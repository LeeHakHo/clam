import collections
import copy

import einops
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import wandb

from clam.models.mlp_policy import MLPPolicy
from clam.models.utils.clam_utils import get_clam_cls, get_la_dim
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.data_utils import Batch
from clam.utils.dataloader import get_dataloader
from clam.utils.general_utils import to_device, to_numpy
from clam.utils.logger import log

import matplotlib.pyplot as plt
import imageio
import os
from matplotlib import colormaps as mpl_cmaps
#import matplotlib.cm as cm


def get_labelled_dataloader(cfg):
    if cfg.data.labelled_data_type == "trajectory":
        # just get enough trajectories to equate to the same amount of labelled examples
        log("loading expert data for training action decoder", "blue")
        log(f"num trajs to load: {cfg.num_labelled_trajs}")
        cfg.data.num_trajs = cfg.num_labelled_trajs
        cfg.data.num_examples = -1
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    elif cfg.data.labelled_data_type == "transition":
        log("loading random transition data for training action decoder", "blue")
        cfg.data.num_trajs = -1
        cfg.data.num_examples = cfg.num_labelled_trajs * cfg.env.max_episode_steps
        log(f"num examples to load: {cfg.data.num_examples}")
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown labelled data type {cfg.data.labelled_data_type}")

    labelled_dataloader = tf.data.Dataset.sample_from_datasets(
        list(labelled_dataloader.values())
    )
    return labelled_dataloader


class CLAMTrainer(OfflineTrainer):
    def __init__(self, cfg):
        self.use_transformer = "transformer" in cfg.model.idm.net.name
        self.use_soda = "soda" in cfg.model.idm.net.name

        log(f"Using transformer: {self.use_transformer}")
        log(f"Using SODA: {self.use_soda}")

        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

        if cfg.joint_action_decoder_training:
            # load data for action decoder
            log("loading labelled data for training action decoder")
            cfg_cpy = copy.deepcopy(cfg)
            self.labelled_dataloader = get_labelled_dataloader(cfg_cpy)

            self.labelled_dataloader_train = (
                self.labelled_dataloader.repeat().as_numpy_iterator()
            )

        # 비디오용 전용 데이터셋 준비 (ds_name이 고정이라면)
        self.video_ds_name = "jesbu1_oxe_rfm_eval_oxe_viola_eval"  # 지금 쓰는 그 이름
        cfg2 = copy.deepcopy(self.cfg)
        cfg2.data.shuffle = False
        cfg2.data.batch_size = 1
        cfg2.data.num_trajs = 1
        cfg2.data.num_examples = -1
        cfg2.env.dataset_name = cfg2.env.eval_dataset_name  # ★ eval 상위 폴더로 group 변경


        ds_dict, *_ = get_dataloader(
            cfg=cfg2,
            dataset_names=[self.video_ds_name],
            dataset_split=[1],
            shuffle=False,
        )
        # 나중에 계속 재사용할 tf.data.Dataset만 저장
        self.video_seq_ds = tf.data.Dataset.sample_from_datasets(
            list(ds_dict.values())
        )

    def setup_action_decoder(self):
        log("---------------------- Initializing Action Decoder ----------------------")

        la_dim = get_la_dim(self.cfg)
        action_decoder = MLPPolicy(
            cfg=self.cfg.model.action_decoder,
            input_dim=la_dim,
            output_dim=self.cfg.env.action_dim,
        )
        action_decoder = action_decoder.to(self.device)
        # self.action_decoder = torch.compile(self.action_decoder)

        log(f"action decoder: {action_decoder}")
        num_params = sum(p.numel() for p in action_decoder.parameters())
        log(f"num params: {num_params}")
        log("=" * 50)
        return action_decoder

    def setup_model(self):
        clam_cls = get_clam_cls(self.cfg.name)
        la_dim = get_la_dim(self.cfg)

        model = clam_cls(self.cfg.model, input_dim=self.obs_shape, la_dim=la_dim)

        if self.cfg.load_from_ckpt:
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="model"
            )

        # create action decoder
        if self.cfg.joint_action_decoder_training:
            self.action_decoder = self.setup_action_decoder()
            self.action_decoder_loss_fn = nn.MSELoss(reduction="none")

            if self.cfg.load_from_ckpt:
                cfg, ckpt = self.action_decoder.load_from_ckpt(
                    self.cfg.ckpt_file,
                    ckpt_step=self.cfg.ckpt_step,
                    key="action_decoder",
                )

        return model

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)

        clam_optimizer, clam_scheduler = super().setup_optimizer_and_scheduler()

        if self.cfg.joint_action_decoder_training:
            log("setting up optimizer and scheduler for action decoder", "green")
            self.action_decoder_optimizer = opt_cls(
                self.action_decoder.parameters(), **self.cfg.decoder_optimizer.params
            )
            self.action_decoder_scheduler = scheduler_cls(
                self.action_decoder_optimizer, **self.cfg.lr_scheduler.params
            )

        return clam_optimizer, clam_scheduler

    def compute_action_decoder_loss(self, batch, train: bool = True):
        obs = batch.observations
        k_step = self.cfg.model.fdm.k_step_pred

        if k_step > 1:
            action_preds = []

            for step in range(k_step):
                end = self.cfg.model.context_len + 1 + step
                obs_splice = obs[:, step:end]

                if self.use_transformer:
                    clam_output = self.model(obs_splice, timesteps=batch.timestep)
                else:
                    clam_output = self.model(obs_splice)

                la = clam_output.la

                if self.cfg.model.distributional_la:
                    la = self.model.reparameterize(la)

                action_pred = self.action_decoder(la)
                action_preds.append(action_pred)

            action_pred = torch.stack(action_preds, dim=1)

            if self.use_transformer:
                action_pred = action_pred[:, 0, :-1]
                gt_actions = batch.actions[:, :-1]
            else:
                gt_actions = batch.actions[:, -2 : (-2 + k_step)]  # TODO: fix this
        else:
            # TODO: clean up this logic
            if self.use_transformer:
                # with torch.no_grad():
                clam_output = self.model(
                    obs, timesteps=batch.timestep, states=batch.states
                )
                la = clam_output.la[:, 1:]

                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[:, :-1]
            else:
                clam_output = self.model(obs)
                la = clam_output.la

                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[
                    :, self.cfg.model.context_len :
                ].squeeze()  # TODO: fix this

        assert action_pred.shape == gt_actions.shape, (
            f"{action_pred.shape}, {gt_actions.shape}"
        )

        gt_actions = to_device(gt_actions, self.device)
        action_decoder_loss = self.action_decoder_loss_fn(action_pred, gt_actions)
        action_decoder_loss = action_decoder_loss.mean()
        return action_decoder_loss, {"action_decoder_loss": action_decoder_loss.item()}

    def compute_loss(self, batch, train: bool = True):
        """
        Compute loss for the Latent Action Model.

        If we are doing k-step prediction,
        we use the reconstructed observation for the next steps prediction.

        e.g. If k_step = 2, we predict o_t+1 and o_t+2

        Step 1:
            o_t-1, o_t => o^_t+1 and z_t
        Step 2:
            o_t, o^_t+1 => o^_t+2 and z_t+1

        Then we compute the summed loss for both o^_t+1 and o^_t+2.
        """
        k_step = self.cfg.model.fdm.k_step_pred

        # check if we are doing k-step prediction
        if k_step > 1:
            # recursively predict the next obs and latent actions
            def splice(x, start, end):
                if x is not None:
                    return x[:, start:end]
                return x

            total_loss = 0.0
            obs_recons = []
            metrics = collections.defaultdict(float)

            for step in range(k_step):
                # need to splice Batch
                end = self.cfg.model.context_len + 1 + step
                batch_splice = dict(
                    map(
                        lambda kv: (kv[0], splice(kv[1], start=step, end=end)),
                        batch.__dict__.items(),
                    )
                )
                batch_splice = Batch(**batch_splice)

                # replace the second to last observation with the reconstructed observation
                if len(obs_recons) > 0:
                    new_observations = batch_splice.observations.clone()
                    new_observations[:, -2] = obs_recons[-1]
                    batch_splice.observations = new_observations

                # compute LAM loss for single step
                step_metrics, obs_recon, step_loss = self.compute_step_loss(
                    batch_splice, train=train
                )

                for k, v in step_metrics.items():
                    metrics[k] += v

                obs_recons.append(obs_recon)
                total_loss += step_loss

            obs_recons = torch.stack(obs_recons, dim=-1)
            metrics = {k: v / k_step for k, v in metrics.items()}
        else:
            metrics, obs_recon, total_loss = self.compute_step_loss(batch, train=train)

        return metrics, total_loss

    def compute_step_loss(self, batch, train: bool = True):
        """
        Computes IDM/FDM loss for a single step.

        e.g. o_t-1, o_t => o_t+1, we also predict z_t
        which is the latent action between o_t and o_t+1
        """

        # reconstructs full sequence if using a transformer model
        if self.use_transformer:
            clam_output = self.model(
                batch.observations, timesteps=batch.timestep, states=batch.states
            )
            obs_gt = batch.observations[:, 1:]

            cheat_pred = batch.observations[:, :-1]  # this refers to o_t
            cheat_loss = self.loss_fn(cheat_pred, obs_gt)
            cheat_loss = cheat_loss.mean()
        else:
            clam_output = self.model(batch.observations)

            if self.cfg.model.fdm.predict_target_embedding:
                # compute target embedding for the next observation
                pass
            else:   
                # reconstruct the next raw observation
                obs_gt = batch.observations[:, self.cfg.model.context_len :].squeeze()

            # compute cheating loss, this is if we just predict the current obs as the next obs
            cheat_pred = batch.observations[
                :, self.cfg.model.context_len - 1 : -1
            ].squeeze()  # this refers to o_t
            cheat_loss = self.loss_fn(cheat_pred, obs_gt)
            cheat_loss = cheat_loss.mean()

        obs_recon = clam_output.reconstructed_obs
        la = clam_output.la
        # make sure the shapes are correct
        assert obs_recon.shape == obs_gt.shape, f"{obs_recon.shape}, {obs_gt.shape}"

        recon_loss = self.loss_fn(obs_recon, obs_gt)
        recon_loss = recon_loss.mean()
        total_loss = self.cfg.model.recon_loss_weight * recon_loss

        # compute KL divergence loss to regularize the latent action
        if self.cfg.model.distributional_la and self.cfg.model.kl_loss_weight:
            la_mean = la[:, : self.cfg.model.la_dim]
            la_logvar = la[:, self.cfg.model.la_dim :]

            posterior = torch.distributions.Normal(la_mean, torch.exp(0.5 * la_logvar))
            prior = torch.distributions.Normal(0, 1)
            kl_loss = torch.distributions.kl.kl_divergence(posterior, prior)
            kl_loss = kl_loss.mean()
            total_loss += self.cfg.model.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "la_mean": la.mean().item(),
            "la_min": la.min().item(),
            "la_max": la.max().item(),
            "obs_recon_mean": obs_recon.mean().item(),
            "obs_recon_min": obs_recon.min().item(),
            "obs_recon_max": obs_recon.max().item(),
            "cheat_loss": cheat_loss.item(),
        }

        if self.cfg.model.idm.quantize_la:
            if clam_output.idm_output.vq_loss is not None:
                total_loss += clam_output.idm_output.vq_loss

            metrics.update(clam_output.idm_output.vq_metrics)

        # run this if we are doing joint training or if this is eval
        if (
            self.cfg.joint_action_decoder_training
            and self.train_step % self.cfg.train_action_decoder_every == 0
            and train
        ):
            labelled_batch = next(self.labelled_dataloader_train)
            labelled_batch = to_device(labelled_batch, self.device)
            labelled_batch = Batch(**labelled_batch)

            action_decoder_loss, action_decoder_metrics = (
                self.compute_action_decoder_loss(labelled_batch)
            )
            total_loss += self.cfg.action_decoder_loss_weight * action_decoder_loss
            metrics.update(action_decoder_metrics)

        return metrics, obs_recon, total_loss

    @property
    def save_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.cfg.joint_action_decoder_training:
            state_dict["action_decoder"] = self.action_decoder.state_dict()
            state_dict["action_decoder_opt"] = (
                self.action_decoder_optimizer.state_dict()
            )
        return state_dict

    #Hayden
    @torch.no_grad()
    def make_episode_video(self, ds_name: str, save_path: str, fps: int = 8, max_steps: int | None = None):
        """
        shuffle=False, batch_size=1로 만든 전용 dataloader에서
        같은 샘플(b=0)의 시간순 윈도우를 이어 붙여
        [o_t | ŏ_{t+1} | o_{t+1}] 프레임을 생성해 MP4로 저장.
        """
        # 0) 비디오 길이 한도
        if max_steps is None:
            max_steps = 50

        # 1) __init__에서 만든 Dataset 사용
        #    (지금 예시는 self.video_seq_ds 하나만 있다고 가정)
        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()   # 이터레이터는 여기서 매번 새로 생성

        frames = []
        steps = 0
        while steps < max_steps:
            try:
                batch_np = next(it)
            except StopIteration:
                break

            batch = to_device(batch_np, self.device)
            batch = Batch(**batch)

            # 모델 추론 (Transformer 기준: reconstructed_obs가 (B,T-1, C,H,W), GT는 (B,T-1,C,H,W))
            if self.use_transformer:
                out  = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
                pred = out.reconstructed_obs[0, :-1]   # (T-1,C,H,W)
                gt   = batch.observations[0, 1:]       # (T-1,C,H,W)
                cur  = batch.observations[0, :-1]      # (T-1,C,H,W)
            else:
                out  = self.model(batch.observations)
                pred = out.reconstructed_obs[0][None]  # (1,C,H,W)
                gt   = batch.observations[0, -1: ]     # (1,C,H,W)
                cur  = batch.observations[0, -2:-1]    # (1,C,H,W)

            # 프레임스택이면 마지막 RGB만
            if self.cfg.env.n_frame_stack > 1:
                C = pred.shape[-3]
                pred = pred[:, C-3:C]; gt = gt[:, C-3:C]; cur = cur[:, C-3:C]

                    
            # ---- diff 전체 스케일 고정 ----
            diff_all = (pred - gt).abs().mean(dim=1, keepdim=True)  # (T',1,H,W)
            vmax = diff_all.max().clamp(min=1e-8)

            # LUT 준비 + 디바이스/타입 맞추기 (루프 바깥에서 1회)
            self._ensure_cmap_lut("magma")
            device = pred.device
            dtype  = pred.dtype
            self._cmap_lut = self._cmap_lut.to(device=device, dtype=dtype)

            Tprime = pred.shape[0]
            for t in range(Tprime):
                diff_t = (diff_all[t] / vmax).clamp(0, 1)
                idx    = (diff_t.squeeze(0) * 255).round().long()
                diff_rgb = self._cmap_lut[idx].permute(2,0,1).contiguous()  # (3,H,W)

                pred_t = torch.clamp(pred[t], 0, 1).to(device=device, dtype=dtype)
                gt_t   = torch.clamp(gt[t],   0, 1).to(device=device, dtype=dtype)

                tile = torch.stack([diff_rgb, pred_t, gt_t], dim=0)
                grid = torchvision.utils.make_grid(tile, nrow=3)
                grid = einops.rearrange(grid, "c h w -> h w c")
                img  = (torch.clamp(grid, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                frames.append(img)
                steps += 1
                if steps >= max_steps:
                    break

        if len(frames) == 0:
            log(f"[make_episode_video] no frames produced for {ds_name}", "yellow")
            return None

        # 3) 저장 + W&B 로깅
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        self.log_to_wandb({f"videos/episode_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")})
        log(f"[make_episode_video] saved: {save_path} (frames={len(frames)}, fps={fps})", "green")
        return save_path


    def _take_one_batch(self, ds):
        # TFDS → numpy → torch 로 동일하게 한 배치만
        it = ds.take(1).as_numpy_iterator()
        batch = next(it)
        batch = to_device(batch, self.device)
        return Batch(**batch)

    def _ensure_cmap_lut(self, name: str = "magma"):
        if not hasattr(self, "_cmap_lut") or self._cmap_lut is None:
            lut_np = mpl_cmaps.get_cmap(name)(np.linspace(0, 1, 256))[..., :3]  # (256,3)
            self._cmap_lut = torch.tensor(lut_np, device=self.device, dtype=torch.float32)  # CUDA

    @torch.no_grad()
    def target_vis(self, sample_dataloader):
        # 1) 한 배치에서 한 샘플만 추출
        batch = self._take_one_batch(sample_dataloader)
        b = 0  # 첫 샘플 고정
        obs = batch.observations[b]  # (T, C, H, W)

        # 2) 예측 실행
        if self.use_transformer:
            out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
            # 예측은 T-1개: o_t -> o_{t+1}
            pred = out.reconstructed_obs[b, :-1]  # (T-1, C, H, W)
            gt   = batch.observations[b, 1:]      # (T-1, C, H, W)
            cur  = batch.observations[b, :-1]     # (T-1, C, H, W) 현재 프레임 o_t
        else:
            out  = self.model(batch.observations)
            # non-transformer는 next만 1개 내는 구조이므로, 윈도우 마지막 step만 보여줌
            pred = out.reconstructed_obs[b][None]     # (1, C, H, W)
            gt   = batch.observations[b, -1: ]        # (1, C, H, W) = o_{t+1}
            cur  = batch.observations[b, -2:-1]       # (1, C, H, W) = o_t

        # 3) 프레임스택이면 마지막 RGB만 사용
        if self.cfg.env.n_frame_stack > 1:
            C = pred.shape[-3]
            pred = pred[:, C-3:C]; gt = gt[:, C-3:C]; cur = cur[:, C-3:C]


        # diff 준비
        diff = (pred - gt).abs()                       # (T', C, H, W)
        # 채널 평균으로 스칼라 diff (또는 원하는 방식)
        diff = diff.mean(dim=1, keepdim=True)          # (T', 1, H, W)

        # LUT 보장
        self._ensure_cmap_lut("magma")
        device = pred.device
        dtype  = pred.dtype
        self._cmap_lut = self._cmap_lut.to(device=device, dtype=dtype)


        # 정규화 범위 (안정적이게)
        eps = 1e-8
        vmax = diff.max().clamp(min=eps)               # 스칼라 (CUDA)

        tiles = []
        for t in range(pred.shape[0]):
            # [0,1] 정규화 → 0..255 인덱스
            diff_t = (diff[t] / vmax).clamp(0, 1)      # (1,H,W)
            idx = (diff_t.squeeze(0) * 255).round().long()  # (H,W), CUDA

            # GPU LUT로 색 입히기 → (H,W,3) → (3,H,W)
            diff_rgb = self._cmap_lut[idx]             # (H,W,3) CUDA
            diff_rgb = diff_rgb.permute(2, 0, 1).contiguous()  # (3,H,W) CUDA

            # pred/gt도 0..1로 클램프(이미 CUDA)
            pred_t = torch.clamp(pred[t], 0, 1)
            gt_t   = torch.clamp(gt[t],   0, 1)

            # [diff_rgb | pred | gt]
            tiles += [diff_rgb, pred_t, gt_t]

        grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=3)  # CUDA
        grid = einops.rearrange(grid, "c h w -> h w c")
        grid = torch.clamp(grid, 0, 1)
        grid = (grid * 255).byte().cpu().numpy()   # 저장/로그 직전에만 CPU로
        return grid


    def eval(self, step: int):
        super().eval(step=step)

        if self.cfg.joint_action_decoder_training:
            log("running action decoder evaluation", "blue")

            self.model.eval()
            self.action_decoder.eval()

            # first run on eval set
            eval_iter = self.eval_dataloader.as_numpy_iterator()

            eval_metrics = collections.defaultdict(list)
            for batch in eval_iter:
                batch = to_device(batch, self.device)
                batch = Batch(**batch)

                with torch.no_grad():
                    action_decoder_loss, action_decoder_metrics = (
                        self.compute_action_decoder_loss(batch)
                    )

                    for k, v in action_decoder_metrics.items():
                        eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(np.array(v))

            self.log_to_wandb(eval_metrics, prefix="action_decoder_eval/")

            # then run on the full labelled dataset
            eval_metrics = collections.defaultdict(list)
            labelled_dataloader_eval = self.labelled_dataloader.as_numpy_iterator()

            for batch in labelled_dataloader_eval:
                batch = to_device(batch, self.device)
                batch = Batch(**batch)

                with torch.no_grad():
                    action_decoder_loss, action_decoder_metrics = (
                        self.compute_action_decoder_loss(batch)
                    )

                    for k, v in action_decoder_metrics.items():
                        eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(np.array(v))

            self.log_to_wandb(eval_metrics, prefix="action_decoder_labelled_eval/")

        if self.cfg.env.image_obs and not self.cfg.model.fdm.predict_target_embedding:
            # visualize the image reconstructions on eval set
            log("visualizing image reconstructions", "blue")

            # # sample a batch
            # eval_iter = self.eval_dataloader.as_numpy_iterator()
            # batch = next(eval_iter)
            # batch = to_device(batch, self.device)
            # batch = Batch(**batch)

            # if self.use_transformer:
            #     clam_output = self.model(
            #         batch.observations, timesteps=batch.timestep, states=batch.states
            #     )
            #     obs_recon = clam_output.reconstructed_obs[:, :-1]
            #     obs_gt = batch.observations[:, 1:]

            #     # we will need to flatten the first two dimensions
            #     obs_recon = einops.rearrange(obs_recon, "B T C H W -> (B T) C H W")
            #     obs_gt = einops.rearrange(obs_gt, "B T C H W -> (B T) C H W")
            # else:
            #     clam_output = self.model(batch.observations)
            #     obs_recon = clam_output.reconstructed_obs
            #     obs_gt = batch.observations[:, -1]

            # # wandb expects [B, H, W, C]
            # assert obs_recon.ndim == 4

            # num_exs = 5
            # obs_recon = obs_recon[:num_exs]
            # obs_gt = obs_gt[:num_exs]

            # # if there is framestack, take the last frame
            # if self.cfg.env.n_frame_stack > 1:
            #     obs_recon = obs_recon[:, -3:]
            #     obs_gt = obs_gt[:, -3:]

            # # mix them together
            # to_vis = torch.cat([obs_recon, obs_gt])
            # to_vis = torchvision.utils.make_grid(to_vis, nrow=num_exs)
            # # make channel last
            # to_vis = einops.rearrange(to_vis, "c h w -> h w c")
            # to_vis = to_numpy(to_vis)

            # #Hayden
            # # output_path = "/home1/hyeonhoo/code/clam/debug_sample_wandb.png"
            # # obs_gt = einops.rearrange(obs_recon[0], "c h w -> h w c")
            # # img = to_numpy(obs_gt)

            # # #to_vis = np.clip(to_vis, 0, 1)
            # # plt.imsave(output_path, img)
            # # checkpoimt()

            # # plot images
            # self.log_to_wandb({"obs_recon_1": wandb.Image(to_vis)}, prefix="images/")

            #Hayden

            #target1 = "jesbu1_oxe_rfm_eval_oxe_viola_eval"
            #target1 = "metaworld_eval"
            # log(f"visualizing image reconstructions - target1={target1}", "blue")
            # to_vis1 = self.target_vis(self.eval_ds[target1])
            # self.log_to_wandb({f"obs_recon_target_eval_{target1}": wandb.Image(to_vis1)},
            #                 prefix="images/")

            # target2 = "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval"
            # log(f"visualizing image reconstructions - target2={target2}", "blue")
            # to_vis2 = self.target_vis(self.eval_ds[target2])
            # self.log_to_wandb({f"obs_recon_target_eval_{target2}": wandb.Image(to_vis2)},
            #                 prefix="images/")

            # target_train = "jesbu1_oxe_rfm_oxe_aloha_mobile"
            # #target_train = "metaworld_train"
            # log(f"visualizing image reconstructions - train={target_train}", "blue")
            # to_vis2 = self.target_vis(self.train_ds[target_train])
            # self.log_to_wandb({f"obs_recon_target_train_{target_train}": wandb.Image(to_vis2)},
            #                 prefix="images/")

            video_target = "jesbu1_oxe_rfm_eval_oxe_viola_eval"
            path = self.make_episode_video(
                ds_name= video_target,
                save_path="results/vis/oxe_eval_ep0.mp4",
                fps=8,
                max_steps=50,
            )
            log(f"visualizing video = {path}", "blue")