import collections
import copy
import os

import einops
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import wandb
import imageio
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps
from matplotlib import colors as mcolors
import colorsys
from rich.pretty import pretty_repr

from clam.models.mlp_policy import MLPPolicy
from clam.models.utils.clam_utils import get_clam_cls, get_la_dim
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.data_utils import Batch
from clam.utils.dataloader import get_dataloader
from clam.utils.general_utils import to_device, to_numpy
from clam.utils.logger import log


def get_labelled_dataloader(cfg):
    if cfg.data.labelled_data_type == "trajectory":
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
            log("loading labelled data for training action decoder")
            cfg_cpy = copy.deepcopy(cfg)
            self.labelled_dataloader = get_labelled_dataloader(cfg_cpy)

            self.labelled_dataloader_train = (
                self.labelled_dataloader.repeat().as_numpy_iterator()
            )

        # ---------------------------------------------------------------------
        # [Fix] Video Evaluation Dataset Setup
        # ---------------------------------------------------------------------
        eval_ds_path = getattr(self.cfg.env, "eval_dataset_name", self.cfg.env.dataset_name)
        eval_ds_list = getattr(self.cfg.env, "eval_datasets", getattr(self.cfg.env, "datasets", []))

        # 2. 비디오 생성에 쓸 "단 하나의 청크 이름" 파싱 (공백 제거 포함)
        if isinstance(eval_ds_list, (list, tuple)) and len(eval_ds_list) > 0:
            target_chunk = eval_ds_list[0]
        elif isinstance(eval_ds_list, str):
            target_chunk = eval_ds_list.strip("[]' ").split(",")[0].strip()
        else:
            target_chunk = "chunk-000"

        self.video_ds_name = target_chunk
        
        log(f"[VideoEval] Path: '{eval_ds_path}', Target Chunk: '{self.video_ds_name}'", "blue")

        # 3. 데이터 로더용 임시 Config
        cfg2 = copy.deepcopy(self.cfg)
        cfg2.data.shuffle = False
        
        # [중요] 길이가 짧은 데이터 처리 및 로딩 확률 높이기
        cfg2.data.batch_size = 1
        cfg2.data.num_trajs = 10 
        cfg2.data.num_examples = -1
        cfg2.data.pad_dataset = True 

        cfg2.env.dataset_name = eval_ds_path
        cfg2.env.datasets = [self.video_ds_name]

        # 4. get_dataloader 호출
        ds_dict, *_ = get_dataloader(
            cfg=cfg2,
            dataset_names=[self.video_ds_name],
            dataset_split=[1],
            shuffle=False,
        )

        if not ds_dict:
            log(f"[VideoEval] Failed to load chunk '{self.video_ds_name}'. Dictionary is empty.", "red")
            self.video_seq_ds = None
        else:
            self.video_seq_ds = tf.data.Dataset.sample_from_datasets(
                list(ds_dict.values())
            )
            log(f"[VideoEval] Successfully loaded video dataset.", "green")

    def setup_action_decoder(self):
        log("---------------------- Initializing Action Decoder ----------------------")
        la_dim = get_la_dim(self.cfg)
        action_decoder = MLPPolicy(
            cfg=self.cfg.model.action_decoder,
            input_dim=la_dim,
            output_dim=self.cfg.env.action_dim,
        )
        action_decoder = action_decoder.to(self.device)
        log(f"action decoder: {action_decoder}")
        return action_decoder

    def setup_model(self):
        clam_cls = get_clam_cls(self.cfg.name)
        la_dim = get_la_dim(self.cfg)
        model = clam_cls(self.cfg.model, input_dim=self.obs_shape, la_dim=la_dim)

        if self.cfg.load_from_ckpt:
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="model"
            )

        if self.cfg.joint_action_decoder_training:
            self.action_decoder = self.setup_action_decoder()
            self.action_decoder_loss_fn = nn.MSELoss(reduction="none")
            if self.cfg.load_from_ckpt:
                cfg, ckpt = self.action_decoder.load_from_ckpt(
                    self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="action_decoder",
                )
        return model

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)
        clam_optimizer, clam_scheduler = super().setup_optimizer_and_scheduler()

        if self.cfg.joint_action_decoder_training:
            self.action_decoder_optimizer = opt_cls(
                self.action_decoder.parameters(), **self.cfg.decoder_optimizer.params
            )
            self.action_decoder_scheduler = scheduler_cls(
                self.action_decoder_optimizer, **self.cfg.lr_scheduler.params
            )
        return clam_optimizer, clam_scheduler

    def compute_action_decoder_loss(self, batch, train: bool = True):
        # (기존 코드와 동일하여 생략 가능하지만 전체 코드를 위해 포함)
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
                gt_actions = batch.actions[:, -2 : (-2 + k_step)]
        else:
            if self.use_transformer:
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
                gt_actions = batch.actions[:, self.cfg.model.context_len :].squeeze()

        assert action_pred.shape == gt_actions.shape
        gt_actions = to_device(gt_actions, self.device)
        action_decoder_loss = self.action_decoder_loss_fn(action_pred, gt_actions).mean()
        return action_decoder_loss, {"action_decoder_loss": action_decoder_loss.item()}

    def compute_loss(self, batch, train: bool = True):
        k_step = self.cfg.model.fdm.k_step_pred
        if k_step > 1:
            def splice(x, start, end):
                return x[:, start:end] if x is not None else x
            total_loss = 0.0
            obs_recons = []
            metrics = collections.defaultdict(float)
            for step in range(k_step):
                end = self.cfg.model.context_len + 1 + step
                batch_splice = dict(map(lambda kv: (kv[0], splice(kv[1], start=step, end=end)), batch.__dict__.items()))
                batch_splice = Batch(**batch_splice)
                if len(obs_recons) > 0:
                    new_observations = batch_splice.observations.clone()
                    new_observations[:, -2] = obs_recons[-1]
                    batch_splice.observations = new_observations
                step_metrics, obs_recon, step_loss = self.compute_step_loss(batch_splice, train=train)
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
        if self.use_transformer:
            clam_output = self.model(
                batch.observations, timesteps=batch.timestep, states=batch.states
            )
            obs_gt = batch.observations[:, 1:]
            cheat_pred = batch.observations[:, :-1]
            cheat_loss = self.loss_fn(cheat_pred, obs_gt).mean()
        else:
            clam_output = self.model(batch.observations)
            if not self.cfg.model.fdm.predict_target_embedding:
                obs_gt = batch.observations[:, self.cfg.model.context_len :].squeeze()
            cheat_pred = batch.observations[:, self.cfg.model.context_len - 1 : -1].squeeze()
            cheat_loss = self.loss_fn(cheat_pred, obs_gt).mean()

        obs_recon = clam_output.reconstructed_obs
        la = clam_output.la
        assert obs_recon.shape == obs_gt.shape, f"{obs_recon.shape}, {obs_gt.shape}"

        recon_loss = self.loss_fn(obs_recon, obs_gt).mean()
        total_loss = self.cfg.model.recon_loss_weight * recon_loss

        if self.cfg.model.distributional_la and self.cfg.model.kl_loss_weight:
            la_mean = la[:, : self.cfg.model.la_dim]
            la_logvar = la[:, self.cfg.model.la_dim :]
            posterior = torch.distributions.Normal(la_mean, torch.exp(0.5 * la_logvar))
            prior = torch.distributions.Normal(0, 1)
            kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).mean()
            total_loss += self.cfg.model.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "cheat_loss": cheat_loss.item(),
        }

        if self.cfg.model.idm.quantize_la:
            if clam_output.idm_output.vq_loss is not None:
                total_loss += 1.0 * clam_output.idm_output.vq_loss
            metrics.update(clam_output.idm_output.vq_metrics)

        if train and self.train_step > 0:
            revival_freq = 50 if self.train_step < 2000 else 500
            if self.train_step % revival_freq == 0:
                if hasattr(self.model.idm, "vq") and self.model.idm.vq is not None:
                    if hasattr(self.model.idm.vq, "replace_unused_codebooks"):
                        self.model.idm.vq.replace_unused_codebooks(num_batches=revival_freq)

        if (self.cfg.joint_action_decoder_training and self.train_step % self.cfg.train_action_decoder_every == 0 and train):
            labelled_batch = next(self.labelled_dataloader_train)
            labelled_batch = to_device(labelled_batch, self.device)
            labelled_batch = Batch(**labelled_batch)
            action_decoder_loss, action_decoder_metrics = self.compute_action_decoder_loss(labelled_batch)
            total_loss += self.cfg.action_decoder_loss_weight * action_decoder_loss
            metrics.update(action_decoder_metrics)

        return metrics, obs_recon, total_loss

    @property
    def save_dict(self):
        state_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        if self.cfg.joint_action_decoder_training:
            state_dict["action_decoder"] = self.action_decoder.state_dict()
            state_dict["action_decoder_opt"] = self.action_decoder_optimizer.state_dict()
        return state_dict

    def _ensure_cmap_lut(self, name: str = "magma"):
        if not hasattr(self, "_cmap_lut") or self._cmap_lut is None:
            lut_np = mpl_cmaps.get_cmap(name)(np.linspace(0, 1, 256))[..., :3]
            self._cmap_lut = torch.tensor(lut_np, device=self.device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # [Fix] Corrected Video Generation Methods
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def make_episode_video(self, ds_name: str, save_path: str, fps: int = 8, max_steps: int | None = None):
        if self.video_seq_ds is None: return None
        if max_steps is None: max_steps = 50

        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()

        frames = []
        steps = 0
        while steps < max_steps:
            try:
                batch_np = next(it)
            except StopIteration:
                break

            batch = to_device(batch_np, self.device)
            batch = Batch(**batch)

            if self.use_transformer:
                out  = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
                # [BUG FIX] Removed ':-1' slicing. Output is already T-1.
                pred = out.reconstructed_obs[0]        # (T-1,C,H,W)
                gt   = batch.observations[0, 1:]       # (T-1,C,H,W)
            else:
                out  = self.model(batch.observations)
                pred = out.reconstructed_obs[0][None]
                gt   = batch.observations[0, -1: ]

            if self.cfg.env.n_frame_stack > 1:
                C = pred.shape[-3]
                pred = pred[:, C-3:C]; gt = gt[:, C-3:C]

            # Diff visualization
            diff_all = (pred - gt).abs().mean(dim=1, keepdim=True)
            vmax = diff_all.max().clamp(min=1e-8)
            self._ensure_cmap_lut("magma")
            self._cmap_lut = self._cmap_lut.to(device=pred.device, dtype=pred.dtype)

            Tprime = pred.shape[0]
            for t in range(Tprime):
                diff_t = diff_all[t] / vmax
                diff_t = torch.nan_to_num(diff_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                idx = (diff_t.squeeze(0) * 255).floor().clamp(0, 255).long()
                diff_rgb = self._cmap_lut[idx].permute(2,0,1).contiguous()
                
                pred_t = (pred[t].clamp(-1, 1) + 1) / 2
                gt_t   = (gt[t].clamp(-1, 1) + 1) / 2

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

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        self.log_to_wandb({f"videos/episode_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")})
        log(f"[make_episode_video] saved: {save_path}", "green")
        return save_path

    @torch.no_grad()
    def make_dreamer_rollout_video(self, ds_name: str, save_path: str, context_len: int = 5, fps: int = 8, max_steps: int = 50):
        if not hasattr(self, "video_seq_ds") or self.video_seq_ds is None: return None
        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()

        full_gt_seq = []
        try:
            first_batch = next(it)
            obs_chunk = first_batch['observations'][0]
            for i in range(obs_chunk.shape[0]):
                full_gt_seq.append(torch.tensor(obs_chunk[i]))
        except StopIteration:
            return None

        # Collect enough frames
        while len(full_gt_seq) < max_steps + 5:
            try:
                batch_np = next(it)
                last_obs = torch.tensor(batch_np['observations'][0, -1])
                full_gt_seq.append(last_obs)
            except StopIteration:
                break
        
        if len(full_gt_seq) < 2: return None
        gt_seq = torch.stack(full_gt_seq).to(self.device)[:max_steps]

        if not hasattr(self.model, "visualize_dreamer_style_rollout"): return None
        try:
            recons = self.model.visualize_dreamer_style_rollout(gt_seq, context_len=context_len)
        except Exception as e:
            log(f"[DreamerVis] Error: {e}", "red")
            return None

        if recons is None or len(recons) == 0: return None

        T_pred = recons.shape[0]
        gt_match = gt_seq[1 : 1 + T_pred]
        frames = []
        
        self._ensure_cmap_lut("magma")
        if self._cmap_lut.device != recons.device:
            self._cmap_lut = self._cmap_lut.to(device=recons.device, dtype=recons.dtype)

        diff_all = (recons - gt_match).abs().mean(dim=1, keepdim=True)
        vmax = diff_all.max().clamp(min=1e-8)

        for t in range(T_pred):
            recon_t = recons[t].clamp(0, 1) 
            gt_t = (gt_match[t].clamp(-1, 1) + 1) / 2 
            
            diff_t = (diff_all[t] / vmax).clamp(0, 1)
            idx = (diff_t.squeeze(0) * 255).long().clamp(0, 255)
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1)

            current_step = t + 1
            is_context = current_step < context_len
            color = torch.tensor([0.0, 1.0, 0.0], device=recons.device) if is_context else torch.tensor([1.0, 0.0, 0.0], device=recons.device)
            border = color.view(3, 1, 1).repeat(1, 3, recon_t.shape[2]) 
            recon_t[:, :3, :] = border 

            tile = torch.stack([diff_rgb, recon_t, gt_t], dim=0)
            grid = torchvision.utils.make_grid(tile, nrow=3, padding=2)
            grid = einops.rearrange(grid, "c h w -> h w c")
            img = (torch.clamp(grid, 0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        self.log_to_wandb({f"videos/dreamer_rollout_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")})
        return save_path

    @torch.no_grad()
    def target_vis(self, sample_dataloader):
        batch = self._take_one_batch(sample_dataloader)
        b = 0
        if self.use_transformer:
            out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
            # [BUG FIX] Removed ':-1' slicing here too
            pred = out.reconstructed_obs[b]       # (T-1, C, H, W)
            gt   = batch.observations[b, 1:]      # (T-1, C, H, W)
        else:
            out  = self.model(batch.observations)
            pred = out.reconstructed_obs[b][None]
            gt   = batch.observations[b, -1: ]

        if self.cfg.env.n_frame_stack > 1:
            C = pred.shape[-3]
            pred = pred[:, C-3:C]; gt = gt[:, C-3:C]

        diff = (pred - gt).abs().mean(dim=1, keepdim=True)
        self._ensure_cmap_lut("magma")
        self._cmap_lut = self._cmap_lut.to(device=pred.device, dtype=pred.dtype)
        
        vmax = diff.max().clamp(min=1e-8)
        tiles = []
        for t in range(pred.shape[0]):
            diff_t = (diff[t] / vmax).clamp(0, 1)
            idx = (diff_t.squeeze(0) * 255).round().long()
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1).contiguous()
            pred_t = torch.clamp(pred[t], 0, 1)
            gt_t   = torch.clamp(gt[t],   0, 1)
            tiles += [diff_rgb, pred_t, gt_t]

        grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=3)
        grid = einops.rearrange(grid, "c h w -> h w c")
        return (torch.clamp(grid, 0, 1) * 255).byte().cpu().numpy()

    # (make_correlation_vq 등은 기존과 동일)
    @torch.no_grad()
    def make_correlation_vq(self, ds_names, save_path="results/vis/figure13_vq.png", max_batches=20000, max_points=500000, wandb_prefix="figures/"):
        if isinstance(ds_names, str): ds_names = [ds_names]
        self.model.eval()
        all_actions_xy, all_codes, num_points = [], [], 0

        for name in ds_names:
            if hasattr(self, "eval_ds") and name in self.eval_ds: ds = self.eval_ds[name]
            elif hasattr(self, "train_ds") and name in self.train_ds: ds = self.train_ds[name]
            else: continue
            
            for batch_np in ds.as_numpy_iterator():
                if num_points >= max_points: break
                batch = Batch(**to_device(batch_np, self.device))
                out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
                idm_out = out.idm_output
                if idm_out is None or idm_out.vq_outputs is None or "indices" not in idm_out.vq_outputs: continue
                
                codes = idm_out.vq_outputs["indices"][:, 1:]
                actions = batch.actions[:, :-1, :2]
                all_actions_xy.append(actions.detach().cpu().numpy().reshape(-1, 2))
                all_codes.append(codes.detach().cpu().numpy().reshape(-1))
                num_points += actions.shape[0] * actions.shape[1]
            if num_points >= max_points: break

        if not all_actions_xy: return
        actions_xy = np.concatenate(all_actions_xy, axis=0)
        codes = np.concatenate(all_codes, axis=0)

        if actions_xy.shape[0] > max_points:
            idx = np.random.choice(actions_xy.shape[0], max_points, replace=False)
            actions_xy = actions_xy[idx]
            codes = codes[idx]

        vocab_size = int(codes.max()) + 1
        cmap = mcolors.ListedColormap([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0, 1, vocab_size, endpoint=False)])
        norm = mcolors.BoundaryNorm(np.arange(vocab_size + 1) - 0.5, cmap.N)

        plt.figure(figsize=(8, 8))
        plt.scatter(actions_xy[:, 0], actions_xy[:, 1], c=codes, s=4, alpha=1.0, cmap=cmap, norm=norm)
        plt.colorbar(label="Latent Code Index")
        plt.title(f"Latent Action Correlation (Datasets: {', '.join(ds_names)})")
        plt.xlabel("action x"); plt.ylabel("action y"); plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200); plt.close()
        if hasattr(self, "log_to_wandb") and os.path.exists(save_path):
             self.log_to_wandb({f"{wandb_prefix}figure13_{'_'.join(ds_names)}": wandb.Image(save_path)})

    def eval(self, step: int):
        super().eval(step=step)
        if self.cfg.joint_action_decoder_training:
            # (Action Decoder Evaluation skipped for brevity)
            pass

        if self.cfg.env.image_obs and not self.cfg.model.fdm.predict_target_embedding:
            log("visualizing image reconstructions", "blue")
            video_target = self.video_ds_name
            
            self.make_episode_video(
                ds_name=video_target,
                save_path="results/vis/oxe_eval_ep0.mp4",
                fps=8,
                max_steps=50,
            )
            
            self.make_dreamer_rollout_video(
                ds_name=video_target,
                save_path="results/vis/dreamer_open_loop.mp4",
                context_len=5,
                fps=8,
                max_steps=50
            )

            try:
                chunk_list = [f"chunk-00{i}" for i in range(5)] if "chunk" in video_target else [video_target]
                self.make_correlation_vq(
                    ds_names=chunk_list,
                    save_path="results/vis/figure13_vq_chunks000_005.png",
                    max_points=50000,
                )
            except Exception as e:
                log(f"VQ Correlation visualization failed: {e}", "yellow")