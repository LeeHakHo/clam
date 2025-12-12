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
from matplotlib import colors as mcolors
#import matplotlib.cm as cm
import colorsys

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

        # dataset for video
        #self.video_ds_name = "jesbu1_oxe_rfm_eval_oxe_viola_eval"
        #self.video_ds_name = "metaworld_eval"
        self.video_ds_name = "chunk-000"
        cfg2 = copy.deepcopy(self.cfg)
        cfg2.data.shuffle = False
        cfg2.data.batch_size = 1
        cfg2.data.num_trajs = 1
        cfg2.data.num_examples = -1
        cfg2.env.dataset_name = cfg2.env.eval_dataset_name


        ds_dict, *_ = get_dataloader(
            cfg=cfg2,
            dataset_names=[self.video_ds_name],
            dataset_split=[1],
            shuffle=False,
        )

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
                total_loss += 1.0 * clam_output.idm_output.vq_loss #일단 weight 0.1 #Hayden

            metrics.update(clam_output.idm_output.vq_metrics)

        # [Fixed] Frequency of Dead Code Revival increased (500 -> 50 for early stage)
        if train and self.train_step > 0:
            revival_freq = 50 if self.train_step < 2000 else 500 #초기에 50
            
            if self.train_step % revival_freq == 0:
                if hasattr(self.model.idm, "vq") and self.model.idm.vq is not None:
                    if hasattr(self.model.idm.vq, "replace_unused_codebooks"):
                        log(f"[Trainer] Running Dead Code Revival at step {self.train_step}", "yellow")
                        self.model.idm.vq.replace_unused_codebooks(num_batches=revival_freq)
                    
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

    #Hayden - video reconstruction for ablation experiment
    @torch.no_grad()
    def make_episode_video_step(self, ds_name: str, save_path: str, fps: int = 8,
                        max_steps: int | None = None):
        import os
        import imageio
        
        if max_steps is None: max_steps = 50

        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()
        
        # n_step data usually overlaps like [t, t+1, t+2]
        full_gt_seq = []
        full_gt_states = []
        
        print(f"Collecting {max_steps} frames from chopped dataloader...")
        
        # Load the first batch (for warm-up, take the whole thing).
        try:
            first_batch = next(it)
            obs_chunk = first_batch['observations'][0]  # (1, T, C, H, W) -> (T, C, H, W)
            state_chunk = first_batch['states'][0]
            
            for i in range(obs_chunk.shape[0]):
                full_gt_seq.append(torch.tensor(obs_chunk[i]))
                full_gt_states.append(torch.tensor(state_chunk[i]))
                
        except StopIteration:
            return None

        # From the second batch onward, loop and collect only the 'last frame'.
        while len(full_gt_seq) < max_steps + 5: # collect with a small margin
            try:
                batch_np = next(it)
                # Take only the last frame (assuming shift=1)
                last_obs = torch.tensor(batch_np['observations'][0, -1]) 
                last_state = torch.tensor(batch_np['states'][0, -1])
                
                full_gt_seq.append(last_obs)
                full_gt_states.append(last_state)
            except StopIteration:
                break

        # Convert lists to tensors
        gt_seq = torch.stack(full_gt_seq).to(self.device)      # (L, C, H, W)
        gt_states = torch.stack(full_gt_states).to(self.device) # (L, D)
        
        # Trim to the desired length
        gt_seq = gt_seq[:max_steps + 1] 
        gt_states = gt_states[:max_steps + 1]

        log(f"Manually stitched GT length: {gt_seq.shape[0]}", "green")


        recons = self.model.rollout_idm_fdm_closed_loop(
            gt_seq, gt_states=gt_states, max_steps=max_steps
        )
        
        N = recons.shape[0]
        if N == 0: return None

        # Visualization and saving 
        min_len = min(len(recons), len(gt_seq) - 1)
        recons = recons[:min_len]
        gt_match = gt_seq[: min_len]
        N = min_len

        # Visualization loop
        frames = []
        device = recons.device
        dtype  = recons.dtype
        diff_all = (recons - gt_match).abs().mean(dim=1, keepdim=True)
        vmax = diff_all.max().clamp(min=1e-8)
        self._ensure_cmap_lut("magma")
        self._cmap_lut = self._cmap_lut.to(device=device, dtype=dtype)

        for t in range(N):
            diff_t = (diff_all[t] / vmax).clamp(0, 1)
            idx    = (diff_t.squeeze(0) * 255).round().long()
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1).contiguous()
            recon_t = torch.clamp(recons[t],   0, 1)
            gt_t    = torch.clamp(gt_match[t], 0, 1)
            tile = torch.stack([diff_rgb, recon_t, gt_t], dim=0)
            grid = torchvision.utils.make_grid(tile, nrow=3)
            grid = einops.rearrange(grid, "c h w -> h w c")
            img  = (torch.clamp(grid, 0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        
        self.log_to_wandb({f"videos/episode_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")})
        return save_path

    #Hayden - video reconstruction
    @torch.no_grad()
    def make_episode_video(self, ds_name: str, save_path: str, fps: int = 8, max_steps: int | None = None):
        """
        From a dedicated dataloader created with shuffle=False, batch_size=1,
        we stitch together time-ordered windows of the same sample (b=0)
        to create [o_t | ŏ_{t+1} | o_{t+1}] frames and save them as an MP4.
        """

        if max_steps is None:
            max_steps = 50

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

            # Model inference (for Transformer: reconstructed_obs is (B,T-1,C,H,W), GT is (B,T-1,C,H,W))
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

            # If frame-stacked, keep only the last RGB channels
            if self.cfg.env.n_frame_stack > 1:
                C = pred.shape[-3]
                pred = pred[:, C-3:C]; gt = gt[:, C-3:C]; cur = cur[:, C-3:C]

                    
            # ---- Fix the overall diff scale ----
            diff_all = (pred - gt).abs().mean(dim=1, keepdim=True)  # (T',1,H,W)
            vmax = diff_all.max().clamp(min=1e-8)

            # Prepare LUT + match device/dtype (once outside the loop)
            self._ensure_cmap_lut("magma")
            device = pred.device
            dtype  = pred.dtype
            self._cmap_lut = self._cmap_lut.to(device=device, dtype=dtype)

            Tprime = pred.shape[0]
            for t in range(Tprime):
                diff_t = diff_all[t] / vmax

                # NaN / Inf 처리
                diff_t = torch.nan_to_num(diff_t, nan=0.0, posinf=1.0, neginf=0.0)
                diff_t = diff_t.clamp(0, 1)

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

        # 3) Save + log to W&B
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        self.log_to_wandb({f"videos/episode_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")})
        log(f"[make_episode_video] saved: {save_path} (frames={len(frames)}, fps={fps})", "green")
        return save_path

    def _take_one_batch(self, ds):
        # Get a single batch, consistently TFDS → numpy → torch
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
        # Take a single sample from one batch
        batch = self._take_one_batch(sample_dataloader)
        b = 0  # Fix to the first sample
        obs = batch.observations[b]  # (T, C, H, W)

        # Run prediction
        if self.use_transformer:
            out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
            # There are T-1 predictions: o_t -> o_{t+1}
            pred = out.reconstructed_obs[b, :-1]  # (T-1, C, H, W)
            gt   = batch.observations[b, 1:]      # (T-1, C, H, W)
            cur  = batch.observations[b, :-1]     # (T-1, C, H, W) current frame o_t
        else:
            out  = self.model(batch.observations)
            pred = out.reconstructed_obs[b][None]     # (1, C, H, W)
            gt   = batch.observations[b, -1: ]        # (1, C, H, W) = o_{t+1}
            cur  = batch.observations[b, -2:-1]       # (1, C, H, W) = o_t

        # If frame-stacked, use only the last RGB channels
        if self.cfg.env.n_frame_stack > 1:
            C = pred.shape[-3]
            pred = pred[:, C-3:C]; gt = gt[:, C-3:C]; cur = cur[:, C-3:C]


        # Prepare diff
        diff = (pred - gt).abs()  
        diff = diff.mean(dim=1, keepdim=True)

        # Ensure LUT exists
        self._ensure_cmap_lut("magma")
        device = pred.device
        dtype  = pred.dtype
        self._cmap_lut = self._cmap_lut.to(device=device, dtype=dtype)

        eps = 1e-8
        vmax = diff.max().clamp(min=eps) 

        tiles = []
        for t in range(pred.shape[0]):
            # Normalize to [0,1] → 0..255 index
            diff_t = (diff[t] / vmax).clamp(0, 1) 
            idx = (diff_t.squeeze(0) * 255).round().long()  

            diff_rgb = self._cmap_lut[idx] 
            diff_rgb = diff_rgb.permute(2, 0, 1).contiguous()

            pred_t = torch.clamp(pred[t], 0, 1)
            gt_t   = torch.clamp(gt[t],   0, 1)

            tiles += [diff_rgb, pred_t, gt_t]

        grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=3)
        grid = einops.rearrange(grid, "c h w -> h w c")
        grid = torch.clamp(grid, 0, 1)
        grid = (grid * 255).byte().cpu().numpy()
        return grid

    @torch.no_grad()
    def make_correlation_vq(
        self,
        ds_names,   # <-- str 또는 list[str]
        save_path: str = "results/vis/figure13_vq.png",
        max_batches: int = 20000,   # <-- 이제 사용하지 않음 (호출 호환용)
        max_points: int = 500000,
        wandb_prefix: str = "figures/",
    ):
        """
        ds_names: str 또는 [str, str, ...]
        예) "chunk-000" 또는 ["chunk-000", "chunk-001", ..., "chunk-005"]
        여러 dataset에서 모은 (action_x, action_y, code index)를 한 그림에 표시.
        """

        # 1) 입력을 리스트 형태로 정규화
        if isinstance(ds_names, str):
            ds_names = [ds_names]

        self.model.eval()

        all_actions_xy = []
        all_codes = []
        num_points = 0  # 전체 point 개수만 체크

        # 2) 여러 dataset을 순차적으로 훑으면서 전부 모으기
        for name in ds_names:
            # 어떤 dataset 쓸지 고르기 (eval 우선, 없으면 train에서 찾기)
            if hasattr(self, "eval_ds") and name in self.eval_ds:
                ds = self.eval_ds[name]
            elif hasattr(self, "train_ds") and name in self.train_ds:
                ds = self.train_ds[name]
            else:
                log(f"[make_correlation_vq] Unknown dataset name: {name}, skip", "yellow")
                continue

            it = ds.as_numpy_iterator()

            for batch_np in it:
                # max_points만 기준으로 제한
                if num_points >= max_points:
                    break

                batch = to_device(batch_np, self.device)
                batch = Batch(**batch)

                # 3) 모델 forward → CLAMOutput
                out = self.model(
                    batch.observations,
                    timesteps=batch.timestep,
                    states=batch.states,
                )

                idm_out = out.idm_output

                # VQ 안 쓰면 스킵
                if (
                    idm_out is None
                    or idm_out.vq_outputs is None
                    or "indices" not in idm_out.vq_outputs
                ):
                    continue

                codes = idm_out.vq_outputs["indices"]   # (B, T)
                codes = codes[:, 1:]                    # (B, T-1), la[:,1:] ↔ actions[:, :-1]
                actions = batch.actions[:, :-1, :2]     # (B, T-1, 2) only x,y

                codes_np = codes.detach().cpu().numpy().reshape(-1)
                acts_np = actions.detach().cpu().numpy().reshape(-1, 2)

                all_actions_xy.append(acts_np)
                all_codes.append(codes_np)

                num_points += acts_np.shape[0]

            # 이 dataset 다 돌았는데도 max_points를 채웠으면 더 이상 다른 ds 볼 필요 없음
            if num_points >= max_points:
                break

        if len(all_actions_xy) == 0:
            log("[make_correlation_vq] No samples collected.", "red")
            return

        actions_xy = np.concatenate(all_actions_xy, axis=0)  # (N, 2)
        codes = np.concatenate(all_codes, axis=0)            # (N,)

        # 너무 많으면 한 번 더 max_points 기준으로 서브샘플링 (안전장치)
        N = actions_xy.shape[0]
        if N > max_points:
            idx = np.random.choice(N, max_points, replace=False)
            actions_xy = actions_xy[idx]
            codes = codes[idx]
            N = max_points

        log(f"[make_correlation_vq] plotting {N} points from {ds_names}", "green")

        # ---- vocab_size 추론 ----
        vocab_size = None
        try:
            vocab_size = int(self.cfg.model.idm.vq.kwargs.codebook_size)
        except Exception:
            if hasattr(self.model, "idm") and hasattr(self.model.idm, "vq"):
                if hasattr(self.model.idm.vq, "codebook_size"):
                    vocab_size = int(self.model.idm.vq.codebook_size)
        if vocab_size is None:
            vocab_size = int(codes.max()) + 1

        # ---- 서로 다른 색 + 진한 색 세팅 ----
        def make_distinct_colors(n: int, s: float = 1.0, v: float = 1.0):
            hues = np.linspace(0.0, 1.0, n, endpoint=False)
            cols = [colorsys.hsv_to_rgb(h, s, v) for h in hues]
            return np.array(cols)

        color_array = make_distinct_colors(vocab_size, s=1.0, v=1.0)
        cmap = mcolors.ListedColormap(color_array)
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(vocab_size + 1) - 0.5,
            ncolors=cmap.N,
        )

        plt.figure(figsize=(8, 8))
        sc = plt.scatter(
            actions_xy[:, 0],
            actions_xy[:, 1],
            c=codes,
            s=4,
            alpha=1.0,
            cmap=cmap,
            norm=norm,
        )
        plt.colorbar(sc, label="Latent Code Index")

        # title에 어떤 chunk들이 들어갔는지도 표시
        if len(ds_names) == 1:
            ds_str = ds_names[0]
        else:
            ds_str = ", ".join(ds_names)

        title = f"Latent Action Correlation (Datasets: {ds_str})"
        if vocab_size is not None:
            title += f"\nVocab Size: {vocab_size}"
        plt.title(title)

        plt.xlabel("action x")
        plt.ylabel("action y")
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()

        log(f"[make_correlation_vq] saved figure to {save_path}", "green")

        # --- 안전하게 wandb 로그하기 ---
        if hasattr(self, "log_to_wandb"):
            key = f"{wandb_prefix}figure13_{'_'.join(ds_names)}"

            if not os.path.exists(save_path):
                log(f"[make_correlation_vq] file not found for wandb: {save_path}", "red")
            elif os.path.getsize(save_path) == 0:
                log(f"[make_correlation_vq] file is empty for wandb: {save_path}", "red")
            else:
                try:
                    img = wandb.Image(save_path)
                    self.log_to_wandb({key: img})
                    log(f"[make_correlation_vq] logged to wandb as {key}", "green")
                except Exception as e:
                    log(f"[make_correlation_vq] wandb.Image failed: {e}", "red")

        return save_path


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

            #Hayden - image reconstruction
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

            # example video reconstrucion
            #video_target = "jesbu1_oxe_rfm_eval_oxe_viola_eval"
            #video_target = "metaworld_eval"
            video_target = "chunk-000"
            path = self.make_episode_video( # or use self.make_episode_video_step
                ds_name= video_target,
                save_path="results/vis/oxe_eval_ep0.mp4",
                fps=8,
                max_steps=50,
            )
            log(f"visualizing video = {path}", "blue")

            # Figure 13 (action 2D + VQ codes)
            chunk_list = [f"chunk-00{i}" for i in range(5)]  # ["chunk-000", ..., "chunk-004"]
            self.make_correlation_vq(
                ds_names=chunk_list,
                save_path="results/vis/figure13_vq_chunks000_005.png",
                max_batches=200,
                max_points=50000,
            )