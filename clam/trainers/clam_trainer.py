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
from omegaconf import OmegaConf

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
        # Video Evaluation Dataset Setup
        # ---------------------------------------------------------------------
        eval_ds_path = getattr(self.cfg.env, "eval_dataset_name", self.cfg.env.dataset_name)

        eval_ds_list = getattr(self.cfg.env, "eval_datasets", None)
        if eval_ds_list is None or len(eval_ds_list) == 0:
            eval_ds_list = getattr(self.cfg.env, "datasets", [])

        self.video_ds_name = eval_ds_list[0]

        log(f"[VideoEval] Path: '{eval_ds_path}', Target Dataset: '{self.video_ds_name}'", "blue")

        cfg2 = copy.deepcopy(self.cfg)
        OmegaConf.set_struct(cfg2, False)
        

        dataset_max_len = getattr(self.cfg.env, "max_episode_steps", 32)
        target_video_len = min(65, dataset_max_len)

        cfg2.data.seq_len = target_video_len 
        cfg2.data.batch_length = target_video_len

        cfg2.data.seq_len = 31
        cfg2.data.batch_length = 31
        cfg2.model.context_len = 5  
        cfg2.data.batch_size = 1   
        cfg2.data.shuffle = False 
        cfg2.data.pad_dataset = True

        cfg2.env.dataset_name = eval_ds_path
        cfg2.env.datasets = [self.video_ds_name]

        try:
            ds_dict, *_ = get_dataloader(
                cfg=cfg2,
                dataset_names=[self.video_ds_name],
                dataset_split=[1],
                shuffle=False,
            )
        except tf.errors.NotFoundError:
            log("[VideoEval] cache missing -> retry with use_cache=False", "yellow")
            cfg2.data.use_cache = False
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

    def anneal_temp(self, global_step):

        temp_start = self.cfg.model.arch.world_model.temp_start
        temp_end = self.cfg.model.arch.world_model.temp_end
        decay_steps = self.cfg.model.arch.world_model.temp_decay_steps
        temp = temp_start - (temp_start - temp_end) * (global_step - self.cfg.model.arch.prefill) / decay_steps

        temp = max(temp, temp_end)

        return temp

    def compute_loss(self, batch, train: bool = True):

        if hasattr(batch, 'done') and batch.done is not None:
            done = batch.done
        else:
            done = torch.zeros(batch.actions.shape[:2], device=self.device)

        if hasattr(batch, 'reward') and batch.reward is not None:
            reward = batch.reward
        else:
            reward = torch.zeros(batch.actions.shape[:2], device=self.device)

        traj_dict = {
            'observations': batch.observations,
            'image': batch.observations,
            'timestep': batch.timestep,
            'reward': reward,
            'done': done,
            'action': batch.actions
        }
        global_step = self.train_step
        temp = self.anneal_temp(global_step)
        model_loss, model_logs, prior_state, post_state= self.model.world_model_loss(global_step, traj_dict, temp)
        
        raw_obs = batch.observations
        #log(f"CHECK - Raw Obs Max: {raw_obs.max().item():.3f}, Min: {raw_obs.min().item():.3f}", "red")
        dec_img = model_logs.get('dec_img')
        gt_img = model_logs.get('gt_img')
        
        if dec_img is not None and gt_img is not None:
            # 원본 예측값(recon)의 범위를 알기 위해 0.5를 다시 뺍니다.
            recon_min = dec_img.min() - 0.5
            recon_max = dec_img.max() - 0.5
            gt_min = gt_img.min() - 0.5
            gt_max = gt_img.max() - 0.5
            
            #log(f"DEBUG [Step {global_step}] Recon Range: [{recon_min:.3f}, {recon_max:.3f}]", "yellow")
            #log(f"DEBUG [Step {global_step}] GT Range: [{gt_min:.3f}, {gt_max:.3f}]", "yellow")





        if 'dec_img' in model_logs:
            self.last_vis_data = {
            'dec_img': model_logs['dec_img'],
            'gt_img': model_logs['gt_img']
            }

        metrics = {}
        for k, v in model_logs.items():
            # 비디오 텐서/큰 텐서는 스킵
            if k in ("dec_img", "gt_img"):
                continue

            # dict는 평균낼 수 없으니 무조건 스킵 (ACT_prior_state 등)
            if isinstance(v, dict):
                continue

            # torch tensor
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    metrics[k] = float(v.detach().cpu().item())
                continue  # numel>1 텐서는 스킵

            # python / numpy scalar
            if isinstance(v, (int, float, np.number)):
                metrics[k] = float(v)
                continue

            # 그 외(list/ndarray/str 등)는 스킵
            continue

        total_loss = model_loss
        return metrics, total_loss


    def compute_step_loss(self, batch, train: bool = True):
        clam_output = self.model(
            batch.observations, timesteps=batch.timestep, states=batch.states
        )
        obs_gt = batch.observations[:, 1:]
        cheat_pred = batch.observations[:, :-1]
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


    def log_to_wandb(self, metrics, prefix: str = "", step: int = None):
        if self.wandb_run is None:
            return

        if step is None:
            if hasattr(self, "_wandb_step_override") and self._wandb_step_override is not None:
                step = int(self._wandb_step_override)
            else:
                step = int(self.train_step) + 1

        return super().log_to_wandb(metrics, prefix=prefix, step=step)



    #---- Visualization---
    @torch.no_grad()
    def make_episode_video(self, ds_name: str, save_path: str, fps: int = 8, max_steps: int = 80):
        if self.video_seq_ds is None: return None
        
        it = self.video_seq_ds.as_numpy_iterator()
        try:
            batch_np = next(it)
        except StopIteration: return None

        batch = to_device(batch_np, self.device)
        batch = Batch(**batch) # [B, T, C, H, W]
        
        log(f"[Debug] batch.observations.shape = {batch.observations.shape}", "yellow")

        temp = self.anneal_temp(self.train_step)
        out = self.model(
            observations=batch.observations, 
            timesteps=batch.timestep, 
            gt_action=batch.actions,
            temp=temp,
            training=False
        )
        
        pred = (out['recon_obs'][0] + 0.5).clamp(0, 1) # [T-1, C, H, W]
        gt = batch.observations[0, 1:].clone().float()
        if gt.max() > 1.0: gt /= 255.
        if gt.min() < 0: gt += 0.5
        gt = gt.clamp(0, 1)

        frames = self._create_frames_from_tensors(pred, gt)
        video_obj = self._save_video(frames, save_path, fps)
        return video_obj

    @torch.no_grad()
    def make_dreamer_rollout_video(self, ds_name: str, save_path: str, context_len: int = 5, fps: int = 8, max_steps: int = 80):
        if self.video_seq_ds is None: return None
        
        it = self.video_seq_ds.as_numpy_iterator()
        try:
            batch_np = next(it)
        except StopIteration: return None

        batch = to_device(batch_np, self.device)
        obs = batch['observations'][:, :max_steps]
        actions = batch['actions'][:, :max_steps]
        
        temp = self.anneal_temp(self.train_step)
        
        out = self.model.visualize_diffusion_style_rollout(
            observations=batch['observations'],
            actions=None,  #Dummy
            timesteps=batch['timestep'],
            states=None, 
            temp=temp,
            context_len=5
        )

        pred = (out['pred_video'] + 0.5).clamp(0, 1)
        gt = obs[0, 1:].clone().float()
        if gt.max() > 1.0: gt /= 255.
        if gt.min() < -0.1: gt += 0.5 
        gt = gt.clamp(0, 1)

        # len gt = len pred
        min_t = min(pred.shape[0], gt.shape[0])
        pred = pred[:min_t]
        gt = gt[:min_t]

        log(f"[Debug] Video Frame Count: {min_t}, Expected: {max_steps-1}", "yellow")

        frames = self._create_frames_from_tensors(pred, gt, context_len=context_len)
        return self._save_video(frames, save_path, fps)

    def _create_frames_from_tensors(self, pred, gt, context_len=None):
        frames = []
        diff_all = (pred - gt).abs().mean(dim=1, keepdim=True)
        vmax = diff_all.max().clamp(min=1e-8)
        self._ensure_cmap_lut("magma")
        
        for t in range(pred.shape[0]):
            # Error Map
            diff_t = (diff_all[t] / vmax).clamp(0, 1)
            idx = (diff_t.squeeze(0) * 255).long().clamp(0, 255)
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1)
            
            p_t, g_t = pred[t], gt[t]
            
            if context_len is not None and (t + 1) >= context_len:
                p_t = p_t.clone()
                p_t[0, :2, :] = 1.0; p_t[1:, :2, :] = 0.0 # Red border top
            
            tile = torch.stack([diff_rgb, p_t, g_t], dim=0)
            grid = torchvision.utils.make_grid(tile, nrow=3)
            img = (einops.rearrange(grid, "c h w -> h w c").cpu().numpy() * 255).astype(np.uint8)
            frames.append(img)
        return frames

    def _save_video(self, frames, save_path, fps):
        video_array = np.stack(frames)  # [T, H, W, C]
        video_array = video_array.transpose(0, 3, 1, 2)  # [T, C, H, W]
        
        return wandb.Video(video_array, fps=fps, format="mp4")

    def eval(self, step: int):
        eval_media = {}
        actual_step =int(step)

        _saved_train_step = self.train_step
        self.train_step = actual_step
        self._wandb_step_override = actual_step

        try:
            if hasattr(self, 'last_vis_data'):
                log(f"[Eval @ Step {step}] Logging Training Reconstruction Video...", "blue")
                
                dec = (self.last_vis_data['dec_img'] + 0.5).clamp(0, 1) 
                raw_gt = self.last_vis_data['gt_img']

                if raw_gt.min() < 0:
                    gt = (raw_gt + 0.5).clamp(0, 1)
                else:
                    gt = raw_gt.clamp(0, 1)

                log(f"DEBUG: Video T dim is {dec.shape[1]}")
                            
                num_samples = min(dec.shape[0], 4)
                frames = []

                for t in range(dec.shape[1]):
                    top_row = torchvision.utils.make_grid(gt[:num_samples, t], nrow=num_samples)
                    bottom_row = torchvision.utils.make_grid(dec[:num_samples, t], nrow=num_samples)
                    
                    combined_frame = torch.cat([top_row, bottom_row], dim=1)
                    
                    img = (combined_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    frames.append(img)
                
                #wandb logging
                video_array = np.stack(frames) # [T, H, W, C]
                eval_media["train/reconstruction_comparison"] = wandb.Video(
                    video_array.transpose(0, 3, 1, 2), fps=8, format="mp4"
                )
                del self.last_vis_data

            if self.cfg.env.image_obs:
                log(f"[Eval @ Step {step}] Generating Eval-set Videos...", "blue")
                
                # 1-step prediction (Reconstruction)
                recon_path = f"results/vis/recon_step_{step}.mp4"
                eval_media["videos/reconstruction"] = self.make_episode_video(
                    ds_name=self.video_ds_name, save_path=recon_path, fps=8, max_steps=65
                )
                
                # Open-loop Rollout (Dreaming)
                dream_path = f"results/vis/dreamer_step_{step}.mp4"
                eval_media["videos/dreamer"] = self.make_dreamer_rollout_video(
                    ds_name=self.video_ds_name, save_path=dream_path, context_len=5, fps=8, max_steps=65
                )

            if eval_media:
                self.wandb_run.log(eval_media, step=actual_step, commit=False)
            out = super().eval(step=actual_step)

            log(f"[Eval @ Step {actual_step}] All metrics and media logged.", "green")
            return out

        finally:
            self.train_step = _saved_train_step
            if hasattr(self, "_wandb_step_override"):
                delattr(self, "_wandb_step_override")