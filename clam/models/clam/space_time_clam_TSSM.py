from typing import Tuple, Optional, List

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from clam.models.base import BaseModel
from clam.models.clam.transformer_clam import TransformerCLAM
from clam.models.space_time_attn.models_v2 import STTransformer
from clam.models.space_time_attn.utils import patchify, unpatchify
from clam.models.utils.transformer_utils import get_pos_encoding
from clam.models.utils.utils import CLAMOutput, IDMOutput
from clam.utils.logger import log

from clam.models.vqNSVQ import NSVQ


# =========================================================
# Helpers
# =========================================================
def extract_state_info(states: torch.Tensor):
    pos_goal = states[:, :, -3:]
    curr_obs_and_prev_obs = states[:, :, :-3]
    curr_obs = curr_obs_and_prev_obs[:, :, : int(curr_obs_and_prev_obs.shape[-1] // 2)]
    hand_pos_gripper = curr_obs[:, :, :4]
    return hand_pos_gripper


def _causal_mask(T: int, device):
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


def diag_gaussian_kl(mu_q, logstd_q, mu_p, logstd_p, eps=1e-8):
    std_q = torch.exp(logstd_q).clamp_min(eps)
    std_p = torch.exp(logstd_p).clamp_min(eps)
    var_q = std_q**2
    var_p = std_p**2
    kl = (logstd_p - logstd_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)  # [B,T]


# =========================================================
# World Model (TSSM Style)
# =========================================================
class CLAMWorldModel(nn.Module):
    """
    CLAM용 World Model (TSSM/RSSM 스타일)
      posterior: q(z_t | o_t)
      prior:     p(z_t | z_{t-1}, a_t)  (Transformer로 deter 만들고 prior param)

    output:
      wm_cond: [B, T-1, model_dim]  (FDM에 additive conditioning)
      kl: scalar (Dreamer-style balanced KL)
      pred_loss: scalar (WM prediction/recon loss, e.g., MSE on img_embed)
    """

    def __init__(self, model_dim: int, la_dim: int, cfg: Optional[DictConfig] = None):
        super().__init__()
        cfg = cfg if cfg is not None else {}

        self.z_dim = int(getattr(cfg, "z_dim", 64))
        self.d_model = int(getattr(cfg, "d_model", 256))
        self.n_layers = int(getattr(cfg, "n_layers", 4))
        self.n_heads = int(getattr(cfg, "n_heads", 8))
        self.ff_mult = int(getattr(cfg, "ff_mult", 4))
        self.dropout = float(getattr(cfg, "dropout", 0.0))

        self.kl_scale = float(getattr(cfg, "kl_scale", 1.0))
        self.kl_balance = float(getattr(cfg, "kl_balance", 0.8))
        self.free_nats = float(getattr(cfg, "free_nats", 1.0))
        self.temp = float(getattr(cfg, "temp", 1.0))

        # (2) WM prediction loss scale
        self.pred_scale = float(getattr(cfg, "pred_scale", 1.0))

        # (1) WM positional embedding
        self.max_seq_len = int(getattr(cfg, "max_seq_len", 512))
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)

        # robot_state 사용 안 함
        self.use_robot_state = False
        self.robot_embed = None
        self.obs_fuse = None

        # posterior q(z|o)
        self.post_net = nn.Sequential(
            nn.Linear(model_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.post_mu = nn.Linear(self.d_model, self.z_dim)
        self.post_logstd = nn.Linear(self.d_model, self.z_dim)

        # prior transition transformer: tokens=[z_{t-1}, a_t] -> deter_t
        self.in_proj = nn.Linear(self.z_dim + la_dim, self.d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_mult * self.d_model,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.trans = nn.TransformerEncoder(enc, num_layers=self.n_layers)

        # prior p(z|deter)
        self.prior_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.prior_mu = nn.Linear(self.d_model, self.z_dim)
        self.prior_logstd = nn.Linear(self.d_model, self.z_dim)

        # FDM conditioning feature
        self.feature_proj = nn.Linear(self.z_dim + self.d_model, model_dim)

        # (2) WM recon/pred head: predict img_embed(o_t) from (z_t, deter_t)
        self.pred_net = nn.Sequential(
            nn.Linear(self.z_dim + self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, model_dim),
        )

        self.min_logstd = -6.0
        self.max_logstd = 2.0

    def fuse_observation(self, img_embed: torch.Tensor, robot_state: Optional[torch.Tensor]):
        return img_embed

    def infer_posterior(self, o: torch.Tensor):
        g = self.post_net(o)
        mu = self.post_mu(g)
        logstd = self.post_logstd(g).clamp(self.min_logstd, self.max_logstd)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logstd) * eps * float(self.temp)
        return z, mu, logstd

    def infer_prior_deter(self, prev_z: torch.Tensor, a: torch.Tensor):
        """
        prev_z: [B, T-1, z_dim]
        a:      [B, T-1, la_dim]
        deter:  [B, T-1, d_model]
        """
        tokens = torch.cat([prev_z, a], dim=-1)         # [B, T-1, z+a]
        tokens = self.in_proj(tokens)                   # [B, T-1, d_model]

        # (1) add positional embedding (learned)
        Tm1 = tokens.size(1)
        if Tm1 > self.max_seq_len:
            raise RuntimeError(f"Sequence length {Tm1} exceeds max_seq_len={self.max_seq_len}. Increase wm.max_seq_len.")
        pos = torch.arange(Tm1, device=tokens.device).unsqueeze(0)  # [1, T-1]
        tokens = tokens + self.pos_emb(pos)                          # [B, T-1, d_model]

        mask = _causal_mask(Tm1, tokens.device)
        deter = self.trans(tokens, mask=mask)                        # [B, T-1, d_model]
        return deter

    def infer_prior(self, deter: torch.Tensor):
        g = self.prior_net(deter)
        mu = self.prior_mu(g)
        logstd = self.prior_logstd(g).clamp(self.min_logstd, self.max_logstd)
        return mu, logstd

    def compute_kl(self, post_mu, post_logstd, prior_mu, prior_logstd):
        """
        (6) free_nats를 원소별 clamp 후 mean으로 변경
        """
        kl_lhs = diag_gaussian_kl(post_mu, post_logstd, prior_mu.detach(), prior_logstd.detach())  # [B, T-1]
        kl_rhs = diag_gaussian_kl(post_mu.detach(), post_logstd.detach(), prior_mu, prior_logstd)  # [B, T-1]

        free = kl_lhs.new_tensor(self.free_nats)
        kl_lhs = torch.clamp(kl_lhs, min=free).mean()
        kl_rhs = torch.clamp(kl_rhs, min=free).mean()

        kl = (1.0 - self.kl_balance) * kl_lhs + self.kl_balance * kl_rhs
        return self.kl_scale * kl

    def predict_next_deter(self, z_history: torch.Tensor, a_history: torch.Tensor):
        """
        Rollout용: 과거 히스토리를 받아 다음 스텝의 deter(h_t)를 예측
        z_history: [B, T_past, z_dim] (z_{0:t-1})
        a_history: [B, T_past, la_dim] (a_{1:t})
        return: last_deter [B, 1, d_model]
        """
        deter_seq = self.infer_prior_deter(z_history, a_history)
        return deter_seq[:, -1:]

    def get_cond(self, z_t: torch.Tensor, deter_t: torch.Tensor):
        """
        z_t: [B, 1, z_dim]
        deter_t: [B, 1, d_model]
        """
        return self.feature_proj(torch.cat([z_t, deter_t], dim=-1))

    def forward(
        self,
        img_embed: torch.Tensor,         # [B,T,model_dim]
        la_delta: torch.Tensor,          # [B,T,la_dim] (t=0 pad 포함)
        robot_state: Optional[torch.Tensor] = None,
    ):
        o = self.fuse_observation(img_embed, robot_state)  # [B,T,model_dim]

        # posterior q(z_t|o_t)
        z_post, post_mu, post_logstd = self.infer_posterior(o)     # [B,T,z]

        # prior uses t=1..T-1: p(z_t | z_{t-1}, a_t)
        prev_z = z_post[:, :-1]     # z_{t-1}  [B,T-1,z]
        a = la_delta[:, 1:]         # a_t      [B,T-1,a]
        deter = self.infer_prior_deter(prev_z, a)                  # [B,T-1,d_model]
        prior_mu, prior_logstd = self.infer_prior(deter)           # [B,T-1,z]

        # KL compare at t=1..T-1
        kl = self.compute_kl(post_mu[:, 1:], post_logstd[:, 1:], prior_mu, prior_logstd)

        # conditioning feature for FDM
        # (5번 요청 제외 -> 그대로 z_post 샘플 사용)
        z_t = z_post[:, 1:]                                             # [B,T-1,z]
        wm_cond = self.feature_proj(torch.cat([z_t, deter], dim=-1))     # [B,T-1,model_dim]

        # (2) WM prediction loss: predict o_t(img_embed_t) from (z_t, deter_t)
        pred_o = self.pred_net(torch.cat([z_t, deter], dim=-1))          # [B,T-1,model_dim]
        target_o = o[:, 1:]                                              # [B,T-1,model_dim]
        pred_loss = F.mse_loss(pred_o, target_o)

        dbg = {
            "z_post": z_post,
            "deter": deter,
            "post_mu": post_mu,
            "post_logstd": post_logstd,
            "prior_mu": prior_mu,
            "prior_logstd": prior_logstd,
            "pred_o": pred_o,
        }
        return wm_cond, kl, self.pred_scale * pred_loss, dbg


# =========================================================
# IDM
# =========================================================
class SpaceTimeIDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: Tuple[int, int, int], la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.name = "SpaceTimeIDM"

        self.la_dim = la_dim
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.model_dim = self.cfg.net.dim_model

        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)

        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

        self.input_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.encoder = STTransformer(cfg=self.cfg.net)
        self.activation = nn.LeakyReLU(0.2)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.ln_pre_head = nn.LayerNorm(self.model_dim)

        # (4) IDM pair 방식 action head: (e_{t-1}, e_t) -> a_t
        self.la_pair_head = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.la_dim),
        )

        # VQ를 계속 쓰려면 la_cont도 필요(기존 NSVQ 입력이 la_dim 기준)
        self.la_head = nn.Linear(self.model_dim, self.la_dim)

        # VQ
        self.vq = None
        if self.cfg.quantize_la:
            log("Initializing NSVQ for LAPA", "green")
            vq_kwargs = dict(self.cfg.vq.kwargs)

            if "codebook_size" in vq_kwargs:
                vq_kwargs["num_embeddings"] = vq_kwargs.pop("codebook_size")
            if "eps" in vq_kwargs:
                vq_kwargs.pop("eps")

            vq_kwargs["dim"] = self.la_dim
            vq_kwargs["embedding_dim"] = self.la_dim
            vq_kwargs["image_size"] = 1
            vq_kwargs["patch_size"] = 1
            vq_kwargs["is_vector_input"] = True
            if "discarding_threshold" not in vq_kwargs:
                vq_kwargs["discarding_threshold"] = 0.01
            vq_kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.vq = NSVQ(**vq_kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")

    def forward(self, observations, timesteps: torch.Tensor, states: torch.Tensor = None, **kwargs) -> IDMOutput:
        B, T, *_ = observations.shape
        observations_hwcn = observations.permute(0, 1, 3, 4, 2)  # B,T,H,W,C

        patches = patchify(observations_hwcn, self.cfg.patch_size)          # [B,T,N,patch_dim]
        patches_embed = self.activation(self.input_embed(patches))          # [B,T,N,model_dim]

        N_aug = self.num_patches
        if self.cfg.net.pos_enc == "learned":
            t_pos = self.temporal_pos_embed(timesteps.long())              # [B,T,E]
            t_pos = einops.repeat(t_pos, "B T E -> B T N E", N=N_aug)

            spatial_coord = torch.arange(N_aug, device=patches_embed.device)
            s_pos = self.spatial_pos_embed(spatial_coord.long())           # [N,E]
            s_pos = einops.repeat(s_pos, "N E -> B T N E", B=B, T=T)

            pos_embed = s_pos + t_pos
        else:
            pos_embed = None

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states)
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(hand_pos_gripper_embed, "B T E -> B T N E", N=N_aug)
            patches_embed = self.input_embed_two(torch.cat([patches_embed, hand_pos_gripper_embed], dim=-1))

        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)  # [B,T,N,E]
        z = z.view(B, T, -1, self.model_dim)

        la_z = z.mean(dim=2)                 # [B,T,model_dim]
        la_z = self.ln_pre_head(la_z)        # img_embed로 쓸 값

        # (4) IDM pair action (continuous)
        pad_la = torch.zeros(B, 1, self.la_dim, device=observations.device)
        if T > 1:
            e_prev = la_z[:, :-1]                                  # [B,T-1,E]
            e_curr = la_z[:, 1:]                                   # [B,T-1,E]
            pair = torch.cat([e_prev, e_curr], dim=-1)             # [B,T-1,2E]
            a_pred = self.la_pair_head(pair)                       # [B,T-1,la_dim]
            la_final = torch.cat([pad_la, a_pred], dim=1)          # [B,T,la_dim]
        else:
            la_final = pad_la

        # VQ 입력용 la_cont (기존 NSVQ 설계 유지)
        la_cont = self.la_head(la_z)                                # [B,T,la_dim]

        vq_outputs, vq_metrics = {}, {}

        # (3) vq_loss 실제 연결
        vq_loss = torch.tensor(0.0, device=observations.device)

        if self.cfg.quantize_la and self.vq is not None and T > 1:
            e_cur = la_cont[:, :-1]                                 # [B,T-1,la_dim]
            e_nxt = la_cont[:, 1:]                                  # [B,T-1,la_dim]
            flat_cur = e_cur.reshape(-1, self.la_dim)
            flat_nxt = e_nxt.reshape(-1, self.la_dim)

            quant_flat, perplexity, vq_aux, idx_flat = self.vq(
                input_data_first=flat_cur,
                input_data_last=flat_nxt,
                codebook_training_only=False,
            )

            quant_delta = quant_flat.view(B, T - 1, self.la_dim)
            indices = idx_flat.view(B, T - 1)

            # VQ 켜면 action은 quant_delta로 교체(기존 동작 유지)
            la_final = torch.cat([pad_la, quant_delta], dim=1)

            pad_idx = torch.zeros(B, 1, dtype=torch.long, device=observations.device)
            vq_outputs["indices"] = torch.cat([pad_idx, indices], dim=1)
            vq_metrics["perplexity"] = float(perplexity.item())

            # NSVQ가 loss/aux를 주면 vq_loss로 사용
            if isinstance(vq_aux, torch.Tensor):
                vq_loss = vq_aux.mean()
            elif isinstance(vq_aux, (list, tuple)) and len(vq_aux) > 0 and isinstance(vq_aux[0], torch.Tensor):
                vq_loss = torch.stack([x.mean() for x in vq_aux]).mean()

        # (7) 큰 텐서를 output dict에 저장할 때는 detach해서 그래프 누수 방지
        # WM 입력으로는 그대로 쓰되, 저장은 detach.
        vq_outputs["img_embed"] = la_z.detach()

        return IDMOutput(
            la=la_final,
            quantized_la=la_final,
            vq_loss=vq_loss,
            vq_metrics=vq_metrics,
            vq_outputs=vq_outputs,
            encoder_out=patches,   # FDM에 원본 패치 전달
        )


# =========================================================
# FDM (wm_cond additive conditioning)
# =========================================================
class SpaceTimeFDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int, use_vq: bool = False):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim

        self.patch_token_dim = C * self.cfg.patch_size**2
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.model_dim = self.cfg.net.dim_model

        self.decoder = STTransformer(cfg=self.cfg.net)
        self.patch_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.la_embed = nn.Linear(la_dim, self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200)
        self.temporal_pos_embed = get_pos_encoding(self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200)
        self.cond_pos_embed = get_pos_encoding(self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200)

        self.to_recon = nn.Linear(self.model_dim, self.patch_token_dim)

        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

    def forward(
        self,
        observations,
        idm_output: IDMOutput,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        wm_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, T, C, H, W = observations.shape

        la = idm_output.la[:, 1:]                   # [B,T-1,la_dim]
        patches = idm_output.encoder_out[:, :-1]    # [B,T-1,N,patch_dim]
        patches = patches.detach()                  # stop-grad

        la_embed = self.la_embed(la)                # [B,T-1,E]
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        patches_embed = self.patch_embed(patches)   # [B,T-1,N,E]
        video_action_patches = la_embed + patches_embed

        # ✅ world model conditioning
        if wm_cond is not None:
            video_action_patches = video_action_patches + wm_cond.unsqueeze(2)

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states[:, :-1])
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches)
            video_action_patches = self.input_embed_two(torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1))

        _, Tm1, N, _ = video_action_patches.shape

        if self.cfg.net.pos_enc == "learned":
            t_pos = self.temporal_pos_embed(timesteps[:, :-1].long())
            t_pos = einops.repeat(t_pos, "B T E -> B T N E", N=N)

            spatial_coord = torch.arange(N, device=video_action_patches.device)
            s_pos = self.spatial_pos_embed(spatial_coord.long())
            s_pos = einops.repeat(s_pos, "N E -> B T N E", B=B, T=Tm1)

            pos_embed = s_pos + t_pos

            cond_pos = self.cond_pos_embed(timesteps[:, 1:].long())
            cond_pos = einops.repeat(cond_pos, "B T E -> B T N E", N=N)
        else:
            pos_embed = None
            cond_pos = None

        video_recon = self.decoder(
            video_action_patches,
            pos_embed=pos_embed,
            causal=True,
            cond=la_embed,
            cond_pos_embed=cond_pos,
        )

        video_recon = self.to_recon(video_recon)
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        return video_recon


# =========================================================
# CLAM Wrapper (forward + rollout)
# =========================================================
class SpaceTimeCLAM_TSSM(TransformerCLAM):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.name = "ST-CLAM_NSVQ_WM"
        self.la_dim = la_dim

        self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

        self.use_wm = bool(getattr(cfg, "use_world_model", True))
        wm_cfg = getattr(cfg, "wm", None)

        if self.use_wm:
            self.world_model = CLAMWorldModel(
                model_dim=cfg.idm.net.dim_model,
                la_dim=la_dim,
                cfg=wm_cfg,
            )
            log("World Model ENABLED (posterior/prior + KL + pred loss)", "green")
        else:
            self.world_model = None
            log("World Model DISABLED", "yellow")

        self.add_wm_kl_to_vq_loss = bool(getattr(cfg, "add_wm_kl_to_vq_loss", True))
        self.add_wm_pred_to_vq_loss = bool(getattr(cfg, "add_wm_pred_to_vq_loss", True))

    def forward(
        self,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        **kwargs,
    ) -> CLAMOutput:

        idm_output = self.idm(observations=observations, timesteps=timesteps, states=states)

        wm_cond = None
        wm_kl = observations.new_tensor(0.0)
        wm_pred = observations.new_tensor(0.0)

        if self.use_wm and (self.world_model is not None):
            img_embed = idm_output.vq_outputs.get("img_embed", None)
            if img_embed is None:
                raise RuntimeError("IDMOutput.vq_outputs['img_embed'] missing. IDM must store it.")

            wm_cond, wm_kl, wm_pred, _dbg = self.world_model(
                img_embed=img_embed,
                la_delta=idm_output.la,
                robot_state=None,   # ✅ robot_state 안 씀
            )

            if isinstance(idm_output.vq_metrics, dict):
                idm_output.vq_metrics["wm_kl"] = float(wm_kl.detach().cpu())
                idm_output.vq_metrics["wm_pred"] = float(wm_pred.detach().cpu())

            # (2) WM loss들을 학습 loss에 포함
            new_vq_loss = idm_output.vq_loss
            if self.add_wm_kl_to_vq_loss:
                new_vq_loss = new_vq_loss + wm_kl
            if self.add_wm_pred_to_vq_loss:
                new_vq_loss = new_vq_loss + wm_pred

            idm_output = IDMOutput(
                la=idm_output.la,
                quantized_la=idm_output.quantized_la,
                vq_loss=new_vq_loss,
                vq_metrics=idm_output.vq_metrics,
                vq_outputs=idm_output.vq_outputs,
                encoder_out=idm_output.encoder_out,
            )

        recon = self.fdm(
            observations=observations,
            idm_output=idm_output,
            timesteps=timesteps,
            states=states,
            wm_cond=wm_cond,
        )

        return CLAMOutput(
            la=idm_output.la,
            reconstructed_obs=recon,
            idm_output=idm_output,
        )

    # rollout/visualize 함수들은 그대로 두되,
    # IDM이 action을 pair-head로 생성하는 것으로 의미만 바뀜(인터페이스 동일).
    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,
        gt_states: torch.Tensor = None,
        max_steps: int | None = None,
    ):
        use_gt_image = True
        use_gt_action = False

        device = gt_seq.device
        T_total, C, H, W = gt_seq.shape
        max_steps = min(max_steps, T_total - 1) if max_steps else T_total - 1

        recons: List[torch.Tensor] = []

        history_z = []
        history_a = []

        def _postprocess_frame(x: torch.Tensor) -> torch.Tensor:
            return x.clamp(-1, 1)

        def _encode_obs_to_z(idm_out):
            if self.world_model is None:
                return None
            img_embed = idm_out.vq_outputs.get("img_embed", None)
            z_post, _, _ = self.world_model.infer_posterior(img_embed)
            return z_post

        def _make_step_output(idm_out_full: IDMOutput, la_override: torch.Tensor) -> IDMOutput:
            return IDMOutput(
                la=la_override,
                quantized_la=la_override,
                vq_loss=getattr(idm_out_full, "vq_loss", gt_seq.new_tensor(0.0)),
                vq_metrics=getattr(idm_out_full, "vq_metrics", {}),
                vq_outputs=getattr(idm_out_full, "vq_outputs", {}),
                encoder_out=getattr(idm_out_full, "encoder_out", None),
            )

        pair0 = gt_seq[0:2].unsqueeze(0)
        ts0 = torch.tensor([[0, 1]], device=device)

        idm_out0 = self.idm(pair0, timesteps=ts0, states=None)

        la0 = idm_out0.la
        if hasattr(self, "cfg") and getattr(self.cfg, "zero_action", False):
            la0 = torch.zeros_like(la0)

        wm_cond0 = None
        if self.use_wm:
            z0 = _encode_obs_to_z(idm_out0)  # [1, 2, z]
            history_z.append(z0)
            history_a.append(la0)

            curr_z_hist = z0[:, :-1]
            curr_a_hist = la0[:, 1:]

            deter0 = self.world_model.predict_next_deter(curr_z_hist, curr_a_hist)
            wm_cond0 = self.world_model.get_cond(z0[:, -1:], deter0)

        idm_step0 = _make_step_output(idm_out0, la0)
        recon0 = self.fdm(pair0, idm_step0, ts0, states=None, wm_cond=wm_cond0)
        recons.append(_postprocess_frame(recon0[0, -1]))

        for t_curr in range(2, max_steps + 1):
            t_prev = t_curr - 1
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)

            if use_gt_image:
                pair_input = gt_seq[t_prev : t_curr + 1].unsqueeze(0)
            else:
                if len(recons) == 1:
                    prev_img = gt_seq[1]
                    curr_img = recons[-1]
                else:
                    prev_img = recons[-2]
                    curr_img = recons[-1]
                pair_input = torch.stack([prev_img, curr_img], dim=0).unsqueeze(0)

            idm_out = self.idm(pair_input, timesteps=ts_pair, states=None)
            current_la = idm_out.la

            if use_gt_action:
                pass

            wm_cond = None
            if self.use_wm:
                z_curr_pair = _encode_obs_to_z(idm_out)  # [1, 2, z]
                z_t = z_curr_pair[:, -1:]
                a_t = current_la[:, -1:]

                history_z.append(z_t)
                history_a.append(a_t)

                full_z = torch.cat(history_z, dim=1)
                full_a = torch.cat(history_a, dim=1)

                inp_z = full_z[:, :-1]
                inp_a = full_a[:, 1:]

                deter_t = self.world_model.predict_next_deter(inp_z, inp_a)
                wm_cond = self.world_model.get_cond(full_z[:, -1:], deter_t)

            idm_step = _make_step_output(idm_out, current_la)
            recon_next = self.fdm(pair_input, idm_step, ts_pair, states=None, wm_cond=wm_cond)

            recons.append(_postprocess_frame(recon_next[0, -1]))

        return torch.stack(recons, dim=0) if recons else torch.empty((0, C, H, W), device=device)

    @torch.no_grad()
    def visualize_dreamer_style_rollout(
        self,
        gt_seq: torch.Tensor,
        context_len: int = 5,
    ):
        device = gt_seq.device
        T_total, C, H, W = gt_seq.shape

        gt_batch = gt_seq.unsqueeze(0)
        ts_full = torch.arange(T_total, device=device).unsqueeze(0)
        idm_out_full = self.idm(gt_batch, timesteps=ts_full, states=None)
        gt_actions = idm_out_full.la

        recons: List[torch.Tensor] = []
        history_z = []
        history_a = []

        def _postprocess_frame(x: torch.Tensor) -> torch.Tensor:
            return x.clamp(-1, 1)

        def _get_posterior_z(idm_out):
            if self.world_model is None:
                return None
            img_embed = idm_out.vq_outputs.get("img_embed", None)
            z_post, _, _ = self.world_model.infer_posterior(img_embed)
            return z_post

        def _sample_prior_z(deter):
            prior_mu, prior_logstd = self.world_model.infer_prior(deter)
            eps = torch.randn_like(prior_mu)
            z_prior = prior_mu + torch.exp(prior_logstd) * eps
            return z_prior

        def _make_step_output(idm_out_full, la_override):
            return IDMOutput(
                la=la_override,
                quantized_la=la_override,
                vq_loss=torch.tensor(0.0, device=device),
                vq_metrics={},
                vq_outputs=getattr(idm_out_full, "vq_outputs", {}),
                encoder_out=getattr(idm_out_full, "encoder_out", None),
            )

        pair0 = gt_seq[0:2].unsqueeze(0)
        ts0 = torch.tensor([[0, 1]], device=device)
        idm_out0 = self.idm(pair0, timesteps=ts0, states=None)

        la0 = torch.zeros_like(idm_out0.la)
        la0[:, 1:2] = gt_actions[:, 1:2]

        wm_cond0 = None
        if self.use_wm:
            z0 = _get_posterior_z(idm_out0)
            history_z.append(z0[:, 0:1])
            history_a.append(la0[:, 0:1])

            deter0 = self.world_model.predict_next_deter(z0[:, 0:1], la0[:, 1:2])
            wm_cond0 = self.world_model.get_cond(z0[:, 1:2], deter0)

            history_z.append(z0[:, 1:2])
            history_a.append(la0[:, 1:2])

        idm_step0 = _make_step_output(idm_out0, la0)
        recon0 = self.fdm(pair0, idm_step0, ts0, states=None, wm_cond=wm_cond0)
        recons.append(_postprocess_frame(recon0[0, -1]))

        for t_curr in range(2, T_total):
            t_prev = t_curr - 1
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)
            is_context = t_curr < context_len

            if is_context:
                pair_input = gt_seq[t_prev : t_curr + 1].unsqueeze(0)
            else:
                if len(recons) == 1:
                    prev_gen = gt_seq[1]
                    curr_gen = recons[-1]
                else:
                    prev_gen = recons[-2]
                    curr_gen = recons[-1]
                pair_input = torch.stack([prev_gen, curr_gen], dim=0).unsqueeze(0)

            idm_out = self.idm(pair_input, timesteps=ts_pair, states=None)

            la_override = torch.zeros_like(idm_out.la)
            la_override[:, 1:2] = gt_actions[:, t_curr : t_curr + 1]

            wm_cond = None
            if self.use_wm:
                a_t = la_override[:, 1:2]

                full_z = torch.cat(history_z, dim=1)
                full_a = torch.cat(history_a, dim=1)

                inp_z = full_z
                inp_a = torch.cat([full_a[:, 1:], a_t], dim=1)

                deter_t = self.world_model.predict_next_deter(inp_z, inp_a)

                if is_context:
                    z_curr_pair = _get_posterior_z(idm_out)
                    z_t = z_curr_pair[:, -1:]
                else:
                    z_t = _sample_prior_z(deter_t)

                history_z.append(z_t)
                history_a.append(a_t)

                wm_cond = self.world_model.get_cond(z_t, deter_t)

            idm_step = _make_step_output(idm_out, la_override)
            recon_next = self.fdm(pair_input, idm_step, ts_pair, states=None, wm_cond=wm_cond)

            recons.append(_postprocess_frame(recon_next[0, -1]))

        return torch.stack(recons, dim=0)
