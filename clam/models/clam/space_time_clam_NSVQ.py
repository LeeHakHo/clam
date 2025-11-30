from typing import Tuple

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from clam.models.base import BaseModel
from clam.models.clam.clam import get_vq_cls
from clam.models.clam.transformer_clam import TransformerCLAM
from clam.models.space_time_attn.models_v2 import STTransformer
from clam.models.space_time_attn.utils import patchify, unpatchify
from clam.models.utils.transformer_utils import get_pos_encoding
from clam.models.utils.utils import CLAMOutput, IDMOutput, compute_perplexity
from clam.utils.logger import log

# 사용자님이 정의하신 NSVQ 클래스 import
# (같은 파일에 있다면 import 불필요, 다른 파일이라면 경로 맞춰주세요)
from clam.models.vqNSVQ import NSVQ 


def extract_state_info(states: torch.Tensor):
    pos_goal = states[:, :, -3:]
    curr_obs_and_prev_obs = states[:, :, :-3]
    curr_obs = curr_obs_and_prev_obs[:, :, : int(curr_obs_and_prev_obs.shape[-1] // 2)]
    hand_pos_gripper = curr_obs[:, :, :4]
    return hand_pos_gripper

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

        self.action_in = nn.Parameter(torch.randn(1, 1, 1, self.model_dim))

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.la_head = nn.Linear(self.model_dim, self.la_dim)

        # ----------------- VQ Init -----------------
        self.vq = None
        if self.cfg.quantize_la:
            log(f"Initializing NSVQ for LAPA", "green")
            vq_kwargs = dict(self.cfg.vq.kwargs)
            
            if "codebook_size" in vq_kwargs:
                vq_kwargs["num_embeddings"] = vq_kwargs.pop("codebook_size")
            
            if "eps" in vq_kwargs:
                vq_kwargs.pop("eps")
            
            # 1x1 Trick
            fake_size = 32
            vq_kwargs["image_size"] = fake_size
            vq_kwargs["patch_size"] = fake_size
            
            vq_kwargs["dim"] = self.model_dim
            vq_kwargs["embedding_dim"] = self.cfg.la_dim
            
            # [Safe Init]
            vq_kwargs["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
            self.vq = NSVQ(**vq_kwargs) # 또는 NSVQ
        else:
            log("Not using vq, continuous latent action space", "red")

    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        
        B, T, *_ = observations.shape
        observations = observations.permute(0, 1, 3, 4, 2)
        patches = patchify(observations, self.cfg.patch_size)
        patches_embed = self.input_embed(patches)
        patches_embed = self.activation(patches_embed)

        if self.cfg.add_action_token:
            action_pad = self.action_in.expand(B, T, 1, self.model_dim)
            patches_embed = torch.cat([action_pad, patches_embed], dim=2)
            N_aug = self.num_patches + 1
        else:
            N_aug = self.num_patches

        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps.long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N_aug)
            
            spatial_coord = torch.arange(N_aug).to(patches_embed.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(spatial_pos_embed, "N E -> B T N E", B=B, T=T)
            
            pos_embed = spatial_pos_embed + t_pos_embed
        else:
            pos_embed = None

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states)
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=N_aug
            )
            patches_embed = self.input_embed_two(
                torch.cat([patches_embed, hand_pos_gripper_embed], dim=-1)
            )

        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)
        z = z.view(B, T, -1, self.model_dim)

        if self.cfg.add_action_token:
            la_z = z[:, :, 0]
        else:
            la_z = z.mean(dim=2)

        la_continuous = self.la_head(la_z)

        # ----------------- NSVQ Forward -----------------
        vq_loss = torch.tensor(0.0, device=observations.device)
        quantized_la = None
        indices = None
        vq_outputs = {}
        vq_metrics = {}

        if self.cfg.quantize_la and self.vq is not None:
            # [CRITICAL FIX]: Reshape to (N, 1, Dim) for NSVQ encode (permute expects 3 dims)
            first = la_z[:, :-1].reshape(-1, 1, self.model_dim)
            last  = la_z[:, 1:].reshape(-1, 1, self.model_dim)

            q_la, perplexity, _, idx_flat = self.vq(
                input_data_first=first,
                input_data_last=last,
                codebook_training_only=False
            )
            
            # --- Output Handling ---
            # idx_flat: [N, 1] or [N] -> flatten first
            idx_flat = idx_flat.reshape(-1) 
            
            idx_reshaped = idx_flat.view(B, T-1) 
            
            # q_la: [N, 1, Dim] -> [B, T-1, Dim]
            q_la_reshaped = q_la.view(B, T-1, -1)

            # Padding (T-1 -> T)
            padding_val = torch.zeros(B, 1, q_la_reshaped.shape[-1], device=q_la_reshaped.device)
            la_final = torch.cat([padding_val, q_la_reshaped], dim=1) 
            
            padding_idx = torch.zeros(B, 1, dtype=idx_reshaped.dtype, device=q_la_reshaped.device)
            indices = torch.cat([padding_idx, idx_reshaped], dim=1)

            vq_outputs = {"indices": indices}
            vq_metrics = {"perplexity": perplexity.item()}
            
            return IDMOutput(
                la=la_final,          
                quantized_la=la_final, 
                vq_loss=vq_loss,
                vq_metrics=vq_metrics,
                vq_outputs=vq_outputs,
                encoder_out=patches,
            )
            
        return IDMOutput(la=la_continuous, encoder_out=patches)

class SpaceTimeFDM(BaseModel):
    # [수정] use_vq 인자 추가
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int, use_vq: bool = False):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.model_dim = self.cfg.net.dim_model

        self.decoder = STTransformer(cfg=self.cfg.net)
        self.patch_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        
        # [수정] NSVQ를 쓴다면 입력 차원은 model_dim (NSVQ가 project_out을 하므로)
        input_la_dim = self.model_dim if use_vq else la_dim
        self.la_embed = nn.Linear(input_la_dim, self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.cond_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.to_recon = nn.Linear(self.model_dim, self.patch_token_dim)

        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

    def forward(
        self,
        observations,
        idm_output: IDMOutput,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ) -> CLAMOutput:
        
        B, T, C, H, W = observations.shape

        # IDMOutput에서 VQ를 거친 Latent 사용 (T=0 패딩 제외)
        la = idm_output.la[:, 1:] # [B, T-1, Dim]
        patches = idm_output.encoder_out

        la_embed = self.la_embed(la)
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        patches_embed = self.patch_embed(patches[:, :-1])
        video_action_patches = la_embed + patches_embed
        #video_action_patches = patches_embed

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states[:, :-1])
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches
            )
            video_action_patches = self.input_embed_two(
                torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1)
            )

        B, T_minus_one, N, E = video_action_patches.shape

        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps[:, :-1].long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N)
        else:
            t_pos_embed = None

        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N).to(video_action_patches.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T - 1
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed if (
            spatial_pos_embed is not None and t_pos_embed is not None
        ) else None

        if self.cfg.net.pos_enc == "learned":
            cond_pos_embed = self.cond_pos_embed(timesteps[:, 1:].long())
            cond_pos_embed = einops.repeat(cond_pos_embed, "B T E -> B T N E", N=N)
        else:
            cond_pos_embed = None

        video_recon = self.decoder(
            video_action_patches,
            pos_embed=pos_embed,
            causal=True,
            cond=la_embed,
            cond_pos_embed=cond_pos_embed,
        )

        video_recon = self.to_recon(video_recon)
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        return video_recon


class SpaceTimeCLAM(BaseModel):
    """
    ST-CLAM with NSVQ (LAPA-style) latent actions.
    TransformerCLAM 에 상속/의존 안 하고, BaseModel에서 바로 시작해서
    IDM -> FDM forward 를 직접 정의하는 버전.
    """

    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        BaseModel.__init__(self, cfg=cfg, input_dim=input_dim)

        self.cfg = cfg
        self.name = "ST-ViVit"
        self.la_dim = la_dim

        # 우리가 정의한 SpaceTimeIDM / SpaceTimeFDM 사용
        self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)

        # NSVQ를 쓰면 FDM 쪽 latent 차원이 model_dim 기준이 되므로 use_vq 플래그 전달
        use_vq = cfg.idm.quantize_la if hasattr(cfg, "idm") else False
        self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim, use_vq=use_vq)

    def forward(
        self,
        observations: torch.Tensor,   # [B, T, C, H, W]
        timesteps: torch.Tensor,      # [B, T]
        states: torch.Tensor = None,  # [B, T, D] or None
        **kwargs,
    ) -> CLAMOutput:
        """
        원래 TransformerCLAM.forward 가 하던 걸
        SpaceTimeIDM/FDM 기반으로 직접 구현한 버전.
        """

        # 1) IDM: latent action (la) + encoder_out (patches) 생성
        idm_output = self.idm(
            observations=observations,
            timesteps=timesteps,
            states=states,
        )

        # 2) FDM: la + patches 로 다음 프레임 재구성
        recon = self.fdm(
            observations=observations,
            idm_output=idm_output,
            timesteps=timesteps,
            states=states,
        )

        # 3) CLAMOutput 으로 래핑
        #    trainer 쪽에서 쓰는 필드는:
        #      - reconstructed_obs
        #      - la
        #      - idm_output (안에 vq_loss, vq_metrics, vq_outputs 등)
        return CLAMOutput(
            la=idm_output.la,
            reconstructed_obs=recon,
            idm_output=idm_output,
        )

    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,    # (T, C, H, W)
        gt_states: torch.Tensor, # (T, D) or None
        max_steps: int | None = None,
    ):
        """
        네가 이미 만들어둔 rollout 함수 그대로 써도 되고,
        아래처럼 SpaceTimeIDM/FDM 기반으로만 돌아가면 됨.
        """
        use_gt_image = True
        use_gt_action = True
        use_zero_action = False
        
        device = gt_seq.device
        T, C, H, W = gt_seq.shape

        if max_steps is None:
            max_steps = T - 1
        else:
            max_steps = min(max_steps, T - 1)

        recons = []

        def sample_la(idm_out):
            return idm_out.la  # NSVQ가 적용된 la

        # step 1: GT 2프레임으로 워밍업
        if max_steps >= 1:
            pair0 = gt_seq[0:2].unsqueeze(0)  # (1,2,C,H,W)
            ts0   = torch.tensor([[0, 1]], device=device)  # (1,2)
            state0 = None if gt_states is None else gt_states[0:2].unsqueeze(0)

            idm_out0 = self.idm(pair0, timesteps=ts0, states=state0)
            la0 = sample_la(idm_out0)
            if use_zero_action:
                la0 = torch.zeros_like(la0)

            idm_step0 = IDMOutput(la=la0, encoder_out=idm_out0.encoder_out)
            recon0 = self.fdm(pair0, idm_step0, ts0, state0)
            recons.append((torch.tanh(recon0[0, -1]) + 1) / 2)

        # step 2..max_steps: closed-loop rollout
        for step in range(2, max_steps + 1):
            t_prev, t_curr = step - 1, step
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)

            state_pair = None
            if gt_states is not None:
                state_pair = gt_states[step-2: step].unsqueeze(0)

            pair_gt = gt_seq[step-2: step].unsqueeze(0)

            if step == 2:
                prev_img, curr_img = gt_seq[1], recons[-1]
            else:
                prev_img, curr_img = recons[-2], recons[-1]

            pair_gen = torch.stack([prev_img, curr_img], dim=0).unsqueeze(0)

            pair_for_idm = pair_gt if use_gt_image else pair_gen

            idm_out = self.idm(pair_for_idm, timesteps=ts_pair, states=state_pair)

            if use_zero_action:
                la_for_fdm = torch.zeros_like(idm_out.la)
            elif use_gt_action:
                idm_out_gt = self.idm(pair_gt, timesteps=ts_pair, states=state_pair)
                la_for_fdm = sample_la(idm_out_gt)
            else:
                la_for_fdm = sample_la(idm_out)

            idm_step = IDMOutput(la=la_for_fdm, encoder_out=idm_out.encoder_out)
            recon_next = self.fdm(pair_for_idm, idm_step, ts_pair, state_pair)
            recons.append((torch.tanh(recon_next[0, -1]) + 1) / 2)

        if len(recons) > 0:
            recons = torch.stack(recons, dim=0)
        else:
            recons = torch.empty((0, C, H, W), device=device)

        return recons
