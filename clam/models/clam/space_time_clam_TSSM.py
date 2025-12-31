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
from collections import defaultdict
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Independent
from clam.models.modules_transformer import TransformerWorldModel, DenseDecoder, ActionDecoder
import pdb


class ActionDecoderV3Full(nn.Module):
    def __init__(
        self, 
        input_size,    # s_t 차원 (h_t + z_t)
        action_size,   # env.action_size
        layers=5,      #
        units=1024,    #
        act='silu',    #
        norm='rms',    #
        unimix=0.01,   #
        outscale=1.0   #
    ):
        super().__init__()
        self._unimix = unimix
        self._action_size = action_size

        # 1. MLP 구조 (JAX의 self.mlp 부분)
        model = []
        for i in range(layers):
            in_dim = input_size if i == 0 else units
            # Dreamer V3 공식 구현 스타일: Linear (bias=True) + RMSNorm
            line = nn.Linear(in_dim, units, bias=True)
            
            # 초기화 로직 이식: trunc_normal
            nn.init.trunc_normal_(line.weight, std=0.02)
            if line.bias is not None:
                nn.init.zeros_(line.bias)
                
            model.append(line)
            
            # RMSNorm (JAX의 'rms' 매칭) 
            model.append(nn.RMSNorm(units))
            
            # SiLU 활성화 (JAX의 'silu' 매칭) 
            model.append(nn.SiLU())
        
        self.mlp = nn.Sequential(*model)

        # 2. Head 출력층 (JAX의 self.head.onehot 부분)
        self.out = nn.Linear(units, action_size, bias=True)
        
        # 출력층 초기화 (outscale 적용)
        nn.init.trunc_normal_(self.out.weight, std=0.02 * outscale)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, features):
        """
        features: RSSM의 post_state로부터 얻은 {h_t, z_t} 결합체 [cite: 204]
        shape: [Batch, Time, D]
        """
        # 1. MLP 통과
        x = self.mlp(features)
        
        # 2. 로짓 계산
        logits = self.out(x)
        
        # 3. Unimix 적용 (Categorical 분포 안정화) [cite: 621, 678]
        # 1%의 균등 분포를 섞어 log(0) 발생 방지
        probs = torch.softmax(logits, dim=-1)
        if self._unimix > 0:
            probs = (1.0 - self._unimix) * probs + self._unimix / self._action_size
            
        return OneHotCategorical(probs=probs)


class SpaceTimeCLAM_TSSM(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim, la_dim):
        super().__init__()

        self.world_model = TransformerWorldModel(cfg)

        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        d_model = cfg.arch.world_model.transformer.d_model
        self.d_model = d_model
        deter_type = cfg.arch.world_model.transformer.deter_type
        n_layers = cfg.arch.world_model.transformer.n_layers
        if deter_type == 'concat_o':
            d_model = n_layers * d_model

        if self.stoch_discrete:
            dense_input_size = d_model + self.stoch_size * self.stoch_discrete
        else:
            dense_input_size = d_model + self.stoch_size
        self.aggregator = cfg.arch.actor.aggregator
        if self.aggregator == 'attn':
            dense_input_size = dense_input_size + self.d_model
        self.actor = ActionDecoder(dense_input_size, cfg.env.action_size, cfg.arch.actor.layers, cfg.arch.actor.num_units,
                                dist=cfg.arch.actor.dist, init_std=cfg.arch.actor.init_std, act=cfg.arch.actor.act)

        self.value = DenseDecoder(dense_input_size, cfg.arch.value.layers, cfg.arch.value.num_units, (1,), act=cfg.arch.value.act)
        self.slow_value = DenseDecoder(dense_input_size, cfg.arch.value.layers, cfg.arch.value.num_units, (1,), act=cfg.arch.value.act)

        self.discount = cfg.rl.discount
        self.lambda_ = cfg.rl.lambda_

        self.actor_loss_type = cfg.arch.actor.actor_loss_type
        self.pcont_scale = cfg.loss.pcont_scale
        self.kl_scale = cfg.loss.kl_scale
        self.kl_balance = cfg.loss.kl_balance
        self.free_nats = cfg.loss.free_nats
        self.H = cfg.arch.H
        self.grad_clip = cfg.optimize.grad_clip
        self.action_size = cfg.env.action_size
        self.log_every_step = cfg.train.log_every_step
        self.batch_length = cfg.train.batch_length
        self.grayscale = cfg.env.grayscale
        self.slow_update = 0
        self.n_sample = cfg.train.n_sample
        self.imag_last_T = cfg.train.imag_last_T
        self.slow_update_step = cfg.slow_update_step
        self.reward_layer = cfg.arch.world_model.reward_layer
        self.log_grad = cfg.train.log_grad
        self.ent_scale = cfg.loss.ent_scale
        self.action_dist = cfg.arch.actor.dist

        self.r_transform = dict(
            tanh=torch.tanh,
            sigmoid=torch.sigmoid,
            none=torch.nn.Identity(),
        )[cfg.rl.r_transform]

        #va_encoder
        shapes = {'image': tuple(input_dim)}
        #self.va_encoder = va_net.VANet(shapes, **cfg.vanet)


        self.pol = ActionDecoderV3Full(dense_input_size, self.action_size)

    
    def va_encode(self, data):
        if self.va_encoder._va_method == 'flow':
            va_action_state = self.va_encoder(None,data['flow'][:,1:])
        else:
            va_action_state = self.va_encoder(data['image'][:,:-1],data['image'][:,1:])
        if self.va_encoder.type == 'mix':
            va_action =  torch.cat([va_action_state['deter'],va_action_state['stoch']],-1)
        else:
            va_action = va_action_state[self.va_encoder.type]
        return va_action_state, self.va_encoder.append_action(va_action, ahead=True)

    def forward(self, observations, timesteps: torch.Tensor, temp, state=None, gt_action=None, done=None, training=True, context_len=49):
        B, T, C, H, W = observations.shape
        device = observations.device
        
        obs_emb = self.world_model.dynamic.img_enc(observations / 255. - 0.5)

        #s_t - stacked
        post_state = self.world_model.dynamic.infer_post_stoch(obs_emb, temp)

        #action padding for first step
        z_actions = torch.zeros(B, T, self.action_size, device=device)

        for t in range(1, T):
            prev_stoch = post_state['stoch'][:, :t]

            if gt_action is not None:
                current_action_input = gt_action[:, :t]
            else:
                current_action_input = z_actions[:, :t]

            #h_t
            prior_step = self.world_model.dynamic.infer_prior_stoch(
                prev_stoch, temp, actions=current_action_input
            )

            deter_t = prior_step['deter'][:, -1] # (h'_t|s_t-1, z_t-1) - predicted
            deter_t = deter_t.squeeze(1)

            stoch_t = post_state['stoch'][:, t].reshape(B, -1) # (h_t| o_t, h_t)

            h_t = torch.cat([deter_t, stoch_t], dim=-1)
            z_action_dist = self.pol(h_t)
            z_action = z_action_dist.sample() if training else z_action_dist.mode

            if t < T - 1:   
                z_actions[:, t] = z_action

        post_state['deter'] = prior_step['deter']
        post_state['o_t'] = prior_step['o_t']

        return {
            'prior_state': prior_step, # end point
            'post_state': post_state,
            'action_state': z_action_dist,
            'pred_action': z_actions
        }


    def write_logs(self, logs, traj, global_step, writer, tag='train', min_idx=None):
        rec_img = logs['dec_img']
        gt_img = logs['gt_img']  # B, {1:T}, C, H, W

        writer.add_video('train/rec - gt',
                        torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0., 1.).cpu(),
                        global_step=global_step)

        for k, v in logs.items():
            if 'loss' in k:
                writer.add_scalar(tag + '_loss/' + k, v, global_step=global_step)
            if 'grad_norm' in k:
                writer.add_scalar(tag + '_grad_norm/' + k, v, global_step=global_step)
            if 'hp' in k:
                writer.add_scalar(tag + '_hp/' + k, v, global_step=global_step)
            if 'ACT' in k:
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            writer.add_histogram(tag + '_ACT/' + k + '-' + kk, vv, global_step=global_step)
                            writer.add_scalar(tag + '_mean_ACT/' + k + '-' + kk, vv.mean(), global_step=global_step)
                        if isinstance(vv, float):
                            writer.add_scalar(tag + '_ACT/' + k + '-' + kk, vv, global_step=global_step)
                else:
                    if isinstance(v, torch.Tensor):
                        writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
                        writer.add_scalar(tag + '_mean_ACT/' + k, v.mean(), global_step=global_step)
                    if isinstance(v, float):
                        writer.add_scalar(tag + '_ACT/' + k, v, global_step=global_step)
            if 'imag_value' in k:
                writer.add_scalar(tag + '_values/' + k, v.mean(), global_step=global_step)
                writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
            if 'actor_target' in k:
                writer.add_scalar(tag + 'actor_target/' + k, v, global_step=global_step)

    def optimize_actor(self, actor_loss, actor_optimizer, writer, global_step):
        actor_loss.backward()
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.actor.named_parameters():
                if p.requires_grad:
                    writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)

        actor_optimizer.step()
        return grad_norm_actor.item()

    def optimize_value(self, value_loss, value_optimizer, writer, global_step):
        value_loss.backward()
        grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.value.named_parameters():
                if p.requires_grad:
                    writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
        value_optimizer.step()

        return grad_norm_value.item()

    def world_model_loss(self, global_step, traj, temp):
        outputs = self.forward(
                observations=traj['observations'], 
                timesteps=traj['timestep'], 
                gt_action=traj['action'],
                done=traj['done'], 
                temp=temp
            )

        model_loss, model_logs, prior_state, post_state= self.world_model.world_model_loss(
            global_step, 
            traj, 
            outputs['prior_state'], 
            outputs['post_state'], 
            temp,
        )
        return model_loss, model_logs, prior_state, post_state

    def actor_and_value_loss(self, global_step, post_state, traj, temp):
        self.update_slow_target(global_step)
        self.value.eval()
        self.value.requires_grad_(False)

        imagine_feat, imagine_state, imagine_action, \
            imagine_reward, imagine_disc, imagine_idx = self.world_model.imagine_ahead(self.actor, post_state, traj, self.batch_length-1, temp)

        target, weights = self.compute_target(imagine_feat, imagine_reward, imagine_disc) # B*T, H-1, 1

        slice_idx = -1

        actor_dist = self.actor(imagine_feat.detach()) # B*T, H
        if self.action_dist == 'onehot':
            indices = imagine_action.max(-1)[1]
            actor_logprob = actor_dist._categorical.log_prob(indices)
        else:
            actor_logprob = actor_dist.log_prob(imagine_action)

        if self.actor_loss_type == 'dynamic':
            actor_loss = target
        elif self.actor_loss_type == 'reinforce':
            baseline = self.value(imagine_feat[:, :slice_idx]).mean
            advantage = (target - baseline).detach()
            actor_loss = actor_logprob[:, :slice_idx].unsqueeze(2) * advantage
        elif self.actor_loss_type == 'both':
            raise NotImplementedError

        actor_entropy = actor_dist.entropy()
        ent_scale = self.ent_scale
        actor_loss = ent_scale * actor_entropy[:, :slice_idx].unsqueeze(2) + actor_loss
        actor_loss = -(weights[:, :slice_idx] * actor_loss).mean()

        self.value.train()
        self.value.requires_grad_(True)
        imagine_value_dist = self.value(imagine_feat[:,:slice_idx].detach())
        log_prob = -imagine_value_dist.log_prob(target.detach())
        value_loss = weights[:, :slice_idx] * log_prob.unsqueeze(2)
        value_loss = value_loss.mean()
        imagine_value = imagine_value_dist.mean

        if global_step % self.log_every_step == 0:
            imagine_dist = Independent(OneHotCategorical(logits=imagine_state['logits']), 1)
            if self.action_dist == 'onehot':
                action_samples = imagine_action.argmax(dim=-1).float().detach()
            else:
                action_samples = imagine_action.detach()
            logs = {
                'value_loss': value_loss.detach().item(),
                'actor_loss': actor_loss.detach().item(),
                'ACT_imag_state': {k: v.detach() for k, v in imagine_state.items()},
                'ACT_imag_entropy': imagine_dist.entropy().mean().detach().item(),
                'ACT_actor_entropy': actor_entropy.mean().item(),
                'ACT_action_prob': actor_dist.mean.detach(),
                'ACT_actor_logprob': actor_logprob.mean().item(),
                'ACT_action_samples': action_samples,
                'ACT_image_discount': imagine_disc.detach(),
                'ACT_imag_value': imagine_value.squeeze(-1).detach(),
                'ACT_actor_target': target.mean().detach(),
                'ACT_target': target.squeeze(-1).detach(),
                'ACT_actor_baseline': baseline.mean().detach(),
                'ACT_imag_reward': imagine_reward.detach(),
                'ACT_imagine_idx': imagine_idx.float(),
            }
        else:
            logs = {}

        return actor_loss, value_loss, logs

    def compute_target(self, imag_feat, reward, discount_arr):
        self.slow_value.eval()
        self.slow_value.requires_grad_(False)

        value = self.slow_value(imag_feat).mean  # B*T, H, 1
        target = self.lambda_return(reward[:, 1:], value[:, :-1], discount_arr[:, 1:],
                                    value[:, -1], self.lambda_)

        discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]), discount_arr[:, :-1]], dim=1)
        weights = torch.cumprod(discount_arr, 1).detach()  # B, T 1
        return target, weights

    def policy(self, prev_obs, obs, action, gradient_step, temp, state=None, training=True, context_len=49):
        obs = obs.unsqueeze(1) / 255. - 0.5 # B, T, C, H, W
        obs_emb = self.world_model.dynamic.img_enc(obs) # B, T, C
        post = self.world_model.dynamic.infer_post_stoch(obs_emb, temp, action=None) # B, T, N, C

        if state is None:
            state = post
            prev_obs = prev_obs.unsqueeze(1) / 255. - 0.5  # B, T, C, H, W
            prev_obs_emb = self.world_model.dynamic.img_enc(prev_obs)  # B, T, C
            prev_post = self.world_model.dynamic.infer_post_stoch(prev_obs_emb, temp, action=None)  # B, T, N, C

            for k, v in post.items():
                state[k] = torch.cat([prev_post[k], v], dim=1)
            s_t = state['stoch']
        else:
            s_t = torch.cat([state['stoch'], post['stoch'][:, -1:]], dim=1)[:, -context_len:]
            for k, v in post.items():
                state[k] = torch.cat([state[k], v], dim=1)[:, -context_len:]


        pred_prior = self.world_model.dynamic.infer_prior_stoch(s_t[:, :-1], temp, action)

        post_state_trimed = {}
        for k, v in state.items():
            if k in ['stoch', 'logits', 'pos', 'mean', 'std']:
                post_state_trimed[k] = v[:, 1:]
            else:
                post_state_trimed[k] = v
        post_state_trimed['deter'] = pred_prior['deter']
        post_state_trimed['o_t'] = pred_prior['o_t']

        rnn_feature = self.world_model.dynamic.get_feature(post_state_trimed, layer=self.reward_layer)
        pred_action_pdf = self.actor(rnn_feature[:, -1:].detach())

        if training:
            pred_action = pred_action_pdf.sample() # B, 1, C
        else:
            if self.action_dist == 'onehot':
                pred_action = pred_action_pdf.mean
                index = pred_action.argmax(dim=-1)[0]
                pred_action = torch.zeros_like(pred_action)
                pred_action[..., index] = 1
            else:
                pred_action = pred_action_pdf.mode

        action = torch.cat([action, pred_action], dim=1)[:, -(context_len-1):] # B, T, C

        return action, state

    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,
        gt_states: torch.Tensor = None,
        max_steps: int | None = None,
    ):
        """
        ✅ [Fix #2/#3]
        - (2) GT slice 오프바이원 수정: (t_prev, t_curr)와 입력 프레임이 일치하도록
        - (3) 스케일 통일: rollout 내부는 [-1, 1]로 유지 (학습/비디오 eval과 일관)
        """
        use_gt_image = True
        use_gt_action = False

        device = gt_seq.device
        T_total, C, H, W = gt_seq.shape
        max_steps = min(max_steps, T_total - 1) if max_steps else T_total - 1

        recons: List[torch.Tensor] = []

        # --- History Buffers for Transformer WM ---
        history_z = []
        history_a = []

        def _postprocess_frame(x: torch.Tensor) -> torch.Tensor:
            # rollout 내부 스케일은 [-1, 1] 유지
            return x.clamp(-1, 1)

        # Helper: IDM 결과에서 z 추출 (Posterior)
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

        # --- 1. Warmup: predict frame 1 from frames (0,1) ---
        pair0 = gt_seq[0:2].unsqueeze(0)          # (0,1)
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

            curr_z_hist = z0[:, :-1]   # z_0
            curr_a_hist = la0[:, 1:]   # a_1

            deter0 = self.world_model.predict_next_deter(curr_z_hist, curr_a_hist)  # deter_1
            wm_cond0 = self.world_model.get_cond(z0[:, -1:], deter0)               # use z_1

        idm_step0 = _make_step_output(idm_out0, la0)
        recon0 = self.fdm(pair0, idm_step0, ts0, states=None, wm_cond=wm_cond0)
        recons.append(_postprocess_frame(recon0[0, -1]))

        # --- 2. Loop: predict frame t from frames (t-1, t) when use_gt_image=True ---
        for t_curr in range(2, max_steps + 1):
            t_prev = t_curr - 1
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)

            if use_gt_image:
                # ✅ [Fix #2] timesteps (t_prev,t_curr)에 맞게 (t_prev,t_curr) 프레임을 넣는다
                pair_input = gt_seq[t_prev : t_curr + 1].unsqueeze(0)
            else:
                # generated closed-loop: (t_prev-1, t_prev) -> next
                # (여기서는 원래 코드 흐름 유지, 스케일만 [-1,1] 유지)
                if len(recons) == 1:
                    prev_img = gt_seq[1]
                    curr_img = recons[-1]
                else:
                    prev_img = recons[-2]
                    curr_img = recons[-1]
                pair_input = torch.stack([prev_img, curr_img], dim=0).unsqueeze(0)

            # IDM
            idm_out = self.idm(pair_input, timesteps=ts_pair, states=None)
            current_la = idm_out.la

            # (optional) GT action override
            if use_gt_action:
                # 여기서 GT action을 쓰는 경우가 있다면, la 인덱싱을 t_curr에 맞춰줘야 함
                # 하지만 현재 코드에서는 use_gt_action=False 기본이므로 pass 유지
                pass

            # WM cond
            wm_cond = None
            if self.use_wm:
                z_curr_pair = _encode_obs_to_z(idm_out)  # [1, 2, z]
                z_t = z_curr_pair[:, -1:]               # z_{t_curr}
                a_t = current_la[:, -1:]                # a_{t_curr}

                history_z.append(z_t)
                history_a.append(a_t)

                full_z = torch.cat(history_z, dim=1)    # z_0 ... z_t
                full_a = torch.cat(history_a, dim=1)    # a_0 ... a_t

                inp_z = full_z[:, :-1]                  # z_0 ... z_{t-1}
                inp_a = full_a[:, 1:]                   # a_1 ... a_t

                deter_t = self.world_model.predict_next_deter(inp_z, inp_a)  # deter_t
                wm_cond = self.world_model.get_cond(full_z[:, -1:], deter_t)

            # FDM
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
        """
        ✅ [Fix #2/#3]
        - (2) GT slice 오프바이원 수정: timesteps와 입력 프레임 정렬
        - (3) 스케일 통일: 내부는 [-1,1] 유지 (비디오 eval과 동일)
        """
        device = gt_seq.device
        T_total, C, H, W = gt_seq.shape

        # Oracle Action Extraction
        gt_batch = gt_seq.unsqueeze(0)
        ts_full = torch.arange(T_total, device=device).unsqueeze(0)
        idm_out_full = self.idm(gt_batch, timesteps=ts_full, states=None)
        gt_actions = idm_out_full.la  # [1, T, la_dim] (index t 는 transition (t-1)->t 로 쓰는 게 맞음)

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

        # --- (A) Warmup: (0,1) -> predict frame 1 ---
        pair0 = gt_seq[0:2].unsqueeze(0)
        ts0 = torch.tensor([[0, 1]], device=device)
        idm_out0 = self.idm(pair0, timesteps=ts0, states=None)

        # ✅ GT action 정렬: frame 1을 만들려면 la[1]이 필요.
        la0 = torch.zeros_like(idm_out0.la)
        la0[:, 1:2] = gt_actions[:, 1:2]

        wm_cond0 = None
        if self.use_wm:
            z0 = _get_posterior_z(idm_out0)  # [1,2,z]
            history_z.append(z0[:, 0:1])     # z_0
            history_a.append(la0[:, 0:1])    # a_0 (pad)

            # deter_1을 만들기 위한 입력: z_0, a_1
            deter0 = self.world_model.predict_next_deter(z0[:, 0:1], la0[:, 1:2])
            wm_cond0 = self.world_model.get_cond(z0[:, 1:2], deter0)

            # history 업데이트: z_1, a_1
            history_z.append(z0[:, 1:2])
            history_a.append(la0[:, 1:2])

        idm_step0 = _make_step_output(idm_out0, la0)
        recon0 = self.fdm(pair0, idm_step0, ts0, states=None, wm_cond=wm_cond0)
        recons.append(_postprocess_frame(recon0[0, -1]))

        # --- (B) Step-by-step dreaming: predict frame t_curr using (t_prev,t_curr) in context, 이후 generated ---
        for t_curr in range(2, T_total):
            t_prev = t_curr - 1
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)
            is_context = t_curr < context_len

            # Input frames
            if is_context:
                # ✅ [Fix #2] context는 GT (t_prev,t_curr)로 맞춘다
                pair_input = gt_seq[t_prev : t_curr + 1].unsqueeze(0)
            else:
                # dreaming: 이전 생성 프레임 사용 (t_prev-1, t_prev 근사)
                if len(recons) == 1:
                    prev_gen = gt_seq[1]
                    curr_gen = recons[-1]
                else:
                    prev_gen = recons[-2]
                    curr_gen = recons[-1]
                pair_input = torch.stack([prev_gen, curr_gen], dim=0).unsqueeze(0)

            # IDM (patch feature용)
            idm_out = self.idm(pair_input, timesteps=ts_pair, states=None)

            # ✅ GT action 정렬: frame t_curr을 만들려면 la[t_curr]을 넣어야 함
            la_override = torch.zeros_like(idm_out.la)
            la_override[:, 1:2] = gt_actions[:, t_curr : t_curr + 1]

            wm_cond = None
            if self.use_wm:
                a_t = la_override[:, 1:2]  # a_{t_curr}

                # history의 full_z/full_a 구성
                full_z = torch.cat(history_z, dim=1)  # z_0 ... z_{t_prev} (이미 쌓여있음)
                full_a = torch.cat(history_a, dim=1)  # a_0 ... a_{t_prev}

                # deter_{t_curr}를 위해: z_{0:t_prev} 와 a_{1:t_curr}
                inp_z = full_z
                inp_a = torch.cat([full_a[:, 1:], a_t], dim=1)

                deter_t = self.world_model.predict_next_deter(inp_z, inp_a)

                if is_context:
                    z_curr_pair = _get_posterior_z(idm_out)
                    z_t = z_curr_pair[:, -1:]
                else:
                    z_t = _sample_prior_z(deter_t)

                # history에 z_t, a_t 추가
                history_z.append(z_t)
                history_a.append(a_t)

                wm_cond = self.world_model.get_cond(z_t, deter_t)

            # FDM prediction
            idm_step = _make_step_output(idm_out, la_override)
            recon_next = self.fdm(pair_input, idm_step, ts_pair, states=None, wm_cond=wm_cond)

            recons.append(_postprocess_frame(recon_next[0, -1]))

        return torch.stack(recons, dim=0)