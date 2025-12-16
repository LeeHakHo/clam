import collections
import time
from collections import defaultdict

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from rich.pretty import pretty_repr

import clam.utils.general_utils as gutl
from clam.trainers.base_trainer import BaseTrainer
from clam.utils.data_utils import Batch
from clam.utils.logger import log
from clam.utils.rollouts import run_eval_rollouts


class OfflineTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_step = 0

        # ✅ eval iterator 반드시 초기화 (repeat로 StopIteration 방지)
        self._eval_iter = self.eval_dataloader.repeat().as_numpy_iterator()

    def train(self):
        # ✅ accelerate면 main process만 eval 돌리기 (중복/충돌 방지)
        if (not self.cfg.skip_first_eval) and (
            (not self.cfg.accelerate.use) or self.accelerator.is_main_process
        ):
            self.eval(step=0)

        self.model.train()
        if hasattr(self, "action_decoder"):
            self.action_decoder.train()

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        for self.train_step in tqdm.tqdm(
            range(self.cfg.num_updates),
            desc=f"{self.cfg.name} train batches",
            disable=False,
            total=self.cfg.num_updates,
        ):
            batch_load_time = time.time()
            batch_np = next(train_iter)

            # put the batch on the device
            batch_np = gutl.to_device(batch_np, self.device)
            batch_load_time = time.time() - batch_load_time
            batch = Batch(**batch_np)

            update_time = time.time()

            self.optimizer.zero_grad()
            if hasattr(self, "action_decoder_optimizer"):
                self.action_decoder_optimizer.zero_grad()

            if self.cfg.accelerate.use:
                with self.accelerator.autocast():
                    metrics, total_loss = self.compute_loss(batch, train=True)

                self.accelerator.backward(total_loss)

                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
                self.optimizer.step()

                if hasattr(self, "action_decoder_optimizer") and (
                    self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.accelerator.clip_grad_norm_(
                        self.action_decoder.parameters(), self.cfg.clip_grad_norm
                    )
                    self.action_decoder_optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    metrics, total_loss = self.compute_loss(batch, train=True)

                self.scaler.scale(total_loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.clip_grad_norm
                )

                self.scaler.step(self.optimizer)

                if hasattr(self, "action_decoder_optimizer") and (
                    self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.scaler.unscale_(self.action_decoder_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.action_decoder.parameters(),
                        max_norm=self.cfg.clip_grad_norm,
                    )
                    self.scaler.step(self.action_decoder_optimizer)

                self.scaler.update()

            self.scheduler.step()

            metrics["time/batch_load"] = batch_load_time
            metrics["time/update"] = time.time() - update_time
            metrics["lr"] = self.scheduler.get_last_lr()[0]

            if hasattr(self, "action_decoder_scheduler"):
                metrics["action_decoder_lr"] = self.action_decoder_scheduler.get_last_lr()[0]

            self.log_to_wandb(metrics, prefix="train/")

            # ✅ eval은 main process만
            if (not self.cfg.accelerate.use) or self.accelerator.is_main_process:
                if ((self.train_step + 1) % self.eval_every) == 0:
                    self.eval(step=self.train_step + 1)
                    self.model.train()
                    if hasattr(self, "action_decoder"):
                        self.action_decoder.train()

                if ((self.train_step + 1) % self.cfg.log_terminal_every) == 0:
                    log(f"step: {self.train_step}, train:")
                    log(f"{pretty_repr(metrics)}")

        # final evaluation (main만)
        if (not self.cfg.accelerate.use) or self.accelerator.is_main_process:
            self.eval(step=self.cfg.num_updates)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, step: int):
        # ✅ accelerate면 main process만 eval
        if self.cfg.accelerate.use and (not self.accelerator.is_main_process):
            return {}

        log("running evaluation", "blue")

        self.model.eval()
        if hasattr(self, "action_decoder"):
            self.action_decoder.eval()

        eval_time = time.time()
        eval_metrics = collections.defaultdict(list)

        # ✅ 혹시라도 iterator가 None/없으면 복구
        if not hasattr(self, "_eval_iter") or self._eval_iter is None:
            self._eval_iter = self.eval_dataloader.repeat().as_numpy_iterator()

        for _ in range(self.num_eval_batches):
            batch_np = next(self._eval_iter)
            batch_np = gutl.to_device(batch_np, self.device)
            batch = Batch(**batch_np)

            with torch.no_grad():
                metrics, _total_eval_loss = self.compute_loss(batch, train=False)

            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                eval_metrics[k].append(v)

            del batch
            del batch_np

        for k, v in eval_metrics.items():
            eval_metrics[k] = float(np.mean(np.array(v)))

        eval_metrics["time"] = time.time() - eval_time
        self.log_to_wandb(eval_metrics, prefix="eval/")

        with open(self.log_dir / "eval.txt", "a+") as f:
            f.write(f"{step}, {eval_metrics}\n")

        log(f"eval: {pretty_repr(eval_metrics)}")

        if self.cfg.run_eval_rollouts:
            rollout_metrics, *_ = run_eval_rollouts(
                cfg=self.cfg, model=self.model, wandb_run=self.wandb_run
            )
            self.log_to_wandb(rollout_metrics, prefix="eval_rollout/")

            with open(self.log_dir / "eval.txt", "a+") as f:
                f.write(f"{step}, {rollout_metrics}\n")

            log(f"eval rollout: {pretty_repr(rollout_metrics)}")

        self.save_model(ckpt_dict=self.save_dict, metrics=eval_metrics, iter=step)
        return eval_metrics
