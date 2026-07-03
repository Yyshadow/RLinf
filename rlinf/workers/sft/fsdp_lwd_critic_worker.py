# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""FSDP worker for LWD-style chunk critic training."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.data.datasets.lwd import (
    LWDChunkDataCollator,
    LWDChunkDataset,
    LWDDataLoaderImpl,
    LWDMixtureDataset,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.lwd_critic.lwd_loss import (
    compute_lwd_losses,
    update_ema_target,
)
from rlinf.models.embodiment.value_model.checkpoint_utils import has_tokenizer_files
from rlinf.models.embodiment.value_model.processing import ValueProcessor
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict


class FSDPLWDCriticWorker(FSDPModelManager, Worker):
    """Train a LWD critic with RLinf's SFT runner and FSDP lifecycle."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self.data_loader, self.eval_data_loaders = self.build_dataloader()
        self.data_iter = iter(self.data_loader) if self.data_loader is not None else None
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.target_model = None

    def init_worker(self):
        self._validate_fsdp_boundary()
        self.setup_model_and_optimizer()
        self.target_model.to(self.device)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def _validate_fsdp_boundary(self) -> None:
        fsdp_cfg = self.cfg.actor.fsdp_config
        if fsdp_cfg.get("strategy", "fsdp") != "fsdp":
            raise ValueError("LWD critic EMA target currently supports fsdp strategy only.")
        if fsdp_cfg.get("sharding_strategy", "no_shard") != "no_shard":
            raise ValueError("LWD critic EMA target currently supports no_shard only.")
        if not fsdp_cfg.get("use_orig_params", False):
            raise ValueError("LWD critic EMA target requires use_orig_params=true.")

    def model_provider_func(self) -> torch.nn.Module:
        model = get_model(self.cfg.actor.model)
        self.target_model = copy.deepcopy(model)
        return model

    def build_dataloader(self):
        data_cfg = self.cfg.get("data", {})
        model_cfg = self.cfg.actor.model
        data_root = data_cfg.get("data_root", None)
        train_num_workers = int(data_cfg.get("train_num_workers", 0))
        eval_num_workers = int(data_cfg.get("eval_num_workers", train_num_workers))
        pin_memory = bool(data_cfg.get("pin_memory", True))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        persistent_workers = bool(data_cfg.get("persistent_workers", True))
        norm_stats_path = data_cfg.get("norm_stats_path", None)
        if norm_stats_path is None:
            raise ValueError("data.norm_stats_path is required for LWD critic training.")

        def worker_kwargs(num_workers: int) -> dict[str, Any]:
            kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
            if num_workers > 0:
                kwargs["persistent_workers"] = persistent_workers
                if prefetch_factor is not None:
                    kwargs["prefetch_factor"] = int(prefetch_factor)
            return kwargs

        tokenizer_path = getattr(model_cfg, "tokenizer_path", None) or getattr(
            model_cfg, "gemma3_path", None
        )
        if tokenizer_path is None or not has_tokenizer_files(Path(tokenizer_path)):
            raise ValueError("Set actor.model.tokenizer_path or actor.model.gemma3_path.")

        processor = ValueProcessor(
            tokenizer_name_or_path=tokenizer_path,
            max_token_len=getattr(model_cfg, "max_token_len", 200),
            image_keys=tuple(data_cfg.get("image_keys", ())),
            do_augment=bool(data_cfg.get("do_augment", True)),
        )
        train_collator = LWDChunkDataCollator(
            processor=processor,
            max_length=getattr(model_cfg, "max_token_len", 200),
            train=True,
        )
        eval_collator = LWDChunkDataCollator(
            processor=processor,
            max_length=getattr(model_cfg, "max_token_len", 200),
            train=False,
        )

        def resolve_path(path: str) -> str:
            if data_root and not os.path.isabs(path):
                return os.path.join(data_root, path)
            return path

        def build_dataset(entry: dict[str, Any]) -> LWDChunkDataset:
            dataset_path = entry.get("dataset_path", None)
            if not dataset_path:
                raise ValueError("Each LWD dataset entry must define dataset_path.")
            return LWDChunkDataset(
                dataset_path=resolve_path(dataset_path),
                action_horizon=entry.get(
                    "action_horizon",
                    data_cfg.get(
                        "action_horizon",
                        getattr(model_cfg, "action_horizon", 50),
                    ),
                ),
                norm_stats_path=resolve_path(norm_stats_path),
                use_quantile_norm=entry.get(
                    "use_quantile_norm",
                    data_cfg.get("use_quantile_norm", True),
                ),
                adapt_to_pi=entry.get("adapt_to_pi", data_cfg.get("adapt_to_pi", True)),
                default_prompt=entry.get("default_prompt", None),
                max_samples=entry.get("max_samples", data_cfg.get("max_samples", None)),
            )

        train_entries = [
            dict(entry) for entry in data_cfg.get("train_data_paths", []) or []
        ]
        if not train_entries:
            raise ValueError("data.train_data_paths must contain at least one dataset.")
        datasets_with_weights = [
            (build_dataset(entry), float(entry.get("weight", 1.0)))
            for entry in train_entries
        ]
        if len(datasets_with_weights) == 1:
            train_dataset = datasets_with_weights[0][0]
        else:
            train_dataset = LWDMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=data_cfg.get("balance_weights", True),
                seed=data_cfg.get("seed", 42),
            )

        train_sampler = None
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                drop_last=True,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=train_collator,
            **worker_kwargs(train_num_workers),
        )
        train_data_loader = LWDDataLoaderImpl(
            {"model_type": "lwd_critic"}, train_loader
        )

        eval_data_loaders = []
        for entry in data_cfg.get("eval_data_paths", []) or []:
            entry = dict(entry)
            eval_dataset = build_dataset(entry)
            eval_sampler = None
            if torch.distributed.is_initialized():
                eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    eval_dataset,
                    num_replicas=self._world_size,
                    rank=self._rank,
                    shuffle=False,
                    drop_last=False,
                )
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.cfg.actor.micro_batch_size,
                shuffle=False,
                sampler=eval_sampler,
                drop_last=False,
                collate_fn=eval_collator,
                **worker_kwargs(eval_num_workers),
            )
            name = entry.get("name", Path(entry["dataset_path"]).stem)
            eval_data_loaders.append(
                (name, LWDDataLoaderImpl({"model_type": "lwd_critic"}, eval_loader))
            )

        return train_data_loader, eval_data_loaders

    def run_training(self) -> dict[str, float]:
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            self.target_model.eval()

            metrics = []
            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )
                batch = self._move_to_device(next(self.data_iter))
                with self.amp_context:
                    loss_out = compute_lwd_losses(
                        model=self.model,
                        target_model=self.target_model,
                        batch=batch,
                        gamma=float(self.cfg.algorithm.gamma),
                        value_loss_weight=float(self.cfg.algorithm.value_loss_weight),
                        q_loss_weight=float(self.cfg.algorithm.q_loss_weight),
                    )
                scaled_loss = loss_out.loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(scaled_loss).backward()
                metrics.append(self._loss_metrics(loss_out))

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            update_ema_target(
                self.model.module,
                self.target_model,
                tau=float(self.cfg.algorithm.ema_tau),
            )
            self.lr_scheduler.step()

            train_metrics = self._average_metrics(metrics)
            train_metrics["grad_norm"] = grad_norm
            train_metrics["lr"] = lr_list[0] if lr_list else 0.0
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
                    self.offload_optimizer()

            return train_metrics

    def run_eval(self) -> dict[str, float]:
        if not self.eval_data_loaders:
            return {}

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)

            self.model.eval()
            self.target_model.eval()
            final_metrics = {}

            with torch.no_grad():
                for name, loader in self.eval_data_loaders:
                    batch_metrics = []
                    group_metrics: dict[str, list[float]] = {}
                    for batch in loader:
                        batch = self._move_to_device(batch)
                        with self.amp_context:
                            loss_out = compute_lwd_losses(
                                model=self.model,
                                target_model=self.target_model,
                                batch=batch,
                                gamma=float(self.cfg.algorithm.gamma),
                                value_loss_weight=float(
                                    self.cfg.algorithm.value_loss_weight
                                ),
                                q_loss_weight=float(self.cfg.algorithm.q_loss_weight),
                            )
                        batch_metrics.append(self._loss_metrics(loss_out))
                        self._collect_group_metrics(loss_out, batch, group_metrics)

                    dataset_metrics = self._average_metrics(batch_metrics)
                    for group_name, values in group_metrics.items():
                        if values:
                            dataset_metrics[group_name] = sum(values) / len(values)
                    for key, value in dataset_metrics.items():
                        final_metrics[f"{name}/{key}"] = value

            final_metrics = all_reduce_dict(
                final_metrics, op=torch.distributed.ReduceOp.AVG
            )

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()

            return final_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            target_path = os.path.join(save_path, "target_model.pt")
            torch.save(self.target_model.state_dict(), target_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        target_path = os.path.join(load_path, "target_model.pt")
        map_location = torch.device(f"cuda:{self.device}")
        state_dict = torch.load(
            target_path,
            map_location=map_location,
            weights_only=False,
        )
        self.target_model.load_state_dict(state_dict)

    def set_global_step(self, step: int):
        loader_len = len(self.data_loader)
        if loader_len == 0:
            return
        epoch = step // self.get_max_steps_per_epoch()
        if getattr(self, "_current_epoch", -1) != epoch:
            self._current_epoch = epoch
            self.data_loader.set_epoch(epoch)
            self.data_iter = iter(self.data_loader)

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        return max(1, len(self.data_loader) // self.gradient_accumulation)

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    @staticmethod
    def _loss_metrics(loss_out) -> dict[str, float]:
        return {
            "loss": loss_out.loss.detach().item(),
            "q_loss": loss_out.q_loss.detach().item(),
            "value_loss": loss_out.value_loss.detach().item(),
            "target_q_mean": loss_out.target_q.detach().mean().item(),
            "target_q_std": loss_out.target_q.detach().std(unbiased=False).item(),
            "reward_sum_mean": loss_out.reward_sum.detach().mean().item(),
            "q_min_mean": loss_out.q_min.detach().mean().item(),
            "q_min_std": loss_out.q_min.detach().std(unbiased=False).item(),
            "q_head_gap": loss_out.q_head_gap.detach().item(),
        }

    @staticmethod
    def _average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        merged: dict[str, list[float]] = {}
        for item in metrics:
            for key, value in item.items():
                merged.setdefault(key, []).append(value)
        return {key: sum(values) / len(values) for key, values in merged.items()}

    @staticmethod
    def _collect_group_metrics(loss_out, batch, group_metrics: dict[str, list[float]]):
        q_min = loss_out.q_min.detach().float().view(-1).cpu()
        sources = batch.get("source", [])
        success = batch.get("success")
        success_list = (
            success.detach().bool().view(-1).cpu().tolist()
            if isinstance(success, torch.Tensor)
            else [False] * len(q_min)
        )
        for idx, q_value in enumerate(q_min.tolist()):
            source = str(sources[idx]) if idx < len(sources) else ""
            if success_list[idx]:
                key = "success/q_min_mean"
            elif "nearmiss" in source:
                key = "nearmiss/q_min_mean"
            elif "failed" in source or "failure" in source:
                key = "failed/q_min_mean"
            else:
                key = "other/q_min_mean"
            group_metrics.setdefault(key, []).append(float(q_value))


__all__ = ["FSDPLWDCriticWorker"]
