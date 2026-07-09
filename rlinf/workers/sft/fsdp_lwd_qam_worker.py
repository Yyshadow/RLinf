# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""FSDP worker for LWD QAM policy extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.lwd import (
    QAMLossOutput,
    clip_by_global_norm,
    flow_ode_step,
    qam_vector_field_loss,
)
from rlinf.algorithms.lwd.qam import bc_flow_matching_loss
from rlinf.data.datasets.lwd import (
    LWDChunkDataCollator,
    LWDChunkDataset,
    LWDDataLoaderImpl,
    LWDMixtureDataset,
    LWDQAMDataCollator,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.value_model.checkpoint_utils import has_tokenizer_files
from rlinf.models.embodiment.value_model.processing import ValueProcessor
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict


class FSDPLWDQAMWorker(FSDPModelManager, Worker):
    """Train an OpenPI actor with QAM guidance from a frozen LWD critic."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self.data_loader = self.build_dataloader()
        self.data_iter = iter(self.data_loader)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        per_step_batch = micro_batch_size * self._world_size
        if global_batch_size % per_step_batch != 0:
            raise ValueError(
                "actor.global_batch_size must be divisible by "
                "actor.micro_batch_size * world_size for LWD QAM training."
            )
        self.gradient_accumulation = global_batch_size // per_step_batch
        self.reference_model = None
        self.critic_model = None
        self.global_step = 0
        self._current_epoch = 0

    def init_worker(self):
        self.setup_model_and_optimizer()
        self._setup_reference_and_critic()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> torch.nn.Module:
        return get_model(self.cfg.actor.model)

    def build_dataloader(self):
        data_cfg = self.cfg.get("data", {})
        data_root = data_cfg.get("data_root", None)
        train_num_workers = int(data_cfg.get("train_num_workers", 0))
        pin_memory = bool(data_cfg.get("pin_memory", True))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        persistent_workers = bool(data_cfg.get("persistent_workers", True))
        norm_stats_path = data_cfg.get("norm_stats_path", None)
        if norm_stats_path is None:
            raise ValueError("data.norm_stats_path is required for LWD QAM training.")

        def worker_kwargs() -> dict[str, Any]:
            kwargs = {"num_workers": train_num_workers, "pin_memory": pin_memory}
            if train_num_workers > 0:
                kwargs["persistent_workers"] = persistent_workers
                if prefetch_factor is not None:
                    kwargs["prefetch_factor"] = int(prefetch_factor)
            return kwargs

        def resolve_path(path: str) -> str:
            if data_root and not os.path.isabs(path):
                return os.path.join(data_root, path)
            return path

        critic_model_cfg = self.cfg.critic.model
        tokenizer_path = getattr(critic_model_cfg, "tokenizer_path", None) or getattr(
            critic_model_cfg,
            "gemma3_path",
            None,
        )
        if tokenizer_path is None or not has_tokenizer_files(Path(tokenizer_path)):
            raise ValueError(
                "Set critic.model.tokenizer_path or critic.model.gemma3_path."
            )

        processor = ValueProcessor(
            tokenizer_name_or_path=tokenizer_path,
            max_token_len=getattr(critic_model_cfg, "max_token_len", 200),
            image_keys=tuple(data_cfg.get("image_keys", ())),
            do_augment=bool(data_cfg.get("do_augment", True)),
        )
        critic_collator = LWDChunkDataCollator(
            processor=processor,
            max_length=getattr(critic_model_cfg, "max_token_len", 200),
            train=True,
        )
        collator = LWDQAMDataCollator(critic_collator=critic_collator)

        def build_dataset(entry: dict[str, Any]) -> LWDChunkDataset:
            dataset_path = entry.get("dataset_path", None)
            if not dataset_path:
                raise ValueError("Each LWD QAM dataset entry must define dataset_path.")
            return LWDChunkDataset(
                dataset_path=resolve_path(dataset_path),
                action_horizon=entry.get(
                    "action_horizon",
                    data_cfg.get(
                        "action_horizon",
                        getattr(self.cfg.actor.model, "num_action_chunks", 50),
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
            collate_fn=collator,
            **worker_kwargs(),
        )
        return LWDDataLoaderImpl({"model_type": "lwd_qam"}, train_loader)

    def run_training(self) -> dict[str, float]:
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            self.reference_model.eval()
            self.critic_model.eval()

            metrics = []
            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )
                batch = self._move_to_device(self._next_batch())

                with self.amp_context:
                    loss_out = self.compute_qam_loss(batch)

                scaled_loss = loss_out.loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(scaled_loss).backward()
                metrics.append(self._loss_metrics(loss_out))

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

            train_metrics = self._average_metrics(metrics)
            train_metrics["grad_norm"] = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            train_metrics["lr"] = lr_list[0] if lr_list else 0.0
            train_metrics = all_reduce_dict(
                train_metrics,
                op=torch.distributed.ReduceOp.AVG,
            )

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
                    self.offload_optimizer()

            return train_metrics

    def run_eval(self) -> dict[str, float]:
        return {}

    def compute_qam_loss(self, batch: dict[str, Any]) -> QAMLossOutput:
        policy_inputs = batch["policy_inputs"]
        critic_observation = batch["critic_observation"]
        replay_actions = batch["action_chunk"].float()
        env_action_dim = replay_actions.shape[-1]
        model_action_dim = self._model_action_dim()
        internal_action_shape = torch.Size(
            (*replay_actions.shape[:-1], model_action_dim)
        )
        num_steps = int(self.cfg.algorithm.get("qam_num_denoise_steps", 5))
        lambda_q = float(self.cfg.algorithm.get("lambda_q", 2.0))
        if num_steps < 2:
            raise ValueError("algorithm.qam_num_denoise_steps must be at least 2.")
        if lambda_q <= 0:
            raise ValueError("algorithm.lambda_q must be positive.")
        qam_loss_weight = float(self.cfg.algorithm.get("qam_loss_weight", 1.0))
        anchor_weight = float(self.cfg.algorithm.get("anchor_weight", 0.0))
        bc_weight = float(self.cfg.algorithm.get("bc_weight", 0.0))
        qam_grad_clip = self.cfg.algorithm.get("qam_grad_clip", 0.05)
        qam_delta_clip = self.cfg.algorithm.get("qam_delta_clip", 5.0)
        min_sigma = float(self.cfg.algorithm.get("min_sigma", 1e-3))
        critic_grad_mode = str(
            self.cfg.algorithm.get("qam_critic_grad_mode", "mean")
        ).lower()
        clip_action_for_critic = bool(
            self.cfg.algorithm.get("qam_clip_action_for_critic", False)
        )

        transitions = self._rollout_reference_flow(
            policy_inputs=policy_inputs,
            action_shape=internal_action_shape,
            num_steps=num_steps,
            dtype=replay_actions.dtype,
        )
        endpoint = transitions[-1]["x_next"].detach().requires_grad_(True)
        critic_endpoint = self._slice_env_action(endpoint, env_action_dim)
        endpoint_saturation_frac = (
            ((critic_endpoint < -1.0) | (critic_endpoint > 1.0)).float().mean()
        )
        critic_action = (
            critic_endpoint.clamp(-1.0, 1.0)
            if clip_action_for_critic
            else critic_endpoint
        )
        q_out = self.critic_model(
            observation=critic_observation,
            action_chunk=critic_action,
        )
        q_values = q_out.q_values.float()
        q_min = q_values.min(dim=-1).values
        q_mean = q_values.mean(dim=-1)
        if critic_grad_mode == "mean":
            q_for_grad = q_mean
        elif critic_grad_mode == "min":
            q_for_grad = q_min
        else:
            raise ValueError(
                "algorithm.qam_critic_grad_mode must be 'mean' or 'min'."
            )
        q_scalar = q_for_grad.sum()
        action_grad = torch.autograd.grad(q_scalar, endpoint)[0]
        action_grad, action_grad_norm, action_grad_clip_frac = clip_by_global_norm(
            action_grad,
            qam_grad_clip,
        )
        adjoint = -action_grad / lambda_q

        adjoints = self._solve_reference_adjoint(transitions, adjoint)

        qam_losses = []
        anchor_losses = []
        delta_norms = []
        delta_clip_fracs = []
        adjoint_norms = []
        # The first OpenPI step is exactly the Gaussian-noise endpoint
        # (t=1, w=0).  QAM's sigma is zero at endpoints, so train only on
        # interior flow states in the discrete objective.
        loss_transitions = transitions[1:] if len(transitions) > 1 else transitions
        loss_adjoints = adjoints[1:] if len(adjoints) > 1 else adjoints
        for transition, step_adjoint in zip(loss_transitions, loss_adjoints):
            x_t = transition["x"].detach()
            timestep = transition["timestep"].detach()
            v_beta = transition["v"].detach()
            v_theta = self._policy_velocity(self.model, policy_inputs, x_t, timestep)
            step_loss, delta_norm, clip_frac = qam_vector_field_loss(
                v_theta=v_theta,
                v_beta=v_beta,
                adjoint=step_adjoint.detach(),
                timesteps=timestep,
                delta_clip=qam_delta_clip,
                min_sigma=min_sigma,
            )
            qam_losses.append(step_loss)
            anchor_losses.append(F.mse_loss(v_theta.float(), v_beta.float()))
            delta_norms.append(delta_norm)
            delta_clip_fracs.append(clip_frac)
            adjoint_norms.append(step_adjoint.float().flatten(1).norm(dim=-1).mean())

        qam_loss = torch.stack(qam_losses).mean()
        anchor_loss = torch.stack(anchor_losses).mean()
        if bc_weight > 0:
            bc_loss = self._bc_flow_loss(
                policy_inputs,
                replay_actions,
                model_action_dim,
            )
        else:
            bc_loss = qam_loss.new_zeros(())
        loss = (
            qam_loss_weight * qam_loss
            + anchor_weight * anchor_loss
            + bc_weight * bc_loss
        )

        return QAMLossOutput(
            loss=loss,
            qam_loss=qam_loss.detach(),
            anchor_loss=anchor_loss.detach(),
            bc_loss=bc_loss.detach(),
            q_mean=q_mean.detach().mean(),
            q_min=q_min.detach().mean(),
            q_std=q_for_grad.detach().std(unbiased=False),
            q_head_gap=q_values.detach().std(dim=-1, unbiased=False).mean(),
            action_grad_norm=action_grad_norm.detach().mean(),
            action_grad_clip_frac=action_grad_clip_frac.detach(),
            adjoint_norm=torch.stack(adjoint_norms).mean().detach(),
            qam_delta_norm=torch.stack(delta_norms).mean().detach(),
            qam_delta_clip_frac=torch.stack(delta_clip_fracs).mean().detach(),
            endpoint_min=critic_endpoint.detach().amin(),
            endpoint_max=critic_endpoint.detach().amax(),
            endpoint_abs_p95=self._abs_p95(critic_endpoint.detach()),
            endpoint_saturation_frac=endpoint_saturation_frac.detach(),
            critic_action_min=critic_action.detach().amin(),
            critic_action_max=critic_action.detach().amax(),
            critic_action_abs_p95=self._abs_p95(critic_action.detach()),
            replay_action_min=replay_actions.detach().amin(),
            replay_action_max=replay_actions.detach().amax(),
            replay_action_abs_p95=self._abs_p95(replay_actions.detach()),
        )

    def _setup_reference_and_critic(self) -> None:
        reference_cfg = OmegaConf.create(
            OmegaConf.to_container(self.cfg.actor.model, resolve=True)
        )
        reference_cfg.model_path = self.cfg.algorithm.reference_model_path
        self.reference_model = get_model(reference_cfg).to(self.device)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        self.critic_model = get_model(self.cfg.critic.model).to(self.device)
        self.critic_model.requires_grad_(False)
        self.critic_model.eval()

    def _rollout_reference_flow(
        self,
        policy_inputs: dict[str, Any],
        action_shape: torch.Size,
        num_steps: int,
        dtype: torch.dtype,
    ) -> list[dict[str, torch.Tensor]]:
        x_t = torch.randn(action_shape, device=self.device, dtype=dtype)
        transitions = []
        for step in range(num_steps):
            x_t = x_t.detach().requires_grad_(True)
            timestep = self._step_timestep(x_t.shape[0], step, num_steps, dtype)
            v_t = self._policy_velocity(
                self.reference_model,
                policy_inputs,
                x_t,
                timestep,
            )
            x_next = flow_ode_step(x_t, v_t, step, num_steps)
            transitions.append(
                {
                    "x": x_t,
                    "x_next": x_next,
                    "v": v_t,
                    "timestep": timestep,
                }
            )
            x_t = x_next.detach()
        return transitions

    def _solve_reference_adjoint(
        self,
        transitions: list[dict[str, torch.Tensor]],
        terminal_adjoint: torch.Tensor,
    ) -> list[torch.Tensor]:
        adjoint = terminal_adjoint.detach()
        adjoints = []
        for transition in reversed(transitions):
            local_adjoint = torch.autograd.grad(
                outputs=transition["x_next"],
                inputs=transition["x"],
                grad_outputs=adjoint.to(transition["x_next"].dtype),
                retain_graph=False,
            )[0]
            adjoints.append(local_adjoint.detach())
            adjoint = local_adjoint.detach()
        return list(reversed(adjoints))

    def _bc_flow_loss(
        self,
        policy_inputs: dict[str, Any],
        replay_actions: torch.Tensor,
        model_action_dim: int,
    ) -> torch.Tensor:
        replay_actions = self._pad_action_chunk(replay_actions, model_action_dim)
        batch_size = replay_actions.shape[0]
        noise = torch.randn_like(replay_actions)
        timestep = torch.rand(
            batch_size,
            device=replay_actions.device,
            dtype=replay_actions.dtype,
        ).clamp(1e-3, 1.0)
        view_shape = (batch_size,) + (1,) * (replay_actions.dim() - 1)
        x_t = (
            timestep.view(view_shape) * noise
            + (1.0 - timestep).view(view_shape) * replay_actions
        )
        target_velocity = noise - replay_actions
        v_theta = self._policy_velocity(self.model, policy_inputs, x_t, timestep)
        return bc_flow_matching_loss(v_theta, target_velocity)

    def _model_action_dim(self) -> int:
        model = self.reference_model
        if model is None:
            model = getattr(self.model, "module", self.model)
        action_in_proj = getattr(model, "action_in_proj", None)
        if action_in_proj is not None:
            return int(action_in_proj.in_features)
        return int(model.config.action_dim)

    @staticmethod
    def _pad_action_chunk(action_chunk: torch.Tensor, action_dim: int) -> torch.Tensor:
        current_dim = action_chunk.shape[-1]
        if current_dim == action_dim:
            return action_chunk
        if current_dim > action_dim:
            raise ValueError(
                f"action chunk dim {current_dim} exceeds model action dim {action_dim}."
            )
        return F.pad(action_chunk, (0, action_dim - current_dim))

    @staticmethod
    def _slice_env_action(
        action_chunk: torch.Tensor,
        env_action_dim: int,
    ) -> torch.Tensor:
        return action_chunk[..., :env_action_dim]

    def _policy_velocity(
        self,
        model: torch.nn.Module,
        policy_inputs: dict[str, Any],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        out = model(
            forward_type=ForwardType.NFT,
            forward_inputs=policy_inputs,
            nft_inputs={
                "x_t": x_t,
                "timesteps": timestep,
            },
        )
        return out["v_theta"]

    def _step_timestep(
        self,
        batch_size: int,
        step: int,
        num_steps: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = 1.0 - step / num_steps
        return torch.full(
            (batch_size,),
            value,
            device=self.device,
            dtype=dtype,
        )

    def set_global_step(self, step: int):
        self.global_step = step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(step)
        loader_len = len(self.data_loader)
        if loader_len == 0:
            return
        epoch = step // self.get_max_steps_per_epoch()
        if getattr(self, "_current_epoch", -1) != epoch:
            self._current_epoch = epoch
            self.data_loader.set_epoch(epoch)
            self.data_iter = iter(self.data_loader)

    def get_max_steps_per_epoch(self):
        return max(1, len(self.data_loader) // self.gradient_accumulation)

    def _next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self._current_epoch += 1
            self.data_loader.set_epoch(self._current_epoch)
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    @staticmethod
    def _loss_metrics(loss_out: QAMLossOutput) -> dict[str, float]:
        return {
            "loss": loss_out.loss.detach().item(),
            "qam_loss": loss_out.qam_loss.item(),
            "anchor_loss": loss_out.anchor_loss.item(),
            "bc_loss": loss_out.bc_loss.item(),
            "q_mean": loss_out.q_mean.item(),
            "q_min": loss_out.q_min.item(),
            "q_std": loss_out.q_std.item(),
            "q_head_gap": loss_out.q_head_gap.item(),
            "action_grad_norm": loss_out.action_grad_norm.item(),
            "action_grad_clip_frac": loss_out.action_grad_clip_frac.item(),
            "adjoint_norm": loss_out.adjoint_norm.item(),
            "qam_delta_norm": loss_out.qam_delta_norm.item(),
            "qam_delta_clip_frac": loss_out.qam_delta_clip_frac.item(),
            "endpoint_min": loss_out.endpoint_min.item(),
            "endpoint_max": loss_out.endpoint_max.item(),
            "endpoint_abs_p95": loss_out.endpoint_abs_p95.item(),
            "endpoint_saturation_frac": loss_out.endpoint_saturation_frac.item(),
            "critic_action_min": loss_out.critic_action_min.item(),
            "critic_action_max": loss_out.critic_action_max.item(),
            "critic_action_abs_p95": loss_out.critic_action_abs_p95.item(),
            "replay_action_min": loss_out.replay_action_min.item(),
            "replay_action_max": loss_out.replay_action_max.item(),
            "replay_action_abs_p95": loss_out.replay_action_abs_p95.item(),
        }

    @staticmethod
    def _average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        merged: dict[str, list[float]] = {}
        for item in metrics:
            for key, value in item.items():
                merged.setdefault(key, []).append(value)
        return {key: sum(values) / len(values) for key, values in merged.items()}

    @staticmethod
    def _abs_p95(value: torch.Tensor) -> torch.Tensor:
        return torch.quantile(value.float().abs().flatten(), 0.95)


__all__ = ["FSDPLWDQAMWorker"]
