# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LWD-style distributional value and chunk-action critic model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput

from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.value_model.configuration import ValueCriticConfig
from rlinf.models.embodiment.value_model.modeling_critic import ValueCriticModel


@dataclass
class LWDCriticConfig:
    """Config for a LWD-style critic.

    The observation backbone follows the existing ValueCriticModel:
    SigLIP/Gemma3 encodes the state, while this module adds a chunk-action
    encoder and clipped double-Q heads.
    """

    model_type: str = "lwd_critic"
    precision: str = "bf16"
    siglip_path: str = ""
    gemma3_path: str = ""
    critic_expert_variant: str = "gemma_1m"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 14
    action_horizon: int = 50
    max_token_len: int = 200
    max_language_len: int = 50
    freeze_vision_encoder: bool = False
    freeze_vlm: bool = False
    train_expert_only: bool = False
    stop_gradient_to_vlm: bool = False
    num_bins: int = 201
    v_min: float = -0.1
    v_max: float = 1.1
    value_dropout: float = 0.0
    quantile_tau: float = 0.6
    action_hidden_dim: int = 256
    q_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    num_q_heads: int = 2
    model_path: Optional[str] = None

    def update_from_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_value_config(self) -> ValueCriticConfig:
        dtype = {
            "bf16": "bfloat16",
            "bf16-mixed": "bfloat16",
            "fp16": "float16",
            "16": "float16",
            "16-mixed": "float16",
            "fp32": "float32",
            "32": "float32",
        }.get(str(self.precision), "float32")
        return ValueCriticConfig(
            critic_expert_variant=self.critic_expert_variant,
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max,
            siglip_path=self.siglip_path,
            gemma3_path=self.gemma3_path,
            value_dropout=self.value_dropout,
            dtype=dtype,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            action_expert_variant=self.action_expert_variant,
            freeze_vision_encoder=self.freeze_vision_encoder,
            freeze_vlm=self.freeze_vlm,
            train_expert_only=self.train_expert_only,
            max_language_len=self.max_language_len,
            stop_gradient_to_vlm=self.stop_gradient_to_vlm,
        )


@dataclass
class LWDCriticOutput(ModelOutput):
    q_values: Optional[torch.FloatTensor] = None
    q_min: Optional[torch.FloatTensor] = None
    value_logits: Optional[torch.FloatTensor] = None
    value_probs: Optional[torch.FloatTensor] = None
    value_mean: Optional[torch.FloatTensor] = None
    value_quantile: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    state_features: Optional[torch.FloatTensor] = None
    action_features: Optional[torch.FloatTensor] = None
    backward_anchor: Optional[torch.FloatTensor] = None


class ActionChunkEncoder(nn.Module):
    """Encode an action chunk with per-step MLP and temporal attention pooling."""

    def __init__(self, action_dim: int, hidden_dim: int, action_horizon: int):
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(torch.zeros(action_horizon, hidden_dim))
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.dim() == 2:
            action_chunk = action_chunk.unsqueeze(1)
        input_dtype = self.step_encoder[0].weight.dtype
        encoded = self.step_encoder(action_chunk.to(dtype=input_dtype))
        encoded = encoded + self.time_embedding[: encoded.shape[1]].to(encoded.dtype)
        weights = torch.softmax(self.score(encoded), dim=1)
        return torch.sum(weights * encoded, dim=1)


class LWDCriticModel(ValueCriticModel):
    """Distributional V(s) plus clipped double-Q Q(s, action_chunk)."""

    def __init__(self, config: LWDCriticConfig):
        self.lwd_config = config
        super().__init__(config.to_value_config())

        hidden_size = self.value_head.hidden_size
        self.action_encoder = ActionChunkEncoder(
            action_dim=config.action_dim,
            hidden_dim=config.action_hidden_dim,
            action_horizon=config.action_horizon,
        )
        self.q_head = MultiQHead(
            hidden_size=hidden_size,
            action_feature_dim=config.action_hidden_dim,
            hidden_dims=config.q_hidden_dims,
            num_q_heads=config.num_q_heads,
        )
        self.quantile_tau = float(config.quantile_tau)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        modules = super()._no_split_modules
        return modules + ["ActionChunkEncoder", "MultiQHead"]

    def encode_state(self, observation) -> tuple[Tensor, Optional[Tensor]]:
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            _,
            _,
        ) = self._preprocess_observation(observation)

        batch_size = lang_tokens.shape[0]
        prefix_embs, prefix_pad_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        _, state_features, _, _, backward_anchor = self._forward_expert(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_ar_masks,
            stop_gradient_to_vlm=getattr(self.config, "stop_gradient_to_vlm", False),
        )
        return state_features, backward_anchor

    def value_from_state_features(
        self,
        state_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values, logits, probs = self._compute_value_from_hidden(state_features)
        quantile = self.quantile_from_probs(probs)
        return values, logits, probs, quantile

    def quantile_from_probs(self, probs: Tensor, tau: Optional[float] = None) -> Tensor:
        tau = self.quantile_tau if tau is None else float(tau)
        cdf = torch.cumsum(probs, dim=-1)
        indices = (cdf >= tau).to(torch.long).argmax(dim=-1)
        atoms = self.value_head.atoms.to(device=probs.device, dtype=probs.dtype)
        return atoms[indices]

    def q_from_state_action(
        self,
        state_features: Tensor,
        action_chunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        action_features = self.action_encoder(action_chunk)
        q_values = self.q_head(state_features, action_features).float()
        return q_values, action_features

    def forward(
        self,
        observation,
        action_chunk: Optional[Tensor] = None,
        **kwargs,
    ) -> LWDCriticOutput:
        state_features, backward_anchor = self.encode_state(observation)
        values, logits, probs, quantile = self.value_from_state_features(state_features)

        q_values = None
        q_min = None
        action_features = None
        if action_chunk is not None:
            q_values, action_features = self.q_from_state_action(
                state_features, action_chunk
            )
            q_min = q_values.min(dim=-1).values

        return LWDCriticOutput(
            q_values=q_values,
            q_min=q_min,
            value_logits=logits,
            value_probs=probs,
            value_mean=values,
            value_quantile=quantile,
            atoms=self.value_head.atoms,
            state_features=state_features,
            action_features=action_features,
            backward_anchor=backward_anchor,
        )

    @torch.no_grad()
    def target_quantile(self, observation, tau: Optional[float] = None) -> Tensor:
        out = self.forward(observation)
        if tau is None:
            return out.value_quantile
        return self.quantile_from_probs(out.value_probs, tau=tau)

    @staticmethod
    def scalar_targets_to_categorical(
        target_values: Tensor,
        atoms: Tensor,
        v_min: float,
        v_max: float,
    ) -> Tensor:
        target_values = target_values.float().view(-1).clamp(v_min, v_max)
        num_bins = atoms.numel()
        delta_z = (v_max - v_min) / (num_bins - 1)
        b = (target_values - v_min) / delta_z
        lower = b.floor().long().clamp(0, num_bins - 1)
        upper = b.ceil().long().clamp(0, num_bins - 1)

        d_to_lower = b - lower.float()
        d_to_upper = upper.float() - b
        same_bin = lower == upper
        d_to_lower = torch.where(same_bin, torch.zeros_like(d_to_lower), d_to_lower)
        d_to_upper = torch.where(same_bin, torch.ones_like(d_to_upper), d_to_upper)

        target_probs = torch.zeros(
            target_values.shape[0],
            num_bins,
            device=target_values.device,
            dtype=atoms.dtype,
        )
        batch_idx = torch.arange(target_values.shape[0], device=target_values.device)
        target_probs[batch_idx, lower] += d_to_upper.to(target_probs.dtype)
        target_probs[batch_idx, upper] += d_to_lower.to(target_probs.dtype)
        return target_probs


__all__ = [
    "ActionChunkEncoder",
    "LWDCriticConfig",
    "LWDCriticModel",
    "LWDCriticOutput",
]
