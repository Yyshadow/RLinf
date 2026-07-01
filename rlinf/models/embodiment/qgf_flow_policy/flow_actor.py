# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QGFFlowActor(nn.Module):
    """Flow-matching actor with explicit velocity and sampling APIs."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        num_action_chunks: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        denoising_steps: int,
        flow_actor_type: str,
        use_batch_norm: bool,
        batch_norm_momentum: float,
        noise_std_head: bool,
        log_std_min_train: float,
        log_std_max_train: float,
        log_std_min_rollout: float,
        log_std_max_rollout: float,
        noise_std_train: float,
        noise_std_rollout: float,
    ):
        super().__init__()
        if flow_actor_type not in {"QGFFlowMLP", "QGFFlowTransformer"}:
            flow_actor_type = "QGFFlowTransformer"

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.flat_action_dim = action_dim * num_action_chunks
        self.denoising_steps = denoising_steps
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout

        self.obs_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.action_proj = nn.Linear(self.flat_action_dim, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if flow_actor_type == "QGFFlowTransformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.backbone_type = "transformer"
        else:
            layers = []
            for _ in range(n_layers):
                layers.extend([nn.Linear(d_model, d_model), nn.SiLU(), nn.LayerNorm(d_model)])
            self.backbone = nn.Sequential(*layers)
            self.backbone_type = "mlp"

        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.flat_action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def velocity(
        self,
        features: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        obs_token = self.obs_proj(features)
        action_token = self.action_proj(noisy_actions)
        time_token = self.time_proj(
            self.timestep_embedding(timesteps, obs_token.shape[-1]).to(obs_token.dtype)
        )
        token = action_token + obs_token + time_token
        if self.backbone_type == "transformer":
            token = self.backbone(token.unsqueeze(1)).squeeze(1)
        else:
            token = self.backbone(token)
        return self.velocity_head(token)

    def sample(
        self,
        features: torch.Tensor,
        train: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        action = torch.randn(
            batch_size,
            self.flat_action_dim,
            device=features.device,
            dtype=features.dtype,
            generator=generator,
        )
        initial_action = action
        dt = 1.0 / float(self.denoising_steps)
        for step in range(self.denoising_steps):
            t = torch.full(
                (batch_size,),
                step / float(self.denoising_steps),
                device=features.device,
                dtype=features.dtype,
            )
            action = action + self.velocity(features, action, t) * dt

        noise_std = self.noise_std_train if train else self.noise_std_rollout
        if noise_std > 0:
            action = action + torch.randn_like(action) * noise_std
        action = action.tanh()

        # Approximate density term for SAC bookkeeping.  The QGF BC/guidance path
        # trains the velocity field directly and does not rely on exact flow log-probs.
        log_prob = -0.5 * initial_action.square().sum(dim=-1, keepdim=True)
        return action, log_prob

    def forward(
        self, features: torch.Tensor, train: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample(features, train=train)

    def flow_matching_loss(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x1 = target_actions.reshape(target_actions.shape[0], -1).clamp(-1.0, 1.0)
        x0 = torch.randn(
            x1.shape,
            device=x1.device,
            dtype=x1.dtype,
            generator=generator,
        )
        step_ids = torch.randint(
            0,
            self.denoising_steps + 1,
            (x1.shape[0],),
            device=x1.device,
            generator=generator,
        )
        t = step_ids.to(dtype=x1.dtype) / float(self.denoising_steps)
        t_view = t.unsqueeze(-1)
        x_t = x0 * (1.0 - t_view) + x1 * t_view
        target_velocity = x1 - x0
        pred_velocity = self.velocity(features, x_t, t)
        return F.mse_loss(pred_velocity, target_velocity, reduction="none")

    def to_chunks(self, flat_actions: torch.Tensor) -> torch.Tensor:
        return flat_actions.reshape(-1, self.num_action_chunks, self.action_dim)
