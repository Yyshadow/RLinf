# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Loss utilities for LWD-style critic training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from .lwd_critic_model import LWDCriticModel


@dataclass
class LWDLossOutput:
    loss: torch.Tensor
    q_loss: torch.Tensor
    value_loss: torch.Tensor
    target_q: torch.Tensor
    reward_sum: torch.Tensor
    q_min: torch.Tensor


def discounted_chunk_sum(reward_chunk: torch.Tensor, gamma: float) -> torch.Tensor:
    rewards = reward_chunk.float()
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
    horizon = rewards.shape[-1]
    discounts = torch.pow(
        torch.full((horizon,), gamma, device=rewards.device, dtype=rewards.dtype),
        torch.arange(horizon, device=rewards.device, dtype=rewards.dtype),
    )
    return torch.sum(rewards * discounts, dim=-1)


def compute_lwd_losses(
    model: LWDCriticModel,
    target_model: LWDCriticModel,
    batch: dict,
    gamma: float,
    value_loss_weight: float = 1.0,
    q_loss_weight: float = 1.0,
) -> LWDLossOutput:
    out = model(
        observation=batch["observation"],
        action_chunk=batch["action_chunk"],
    )

    with torch.no_grad():
        next_quantile = target_model.target_quantile(batch["next_observation"])
        reward_sum = discounted_chunk_sum(batch["reward_chunk"], gamma)
        done = batch["done"].float().view(-1)
        horizon = batch["reward_chunk"].shape[-1]
        bootstrap = (gamma**horizon) * (1.0 - done) * next_quantile.float()
        target_q = reward_sum + bootstrap

        target_out = target_model(
            observation=batch["observation"],
            action_chunk=batch["action_chunk"],
        )
        value_targets = target_out.q_values.min(dim=-1).values

    q_targets = target_q[:, None].expand_as(out.q_values)
    q_loss = F.mse_loss(out.q_values.float(), q_targets)

    value_target_probs = model.scalar_targets_to_categorical(
        value_targets,
        atoms=out.atoms.to(value_targets.device),
        v_min=model.v_min,
        v_max=model.v_max,
    )
    value_loss = -(
        value_target_probs * F.log_softmax(out.value_logits.float(), dim=-1)
    ).sum(dim=-1).mean()

    loss = q_loss_weight * q_loss + value_loss_weight * value_loss
    return LWDLossOutput(
        loss=loss,
        q_loss=q_loss,
        value_loss=value_loss,
        target_q=target_q,
        reward_sum=reward_sum,
        q_min=out.q_values.min(dim=-1).values,
    )


@torch.no_grad()
def update_ema_target(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    tau: float,
) -> None:
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


__all__ = [
    "LWDLossOutput",
    "compute_lwd_losses",
    "discounted_chunk_sum",
    "update_ema_target",
]
