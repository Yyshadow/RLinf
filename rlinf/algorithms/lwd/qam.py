# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Q-learning with Adjoint Matching utilities for LWD policy extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@dataclass
class QAMLossOutput:
    """Container for QAM loss terms and diagnostics."""

    loss: Tensor
    qam_loss: Tensor
    anchor_loss: Tensor
    bc_loss: Tensor
    q_mean: Tensor
    q_min: Tensor
    q_std: Tensor
    q_head_gap: Tensor
    action_grad_norm: Tensor
    action_grad_clip_frac: Tensor
    adjoint_norm: Tensor
    qam_delta_norm: Tensor
    qam_delta_clip_frac: Tensor
    endpoint_min: Tensor
    endpoint_max: Tensor
    endpoint_abs_p95: Tensor
    endpoint_saturation_frac: Tensor
    critic_action_min: Tensor
    critic_action_max: Tensor
    critic_action_abs_p95: Tensor
    replay_action_min: Tensor
    replay_action_max: Tensor
    replay_action_abs_p95: Tensor
    q_ref_mean: Tensor | None = None
    q_cur_mean: Tensor | None = None
    q_cur_minus_ref: Tensor | None = None
    q_ref_min: Tensor | None = None
    q_cur_min: Tensor | None = None
    q_cur_minus_ref_min: Tensor | None = None
    cur_ref_endpoint_l2: Tensor | None = None
    cur_endpoint_abs_p95: Tensor | None = None
    cur_endpoint_saturation_frac: Tensor | None = None


def clip_by_global_norm(
    value: Tensor,
    max_norm: float | None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Clip each batch item by its flattened norm.

    Args:
        value: Tensor with batch as the first dimension.
        max_norm: Maximum norm. ``None`` or non-positive disables clipping.
        eps: Numerical stabilizer.

    Returns:
        The clipped tensor, original per-item norm, and clipped fraction.
    """

    norm = value.float().flatten(1).norm(dim=-1)
    if max_norm is None or max_norm <= 0:
        return value, norm, value.new_zeros(())

    scale = (float(max_norm) / (norm + eps)).clamp(max=1.0)
    clipped = value * scale.view(-1, *([1] * (value.dim() - 1))).to(value.dtype)
    clip_frac = (norm > float(max_norm)).float().mean()
    return clipped, norm, clip_frac


def flow_sigmas(
    timesteps: Tensor,
    min_sigma: float = 1e-3,
) -> Tensor:
    """Return the QAM stochastic-interpolant sigma for OpenPI timesteps.

    OpenPI denoising starts at ``t=1`` for Gaussian noise and moves toward
    ``t=0`` for the action chunk.  The QAM paper writes the path variable as
    ``w`` from noise to data, so we use ``w = 1 - t``.
    """

    w = (1.0 - timesteps.float()).clamp(0.0, 1.0)
    sigma = torch.sqrt((2.0 * (1.0 - w) * w).clamp_min(min_sigma**2))
    return sigma.to(dtype=timesteps.dtype)


def flow_ode_step(
    x_t: Tensor,
    v_t: Tensor,
    step_index: int,
    num_steps: int,
) -> Tensor:
    """Apply one deterministic OpenPI flow-ODE denoising step.

    This mirrors ``OpenPi0ForRLActionPrediction.sample_mean_var_val`` for
    ``sample_method == "flow_ode"`` without recomputing the vector field.
    """

    device = x_t.device
    dtype = x_t.dtype
    timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device, dtype=dtype)
    timesteps = torch.cat([timesteps, torch.zeros(1, device=device, dtype=dtype)])
    t_input = timesteps[step_index]
    delta = timesteps[step_index] - timesteps[step_index + 1]

    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1.0 - t_input)
    x0_weight = 1.0 - (t_input - delta)
    x1_weight = t_input - delta
    return x0_pred * x0_weight + x1_pred * x1_weight


def qam_vector_field_loss(
    v_theta: Tensor,
    v_beta: Tensor,
    adjoint: Tensor,
    timesteps: Tensor,
    delta_clip: float | None = None,
    min_sigma: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the local QAM vector-field regression loss.

    The QAM paper and reference implementation write the residual as
    ``2 * (f_theta - f_beta) / sigma + sigma * g`` for a noise-to-action
    vector field.  OpenPI trains the reverse-time velocity
    ``v = noise - action`` and integrates it with ``x_next = x - dt * v``.
    In this coordinate system, ``f_theta - f_beta`` becomes
    ``v_beta - v_theta``.
    """

    sigma = flow_sigmas(timesteps, min_sigma=min_sigma)
    sigma = sigma.view(-1, *([1] * (v_theta.dim() - 1))).to(v_theta.dtype)
    delta = v_beta - v_theta
    delta, delta_norm, clip_frac = clip_by_global_norm(delta, delta_clip)
    residual = 2.0 * delta / sigma + sigma * adjoint.to(v_theta.dtype)
    return residual.float().square().mean(), delta_norm.mean(), clip_frac


def bc_flow_matching_loss(
    v_theta: Tensor,
    target_velocity: Tensor,
) -> Tensor:
    """Standard OpenPI-direction flow matching loss."""

    return F.mse_loss(v_theta.float(), target_velocity.float())


__all__ = [
    "QAMLossOutput",
    "bc_flow_matching_loss",
    "clip_by_global_norm",
    "flow_ode_step",
    "flow_sigmas",
    "qam_vector_field_loss",
]
