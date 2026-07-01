# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _load_stats(stats_path: str | None) -> dict[str, Any]:
    if not stats_path:
        return {}
    return torch.load(stats_path, map_location="cpu", weights_only=False)


class QGFActionTransform(nn.Module):
    """Map between QGF model actions and environment actions.

    QGF learns in a normalized model-action space.  For Robotwin qpos control,
    arm joints are interpreted as deltas from the current qpos, while grippers
    stay absolute in Robotwin's normalized [0, 1] gripper coordinate.
    """

    def __init__(
        self,
        action_space: str,
        action_dim: int,
        num_action_chunks: int,
        stats_path: str | None,
        robotwin_delta_scale: float | list[float],
        robotwin_gripper_indices: list[int] | None = None,
    ):
        super().__init__()
        self.action_space = action_space
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.flat_action_dim = action_dim * num_action_chunks

        stats = _load_stats(stats_path)
        delta_scale = stats.get("robotwin_delta_scale", robotwin_delta_scale)
        delta_scale = torch.as_tensor(delta_scale, dtype=torch.float32).flatten()
        if delta_scale.numel() == 1:
            delta_scale = delta_scale.repeat(action_dim)
        elif delta_scale.numel() != action_dim:
            raise ValueError(
                "robotwin_delta_scale must be scalar or action_dim-sized, "
                f"got {delta_scale.numel()} for action_dim={action_dim}."
            )
        self.register_buffer("robotwin_delta_scale", delta_scale.clamp_min(1e-6))

        gripper_indices = (
            robotwin_gripper_indices
            if robotwin_gripper_indices is not None
            else self._infer_gripper_indices(action_dim)
        )
        gripper_mask = torch.zeros(action_dim, dtype=torch.bool)
        if gripper_indices:
            gripper_indices_t = torch.tensor(gripper_indices, dtype=torch.long)
            if (gripper_indices_t < 0).any() or (gripper_indices_t >= action_dim).any():
                raise ValueError(
                    "robotwin_gripper_indices must be within action_dim, "
                    f"got {gripper_indices} for action_dim={action_dim}."
                )
            gripper_mask[gripper_indices_t] = True
        self.register_buffer("gripper_mask", gripper_mask)
        self.register_buffer("arm_mask", ~gripper_mask)

    @staticmethod
    def _infer_gripper_indices(action_dim: int) -> list[int]:
        if action_dim % 7 == 0:
            return [6 + 7 * arm_id for arm_id in range(action_dim // 7)]
        return []

    def to_chunks(self, flat_actions: torch.Tensor) -> torch.Tensor:
        return flat_actions.reshape(-1, self.num_action_chunks, self.action_dim)

    def to_flat(self, chunk_actions: torch.Tensor) -> torch.Tensor:
        return chunk_actions.reshape(chunk_actions.shape[0], self.flat_action_dim)

    def decode(self, model_actions: torch.Tensor, raw_states: torch.Tensor) -> torch.Tensor:
        """Decode normalized model actions to actions executed by the environment."""
        if self.action_space == "identity":
            return self.to_chunks(model_actions)
        if self.action_space != "robotwin_delta_qpos":
            raise ValueError(f"Unsupported QGF action_space={self.action_space!r}.")

        model_chunks = self.to_chunks(model_actions).clamp(-1.0, 1.0)
        current = raw_states[:, : self.action_dim].to(
            device=model_chunks.device, dtype=model_chunks.dtype
        )
        env_chunks = []
        for chunk_id in range(self.num_action_chunks):
            model_chunk = model_chunks[:, chunk_id]
            env_chunk = current.clone()
            env_chunk[:, self.arm_mask] = (
                current[:, self.arm_mask]
                + model_chunk[:, self.arm_mask]
                * self.robotwin_delta_scale[self.arm_mask].to(model_chunk.device)
            )
            if self.gripper_mask.any():
                env_chunk[:, self.gripper_mask] = (
                    model_chunk[:, self.gripper_mask] + 1.0
                ) * 0.5
            env_chunks.append(env_chunk)
            current = env_chunk
        return torch.stack(env_chunks, dim=1).to(model_actions.dtype)

    def encode(self, env_actions: torch.Tensor, raw_states: torch.Tensor) -> torch.Tensor:
        """Encode environment qpos actions into QGF normalized model actions."""
        if self.action_space == "identity":
            if env_actions.dim() == 3:
                return self.to_flat(env_actions)
            return env_actions.reshape(env_actions.shape[0], self.flat_action_dim)
        if self.action_space != "robotwin_delta_qpos":
            raise ValueError(f"Unsupported QGF action_space={self.action_space!r}.")

        if env_actions.dim() == 2:
            env_chunks = self.to_chunks(env_actions)
        else:
            env_chunks = env_actions
        env_chunks = env_chunks.to(device=raw_states.device, dtype=raw_states.dtype)
        current = raw_states[:, : self.action_dim].to(env_chunks)
        model_chunks = []
        for chunk_id in range(self.num_action_chunks):
            env_chunk = env_chunks[:, chunk_id]
            model_chunk = torch.zeros_like(env_chunk)
            model_chunk[:, self.arm_mask] = (
                env_chunk[:, self.arm_mask] - current[:, self.arm_mask]
            ) / self.robotwin_delta_scale[self.arm_mask].to(env_chunk.device)
            if self.gripper_mask.any():
                model_chunk[:, self.gripper_mask] = (
                    env_chunk[:, self.gripper_mask].clamp(0.0, 1.0) * 2.0 - 1.0
                )
            model_chunks.append(model_chunk.clamp(-1.0, 1.0))
            current = env_chunk
        return self.to_flat(torch.stack(model_chunks, dim=1))
