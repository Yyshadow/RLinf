#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def iter_trajectory_files(path: Path):
    if path.is_file():
        yield path
        return
    for pattern in ("trajectory_*.pt", "*.pt"):
        yielded = False
        for file_path in sorted(path.rglob(pattern)):
            yielded = True
            yield file_path
        if yielded:
            return


def flatten_time_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 3:
        return tensor
    return tensor.reshape(-1, *tensor.shape[2:])


def collect_stats(input_path: Path, action_dim: int, percentile: float) -> dict:
    states = []
    env_actions = []
    model_actions = []

    for file_path in iter_trajectory_files(input_path):
        traj = torch.load(file_path, map_location="cpu", weights_only=False)
        curr_obs = traj.get("curr_obs", {})
        forward_inputs = traj.get("forward_inputs", {})
        if "states" in curr_obs:
            states.append(flatten_time_batch(curr_obs["states"]).float())
        elif "states" in forward_inputs:
            states.append(flatten_time_batch(forward_inputs["states"]).float())

        if "env_action" in forward_inputs:
            env_actions.append(flatten_time_batch(forward_inputs["env_action"]).float())
        if "model_action" in forward_inputs:
            model_actions.append(flatten_time_batch(forward_inputs["model_action"]).float())

    if not states:
        raise RuntimeError(f"No states found under {input_path}")

    states_t = torch.cat(states, dim=0)
    stats = {
        "state_mean": states_t.mean(dim=0),
        "state_std": states_t.std(dim=0).clamp_min(1e-6),
        "num_state_samples": torch.tensor(states_t.shape[0]),
    }

    if env_actions:
        env_actions_t = torch.cat(env_actions, dim=0).reshape(-1, action_dim)
        current = states_t[: env_actions_t.shape[0], :action_dim]
        delta = env_actions_t - current
        gripper_mask = torch.zeros(action_dim, dtype=torch.bool)
        if action_dim % 7 == 0:
            gripper_mask[torch.arange(6, action_dim, 7)] = True
        arm_delta = delta[:, ~gripper_mask].abs()
        scale = torch.ones(action_dim)
        if arm_delta.numel() > 0:
            scale[~gripper_mask] = torch.quantile(
                arm_delta, percentile / 100.0, dim=0
            ).clamp_min(1e-3)
        stats["robotwin_delta_scale"] = scale
        stats["env_action_min"] = env_actions_t.min(dim=0).values
        stats["env_action_max"] = env_actions_t.max(dim=0).values

    if model_actions:
        model_actions_t = torch.cat(model_actions, dim=0)
        stats["model_action_mean"] = model_actions_t.mean(dim=0)
        stats["model_action_std"] = model_actions_t.std(dim=0).clamp_min(1e-6)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build QGF normalization stats.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    stats = collect_stats(args.input, args.action_dim, args.percentile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)
    print(f"saved stats to {args.output}")
    for key, value in stats.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} mean={value.float().mean().item():.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
