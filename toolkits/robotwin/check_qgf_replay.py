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
from typing import Any

import torch


def iter_trajectory_files(path: Path):
    if path.is_file():
        yield path
        return
    for pattern in ("trajectory_*.pt", "*.pt"):
        files = sorted(path.rglob(pattern))
        if files:
            yield from files
            return


def load_trajectory(path: Path) -> dict[str, Any]:
    traj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(traj, "__dataclass_fields__"):
        return {
            field: getattr(traj, field)
            for field in traj.__dataclass_fields__
            if getattr(traj, field, None) is not None
        }
    return traj


def flatten_time_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 3:
        return tensor
    return tensor.reshape(-1, *tensor.shape[2:])


def tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = flatten_time_batch(tensor).float()
    flat = tensor.reshape(tensor.shape[0], -1)
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={flat.min().item():.5f} max={flat.max().item():.5f} "
        f"mean={flat.mean().item():.5f} std={flat.std().item():.5f}"
    )


def summarize(path: Path, action_dim: int):
    files = list(iter_trajectory_files(path))
    if not files:
        raise RuntimeError(f"No trajectory .pt files found under {path}")

    print(f"files: {len(files)}")
    totals = {
        "actions": [],
        "model_action": [],
        "env_action": [],
        "states": [],
        "rewards": [],
        "dones": [],
    }

    for file_path in files:
        traj = load_trajectory(file_path)
        forward_inputs = traj.get("forward_inputs", {}) or {}
        curr_obs = traj.get("curr_obs", {}) or {}

        if torch.is_tensor(traj.get("actions", None)):
            totals["actions"].append(flatten_time_batch(traj["actions"]))
        if torch.is_tensor(forward_inputs.get("model_action", None)):
            totals["model_action"].append(flatten_time_batch(forward_inputs["model_action"]))
        if torch.is_tensor(forward_inputs.get("env_action", None)):
            totals["env_action"].append(flatten_time_batch(forward_inputs["env_action"]))
        if torch.is_tensor(curr_obs.get("states", None)):
            totals["states"].append(flatten_time_batch(curr_obs["states"]))
        elif torch.is_tensor(forward_inputs.get("states", None)):
            totals["states"].append(flatten_time_batch(forward_inputs["states"]))
        if torch.is_tensor(traj.get("rewards", None)):
            totals["rewards"].append(flatten_time_batch(traj["rewards"]))
        if torch.is_tensor(traj.get("dones", None)):
            totals["dones"].append(flatten_time_batch(traj["dones"]))

    for key, tensors in totals.items():
        if tensors:
            print(tensor_stats(key, torch.cat(tensors, dim=0)))

    if totals["actions"]:
        actions = torch.cat(totals["actions"], dim=0).reshape(-1, action_dim)
        print(
            "actions_normalized_check: "
            f"min={actions.min().item():.5f} max={actions.max().item():.5f}"
        )

    if totals["env_action"]:
        env_action = torch.cat(totals["env_action"], dim=0).reshape(-1, action_dim)
        if action_dim % 7 == 0:
            gripper = env_action[:, torch.arange(6, action_dim, 7)]
            print(
                "robotwin_gripper_check: "
                f"min={gripper.min().item():.5f} max={gripper.max().item():.5f} "
                f"in_[0,1]={bool(((gripper >= 0) & (gripper <= 1)).all())}"
            )

    if totals["dones"]:
        dones = torch.cat(totals["dones"], dim=0).bool()
        print(f"done_count: {int(dones.sum().item())}")


def main():
    parser = argparse.ArgumentParser(description="Inspect QGF Robotwin replay tensors.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--action-dim", type=int, default=14)
    args = parser.parse_args()
    summarize(args.input, args.action_dim)


if __name__ == "__main__":
    main()
