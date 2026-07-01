#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FEATURES = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.state",
    "action",
}


def load_info(dataset: Path) -> dict:
    with (dataset / "meta" / "info.json").open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def check_feature_schema(info: dict, action_dim: int) -> None:
    features = info["features"]
    missing = REQUIRED_FEATURES - set(features)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    state_shape = tuple(features["observation.state"]["shape"])
    action_shape = tuple(features["action"]["shape"])
    if state_shape != (action_dim,):
        raise ValueError(f"observation.state shape must be ({action_dim},), got {state_shape}")
    if action_shape != (action_dim,):
        raise ValueError(f"action shape must be ({action_dim},), got {action_shape}")

    for key in [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]:
        if features[key]["dtype"] not in ("image", "video"):
            raise ValueError(f"{key} must be an image/video feature")


def validate_openpi_transform(dataset: Path, config_name: str) -> None:
    import openpi.training.data_loader as data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(config_name, repo_id=str(dataset))
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    transformed = data_loader.TransformedDataset(
        raw_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    sample = transformed[0]
    state = np.asarray(sample["state"])
    actions = np.asarray(sample["actions"])
    images = sample.get("images") or sample.get("image")
    if images is None:
        raise ValueError(f"OpenPI transform output has no image field: {sorted(sample)}")

    print("OpenPI transform check:")
    print(f"  state: shape={state.shape} dtype={state.dtype}")
    print(f"  actions: shape={actions.shape} dtype={actions.dtype}")
    if isinstance(images, dict):
        print(f"  image keys: {sorted(images.keys())}")
    else:
        print(f"  image: shape={np.asarray(images).shape} dtype={np.asarray(images).dtype}")
    print(f"  prompt: {sample.get('prompt', '')}")


def summarize_dataset(dataset: Path) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset.name, root=dataset)
    success_count = 0
    done_count = 0
    state_values = []
    action_values = []
    for idx in range(len(ds)):
        item = ds[idx]
        state_values.append(np.asarray(item["observation.state"], dtype=np.float32))
        action_values.append(np.asarray(item["action"], dtype=np.float32))
        if "success" in item:
            success_count += int(bool(np.asarray(item["success"]).reshape(-1)[0]))
        if "done" in item:
            done_count += int(bool(np.asarray(item["done"]).reshape(-1)[0]))

    states = np.stack(state_values)
    actions = np.stack(action_values)
    print("Dataset summary:")
    print(f"  episodes={ds.num_episodes} frames={ds.num_frames} tasks={ds.meta.tasks}")
    print(f"  success_frames={success_count} done_frames={done_count}")
    print(f"  state: min={states.min():.5f} max={states.max():.5f} mean={states.mean():.5f}")
    print(f"  action: min={actions.min():.5f} max={actions.max():.5f} mean={actions.mean():.5f}")
    if actions.shape[-1] % 7 == 0:
        gripper = actions[:, np.arange(6, actions.shape[-1], 7)]
        print(
            "  gripper: "
            f"min={gripper.min():.5f} max={gripper.max():.5f} "
            f"in_[0,1]={bool(((gripper >= 0) & (gripper <= 1)).all())}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a RoboTwin Aloha LeRobot dataset for OpenPI.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config-name", default="pi05_aloha_robotwin")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--skip-openpi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset)
    check_feature_schema(info, args.action_dim)
    summarize_dataset(args.dataset)
    if not args.skip_openpi:
        validate_openpi_transform(args.dataset, args.config_name)
    print("validation passed")


if __name__ == "__main__":
    main()
