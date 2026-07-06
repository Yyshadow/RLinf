# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Export held-out LWD LeRobot episodes as quick-look MP4 videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from rlinf.data.datasets.lwd.chunk_dataset import LWDChunkDataset


DEFAULT_DATA_ROOT = Path("/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split")
DEFAULT_OUTPUT_DIR = Path("/data/wam_codebase/RLinf/outputs/lwd_critic_value_curves/videos")
DEFAULT_SPLITS = (
    ("success", "beat_block_hammer_success_eval"),
    ("failed", "beat_block_hammer_failed_eval"),
    ("nearmiss", "beat_block_hammer_nearmiss_eval"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--camera-key",
        action="append",
        default=None,
        help="Camera key to include. Repeat for multiple cameras.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Eval split in label:dataset_dir format. Defaults to success/failed/nearmiss.",
    )
    return parser.parse_args()


def parse_splits(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_SPLITS)
    splits = []
    for value in values:
        if ":" not in value:
            raise ValueError("--split must use label:dataset_dir format.")
        label, dataset_name = value.split(":", 1)
        splits.append((label.strip(), dataset_name.strip()))
    return splits


def tensor_to_uint8_image(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return array


def add_title_bar(image: np.ndarray, text: str) -> np.ndarray:
    height, width = image.shape[:2]
    bar = np.full((34, width, 3), 245, dtype=np.uint8)
    out = np.concatenate([bar, image], axis=0)
    cv2.putText(
        out,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return out


def make_frame(
    images: dict[str, Any],
    camera_keys: list[str],
    *,
    label: str,
    dataset_name: str,
    episode_index: int,
    frame_idx: int,
    fps: int,
) -> np.ndarray:
    panels = []
    for key in camera_keys:
        image = tensor_to_uint8_image(images[key])
        image = add_title_bar(image, key)
        panels.append(image)
    frame = np.concatenate(panels, axis=1)
    text = (
        f"{label} | {dataset_name} | episode {episode_index:03d} | "
        f"frame {frame_idx:04d} | t={frame_idx / fps:.1f}s"
    )
    return add_title_bar(frame, text)


def export_video(
    *,
    data_root: Path,
    output_dir: Path,
    label: str,
    dataset_name: str,
    episode_index: int,
    camera_keys: list[str],
) -> dict[str, Any]:
    dataset = LWDChunkDataset(
        dataset_path=str(data_root / dataset_name),
        action_horizon=50,
        norm_stats_path=str(data_root / "pi05_norm_stats.json"),
        use_quantile_norm=True,
        adapt_to_pi=True,
    )
    starts = dataset.dataset.episode_data_index["from"]
    stops = dataset.dataset.episode_data_index["to"]
    if episode_index < 0 or episode_index >= len(starts):
        raise IndexError(
            f"episode_index={episode_index} is out of range for {dataset_name}; "
            f"num_episodes={len(starts)}"
        )

    start = int(starts[episode_index].item())
    stop = int(stops[episode_index].item())
    fps = int(dataset.fps)

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    video_path = label_dir / f"{dataset_name}_episode_{episode_index:03d}_triplet.mp4"

    writer = None
    try:
        for index in range(start, stop):
            sample = dataset.dataset[index]
            images = dataset._extract_images(sample)
            missing = [key for key in camera_keys if key not in images]
            if missing:
                raise KeyError(f"{dataset_name} missing camera keys: {missing}")
            frame_idx = int(sample["frame_index"].item())
            frame = make_frame(
                images,
                camera_keys,
                label=label,
                dataset_name=dataset_name,
                episode_index=episode_index,
                frame_idx=frame_idx,
                fps=fps,
            )
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {video_path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()

    return {
        "label": label,
        "dataset": dataset_name,
        "episode_index": episode_index,
        "fps": fps,
        "frames": stop - start,
        "camera_keys": camera_keys,
        "video_path": str(video_path),
    }


def main() -> None:
    args = parse_args()
    camera_keys = args.camera_key or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for label, dataset_name in parse_splits(args.split):
        item = export_video(
            data_root=args.data_root,
            output_dir=args.output_dir,
            label=label,
            dataset_name=dataset_name,
            episode_index=args.episode_index,
            camera_keys=camera_keys,
        )
        manifest.append(item)
        print(f"Saved video: {item['video_path']}")

    manifest_path = args.output_dir / f"episode_{args.episode_index:03d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
