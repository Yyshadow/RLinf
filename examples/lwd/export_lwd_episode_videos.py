# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Export LWD LeRobot episodes as quick-look MP4 videos.

The script is intended for visually auditing failed/nearmiss data.  It exports
triplet camera videos directly from the offline LeRobot parquet files, so it
does not need to run RoboTwin simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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
CORE_INDEX_FIELDS = (
    "label",
    "dataset",
    "episode_index",
    "video_path",
    "frames",
    "fps",
    "stride",
    "seed",
    "source_episode_id",
    "source_episode",
    "perturb_type",
    "perturb_scale",
    "active_arm",
    "replay_start_fraction",
    "perturb_start_frame",
    "drop_hammer",
    "bad_grasp_or_skewed",
    "miss_block_far",
    "near_miss",
    "timeout_or_no_contact",
    "other_note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Episode index to export. Repeat for multiple episodes. If omitted, "
            "the script exports episode 0 unless --all-episodes, --max-episodes, "
            "or --episodes-per-perturb is set."
        ),
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Export all episodes from each selected split.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Export the first N selected episodes from each split.",
    )
    parser.add_argument(
        "--episodes-per-perturb",
        type=int,
        default=None,
        help="Export at most N episodes for each perturb_type in each split.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one frame every STRIDE frames. Use 2 or 3 for cheaper previews.",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=None,
        help="Output video FPS. Defaults to dataset_fps / stride.",
    )
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episode_metadata(dataset_path: Path) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(dataset_path / "meta" / "robotwin_episode_metadata.jsonl")
    metadata = {}
    for row in rows:
        episode_index = row.get("episode_id", row.get("episode_index"))
        if episode_index is not None:
            metadata[int(episode_index)] = row
    return metadata


def perturb_start_frame(metadata: dict[str, Any], fallback_frames: int) -> int | None:
    fraction = metadata.get("replay_start_fraction")
    if fraction is None:
        return None
    num_steps = int(metadata.get("num_steps") or fallback_frames)
    return int(round(float(fraction) * num_steps))


def safe_name(value: Any) -> str:
    text = str(value) if value not in (None, "") else "na"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def video_filename(dataset_name: str, episode_index: int, metadata: dict[str, Any]) -> str:
    parts = [
        dataset_name,
        f"episode_{episode_index:03d}",
        safe_name(metadata.get("perturb_type")),
        safe_name(metadata.get("active_arm")),
        f"seed{safe_name(metadata.get('seed'))}",
    ]
    return "_".join(parts) + "_triplet.mp4"


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
    metadata: dict[str, Any] | None = None,
    perturb_start: int | None = None,
) -> np.ndarray:
    panels = []
    for key in camera_keys:
        image = tensor_to_uint8_image(images[key])
        image = add_title_bar(image, key)
        panels.append(image)
    frame = np.concatenate(panels, axis=1)
    phase = ""
    if perturb_start is not None:
        phase = "pre" if frame_idx < perturb_start else "post"
        phase = f" | perturb@{perturb_start:04d} {phase}"
    meta_text = ""
    if metadata:
        perturb_type = metadata.get("perturb_type", "na")
        active_arm = metadata.get("active_arm", "na")
        seed = metadata.get("seed", "na")
        meta_text = f" | {perturb_type} {active_arm} seed={seed}"
    text = (
        f"{label} | {dataset_name} | episode {episode_index:03d} | "
        f"frame {frame_idx:04d} | t={frame_idx / fps:.1f}s"
        f"{phase}{meta_text}"
    )
    return add_title_bar(frame, text)


def select_episode_indices(
    *,
    num_episodes: int,
    metadata_by_episode: dict[int, dict[str, Any]],
    episode_indices: list[int] | None,
    all_episodes: bool,
    max_episodes: int | None,
    episodes_per_perturb: int | None,
) -> list[int]:
    if episode_indices:
        selected = episode_indices
    elif episodes_per_perturb is not None:
        groups: dict[str, list[int]] = defaultdict(list)
        for episode_index in range(num_episodes):
            metadata = metadata_by_episode.get(episode_index, {})
            perturb_type = str(metadata.get("perturb_type", "unknown"))
            groups[perturb_type].append(episode_index)
        selected = []
        for perturb_type in sorted(groups):
            selected.extend(groups[perturb_type][:episodes_per_perturb])
    elif all_episodes or max_episodes is not None:
        selected = list(range(num_episodes))
    else:
        selected = [0]

    if max_episodes is not None:
        selected = selected[:max_episodes]
    invalid = [idx for idx in selected if idx < 0 or idx >= num_episodes]
    if invalid:
        raise IndexError(
            f"Episode indices out of range for num_episodes={num_episodes}: {invalid}"
        )
    return selected


def export_episode_video(
    *,
    dataset: LWDChunkDataset,
    output_dir: Path,
    label: str,
    dataset_name: str,
    episode_index: int,
    metadata: dict[str, Any],
    camera_keys: list[str],
    stride: int,
    output_fps: int | None,
) -> dict[str, Any]:
    starts = dataset.dataset.episode_data_index["from"]
    stops = dataset.dataset.episode_data_index["to"]

    start = int(starts[episode_index].item())
    stop = int(stops[episode_index].item())
    fps = int(dataset.fps)
    writer_fps = output_fps or max(1, int(round(fps / stride)))
    perturb_start = perturb_start_frame(metadata, stop - start)

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    video_path = label_dir / video_filename(dataset_name, episode_index, metadata)

    writer = None
    try:
        for index in range(start, stop, stride):
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
                metadata=metadata,
                perturb_start=perturb_start,
            )
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    writer_fps,
                    (width, height),
                )
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
        "fps": writer_fps,
        "dataset_fps": fps,
        "stride": stride,
        "frames": stop - start,
        "camera_keys": camera_keys,
        "video_path": str(video_path),
        "seed": metadata.get("seed"),
        "source_episode_id": metadata.get("source_episode_id"),
        "source_episode": metadata.get("source_episode"),
        "perturb_type": metadata.get("perturb_type"),
        "perturb_scale": metadata.get("perturb_scale"),
        "active_arm": metadata.get("active_arm"),
        "replay_start_fraction": metadata.get("replay_start_fraction"),
        "perturb_start_frame": perturb_start,
        "drop_hammer": "",
        "bad_grasp_or_skewed": "",
        "miss_block_far": "",
        "near_miss": "",
        "timeout_or_no_contact": "",
        "other_note": "",
    }


def export_split(
    *,
    data_root: Path,
    output_dir: Path,
    label: str,
    dataset_name: str,
    camera_keys: list[str],
    episode_indices: list[int] | None,
    all_episodes: bool,
    max_episodes: int | None,
    episodes_per_perturb: int | None,
    stride: int,
    output_fps: int | None,
) -> list[dict[str, Any]]:
    dataset_path = data_root / dataset_name
    dataset = LWDChunkDataset(
        dataset_path=str(dataset_path),
        action_horizon=1,
        norm_stats_path=str(data_root / "norm_stats.json"),
        use_quantile_norm=True,
        adapt_to_pi=True,
    )
    starts = dataset.dataset.episode_data_index["from"]
    metadata_by_episode = load_episode_metadata(dataset_path)
    selected_indices = select_episode_indices(
        num_episodes=len(starts),
        metadata_by_episode=metadata_by_episode,
        episode_indices=episode_indices,
        all_episodes=all_episodes,
        max_episodes=max_episodes,
        episodes_per_perturb=episodes_per_perturb,
    )

    manifest = []
    for episode_index in selected_indices:
        item = export_episode_video(
            dataset=dataset,
            output_dir=output_dir,
            label=label,
            dataset_name=dataset_name,
            episode_index=episode_index,
            metadata=metadata_by_episode.get(episode_index, {}),
            camera_keys=camera_keys,
            stride=stride,
            output_fps=output_fps,
        )
        manifest.append(item)
        print(f"Saved video: {item['video_path']}")
    return manifest


def write_index(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "video_index.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        for item in manifest:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_path = output_dir / "video_index.csv"
    fieldnames = list(CORE_INDEX_FIELDS)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            writer.writerow({key: item.get(key, "") for key in fieldnames})

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# LWD Failure Video Audit",
                "",
                "每个 mp4 是从离线 LeRobot 数据集直接导出的三视角视频。",
                "",
                "建议在 `video_index.csv` 中人工填写这些列：",
                "",
                "- `drop_hammer`: 锤子明显掉落或失去控制。",
                "- `bad_grasp_or_skewed`: 抓取姿态明显歪、夹持不稳。",
                "- `miss_block_far`: 锤头明显没有到红色方块附近。",
                "- `near_miss`: 视觉上接近成功，但可能差几厘米或没有敲下去。",
                "- `timeout_or_no_contact`: 动作完成但没有明显接触/敲击。",
                "- `other_note`: 其它备注。",
                "",
                "视频标题里的 `perturb@frame` 是离线失败数据开始加入扰动的大致帧号。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved index: {manifest_path}")
    print(f"Saved CSV for manual labels: {csv_path}")


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    camera_keys = args.camera_key or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for label, dataset_name in parse_splits(args.split):
        manifest.extend(
            export_split(
            data_root=args.data_root,
            output_dir=args.output_dir,
            label=label,
            dataset_name=dataset_name,
            camera_keys=camera_keys,
                episode_indices=args.episode_index,
                all_episodes=args.all_episodes,
                max_episodes=args.max_episodes,
                episodes_per_perturb=args.episodes_per_perturb,
                stride=args.stride,
                output_fps=args.output_fps,
            )
        )
    write_index(args.output_dir, manifest)


if __name__ == "__main__":
    main()
