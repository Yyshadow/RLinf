#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


CAM_HIGH = "observation.images.cam_high"
CAM_LEFT = "observation.images.cam_left_wrist"
CAM_RIGHT = "observation.images.cam_right_wrist"
STATE = "observation.state"
ACTION = "action"


def image_feature(shape: tuple[int, int, int]) -> dict[str, Any]:
    return {"dtype": "image", "shape": list(shape), "names": ["height", "width", "channel"]}


def vector_feature(dim: int) -> dict[str, Any]:
    return {"dtype": "float32", "shape": (dim,), "names": ["value"]}


def scalar_feature(dtype: str, name: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": (1,), "names": [name]}


def aloha_features(
    image_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
) -> dict[str, dict[str, Any]]:
    return {
        CAM_HIGH: image_feature(image_shape),
        CAM_LEFT: image_feature(image_shape),
        CAM_RIGHT: image_feature(image_shape),
        STATE: vector_feature(state_dim),
        ACTION: vector_feature(action_dim),
        "reward": scalar_feature("float32", "reward"),
        "done": scalar_feature("bool", "done"),
        "success": scalar_feature("bool", "success"),
    }


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        arr = np.asarray(value.convert("RGB"))
    elif isinstance(value, dict) and ("bytes" in value or "path" in value):
        raw = value.get("bytes")
        if raw is None:
            value = value["path"]
            arr = np.asarray(Image.open(value).convert("RGB"))
        else:
            arr = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
    else:
        arr = to_numpy(value)

    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def bytes_to_image(raw: bytes | np.bytes_) -> np.ndarray:
    # RoboTwin hdf5 stores JPEG bytes produced from RGB frames via cv2.imencode,
    # which writes them as if they were BGR. Swap channels back here.
    image = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
    return image[..., ::-1]


def get_first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def scalar_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return bool(arr.reshape(-1)[0])


def scalar_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def make_frame(
    image: np.ndarray,
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    reward: float,
    done: bool,
    success: bool,
) -> dict[str, Any]:
    if left_wrist is None:
        left_wrist = np.zeros_like(image)
    if right_wrist is None:
        right_wrist = np.zeros_like(image)
    return {
        CAM_HIGH: to_hwc_uint8(image),
        CAM_LEFT: to_hwc_uint8(left_wrist),
        CAM_RIGHT: to_hwc_uint8(right_wrist),
        STATE: np.asarray(state, dtype=np.float32).reshape(-1),
        ACTION: np.asarray(action, dtype=np.float32).reshape(-1),
        "reward": np.array([reward], dtype=np.float32),
        "done": np.array([done], dtype=bool),
        "success": np.array([success], dtype=bool),
        "task": task,
    }


def iter_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.hdf5"))


def infer_task_from_hdf5(path: Path, explicit_task: str | None) -> str:
    if explicit_task:
        return explicit_task
    if path.parent.name == "data":
        return path.parent.parent.name
    return path.stem


def hdf5_episode_to_frames(
    path: Path,
    *,
    task: str,
    success: bool,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with h5py.File(path, "r") as data:
        qpos = np.asarray(data["joint_action/vector"], dtype=np.float32)
        if qpos.shape[0] < 2:
            return frames

        head = data["observation/head_camera/rgb"]
        left = data["observation/left_camera/rgb"] if "observation/left_camera/rgb" in data else None
        right = data["observation/right_camera/rgb"] if "observation/right_camera/rgb" in data else None

        for i in range(qpos.shape[0] - 1):
            is_final = i == qpos.shape[0] - 2
            frame_success = success and is_final
            frames.append(
                make_frame(
                    image=bytes_to_image(head[i]),
                    left_wrist=bytes_to_image(left[i]) if left is not None else None,
                    right_wrist=bytes_to_image(right[i]) if right is not None else None,
                    state=qpos[i],
                    action=qpos[i + 1],
                    task=task,
                    reward=1.0 if frame_success else 0.0,
                    done=is_final,
                    success=frame_success,
                )
            )
    return frames


def load_lerobot_tasks(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if not tasks_path.exists():
        return tasks
    with tasks_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def iter_lerobot_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/episode_*.parquet"))


def lerobot_episode_to_frames(
    parquet_path: Path,
    *,
    tasks: dict[int, str],
    default_task: str,
    default_success: bool,
) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    frames: list[dict[str, Any]] = []
    if not rows:
        return frames

    for row in rows:
        image = get_first(row, [CAM_HIGH, "image", "main_image"])
        left = get_first(row, [CAM_LEFT, "wrist_image-0", "wrist_image/0", "wrist_image"])
        right = get_first(row, [CAM_RIGHT, "wrist_image-1", "wrist_image/1"])
        state = get_first(row, [STATE, "state", "states"])
        action = get_first(row, [ACTION, "actions", "action"])
        if image is None or state is None or action is None:
            continue

        task = tasks.get(int(row.get("task_index", -1)), default_task)
        success = scalar_bool(
            get_first(row, ["success", "is_success"]),
            default=default_success,
        )
        done = scalar_bool(row.get("done"), default=False)
        reward = scalar_float(row.get("reward"), default=1.0 if done and success else 0.0)

        frames.append(
            make_frame(
                image=to_hwc_uint8(image),
                left_wrist=to_hwc_uint8(left) if left is not None else None,
                right_wrist=to_hwc_uint8(right) if right is not None else None,
                state=state,
                action=action,
                task=task,
                reward=reward,
                done=done,
                success=success,
            )
        )
    if frames and not any(bool(frame["done"][0]) for frame in frames):
        frames[-1]["done"] = np.array([True], dtype=bool)
    return frames


def discover_source_kind(path: Path) -> str:
    if (path / "meta" / "info.json").exists():
        return "lerobot"
    if path.is_file() and path.suffix == ".hdf5":
        return "hdf5"
    if path.is_dir() and list(path.rglob("*.hdf5")):
        return "hdf5"
    raise ValueError(f"Unsupported source: {path}")


def build_episodes(args: argparse.Namespace) -> list[tuple[str, list[dict[str, Any]]]]:
    source = args.source
    source_kind = args.source_type if args.source_type != "auto" else discover_source_kind(source)
    episodes: list[tuple[str, list[dict[str, Any]]]] = []

    if source_kind == "hdf5":
        for path in iter_hdf5_files(source):
            task = infer_task_from_hdf5(path, args.task)
            frames = hdf5_episode_to_frames(path, task=task, success=args.success)
            if frames:
                episodes.append((path.stem, frames))
        return episodes

    tasks = load_lerobot_tasks(source)
    default_task = args.task or source.name
    for path in iter_lerobot_parquets(source):
        frames = lerobot_episode_to_frames(
            path,
            tasks=tasks,
            default_task=default_task,
            default_success=args.success,
        )
        if frames:
            episodes.append((path.stem, frames))
    return episodes


def write_metadata(output: Path, records: list[dict[str, Any]]) -> None:
    meta_path = output / "meta" / "robotwin_episode_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as file_obj:
        for row in records:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lerobot_dataset(args: argparse.Namespace, episodes: list[tuple[str, list[dict[str, Any]]]]) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise RuntimeError(f"No valid episodes found in {args.source}")

    first_frame = episodes[0][1][0]
    state_dim = int(first_frame[STATE].shape[-1])
    action_dim = int(first_frame[ACTION].shape[-1])
    image_shape = tuple(first_frame[CAM_HIGH].shape)

    dataset = LeRobotDataset.create(
        repo_id=args.output.name,
        root=args.output,
        robot_type=args.robot_type,
        fps=args.fps,
        features=aloha_features(image_shape, state_dim, action_dim),
        use_videos=False,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    metadata: list[dict[str, Any]] = []
    for episode_idx, (source_episode, frames) in enumerate(episodes):
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        rewards = [scalar_float(frame["reward"]) for frame in frames]
        success = any(scalar_bool(frame["success"]) for frame in frames)
        metadata.append(
            {
                "episode_id": episode_idx,
                "source_episode": source_episode,
                "source": args.source_label,
                "task": frames[0]["task"],
                "success": success,
                "return": float(np.sum(rewards)),
                "num_steps": len(frames),
            }
        )

    write_metadata(args.output, metadata)
    print(f"wrote {len(episodes)} episode(s) to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin hdf5 or generic LeRobot data into OpenPI Aloha LeRobot format."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-type", choices=["auto", "hdf5", "lerobot"], default="auto")
    parser.add_argument("--task", default=None)
    parser.add_argument("--source-label", default="expert_success")
    parser.add_argument("--robot-type", default="aloha")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = build_episodes(args)
    write_lerobot_dataset(args, episodes)


if __name__ == "__main__":
    main()
