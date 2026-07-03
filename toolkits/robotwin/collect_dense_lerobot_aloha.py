#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkits.robotwin.collect_failed_lerobot_aloha import (
    get_images_and_state,
    make_task,
    resolve_robotwin_args,
)
from toolkits.robotwin.prepare_lerobot_aloha import (
    ACTION,
    CAM_HIGH,
    CAM_LEFT,
    CAM_RIGHT,
    STATE,
    aloha_features,
    scalar_bool,
    scalar_float,
    to_hwc_uint8,
    write_metadata,
)


JOINT_IDX = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_IDX = np.asarray([6, 13])


def use_fast_sapien_render() -> None:
    import sapien

    set_camera_shader_dir = sapien.render.set_camera_shader_dir
    sapien.render.set_camera_shader_dir = lambda _shader_dir: set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel = lambda *_args, **_kwargs: None
    sapien.render.set_ray_tracing_path_depth = lambda *_args, **_kwargs: None
    sapien.render.set_ray_tracing_denoiser = lambda *_args, **_kwargs: None


def sorted_episode_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/episode_*.parquet"))


def load_robotwin_metadata(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "robotwin_episode_metadata.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_source_actions(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=[ACTION])
    return np.asarray(table.column(ACTION).to_pylist(), dtype=np.float32)


def load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def pkl_cache_to_frames(cache_dir: Path, task: str, success: bool) -> list[dict[str, Any]]:
    pkl_paths = sorted(cache_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    frames: list[dict[str, Any]] = []
    if len(pkl_paths) < 2:
        return frames

    current_rows = [load_pkl(path) for path in pkl_paths]
    for idx in range(len(current_rows) - 1):
        row = current_rows[idx]
        next_row = current_rows[idx + 1]
        images = row["observation"]
        is_final = idx == len(current_rows) - 2
        frame_success = success and is_final
        frames.append(
            {
                CAM_HIGH: to_hwc_uint8(images["head_camera"]["rgb"]),
                CAM_LEFT: to_hwc_uint8(images["left_camera"]["rgb"]),
                CAM_RIGHT: to_hwc_uint8(images["right_camera"]["rgb"]),
                STATE: np.asarray(row["joint_action"]["vector"], dtype=np.float32),
                ACTION: np.asarray(next_row["joint_action"]["vector"], dtype=np.float32),
                "reward": np.array([1.0 if frame_success else 0.0], dtype=np.float32),
                "done": np.array([is_final], dtype=bool),
                "success": np.array([frame_success], dtype=bool),
                "task": task,
            }
        )
    return frames


def create_lerobot_dataset(output: Path, first_frame: dict[str, Any], robot_type: str, fps: int):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        robot_type=robot_type,
        fps=fps,
        features=aloha_features(
            tuple(first_frame[CAM_HIGH].shape),
            int(first_frame[STATE].shape[-1]),
            int(first_frame[ACTION].shape[-1]),
        ),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=int(os.environ.get("LEROBOT_IMAGE_WRITER_THREADS", "2")),
    )


def append_episode(dataset, frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        dataset.add_frame(frame)
    dataset.save_episode()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)


def collect_success(args: argparse.Namespace) -> None:
    prepare_output(args.output, args.overwrite)
    cache_root = args.output.parent / f".{args.output.name}_robotwin_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cfg = resolve_robotwin_args(args.robotwin_root, args.task, args.task_config)
    cfg.update(
        {
            "save_path": str(cache_root),
            "render_freq": 0,
            "save_data": True,
            "collect_data": False,
            "need_plan": True,
            "eval_mode": False,
            "eval_video_log": False,
            "save_freq": args.save_freq,
            "planner_backend": args.planner_backend,
        }
    )

    dataset = None
    metadata: list[dict[str, Any]] = []
    failures = 0

    for attempt in range(args.max_attempts):
        if len(metadata) >= args.num_episodes:
            break

        episode_id = len(metadata)
        seed = args.seed_start + attempt
        task_env = make_task(args.robotwin_root, args.task)
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed, **cfg)
            task_env.play_once()
            success = bool(task_env.plan_success and task_env.check_success())
            cache_dir = Path(task_env.folder_path["cache"]) if hasattr(task_env, "folder_path") else None
            frames = pkl_cache_to_frames(cache_dir, args.task, success=True) if success and cache_dir else []
            if not success or not frames:
                failures += 1
                print(f"[skip] seed={seed} success={success} frames={len(frames)}")
                continue

            if dataset is None:
                dataset = create_lerobot_dataset(args.output, frames[0], args.robot_type, args.fps)
            append_episode(dataset, frames)

            rewards = [scalar_float(frame["reward"]) for frame in frames]
            metadata.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "source": "planner_success",
                    "perturb_type": "",
                    "perturb_scale": 0.0,
                    "policy_ckpt": "",
                    "success": True,
                    "return": float(np.sum(rewards)),
                    "num_steps": len(frames),
                    "failure_stage": "",
                    "task": args.task,
                    "robotwin_source": str(args.robotwin_root),
                    "planner": args.planner_backend,
                }
            )
            write_metadata(args.output, metadata)
            print(f"[{len(metadata)}/{args.num_episodes}] success seed={seed} steps={len(frames)}")
        except Exception as exc:
            failures += 1
            print(f"[skip] seed={seed} error={type(exc).__name__}: {exc}")
        finally:
            try:
                if hasattr(task_env, "folder_path"):
                    task_env.remove_data_cache()
            except Exception:
                pass
            try:
                task_env.close_env(clear_cache=(attempt % 5 == 4))
            except Exception:
                pass

    shutil.rmtree(cache_root, ignore_errors=True)
    if dataset is None or len(metadata) < args.num_episodes:
        raise RuntimeError(f"Collected {len(metadata)} / {args.num_episodes} success episodes, failures={failures}")
    print(f"LeRobot success dataset: {args.output}")


def active_arm(actions: np.ndarray, start: int) -> str:
    left_motion = np.linalg.norm(np.diff(actions[start:, :6], axis=0), axis=1).sum()
    right_motion = np.linalg.norm(np.diff(actions[start:, 7:13], axis=0), axis=1).sum()
    return "left" if left_motion >= right_motion else "right"


def perturb_actions(
    actions: np.ndarray,
    *,
    rng: np.random.Generator,
    kind: str,
    mode: str,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    perturbed = actions.copy()
    start_fraction = rng.uniform(0.45, 0.72) if kind == "nearmiss" else rng.uniform(0.18, 0.55)
    start = min(len(perturbed) - 1, max(1, int(len(perturbed) * start_fraction)))
    ramp_len = max(1, int(len(perturbed) * (0.12 if kind == "nearmiss" else 0.2)))
    arm = active_arm(perturbed, start)
    arm_joint_idx = np.arange(0, 6) if arm == "left" else np.arange(7, 13)
    gripper_idx = 6 if arm == "left" else 13

    if mode == "joint_bias":
        bias = rng.normal(0.0, scale, size=len(arm_joint_idx)).astype(np.float32)
        for idx in range(start, len(perturbed)):
            ramp = min(1.0, (idx - start + 1) / ramp_len)
            perturbed[idx, arm_joint_idx] += ramp * bias
    elif mode == "smooth_noise":
        noise = rng.normal(0.0, scale, size=(len(perturbed) - start, len(arm_joint_idx))).astype(np.float32)
        for idx in range(1, len(noise)):
            noise[idx] = 0.85 * noise[idx - 1] + 0.15 * noise[idx]
        perturbed[start:, arm_joint_idx] += noise
    elif mode == "action_lag":
        lag = int(rng.integers(2, 6 if kind == "nearmiss" else 12))
        for idx in range(start, len(perturbed)):
            perturbed[idx, arm_joint_idx] = actions[max(0, idx - lag), arm_joint_idx]
    elif mode == "early_release":
        release = float(rng.uniform(0.55, 0.85))
        end = min(len(perturbed), start + max(ramp_len, len(perturbed) // 4))
        perturbed[start:end, gripper_idx] = np.maximum(perturbed[start:end, gripper_idx], release)
    elif mode == "gripper_delay":
        lag = int(rng.integers(3, 8 if kind == "nearmiss" else 16))
        for idx in range(start, len(perturbed)):
            perturbed[idx, gripper_idx] = actions[max(0, idx - lag), gripper_idx]
    else:
        raise ValueError(f"Unsupported perturb mode: {mode}")

    perturbed[:, GRIPPER_IDX] = np.clip(perturbed[:, GRIPPER_IDX], 0.0, 1.0)
    return perturbed.astype(np.float32), {
        "perturb_mode": mode,
        "replay_start_fraction": float(start_fraction),
        "active_arm": arm,
    }


def rollout_replay_episode(
    task_env,
    *,
    actions: np.ndarray,
    task: str,
) -> tuple[list[dict[str, Any]], bool, float]:
    frames: list[dict[str, Any]] = []
    success = False
    episode_return = 0.0

    for idx, action in enumerate(actions):
        head, left, right, state = get_images_and_state(task_env)
        task_env.take_action(action, action_type="qpos")
        step_success = bool(task_env.check_success())
        success = success or step_success
        is_final = idx == len(actions) - 1
        reward = 1.0 if step_success else 0.0
        episode_return += reward
        frames.append(
            {
                CAM_HIGH: head,
                CAM_LEFT: left,
                CAM_RIGHT: right,
                STATE: state.astype(np.float32),
                ACTION: action.astype(np.float32),
                "reward": np.array([reward], dtype=np.float32),
                "done": np.array([is_final], dtype=bool),
                "success": np.array([step_success], dtype=bool),
                "task": task,
            }
        )
    if frames:
        frames[-1]["done"] = np.array([True], dtype=bool)
    return frames, success, episode_return


def collect_replay_failures(args: argparse.Namespace) -> None:
    prepare_output(args.output, args.overwrite)
    source_metadata = load_robotwin_metadata(args.expert_root)
    source_parquets = sorted_episode_parquets(args.expert_root)
    if not source_metadata or not source_parquets:
        raise RuntimeError(f"No source episodes found in {args.expert_root}")

    cfg = resolve_robotwin_args(args.robotwin_root, args.task, args.task_config)
    cfg.update(
        {
            "render_freq": 0,
            "save_data": False,
            "collect_data": False,
            "need_plan": False,
            "eval_mode": True,
            "planner_backend": args.planner_backend,
        }
    )

    modes = args.perturb_modes.split(",")
    dataset = None
    metadata: list[dict[str, Any]] = []
    attempts = 0

    while len(metadata) < args.num_episodes and attempts < args.max_attempts:
        source_idx = attempts % min(len(source_parquets), len(source_metadata))
        source_row = source_metadata[source_idx]
        seed = int(source_row["seed"])
        rng = np.random.default_rng(args.seed_start + attempts)
        source_actions = load_source_actions(source_parquets[source_idx])
        mode = modes[attempts % len(modes)]
        actions, extra = perturb_actions(
            source_actions,
            rng=rng,
            kind=args.failure_kind,
            mode=mode,
            scale=args.perturb_scale,
        )

        task_env = make_task(args.robotwin_root, args.task)
        try:
            task_env.setup_demo(now_ep_num=len(metadata), seed=seed, **cfg)
            frames, success, episode_return = rollout_replay_episode(task_env, actions=actions, task=args.task)
            if not frames or success:
                print(
                    f"[skip] source_episode={source_idx} seed={seed} "
                    f"success={success} frames={len(frames)} mode={mode}"
                )
                attempts += 1
                continue

            if dataset is None:
                dataset = create_lerobot_dataset(args.output, frames[0], args.robot_type, args.fps)
            append_episode(dataset, frames)

            episode_id = len(metadata)
            metadata.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "source": f"expert_replay_{args.failure_kind}",
                    "source_episode": source_idx,
                    "perturb_type": mode,
                    "perturb_scale": args.perturb_scale,
                    "policy_ckpt": "",
                    "success": False,
                    "return": float(episode_return),
                    "num_steps": len(frames),
                    "failure_stage": "dense_expert_replay_perturb",
                    "task": args.task,
                    "robotwin_source": str(args.robotwin_root),
                    "expert_source": str(args.expert_root),
                    "planner": args.planner_backend,
                    **extra,
                }
            )
            write_metadata(args.output, metadata)
            print(
                f"[{len(metadata)}/{args.num_episodes}] {args.failure_kind} "
                f"source_episode={source_idx} seed={seed} mode={mode} steps={len(frames)}"
            )
        except Exception as exc:
            print(f"[skip] source_episode={source_idx} seed={seed} error={type(exc).__name__}: {exc}")
        finally:
            attempts += 1
            try:
                task_env.close_env(clear_cache=(attempts % 5 == 0))
            except Exception:
                pass

    if dataset is None or len(metadata) < args.num_episodes:
        raise RuntimeError(f"Collected {len(metadata)} / {args.num_episodes} {args.failure_kind} episodes")
    print(f"LeRobot {args.failure_kind} dataset: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dense RoboTwin episodes into OpenPI Aloha LeRobot format.")
    parser.add_argument("--mode", choices=["success", "replay_failure"], required=True)
    parser.add_argument("--robotwin-root", type=Path, default=Path("/data/wam_codebase/RoboTwin_RLinf"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--planner-backend", choices=["curobo", "mplib"], default="mplib")
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-type", default="aloha")
    parser.add_argument("--fast-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expert-root", type=Path)
    parser.add_argument("--failure-kind", choices=["failed", "nearmiss"], default="failed")
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument(
        "--perturb-modes",
        default="joint_bias,smooth_noise,action_lag,early_release,gripper_delay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("ASSETS_PATH", str(args.robotwin_root))
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    sys.path.insert(0, str(args.robotwin_root))
    if args.fast_render:
        use_fast_sapien_render()

    if args.mode == "success":
        collect_success(args)
    else:
        if args.expert_root is None:
            raise ValueError("--expert-root is required for replay_failure mode")
        collect_replay_failures(args)


if __name__ == "__main__":
    main()
