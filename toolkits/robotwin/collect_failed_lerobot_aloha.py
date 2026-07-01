#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

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


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def resolve_robotwin_args(robotwin_root: Path, task: str, task_config: str) -> dict[str, Any]:
    sys.path.insert(0, str(robotwin_root))
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    cfg = load_yaml(Path(CONFIGS_PATH) / f"{task_config}.yml")
    embodiment_cfg = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

    def get_embodiment_file(name: str) -> str:
        robot_file = embodiment_cfg[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment file for {name}")
        robot_path = Path(robot_file)
        if not robot_path.is_absolute():
            robot_path = robotwin_root / robot_path
        return str(robot_path)

    def get_embodiment_config(robot_file: str) -> dict[str, Any]:
        return load_yaml(Path(robot_file) / "config.yml")

    embodiment_type = cfg["embodiment"]
    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
        cfg["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
        cfg["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must contain either 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])
    cfg["task_name"] = task
    cfg["task_config"] = task_config
    return cfg


def make_task(robotwin_root: Path, task: str):
    sys.path.insert(0, str(robotwin_root))
    module = importlib.import_module(f"envs.{task}")
    return getattr(module, task)()


def get_qpos(task_env) -> np.ndarray:
    left = task_env.robot.get_left_arm_jointState()
    right = task_env.robot.get_right_arm_jointState()
    return np.asarray(left + right, dtype=np.float32)


def get_images_and_state(task_env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = task_env.get_obs()
    images = obs["observation"]
    head = to_hwc_uint8(images["head_camera"]["rgb"])
    left = to_hwc_uint8(images["left_camera"]["rgb"])
    right = to_hwc_uint8(images["right_camera"]["rgb"])
    state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    return head, left, right, state


def make_action(current: np.ndarray, rng: np.random.Generator, mode: str, scale: float) -> np.ndarray:
    action = current.copy()
    if mode == "idle":
        return action
    if mode == "random_walk":
        joint_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        action[joint_idx] += rng.normal(0.0, scale, size=len(joint_idx)).astype(np.float32)
        action[[6, 13]] = np.clip(action[[6, 13]], 0.0, 1.0)
        return action
    if mode == "bad_gripper":
        action[[6, 13]] = 1.0
        return action
    raise ValueError(f"Unsupported failure mode: {mode}")


def rollout_failed_episode(
    task_env,
    *,
    rng: np.random.Generator,
    mode: str,
    steps: int,
    chunk_size: int,
    perturb_scale: float,
    task_name: str,
) -> tuple[list[dict[str, Any]], bool, float]:
    frames: list[dict[str, Any]] = []
    episode_return = 0.0

    for _ in range(steps):
        head, left, right, state = get_images_and_state(task_env)
        actions = []
        current = state.copy()
        for _chunk_idx in range(chunk_size):
            current = make_action(current, rng, mode, perturb_scale)
            actions.append(current.copy())

        chunk_actions = np.asarray(actions, dtype=np.float32)
        reward, termination, truncation, info = task_env.gen_sparse_reward_data(chunk_actions, action_type="qpos")
        step_success = bool(info.get("success", False))
        done = bool(np.asarray(termination).any() or np.asarray(truncation).any())
        episode_return += float(np.asarray(reward).reshape(-1)[0])

        frames.append(
            {
                CAM_HIGH: head,
                CAM_LEFT: left,
                CAM_RIGHT: right,
                STATE: state.astype(np.float32),
                ACTION: chunk_actions[0].astype(np.float32),
                "reward": np.asarray(reward, dtype=np.float32).reshape(1),
                "done": np.array([done], dtype=bool),
                "success": np.array([step_success], dtype=bool),
                "task": task_name,
            }
        )
        if done:
            break

    if frames:
        frames[-1]["done"] = np.array([True], dtype=bool)
    return frames, any(scalar_bool(frame["success"]) for frame in frames), episode_return


def write_lerobot_dataset(
    *,
    output: Path,
    episodes: list[list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    robot_type: str,
    fps: int,
) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise RuntimeError("No failed episodes collected")

    first = episodes[0][0]
    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        robot_type=robot_type,
        fps=fps,
        features=aloha_features(tuple(first[CAM_HIGH].shape), int(first[STATE].shape[-1]), int(first[ACTION].shape[-1])),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    for frames in episodes:
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

    write_metadata(output, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect failed RoboTwin rollouts into OpenPI Aloha LeRobot format.")
    parser.add_argument("--robotwin-root", type=Path, default=Path("/data/wam_codebase/RoboTwin_RLinf"))
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_30")
    parser.add_argument("--output", type=Path, default=Path("/data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer_failed_20ep"))
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--failure-mode", choices=["idle", "random_walk", "bad_gripper"], default="random_walk")
    parser.add_argument("--perturb-scale", type=float, default=0.08)
    parser.add_argument("--planner-backend", choices=["curobo", "mplib"], default="mplib")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-type", default="aloha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("ASSETS_PATH", str(args.robotwin_root))
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

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

    episodes: list[list[dict[str, Any]]] = []
    metadata: list[dict[str, Any]] = []

    for attempt in range(args.max_attempts):
        if len(episodes) >= args.num_episodes:
            break
        seed = args.seed_start + attempt
        task_env = make_task(args.robotwin_root, args.task)
        try:
            task_env.setup_demo(now_ep_num=len(episodes), seed=seed, **cfg)
            rng = np.random.default_rng(seed)
            frames, success, episode_return = rollout_failed_episode(
                task_env,
                rng=rng,
                mode=args.failure_mode,
                steps=args.steps,
                chunk_size=args.chunk_size,
                perturb_scale=args.perturb_scale,
                task_name=args.task,
            )
            if frames and not success:
                episode_id = len(episodes)
                episodes.append(frames)
                metadata.append(
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "source": "policy_failure",
                        "perturb_type": args.failure_mode,
                        "perturb_scale": args.perturb_scale,
                        "policy_ckpt": "",
                        "success": False,
                        "return": float(episode_return),
                        "num_steps": len(frames),
                        "failure_stage": "rollout_no_success",
                        "task": args.task,
                        "robotwin_source": str(args.robotwin_root),
                        "planner": args.planner_backend,
                    }
                )
                print(f"[{len(episodes)}/{args.num_episodes}] saved failed seed={seed} steps={len(frames)}")
            else:
                print(f"[skip] seed={seed} success={success} frames={len(frames)}")
        except Exception as exc:
            print(f"[skip] seed={seed} error={type(exc).__name__}: {exc}")
        finally:
            try:
                task_env.close_env(clear_cache=(attempt % 5 == 4))
            except Exception:
                pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_lerobot_dataset(output=args.output, episodes=episodes, metadata=metadata, robot_type=args.robot_type, fps=args.fps)
    print(f"wrote {len(episodes)} failed episode(s) to {args.output}")


if __name__ == "__main__":
    main()
