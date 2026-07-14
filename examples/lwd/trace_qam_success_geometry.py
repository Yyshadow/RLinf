# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Trace beat_block_hammer success geometry for paired SFT/QAM rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-path",
        default="/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_allstats/global_step_11000",
    )
    parser.add_argument(
        "--qam-path",
        default="/data/wam_codebase/RLinf/checkpoints/rlinf_lwd_qam_allstats/"
        "robotwin_beat_block_hammer_lwd_qam_openpi_pi05_allstats_probe_lq005_gc005/"
        "checkpoints/global_step_300",
    )
    parser.add_argument(
        "--data-root",
        default="/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split",
    )
    parser.add_argument(
        "--seeds",
        default="100137506,100112514,100175033",
        help="Comma-separated env seeds. Keep target seed at env_id=2 to match the paired video.",
    )
    parser.add_argument("--target-env-id", type=int, default=2)
    parser.add_argument("--fixed-noise-seed", type=int, default=20260713)
    parser.add_argument("--action-exec-horizon", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        default="outputs/paired_sft_qam_geometry_trace_seed100175033",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def setup_paths() -> Path:
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("REPO_PATH", str(repo))
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples" / "embodiment"))
    os.environ.setdefault("ROBOTWIN_PATH", "/data/wam_codebase/RoboTwin_RLinf")
    os.environ.setdefault("ROBOTWIN_ASSETS_PATH", "/data/wam_codebase/RoboTwin_RLinf")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    robotwin_path = os.environ["ROBOTWIN_PATH"]
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)
    return repo


def patch_success_trace() -> None:
    """Record geometry every time beat_block_hammer.check_success is evaluated."""

    from envs.beat_block_hammer import beat_block_hammer

    if getattr(beat_block_hammer.check_success, "_lwd_trace_patched", False):
        return

    original_check_success = beat_block_hammer.check_success

    def traced_check_success(self):
        hammer_xy = self.hammer.get_functional_point(0, "pose").p[:2].copy()
        block_xy = self.block.get_functional_point(1, "pose").p[:2].copy()
        diff_xy = hammer_xy - block_xy
        abs_xy = np.abs(diff_xy)
        contact = bool(self.check_actors_contact(self.hammer.get_name(), self.block.get_name()))
        success = bool(np.all(abs_xy < np.array([0.02, 0.02])) and contact)
        trace = getattr(self, "_lwd_success_trace", None)
        if trace is not None:
            trace.append(
                {
                    "check_idx": len(trace),
                    "take_action_cnt": int(getattr(self, "take_action_cnt", -1)),
                    "run_steps": int(getattr(self, "run_steps", -1)),
                    "hammer_x": float(hammer_xy[0]),
                    "hammer_y": float(hammer_xy[1]),
                    "block_x": float(block_xy[0]),
                    "block_y": float(block_xy[1]),
                    "dx": float(diff_xy[0]),
                    "dy": float(diff_xy[1]),
                    "abs_dx": float(abs_xy[0]),
                    "abs_dy": float(abs_xy[1]),
                    "xy_linf": float(abs_xy.max()),
                    "xy_l2": float(np.linalg.norm(diff_xy)),
                    "contact": contact,
                    "success": success,
                }
            )
        original_success = bool(original_check_success(self))
        if original_success != success:
            raise RuntimeError("Patched success computation diverged from RoboTwin check_success.")
        return original_success

    traced_check_success._lwd_trace_patched = True
    beat_block_hammer.check_success = traced_check_success


def build_cfg(args: argparse.Namespace, ckpt_path: str, out_dir: Path):
    repo = setup_paths()
    config_dir = str(repo / "evaluations" / "robotwin")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="robotwin_beat_block_hammer_openpi_pi05_eval",
            overrides=[
                f"runner.logger.log_path={out_dir}",
                f"runner.ckpt_path={ckpt_path}/actor/model_state_dict/full_weights.pt",
                f"rollout.model.model_path={ckpt_path}",
                f"rollout.model.openpi.norm_stats_path={args.data_root}/norm_stats.json",
                "rollout.model.openpi.noise_method=flow_ode",
                "rollout.model.openpi.noise_level=0.0",
                f"rollout.model.openpi.fixed_eval_noise_seed={args.fixed_noise_seed}",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    cfg.env.eval.total_num_envs = len(seeds)
    cfg.env.eval.group_size = 1
    cfg.env.eval.rollout_epoch = 1
    cfg.env.eval.auto_reset = False
    cfg.env.eval.ignore_terminations = True
    cfg.env.eval.use_fixed_reset_state_ids = True
    cfg.env.eval.max_episode_steps = int(args.max_episode_steps)
    cfg.env.eval.max_steps_per_rollout_epoch = int(args.max_episode_steps)
    cfg.env.eval.action_exec_horizon = int(args.action_exec_horizon)
    cfg.env.eval.video_cfg.save_video = False
    cfg.env.eval.task_config.render_freq = 0
    cfg.env.eval.task_config.domain_randomization.random_background = False
    cfg.env.eval.task_config.domain_randomization.cluttered_table = False
    cfg.env.eval.task_config.domain_randomization.random_table_height = 0
    cfg.env.eval.task_config.domain_randomization.random_light = False
    return cfg, seeds


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


@torch.no_grad()
def sample_env_action(actor, env_obs: dict[str, Any], device: torch.device) -> torch.Tensor:
    env_obs = dict(env_obs)
    env_obs.setdefault("extra_view_images", None)
    actions, _ = actor.predict_action_batch(
        env_obs=env_obs,
        mode="eval",
        compute_values=False,
    )
    return actions.to(device)


def reset_trace_buffers(env: RoboTwinEnv) -> None:
    for sub_env in env.venv.envs:
        sub_env.task._lwd_success_trace = []


def collect_trace_buffers(env: RoboTwinEnv) -> list[list[dict[str, Any]]]:
    return [list(sub_env.task._lwd_success_trace) for sub_env in env.venv.envs]


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_checks": 0, "success": False}
    success_indices = [idx for idx, item in enumerate(records) if item["success"]]
    contact_indices = [idx for idx, item in enumerate(records) if item["contact"]]
    within_indices = [
        idx
        for idx, item in enumerate(records)
        if item["abs_dx"] < 0.02 and item["abs_dy"] < 0.02
    ]
    near_indices = [idx for idx, item in enumerate(records) if item["xy_linf"] < 0.03]
    min_linf_item = min(records, key=lambda item: item["xy_linf"])
    min_l2_item = min(records, key=lambda item: item["xy_l2"])
    first_success = success_indices[0] if success_indices else None

    def consecutive_count(indices: list[int], start_idx: int | None) -> int:
        if start_idx is None:
            return 0
        index_set = set(indices)
        count = 0
        cursor = start_idx
        while cursor in index_set:
            count += 1
            cursor += 1
        return count

    return {
        "num_checks": len(records),
        "success": bool(success_indices),
        "first_success_check_idx": first_success,
        "success_check_count": len(success_indices),
        "within_2cm_check_count": len(within_indices),
        "near_3cm_check_count": len(near_indices),
        "contact_check_count": len(contact_indices),
        "consecutive_success_from_first": consecutive_count(success_indices, first_success),
        "consecutive_within_2cm_from_first_success": consecutive_count(within_indices, first_success),
        "min_xy_linf_m": min_linf_item["xy_linf"],
        "min_xy_l2_m": min_l2_item["xy_l2"],
        "min_linf_check_idx": min_linf_item["check_idx"],
        "min_l2_check_idx": min_l2_item["check_idx"],
        "final_xy_linf_m": records[-1]["xy_linf"],
        "final_xy_l2_m": records[-1]["xy_l2"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_model(
    name: str,
    ckpt_path: str,
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    model_out_dir = out_root / name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    cfg, seeds = build_cfg(args, ckpt_path, model_out_dir)
    actor = get_openpi_model(cfg.rollout.model)
    state_dict_path = Path(ckpt_path) / "actor" / "model_state_dict" / "full_weights.pt"
    state_dict = torch.load(state_dict_path, map_location="cpu")
    actor.load_state_dict(state_dict)
    del state_dict
    actor = actor.to(device).eval()
    env = RoboTwinEnv(
        cfg.env.eval,
        num_envs=len(seeds),
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    try:
        obs, _ = env.reset(env_seeds=seeds)
        obs.setdefault("extra_view_images", None)
        reset_trace_buffers(env)

        n_chunks = int(args.max_episode_steps // args.action_exec_horizon)
        last_infos = {}
        for _ in range(n_chunks):
            env_action = sample_env_action(actor, obs, device)
            step_action = env_action[:, : args.action_exec_horizon, :]
            obs, _reward, _terminated, _truncated, last_infos = env.step(
                step_action,
                auto_reset=False,
            )
            obs.setdefault("extra_view_images", None)

        traces = collect_trace_buffers(env)
        summaries = []
        for env_id, trace in enumerate(traces):
            rows = [
                {"model": name, "env_id": env_id, "seed": seeds[env_id], **item}
                for item in trace
            ]
            write_csv(model_out_dir / f"env{env_id}_seed{seeds[env_id]}_trace.csv", rows)
            summaries.append(
                {
                    "model": name,
                    "env_id": env_id,
                    "seed": seeds[env_id],
                    **summarize_trace(trace),
                }
            )

        success_once = (
            last_infos.get("episode", {})
            .get("success_once", torch.zeros(len(seeds), dtype=torch.bool))
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
            .tolist()
        )
        result = {
            "model": name,
            "checkpoint": ckpt_path,
            "seeds": seeds,
            "target_env_id": args.target_env_id,
            "success_once": success_once,
            "summaries": summaries,
        }
        (model_out_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result
    finally:
        env.close()
        del actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    setup_paths()
    patch_success_trace()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = [
        run_model("sft", args.sft_path, args, out_root, device),
        run_model("qam", args.qam_path, args, out_root, device),
    ]
    target = {
        result["model"]: result["summaries"][args.target_env_id]
        for result in results
    }
    combined = {
        "settings": vars(args),
        "target": target,
        "all_results": results,
    }
    (out_root / "paired_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
