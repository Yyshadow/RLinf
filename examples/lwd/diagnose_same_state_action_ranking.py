# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Diagnose same-state action ranking for the LWD critic.

This pilot intentionally avoids pretending that qpos actions are Cartesian
offsets.  It restores a state by replaying an identical SFT prefix from the
same RoboTwin seed, validates that restoration, then compares critic scores
against real rollouts for SFT/QAM/mirror/gradient candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.lwd_critic import get_model as get_lwd_critic_model
from rlinf.models.embodiment.openpi import get_model as get_openpi_model
from rlinf.models.embodiment.value_model.processing import ValueProcessor

from diagnose_critic_gradient_edit import (
    LWDChunkDataCollator,
    _load_norm_stats,
    critic_observation_from_env,
    move_to_device,
)


DEFAULT_SEEDS = "100137506,100112514,100175033,100187555"
ACTION_DIM = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "pilot", "full"], default="pilot")
    parser.add_argument("--num-states", type=int, default=None)
    parser.add_argument("--query-index", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--snapshot-repeats", type=int, default=3)
    parser.add_argument("--fixed-noise-seed", type=int, default=20260713)
    parser.add_argument("--action-exec-horizon", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="outputs/lwd_same_state_action_ranking_pilot")
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
        "--critic-path",
        default="/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/"
        "global_step_8000/actor",
    )
    parser.add_argument(
        "--data-root",
        default="/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split",
    )
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
    from envs.beat_block_hammer import beat_block_hammer

    if getattr(beat_block_hammer.check_success, "_same_state_trace_patched", False):
        return

    original_check_success = beat_block_hammer.check_success

    def traced_check_success(self):
        hammer_pose = self.hammer.get_functional_point(0, "pose")
        block_pose = self.block.get_functional_point(1, "pose")
        hammer_xy = hammer_pose.p[:2].copy()
        block_xy = block_pose.p[:2].copy()
        diff_xy = hammer_xy - block_xy
        abs_xy = np.abs(diff_xy)
        contact = bool(self.check_actors_contact(self.hammer.get_name(), self.block.get_name()))
        success = bool(np.all(abs_xy < np.array([0.02, 0.02])) and contact)
        trace = getattr(self, "_same_state_success_trace", None)
        if trace is not None:
            trace.append(
                {
                    "check_idx": len(trace),
                    "take_action_cnt": int(getattr(self, "take_action_cnt", -1)),
                    "hammer_x": float(hammer_pose.p[0]),
                    "hammer_y": float(hammer_pose.p[1]),
                    "hammer_z": float(hammer_pose.p[2]),
                    "block_x": float(block_pose.p[0]),
                    "block_y": float(block_pose.p[1]),
                    "block_z": float(block_pose.p[2]),
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

    traced_check_success._same_state_trace_patched = True
    beat_block_hammer.check_success = traced_check_success


def build_eval_cfg(args: argparse.Namespace, model_path: str, out_dir: Path):
    repo = setup_paths()
    config_dir = str(repo / "evaluations" / "robotwin")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="robotwin_beat_block_hammer_openpi_pi05_eval",
            overrides=[
                f"runner.logger.log_path={out_dir}",
                f"runner.ckpt_path={model_path}/actor/model_state_dict/full_weights.pt",
                f"rollout.model.model_path={model_path}",
                f"rollout.model.openpi.norm_stats_path={args.data_root}/norm_stats.json",
                "rollout.model.openpi.noise_method=flow_ode",
                "rollout.model.openpi.noise_level=0.0",
                f"rollout.model.openpi.fixed_eval_noise_seed={args.fixed_noise_seed}",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.env.eval.total_num_envs = 1
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
    return cfg


def load_actor(cfg, checkpoint_path: str, device: torch.device):
    actor = get_openpi_model(cfg.rollout.model)
    state_path = Path(checkpoint_path) / "actor" / "model_state_dict" / "full_weights.pt"
    state_dict = torch.load(state_path, map_location="cpu")
    actor.load_state_dict(state_dict)
    del state_dict
    return actor.to(device).eval()


def reset_actor_noise(actor, seed: int) -> None:
    object.__setattr__(actor.config, "fixed_eval_noise_seed", int(seed))
    actor._fixed_eval_noise_generators = {}


@torch.no_grad()
def sample_policy_action(actor, env_obs: dict[str, Any], noise_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    reset_actor_noise(actor, noise_seed)
    env_obs = dict(env_obs)
    env_obs.setdefault("extra_view_images", None)
    actions, result = actor.predict_action_batch(
        env_obs=env_obs,
        mode="eval",
        compute_values=False,
    )
    model_flat = result["forward_inputs"]["model_action"]
    horizon = int(actor.config.action_chunk)
    model_action = model_flat.reshape(model_flat.shape[0], horizon, -1).detach()
    return actions.detach(), model_action


@torch.no_grad()
def model_to_env_action(actor, env_obs: dict[str, Any], model_action: torch.Tensor) -> torch.Tensor:
    env_obs = dict(env_obs)
    env_obs.setdefault("extra_view_images", None)
    processed = actor.obs_processor(env_obs)
    processed = actor.input_transform(processed, transpose=False)
    processed = actor.precision_processor(processed)
    from openpi.models import model as _model

    observation = _model.Observation.from_dict(processed)
    return actor.output_transform(
        {"actions": model_action.detach(), "state": observation.state}
    )["actions"].to(model_action.device)


def make_env(cfg) -> RoboTwinEnv:
    return RoboTwinEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )


def reset_trace(env: RoboTwinEnv) -> None:
    env.venv.envs[0].task._same_state_success_trace = []


def get_trace(env: RoboTwinEnv) -> list[dict[str, Any]]:
    return list(env.venv.envs[0].task._same_state_success_trace)


def task_signature(env: RoboTwinEnv) -> dict[str, Any]:
    task = env.venv.envs[0].task
    hammer_pose = task.hammer.get_pose()
    block_pose = task.block.get_pose()
    left_state = task.robot.get_left_arm_jointState()
    right_state = task.robot.get_right_arm_jointState()
    return {
        "hammer_p": np.asarray(hammer_pose.p, dtype=np.float64).tolist(),
        "hammer_q": np.asarray(hammer_pose.q, dtype=np.float64).tolist(),
        "block_p": np.asarray(block_pose.p, dtype=np.float64).tolist(),
        "block_q": np.asarray(block_pose.q, dtype=np.float64).tolist(),
        "left_joint": np.asarray(left_state, dtype=np.float64).tolist(),
        "right_joint": np.asarray(right_state, dtype=np.float64).tolist(),
    }


def signature_diffs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    diffs = {}
    for key in ["hammer_p", "hammer_q", "block_p", "block_q", "left_joint", "right_joint"]:
        av = np.asarray(a[key], dtype=np.float64)
        bv = np.asarray(b[key], dtype=np.float64)
        diffs[f"{key}_max_abs"] = float(np.max(np.abs(av - bv)))
    return diffs


def obs_hash(obs: dict[str, Any]) -> str:
    hasher = hashlib.sha1()
    for key in ["main_images", "states"]:
        value = obs[key]
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def replay_prefix(
    env: RoboTwinEnv,
    seed: int,
    prefix_actions: list[torch.Tensor],
) -> dict[str, Any]:
    obs, _ = env.reset(env_seeds=[int(seed)])
    obs.setdefault("extra_view_images", None)
    for action in prefix_actions:
        obs, _reward, _terminated, _truncated, _infos = env.step(
            action[:, : env.cfg.action_exec_horizon, :],
            auto_reset=False,
        )
        obs.setdefault("extra_view_images", None)
    return obs


def collect_canonical_state(
    env: RoboTwinEnv,
    actor,
    seed: int,
    query_index: int,
    base_noise_seed: int,
) -> tuple[dict[str, Any], list[torch.Tensor], dict[str, Any]]:
    obs, _ = env.reset(env_seeds=[int(seed)])
    obs.setdefault("extra_view_images", None)
    prefix_actions = []
    for query_idx in range(query_index):
        env_action, _model_action = sample_policy_action(
            actor,
            obs,
            base_noise_seed + query_idx,
        )
        prefix_actions.append(env_action.detach().clone())
        obs, _reward, _terminated, _truncated, _infos = env.step(
            env_action[:, : env.cfg.action_exec_horizon, :],
            auto_reset=False,
        )
        obs.setdefault("extra_view_images", None)
    return obs, prefix_actions, task_signature(env)


def validate_restore(
    cfg,
    actor,
    seed: int,
    prefix_actions: list[torch.Tensor],
    reference_signature: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    results = []
    env = make_env(cfg)
    try:
        for _ in range(repeats):
            obs = replay_prefix(env, seed, prefix_actions)
            sig = task_signature(env)
            diffs = signature_diffs(reference_signature, sig)
            results.append({"obs_hash": obs_hash(obs), **diffs})
    finally:
        env.close()
    max_diffs = {
        key: max(item[key] for item in results)
        for key in results[0]
        if key != "obs_hash"
    }
    return {
        "repeats": repeats,
        "obs_hashes": [item["obs_hash"] for item in results],
        "max_diffs": max_diffs,
        "passed": bool(
            max_diffs["hammer_p_max_abs"] < 1e-4
            and max_diffs["block_p_max_abs"] < 1e-6
            and max_diffs["left_joint_max_abs"] < 1e-4
            and max_diffs["right_joint_max_abs"] < 1e-4
        ),
    }


def critic_scores(
    critic,
    critic_obs: dict[str, Any],
    model_action: torch.Tensor,
    env_action_dim: int,
    q_mode: str,
) -> dict[str, float]:
    action = model_action[..., :env_action_dim].detach().float()
    with torch.no_grad():
        out = critic(observation=critic_obs, action_chunk=action)
    q_values = out.q_values.float().reshape(-1)
    q_min = float(q_values.min().item())
    q_mean = float(q_values.mean().item())
    return {
        "q1": float(q_values[0].item()),
        "q2": float(q_values[1].item()) if q_values.numel() > 1 else float("nan"),
        "q_min": q_min,
        "q_mean": q_mean,
        "q_used": q_min if q_mode == "min" else q_mean,
    }


def critic_gradient(
    critic,
    critic_obs: dict[str, Any],
    model_action: torch.Tensor,
    env_action_dim: int,
    q_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    action = model_action[..., :env_action_dim].detach().float().requires_grad_(True)
    out = critic(observation=critic_obs, action_chunk=action)
    q_values = out.q_values.float()
    q_used = q_values.min(dim=-1).values if q_mode == "min" else q_values.mean(dim=-1)
    grad = torch.autograd.grad(q_used.sum(), action)[0]
    return grad.detach(), {
        "base_q1": float(q_values[0, 0].item()),
        "base_q2": float(q_values[0, 1].item()),
        "grad_norm": float(grad.flatten(1).norm(dim=-1)[0].item()),
    }


def build_candidates(
    args: argparse.Namespace,
    sft_actor,
    qam_actor,
    critic,
    critic_obs: dict[str, Any],
    obs: dict[str, Any],
    state_idx: int,
    q_mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    current_noise_seed = args.fixed_noise_seed + 10000 + state_idx
    sft_env, sft_model = sample_policy_action(sft_actor, obs, current_noise_seed)
    qam_env, qam_model = sample_policy_action(qam_actor, obs, current_noise_seed)

    candidates = {
        "sft": {
            "model_action": sft_model,
            "env_action": sft_env,
            "kind": "policy",
        },
        "qam_same_noise": {
            "model_action": qam_model,
            "env_action": qam_env,
            "kind": "policy",
        },
    }

    mirror_model = (2.0 * sft_model - qam_model).clamp(-1.0, 1.0)
    candidates["qam_mirror"] = {
        "model_action": mirror_model,
        "env_action": model_to_env_action(sft_actor, obs, mirror_model),
        "kind": "mirror",
    }

    grad, grad_info = critic_gradient(critic, critic_obs, sft_model, ACTION_DIM, q_mode)
    target_delta = (qam_model[..., :ACTION_DIM] - sft_model[..., :ACTION_DIM]).flatten(1)
    target_norm = float(target_delta.norm(dim=-1)[0].item())
    if target_norm < 1e-5:
        target_norm = 1e-3
    grad_norm = grad.flatten(1).norm(dim=-1).clamp_min(1e-6)
    grad_step = grad / grad_norm.view(-1, 1, 1) * target_norm

    rand = torch.randn_like(grad)
    rand = rand / rand.flatten(1).norm(dim=-1).clamp_min(1e-6).view(-1, 1, 1) * target_norm

    for name, step in [
        ("grad_plus", grad_step),
        ("grad_minus", -grad_step),
        ("random_plus", rand),
        ("random_minus", -rand),
    ]:
        edited = sft_model.clone()
        edited[..., :ACTION_DIM] = (edited[..., :ACTION_DIM] + step).clamp(-1.0, 1.0)
        candidates[name] = {
            "model_action": edited,
            "env_action": model_to_env_action(sft_actor, obs, edited),
            "kind": "gradient" if name.startswith("grad") else "random",
        }

    meta = {
        **grad_info,
        "current_noise_seed": current_noise_seed,
        "gradient_target_norm_source": "qam_delta_model_action_env_dims",
        "gradient_target_norm": target_norm,
        "skipped_candidates": {
            "x_plus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
            "x_minus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
            "y_plus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
            "y_minus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
            "z_plus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
            "z_minus_2mm": "Current executable action is qpos; no safe Cartesian IK/controller conversion API is wired here.",
        },
    }
    return candidates, meta


def summarize_trace(trace: list[dict[str, Any]], initial_block_p: list[float], final_sig: dict[str, Any]) -> dict[str, Any]:
    if not trace:
        return {
            "success": False,
            "contact_count": 0,
            "min_xy_linf_cm": None,
            "min_xy_l2_cm": None,
            "margin_cm": None,
            "termination_reason": "no_trace",
        }
    success = any(item["success"] for item in trace)
    contact_rows = [item for item in trace if item["contact"]]
    rows_for_min = contact_rows if contact_rows else trace
    min_linf = min(item["xy_linf"] for item in rows_for_min)
    min_l2 = min(item["xy_l2"] for item in rows_for_min)
    final_block_p = np.asarray(final_sig["block_p"], dtype=np.float64)
    initial_block = np.asarray(initial_block_p, dtype=np.float64)
    return {
        "success": bool(success),
        "termination_reason": "success" if success else "timeout",
        "contact_count": int(sum(item["contact"] for item in trace)),
        "first_contact_check": next((item["check_idx"] for item in trace if item["contact"]), None),
        "first_success_check": next((item["check_idx"] for item in trace if item["success"]), None),
        "min_xy_linf_cm": float(min_linf * 100.0),
        "min_xy_l2_cm": float(min_l2 * 100.0),
        "margin_cm": float(2.0 - min_linf * 100.0) if contact_rows else None,
        "block_displacement_cm": float(np.linalg.norm(final_block_p - initial_block) * 100.0),
        "grasp_success": None,
        "hammer_dropped": None,
    }


def run_candidate_rollout(
    cfg,
    sft_actor,
    seed: int,
    state_idx: int,
    query_index: int,
    prefix_actions: list[torch.Tensor],
    candidate_action: torch.Tensor,
    initial_signature: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = make_env(cfg)
    try:
        obs = replay_prefix(env, seed, prefix_actions)
        reset_trace(env)
        obs, _reward, _terminated, _truncated, infos = env.step(
            candidate_action[:, : args.action_exec_horizon, :],
            auto_reset=False,
        )
        obs.setdefault("extra_view_images", None)
        success_once = bool(infos.get("episode", {}).get("success_once", torch.zeros(1))[0].item())
        n_chunks = args.max_episode_steps // args.action_exec_horizon
        for cont_query in range(query_index + 1, n_chunks):
            if success_once:
                break
            cont_seed = args.fixed_noise_seed + 20000 + state_idx * 100 + cont_query
            cont_action, _ = sample_policy_action(sft_actor, obs, cont_seed)
            obs, _reward, _terminated, _truncated, infos = env.step(
                cont_action[:, : args.action_exec_horizon, :],
                auto_reset=False,
            )
            obs.setdefault("extra_view_images", None)
            success_once = bool(infos.get("episode", {}).get("success_once", torch.zeros(1))[0].item())
        trace = get_trace(env)
        final_sig = task_signature(env)
        return summarize_trace(trace, initial_signature["block_p"], final_sig)
    finally:
        env.close()


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    unique_vals = np.unique(arr)
    for val in unique_vals:
        mask = arr == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def per_state_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons = []
    success_hits = 0
    success_total = 0
    margin_hits = 0
    margin_total = 0
    valid_margins = []
    valid_qs = []
    for idx, left in enumerate(rows):
        if left["margin_cm"] is not None:
            valid_margins.append(float(left["margin_cm"]))
            valid_qs.append(float(left["q_used"]))
        for right in rows[idx + 1 :]:
            q_diff = float(left["q_used"] - right["q_used"])
            if bool(left["success"]) != bool(right["success"]):
                true_diff = 1.0 if left["success"] else -1.0
                hit = q_diff * true_diff > 0
                success_hits += int(hit)
                success_total += 1
                comparisons.append(
                    {
                        "candidate_a": left["candidate"],
                        "candidate_b": right["candidate"],
                        "type": "success",
                        "q_diff": q_diff,
                        "true_diff": true_diff,
                        "hit": hit,
                    }
                )
            if left["margin_cm"] is not None and right["margin_cm"] is not None:
                margin_diff = float(left["margin_cm"] - right["margin_cm"])
                if abs(margin_diff) > 0.05:
                    hit = q_diff * margin_diff > 0
                    margin_hits += int(hit)
                    margin_total += 1
                    comparisons.append(
                        {
                            "candidate_a": left["candidate"],
                            "candidate_b": right["candidate"],
                            "type": "margin",
                            "q_diff": q_diff,
                            "true_diff": margin_diff,
                            "hit": hit,
                        }
                    )
    by_name = {row["candidate"]: row for row in rows}

    def sign_agreement(a: str, b: str, field: str) -> bool | None:
        if a not in by_name or b not in by_name:
            return None
        qa = float(by_name[a]["q_used"])
        qb = float(by_name[b]["q_used"])
        va = by_name[a][field]
        vb = by_name[b][field]
        if va is None or vb is None:
            return None
        qd = qa - qb
        vd = float(va) - float(vb)
        if abs(qd) < 1e-8 or abs(vd) < 0.05:
            return None
        return bool(qd * vd > 0)

    grad_margin_better = None
    if "grad_plus" in by_name and "grad_minus" in by_name:
        gp = by_name["grad_plus"]["margin_cm"]
        gm = by_name["grad_minus"]["margin_cm"]
        if gp is not None and gm is not None:
            grad_margin_better = bool(float(gp) > float(gm) + 0.05)

    summary = {
        "success_pair_acc": success_hits / success_total if success_total else None,
        "success_pair_count": success_total,
        "margin_pair_acc": margin_hits / margin_total if margin_total else None,
        "margin_pair_count": margin_total,
        "spearman_q_margin": spearman(valid_qs, valid_margins),
        "qam_vs_sft_margin_sign_agree": sign_agreement("qam_same_noise", "sft", "margin_cm"),
        "qam_vs_mirror_margin_sign_agree": sign_agreement("qam_same_noise", "qam_mirror", "margin_cm"),
        "grad_plus_margin_better_than_minus": grad_margin_better,
    }
    return summary, comparisons


def bootstrap_ci(values: list[float], n: int = 2000) -> list[float | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return [None, None]
    rng = np.random.default_rng(20260713)
    samples = []
    arr = np.asarray(clean)
    for _ in range(n):
        samples.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def decide(summary: dict[str, Any]) -> str:
    margin = summary.get("mean_margin_pair_acc")
    success = summary.get("mean_success_pair_acc")
    spear = summary.get("median_spearman_q_margin")
    grad = summary.get("grad_plus_better_rate")
    if margin is None and success is None:
        return "INCONCLUSIVE"
    primary = margin if margin is not None else success
    if primary is not None and primary <= 0.55 and (spear is None or spear <= 0.0):
        return "NO_GO"
    if primary is not None and primary > 0.65 and (spear is not None and spear > 0.0):
        if grad is not None and grad > 0.5:
            return "CRITIC_PASS"
        return "RANKING_ONLY"
    return "INCONCLUSIVE"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    keys = list(rows[0].keys())
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    repo = setup_paths()
    patch_success_trace()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.num_states is None:
        args.num_states = 1 if args.mode == "dry-run" else (4 if args.mode == "pilot" else 20)
    seeds = seeds[: args.num_states]
    if args.mode == "dry-run":
        args.snapshot_repeats = min(args.snapshot_repeats, 1)
        candidate_allowlist = {"sft", "qam_same_noise"}
    else:
        candidate_allowlist = None

    cfg = build_eval_cfg(args, args.sft_path, out_dir)
    q_mode = "mean"
    sft_actor = load_actor(cfg, args.sft_path, device)
    qam_cfg = build_eval_cfg(args, args.qam_path, out_dir)
    qam_actor = load_actor(qam_cfg, args.qam_path, device)

    lwd_cfg = OmegaConf.load(repo / "examples" / "lwd" / "config" / "model" / "lwd_critic.yaml")
    lwd_cfg.model_path = args.critic_path
    critic = get_lwd_critic_model(lwd_cfg).to(device).eval()
    critic.requires_grad_(False)
    processor = ValueProcessor(
        tokenizer_name_or_path=lwd_cfg.tokenizer_path,
        max_token_len=int(lwd_cfg.max_token_len),
        do_augment=False,
    )
    collator = LWDChunkDataCollator(processor=processor, train=False)
    norm_stats = _load_norm_stats(Path(args.data_root) / "norm_stats.json")

    config_record = {
        **vars(args),
        "q_mode": q_mode,
        "action_semantics": "qpos",
        "cartesian_perturbations": "skipped_no_safe_ik_conversion",
    }
    write_json(out_dir / "config.json", config_record)

    states_manifest: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    rollout_results: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    per_state_rows: list[dict[str, Any]] = []

    for state_idx, seed in enumerate(seeds):
        state_id = f"seed{seed}_q{args.query_index}"
        env = make_env(cfg)
        try:
            obs, prefix_actions, signature = collect_canonical_state(
                env,
                sft_actor,
                seed,
                args.query_index,
                args.fixed_noise_seed + state_idx * 1000,
            )
            obs_hash_value = obs_hash(obs)
        finally:
            env.close()

        restore = validate_restore(
            cfg,
            sft_actor,
            seed,
            prefix_actions,
            signature,
            args.snapshot_repeats,
        )
        manifest = {
            "state_id": state_id,
            "seed": seed,
            "query_index": args.query_index,
            "restore": restore,
            "observation_hash": obs_hash_value,
            "snapshot_method": "same_seed_plus_sft_prefix_replay",
            "active_arm": None,
            "gripper_close_step": None,
        }
        states_manifest.append(manifest)
        if not restore["passed"]:
            per_state_rows.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "status": "restore_failed",
                    "success_pair_acc": None,
                    "margin_pair_acc": None,
                    "spearman_q_margin": None,
                }
            )
            continue

        critic_obs = critic_observation_from_env(obs, collator, norm_stats, device)
        candidates, candidate_meta = build_candidates(
            args,
            sft_actor,
            qam_actor,
            critic,
            critic_obs,
            move_to_device(obs, device),
            state_idx,
            q_mode,
        )
        if candidate_allowlist is not None:
            candidates = {k: v for k, v in candidates.items() if k in candidate_allowlist}

        state_rows = []
        sft_score_for_delta = None
        for candidate_name, candidate in candidates.items():
            scores = critic_scores(
                critic,
                critic_obs,
                candidate["model_action"],
                ACTION_DIM,
                q_mode,
            )
            if candidate_name == "sft":
                sft_score_for_delta = scores["q_used"]
            candidate_scores.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "candidate": candidate_name,
                    "kind": candidate["kind"],
                    **scores,
                    "action_model_l2_from_sft": float(
                        (
                            candidate["model_action"][..., :ACTION_DIM]
                            - candidates["sft"]["model_action"][..., :ACTION_DIM]
                        )
                        .flatten(1)
                        .norm(dim=-1)[0]
                        .item()
                    ),
                }
            )

        for candidate_name, candidate in candidates.items():
            for repeat_idx in range(args.repeats):
                result = run_candidate_rollout(
                    cfg,
                    sft_actor,
                    seed,
                    state_idx * 100 + repeat_idx,
                    args.query_index,
                    prefix_actions,
                    candidate["env_action"],
                    signature,
                    args,
                )
                score = next(
                    row
                    for row in candidate_scores
                    if row["state_id"] == state_id and row["candidate"] == candidate_name
                )
                row = {
                    "state_id": state_id,
                    "seed": seed,
                    "repeat": repeat_idx,
                    "candidate": candidate_name,
                    **score,
                    "q_delta_from_sft": float(score["q_used"] - sft_score_for_delta),
                    **result,
                }
                rollout_results.append(row)
                state_rows.append(row)

        # repeats are averaged per candidate before pairwise state metrics.
        averaged_rows = []
        for candidate_name in candidates:
            rows = [row for row in state_rows if row["candidate"] == candidate_name]
            base = rows[0].copy()
            for field in ["success", "contact_count"]:
                base[field] = float(np.mean([float(row[field]) for row in rows]))
            for field in ["min_xy_linf_cm", "min_xy_l2_cm", "margin_cm", "block_displacement_cm"]:
                vals = [row[field] for row in rows if row[field] is not None]
                base[field] = float(np.mean(vals)) if vals else None
            base["success"] = bool(base["success"] >= 0.5)
            averaged_rows.append(base)

        state_summary, comparisons = per_state_metrics(averaged_rows)
        per_state_rows.append(
            {
                "state_id": state_id,
                "seed": seed,
                "status": "ok",
                **state_summary,
                "skipped_candidates": json.dumps(candidate_meta["skipped_candidates"], ensure_ascii=False),
                "gradient_target_norm": candidate_meta["gradient_target_norm"],
                "critic_grad_norm": candidate_meta["grad_norm"],
            }
        )
        for comp in comparisons:
            pairwise_rows.append({"state_id": state_id, "seed": seed, **comp})

    valid_states = [row for row in per_state_rows if row["status"] == "ok"]
    success_accs = [row["success_pair_acc"] for row in valid_states if row["success_pair_acc"] is not None]
    margin_accs = [row["margin_pair_acc"] for row in valid_states if row["margin_pair_acc"] is not None]
    spears = [row["spearman_q_margin"] for row in valid_states if row["spearman_q_margin"] is not None]
    qam_sft = [
        row["qam_vs_sft_margin_sign_agree"]
        for row in valid_states
        if row["qam_vs_sft_margin_sign_agree"] is not None
    ]
    qam_mirror = [
        row["qam_vs_mirror_margin_sign_agree"]
        for row in valid_states
        if row["qam_vs_mirror_margin_sign_agree"] is not None
    ]
    grad_better = [
        row["grad_plus_margin_better_than_minus"]
        for row in valid_states
        if row["grad_plus_margin_better_than_minus"] is not None
    ]
    summary = {
        "num_states_requested": args.num_states,
        "num_states_valid": len(valid_states),
        "mean_success_pair_acc": float(np.mean(success_accs)) if success_accs else None,
        "success_pair_acc_ci95": bootstrap_ci(success_accs),
        "mean_margin_pair_acc": float(np.mean(margin_accs)) if margin_accs else None,
        "margin_pair_acc_ci95": bootstrap_ci(margin_accs),
        "median_spearman_q_margin": float(np.median(spears)) if spears else None,
        "spearman_ci95": bootstrap_ci(spears),
        "qam_vs_sft_sign_agree_rate": float(np.mean(qam_sft)) if qam_sft else None,
        "qam_vs_mirror_sign_agree_rate": float(np.mean(qam_mirror)) if qam_mirror else None,
        "grad_plus_better_rate": float(np.mean(grad_better)) if grad_better else None,
        "decision": None,
        "limitations": [
            "Simulator state is restored by same seed plus SFT prefix replay, not by a low-level Sapien snapshot API.",
            "Cartesian +/-2mm candidates are skipped because current executable action is qpos and no safe IK conversion is wired.",
            "Pilot sample is small; decision is diagnostic, not a publication-level estimate.",
        ],
    }
    summary["decision"] = decide(summary)

    write_jsonl(out_dir / "states_manifest.jsonl", states_manifest)
    write_jsonl(out_dir / "candidate_scores.jsonl", candidate_scores)
    write_jsonl(out_dir / "rollout_results.jsonl", rollout_results)
    write_csv(out_dir / "per_state_summary.csv", per_state_rows)
    write_csv(out_dir / "pairwise_comparisons.csv", pairwise_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Close models after all outputs are already on disk.
    del sft_actor, qam_actor, critic
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
