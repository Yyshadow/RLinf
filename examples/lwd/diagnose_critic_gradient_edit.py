# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Run a small closed-loop critic action-gradient edit diagnostic.

The diagnostic compares four policies from matched RoboTwin reset seeds:

* ``base``: the SFT/OpenPI action chunk.
* ``plus``: the executed prefix edited along ``+grad_A Q``.
* ``minus``: the executed prefix edited along ``-grad_A Q``.
* ``random``: the executed prefix edited along a random direction with the
  same normalized step size.

This tests whether the critic's local action gradient has useful closed-loop
meaning before routing it through QAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.datasets.lwd.chunk_dataset import (
    PI05_STATE_BINS,
    LWDChunkDataCollator,
    _decode_state,
    _load_norm_stats,
)
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.lwd_critic import get_model as get_lwd_critic_model
from rlinf.models.embodiment.openpi import get_model as get_openpi_model
from rlinf.models.embodiment.value_model.processing import ValueProcessor


VARIANTS = ("base", "plus", "minus", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--out-dir", default="outputs/lwd_critic_grad_edit_diag_n10")
    parser.add_argument(
        "--seeds",
        default="600001,600002,600004,600006,600007,600008,600010,600012,600013,600014",
    )
    parser.add_argument(
        "--actor-path",
        default="/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_allstats/global_step_11000",
    )
    parser.add_argument(
        "--critic-path",
        default="/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor",
    )
    parser.add_argument(
        "--data-root",
        default="/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split",
    )
    parser.add_argument("--fixed-noise-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def load_eval_cfg(args: argparse.Namespace):
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("REPO_PATH", str(repo))
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples" / "embodiment"))
    os.environ.setdefault("ROBOTWIN_PATH", "/data/wam_codebase/RoboTwin_RLinf")
    if os.environ["ROBOTWIN_PATH"] not in sys.path:
        sys.path.insert(0, os.environ["ROBOTWIN_PATH"])
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    config_dir = str(repo / "evaluations" / "robotwin")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="robotwin_beat_block_hammer_openpi_pi05_eval",
            overrides=[
                f"runner.logger.log_path={args.out_dir}",
                f"runner.ckpt_path={args.actor_path}/actor/model_state_dict/full_weights.pt",
                f"rollout.model.model_path={args.actor_path}",
                f"rollout.model.openpi.norm_stats_path={args.data_root}/norm_stats.json",
                "rollout.model.openpi.noise_method=flow_ode",
                "rollout.model.openpi.noise_level=0.0",
                f"rollout.model.openpi.fixed_eval_noise_seed={args.fixed_noise_seed}",
            ],
        )

    lwd_cfg = OmegaConf.load(repo / "examples" / "lwd" / "config" / "model" / "lwd_critic.yaml")
    lwd_cfg.model_path = args.critic_path
    OmegaConf.set_struct(cfg, False)
    cfg.critic = OmegaConf.create({"model": lwd_cfg})

    cfg.env.eval.total_num_envs = len(VARIANTS)
    cfg.env.eval.group_size = 1
    cfg.env.eval.rollout_epoch = 1
    cfg.env.eval.auto_reset = False
    cfg.env.eval.video_cfg.save_video = bool(args.save_video)
    cfg.env.eval.video_cfg.video_base_dir = str(Path(args.out_dir) / "video")
    return cfg


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def normalize_state_for_prompt(state: torch.Tensor, norm_stats: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    state_pi = _decode_state(state.detach().cpu().numpy(), adapt_to_pi=True)
    stats = norm_stats["state"]
    low = stats["q01"][..., : state_pi.shape[-1]]
    high = stats["q99"][..., : state_pi.shape[-1]]
    return (state_pi - low) / (high - low + 1e-6) * 2.0 - 1.0


def pi05_state_prompt(task_prompt: str, state_norm: np.ndarray) -> str:
    bins = np.linspace(-1, 1, PI05_STATE_BINS + 1)[:-1]
    discretized_state = np.digitize(state_norm, bins=bins) - 1
    discretized_state = np.clip(discretized_state, 0, PI05_STATE_BINS - 1)
    state_str = " ".join(map(str, discretized_state.astype(np.int64).tolist()))
    cleaned_prompt = task_prompt.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned_prompt}, State: {state_str};\nAction: "


def critic_observation_from_env(
    env_obs: dict[str, Any],
    collator: LWDChunkDataCollator,
    norm_stats: dict[str, dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, Any]:
    images_list = []
    prompts = []
    batch_size = int(env_obs["main_images"].shape[0])
    for idx in range(batch_size):
        images = {"cam_high": env_obs["main_images"][idx]}
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is not None:
            images["cam_left_wrist"] = wrist_images[idx, 0]
            images["cam_right_wrist"] = wrist_images[idx, 1]
        images_list.append(images)

        state_norm = normalize_state_for_prompt(env_obs["states"][idx], norm_stats)
        prompts.append(pi05_state_prompt(env_obs["task_descriptions"][idx], state_norm))

    observation = collator._build_observation(images_list, prompts)
    return move_to_device(observation, device)


@torch.no_grad()
def sample_model_action(
    actor,
    env_obs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    env_obs = dict(env_obs)
    env_obs.setdefault("extra_view_images", None)
    to_process_obs = actor.obs_processor(env_obs)
    processed_obs = actor.input_transform(to_process_obs, transpose=False)
    processed_obs = actor.precision_processor(processed_obs)

    from openpi.models import model as _model

    observation = _model.Observation.from_dict(processed_obs)
    outputs = actor.sample_actions(observation, mode="eval", compute_values=False)
    model_action = outputs["actions"].detach()
    env_action = actor.output_transform(
        {"actions": model_action, "state": observation.state}
    )["actions"].to(device)
    return model_action.to(device), env_action


def to_env_action(actor, env_obs: dict[str, Any], model_action: torch.Tensor) -> torch.Tensor:
    env_obs = dict(env_obs)
    env_obs.setdefault("extra_view_images", None)
    to_process_obs = actor.obs_processor(env_obs)
    processed_obs = actor.input_transform(to_process_obs, transpose=False)
    processed_obs = actor.precision_processor(processed_obs)

    from openpi.models import model as _model

    observation = _model.Observation.from_dict(processed_obs)
    return actor.output_transform(
        {"actions": model_action.detach(), "state": observation.state}
    )["actions"].to(model_action.device)


def critic_gradient(
    critic,
    critic_observation: dict[str, Any],
    model_action: torch.Tensor,
    env_action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action = model_action[..., :env_action_dim].detach().float().requires_grad_(True)
    q_values = critic(observation=critic_observation, action_chunk=action).q_values.float()
    q_mean = q_values.mean(dim=-1)
    grad = torch.autograd.grad(q_mean.sum(), action)[0]
    return grad.detach(), q_mean.detach(), q_values.detach()


@torch.no_grad()
def critic_q_mean(
    critic,
    critic_observation: dict[str, Any],
    model_action: torch.Tensor,
    env_action_dim: int,
) -> torch.Tensor:
    action = model_action[..., :env_action_dim].detach().float()
    q_values = critic(observation=critic_observation, action_chunk=action).q_values.float()
    return q_values.mean(dim=-1).detach()


def apply_variant_edits(
    model_action: torch.Tensor,
    grad: torch.Tensor,
    epsilon: float,
    exec_horizon: int,
    rng: torch.Generator,
) -> torch.Tensor:
    edited = model_action.clone()
    env_dim = grad.shape[-1]
    mask = torch.zeros_like(grad)
    mask[:, :exec_horizon, :] = 1.0
    grad = grad * mask
    grad_norm = grad.flatten(1).norm(dim=-1).clamp_min(1e-6)
    grad_dir = grad / grad_norm.view(-1, 1, 1)

    random_dir = torch.randn(
        grad.shape,
        generator=rng,
        device=grad.device,
        dtype=grad.dtype,
    )
    random_dir = random_dir * mask
    random_norm = random_dir.flatten(1).norm(dim=-1).clamp_min(1e-6)
    random_dir = random_dir / random_norm.view(-1, 1, 1)

    for env_idx, variant in enumerate(VARIANTS):
        if variant == "plus":
            edited[env_idx, :, :env_dim] += epsilon * grad_dir[env_idx]
        elif variant == "minus":
            edited[env_idx, :, :env_dim] -= epsilon * grad_dir[env_idx]
        elif variant == "random":
            edited[env_idx, :, :env_dim] += epsilon * random_dir[env_idx]
    edited[..., :env_dim] = edited[..., :env_dim].clamp(-1.0, 1.0)
    return edited


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"num_cases": len(records), "variants": {}}
    for variant in VARIANTS:
        successes = [bool(item["success"][variant]) for item in records]
        returns = [float(item["return"][variant]) for item in records]
        summary["variants"][variant] = {
            "success": int(sum(successes)),
            "total": len(successes),
            "success_rate": float(sum(successes) / max(1, len(successes))),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
        }
    summary["paired"] = {
        "plus_only_over_base": sum(
            (not item["success"]["base"]) and item["success"]["plus"] for item in records
        ),
        "base_only_over_plus": sum(
            item["success"]["base"] and (not item["success"]["plus"]) for item in records
        ),
        "minus_only_over_base": sum(
            (not item["success"]["base"]) and item["success"]["minus"] for item in records
        ),
        "base_only_over_minus": sum(
            item["success"]["base"] and (not item["success"]["minus"]) for item in records
        ),
        "random_only_over_base": sum(
            (not item["success"]["base"]) and item["success"]["random"] for item in records
        ),
        "base_only_over_random": sum(
            item["success"]["base"] and (not item["success"]["random"]) for item in records
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_eval_cfg(args)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    actor = get_openpi_model(cfg.rollout.model).to(device).eval()
    critic = get_lwd_critic_model(cfg.critic.model).to(device).eval()
    critic.requires_grad_(False)

    processor = ValueProcessor(
        tokenizer_name_or_path=cfg.critic.model.tokenizer_path,
        max_token_len=int(cfg.critic.model.max_token_len),
        do_augment=False,
    )
    collator = LWDChunkDataCollator(processor=processor, train=False)
    norm_stats = _load_norm_stats(Path(args.data_root) / "norm_stats.json")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    seeds = seeds[: args.num_cases]
    rng = torch.Generator(device=device)
    rng.manual_seed(20260713)

    records = []
    jsonl_path = out_dir / "critic_grad_edit_episode_metrics.jsonl"
    env = RoboTwinEnv(
        cfg.env.eval,
        num_envs=len(VARIANTS),
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for case_idx, seed in enumerate(seeds):
            obs, _ = env.reset(env_seeds=[seed] * len(VARIANTS))
            obs.setdefault("extra_view_images", None)

            chunk_q_means: list[dict[str, float]] = []
            chunk_q_deltas: list[dict[str, float]] = []
            n_chunks = int(cfg.env.eval.max_episode_steps // cfg.env.eval.action_exec_horizon)
            for _chunk_idx in range(n_chunks):
                model_action, _ = sample_model_action(actor, obs, device)
                critic_obs = critic_observation_from_env(obs, collator, norm_stats, device)
                grad, q_before, _ = critic_gradient(
                    critic,
                    critic_obs,
                    model_action,
                    int(cfg.rollout.model.action_dim),
                )
                edited_model_action = apply_variant_edits(
                    model_action,
                    grad,
                    float(args.epsilon),
                    int(cfg.env.eval.action_exec_horizon),
                    rng,
                )
                q_after = critic_q_mean(
                    critic,
                    critic_obs,
                    edited_model_action,
                    int(cfg.rollout.model.action_dim),
                )
                env_action = to_env_action(actor, obs, edited_model_action)
                step_action = env_action[:, : int(cfg.env.eval.action_exec_horizon), :]
                obs, _reward, _terminated, _truncated, infos = env.step(
                    step_action,
                    auto_reset=False,
                )
                obs.setdefault("extra_view_images", None)
                chunk_q_means.append(
                    {
                        variant: float(q_before[idx].item())
                        for idx, variant in enumerate(VARIANTS)
                    }
                )
                chunk_q_deltas.append(
                    {
                        variant: float((q_after[idx] - q_before[idx]).item())
                        for idx, variant in enumerate(VARIANTS)
                    }
                )

            episode = infos["episode"]
            success = {
                variant: bool(episode["success_once"][idx].item())
                for idx, variant in enumerate(VARIANTS)
            }
            returns = {
                variant: float(episode["return"][idx].item())
                for idx, variant in enumerate(VARIANTS)
            }
            record = {
                "case_idx": case_idx,
                "seed": seed,
                "epsilon": float(args.epsilon),
                "variants": list(VARIANTS),
                "success": success,
                "return": returns,
                "chunk_q_mean": chunk_q_means,
                "chunk_q_delta_after_edit": chunk_q_deltas,
            }
            records.append(record)
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            print(json.dumps(record, ensure_ascii=False))
            env.close()

    summary = summarize_records(records)
    summary_path = out_dir / "critic_grad_edit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
