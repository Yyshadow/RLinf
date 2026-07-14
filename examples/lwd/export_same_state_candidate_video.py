# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Export a front-camera video for one same-state candidate rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import diagnose_same_state_action_ranking as diag
from rlinf.models.embodiment.lwd_critic import get_model as get_lwd_critic_model
from rlinf.models.embodiment.value_model.processing import ValueProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100137506)
    parser.add_argument("--state-idx", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=2)
    parser.add_argument("--candidate", default="grad_plus")
    parser.add_argument("--camera", default="front_camera")
    parser.add_argument("--video-stride", type=int, default=5)
    parser.add_argument("--fixed-noise-seed", type=int, default=20260713)
    parser.add_argument("--action-exec-horizon", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--snapshot-repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir",
        default="outputs/same_state_candidate_videos/seed100137506_q2_grad_plus_front",
    )
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


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = diag.setup_paths()
    diag.patch_success_trace()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = diag.build_eval_cfg(args, args.sft_path, out_dir)
    sft_actor = diag.load_actor(cfg, args.sft_path, device)
    qam_cfg = diag.build_eval_cfg(args, args.qam_path, out_dir)
    qam_actor = diag.load_actor(qam_cfg, args.qam_path, device)

    lwd_cfg = OmegaConf.load(repo / "examples" / "lwd" / "config" / "model" / "lwd_critic.yaml")
    lwd_cfg.model_path = args.critic_path
    critic = get_lwd_critic_model(lwd_cfg).to(device).eval()
    critic.requires_grad_(False)
    processor = ValueProcessor(
        tokenizer_name_or_path=lwd_cfg.tokenizer_path,
        max_token_len=int(lwd_cfg.max_token_len),
        do_augment=False,
    )
    collator = diag.LWDChunkDataCollator(processor=processor, train=False)
    norm_stats = diag._load_norm_stats(Path(args.data_root) / "norm_stats.json")

    state_id = f"seed{args.seed}_q{args.query_index}"
    env = diag.make_env(cfg)
    try:
        obs, prefix_actions, signature = diag.collect_canonical_state(
            env,
            sft_actor,
            args.seed,
            args.query_index,
            args.fixed_noise_seed + args.state_idx * 1000,
        )
        obs_hash_value = diag.obs_hash(obs)
    finally:
        env.close()

    restore = diag.validate_restore(
        cfg,
        sft_actor,
        args.seed,
        prefix_actions,
        signature,
        args.snapshot_repeats,
    )
    if not restore["passed"]:
        raise RuntimeError(f"State restore failed for {state_id}: {restore}")

    critic_obs = diag.critic_observation_from_env(obs, collator, norm_stats, device)
    candidates, candidate_meta = diag.build_candidates(
        args,
        sft_actor,
        qam_actor,
        critic,
        critic_obs,
        diag.move_to_device(obs, device),
        args.state_idx,
        "mean",
    )
    if args.candidate not in candidates:
        raise KeyError(f"Candidate {args.candidate!r} not found in {sorted(candidates)}")

    score = diag.critic_scores(
        critic,
        critic_obs,
        candidates[args.candidate]["model_action"],
        diag.ACTION_DIM,
        "mean",
    )

    video_cfg = diag.build_eval_cfg(args, args.sft_path, out_dir)
    video_cfg.env.eval.video_cfg.save_video = True
    video_cfg.env.eval.video_cfg.record_internal_camera = True
    video_cfg.env.eval.video_cfg.camera = args.camera
    video_cfg.env.eval.video_cfg.write_every_n_sim_steps = int(args.video_stride)
    video_cfg.env.eval.video_cfg.video_base_dir = str(out_dir / "video")

    result = diag.run_candidate_rollout(
        video_cfg,
        sft_actor,
        args.seed,
        args.state_idx,
        args.query_index,
        prefix_actions,
        candidates[args.candidate]["env_action"],
        signature,
        args,
    )

    expected_video_path = (
        out_dir
        / "video"
        / "beat_block_hammer"
        / f"seed_{args.seed}"
        / "env_0"
        / "episode.mp4"
    )
    summary = {
        "state_id": state_id,
        "seed": args.seed,
        "query_index": args.query_index,
        "candidate": args.candidate,
        "camera": args.camera,
        "video_stride": args.video_stride,
        "expected_video_path": str(expected_video_path),
        "observation_hash": obs_hash_value,
        "restore": restore,
        "score": score,
        "candidate_meta": candidate_meta,
        "rollout_result": result,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    del sft_actor, qam_actor, critic
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
