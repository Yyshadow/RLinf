#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Diagnose source-paired LWD critic action dependence.

This script pairs failed/nearmiss replay episodes with their original success
source episode, then compares critic scores on the same source observation with
different action chunks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from examples.lwd.visualize_lwd_critic_episode_values import (  # noqa: E402
    build_collator,
    build_model,
    move_to_device,
    resolve_checkpoint,
    resolve_device,
)
from rlinf.data.datasets.lwd.chunk_dataset import LWDChunkDataset  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split")
DEFAULT_CHECKPOINT = Path("/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100")
DEFAULT_OUTPUT_DIR = Path("/data/wam_codebase/RLinf/outputs/lwd_source_paired_action_dependence")
DEFAULT_PRETRAINED_ROOT = Path("/data/wam_codebase/RLinf/checkpoints/pretrained")


@dataclass
class PairPoint:
    label: str
    target_dataset: str
    target_episode: int
    source_dataset: str
    source_episode: int
    source_episode_id: int
    frame_idx: int
    segment: str
    perturb_start: int
    replay_start_fraction: float
    perturb_type: str
    source_sample: dict[str, Any]
    target_sample: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--siglip-path",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT / "siglip2-so400m-patch14-224",
    )
    parser.add_argument(
        "--gemma3-path",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT / "gemma-3-270m",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT / "gemma-3-270m",
    )
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--precision", type=str, default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--quantile-tau", type=float, default=0.6)
    parser.add_argument("--points-per-episode", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_dataset(data_root: Path, name: str, action_horizon: int) -> LWDChunkDataset:
    return LWDChunkDataset(
        dataset_path=str(data_root / name),
        action_horizon=action_horizon,
        norm_stats_path=str(data_root / "norm_stats.json"),
        use_quantile_norm=True,
        adapt_to_pi=True,
    )


def episode_span(dataset: LWDChunkDataset, episode_index: int) -> tuple[int, int]:
    starts = dataset.dataset.episode_data_index["from"]
    stops = dataset.dataset.episode_data_index["to"]
    return int(starts[episode_index].item()), int(stops[episode_index].item())


def source_index_map(
    data_root: Path,
    success_train: LWDChunkDataset,
    success_eval: LWDChunkDataset,
) -> dict[int, tuple[str, LWDChunkDataset, int]]:
    mapping: dict[int, tuple[str, LWDChunkDataset, int]] = {}
    for name, dataset in [
        ("beat_block_hammer_success_train", success_train),
        ("beat_block_hammer_success_eval", success_eval),
    ]:
        rows = load_jsonl(data_root / name / "meta" / "robotwin_episode_metadata.jsonl")
        for row in rows:
            source_episode_id = int(row.get("source_episode_id", row["episode_id"]))
            mapping[source_episode_id] = (name, dataset, int(row["episode_id"]))
    return mapping


def episode_points(start: int, stop: int, num_points: int) -> list[int]:
    if stop <= start:
        return []
    count = min(num_points, stop - start)
    return np.unique(np.linspace(start, stop - 1, count).round().astype(int)).tolist()


def segment_for_frame(frame_idx: int, perturb_start: int, horizon: int) -> str:
    if frame_idx + horizon <= perturb_start:
        return "clean"
    if frame_idx < perturb_start < frame_idx + horizon:
        return "cross"
    return "post"


def clone_with_action(sample: dict[str, Any], action_chunk: torch.Tensor) -> dict[str, Any]:
    cloned = dict(sample)
    cloned["action_chunk"] = action_chunk.clone()
    return cloned


def clone_with_shuffled_action(sample: dict[str, Any]) -> dict[str, Any]:
    action = sample["action_chunk"]
    indices = torch.arange(action.shape[0] - 1, -1, -1)
    return clone_with_action(sample, action[indices])


@torch.inference_mode()
def score_examples(
    *,
    model,
    collator,
    examples: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values_quantile: list[np.ndarray] = []
    values_mean: list[np.ndarray] = []
    q_min: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = move_to_device(collator(batch_examples), device)
        out = model(batch["observation"], batch["action_chunk"])
        values_quantile.append(out.value_quantile.detach().float().cpu().numpy())
        values_mean.append(out.value_mean.detach().float().cpu().numpy())
        q_min.append(out.q_min.detach().float().cpu().numpy())
    return {
        "value_quantile": np.concatenate(values_quantile, axis=0),
        "value_mean": np.concatenate(values_mean, axis=0),
        "q_min": np.concatenate(q_min, axis=0),
    }


def collect_pair_points(args: argparse.Namespace) -> list[PairPoint]:
    data_root = args.data_root
    action_horizon = int(args.action_horizon)
    success_train = build_dataset(data_root, "beat_block_hammer_success_train", action_horizon)
    success_eval = build_dataset(data_root, "beat_block_hammer_success_eval", action_horizon)
    source_map = source_index_map(data_root, success_train, success_eval)

    target_specs = [
        ("failed", "beat_block_hammer_failed_eval"),
        ("nearmiss", "beat_block_hammer_nearmiss_eval"),
    ]
    points: list[PairPoint] = []
    for label, dataset_name in target_specs:
        target_dataset = build_dataset(data_root, dataset_name, action_horizon)
        rows = load_jsonl(data_root / dataset_name / "meta" / "robotwin_episode_metadata.jsonl")
        if args.max_episodes_per_split > 0:
            rows = rows[: args.max_episodes_per_split]
        for row in rows:
            target_episode = int(row["episode_id"])
            source_episode_id = int(row["source_episode"])
            if source_episode_id not in source_map:
                continue
            source_dataset_name, source_dataset, source_episode = source_map[source_episode_id]
            target_start, target_stop = episode_span(target_dataset, target_episode)
            source_start, source_stop = episode_span(source_dataset, source_episode)
            local_length = min(target_stop - target_start, source_stop - source_start)
            if local_length <= 0:
                continue
            replay_start_fraction = float(row["replay_start_fraction"])
            perturb_start = max(1, min(local_length - 1, int(local_length * replay_start_fraction)))
            for target_abs_idx in episode_points(
                target_start,
                target_start + local_length,
                args.points_per_episode,
            ):
                frame_idx = target_abs_idx - target_start
                source_abs_idx = source_start + frame_idx
                source_sample = source_dataset[source_abs_idx]
                target_sample = target_dataset[target_abs_idx]
                points.append(
                    PairPoint(
                        label=label,
                        target_dataset=dataset_name,
                        target_episode=target_episode,
                        source_dataset=source_dataset_name,
                        source_episode=source_episode,
                        source_episode_id=source_episode_id,
                        frame_idx=frame_idx,
                        segment=segment_for_frame(
                            frame_idx,
                            perturb_start,
                            target_dataset.action_horizon,
                        ),
                        perturb_start=perturb_start,
                        replay_start_fraction=replay_start_fraction,
                        perturb_type=str(row.get("perturb_type", "")),
                        source_sample=source_sample,
                        target_sample=target_sample,
                    )
                )
    return points


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_points": len(rows)}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["label"], row["segment"])
        groups.setdefault(key, []).append(row)

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    segment_summary = {}
    for (label, segment), items in sorted(groups.items()):
        q_delta = [float(item["q_failed_on_source_obs"] - item["q_success"]) for item in items]
        q_pref = [float(item["q_success"] > item["q_failed_on_source_obs"]) for item in items]
        v_delta = [float(item["v_failed_own"] - item["v_success"]) for item in items]
        action_range = [float(item["action_q_range"]) for item in items]
        target_gap = [
            abs(float(item["q_failed_on_source_obs"] - item["q_success"]))
            for item in items
        ]
        random_gap = [
            abs(float(item["q_random_on_source_obs"] - item["q_success"]))
            for item in items
        ]
        segment_summary[f"{label}/{segment}"] = {
            "count": len(items),
            "q_success_gt_failed_rate": float(np.mean(q_pref)) if q_pref else None,
            "q_failed_minus_success": stats(q_delta),
            "v_failed_minus_success": stats(v_delta),
            "abs_q_failed_success_gap": stats(target_gap),
            "abs_q_random_success_gap": stats(random_gap),
            "action_q_range": stats(action_range),
        }
    summary["segments"] = segment_summary

    all_action_ranges = [float(row["action_q_range"]) for row in rows]
    all_failed_gaps = [abs(float(row["q_failed_on_source_obs"] - row["q_success"])) for row in rows]
    all_random_gaps = [abs(float(row["q_random_on_source_obs"] - row["q_success"])) for row in rows]
    summary["action_dependence"] = {
        "action_q_range": stats(all_action_ranges),
        "abs_q_failed_success_gap": stats(all_failed_gaps),
        "abs_q_random_success_gap": stats(all_random_gaps),
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "target_dataset",
        "target_episode",
        "source_dataset",
        "source_episode",
        "source_episode_id",
        "frame_idx",
        "segment",
        "perturb_start",
        "replay_start_fraction",
        "perturb_type",
        "v_success",
        "v_failed_own",
        "q_success",
        "q_failed_on_source_obs",
        "q_failed_own",
        "q_random_on_source_obs",
        "q_shuffled_on_source_obs",
        "action_q_range",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = build_model(args, checkpoint, device)
    collator = build_collator(args)

    points = collect_pair_points(args)
    if not points:
        raise RuntimeError("No source-paired points were collected.")
    print(f"Collected {len(points)} source-paired points.")

    source_examples = [point.source_sample for point in points]
    target_own_examples = [point.target_sample for point in points]
    source_with_target_action = [
        clone_with_action(point.source_sample, point.target_sample["action_chunk"])
        for point in points
    ]
    random_examples = []
    for idx, point in enumerate(points):
        other = points[(idx * 37 + 17) % len(points)].target_sample
        random_examples.append(clone_with_action(point.source_sample, other["action_chunk"]))
    shuffled_examples = [clone_with_shuffled_action(point.source_sample) for point in points]

    scored_source = score_examples(
        model=model,
        collator=collator,
        examples=source_examples,
        batch_size=args.batch_size,
        device=device,
    )
    scored_target_own = score_examples(
        model=model,
        collator=collator,
        examples=target_own_examples,
        batch_size=args.batch_size,
        device=device,
    )
    scored_source_target_action = score_examples(
        model=model,
        collator=collator,
        examples=source_with_target_action,
        batch_size=args.batch_size,
        device=device,
    )
    scored_random = score_examples(
        model=model,
        collator=collator,
        examples=random_examples,
        batch_size=args.batch_size,
        device=device,
    )
    scored_shuffled = score_examples(
        model=model,
        collator=collator,
        examples=shuffled_examples,
        batch_size=args.batch_size,
        device=device,
    )

    rows = []
    for idx, point in enumerate(points):
        q_candidates = [
            float(scored_source["q_min"][idx]),
            float(scored_source_target_action["q_min"][idx]),
            float(scored_random["q_min"][idx]),
            float(scored_shuffled["q_min"][idx]),
        ]
        rows.append(
            {
                "label": point.label,
                "target_dataset": point.target_dataset,
                "target_episode": point.target_episode,
                "source_dataset": point.source_dataset,
                "source_episode": point.source_episode,
                "source_episode_id": point.source_episode_id,
                "frame_idx": point.frame_idx,
                "segment": point.segment,
                "perturb_start": point.perturb_start,
                "replay_start_fraction": point.replay_start_fraction,
                "perturb_type": point.perturb_type,
                "v_success": float(scored_source["value_quantile"][idx]),
                "v_failed_own": float(scored_target_own["value_quantile"][idx]),
                "q_success": float(scored_source["q_min"][idx]),
                "q_failed_on_source_obs": float(scored_source_target_action["q_min"][idx]),
                "q_failed_own": float(scored_target_own["q_min"][idx]),
                "q_random_on_source_obs": float(scored_random["q_min"][idx]),
                "q_shuffled_on_source_obs": float(scored_shuffled["q_min"][idx]),
                "action_q_range": max(q_candidates) - min(q_candidates),
            }
        )

    summary = summarize(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "data_root": str(args.data_root),
            "points_per_episode": args.points_per_episode,
            "batch_size": args.batch_size,
        }
    )
    write_csv(args.output_dir / "source_paired_action_dependence.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
