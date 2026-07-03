#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Merge local LeRobot datasets into one clean, renumbered dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkits.robotwin.collect_dense_lerobot_aloha import append_episode, create_lerobot_dataset
from toolkits.robotwin.prepare_lerobot_aloha import (
    ACTION,
    CAM_HIGH,
    STATE,
    iter_lerobot_parquets,
    lerobot_episode_to_frames,
    load_lerobot_tasks,
    scalar_bool,
    scalar_float,
    write_metadata,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_robotwin_metadata(root: Path) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(root / "meta" / "robotwin_episode_metadata.jsonl")
    return {int(row["episode_id"]): row for row in rows if "episode_id" in row}


def episode_id_from_parquet(path: Path) -> int:
    return int(path.stem.replace("episode_", ""))


def dedupe_key(meta: dict[str, Any], frames: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Return a stable episode key across retry/resume output directories."""

    if "source_episode" in meta:
        return (
            meta.get("task"),
            meta.get("source"),
            meta.get("source_episode"),
            meta.get("seed"),
            meta.get("perturb_type"),
            meta.get("perturb_mode"),
            meta.get("perturb_scale"),
            round(float(meta.get("replay_start_fraction", -1.0)), 12),
            meta.get("active_arm"),
            meta.get("num_steps", len(frames)),
        )

    return (
        meta.get("task"),
        meta.get("source"),
        meta.get("seed"),
        meta.get("policy_ckpt"),
        meta.get("success"),
        meta.get("num_steps", len(frames)),
    )


def infer_metadata(
    *,
    old_meta: dict[str, Any],
    frames: list[dict[str, Any]],
    episode_id: int,
    source_root: Path,
    source_episode_id: int,
) -> dict[str, Any]:
    rewards = [scalar_float(frame.get("reward")) for frame in frames]
    success = any(scalar_bool(frame.get("success")) for frame in frames)
    row = dict(old_meta)
    row.update(
        {
            "episode_id": episode_id,
            "success": success,
            "return": float(np.sum(rewards)),
            "num_steps": len(frames),
            "merged_from_dataset": source_root.name,
            "merged_from_episode_id": source_episode_id,
        }
    )
    row.setdefault("task", frames[0].get("task", source_root.name))
    row.setdefault("source", "merged_lerobot")
    return row


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)


def merge_datasets(args: argparse.Namespace) -> None:
    prepare_output(args.output, args.overwrite)

    dataset = None
    metadata: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    skipped_duplicates = 0

    for source_root in args.sources:
        tasks = load_lerobot_tasks(source_root)
        meta_by_episode = load_robotwin_metadata(source_root)
        default_task = args.task or source_root.name

        for parquet_path in iter_lerobot_parquets(source_root):
            if args.max_episodes is not None and len(metadata) >= args.max_episodes:
                break

            source_episode_id = episode_id_from_parquet(parquet_path)
            frames = lerobot_episode_to_frames(
                parquet_path,
                tasks=tasks,
                default_task=default_task,
                default_success=args.default_success,
            )
            if not frames:
                continue

            old_meta = meta_by_episode.get(source_episode_id, {})
            key = dedupe_key(old_meta, frames)
            if args.dedupe and key in seen:
                skipped_duplicates += 1
                continue
            seen.add(key)

            if dataset is None:
                dataset = create_lerobot_dataset(args.output, frames[0], args.robot_type, args.fps)
            append_episode(dataset, frames)
            metadata.append(
                infer_metadata(
                    old_meta=old_meta,
                    frames=frames,
                    episode_id=len(metadata),
                    source_root=source_root,
                    source_episode_id=source_episode_id,
                )
            )
            write_metadata(args.output, metadata)

        if args.max_episodes is not None and len(metadata) >= args.max_episodes:
            break

    if dataset is None:
        raise RuntimeError("No episodes were written.")
    if args.max_episodes is not None and len(metadata) < args.max_episodes:
        raise RuntimeError(f"Collected only {len(metadata)} / {args.max_episodes} merged episodes.")

    print(f"wrote {len(metadata)} episode(s) to {args.output}")
    print(f"skipped {skipped_duplicates} duplicate episode(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--robot-type", default="aloha")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--default-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    merge_datasets(parse_args())


if __name__ == "__main__":
    main()
