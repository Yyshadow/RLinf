#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE = "observation.state"
ACTION = "action"
CHUNK_SCHEMA = pa.schema(
    [
        ("dataset_path", pa.string()),
        ("episode_id", pa.int64()),
        ("frame_idx", pa.int64()),
        ("next_frame_idx", pa.int64()),
        ("horizon", pa.int64()),
        ("task", pa.string()),
        ("source", pa.string()),
        ("success", pa.bool_()),
        ("done", pa.bool_()),
        ("state", pa.list_(pa.float32())),
        ("next_state", pa.list_(pa.float32())),
        ("action_chunk", pa.list_(pa.list_(pa.float32()))),
        ("reward_chunk", pa.list_(pa.float32())),
        ("reward_sum", pa.float32()),
    ]
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tasks(dataset: Path) -> dict[int, str]:
    return {
        int(row["task_index"]): str(row["task"])
        for row in load_jsonl(dataset / "meta" / "tasks.jsonl")
    }


def load_episode_metadata(dataset: Path) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(dataset / "meta" / "robotwin_episode_metadata.jsonl")
    return {int(row["episode_id"]): row for row in rows if "episode_id" in row}


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


def to_float_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def episode_parquets(dataset: Path) -> list[Path]:
    return sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))


def build_chunks_for_episode(
    parquet_path: Path,
    *,
    tasks: dict[int, str],
    metadata: dict[int, dict[str, Any]],
    horizon: int,
    stride: int,
    default_source: str,
) -> list[dict[str, Any]]:
    rows = pq.read_table(parquet_path).to_pylist()
    if len(rows) <= horizon:
        return []

    episode_id = int(rows[0]["episode_index"])
    meta = metadata.get(episode_id, {})
    task = meta.get("task") or tasks.get(int(rows[0].get("task_index", -1)), "")
    source = meta.get("source", default_source)
    episode_success = bool(meta.get("success", any(scalar_bool(row.get("success")) for row in rows)))

    records = []
    for start in range(0, len(rows) - horizon, stride):
        end = start + horizon
        reward_chunk = [scalar_float(row.get("reward"), 0.0) for row in rows[start:end]]
        done_chunk = [scalar_bool(row.get("done"), False) for row in rows[start:end]]
        action_chunk = [to_float_list(row[ACTION]) for row in rows[start:end]]
        records.append(
            {
                "dataset_path": str(parquet_path.parents[2]),
                "episode_id": episode_id,
                "frame_idx": int(rows[start]["frame_index"]),
                "next_frame_idx": int(rows[end]["frame_index"]),
                "horizon": horizon,
                "task": task,
                "source": source,
                "success": episode_success,
                "done": bool(any(done_chunk)),
                "state": to_float_list(rows[start][STATE]),
                "next_state": to_float_list(rows[end][STATE]),
                "action_chunk": action_chunk,
                "reward_chunk": reward_chunk,
                "reward_sum": np.float32(np.sum(reward_chunk)).item(),
            }
        )
    return records


def build_chunks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = load_tasks(args.dataset)
    metadata = load_episode_metadata(args.dataset)
    records: list[dict[str, Any]] = []
    for parquet_path in episode_parquets(args.dataset):
        records.extend(
            build_chunks_for_episode(
                parquet_path,
                tasks=tasks,
                metadata=metadata,
                horizon=args.horizon,
                stride=args.stride,
                default_source=args.default_source,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LWD/BPG chunk indexes from a RoboTwin Aloha dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--default-source", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_chunks(args)
    if not records:
        raise RuntimeError("No chunks were generated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=CHUNK_SCHEMA), args.output)
    print(f"wrote {len(records)} chunks to {args.output}")


if __name__ == "__main__":
    main()
