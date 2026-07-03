#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Split one LeRobot dataset into episode-level train/eval datasets."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return root / data_path.format(
        episode_chunk=chunk_idx,
        episode_index=episode_index,
    )


def _reindex_episode_stats(
    stats: dict[str, Any],
    *,
    new_episode_index: int,
    old_frame_start: int,
    new_frame_start: int,
) -> dict[str, Any]:
    out = copy.deepcopy(stats)
    frame_offset = new_frame_start - old_frame_start

    if "episode_index" in out:
        count = out["episode_index"].get("count", [1])
        out["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": count,
        }

    if "index" in out:
        index_stats = out["index"]
        out["index"] = {
            "min": [v + frame_offset for v in _as_list(index_stats.get("min", [0]))],
            "max": [v + frame_offset for v in _as_list(index_stats.get("max", [0]))],
            "mean": [
                v + frame_offset for v in _as_list(index_stats.get("mean", [0.0]))
            ],
            "std": index_stats.get("std", [0.0]),
            "count": index_stats.get("count", [1]),
        }

    return out


def _task_map_for_episodes(episodes: list[dict[str, Any]]) -> dict[str, int]:
    task_to_index: dict[str, int] = {}
    for episode in episodes:
        for task in episode.get("tasks", []):
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
    return task_to_index


def _write_subset(
    source: Path,
    output: Path,
    episode_indices: list[int],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = _read_json(source / "meta" / "info.json")
    episodes = {
        int(row["episode_index"]): row
        for row in _read_jsonl(source / "meta" / "episodes.jsonl")
    }
    episode_stats = {
        int(row["episode_index"]): row
        for row in _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }
    robotwin_metadata = {
        int(row["episode_id"]): row
        for row in _read_jsonl(source / "meta" / "robotwin_episode_metadata.jsonl")
        if "episode_id" in row
    }

    selected_episodes = [episodes[idx] for idx in episode_indices]
    task_to_index = _task_map_for_episodes(selected_episodes)

    output.mkdir(parents=True, exist_ok=False)
    (output / "meta").mkdir()
    output_chunks_size = int(info.get("chunks_size", 1000))

    global_frame_index = 0
    out_episodes = []
    out_episode_stats = []
    out_robotwin_metadata = []

    for new_episode_index, old_episode_index in enumerate(episode_indices):
        parquet_path = _episode_path(source, info, old_episode_index)
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        frame_count = len(df)
        old_frame_start = int(df["index"].min()) if "index" in df.columns else 0

        episode = episodes[old_episode_index]
        tasks = episode.get("tasks", ["unknown task"])
        task_index = task_to_index.get(tasks[0], 0)
        df["episode_index"] = new_episode_index
        df["index"] = range(global_frame_index, global_frame_index + frame_count)
        if "task_index" in df.columns:
            df["task_index"] = task_index

        chunk_idx = new_episode_index // output_chunks_size
        chunk_dir = output / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = chunk_dir / f"episode_{new_episode_index:06d}.parquet"

        out_table = pa.Table.from_pandas(df, preserve_index=False)
        if table.schema.metadata:
            out_table = out_table.cast(out_table.schema.with_metadata(table.schema.metadata))
        pq.write_table(out_table, out_parquet)

        out_episode = {
            **{
                key: value
                for key, value in episode.items()
                if key not in {"episode_index", "length"}
            },
            "episode_index": new_episode_index,
            "length": frame_count,
        }
        out_episodes.append(out_episode)

        if old_episode_index in episode_stats:
            out_episode_stats.append(
                {
                    "episode_index": new_episode_index,
                    "stats": _reindex_episode_stats(
                        episode_stats[old_episode_index]["stats"],
                        new_episode_index=new_episode_index,
                        old_frame_start=old_frame_start,
                        new_frame_start=global_frame_index,
                    ),
                }
            )

        if old_episode_index in robotwin_metadata:
            metadata = dict(robotwin_metadata[old_episode_index])
            metadata["episode_id"] = new_episode_index
            metadata["source_episode_id"] = old_episode_index
            out_robotwin_metadata.append(metadata)

        global_frame_index += frame_count

    total_chunks = max(1, (len(episode_indices) + output_chunks_size - 1) // output_chunks_size)
    out_info = dict(info)
    out_info.update(
        {
            "total_episodes": len(episode_indices),
            "total_frames": global_frame_index,
            "total_tasks": len(task_to_index),
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": output_chunks_size,
            "splits": {"train": f"0:{len(episode_indices)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
    )
    out_info.pop("video_path", None)

    _write_json(output / "meta" / "info.json", out_info)
    _write_jsonl(output / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(
        output / "meta" / "tasks.jsonl",
        [
            {"task_index": task_index, "task": task}
            for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
        ],
    )
    if out_episode_stats:
        _write_jsonl(output / "meta" / "episodes_stats.jsonl", out_episode_stats)
    if out_robotwin_metadata:
        _write_jsonl(
            output / "meta" / "robotwin_episode_metadata.jsonl",
            out_robotwin_metadata,
        )


def split_dataset(
    source: Path,
    output_root: Path,
    train_name: str,
    eval_name: str,
    train_episodes: int,
    eval_episodes: int,
    overwrite: bool,
) -> None:
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    total = len(episodes)
    if train_episodes + eval_episodes > total:
        raise ValueError(
            f"Requested {train_episodes}+{eval_episodes} episodes, but {source} has {total}."
        )

    train_output = output_root / train_name
    eval_output = output_root / eval_name
    if overwrite:
        for path in (train_output, eval_output):
            if path.exists():
                shutil.rmtree(path)

    train_indices = list(range(train_episodes))
    eval_indices = list(range(total - eval_episodes, total))
    _write_subset(source, train_output, train_indices)
    _write_subset(source, eval_output, eval_indices)
    print(f"[split] {source.name}: train={len(train_indices)} -> {train_output}")
    print(f"[split] {source.name}: eval={len(eval_indices)} -> {eval_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--train-episodes", type=int, default=480)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dataset(
        source=args.source,
        output_root=args.output_root,
        train_name=args.train_name,
        eval_name=args.eval_name,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
