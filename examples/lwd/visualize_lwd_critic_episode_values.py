# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Plot LWD critic values along held-out RoboTwin episodes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlinf.data.datasets.lwd.chunk_dataset import (
    LWDChunkDataCollator,
    LWDChunkDataset,
)
from rlinf.models.embodiment.lwd_critic.lwd_critic_model import (
    LWDCriticConfig,
    LWDCriticModel,
)
from rlinf.models.embodiment.value_model.checkpoint_utils import (
    load_state_dict_from_checkpoint,
)
from rlinf.models.embodiment.value_model.processing import ValueProcessor


DEFAULT_DATA_ROOT = Path("/data/wam_codebase/RLinf/datasets/robotwin_aloha_lwd_split")
DEFAULT_CHECKPOINT = Path(
    "/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100"
)
DEFAULT_OUTPUT_DIR = Path("/data/wam_codebase/RLinf/outputs/lwd_critic_value_curves")
DEFAULT_PRETRAINED_ROOT = Path("/data/wam_codebase/RLinf/checkpoints/pretrained")
DEFAULT_SPLITS = (
    ("success", "beat_block_hammer_success_eval"),
    ("failed", "beat_block_hammer_failed_eval"),
    ("nearmiss", "beat_block_hammer_nearmiss_eval"),
)


@dataclass
class EpisodeCurve:
    label: str
    dataset_name: str
    episode_index: int
    fps: int
    records: list[dict[str, Any]]
    thumbnails: list[dict[str, Any]]


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
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--camera-key", type=str, default="cam_high")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--quantile-tau", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--precision", type=str, default="bf16", choices=("bf16", "fp32"))
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Eval split in label:dataset_dir format. Defaults to success/failed/nearmiss.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    direct = path / "actor" / "model_state_dict" / "full_weights.pt"
    if direct.exists():
        return direct

    checkpoints_dir = path / "checkpoints"
    candidates = []
    if checkpoints_dir.exists():
        for step_dir in checkpoints_dir.glob("global_step_*"):
            weight_path = step_dir / "actor" / "model_state_dict" / "full_weights.pt"
            if weight_path.exists():
                match = re.search(r"global_step_(\d+)", step_dir.name)
                step = int(match.group(1)) if match else -1
                candidates.append((step, weight_path))
    if candidates:
        return sorted(candidates)[-1][1]

    raise FileNotFoundError(f"Could not resolve full_weights.pt from {path}")


def parse_splits(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_SPLITS)
    splits = []
    for value in values:
        if ":" not in value:
            raise ValueError("--split must use label:dataset_dir format.")
        label, dataset_name = value.split(":", 1)
        splits.append((label.strip(), dataset_name.strip()))
    return splits


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def build_model(args: argparse.Namespace, checkpoint_path: Path, device: torch.device):
    config = LWDCriticConfig(
        precision=args.precision,
        siglip_path=str(args.siglip_path),
        gemma3_path=str(args.gemma3_path),
        action_dim=14,
        action_horizon=int(getattr(args, "action_horizon", 50)),
        max_token_len=200,
        max_language_len=50,
        critic_expert_variant="gemma_1m",
        action_expert_variant="gemma_300m",
        num_bins=201,
        v_min=-0.1,
        v_max=1.1,
        quantile_tau=float(getattr(args, "quantile_tau", 0.6)),
        action_hidden_dim=256,
        q_hidden_dims=[512, 256, 128],
        num_q_heads=2,
    )
    model = LWDCriticModel(config)
    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model.to(device)
    model.eval()
    return model


def build_collator(args: argparse.Namespace) -> LWDChunkDataCollator:
    processor = ValueProcessor(
        tokenizer_name_or_path=str(args.tokenizer_path),
        max_token_len=200,
        image_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        do_augment=False,
    )
    return LWDChunkDataCollator(processor=processor, max_length=200, train=False)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def episode_indices(dataset: LWDChunkDataset, episode_index: int, num_points: int) -> list[int]:
    starts = dataset.dataset.episode_data_index["from"]
    stops = dataset.dataset.episode_data_index["to"]
    if episode_index < 0 or episode_index >= len(starts):
        raise IndexError(
            f"episode_index={episode_index} is out of range for "
            f"{dataset.dataset_path.name}; num_episodes={len(starts)}"
        )
    start = int(starts[episode_index].item())
    stop = int(stops[episode_index].item())
    return np.unique(np.linspace(start, stop - 1, num_points).round().astype(int)).tolist()


def tensor_to_image(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
    return array


def evaluate_episode(
    *,
    model: LWDCriticModel,
    collator: LWDChunkDataCollator,
    dataset: LWDChunkDataset,
    label: str,
    episode_index: int,
    indices: list[int],
    batch_size: int,
    device: torch.device,
    camera_key: str,
    frame_count: int,
) -> EpisodeCurve:
    raw_samples = [dataset[index] for index in indices]
    records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for start in range(0, len(raw_samples), batch_size):
            examples = raw_samples[start : start + batch_size]
            batch = move_to_device(collator(examples), device)
            out = model(batch["observation"], batch["action_chunk"])
            value_quantile = out.value_quantile.detach().float().cpu().numpy()
            value_mean = out.value_mean.detach().float().cpu().numpy()
            q_min = out.q_min.detach().float().cpu().numpy()
            reward_sum = batch["reward_chunk"].detach().float().sum(dim=-1).cpu().numpy()
            done = batch["done"].detach().bool().cpu().numpy()
            success = batch["success"].detach().bool().cpu().numpy()
            frame_idx = batch["frame_idx"].detach().long().cpu().numpy()

            for item_idx in range(len(examples)):
                records.append(
                    {
                        "label": label,
                        "dataset": dataset.dataset_path.name,
                        "episode_index": episode_index,
                        "frame_idx": int(frame_idx[item_idx]),
                        "time_s": float(frame_idx[item_idx] / dataset.fps),
                        "value_quantile": float(value_quantile[item_idx]),
                        "value_mean": float(value_mean[item_idx]),
                        "q_min": float(q_min[item_idx]),
                        "reward_sum": float(reward_sum[item_idx]),
                        "done": bool(done[item_idx]),
                        "success": bool(success[item_idx]),
                    }
                )

    records.sort(key=lambda item: item["frame_idx"])
    thumb_positions = np.unique(
        np.linspace(0, len(raw_samples) - 1, min(frame_count, len(raw_samples)))
        .round()
        .astype(int)
    )
    thumbnails = []
    for pos in thumb_positions:
        sample = raw_samples[int(pos)]
        images = sample["images"]
        if camera_key not in images:
            raise KeyError(f"Camera key {camera_key!r} not found in {images.keys()}.")
        thumbnails.append(
            {
                "frame_idx": int(sample["frame_idx"]),
                "time_s": float(sample["frame_idx"] / dataset.fps),
                "image": tensor_to_image(images[camera_key]),
            }
        )

    return EpisodeCurve(
        label=label,
        dataset_name=dataset.dataset_path.name,
        episode_index=episode_index,
        fps=dataset.fps,
        records=records,
        thumbnails=thumbnails,
    )


def save_csv(curves: list[EpisodeCurve], output_path: Path) -> None:
    fieldnames = [
        "label",
        "dataset",
        "episode_index",
        "frame_idx",
        "time_s",
        "value_quantile",
        "value_mean",
        "q_min",
        "reward_sum",
        "done",
        "success",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            writer.writerows(curve.records)


def plot_curves(curves: list[EpisodeCurve], output_png: Path, output_pdf: Path) -> None:
    colors = {
        "success": "#2f7d46",
        "failed": "#b4443f",
        "nearmiss": "#c28a19",
    }
    fig = plt.figure(figsize=(14, 4.2 * len(curves)), constrained_layout=True)
    outer = fig.add_gridspec(
        nrows=2 * len(curves),
        ncols=1,
        height_ratios=[1.0, 2.1] * len(curves),
    )

    for row, curve in enumerate(curves):
        frame_grid = outer[2 * row].subgridspec(1, len(curve.thumbnails), wspace=0.05)
        for col, thumb in enumerate(curve.thumbnails):
            ax_img = fig.add_subplot(frame_grid[0, col])
            ax_img.imshow(thumb["image"])
            ax_img.set_title(f"t={thumb['time_s']:.1f}s", fontsize=10)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_visible(False)

        ax = fig.add_subplot(outer[2 * row + 1])
        times = np.asarray([item["time_s"] for item in curve.records])
        value_quantile = np.asarray([item["value_quantile"] for item in curve.records])
        value_mean = np.asarray([item["value_mean"] for item in curve.records])
        q_min = np.asarray([item["q_min"] for item in curve.records])
        reward_sum = np.asarray([item["reward_sum"] for item in curve.records])

        color = colors.get(curve.label, "#3d6fb6")
        ax.plot(times, value_quantile, color=color, linewidth=2.4, label="V quantile")
        ax.plot(times, value_mean, color=color, linewidth=1.6, alpha=0.45, label="V mean")
        ax.plot(times, q_min, color="#4b5563", linewidth=1.8, linestyle="--", label="Q min")
        if np.any(reward_sum > 0):
            ax.scatter(
                times[reward_sum > 0],
                np.full(np.count_nonzero(reward_sum > 0), 1.06),
                color="#111827",
                s=24,
                marker="|",
                label="future reward in chunk",
            )

        for thumb in curve.thumbnails:
            ax.axvline(thumb["time_s"], color="#9ca3af", linewidth=0.8, alpha=0.45)

        ax.set_title(
            f"{curve.label} | {curve.dataset_name} | episode {curve.episode_index}",
            loc="left",
            fontsize=12,
        )
        ax.set_ylabel("critic value")
        ax.set_xlabel("time (s)")
        ax.set_ylim(-0.15, 1.15)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(loc="upper left", ncols=4, fontsize=9, frameon=False)

    fig.suptitle("LWD Critic Values on Held-out RoboTwin Episodes", fontsize=15)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    curves: list[EpisodeCurve],
    output_path: Path,
) -> None:
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(args.data_root),
        "episode_index": args.episode_index,
        "num_points": args.num_points,
        "batch_size": args.batch_size,
        "camera_key": args.camera_key,
        "curves": [
            {
                "label": curve.label,
                "dataset": curve.dataset_name,
                "num_records": len(curve.records),
                "value_quantile_first": curve.records[0]["value_quantile"],
                "value_quantile_last": curve.records[-1]["value_quantile"],
                "q_min_first": curve.records[0]["q_min"],
                "q_min_last": curve.records[-1]["q_min"],
            }
            for curve in curves
        ],
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = build_model(args, checkpoint_path, device)
    collator = build_collator(args)
    splits = parse_splits(args.split)

    curves = []
    for label, dataset_name in splits:
        dataset = LWDChunkDataset(
            dataset_path=str(args.data_root / dataset_name),
            action_horizon=int(args.action_horizon),
            norm_stats_path=str(args.data_root / "norm_stats.json"),
            use_quantile_norm=True,
            adapt_to_pi=True,
        )
        indices = episode_indices(dataset, args.episode_index, args.num_points)
        print(
            f"Evaluating {label}: {dataset_name}, episode={args.episode_index}, "
            f"points={len(indices)}"
        )
        curves.append(
            evaluate_episode(
                model=model,
                collator=collator,
                dataset=dataset,
                label=label,
                episode_index=args.episode_index,
                indices=indices,
                batch_size=args.batch_size,
                device=device,
                camera_key=args.camera_key,
                frame_count=args.frame_count,
            )
        )

    csv_path = output_dir / "robotwin_lwd_critic_episode_values.csv"
    png_path = output_dir / "robotwin_lwd_critic_episode_values.png"
    pdf_path = output_dir / "robotwin_lwd_critic_episode_values.pdf"
    summary_path = output_dir / "robotwin_lwd_critic_episode_values_summary.json"

    save_csv(curves, csv_path)
    plot_curves(curves, png_path, pdf_path)
    save_summary(
        args=args,
        checkpoint_path=checkpoint_path,
        curves=curves,
        output_path=summary_path,
    )

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved values: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
