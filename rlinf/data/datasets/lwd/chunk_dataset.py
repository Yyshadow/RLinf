# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LeRobot chunk-transition dataset for LWD-style critics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

from rlinf.data.datasets.recap.common import BaseDataLoaderImpl, ReCapMixtureDataset
from rlinf.data.datasets.recap.utils import (
    decode_image_struct_batch,
    load_task_descriptions,
)
from rlinf.models.embodiment.openpi.policies.aloha_policy import (
    _decode_state,
    _encode_actions_inv,
)
from rlinf.models.embodiment.value_model.data_collator import stack_tensors

logger = logging.getLogger(__name__)


PI05_DELTA_ACTION_MASK = np.asarray(
    [True] * 6 + [False] + [True] * 6 + [False],
    dtype=bool,
)
PI05_STATE_BINS = 256


def _as_python_int(value: Any) -> int:
    return int(value.item() if isinstance(value, torch.Tensor) else value)


def _as_python_bool(value: Any) -> bool:
    return bool(value.item() if isinstance(value, torch.Tensor) else value)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _load_norm_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    data = data.get("norm_stats", data)
    return {
        key: {
            stat_key: np.asarray(stat_value, dtype=np.float32)
            for stat_key, stat_value in value.items()
            if stat_value is not None
        }
        for key, value in data.items()
    }


class LWDChunkDataset(Dataset):
    """Build pi0.5-aligned chunk transitions from RoboTwin LeRobot data."""

    def __init__(
        self,
        dataset_path: str,
        action_horizon: int = 50,
        norm_stats_path: str | None = None,
        use_quantile_norm: bool = True,
        adapt_to_pi: bool = True,
        default_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path).absolute()
        self.action_horizon = int(action_horizon)
        self.use_quantile_norm = bool(use_quantile_norm)
        self.adapt_to_pi = bool(adapt_to_pi)
        self.default_prompt = default_prompt or "perform the task"
        self.max_samples = max_samples
        if norm_stats_path is None:
            raise ValueError(
                "LWDChunkDataset requires norm_stats_path for pi0.5 "
                "state/action alignment."
            )
        self.norm_stats = _load_norm_stats(Path(norm_stats_path))

        self.metadata = LeRobotDatasetMetadata(
            self.dataset_path.name,
            root=self.dataset_path,
        )
        self.fps = int(self.metadata.fps)
        delta_timestamps = {
            "action": [i / self.fps for i in range(self.action_horizon)],
            "reward": [i / self.fps for i in range(self.action_horizon)],
        }
        self.dataset = LeRobotDataset(
            self.dataset_path.name,
            root=self.dataset_path,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )
        self.dataset.hf_dataset.set_transform(decode_image_struct_batch)
        self.tasks = load_task_descriptions(self.dataset_path)

        n = len(self.dataset)
        logger.info(
            "LWDChunkDataset: %s, samples=%d, action_horizon=%d, norm_stats=%s",
            self.dataset_path,
            min(n, max_samples or n),
            self.action_horizon,
            norm_stats_path,
        )

    def __len__(self) -> int:
        n = len(self.dataset)
        return min(n, self.max_samples) if self.max_samples else n

    @staticmethod
    def _extract_images(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = {}
        prefix = "observation.images."
        for key, value in sample.items():
            if key.startswith(prefix):
                images[key.removeprefix(prefix)] = value
        return images

    def _prompt_for_sample(self, sample: dict[str, Any]) -> str:
        if "task" in sample and sample["task"]:
            return str(sample["task"])
        if "task_index" in sample:
            task_index = _as_python_int(sample["task_index"])
            if task_index in self.tasks:
                return self.tasks[task_index]
        return self.default_prompt

    def _normalize(self, key: str, value: np.ndarray) -> np.ndarray:
        stats = self.norm_stats.get(key)
        if stats is None and key == "actions":
            stats = self.norm_stats.get("action")
        if stats is None:
            raise KeyError(f"norm_stats must contain '{key}' statistics.")

        if self.use_quantile_norm:
            low = stats["q01"][..., : value.shape[-1]]
            high = stats["q99"][..., : value.shape[-1]]
            return (value - low) / (high - low + 1e-6) * 2.0 - 1.0

        mean = stats["mean"][..., : value.shape[-1]]
        std = stats["std"][..., : value.shape[-1]]
        return (value - mean) / (std + 1e-6)

    def _pi05_state_prompt(self, prompt: str, state: np.ndarray) -> str:
        bins = np.linspace(-1, 1, PI05_STATE_BINS + 1)[:-1]
        discretized_state = np.digitize(state, bins=bins) - 1
        discretized_state = np.clip(discretized_state, 0, PI05_STATE_BINS - 1)
        state_str = " ".join(map(str, discretized_state.astype(np.int64).tolist()))
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        return f"Task: {cleaned_prompt}, State: {state_str};\nAction: "

    def _pi05_state(self, state: Any) -> np.ndarray:
        state_pi = _decode_state(_to_numpy(state), adapt_to_pi=self.adapt_to_pi)
        return self._normalize("state", state_pi)

    def _pi05_state_and_actions(
        self,
        state: Any,
        action_chunk: Any,
    ) -> tuple[np.ndarray, torch.Tensor]:
        state_pi = _decode_state(_to_numpy(state), adapt_to_pi=self.adapt_to_pi)
        actions_pi = _encode_actions_inv(
            _to_numpy(action_chunk),
            adapt_to_pi=self.adapt_to_pi,
        )
        actions_pi[..., PI05_DELTA_ACTION_MASK] -= state_pi[PI05_DELTA_ACTION_MASK]

        state_norm = self._normalize("state", state_pi)
        action_norm = self._normalize("actions", actions_pi)
        return state_norm, torch.from_numpy(action_norm.astype(np.float32))

    def _episode_end_index(self, episode_index: int) -> int:
        return int(self.dataset.episode_data_index["to"][episode_index].item())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        episode_index = _as_python_int(sample["episode_index"])
        episode_end = self._episode_end_index(episode_index)
        next_idx = min(idx + self.action_horizon, episode_end - 1)
        next_sample = self.dataset[next_idx]

        reward_chunk = sample["reward"].float()
        reward_pad = sample.get("reward_is_pad")
        if reward_pad is not None:
            reward_chunk = reward_chunk.masked_fill(reward_pad.bool(), 0.0)

        action_pad = sample.get("action_is_pad")
        done = _as_python_bool(sample["done"]) or _as_python_bool(next_sample["done"])
        if action_pad is not None:
            done = done or bool(action_pad.bool().any().item())

        state_norm, action_chunk = self._pi05_state_and_actions(
            sample["observation.state"],
            sample["action"],
        )
        next_state_norm = self._pi05_state(next_sample["observation.state"])
        prompt = self._prompt_for_sample(sample)

        return {
            "images": self._extract_images(sample),
            "next_images": self._extract_images(next_sample),
            "prompt": self._pi05_state_prompt(prompt, state_norm),
            "next_prompt": self._pi05_state_prompt(prompt, next_state_norm),
            "action_chunk": action_chunk,
            "reward_chunk": reward_chunk,
            "done": done,
            "success": _as_python_bool(next_sample.get("success", False)),
            "episode_id": episode_index,
            "frame_idx": _as_python_int(sample["frame_index"]),
            "source": self.dataset_path.name,
        }


@dataclass
class LWDChunkDataCollator(DataCollatorMixin):
    """Collate raw LeRobot chunk samples into LWDCriticModel inputs."""

    processor: Any
    max_length: int = 200
    return_tensors: str = "pt"
    train: bool = True

    def _tokenize_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        batch_tokens = []
        batch_masks = []
        for prompt in prompts:
            tokens = self.processor.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )
            if len(tokens) < self.max_length:
                pad = self.max_length - len(tokens)
                mask = [True] * len(tokens) + [False] * pad
                tokens = tokens + [pad_token_id] * pad
            else:
                tokens = tokens[: self.max_length]
                mask = [True] * self.max_length
            batch_tokens.append(tokens)
            batch_masks.append(mask)
        return {
            "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.bool),
        }

    def _build_observation(
        self,
        images_list: list[dict[str, torch.Tensor]],
        prompts: list[str],
    ) -> dict[str, Any]:
        images = stack_tensors(images_list)
        processed_img = self.processor.image_processor(
            images=images,
            image_masks={},
            return_tensors="pt",
            train=self.train,
        )
        processed_txt = self._tokenize_prompts(prompts)

        observation = {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": processed_txt["input_ids"],
            "tokenized_prompt_mask": processed_txt["attention_mask"].bool(),
        }
        return observation

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [ex["prompt"] for ex in examples]
        next_prompts = [ex["next_prompt"] for ex in examples]

        observation = self._build_observation(
            [ex["images"] for ex in examples],
            prompts,
        )
        next_observation = self._build_observation(
            [ex["next_images"] for ex in examples],
            next_prompts,
        )

        return {
            "observation": observation,
            "next_observation": next_observation,
            "action_chunk": torch.stack([ex["action_chunk"] for ex in examples]).float(),
            "reward_chunk": torch.stack([ex["reward_chunk"] for ex in examples]).float(),
            "done": torch.tensor([ex["done"] for ex in examples], dtype=torch.bool),
            "success": torch.tensor(
                [ex["success"] for ex in examples],
                dtype=torch.bool,
            ),
            "episode_id": torch.tensor(
                [ex["episode_id"] for ex in examples],
                dtype=torch.long,
            ),
            "frame_idx": torch.tensor(
                [ex["frame_idx"] for ex in examples],
                dtype=torch.long,
            ),
            "source": [ex["source"] for ex in examples],
            "prompt": prompts,
        }


class LWDDataLoaderImpl(BaseDataLoaderImpl):
    """Lightweight wrapper that yields LWD critic batches."""

    def __iter__(self):
        yield from self._data_loader


class LWDMixtureDataset(ReCapMixtureDataset):
    """Weighted mixture of LWD chunk datasets."""

    mixture_name = "LWDMixtureDataset"


__all__ = [
    "LWDChunkDataCollator",
    "LWDChunkDataset",
    "LWDDataLoaderImpl",
    "LWDMixtureDataset",
]
