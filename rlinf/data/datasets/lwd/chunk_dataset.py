# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LeRobot chunk-transition dataset for LWD-style critics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

from rlinf.data.datasets.recap.utils import (
    decode_image_struct_batch,
    load_task_descriptions,
)
from rlinf.models.embodiment.value_model.data_collator import stack_tensors

logger = logging.getLogger(__name__)


def _as_python_int(value: Any) -> int:
    return int(value.item() if isinstance(value, torch.Tensor) else value)


def _as_python_bool(value: Any) -> bool:
    return bool(value.item() if isinstance(value, torch.Tensor) else value)


class LWDChunkDataset(Dataset):
    """Build `(s_t, a_t:t+H, r_t:t+H, s_t+H, done)` from LeRobot data."""

    def __init__(
        self,
        dataset_path: str,
        action_horizon: int = 10,
        default_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path).absolute()
        self.action_horizon = int(action_horizon)
        self.default_prompt = default_prompt or "perform the task"
        self.max_samples = max_samples

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
            "LWDChunkDataset: %s, samples=%d, action_horizon=%d",
            self.dataset_path,
            min(n, max_samples or n),
            self.action_horizon,
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

        return {
            "images": self._extract_images(sample),
            "next_images": self._extract_images(next_sample),
            "state": sample.get("observation.state"),
            "next_state": next_sample.get("observation.state"),
            "prompt": self._prompt_for_sample(sample),
            "action_chunk": sample["action"].float(),
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

    def _build_observation(
        self,
        images_list: list[dict[str, torch.Tensor]],
        prompts: list[str],
        states: list[Any],
    ) -> dict[str, Any]:
        images = stack_tensors(images_list)
        processed_img = self.processor.image_processor(
            images=images,
            image_masks={},
            return_tensors="pt",
            train=self.train,
        )
        processed_txt = self.processor.process_text(
            prompts=prompts,
            max_length=self.max_length,
            return_tensors="pt",
        )

        observation = {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": processed_txt["input_ids"],
            "tokenized_prompt_mask": processed_txt["attention_mask"].bool(),
        }

        if states[0] is not None:
            observation["state"] = torch.stack(
                [
                    state if isinstance(state, torch.Tensor) else torch.tensor(state)
                    for state in states
                ]
            ).float()
        return observation

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [ex["prompt"] for ex in examples]

        observation = self._build_observation(
            [ex["images"] for ex in examples],
            prompts,
            [ex.get("state") for ex in examples],
        )
        next_observation = self._build_observation(
            [ex["next_images"] for ex in examples],
            prompts,
            [ex.get("next_state") for ex in examples],
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


__all__ = ["LWDChunkDataCollator", "LWDChunkDataset"]
