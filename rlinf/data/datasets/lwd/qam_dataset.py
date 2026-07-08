# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Data collation for LWD QAM policy extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.data.data_collator import DataCollatorMixin

from rlinf.models.embodiment.value_model.data_collator import stack_tensors

from .chunk_dataset import LWDChunkDataCollator


@dataclass
class LWDQAMDataCollator(DataCollatorMixin):
    """Collate LWD replay samples into critic and OpenPI policy views."""

    critic_collator: LWDChunkDataCollator
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        critic_batch = self.critic_collator.torch_call(examples)
        policy_inputs = {
            "images": stack_tensors([ex["images"] for ex in examples]),
            "state": torch.stack(
                [
                    ex["state"]
                    if isinstance(ex["state"], torch.Tensor)
                    else torch.as_tensor(ex["state"])
                    for ex in examples
                ]
            ).float(),
            "prompt": [ex["task_prompt"] for ex in examples],
        }
        return {
            "critic_observation": critic_batch["observation"],
            "policy_inputs": policy_inputs,
            "action_chunk": critic_batch["action_chunk"],
            "success": critic_batch["success"],
            "episode_id": critic_batch["episode_id"],
            "frame_idx": critic_batch["frame_idx"],
            "source": critic_batch["source"],
            "prompt": critic_batch["prompt"],
        }


__all__ = ["LWDQAMDataCollator"]
