# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.value_head import ValueHead


class QGFValueModel(nn.Module):
    """IQL-style state value V(s)."""

    def __init__(self, feature_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.value_head = ValueHead(
            input_dim=feature_dim,
            hidden_sizes=tuple(hidden_dims),
            activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features).squeeze(-1)
