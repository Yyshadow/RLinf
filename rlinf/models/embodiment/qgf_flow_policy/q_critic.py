# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.q_head import MultiQHead


class QGFQCritic(nn.Module):
    """Q ensemble used by SAC/RLPD and later Q-guided flow sampling."""

    def __init__(
        self,
        feature_dim: int,
        flat_action_dim: int,
        hidden_dims: list[int],
        num_q_heads: int,
    ):
        super().__init__()
        self.flat_action_dim = flat_action_dim
        self.q_head = MultiQHead(
            hidden_size=feature_dim,
            hidden_dims=hidden_dims,
            num_q_heads=num_q_heads,
            action_feature_dim=flat_action_dim,
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_head(features, actions.reshape(actions.shape[0], -1))
