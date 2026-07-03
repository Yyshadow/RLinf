# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for pure LWD critic loss utilities."""

import torch

from rlinf.models.embodiment.lwd_critic.lwd_critic_model import (
    ActionChunkEncoder,
    LWDCriticModel,
)
from rlinf.models.embodiment.lwd_critic.lwd_loss import discounted_chunk_sum
from rlinf.models.embodiment.modules.q_head import MultiQHead


def test_discounted_chunk_sum_batches_rewards() -> None:
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]])

    discounted = discounted_chunk_sum(rewards, gamma=0.5)

    torch.testing.assert_close(discounted, torch.tensor([1.75, 1.0]))


def test_scalar_targets_to_categorical_projects_and_clamps() -> None:
    atoms = torch.linspace(-1.0, 1.0, steps=5)
    targets = torch.tensor([-1.0, -0.25, 0.0, 2.0])

    projected = LWDCriticModel.scalar_targets_to_categorical(
        targets,
        atoms=atoms,
        v_min=-1.0,
        v_max=1.0,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(projected, expected)
    torch.testing.assert_close(projected.sum(dim=-1), torch.ones(targets.shape[0]))


def test_q_from_state_action_matches_head_dtype() -> None:
    critic = LWDCriticModel.__new__(LWDCriticModel)
    critic.action_encoder = ActionChunkEncoder(
        action_dim=4,
        hidden_dim=8,
        action_horizon=3,
    ).to(dtype=torch.float64)
    critic.q_head = MultiQHead(
        hidden_size=6,
        action_feature_dim=8,
        hidden_dims=[8],
        num_q_heads=2,
    ).to(dtype=torch.float64)

    state_features = torch.randn(2, 6, dtype=torch.float32)
    action_chunk = torch.randn(2, 3, 4, dtype=torch.float32)

    q_values, action_features = critic.q_from_state_action(
        state_features,
        action_chunk,
    )

    assert q_values.dtype == torch.float32
    assert action_features.dtype == torch.float64
    assert q_values.shape == (2, 2)
