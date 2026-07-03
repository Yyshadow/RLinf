# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for pure LWD critic loss utilities."""

import torch

from rlinf.models.embodiment.lwd_critic.lwd_critic_model import LWDCriticModel
from rlinf.models.embodiment.lwd_critic.lwd_loss import discounted_chunk_sum


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
