# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LWD-style critic model factory."""

import logging
import os

from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.value_model.checkpoint_utils import (
    load_state_dict_from_checkpoint,
)

from .lwd_critic_model import LWDCriticConfig, LWDCriticModel, LWDCriticOutput

logger = logging.getLogger(__name__)


def get_model(cfg: DictConfig, torch_dtype=None) -> LWDCriticModel:
    config = LWDCriticConfig()
    config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
    model = LWDCriticModel(config)

    model_path = getattr(config, "model_path", None)
    if model_path and os.path.exists(model_path):
        state_dict = load_state_dict_from_checkpoint(model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded LWD critic checkpoint from %s (missing=%d, unexpected=%d)",
            model_path,
            len(missing),
            len(unexpected),
        )

    return model


__all__ = [
    "get_model",
    "LWDCriticConfig",
    "LWDCriticModel",
    "LWDCriticOutput",
]
