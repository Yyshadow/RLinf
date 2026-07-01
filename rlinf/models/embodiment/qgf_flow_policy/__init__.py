# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=None):
    from rlinf.models.embodiment.qgf_flow_policy.qgf_flow_policy import (
        QGFFlowConfig,
        QGFFlowPolicy,
    )

    model_config = QGFFlowConfig()
    model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
    return QGFFlowPolicy(model_config)
