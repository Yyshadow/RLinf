# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Entry point for RLinf-native LWD critic training."""

from __future__ import annotations

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_lwd_critic_worker import FSDPLWDCriticWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="robotwin_lwd_critic")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    critic_group = FSDPLWDCriticWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    runner = SFTRunner(cfg=cfg, actor=critic_group)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
