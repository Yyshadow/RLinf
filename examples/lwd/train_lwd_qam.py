# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Entry point for LWD QAM policy extraction."""

from __future__ import annotations

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_lwd_qam_worker import FSDPLWDQAMWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_beat_block_hammer_lwd_qam_openpi_pi05",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPLWDQAMWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    runner = SFTRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
