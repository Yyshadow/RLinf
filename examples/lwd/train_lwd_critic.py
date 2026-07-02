# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Single-GPU trainer for the LWD-style critic smoke path."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from rlinf.data.datasets.lwd import LWDChunkDataCollator, LWDChunkDataset
from rlinf.models import get_model
from rlinf.models.embodiment.lwd_critic.lwd_loss import (
    compute_lwd_losses,
    update_ema_target,
)
from rlinf.models.embodiment.value_model.processing import ValueProcessor

logger = logging.getLogger(__name__)


def _build_dataset(cfg: DictConfig):
    datasets = []
    weights = []
    for entry in cfg.data.train_data_paths:
        dataset = LWDChunkDataset(
            dataset_path=entry.dataset_path,
            action_horizon=cfg.data.action_horizon,
            max_samples=cfg.data.get("max_samples", None),
        )
        datasets.append(dataset)
        weights.extend([float(entry.get("weight", 1.0))] * len(dataset))
    return ConcatDataset(datasets), torch.as_tensor(weights, dtype=torch.double)


def _move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return batch
    return batch


@hydra.main(version_base=None, config_path="config", config_name="robotwin_lwd_critic")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_path,
        add_bos_token=True,
        local_files_only=True,
    )
    processor = ValueProcessor(
        tokenizer=tokenizer,
        max_token_len=cfg.model.max_token_len,
        image_keys=tuple(cfg.data.image_keys),
    )

    dataset, sample_weights = _build_dataset(cfg)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    collator = LWDChunkDataCollator(
        processor=processor,
        max_length=cfg.model.max_token_len,
        train=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    model = get_model(cfg.model).to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.requires_grad_(False)
    target_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    output_dir = Path(cfg.runner.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)
    for step in range(1, int(cfg.runner.max_steps) + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = _move_to_device(batch, device)
        model.train()
        loss_out = compute_lwd_losses(
            model=model,
            target_model=target_model,
            batch=batch,
            gamma=float(cfg.optim.gamma),
            value_loss_weight=float(cfg.optim.value_loss_weight),
            q_loss_weight=float(cfg.optim.q_loss_weight),
        )

        optimizer.zero_grad(set_to_none=True)
        loss_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        optimizer.step()
        update_ema_target(model, target_model, tau=float(cfg.optim.ema_tau))

        if step % int(cfg.runner.log_interval) == 0 or step == 1:
            logger.info(
                "step=%d loss=%.4f q_loss=%.4f value_loss=%.4f "
                "target_q=%.4f q_min=%.4f reward_sum=%.4f",
                step,
                loss_out.loss.item(),
                loss_out.q_loss.item(),
                loss_out.value_loss.item(),
                loss_out.target_q.mean().item(),
                loss_out.q_min.mean().item(),
                loss_out.reward_sum.mean().item(),
            )

        if step % int(cfg.runner.save_interval) == 0:
            ckpt_path = output_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "target_model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "step": step,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
