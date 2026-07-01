# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.qgf_flow_policy.action_transform import QGFActionTransform
from rlinf.models.embodiment.qgf_flow_policy.flow_actor import QGFFlowActor
from rlinf.models.embodiment.qgf_flow_policy.obs_encoder import QGFObsEncoder
from rlinf.models.embodiment.qgf_flow_policy.q_critic import QGFQCritic
from rlinf.models.embodiment.qgf_flow_policy.value_model import QGFValueModel


@dataclass
class QGFFlowConfig:
    input_type: str = "state"
    image_size: list[int] = field(default_factory=lambda: [3, 128, 128])
    image_num: int = 1
    state_dim: int = 29
    action_dim: int = 4
    num_action_chunks: int = 1
    model_path: Optional[str] = ""
    encoder_config: dict[str, Any] = field(default_factory=dict)
    stats_path: Optional[str] = ""
    state_normalization: str = "none"
    image_normalization: str = "imagenet"
    action_space: str = "identity"
    robotwin_delta_scale: float | list[float] = 0.08
    robotwin_gripper_indices: Optional[list[int]] = None

    feature_dim: int = 256
    state_latent_dim: int = 64
    add_q_head: bool = True
    add_value_head: bool = False
    num_q_heads: int = 2
    q_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    value_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])

    denoising_steps: int = 4
    d_model: int = 128
    n_head: int = 4
    n_layers: int = 2
    flow_actor_type: str = "QGFFlowTransformer"
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    noise_std_head: bool = False
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    noise_std_train: float = 0.3
    noise_std_rollout: float = 0.02

    is_lora: bool = False
    lora_rank: int = 32
    precision: str = "32"
    sharding_strategy: str = "no_shard"

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._update_info()

    def _update_info(self):
        if self.input_type == "mixed":
            assert self.model_path is not None
            assert "ckpt_name" in self.encoder_config
            ckpt_path = os.path.join(self.model_path, self.encoder_config["ckpt_name"])
            assert os.path.exists(ckpt_path), (
                f"Pretrained encoder weights not found at {ckpt_path}"
            )
            self.encoder_config["ckpt_path"] = ckpt_path


class QGFFlowPolicy(nn.Module, BasePolicy):
    """RLinf-native QGF flow policy with SAC Q and IQL value interfaces."""

    def __init__(self, cfg: QGFFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.flat_action_dim = cfg.action_dim * cfg.num_action_chunks
        self.cuda_graph_manager = None

        self.obs_encoder = QGFObsEncoder(
            input_type=cfg.input_type,
            state_dim=cfg.state_dim,
            feature_dim=cfg.feature_dim,
            image_size=cfg.image_size,
            image_num=cfg.image_num,
            state_latent_dim=cfg.state_latent_dim,
            encoder_config=cfg.encoder_config,
            stats_path=cfg.stats_path,
            state_normalization=cfg.state_normalization,
            image_normalization=cfg.image_normalization,
        )
        self.action_transform = QGFActionTransform(
            action_space=cfg.action_space,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            stats_path=cfg.stats_path,
            robotwin_delta_scale=cfg.robotwin_delta_scale,
            robotwin_gripper_indices=cfg.robotwin_gripper_indices,
        )
        self.flow_actor = QGFFlowActor(
            feature_dim=self.obs_encoder.out_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            n_layers=cfg.n_layers,
            denoising_steps=cfg.denoising_steps,
            flow_actor_type=cfg.flow_actor_type,
            use_batch_norm=cfg.use_batch_norm,
            batch_norm_momentum=cfg.batch_norm_momentum,
            noise_std_head=cfg.noise_std_head,
            log_std_min_train=cfg.log_std_min_train,
            log_std_max_train=cfg.log_std_max_train,
            log_std_min_rollout=cfg.log_std_min_rollout,
            log_std_max_rollout=cfg.log_std_max_rollout,
            noise_std_train=cfg.noise_std_train,
            noise_std_rollout=cfg.noise_std_rollout,
        )
        if cfg.add_q_head:
            self.q_head = QGFQCritic(
                feature_dim=self.obs_encoder.out_dim,
                flat_action_dim=self.flat_action_dim,
                hidden_dims=cfg.q_hidden_dims,
                num_q_heads=cfg.num_q_heads,
            )
        if cfg.add_value_head:
            self.value_head = QGFValueModel(
                feature_dim=self.obs_encoder.out_dim,
                hidden_dims=cfg.value_hidden_dims,
            )

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        return self.obs_encoder.preprocess(env_obs, next(self.parameters()).device)

    def _raw_states(self, env_obs):
        return env_obs["states"].to(next(self.parameters()).device).float()

    def encode_obs(self, obs, detach_encoder: bool = False):
        return self.obs_encoder(obs, detach_visual=detach_encoder)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type in {
            ForwardType.IQL_ACTOR,
            ForwardType.IQL_CRITIC,
            ForwardType.IQL_VALUE,
        }:
            return self.iql_forward(forward_type=forward_type, **kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError

    def sac_forward(self, obs, **kwargs):
        features = self.encode_obs(obs)
        action, log_prob = self.flow_actor(features, train=True)
        return action, log_prob, features

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        features = shared_feature
        if features is None:
            features = self.encode_obs(obs, detach_encoder)
        elif detach_encoder:
            features = features.detach()
        return self.q_head(features, actions)

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.get_q_values(obs, actions, shared_feature, detach_encoder)

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = self._obs_from_forward_inputs(forward_inputs)
        features = self.encode_obs(self.preprocess_env_obs(obs))
        action, log_prob = self.flow_actor(features, train=False)
        output = {"action": action}
        if compute_logprobs:
            output["logprobs"] = log_prob
        if compute_entropy:
            output["entropy"] = -log_prob
        if compute_values:
            if not hasattr(self, "value_head"):
                raise NotImplementedError("QGFFlowPolicy was built without value_head.")
            output["values"] = self.value_head(features).unsqueeze(-1)
        return output

    def sft_forward(self, data, **kwargs):
        obs = self._obs_from_forward_inputs(data)
        raw_states = self._raw_states(obs)
        target_action = self._resolve_model_action(data, raw_states)
        features = self.encode_obs(self.preprocess_env_obs(obs))
        return self.flow_actor.flow_matching_loss(features, target_action)

    def prepare_dagger_sft_batch(self, batch):
        data = {}
        if "model_action" in batch:
            data["model_action"] = batch["model_action"]
        if "env_action" in batch:
            data["env_action"] = batch["env_action"]
        if "action" in batch:
            data["action"] = batch["action"]
        for key in ("states", "main_images", "wrist_images", "extra_view_images"):
            if key in batch:
                data[key] = batch[key]
        return data

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        obs = self.preprocess_env_obs(env_obs)
        raw_states = self._raw_states(env_obs)
        features = self.encode_obs(obs)
        model_action, log_prob = self.flow_actor(features, train=(mode == "train"))
        env_actions = self.action_transform.decode(model_action, raw_states)

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(features).unsqueeze(-1)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {
            "action": model_action,
            "model_action": model_action,
            "env_action": env_actions.reshape(env_actions.shape[0], -1),
        }
        if return_obs:
            for key in ("states", "main_images", "wrist_images", "extra_view_images"):
                if key in env_obs:
                    forward_inputs[key] = env_obs[key]

        return env_actions, {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

    def iql_forward(self, forward_type, **kwargs):
        observations = kwargs.get("observations")
        if observations is None:
            obs = kwargs.get("obs")
            features = self.encode_obs(obs) if isinstance(obs, dict) else None
        elif isinstance(observations, dict):
            features = self.encode_obs(self.preprocess_env_obs(observations))
        else:
            features = self.obs_encoder.encoder(observations)

        if forward_type == ForwardType.IQL_ACTOR:
            actions = kwargs.get("actions")
            if actions is None:
                action, _ = self.flow_actor(features, train=True)
                return action
            loss = self.flow_actor.flow_matching_loss(
                features,
                actions.reshape(actions.shape[0], -1),
            )
            return -loss.mean(dim=-1)
        if forward_type == ForwardType.IQL_CRITIC:
            return self.q_head(features, kwargs["actions"]).min(dim=-1).values
        if forward_type == ForwardType.IQL_VALUE:
            if not hasattr(self, "value_head"):
                raise NotImplementedError("QGFFlowPolicy was built without value_head.")
            return self.value_head(features)
        raise NotImplementedError

    def _resolve_model_action(self, data, raw_states):
        if "model_action" in data:
            action = data["model_action"]
        elif "env_action" in data:
            action = self.action_transform.encode(data["env_action"], raw_states)
        else:
            action = data["action"]
        return action.reshape(action.shape[0], -1).float()

    def _obs_from_forward_inputs(self, inputs):
        obs = {"states": inputs["states"]}
        for key in ("main_images", "wrist_images", "extra_view_images"):
            if key in inputs:
                obs[key] = inputs[key]
        return obs
