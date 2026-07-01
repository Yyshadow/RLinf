# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, layer_init, make_mlp


class QGFObsEncoder(nn.Module):
    """Encode RLinf embodied observations into a single policy feature."""

    def __init__(
        self,
        input_type: str,
        state_dim: int,
        feature_dim: int,
        image_size: list[int],
        image_num: int,
        state_latent_dim: int,
        encoder_config: dict,
        stats_path: str | None,
        state_normalization: str,
        image_normalization: str,
    ):
        super().__init__()
        self.input_type = input_type
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.image_num = image_num
        self.state_normalization = state_normalization
        self.image_normalization = image_normalization

        stats = torch.load(stats_path, map_location="cpu") if stats_path else {}
        state_mean = stats.get("state_mean", torch.zeros(state_dim))
        state_std = stats.get("state_std", torch.ones(state_dim))
        state_mean = torch.as_tensor(state_mean, dtype=torch.float32).flatten()
        state_std = torch.as_tensor(state_std, dtype=torch.float32).flatten()
        if state_mean.numel() != state_dim:
            state_mean = torch.zeros(state_dim)
        if state_std.numel() != state_dim:
            state_std = torch.ones(state_dim)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(1e-6))
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        )

        if input_type == "state":
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(state_dim, feature_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(feature_dim, feature_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(feature_dim, feature_dim)),
                nn.Tanh(),
            )
            self.out_dim = feature_dim
            return

        if input_type != "mixed":
            raise NotImplementedError(f"Unsupported qgf_flow_policy input_type={input_type}")

        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        sample_x = torch.randn(1, *image_size)
        for _ in range(image_num):
            encoder = ResNetEncoder(sample_x, out_dim=256, encoder_cfg=encoder_config)
            self.encoders.append(encoder)
            encoder_out_dim += encoder.out_dim

        self.state_proj = nn.Sequential(
            *make_mlp(
                in_channels=state_dim,
                mlp_channels=[state_latent_dim],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        self.state_proj._fsdp_wrap_name = "state_proj"
        init_mlp_weights(self.state_proj, nonlinearity="tanh")

        self.mix_proj = nn.Sequential(
            *make_mlp(
                in_channels=encoder_out_dim + state_latent_dim,
                mlp_channels=[feature_dim, feature_dim],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        init_mlp_weights(self.mix_proj, nonlinearity="tanh")
        self.out_dim = feature_dim

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        detach_visual: bool = False,
    ) -> torch.Tensor:
        if self.input_type == "state":
            return self.encoder(obs["states"])

        visual_features = []
        extra_view_images = obs.get("extra_view_images", None)
        wrist_images = obs.get("wrist_images", None)

        for img_id in range(self.image_num):
            if img_id == 0:
                images = obs["main_images"]
            elif extra_view_images is not None:
                images = extra_view_images[:, img_id - 1]
            else:
                images = wrist_images[:, img_id - 1]

            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            feat = self.encoders[img_id](images)
            if detach_visual:
                feat = feat.detach()
            visual_features.append(feat)

        visual_feature = torch.cat(visual_features, dim=-1)
        state_feature = self.state_proj(obs["states"])
        return self.mix_proj(torch.cat([visual_feature, state_feature], dim=-1))

    def preprocess(self, env_obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
        states = env_obs["states"].to(device).float()
        if self.state_normalization == "zscore":
            states = (states - self.state_mean.to(device)) / self.state_std.to(device)
        obs = {"states": states}
        if self.input_type == "state":
            return obs

        obs["main_images"] = self._preprocess_image(env_obs["main_images"], device)
        if env_obs.get("extra_view_images", None) is not None:
            obs["extra_view_images"] = self._preprocess_image(
                env_obs["extra_view_images"], device
            )
        if env_obs.get("wrist_images", None) is not None:
            obs["wrist_images"] = self._preprocess_image(
                env_obs["wrist_images"], device
            )
        return obs

    def _preprocess_image(self, image: torch.Tensor, device: torch.device) -> torch.Tensor:
        image = image.to(device)
        if image.dtype.is_floating_point:
            image = image.float()
            if image.max() > 2.0:
                image = image / 255.0
        else:
            image = image.float() / 255.0
        if self.image_normalization == "imagenet":
            mean = self.img_mean.to(device)
            std = self.img_std.to(device)
            if image.dim() == 5:
                mean = mean.unsqueeze(1)
                std = std.unsqueeze(1)
            image = (image - mean) / std
        return image
