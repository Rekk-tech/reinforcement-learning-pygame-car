"""
src/core/cnn_encoder.py
────────────────────────
CNN Feature Extractor cho Computer Vision mode.

Nhan anh pixel (4, 84, 84) tu camera sensor va trich xuat
feature vector 512 chieu de dua vao Actor-Critic network.

Kien truc:
  Conv2d(4, 32, 8, stride=4) -> ReLU
  Conv2d(32, 64, 4, stride=2) -> ReLU
  Conv2d(64, 64, 3, stride=1) -> ReLU
  Flatten -> FC(3136, 512) -> ReLU
  -> output: 512-dim feature vector
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class CNNEncoder(nn.Module):
    """
    Convolutional feature extractor cho pixel observations.

    Input:  (batch, in_channels, 84, 84)
    Output: (batch, feature_dim) — default 512

    Args:
        in_channels: So kenh input (frame_stack * channels_per_frame)
        feature_dim: Kich thuoc feature vector output
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 512,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # 3 lop convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Tinh kich thuoc sau flatten (phu thuoc vao input size)
        self._conv_output_size = self._get_conv_output_size(in_channels, 84, 84)

        # Fully connected -> feature vector
        self.fc = nn.Sequential(
            nn.Linear(self._conv_output_size, feature_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _get_conv_output_size(self, c: int, h: int, w: int) -> int:
        """Tinh kich thuoc output cua conv layers."""
        dummy = torch.zeros(1, c, h, w)
        out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (batch, in_channels, 84, 84) — pixel observation

        Returns:
            (batch, feature_dim) — feature vector
        """
        conv_out = self.conv(obs)
        flat = conv_out.reshape(conv_out.size(0), -1)
        features = self.fc(flat)
        return features


class CNNActorCritic(nn.Module):
    """
    Actor-Critic network voi CNN encoder.

    Pipeline:
      pixel_obs (4, 84, 84) -> CNNEncoder -> FC(512) -> [Actor head, Critic head]

    Args:
        in_channels: So kenh input
        action_dim: So chieu action (2: turn, engine)
        feature_dim: Kich thuoc feature vector tu CNN
    """

    def __init__(
        self,
        in_channels: int = 4,
        action_dim: int = 2,
        feature_dim: int = 512,
    ):
        super().__init__()

        self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # Actor head
        self.actor_mean = nn.Linear(feature_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(feature_dim, 1)

        # Khoi tao weights nho
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action tu policy."""
        from torch.distributions import Normal

        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under current policy."""
        from torch.distributions import Normal

        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy
