"""
src/core/ppo_agent.py
──────────────────────
Proximal Policy Optimization (PPO) Agent cho Deep RL.

Thay thế Genetic Algorithm bằng gradient-based optimization:
  - Actor-Critic architecture (shared backbone)
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Rollout buffer cho on-policy training

Kiến trúc mặc định (sensor mode):
  obs(5) -> FC(64) -> FC(64) -> Actor head(2) + Critic head(1)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Tuple, Dict, Optional

from src.core.cnn_encoder import CNNActorCritic


# ── Actor-Critic Network ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared backbone + 2 heads:
      - Actor: outputs mean of continuous actions [turn, engine]
      - Critic: outputs scalar value estimate V(s)
    """

    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 2,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64]

        # Shared backbone
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.Tanh())
            prev_size = h
        self.backbone = nn.Sequential(*layers)

        # Actor head: mean of actions
        self.actor_mean = nn.Linear(prev_size, action_dim)
        # Learnable log_std (per-action)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: value estimate
        self.critic = nn.Linear(prev_size, 1)

        # Khởi tạo weights nhỏ cho ổn định
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor output nhỏ hơn → hành vi ban đầu gần uniform
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass qua backbone."""
        features = self.backbone(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action từ policy distribution.

        Returns:
            action: sampled action (clipped to [-1, 1])
            log_prob: log probability of the action
            value: V(s) estimate
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)  # bound to [-1, 1]

        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy (dùng trong PPO update).

        Returns:
            log_probs, values, entropy
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy


# ── Rollout Buffer ────────────────────────────────────────────────────

class RolloutBuffer:
    """
    On-policy buffer lưu trajectories cho PPO update.
    Reset sau mỗi lần update.
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def get_tensors(self, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Convert buffer sang tensors cho training."""
        device = device or torch.device("cpu")
        return {
            "states": torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.float32, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
        }


# ── PPO Agent ─────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO Agent quản lý toàn bộ training pipeline.

    Args:
        obs_dim: Số chiều observation (5 cho sensor mode)
        action_dim: Số chiều action (2: turn, engine)
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clipping range
        epochs: Số epoch update mỗi batch
        batch_size: Mini-batch size
        entropy_coeff: Hệ số entropy bonus
        value_coeff: Hệ số value loss
    """

    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 2,
        hidden_sizes: List[int] = None,
        is_cnn: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cpu")
        self.is_cnn = is_cnn

        if self.is_cnn:
            self.network = CNNActorCritic(
                in_channels=obs_dim,  # obs_dim ở đây hiểu là in_channels (ví dụ: 4)
                action_dim=action_dim,
                feature_dim=512,
            ).to(self.device)
        else:
            self.network = ActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes or [64, 64],
            ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        # Training stats
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []

    # ── Action selection ──────────────────────────────────────────────

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Chọn action từ policy.

        Args:
            obs: Observation vector (sensor readings)
            deterministic: True cho evaluation (không exploration)

        Returns:
            action (np.ndarray), log_prob (float), value (float)
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.network.get_action(obs_t, deterministic)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # ── GAE computation ───────────────────────────────────────────────

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation.

        Returns:
            advantages: A_t (chuẩn hóa)
            returns: R_t = A_t + V_t
        """
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Chuẩn hóa advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ── PPO Update ────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        PPO update trên buffer hiện tại.

        Returns:
            Dict chứa loss stats
        """
        if len(self.buffer) == 0:
            return {}

        data = self.buffer.get_tensors(self.device)
        advantages, returns = self.compute_gae(
            data["rewards"], data["values"], data["dones"]
        )

        old_log_probs = data["log_probs"]
        states = data["states"]
        actions = data["actions"]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            # Mini-batch random shuffle
            indices = torch.randperm(len(self.buffer))
            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Evaluate under current policy
                new_log_probs, new_values, entropy = self.network.evaluate(
                    b_states, b_actions
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, b_returns)

                # Entropy bonus (khuyến khích exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

        self.buffer.clear()

        stats = {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }
        self.policy_losses.append(stats["policy_loss"])
        self.value_losses.append(stats["value_loss"])

        return stats

    # ── Checkpoint ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Lưu model weights."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
        }, path)

    def load(self, path: str) -> None:
        """Tải model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint.get("episode_count", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])

    # ── Logging ───────────────────────────────────────────────────────

    def log_episode(self, episode_reward: float) -> str:
        """Format log string cho 1 episode."""
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)

        recent = self.episode_rewards[-100:]
        avg = sum(recent) / len(recent)
        best = max(self.episode_rewards)

        return (
            f"Ep {self.episode_count:4d} | "
            f"reward={episode_reward:8.1f} | avg100={avg:8.1f} | "
            f"best={best:8.1f} | steps={self.total_steps}"
        )

    def __repr__(self) -> str:
        return (
            f"PPOAgent(episodes={self.episode_count}, "
            f"steps={self.total_steps}, "
            f"lr={self.optimizer.param_groups[0]['lr']})"
        )
