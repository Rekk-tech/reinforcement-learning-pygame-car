"""
src/simulation/reward.py
─────────────────────────
Shaped Reward Functions cho Deep Reinforcement Learning.

Thay vì chỉ dùng quãng đường đơn giản (fitness = tổng speed),
shaped reward cung cấp tín hiệu phong phú hơn để agent học nhanh hơn.

Components:
  +speed_bonus     : Thưởng duy trì tốc độ cao
  -wall_penalty    : Phạt khi va chạm tường
  +checkpoint_bonus: Thưởng khi qua checkpoint mới
  -stuck_penalty   : Phạt đứng im quá lâu
  +alive_bonus     : Thưởng nhỏ mỗi frame sống sót
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.car import Car
    from src.simulation.track import Track


@dataclass
class RewardConfig:
    """Tham số shaped reward — có thể override từ config.yaml."""
    speed_bonus_weight: float = 0.1      # Hệ số thưởng tốc độ
    wall_penalty: float = -10.0          # Phạt va chạm tường
    checkpoint_bonus: float = 50.0       # Thưởng qua checkpoint
    stuck_penalty: float = -0.5          # Phạt mỗi frame stuck
    alive_bonus: float = 0.05           # Thưởng nhỏ mỗi frame sống
    max_speed_ref: float = 3.0           # Tốc độ tham chiếu cho chuẩn hóa


class RewardCalculator:
    """
    Tính shaped reward cho mỗi frame của agent.

    Usage:
        calc = RewardCalculator(config)
        reward = calc.compute(car, track, done)
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self._last_checkpoint_idx: int = -1

    def reset(self) -> None:
        """Reset state khi bắt đầu episode mới."""
        self._last_checkpoint_idx = -1

    def compute(
        self,
        car: "Car",
        track: "Track",
        done: bool,
        checkpoint_idx: int = -1,
    ) -> float:
        """
        Tính reward cho 1 frame.

        Args:
            car: Agent xe hiện tại
            track: Sa hình đường đua
            done: True nếu episode kết thúc (va chạm hoặc hết thời gian)
            checkpoint_idx: Index checkpoint hiện tại (-1 nếu chưa qua)

        Returns:
            float: Shaped reward tổng hợp
        """
        cfg = self.config
        reward = 0.0

        if done:
            # Phạt va chạm tường
            reward += cfg.wall_penalty
            return reward

        # 1. Alive bonus — khuyến khích sống sót
        reward += cfg.alive_bonus

        # 2. Speed bonus — thưởng tốc độ cao (chuẩn hóa theo max_speed)
        speed_ratio = car.speed / max(cfg.max_speed_ref, 0.01)
        reward += cfg.speed_bonus_weight * speed_ratio

        # 3. Checkpoint bonus — thưởng khi qua checkpoint mới
        if checkpoint_idx > self._last_checkpoint_idx:
            reward += cfg.checkpoint_bonus
            self._last_checkpoint_idx = checkpoint_idx

        # 4. Stuck penalty — phạt nếu tốc độ quá thấp
        if car.speed < 0.1:
            reward += cfg.stuck_penalty

        return reward

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardCalculator":
        """Khởi tạo từ dict config (section 'reward' trong config.yaml)."""
        rc = RewardConfig(
            speed_bonus_weight=cfg.get("speed_bonus_weight", 0.1),
            wall_penalty=cfg.get("wall_penalty", -10.0),
            checkpoint_bonus=cfg.get("checkpoint_bonus", 50.0),
            stuck_penalty=cfg.get("stuck_penalty", -0.5),
            alive_bonus=cfg.get("alive_bonus", 0.05),
            max_speed_ref=cfg.get("max_speed_ref", 3.0),
        )
        return cls(rc)
