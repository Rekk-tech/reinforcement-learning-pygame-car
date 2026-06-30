"""
src/simulation/car.py
──────────────────────
Agent xe ô tô trong môi trường sa hình.

Mỗi Car sở hữu:
  - Một NeuralNetwork (bộ não) nhận sensor input → quyết định lái
  - 5 tia cảm biến (raycast) phát ra từ đầu xe
  - Trạng thái vật lý: vị trí, góc, tốc độ
  - Fitness score: tổng quãng đường tích lũy

Vòng đời 1 xe trong 1 generation:
  observe() → forward(NN) → act() → update_physics() → check_collision()
  → [alive] tiếp tục  hoặc  [dead] chốt fitness
"""

from __future__ import annotations
import numpy as np
import math
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.neural_network import NeuralNetwork
    from src.simulation.track import Track


class Car:
    """
    Agent xe — kết hợp sensor, physics và NN inference.

    Attributes:
        x, y (float)     : Vị trí tâm xe (pixels)
        angle (float)    : Hướng hiện tại (radians)
        speed (float)    : Tốc độ hiện tại (pixel/frame)
        fitness (float)  : Điểm số tích lũy
        alive (bool)     : False khi va chạm tường
        nn (NeuralNetwork): Bộ não điều khiển
        trail (list)     : Lịch sử vị trí gần nhất
        is_elite (bool)  : Được đánh dấu là elite của gen trước
    """

    # ── Class-level physics constants (override từ config) ──────────
    MAX_SPEED: float = 3.0
    ACCELERATION: float = 0.15
    FRICTION: float = 0.02
    MAX_STEER: float = 0.08
    N_SENSORS: int = 5
    SENSOR_RANGE: float = 120.0
    SENSOR_ANGLES: List[float] = [-0.6, -0.3, 0.0, 0.3, 0.6]
    TRAIL_LENGTH: int = 40
    MAX_STUCK_FRAMES: int = 120

    def __init__(
        self,
        nn: "NeuralNetwork",
        start_x: float,
        start_y: float,
        start_angle: float = 0.0,
        is_elite: bool = False,
    ):
        self.nn = nn
        self.x = start_x
        self.y = start_y
        self.angle = start_angle
        self.speed = 0.0
        self.fitness = 0.0
        self.alive = True
        self.is_elite = is_elite

        self.trail: List[Tuple[float, float]] = []
        self._stuck_counter = 0
        self._frame_count = 0
        self.cp_idx = 0
        self.total_cp_passed = 0
        self.last_cp_passed_frame = 0
        self.target_checkpoint = None
        self.angle_diff = 0.0

        # Cache sensor readings cho renderer
        self.sensor_endpoints: List[Tuple[float, float]] = []
        self.sensor_values: List[float] = [1.0] * self.N_SENSORS

    # ── Sensor system ─────────────────────────────────────────────────

    def cast_sensors(self, track: "Track") -> List[float]:
        """
        Phát 5 tia raycast, trả về [0,1] chuẩn hóa.
        0 = tường ngay sát, 1 = không thấy tường trong sensor_range.
        """
        values = []
        self.sensor_endpoints = []

        for delta_angle in self.SENSOR_ANGLES:
            ray_angle = self.angle + delta_angle
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)

            dist = self.SENSOR_RANGE  # giả sử không thấy tường
            for d in range(5, int(self.SENSOR_RANGE), 3):
                px = self.x + cos_a * d
                py = self.y + sin_a * d
                if track.is_off_track(px, py):
                    dist = d
                    break

            normalized = dist / self.SENSOR_RANGE
            values.append(normalized)
            self.sensor_endpoints.append(
                (self.x + cos_a * dist, self.y + sin_a * dist)
            )

        # 8th input: Angle difference to next checkpoint
        if hasattr(track, "checkpoints") and len(track.checkpoints) > 0:
            cx, cy, _ = track.checkpoints[self.cp_idx]
            self.target_checkpoint = (cx, cy)
            target_angle = math.atan2(cy - self.y, cx - self.x)
            # Normalize angle diff to [-pi, pi]
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle_diff = angle_diff
            
            # Kill if going backwards (angle diff > 90 degrees)
            if abs(angle_diff) > math.pi / 2:
                self.alive = False
                
            values.append(angle_diff / math.pi) # [-1, 1]
        else:
            values.append(0.0)

        self.sensor_values = values
        return values

    # ── Physics update ────────────────────────────────────────────────

    def act(self, turn: float, engine: float) -> None:
        """
        Áp dụng output NN vào vật lý xe.

        Args:
            turn   (float): ∈ (-1, 1) — âm = rẽ trái, dương = rẽ phải
            engine (float): ∈ (-1, 1) — âm = phanh/lùi, dương = tăng ga
        """
        # Góc lái
        self.angle += turn * self.MAX_STEER * (self.speed / self.MAX_SPEED + 0.3)

        # Tốc độ
        self.speed += engine * self.ACCELERATION
        self.speed -= self.FRICTION  # ma sát
        self.speed = max(0.0, min(self.speed, self.MAX_SPEED))

    def update_position(self, track: "Track") -> bool:
        """
        Di chuyển xe theo hướng + tốc độ hiện tại.
        Trả về False nếu va chạm (xe chết).
        """
        nx = self.x + math.cos(self.angle) * self.speed
        ny = self.y + math.sin(self.angle) * self.speed

        if track.is_off_track(nx, ny):
            self.alive = False
            return False

        self.x = nx
        self.y = ny

        # Checkpoint logic
        if hasattr(track, "checkpoints") and len(track.checkpoints) > 0:
            cx, cy, radius = track.checkpoints[self.cp_idx]
            dist_to_cp = math.hypot(self.x - cx, self.y - cy)
            if dist_to_cp < radius:
                self.cp_idx = (self.cp_idx + 1) % len(track.checkpoints)
                self.total_cp_passed += 1
                self.last_cp_passed_frame = self._frame_count
            
            # Fitness dựa hoàn toàn vào checkpoint
            self.fitness = self.total_cp_passed * 1000 - dist_to_cp
        else:
            self.fitness += self.speed  # Fallback

        # Trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)

        return True

    def check_stuck(self) -> None:
        """Tự chết nếu không qua checkpoint mới trong MAX_STUCK_FRAMES frames."""
        self._frame_count += 1
        frames_since_cp = self._frame_count - self.last_cp_passed_frame
        if frames_since_cp >= 150:  # Khoảng 2.5 giây không qua checkpoint
            self.alive = False

    # ── Main step (GA mode) ─────────────────────────────────────────────

    def step(self, track: "Track") -> bool:
        """
        1 frame update: observe → decide → act → physics → check.
        Trả về True nếu xe vẫn alive.
        (Dùng cho GA mode — NN quyết định hành động)
        """
        if not self.alive:
            return False

        # 1. Quan sát
        sensor_input = self.cast_sensors(track)

        # 2. Quyết định từ NN
        turn, engine = self.nn.forward(sensor_input)

        # 3. Áp dụng hành động
        self.act(float(turn), float(engine))

        # 4. Di chuyển + kiểm tra va chạm
        self.update_position(track)

        # 5. Chống stuck
        if self.alive:
            self.check_stuck()

        return self.alive

    # ── Gym-like API (PPO mode) ───────────────────────────────────────

    def get_observation(self, track: "Track") -> List[float]:
        """Trả về observation hiện tại (sensor readings)."""
        return self.cast_sensors(track)

    def gym_reset(self, start_x: float, start_y: float, start_angle: float) -> None:
        """Reset xe về trạng thái ban đầu cho episode mới."""
        self.x = start_x
        self.y = start_y
        self.angle = start_angle
        self.speed = 0.0
        self.fitness = 0.0
        self.alive = True
        self.trail = []
        self._frame_count = 0
        self.cp_idx = 0
        self.total_cp_passed = 0
        self.last_cp_passed_frame = 0
        self.target_checkpoint = None
        self.angle_diff = 0.0
        self.sensor_endpoints = []
        self.sensor_values = [1.0] * (self.N_SENSORS + 1)

    def gym_step(
        self, action: "Tuple[float, float]", track: "Track"
    ) -> "Tuple[List[float], float, bool, dict]":
        """
        Gym-like step: nhận action từ bên ngoài (PPO agent).

        Args:
            action: (turn, engine) — mỗi giá trị ∈ [-1, 1]
            track: Sa hình đường đua

        Returns:
            obs: Sensor readings [0,1] x N_SENSORS
            reward: Quãng đường di chuyển frame này
            done: True nếu xe chết
            info: Dict thông tin phụ
        """
        if not self.alive:
            obs = [0.0] * (self.N_SENSORS + 1)  # 7 sensors + 1 compass
            return obs, 0.0, True, {"reason": "already_dead"}

        turn, engine = float(action[0]), float(action[1])

        # Áp dụng hành động
        self.act(turn, engine)

        # Di chuyển
        old_fitness = self.fitness
        alive_before = self.alive
        self.update_position(track)

        # Chống stuck
        if self.alive:
            self.check_stuck()

        done = not self.alive
        step_reward = self.fitness - old_fitness  # quãng đường frame này

        # Quan sát mới
        if self.alive:
            obs = self.cast_sensors(track)
        else:
            obs = [0.0] * (self.N_SENSORS + 1)  # 7 sensors + 1 compass

        info = {
            "speed": self.speed,
            "fitness": self.fitness,
            "frame": self._frame_count,
        }

        return obs, step_reward, done, info

    # ── Rendering helpers ─────────────────────────────────────────────

    @property
    def corners(self) -> List[Tuple[float, float]]:
        """4 góc xe để vẽ hình chữ nhật xoay."""
        W, H = 13, 8
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        pts = [(-6, -4), (7, -4), (7, 4), (-6, 4)]
        return [
            (self.x + cos_a * px - sin_a * py,
             self.y + sin_a * px + cos_a * py)
            for px, py in pts
        ]

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return f"Car({status}, fit={self.fitness:.1f}, x={self.x:.0f}, y={self.y:.0f})"

