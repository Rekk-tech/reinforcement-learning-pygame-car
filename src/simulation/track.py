"""
src/simulation/track.py
────────────────────────
Sa hình đường đua 2D định nghĩa bằng waypoints.

Thiết kế:
  - Track = chuỗi waypoints (x,y) khép kín
  - "Trên track" = khoảng cách từ điểm đến segment gần nhất <= track_width/2
  - Hỗ trợ nhiều preset: oval, figure8, city
  - Checkpoints tự động sinh từ waypoints để đo fitness chính xác hơn

Thuật toán is_off_track():
  Với mỗi điểm P cần kiểm tra:
  1. Tìm segment (A,B) gần nhất trong track
  2. Tính khoảng cách vuông góc từ P đến segment đó
  3. Nếu distance > half_width → off track
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict


# ── Preset tracks ─────────────────────────────────────────────────────

PRESET_TRACKS: Dict[str, List[Tuple[float, ...]]] = {
    "oval": [
        (150, 90), (320, 70), (490, 80), (600, 140), (620, 220),
        (610, 310), (530, 370), (400, 395), (260, 390), (140, 360),
        (80, 280), (75, 190), (100, 130),
    ],
    "figure8": [
        (180, 100), (300, 80), (420, 100), (500, 170), (480, 250),
        (380, 290), (300, 310), (220, 290), (140, 250), (130, 170),
        (200, 130), (300, 180), (400, 230), (460, 310), (420, 390),
        (300, 420), (180, 390), (120, 300), (140, 210), (200, 150),
    ],
    "city_simple": [
        # Khối chữ nhật với bo góc (fillet)
        (200, 100, 60), (600, 100, 60), 
        (700, 200, 80), (700, 400, 80), 
        (600, 500, 60), (200, 500, 60), 
        (100, 400, 80), (100, 200, 80)
    ],
    "city": [
        # Hình số 8 vuông bám viền ngoài 2 blocks
        # Top street
        (200, 100, 60), (400, 100, 60), (600, 100, 60),
        # Right curve & street
        (700, 200, 80), (700, 400, 80),
        # Bottom right curve & street
        (600, 500, 60), (500, 500, 60),
        # Inner turn UP
        (420, 420, 90), (420, 300, 70),
        # Inner turn LEFT
        (380, 260, 90), (300, 260, 70),
        # Inner turn DOWN
        (260, 300, 90), (260, 420, 70),
        # Bottom left turn & street
        (180, 500, 90), (100, 500, 60),
        # Left curve & street
        (0, 400, 80), (0, 200, 80),
        # Top left curve
        (100, 100, 80)
    ],
}


class Track:
    """
    Sa hình đường đua với API kiểm tra va chạm nhanh.

    Args:
        waypoints: List[(x,y)] — tâm đường đua theo thứ tự
        track_width: Chiều rộng đường (pixels)
        name: Tên track để debug

    Attributes:
        waypoints: List tọa độ waypoints
        half_width: Bán kính track (track_width / 2)
        start_x, start_y: Vị trí xuất phát xe
        start_angle: Góc xuất phát (hướng từ waypoint 0 → 1)
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, ...]] = None,
        track_width: int = 56,
        name: str = "custom",
        checkpoint_interval: int = 60,
    ):
        self.waypoints_raw = waypoints or PRESET_TRACKS["oval"]
        self.track_width = track_width
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.n = len(self.waypoints_raw)

        self.waypoints = []
        self.widths = []
        for p in self.waypoints_raw:
            self.waypoints.append((float(p[0]), float(p[1])))
            w = float(p[2]) if len(p) > 2 else float(track_width)
            self.widths.append(w)

        # Vị trí + góc xuất phát
        p0 = self.waypoints[0]
        p1 = self.waypoints[1]
        self.start_x = p0[0]
        self.start_y = p0[1]
        self.start_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        # Cache numpy arrays cho vectorized distance check
        self._pts = np.array(self.waypoints, dtype=np.float32)
        self._segs_a = self._pts                         # shape (n, 2)
        self._segs_b = np.roll(self._pts, -1, axis=0)   # shifted by 1
        
        self.generate_checkpoints()

    def generate_checkpoints(self) -> None:
        """Tự động sinh checkpoints dọc theo các segment của đường đua."""
        self.checkpoints: List[Tuple[float, float, float]] = []
        for i in range(self.n):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[(i + 1) % self.n]
            wa = self.widths[i]
            wb = self.widths[(i + 1) % self.n]
            
            dx, dy = bx - ax, by - ay
            dist = math.hypot(dx, dy)
            if dist == 0: continue
            
            # Số lượng checkpoint trên đoạn này
            count = max(1, int(dist / self.checkpoint_interval))
            for j in range(count):
                t = (j + 1) / (count + 1)  # Không rải ở mép waypoint để tránh trùng
                cx = ax + t * dx
                cy = ay + t * dy
                cw = wa + t * (wb - wa)
                # Bán kính checkpoint lớn hơn nửa độ rộng đường một chút để xe dễ chạm
                radius = (cw / 2) + 15
                self.checkpoints.append((cx, cy, radius))

    def center(self, screen_width: int, screen_height: int) -> None:
        """Dịch chuyển toàn bộ track ra giữa màn hình."""
        xs = [p[0] for p in self.waypoints]
        ys = [p[1] for p in self.waypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        track_w = max_x - min_x
        track_h = max_y - min_y
        
        offset_x = (screen_width - track_w) / 2 - min_x
        offset_y = (screen_height - track_h) / 2 - min_y
        
        self.waypoints = [(x + offset_x, y + offset_y) for x, y in self.waypoints]
        
        p0 = self.waypoints[0]
        p1 = self.waypoints[1]
        self.start_x = float(p0[0])
        self.start_y = float(p0[1])
        self.start_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        
        self._pts = np.array(self.waypoints, dtype=np.float32)
        self._segs_a = self._pts
        self._segs_b = np.roll(self._pts, -1, axis=0)
        
        self.generate_checkpoints()

    # ── Track geometry helpers ────────────────────────────────────────

    @staticmethod
    def _dist_point_to_segment_ratio(
        px: float, py: float,
        ax: float, ay: float, aw: float,
        bx: float, by: float, bw: float,
    ) -> float:
        """Khoảng cách tỉ lệ từ điểm P đến đoạn thẳng AB (d / local_half_width)."""
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-9:
            dist = math.hypot(px - ax, py - ay)
            return dist / (aw / 2)
        
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        
        dist = math.hypot(px - closest_x, py - closest_y)
        local_half_width = (aw + t * (bw - aw)) / 2
        return dist / local_half_width

    def is_off_track(self, x: float, y: float) -> bool:
        """True nếu điểm (x,y) nằm ngoài biên đường đua (ratio > 1.0 cho tất cả segments)."""
        min_ratio = float("inf")
        for i in range(self.n):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[(i + 1) % self.n]
            aw = self.widths[i]
            bw = self.widths[(i + 1) % self.n]
            
            ratio = self._dist_point_to_segment_ratio(x, y, ax, ay, aw, bx, by, bw)
            if ratio <= 1.0:
                return False  # Nằm trong đoạn này => on track
            if ratio < min_ratio:
                min_ratio = ratio
                
        return True # Nằm ngoài tất cả các đoạn

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str, track_width: int = 56) -> "Track":
        """Tạo track từ preset name."""
        if name not in PRESET_TRACKS:
            available = list(PRESET_TRACKS.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return cls(
            waypoints=PRESET_TRACKS[name],
            track_width=track_width,
            name=name,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Track":
        """Load track từ file YAML custom."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        
        waypoints = []
        for p in data["waypoints"]:
            if "w" in p:
                waypoints.append((p["x"], p["y"], p["w"]))
            else:
                waypoints.append((p["x"], p["y"]))
                
        return cls(
            waypoints=waypoints,
            track_width=data.get("track_width", 56),
            name=data.get("name", "custom"),
        )

    # ── Debug ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Track(name={self.name!r}, "
            f"waypoints={self.n}, "
            f"width={self.track_width})"
        )
