"""
src/rendering/renderer.py
──────────────────────────
Pygame renderer cho toàn bộ simulation.

Tách biệt hoàn toàn với logic AI/physics:
  Renderer chỉ đọc state, không bao giờ ghi vào Car hay Track.

Layers (vẽ theo thứ tự):
  1. Background
  2. Track (fill + center line)
  3. Sensor rays (các xe còn sống)
  4. Car trails
  5. Cars (dead → alive → best)
  6. HUD overlay (gen, fitness, population bar, speed)
"""

from __future__ import annotations
import math
import pygame
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.car import Car
    from src.simulation.track import Track


# ── Color palette ─────────────────────────────────────────────────────

COLORS = {
    "bg":           (18,  18,  24),
    "track_fill":   (42,  42,  52),
    "track_edge":   (65,  65,  80),
    "center_line":  (80,  80,  100),
    "car_normal":   (55,  138, 221),   # xanh dương
    "car_elite":    (29,  158, 117),   # xanh lá
    "car_best":     (239, 159, 39),    # vàng cam
    "car_dead":     (226, 75,  74),    # đỏ
    "sensor":       (255, 255, 255),
    "trail_normal": (55,  138, 221),
    "trail_best":   (239, 159, 39),
    "hud_bg":       (12,  12,  18),
    "hud_text":     (200, 200, 210),
    "hud_accent":   (29,  158, 117),
    "bar_bg":       (40,  40,  52),
    "bar_fill":     (29,  158, 117),
}


class Renderer:
    """
    Quản lý toàn bộ Pygame window và render pipeline.

    Usage:
        renderer = Renderer(width=1280, height=720)
        renderer.draw_frame(cars, track, stats)
        pygame.display.flip()
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        show_sensors: bool = True,
        show_trails: bool = True,
        show_checkpoints: bool = True,
        headless: bool = False,
    ):
        pygame.init()
        flags = pygame.HIDDEN if headless else 0
        self.screen = pygame.display.set_mode((width, height), flags=flags)
        if not headless:
            pygame.display.set_caption("Deep Learning Cars — Neuroevolution")
        self.width = width
        self.height = height
        self.fps = fps
        self.show_sensors = show_sensors
        self.show_trails = show_trails
        self.show_checkpoints = show_checkpoints
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_mid   = pygame.font.SysFont("monospace", 15)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # Surface riêng cho semi-transparent elements
        self._overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    # ── Track rendering ───────────────────────────────────────────────

    def _draw_track(self, track: "Track") -> None:
        n = len(track.waypoints)
        if n < 2:
            return

        # 1. Edge highlight (draw thick segments and node circles)
        for i in range(n):
            a = track.waypoints[i]
            b = track.waypoints[(i + 1) % n]
            wa = track.widths[i]
            wb = track.widths[(i + 1) % n]
            
            avg_w = int((wa + wb) / 2)
            pygame.draw.line(self.screen, COLORS["track_edge"], (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), avg_w)
            pygame.draw.circle(self.screen, COLORS["track_edge"], (int(a[0]), int(a[1])), int(wa / 2))

        # 2. Road fill (draw slightly thinner segments to reveal edge)
        for i in range(n):
            a = track.waypoints[i]
            b = track.waypoints[(i + 1) % n]
            wa = track.widths[i]
            wb = track.widths[(i + 1) % n]
            
            avg_w = max(2, int((wa + wb) / 2) - 8)
            radius = max(1, int(wa / 2) - 4)
            
            pygame.draw.line(self.screen, COLORS["track_fill"], (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), avg_w)
            pygame.draw.circle(self.screen, COLORS["track_fill"], (int(a[0]), int(a[1])), radius)

        # 3. Center dashed line
        for i in range(n):
            a = track.waypoints[i]
            b = track.waypoints[(i + 1) % n]
            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            pygame.draw.line(
                self.screen, COLORS["center_line"],
                (int(a[0]), int(a[1])), (int(mx), int(my)), 1
            )

    def _draw_checkpoints(self, track: "Track", cars: List["Car"]) -> None:
        if not hasattr(track, "checkpoints") or not track.checkpoints:
            return
            
        # Vẽ các vùng checkpoint mờ
        for cx, cy, radius in track.checkpoints:
            pygame.draw.circle(self._overlay, (255, 200, 0, 30), (int(cx), int(cy)), int(radius), 2)

        # Vẽ tia chỉ đường (vàng rực) cho xe dẫn đầu hoặc tất cả
        best_car = None
        alive_cars = [c for c in cars if c.alive]
        if alive_cars:
            best_car = max(alive_cars, key=lambda c: c.fitness)
            
        for car in alive_cars:
            if car.target_checkpoint:
                cx, cy = car.target_checkpoint
                color = (255, 200, 0, 180) if car is best_car else (255, 200, 0, 40)
                pygame.draw.line(
                    self._overlay, color,
                    (int(car.x), int(car.y)), (int(cx), int(cy)), 2 if car is best_car else 1
                )

    # ── Sensor rays ───────────────────────────────────────────────────

    def _draw_sensors(self, car: "Car") -> None:
        if not car.alive or not car.sensor_endpoints:
            return
        for i, (ex, ey) in enumerate(car.sensor_endpoints):
            val = car.sensor_values[i]
            # Màu ray: đỏ khi gần tường, trắng khi xa
            alpha = max(30, int(val * 180))
            color = (
                int(255 * (1 - val)),  # R: cao khi gần
                int(200 * val),        # G: cao khi xa
                255,                   # B
            )
            pygame.draw.line(
                self.screen, color,
                (int(car.x), int(car.y)),
                (int(ex), int(ey)), 1
            )
            # Endpoint dot
            pygame.draw.circle(
                self.screen, color,
                (int(ex), int(ey)), 2
            )

    # ── Car rendering ─────────────────────────────────────────────────

    def _car_color(self, car: "Car", is_best: bool) -> Tuple[int, int, int]:
        if not car.alive:
            return COLORS["car_dead"]
        if is_best:
            return COLORS["car_best"]
        if car.is_elite:
            return COLORS["car_elite"]
        return COLORS["car_normal"]

    def _draw_trail(self, car: "Car", is_best: bool) -> None:
        if len(car.trail) < 2:
            return
        base_color = COLORS["trail_best"] if is_best else COLORS["trail_normal"]
        n = len(car.trail)
        for i in range(1, n):
            alpha = int(180 * i / n)
            color = (*base_color, alpha)
            try:
                pygame.draw.line(
                    self._overlay, color,
                    (int(car.trail[i-1][0]), int(car.trail[i-1][1])),
                    (int(car.trail[i][0]),   int(car.trail[i][1])), 2
                )
            except Exception:
                pass

    def _draw_car(self, car: "Car", is_best: bool) -> None:
        color = self._car_color(car, is_best)
        corners = [(int(x), int(y)) for x, y in car.corners]
        pygame.draw.polygon(self.screen, color, corners)
        # Viền trắng mỏng
        pygame.draw.polygon(self.screen, (255, 255, 255, 80), corners, 1)
        # Windshield nhỏ
        if car.alive:
            cx = sum(x for x, _ in corners[:2]) // 2
            cy = sum(y for _, y in corners[:2]) // 2
            pygame.draw.circle(self.screen, (200, 220, 255), (cx, cy), 2)

    # ── HUD ───────────────────────────────────────────────────────────

    def _draw_hud(self, stats: dict, n_alive: int, population_size: int) -> None:
        hud_x, hud_y = 16, 16
        line_h = 22
        padding = 10

        lines = [
            (f"Generation   {stats.get('generation', 0):>4}", COLORS["hud_accent"]),
            (f"Alive        {n_alive:>2}/{population_size}", COLORS["hud_text"]),
            (f"Best fit     {stats.get('current_best', 0):>7.1f}", COLORS["hud_text"]),
            (f"Record       {stats.get('all_time_best', 0):>7.1f}", COLORS["hud_accent"]),
        ]

        # Background panel
        panel_w = 210
        panel_h = len(lines) * line_h + padding * 2 + 20
        pygame.draw.rect(
            self.screen, (*COLORS["hud_bg"], 200),
            (hud_x - padding, hud_y - padding, panel_w, panel_h),
            border_radius=8
        )

        for i, (text, color) in enumerate(lines):
            surf = self.font_mid.render(text, True, color)
            self.screen.blit(surf, (hud_x, hud_y + i * line_h))

        # Population progress bar
        bar_y = hud_y + len(lines) * line_h + 4
        bar_w = panel_w - padding * 2
        pygame.draw.rect(
            self.screen, COLORS["bar_bg"],
            (hud_x, bar_y, bar_w, 6), border_radius=3
        )
        fill_w = int(bar_w * n_alive / max(1, population_size))
        if fill_w > 0:
            pygame.draw.rect(
                self.screen, COLORS["bar_fill"],
                (hud_x, bar_y, fill_w, 6), border_radius=3
            )

    # ── Main draw call ────────────────────────────────────────────────

    def draw_frame(
        self,
        cars: List["Car"],
        track: "Track",
        stats: dict,
    ) -> None:
        """
        Vẽ 1 frame hoàn chỉnh.

        Args:
            cars: Toàn bộ population (alive + dead)
            track: Sa hình hiện tại
            stats: Dict chứa generation, current_best, all_time_best, ...
        """
        # Clear
        self.screen.fill(COLORS["bg"])
        self._overlay.fill((0, 0, 0, 0))

        # Layer 1: Track
        self._draw_track(track)
        
        # Layer 1.5: Checkpoints
        if self.show_checkpoints:
            self._draw_checkpoints(track, cars)

        # Layer 2: Trails
        if self.show_trails:
            alive_cars = [c for c in cars if c.alive]
            best = max(alive_cars, key=lambda c: c.fitness) if alive_cars else None
            for car in cars:
                self._draw_trail(car, car is best)
            self.screen.blit(self._overlay, (0, 0))

        # Layer 3: Sensors (alive only)
        if self.show_sensors:
            for car in cars:
                if car.alive:
                    self._draw_sensors(car)

        # Layer 4: Cars — dead first, then alive, best on top
        alive_cars = [c for c in cars if c.alive]
        dead_cars  = [c for c in cars if not c.alive]
        best_car = max(alive_cars, key=lambda c: c.fitness) if alive_cars else None

        for car in dead_cars:
            self._draw_car(car, False)
        for car in alive_cars:
            if car is not best_car:
                self._draw_car(car, False)
        if best_car:
            self._draw_car(best_car, True)

        # Layer 5: HUD
        n_alive = len(alive_cars)
        self._draw_hud(stats, n_alive, len(cars))

        self.clock.tick(self.fps)

    def handle_quit(self) -> bool:
        """Trả về True nếu user đóng cửa sổ."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def quit(self) -> None:
        pygame.quit()
