"""
src/rendering/hud.py
─────────────────────
Heads-Up Display — ve cac lop thong tin de len man hinh.

Components:
  - Stats panel: generation, alive, best fitness, record
  - Population progress bar: ty le xe con song
  - Fitness chart: bieu do duong fitness/reward theo thoi gian (goc duoi phai)
  - Algorithm badge: hien thi GA hoac PPO
"""

from __future__ import annotations
import pygame
from collections import deque
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ── Color palette ─────────────────────────────────────────────────────

HUD_COLORS = {
    "panel_bg":     (12,  12,  18,  200),
    "text":         (200, 200, 210),
    "accent":       (29,  158, 117),
    "warning":      (239, 159, 39),
    "danger":       (226, 75,  74),
    "bar_bg":       (40,  40,  52),
    "bar_fill":     (29,  158, 117),
    "chart_bg":     (20,  20,  30,  180),
    "chart_line":   (55,  180, 140),
    "chart_mean":   (100, 120, 200),
    "chart_grid":   (50,  50,  65),
    "badge_ga":     (55,  138, 221),
    "badge_ppo":    (239, 159, 39),
}


class HUD:
    """
    Heads-Up Display — lop de len man hinh cua Renderer.

    Usage:
        hud = HUD(screen_width=1280, screen_height=720)
        hud.update(stats)
        hud.draw(screen)
    """

    def __init__(
        self,
        screen_width: int = 1280,
        screen_height: int = 720,
        chart_max_points: int = 100,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts
        pygame.font.init()
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_mid   = pygame.font.SysFont("monospace", 15)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # Chart data
        self.best_history: deque = deque(maxlen=chart_max_points)
        self.mean_history: deque = deque(maxlen=chart_max_points)

        # Algorithm badge
        self.algorithm: str = "GA"

    def set_algorithm(self, algo: str) -> None:
        """Set algorithm label (GA or PPO)."""
        self.algorithm = algo.upper()

    def update(self, stats: dict) -> None:
        """Cap nhat chart data tu stats moi."""
        if not stats:
            return
        best = stats.get("current_best", 0)
        mean = stats.get("mean_fitness", 0)
        self.best_history.append(best)
        if mean > 0:
            self.mean_history.append(mean)

    # ── Stats Panel (goc tren trai) ─────────────────────────────────

    def _draw_stats_panel(
        self, screen: pygame.Surface, stats: dict,
        n_alive: int, population_size: int,
    ) -> None:
        """Ve panel thong ke o goc tren trai."""
        hud_x, hud_y = 16, 16
        line_h = 22
        padding = 10

        gen = stats.get("generation", 0)
        best = stats.get("current_best", 0)
        record = stats.get("all_time_best", 0)

        lines = [
            (f"Generation   {gen:>4}", HUD_COLORS["accent"]),
            (f"Alive        {n_alive:>2}/{population_size}", HUD_COLORS["text"]),
            (f"Best fit     {best:>7.1f}", HUD_COLORS["text"]),
            (f"Record       {record:>7.1f}", HUD_COLORS["accent"]),
        ]

        # Background panel
        panel_w = 210
        panel_h = len(lines) * line_h + padding * 2 + 20

        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill(HUD_COLORS["panel_bg"])
        screen.blit(panel_surface, (hud_x - padding, hud_y - padding))

        for i, (text, color) in enumerate(lines):
            surf = self.font_mid.render(text, True, color)
            screen.blit(surf, (hud_x, hud_y + i * line_h))

        # Population progress bar
        bar_y = hud_y + len(lines) * line_h + 4
        bar_w = panel_w - padding * 2
        pygame.draw.rect(
            screen, HUD_COLORS["bar_bg"],
            (hud_x, bar_y, bar_w, 6), border_radius=3,
        )
        fill_w = int(bar_w * n_alive / max(1, population_size))
        if fill_w > 0:
            pygame.draw.rect(
                screen, HUD_COLORS["bar_fill"],
                (hud_x, bar_y, fill_w, 6), border_radius=3,
            )

    # ── Algorithm Badge (goc tren phai) ──────────────────────────────

    def _draw_badge(self, screen: pygame.Surface) -> None:
        """Ve badge thuat toan (GA / PPO) o goc tren phai."""
        badge_color = HUD_COLORS["badge_ga"] if self.algorithm == "GA" else HUD_COLORS["badge_ppo"]
        text = f" {self.algorithm} "
        surf = self.font_large.render(text, True, (255, 255, 255))
        rect = surf.get_rect()
        rect.topright = (self.screen_width - 16, 16)

        # Badge background
        bg_rect = rect.inflate(12, 6)
        badge_bg = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(badge_bg, (*badge_color, 220), badge_bg.get_rect(), border_radius=6)
        screen.blit(badge_bg, bg_rect.topleft)
        screen.blit(surf, rect)

    # ── Fitness Chart (goc duoi phai) ────────────────────────────────

    def _draw_chart(self, screen: pygame.Surface) -> None:
        """Ve bieu do duong fitness theo thoi gian."""
        if len(self.best_history) < 2:
            return

        chart_w = 280
        chart_h = 120
        margin = 16
        cx = self.screen_width - chart_w - margin
        cy = self.screen_height - chart_h - margin

        # Background
        chart_bg = pygame.Surface((chart_w, chart_h), pygame.SRCALPHA)
        chart_bg.fill(HUD_COLORS["chart_bg"])
        screen.blit(chart_bg, (cx, cy))

        # Border
        pygame.draw.rect(screen, HUD_COLORS["chart_grid"], (cx, cy, chart_w, chart_h), 1)

        # Grid lines (3 ngang)
        for i in range(1, 4):
            y = cy + int(chart_h * i / 4)
            pygame.draw.line(screen, HUD_COLORS["chart_grid"], (cx, y), (cx + chart_w, y), 1)

        # Tinh scale
        data = list(self.best_history)
        max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        val_range = max(max_val - min_val, 1)

        # Ve duong best fitness
        n = len(data)
        points = []
        for i, val in enumerate(data):
            x = cx + int(chart_w * i / max(1, n - 1))
            y = cy + chart_h - int(chart_h * (val - min_val) / val_range * 0.85) - 10
            y = max(cy + 2, min(cy + chart_h - 2, y))
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(screen, HUD_COLORS["chart_line"], False, points, 2)

        # Ve duong mean fitness (neu co)
        if len(self.mean_history) >= 2:
            mean_data = list(self.mean_history)
            mean_points = []
            for i, val in enumerate(mean_data):
                x = cx + int(chart_w * i / max(1, len(mean_data) - 1))
                y = cy + chart_h - int(chart_h * (val - min_val) / val_range * 0.85) - 10
                y = max(cy + 2, min(cy + chart_h - 2, y))
                mean_points.append((x, y))
            if len(mean_points) >= 2:
                pygame.draw.lines(screen, HUD_COLORS["chart_mean"], False, mean_points, 1)

        # Labels
        title = self.font_small.render("Fitness", True, HUD_COLORS["text"])
        screen.blit(title, (cx + 4, cy + 2))

        max_label = self.font_small.render(f"{max_val:.0f}", True, HUD_COLORS["chart_line"])
        screen.blit(max_label, (cx + 4, cy + 14))

        # Legend
        legend_y = cy + chart_h - 14
        pygame.draw.line(screen, HUD_COLORS["chart_line"], (cx + 4, legend_y + 5), (cx + 20, legend_y + 5), 2)
        best_label = self.font_small.render("best", True, HUD_COLORS["chart_line"])
        screen.blit(best_label, (cx + 24, legend_y))

        if len(self.mean_history) >= 2:
            pygame.draw.line(screen, HUD_COLORS["chart_mean"], (cx + 70, legend_y + 5), (cx + 86, legend_y + 5), 1)
            mean_label = self.font_small.render("mean", True, HUD_COLORS["chart_mean"])
            screen.blit(mean_label, (cx + 90, legend_y))

    # ── Main draw ────────────────────────────────────────────────────

    def draw(
        self,
        screen: pygame.Surface,
        stats: dict,
        n_alive: int = 0,
        population_size: int = 20,
    ) -> None:
        """
        Ve toan bo HUD len screen.

        Args:
            screen: Pygame surface chinh
            stats: Dict tu training loop (generation, current_best, all_time_best, ...)
            n_alive: So xe con song
            population_size: Tong so xe
        """
        if not stats:
            return

        self._draw_stats_panel(screen, stats, n_alive, population_size)
        self._draw_badge(screen)
        self._draw_chart(screen)
