"""
src/ui/controls.py
───────────────────
Giao dien tuong tac voi nguoi dung trong qua trinh huan luyen.

Components:
  - Speed slider: Thanh truot chinh toc do mo phong (1x - 10x)
  - Pause button: Tam dung / tiep tuc huan luyen
  - Reset button: Khoi dong lai tu dau (population moi)
  - Keyboard shortcuts: phim tat tuong ung

Keyboard shortcuts:
  Space  : Pause / Resume
  R      : Reset
  + / =  : Tang toc
  - / _  : Giam toc
  1-9    : Dat toc do truc tiep (1x - 9x)
  0      : Dat toc do 10x
  Escape : Thoat
"""

from __future__ import annotations
import pygame
from typing import Optional, Tuple


# ── Color palette ─────────────────────────────────────────────────────

CTRL_COLORS = {
    "panel_bg":     (12,  12,  18,  200),
    "text":         (200, 200, 210),
    "accent":       (29,  158, 117),
    "btn_normal":   (50,  50,  65),
    "btn_hover":    (70,  70,  90),
    "btn_active":   (29,  158, 117),
    "btn_danger":   (226, 75,  74),
    "slider_bg":    (40,  40,  52),
    "slider_fill":  (55,  138, 221),
    "slider_knob":  (220, 220, 230),
    "pause_color":  (239, 159, 39),
}


class UIControls:
    """
    Giao dien tuong tac: speed slider, pause, reset.

    Usage:
        controls = UIControls(screen_width=1280, screen_height=720)
        controls.handle_events(events)    # Goi moi frame
        speed = controls.speed            # Doc toc do hien tai
        if controls.paused:               # Kiem tra trang thai
            ...
        if controls.reset_requested:      # Kiem tra yeu cau reset
            controls.acknowledge_reset()
    """

    def __init__(
        self,
        screen_width: int = 1280,
        screen_height: int = 720,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # State
        self.speed: int = 1               # 1x - 10x
        self.paused: bool = False
        self.reset_requested: bool = False
        self._quit_requested: bool = False

        # UI layout
        self._bar_height = 36
        self._bar_y = screen_height - self._bar_height

        # Slider geometry
        self._slider_x = 100
        self._slider_w = 160
        self._slider_y = self._bar_y + 12
        self._dragging = False

        # Button geometry
        self._pause_btn = pygame.Rect(screen_width - 180, self._bar_y + 4, 80, 28)
        self._reset_btn = pygame.Rect(screen_width - 90, self._bar_y + 4, 80, 28)

        # Hover state
        self._hover_pause = False
        self._hover_reset = False

        # Fonts
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)

    # ── Event handling ────────────────────────────────────────────────

    def handle_events(self, events: list) -> bool:
        """
        Xu ly events. Tra ve True neu user yeu cau thoat (Quit / Escape).

        Args:
            events: list pygame events tu pygame.event.get()

        Returns:
            True neu can thoat ung dung
        """
        mouse_pos = pygame.mouse.get_pos()
        self._hover_pause = self._pause_btn.collidepoint(mouse_pos)
        self._hover_reset = self._reset_btn.collidepoint(mouse_pos)

        for event in events:
            if event.type == pygame.QUIT:
                self._quit_requested = True
                return True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                    return True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_requested = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed = min(10, self.speed + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    self.speed = max(1, self.speed - 1)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    self.speed = event.key - pygame.K_0
                elif event.key == pygame.K_0:
                    self.speed = 10

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Slider click
                    slider_rect = pygame.Rect(
                        self._slider_x, self._slider_y - 6,
                        self._slider_w, 12
                    )
                    if slider_rect.collidepoint(event.pos):
                        self._dragging = True
                        self._update_slider(event.pos[0])

                    # Button clicks
                    if self._pause_btn.collidepoint(event.pos):
                        self.paused = not self.paused
                    elif self._reset_btn.collidepoint(event.pos):
                        self.reset_requested = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self._dragging:
                    self._update_slider(event.pos[0])

        return False

    def _update_slider(self, mouse_x: int) -> None:
        """Cap nhat toc do tu vi tri chuot tren slider."""
        ratio = (mouse_x - self._slider_x) / self._slider_w
        ratio = max(0.0, min(1.0, ratio))
        self.speed = max(1, min(10, int(ratio * 9) + 1))

    def acknowledge_reset(self) -> None:
        """Goi sau khi da xu ly reset xong."""
        self.reset_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    # ── Drawing ───────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        """Ve control bar o duoi cung man hinh."""

        # Background bar
        bar_surface = pygame.Surface((self.screen_width, self._bar_height), pygame.SRCALPHA)
        bar_surface.fill(CTRL_COLORS["panel_bg"])
        screen.blit(bar_surface, (0, self._bar_y))

        # ── Speed label ──
        speed_label = self.font.render(f"Speed:", True, CTRL_COLORS["text"])
        screen.blit(speed_label, (16, self._bar_y + 10))

        # ── Speed slider ──
        # Track
        pygame.draw.rect(
            screen, CTRL_COLORS["slider_bg"],
            (self._slider_x, self._slider_y, self._slider_w, 4),
            border_radius=2,
        )

        # Fill
        fill_ratio = (self.speed - 1) / 9.0
        fill_w = int(self._slider_w * fill_ratio)
        if fill_w > 0:
            pygame.draw.rect(
                screen, CTRL_COLORS["slider_fill"],
                (self._slider_x, self._slider_y, fill_w, 4),
                border_radius=2,
            )

        # Knob
        knob_x = self._slider_x + fill_w
        pygame.draw.circle(screen, CTRL_COLORS["slider_knob"], (knob_x, self._slider_y + 2), 7)
        pygame.draw.circle(screen, CTRL_COLORS["slider_fill"], (knob_x, self._slider_y + 2), 5)

        # Speed value
        speed_val = self.font.render(f"{self.speed}x", True, CTRL_COLORS["accent"])
        screen.blit(speed_val, (self._slider_x + self._slider_w + 10, self._bar_y + 10))

        # ── Pause button ──
        pause_color = CTRL_COLORS["pause_color"] if self.paused else (
            CTRL_COLORS["btn_hover"] if self._hover_pause else CTRL_COLORS["btn_normal"]
        )
        pygame.draw.rect(screen, pause_color, self._pause_btn, border_radius=6)
        pause_text = "RESUME" if self.paused else "PAUSE"
        pause_surf = self.font.render(pause_text, True, (255, 255, 255))
        pause_rect = pause_surf.get_rect(center=self._pause_btn.center)
        screen.blit(pause_surf, pause_rect)

        # ── Reset button ──
        reset_color = (
            CTRL_COLORS["btn_danger"] if self._hover_reset else CTRL_COLORS["btn_normal"]
        )
        pygame.draw.rect(screen, reset_color, self._reset_btn, border_radius=6)
        reset_surf = self.font.render("RESET", True, (255, 255, 255))
        reset_rect = reset_surf.get_rect(center=self._reset_btn.center)
        screen.blit(reset_surf, reset_rect)

        # ── Keyboard hints ──
        hints = "[Space] Pause  [+/-] Speed  [R] Reset  [Esc] Quit"
        hint_surf = self.font_small.render(hints, True, (100, 100, 120))
        screen.blit(hint_surf, (self._slider_x + self._slider_w + 60, self._bar_y + 12))

        # ── Paused overlay ──
        if self.paused:
            pause_overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pause_overlay.fill((0, 0, 0, 80))
            screen.blit(pause_overlay, (0, 0))

            big_font = pygame.font.SysFont("monospace", 48, bold=True)
            paused_text = big_font.render("PAUSED", True, CTRL_COLORS["pause_color"])
            rect = paused_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            screen.blit(paused_text, rect)

            sub_font = pygame.font.SysFont("monospace", 18)
            sub_text = sub_font.render("Press SPACE to resume", True, CTRL_COLORS["text"])
            sub_rect = sub_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 40))
            screen.blit(sub_text, sub_rect)
