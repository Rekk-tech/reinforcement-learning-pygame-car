"""
src/simulation/camera.py
─────────────────────────
Camera sensor cho Computer Vision mode.

Chup (capture) vung 84x84 pixel xung quanh xe tu Pygame surface,
chuyen sang grayscale, va thuc hien frame stacking (4 frames)
de agent CNN hoc duoc thong tin ve chuyen dong.

Pipeline:
  Pygame Surface -> crop 84x84 quanh xe -> grayscale -> stack 4 frames
  -> tensor (4, 84, 84) -> CNN encoder -> feature vector 512d
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import TYPE_CHECKING, Optional

try:
    import pygame
except ImportError:
    pygame = None

if TYPE_CHECKING:
    from src.simulation.car import Car


class CameraSensor:
    """
    Camera gia lap: chup vung pixel quanh xe tu Pygame surface.

    Args:
        capture_size: Kich thuoc vung chup (pixels, vuong)
        frame_stack: So frames gom lai (default 4)
        grayscale: Chuyen sang anh xam (1 channel)
    """

    def __init__(
        self,
        capture_size: int = 84,
        frame_stack: int = 4,
        grayscale: bool = True,
    ):
        self.capture_size = capture_size
        self.frame_stack = frame_stack
        self.grayscale = grayscale

        # Buffer luu cac frame gan nhat
        channels = 1 if grayscale else 3
        blank = np.zeros((channels, capture_size, capture_size), dtype=np.float32)
        self._frames: deque = deque(
            [blank.copy() for _ in range(frame_stack)],
            maxlen=frame_stack,
        )

    def reset(self) -> None:
        """Reset frame buffer (dung khi bat dau episode moi)."""
        channels = 1 if self.grayscale else 3
        blank = np.zeros(
            (channels, self.capture_size, self.capture_size),
            dtype=np.float32,
        )
        self._frames = deque(
            [blank.copy() for _ in range(self.frame_stack)],
            maxlen=self.frame_stack,
        )

    def capture(self, surface: "pygame.Surface", car: "Car") -> np.ndarray:
        """
        Chup 1 frame tu Pygame surface, centered quanh vi tri xe.

        Args:
            surface: Pygame surface (man hinh game)
            car: Agent xe (can x, y de xac dinh tam)

        Returns:
            np.ndarray shape (C, H, W) — anh da xu ly (0.0 - 1.0)
        """
        if pygame is None:
            raise RuntimeError("Pygame not available for camera capture")

        half = self.capture_size // 2
        sw, sh = surface.get_size()

        # Tinh vung crop (clamp vao trong surface)
        left = max(0, int(car.x) - half)
        top  = max(0, int(car.y) - half)
        right = min(sw, left + self.capture_size)
        bottom = min(sh, top + self.capture_size)

        # Dieu chinh neu vuot bien
        if right - left < self.capture_size:
            left = max(0, right - self.capture_size)
        if bottom - top < self.capture_size:
            top = max(0, bottom - self.capture_size)

        # Crop tu surface
        rect = pygame.Rect(left, top, self.capture_size, self.capture_size)
        subsurface = surface.subsurface(rect)

        # Chuyen sang numpy array
        pixels = pygame.surfarray.array3d(subsurface)  # (W, H, 3)
        pixels = np.transpose(pixels, (1, 0, 2))       # (H, W, 3)

        # Chuyen sang grayscale neu can
        if self.grayscale:
            # Luminance formula: 0.2989*R + 0.5870*G + 0.1140*B
            gray = (
                0.2989 * pixels[:, :, 0]
                + 0.5870 * pixels[:, :, 1]
                + 0.1140 * pixels[:, :, 2]
            )
            frame = gray[np.newaxis, :, :].astype(np.float32) / 255.0
        else:
            frame = np.transpose(pixels, (2, 0, 1)).astype(np.float32) / 255.0

        return frame

    def observe(self, surface: "pygame.Surface", car: "Car") -> np.ndarray:
        """
        Chup frame moi va tra ve stacked observation.

        Returns:
            np.ndarray shape (frame_stack * C, H, W)
            Vi du: grayscale + stack 4 -> (4, 84, 84)
        """
        frame = self.capture(surface, car)
        self._frames.append(frame)

        # Stack tat ca frames: (stack*C, H, W)
        stacked = np.concatenate(list(self._frames), axis=0)
        return stacked

    @property
    def observation_shape(self) -> tuple:
        """Shape cua observation tensor."""
        channels = 1 if self.grayscale else 3
        return (self.frame_stack * channels, self.capture_size, self.capture_size)
