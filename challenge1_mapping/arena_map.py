"""
Arena map in UWB world coordinates (North/East), assembled during the survey.

The mapping drone's camera looks straight down (organizer "top-down depth map"),
so a marker seen at camera-frame offset (x_m right, y_m down) maps to world:
    East  = drone_E + x_m
    North = drone_N - y_m
(z_m is height above the pad, not used for the 2-D map.)

This whole module is pure numpy/OpenCV — unit-testable without any hardware.
Sign conventions are configurable because they depend on how the camera is
physically mounted; confirm in the arena and flip via ArenaMapConfig if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ArenaMapConfig:
    n_min: float = -5.0
    n_max: float = 5.0
    e_min: float = -5.0
    e_max: float = 5.0
    resolution_m: float = 0.05
    east_sign: float = 1.0   # camera x_m -> East
    north_sign: float = -1.0  # camera y_m -> North (image-down is South)


def marker_world_position(
    drone_n: float,
    drone_e: float,
    x_m: float,
    y_m: float,
    cfg: ArenaMapConfig | None = None,
) -> tuple[float, float]:
    """Camera-frame marker offset -> world (North, East) for a down-facing camera."""
    c = cfg or ArenaMapConfig()
    world_e = drone_e + c.east_sign * x_m
    world_n = drone_n + c.north_sign * y_m
    return world_n, world_e


@dataclass
class LandingPad:
    marker_id: int
    valid: bool
    n: float
    e: float


class ArenaMap:
    """Top-down occupancy + landing-pad map indexed by UWB North/East."""

    def __init__(self, config: ArenaMapConfig | None = None) -> None:
        self.cfg = config or ArenaMapConfig()
        self.rows = int(round((self.cfg.n_max - self.cfg.n_min) / self.cfg.resolution_m))
        self.cols = int(round((self.cfg.e_max - self.cfg.e_min) / self.cfg.resolution_m))
        self.occupancy = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.path: list[tuple[float, float]] = []
        self.pads: list[LandingPad] = []

    def in_bounds(self, n: float, e: float) -> bool:
        return self.cfg.n_min <= n < self.cfg.n_max and self.cfg.e_min <= e < self.cfg.e_max

    def world_to_cell(self, n: float, e: float) -> tuple[int, int]:
        """Return (row, col). North increases upward (row 0 = top = n_max)."""
        col = int((e - self.cfg.e_min) / self.cfg.resolution_m)
        row = int((self.cfg.n_max - n) / self.cfg.resolution_m)
        return row, col

    def stamp_obstacle(self, n: float, e: float) -> None:
        if not self.in_bounds(n, e):
            return
        r, c = self.world_to_cell(n, e)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.occupancy[r, c] = 255

    def add_path_point(self, n: float, e: float) -> None:
        self.path.append((n, e))

    def add_landing_pad(self, marker_id: int, valid: bool, n: float, e: float) -> None:
        self.pads.append(LandingPad(marker_id=marker_id, valid=valid, n=n, e=e))

    def render_bgr(self) -> np.ndarray:
        img = cv2.cvtColor(self.occupancy, cv2.COLOR_GRAY2BGR)

        for i in range(1, len(self.path)):
            r0, c0 = self.world_to_cell(*self.path[i - 1])
            r1, c1 = self.world_to_cell(*self.path[i])
            cv2.line(img, (c0, r0), (c1, r1), (255, 200, 0), 1)

        for pad in self.pads:
            r, c = self.world_to_cell(pad.n, pad.e)
            color = (0, 200, 0) if pad.valid else (0, 0, 255)
            cv2.circle(img, (c, r), 5, color, -1)
            cv2.putText(
                img, str(pad.marker_id), (c + 6, r),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )
        return img
