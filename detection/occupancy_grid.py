"""
Top-down obstacle grid from a down-facing RealSense depth image.

The camera looks down at the arena floor:
    RealSense frame: Z is optical/down, X is image-right, Y is image-down
    Local map: North = -Y, East = +X, centered on the drone/reference point

The grid marks surfaces that are closer than the estimated floor plane by at
least `min_obstacle_height_m`. This makes obstacle distance measurements useful
for the Challenge 1 map instead of treating vertical camera depth as forward
ground distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridConfig:
    min_depth_m: float = 0.05
    max_depth_m: float = 5.0
    resolution_m: float = 0.05  # meters per cell
    width_cells: int = 200      # 10 m East/West
    height_cells: int = 200     # 10 m North/South
    min_obstacle_height_m: float = 0.10
    floor_percentile: float = 90.0
    denoise: bool = True


@dataclass(frozen=True)
class ObstaclePoint:
    """Obstacle top point in the camera frame."""

    x_m: float       # camera right, normally +East
    y_m: float       # image down, normally -North
    height_m: float  # estimated height above floor
    distance_m: float


def _obstacle_projection(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cfg: GridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth_m.shape[:2]
    u_coords, v_coords = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )

    valid = (depth_m >= cfg.min_depth_m) & (depth_m < cfg.max_depth_m)
    valid_depth = depth_m[valid]
    if valid_depth.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    # For a downward camera, the floor is the farthest dominant surface.
    floor_depth_m = float(np.percentile(valid_depth, cfg.floor_percentile))
    height_m = floor_depth_m - depth_m
    obstacle = valid & (height_m >= cfg.min_obstacle_height_m)

    z = depth_m[obstacle]
    x = (u_coords[obstacle] - cx) * z / fx
    y = (v_coords[obstacle] - cy) * z / fy
    hgt = height_m[obstacle]
    return x.astype(np.float32), y.astype(np.float32), hgt.astype(np.float32)


def obstacle_points_from_depth(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: GridConfig | None = None,
) -> list[ObstaclePoint]:
    """Return one representative obstacle point per local map cell."""
    cfg = config or GridConfig()
    x, y, height = _obstacle_projection(depth_m, fx, fy, cx, cy, cfg)
    if x.size == 0:
        return []

    center_r = cfg.height_cells // 2
    center_c = cfg.width_cells // 2
    local_n = -y
    local_e = x
    cols = (local_e / cfg.resolution_m).astype(np.int32) + center_c
    rows = center_r - (local_n / cfg.resolution_m).astype(np.int32)
    in_grid = (
        (rows >= 0) & (rows < cfg.height_cells) &
        (cols >= 0) & (cols < cfg.width_cells)
    )

    points: dict[tuple[int, int], ObstaclePoint] = {}
    for r, c, px, py, ph in zip(
        rows[in_grid], cols[in_grid], x[in_grid], y[in_grid], height[in_grid]
    ):
        key = (int(r), int(c))
        dist = float(np.hypot(px, py))
        existing = points.get(key)
        if existing is None or float(ph) > existing.height_m:
            points[key] = ObstaclePoint(
                x_m=float(px),
                y_m=float(py),
                height_m=float(ph),
                distance_m=dist,
            )
    return list(points.values())


def build_occupancy_grid(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: GridConfig | None = None,
) -> np.ndarray:
    """
    depth_m: HxW depth in meters.
    Returns: uint8 grid, 255 = obstacle, 128 = drone/reference point.
    """
    cfg = config or GridConfig()
    occupancy = np.zeros((cfg.height_cells, cfg.width_cells), dtype=np.uint8)
    center_r = cfg.height_cells // 2
    center_c = cfg.width_cells // 2

    x, y, _height = _obstacle_projection(depth_m, fx, fy, cx, cy, cfg)
    if x.size:
        local_n = -y
        local_e = x
        gx = (local_e / cfg.resolution_m).astype(np.int32) + center_c
        gy = center_r - (local_n / cfg.resolution_m).astype(np.int32)
        in_grid = (
            (gx >= 0) & (gx < cfg.width_cells) &
            (gy >= 0) & (gy < cfg.height_cells)
        )
        occupancy[gy[in_grid], gx[in_grid]] = 255

    if cfg.denoise:
        kernel = np.ones((3, 3), np.uint8)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_OPEN, kernel)

    marker = np.zeros_like(occupancy)
    cv2.circle(marker, (center_c, center_r), 5, 128, -1)
    occupancy[(marker == 128) & (occupancy == 0)] = 128
    return occupancy
