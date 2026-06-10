"""Top-down occupancy grid built from synthetic depth maps (no camera)."""

import numpy as np

from detection.occupancy_grid import (
    GridConfig,
    build_occupancy_grid,
    obstacle_points_from_depth,
)

# Typical RealSense-ish intrinsics for 640x480
FX = FY = 600.0
CX, CY = 320.0, 240.0


def test_grid_shape_and_camera_marker():
    depth = np.zeros((480, 640), dtype=np.float32)  # all invalid
    cfg = GridConfig()
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    assert grid.shape == (cfg.height_cells, cfg.width_cells)
    # Down-facing local map is centered on the drone/reference point.
    assert grid[cfg.height_cells // 2, cfg.width_cells // 2] == 128


def test_empty_when_depth_out_of_range():
    depth = np.full((480, 640), 100.0, dtype=np.float32)  # beyond max_depth
    grid = build_occupancy_grid(depth, FX, FY, CX, CY)
    # Only the camera marker should be set (128), no occupied (255)
    assert np.count_nonzero(grid == 255) == 0


def test_obstacle_registers_in_grid():
    depth = np.full((480, 640), 2.0, dtype=np.float32)  # floor
    depth[200:280, 280:360] = 1.5  # obstacle top, closer than floor
    cfg = GridConfig(denoise=False)
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    assert np.count_nonzero(grid == 255) > 0
    # Patch is near image center, so it should map near the drone reference.
    occupied_rows = np.where((grid == 255).any(axis=1))[0]
    assert abs(int(occupied_rows.mean()) - cfg.height_cells // 2) < 15


def test_centered_obstacle_maps_near_center_column():
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    depth[230:250, 310:330] = 1.5  # near principal point -> X/Y ~ 0
    cfg = GridConfig(denoise=False)
    grid = build_occupancy_grid(depth, FX, FY, CX, CY, cfg)
    cols = np.where((grid == 255).any(axis=0))[0]
    assert abs(int(cols.mean()) - cfg.width_cells // 2) < 5


def test_obstacle_points_export_distances():
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    depth[230:250, 310:330] = 1.5
    points = obstacle_points_from_depth(depth, FX, FY, CX, CY, GridConfig(denoise=False))
    assert points
    p = min(points, key=lambda item: item.distance_m)
    assert p.height_m > 0.4
    assert p.distance_m < 0.1
