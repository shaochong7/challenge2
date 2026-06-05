"""Arena map world-coordinate logic (no hardware)."""

from challenge1_mapping.arena_map import (
    ArenaMap,
    ArenaMapConfig,
    marker_world_position,
)


def test_marker_world_position_down_camera():
    # Drone at (2, 3), marker 1 m to image-right, 0.5 m image-down
    n, e = marker_world_position(2.0, 3.0, x_m=1.0, y_m=0.5)
    assert abs(e - 4.0) < 1e-6     # East = drone_e + x_m
    assert abs(n - 1.5) < 1e-6     # North = drone_n - y_m (default north_sign=-1)


def test_marker_world_position_sign_override():
    cfg = ArenaMapConfig(north_sign=1.0)
    n, e = marker_world_position(0.0, 0.0, x_m=0.0, y_m=0.5, cfg=cfg)
    assert abs(n - 0.5) < 1e-6


def test_world_to_cell_top_left_is_n_max_e_min():
    arena = ArenaMap(ArenaMapConfig(n_min=-5, n_max=5, e_min=-5, e_max=5, resolution_m=0.05))
    r, c = arena.world_to_cell(5.0, -5.0)  # top-left corner
    assert r == 0 and c == 0


def test_stamp_obstacle_marks_cell():
    arena = ArenaMap(ArenaMapConfig())
    arena.stamp_obstacle(0.0, 0.0)
    r, c = arena.world_to_cell(0.0, 0.0)
    assert arena.occupancy[r, c] == 255


def test_out_of_bounds_ignored():
    arena = ArenaMap(ArenaMapConfig(n_min=-1, n_max=1, e_min=-1, e_max=1, resolution_m=0.05))
    arena.stamp_obstacle(100.0, 100.0)  # far outside
    assert arena.occupancy.sum() == 0


def test_render_has_landing_pads():
    arena = ArenaMap(ArenaMapConfig())
    arena.add_path_point(0.0, 0.0)
    arena.add_path_point(1.0, 1.0)
    arena.add_landing_pad(3, True, 1.0, 1.0)
    arena.add_landing_pad(10, False, -1.0, -1.0)
    img = arena.render_bgr()
    assert img.shape[2] == 3
    # green channel set somewhere (valid pad), red channel set somewhere (invalid)
    assert (img[:, :, 1] > 100).any()
    assert (img[:, :, 2] > 100).any()
