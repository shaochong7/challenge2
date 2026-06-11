"""Obstacle sensing + go-around avoidance (Challenge 2, brief: no flying over)."""

import math

from challenge2_swarm.obstacle import (
    DIR_DELTA,
    HulaObstacleSensor,
    MapObstacleSensor,
    ObstacleBox,
)
from challenge2_swarm.uwb_nav import (
    Direction,
    candidate_directions,
    obstacle_speed_limit,
    uwb_nav_tick,
)
from common.uwb_c2 import SimulatedUWBC2
from common.velocity_nav import NavGains


def test_box_inflation_grows_footprint():
    b = ObstacleBox(0.0, 0.0, 1.0, 1.0).inflated(0.5)
    assert b.n0 == -0.5 and b.e1 == 1.5


def test_ray_distance_ahead_and_clear():
    s = MapObstacleSensor([ObstacleBox(2.0, -1.0, 3.0, 1.0)], clearance_m=0.0)
    # heading +North toward the box from (0,0): distance to n0=2.0
    dn, de = DIR_DELTA[Direction.FORWARD]
    assert abs(s.distance_ahead(0.0, 0.0, dn, de) - 2.0) < 1e-9
    # heading +East: box is not in the East lane -> clear
    dn, de = DIR_DELTA[Direction.RIGHT]
    assert s.distance_ahead(0.0, 0.0, dn, de) == math.inf


def test_candidate_directions_prefers_dominant_then_laterals():
    dirs = candidate_directions(0.4, 0.05)  # strong North, tiny East
    assert dirs[0] == Direction.FORWARD
    # lateral sidesteps (E/W) must both be present for go-around
    assert Direction.RIGHT in dirs and Direction.LEFT in dirs


def test_nav_tick_routes_around_blocking_obstacle():
    # Target is due North; a box blocks straight ahead -> must sidestep (not North).
    uwb = SimulatedUWBC2({0: (0.0, 0.0)})
    box = ObstacleBox(0.3, -0.5, 0.6, 0.5)
    sensor = MapObstacleSensor([box], clearance_m=0.0)
    tick = uwb_nav_tick(
        uwb, 0, 2.0, 0.0, NavGains(), max_speed=0.5,
        obstacle_sensor=sensor, stop_distance=0.5,
    )
    assert tick.direction != Direction.FORWARD
    assert tick.avoiding is True
    assert tick.blocked is False


def test_obstacle_speed_limit_ladder():
    assert obstacle_speed_limit(0.31, 0.5) == 0.5
    assert obstacle_speed_limit(0.30, 0.5) == 0.3
    assert obstacle_speed_limit(0.20, 0.5) == 0.2
    assert obstacle_speed_limit(0.10, 0.5) == 0.1


def test_nav_tick_slows_down_near_obstacle():
    uwb = SimulatedUWBC2({0: (0.0, 0.0)})
    box = ObstacleBox(0.25, -0.5, 0.6, 0.5)
    sensor = MapObstacleSensor([box], clearance_m=0.0)
    tick = uwb_nav_tick(
        uwb, 0, 10.0, 0.0, NavGains(), max_speed=0.5,
        obstacle_sensor=sensor, stop_distance=0.05,
    )
    assert tick.direction == Direction.FORWARD
    assert tick.speed == 0.3
    assert tick.obstacle_distance_m == 0.25


def test_nav_tick_blocked_when_all_directions_obstructed():
    # Surround the drone so every axis-aligned move hits a wall within stop_distance.
    uwb = SimulatedUWBC2({0: (0.0, 0.0)})
    boxes = [
        ObstacleBox(0.2, -1.0, 0.4, 1.0),   # North
        ObstacleBox(-0.4, -1.0, -0.2, 1.0),  # South
        ObstacleBox(-1.0, 0.2, 1.0, 0.4),   # East
        ObstacleBox(-1.0, -0.4, 1.0, -0.2),  # West
    ]
    sensor = MapObstacleSensor(boxes, clearance_m=0.0)
    tick = uwb_nav_tick(
        uwb, 0, 2.0, 2.0, NavGains(), max_speed=0.5,
        obstacle_sensor=sensor, stop_distance=0.5,
    )
    assert tick.blocked is True
    assert tick.direction is None


def test_hula_obstacle_sensor_fails_safe_when_unwired():
    sensor = HulaObstacleSensor(api=None)
    assert sensor.is_wired() is False
    # Unwired -> reports blocked (0.0) so we never assume a clear path.
    assert sensor.distance_ahead(0.0, 0.0, 1.0, 0.0) == 0.0


def test_hula_obstacle_sensor_uses_wired_reader():
    data = {Direction.FORWARD: 5.0, Direction.RIGHT: 0.1}
    sensor = HulaObstacleSensor(api=None, reader=lambda: data)
    assert sensor.is_wired() is True
    assert sensor.distance_ahead(0.0, 0.0, 1.0, 0.0) == 5.0
    assert sensor.distance_ahead(0.0, 0.0, 0.0, 1.0) == 0.1


def test_hula_obstacle_sensor_reads_pyhulax_obstacle_flags():
    class Obstacles:
        forward = True
        back = False
        left = False
        right = True

    class Api:
        def get_obstacles(self):
            return Obstacles()

    sensor = HulaObstacleSensor(api=Api())
    assert sensor.is_wired() is True
    assert sensor.distance_ahead(0.0, 0.0, 1.0, 0.0) == 0.0
    assert sensor.distance_ahead(0.0, 0.0, -1.0, 0.0) == math.inf
    assert sensor.distance_ahead(0.0, 0.0, 0.0, 1.0) == 0.0
    assert sensor.distance_ahead(0.0, 0.0, 0.0, -1.0) == math.inf


def test_hula_obstacle_sensor_reads_legacy_barrier_dict():
    class Server:
        def Plane_getBarrier(self):
            return {"forward": False, "back": True, "left": True, "right": False}

    class Api:
        _server = Server()

    sensor = HulaObstacleSensor(api=Api())
    assert sensor.is_wired() is True
    assert sensor.distance_ahead(0.0, 0.0, 1.0, 0.0) == math.inf
    assert sensor.distance_ahead(0.0, 0.0, -1.0, 0.0) == 0.0


def test_hula_obstacle_sensor_reads_numeric_distances():
    class Api:
        def get_obstacles(self):
            return {"forward_cm": 25, "left_m": 0.18, "right": False}

    sensor = HulaObstacleSensor(api=Api())
    assert sensor.distance_ahead(0.0, 0.0, 1.0, 0.0) == 0.25
    assert sensor.distance_ahead(0.0, 0.0, 0.0, -1.0) == 0.18
    assert sensor.distance_ahead(0.0, 0.0, 0.0, 1.0) == math.inf
