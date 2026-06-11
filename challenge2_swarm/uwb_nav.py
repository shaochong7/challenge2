"""
UWB-guided navigation for HULA drones via pyhulax discrete move() commands.

Uses the same P-controller math as the mapping drone (compute_nav_velocity),
but outputs Direction + speed for pyhulax instead of MAVSDK velocity setpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from challenge2_swarm.obstacle import DIR_DELTA, ObstacleSensor
from common.geofence import ArenaBounds
from common.uwb_c2 import UWBSource
from common.velocity_nav import NavGains, compute_nav_velocity

try:
    from pyhulax.core import Direction
except ImportError:

    class Direction:  # type: ignore[no-redef]
        FORWARD = "FORWARD"
        BACKWARD = "BACKWARD"
        LEFT = "LEFT"
        RIGHT = "RIGHT"


BACK_DIRECTION = getattr(Direction, "BACK", getattr(Direction, "BACKWARD", None))
if BACK_DIRECTION is None:
    raise AttributeError("pyhulax Direction must expose BACK or BACKWARD")


@dataclass
class NavTickResult:
    at_goal: bool
    ready: bool
    current_n: float
    current_e: float
    direction: object | None = None
    speed: float = 0.0
    geofence_violation: bool = False
    blocked: bool = False        # obstacle ahead and no clear way around this tick
    avoiding: bool = False       # moving around an obstacle (not straight to target)
    obstacle_distance_m: float = math.inf


def obstacle_speed_limit(
    distance_m: float,
    max_speed: float,
    slowdowns: list[tuple[float, float]] | None = None,
) -> float:
    """Limit speed by nearest obstacle distance.

    Thresholds are sorted closest-first so 0.10 m -> 0.1 m/s, then
    0.20 m -> 0.2 m/s, then 0.30 m -> 0.3 m/s.
    """

    if not math.isfinite(distance_m):
        return max_speed
    rules = slowdowns or [(0.10, 0.1), (0.20, 0.2), (0.30, 0.3)]
    for threshold_m, speed_mps in sorted(rules):
        if distance_m <= threshold_m:
            return min(max_speed, speed_mps)
    return max_speed


def velocity_to_direction(vn: float, ve: float, max_speed: float):
    mag = math.hypot(vn, ve)
    if mag < 1e-6:
        return Direction.FORWARD, 0.0
    speed = min(mag, max_speed)
    if abs(vn) >= abs(ve):
        direction = Direction.FORWARD if vn > 0 else BACK_DIRECTION
    else:
        direction = Direction.RIGHT if ve > 0 else Direction.LEFT
    return direction, speed


def candidate_directions(vn: float, ve: float) -> list:
    """Direction preference for reaching a target, then go-around fallbacks.

    Order: dominant-error axis, secondary-error axis, then the two lateral
    sidesteps perpendicular to the dominant axis (to route around an obstacle).
    """
    n_dir = Direction.FORWARD if vn > 0 else BACK_DIRECTION
    e_dir = Direction.RIGHT if ve > 0 else Direction.LEFT
    dirs: list = []
    if abs(vn) >= abs(ve):
        if abs(vn) > 1e-9:
            dirs.append(n_dir)
        if abs(ve) > 1e-9:
            dirs.append(e_dir)
        laterals = [e_dir, Direction.LEFT if e_dir == Direction.RIGHT else Direction.RIGHT]
    else:
        if abs(ve) > 1e-9:
            dirs.append(e_dir)
        if abs(vn) > 1e-9:
            dirs.append(n_dir)
        laterals = [n_dir, BACK_DIRECTION if n_dir == Direction.FORWARD else Direction.FORWARD]
    for d in laterals:
        if d not in dirs:
            dirs.append(d)
    return dirs


def uwb_nav_tick(
    uwb: UWBSource,
    tag_id: int,
    target_n: float,
    target_e: float,
    gains: NavGains,
    max_speed: float,
    geofence: ArenaBounds | None = None,
    obstacle_sensor: ObstacleSensor | None = None,
    stop_distance: float = 0.0,
    slowdowns: list[tuple[float, float]] | None = None,
) -> NavTickResult:
    n, e, ready = uwb.get_tag_ne(tag_id)
    if not ready:
        return NavTickResult(at_goal=False, ready=False, current_n=n, current_e=e)

    if geofence is not None and not geofence.in_anchor_zone(n, e):
        return NavTickResult(
            at_goal=False,
            ready=True,
            current_n=n,
            current_e=e,
            geofence_violation=True,
        )

    vn, ve, _, at_goal = compute_nav_velocity(
        target_n - n, target_e - e, 0.0, gains, ignore_height=True
    )
    if at_goal:
        return NavTickResult(at_goal=True, ready=True, current_n=n, current_e=e)

    speed = min(math.hypot(vn, ve), max_speed)

    if obstacle_sensor is None:
        direction, speed = velocity_to_direction(vn, ve, max_speed)
        return NavTickResult(
            at_goal=False, ready=True, current_n=n, current_e=e,
            direction=direction, speed=speed,
        )

    # Obstacle-aware: pick the first candidate direction that is not in the
    # hard-stop zone, then reduce speed based on proximity.
    prefs = candidate_directions(vn, ve)
    for idx, direction in enumerate(prefs):
        dn, de = DIR_DELTA[direction]
        distance_m = obstacle_sensor.distance_ahead(n, e, dn, de)
        if distance_m > stop_distance:
            limited_speed = obstacle_speed_limit(distance_m, speed, slowdowns)
            return NavTickResult(
                at_goal=False, ready=True, current_n=n, current_e=e,
                direction=direction,
                speed=limited_speed,
                avoiding=(idx > 0),
                obstacle_distance_m=distance_m,
            )
    # Every direction toward/around the obstacle is blocked -> hold position.
    return NavTickResult(
        at_goal=False, ready=True, current_n=n, current_e=e, blocked=True,
    )


def apply_nav_tick(api, tick: NavTickResult, *, min_speed: float = 0.05) -> bool:
    """Apply one nav tick. Returns True if geofence was violated."""
    def _hover() -> None:
        try:
            api.hover(1, blocking=False)
        except TypeError:
            try:
                api.hover(1)
            except TypeError:
                api.hover()

    if not tick.ready:
        # No UWB position yet. The drone holds position after takeoff; repeatedly
        # sending hover commands here just spams pyhulax while waiting for UWB.
        return False

    if tick.geofence_violation:
        _hover()
        return True
    if tick.at_goal or tick.blocked:
        # blocked = obstacle ahead with no clear way around -> hold, don't fly over
        _hover()
        return False
    speed = max(tick.speed, min_speed) if tick.speed > 0 else 0.0
    if speed <= 0.0:
        _hover()
        return False
    api.move(tick.direction, speed)
    return False


def hover_hold(api) -> None:
    """Hold briefly across pyhulax versions."""
    try:
        api.hover(1, blocking=False)
    except TypeError:
        try:
            api.hover(1)
        except TypeError:
            api.hover()
