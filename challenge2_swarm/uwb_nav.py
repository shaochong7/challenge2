"""
UWB-guided navigation for HULA drones via pyhulax discrete move() commands.

Uses the same P-controller math as the mapping drone (compute_nav_velocity),
but outputs Direction + speed for pyhulax instead of MAVSDK velocity setpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass
class NavTickResult:
    at_goal: bool
    ready: bool
    current_n: float
    current_e: float
    direction: object | None = None
    speed: float = 0.0


def velocity_to_direction(vn: float, ve: float, max_speed: float):
    mag = math.hypot(vn, ve)
    if mag < 1e-6:
        return Direction.FORWARD, 0.0
    speed = min(mag, max_speed)
    if abs(vn) >= abs(ve):
        direction = Direction.FORWARD if vn > 0 else Direction.BACKWARD
    else:
        direction = Direction.RIGHT if ve > 0 else Direction.LEFT
    return direction, speed


def uwb_nav_tick(
    uwb: UWBSource,
    tag_id: int,
    target_n: float,
    target_e: float,
    gains: NavGains,
    max_speed: float,
) -> NavTickResult:
    n, e, ready = uwb.get_tag_ne(tag_id)
    if not ready:
        return NavTickResult(at_goal=False, ready=False, current_n=n, current_e=e)

    vn, ve, _, at_goal = compute_nav_velocity(
        target_n - n, target_e - e, 0.0, gains, ignore_height=True
    )
    if at_goal:
        return NavTickResult(at_goal=True, ready=True, current_n=n, current_e=e)

    direction, speed = velocity_to_direction(vn, ve, max_speed)
    return NavTickResult(
        at_goal=False,
        ready=True,
        current_n=n,
        current_e=e,
        direction=direction,
        speed=speed,
    )


def apply_nav_tick(api, tick: NavTickResult, *, min_speed: float = 0.05) -> None:
    if not tick.ready:
        api.hover()
        return
    if tick.at_goal:
        api.hover()
        return
    speed = max(tick.speed, min_speed) if tick.speed > 0 else 0.0
    if speed <= 0.0:
        api.hover()
        return
    api.move(tick.direction, speed)
