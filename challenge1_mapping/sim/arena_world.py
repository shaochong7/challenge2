"""
Simulated arena layout for dry-run.

Landing pads are placed at fixed world (N, E) coordinates. The fake camera
renders whichever pads fall inside its field of view at each waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimLandingPad:
    marker_id: int
    n: float
    e: float
    valid: bool


@dataclass(frozen=True)
class SimObstacle:
    """Axis-aligned box on the ground (N/E corners), height in meters."""
    n0: float
    e0: float
    n1: float
    e1: float
    height_m: float = 0.5


# A more realistic practice layout: five landing sites plus extra invalid markers
# and several obstacles spread across a 4 m x 4 m arena.
DEFAULT_PADS = [
    SimLandingPad(11, 0.45, 0.45, True),
    SimLandingPad(45, 1.55, 0.95, True),
    SimLandingPad(51, 2.85, 0.55, True),
    SimLandingPad(67, 3.45, 2.15, True),
    SimLandingPad(101, 0.75, 3.35, True),
    SimLandingPad(201, 1.15, 2.45, False),
    SimLandingPad(202, 2.25, 1.85, False),
    SimLandingPad(203, 3.35, 3.35, False),
]

DEFAULT_OBSTACLES = [
    SimObstacle(0.85, 0.25, 1.25, 0.65, height_m=0.45),
    SimObstacle(0.55, 1.55, 1.05, 2.05, height_m=0.65),
    SimObstacle(1.65, 2.75, 2.15, 3.35, height_m=0.50),
    SimObstacle(2.55, 1.15, 3.10, 1.55, height_m=0.75),
    SimObstacle(3.20, 2.65, 3.70, 3.05, height_m=0.40),
]
