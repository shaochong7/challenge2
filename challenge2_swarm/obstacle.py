"""
Obstacle sensing + avoidance for the HULA swarm.

BRIEF RULE: "Strictly no flying over obstacles" — violation invalidates the score.
The HULA flies at a fixed low height (~1.1 m), so obstacles must be avoided
*horizontally* (go around), never by climbing over them.

Design: the nav layer asks an ObstacleSensor "how far is the nearest obstacle if I
move in direction (dn, de)?". Two implementations:

  MapObstacleSensor  -> known obstacle boxes in arena N/E (dry-run, or from the
                        Challenge 1 map). Pure + unit-testable.
  HulaObstacleSensor -> reads the HULA's onboard obstacle sensing (lidar) via
                        pyhulax. The exact SDK call is confirmed on the unit; until
                        wired it FAILS SAFE (reports blocked) so we never pretend the
                        path is clear and break the no-fly-over rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

try:
    from pyhulax.core import Direction
except ImportError:

    class Direction:  # type: ignore[no-redef]
        FORWARD = "FORWARD"
        BACK = "BACKWARD"
        LEFT = "LEFT"
        RIGHT = "RIGHT"


# Direction -> unit (dNorth, dEast)
DIR_DELTA = {
    Direction.FORWARD: (1.0, 0.0),
    Direction.BACK: (-1.0, 0.0),
    Direction.RIGHT: (0.0, 1.0),
    Direction.LEFT: (0.0, -1.0),
}


class ObstacleSensor(Protocol):
    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        """Meters to the nearest obstacle from (n,e) along unit dir (dn,de).
        Return math.inf if clear."""
        ...


@dataclass(frozen=True)
class ObstacleBox:
    """Axis-aligned obstacle footprint in arena N/E (meters)."""

    n0: float
    e0: float
    n1: float
    e1: float

    def inflated(self, margin: float) -> "ObstacleBox":
        return ObstacleBox(
            self.n0 - margin, self.e0 - margin, self.n1 + margin, self.e1 + margin
        )


def _ray_box_distance(
    n: float, e: float, dn: float, de: float, box: ObstacleBox
) -> float:
    """Distance from (n,e) to an axis-aligned box along an axis-aligned ray.

    Movement is quantized to N/E, so exactly one of dn/de is non-zero.
    """
    if dn > 0:  # heading +North
        if box.e0 <= e <= box.e1 and box.n1 >= n:
            return max(0.0, box.n0 - n)
    elif dn < 0:  # heading -North
        if box.e0 <= e <= box.e1 and box.n0 <= n:
            return max(0.0, n - box.n1)
    elif de > 0:  # heading +East
        if box.n0 <= n <= box.n1 and box.e1 >= e:
            return max(0.0, box.e0 - e)
    elif de < 0:  # heading -East
        if box.n0 <= n <= box.n1 and box.e0 <= e:
            return max(0.0, e - box.e1)
    return math.inf


class MapObstacleSensor:
    """Obstacle sensing from known boxes (sim, or Challenge 1 map)."""

    def __init__(self, boxes: list[ObstacleBox], clearance_m: float = 0.3) -> None:
        self.boxes = [b.inflated(clearance_m) for b in boxes]

    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        best = math.inf
        for box in self.boxes:
            best = min(best, _ray_box_distance(n, e, dn, de, box))
        return best


class HulaObstacleSensor:
    """HULA onboard obstacle sensing (lidar) via pyhulax.

    pyhulax usually exposes barrier sensors as directional boolean flags rather
    than precise distances. We adapt those flags to the nav interface:
      blocked -> 0.0 m
      clear   -> math.inf

    If a reader or future SDK build returns directional numeric distances
    (for example forward_m or forward_cm), those are used directly.

    FAIL SAFE: if no reader is available or a read fails, distance_ahead returns
    0.0 (blocked) so the drone refuses to advance into unknown space.
    """

    def __init__(self, api, reader=None) -> None:
        self.api = api
        self._wired = reader is not None or self._has_default_reader()
        self.reader = reader or self._default_reader
        self._warned = False

    def _has_default_reader(self) -> bool:
        return (
            hasattr(self.api, "get_obstacles")
            or hasattr(getattr(self.api, "_server", None), "Plane_getBarrier")
        )

    @staticmethod
    def _value(data, name: str):
        if isinstance(data, dict):
            return data.get(name)
        return getattr(data, name, None)

    @classmethod
    def _distance(cls, data, *names: str) -> float | None:
        for name in names:
            value = cls._value(data, name)
            if value is None:
                continue
            if isinstance(value, bool):
                return 0.0 if value else math.inf
            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue
            if name.endswith("_cm"):
                distance /= 100.0
            return max(0.0, distance)
        return None

    @classmethod
    def _flags_to_distances(cls, data) -> dict:
        forward = cls._distance(
            data,
            "forward_m", "front_m", "forward_distance_m", "front_distance_m",
            "forward_cm", "front_cm", "forward_distance_cm", "front_distance_cm",
            "forward", "front",
        )
        back = cls._distance(
            data,
            "back_m", "backward_m", "back_distance_m", "backward_distance_m",
            "back_cm", "backward_cm", "back_distance_cm", "backward_distance_cm",
            "back", "backward",
        )
        left = cls._distance(
            data,
            "left_m", "left_distance_m", "left_cm", "left_distance_cm", "left",
        )
        right = cls._distance(
            data,
            "right_m", "right_distance_m", "right_cm", "right_distance_cm", "right",
        )
        return {
            Direction.FORWARD: math.inf if forward is None else forward,
            Direction.BACK: math.inf if back is None else back,
            Direction.LEFT: math.inf if left is None else left,
            Direction.RIGHT: math.inf if right is None else right,
        }

    def _default_reader(self):
        if hasattr(self.api, "get_obstacles"):
            return self._flags_to_distances(self.api.get_obstacles())
        server = getattr(self.api, "_server", None)
        if hasattr(server, "Plane_getBarrier"):
            return self._flags_to_distances(server.Plane_getBarrier())
        return None

    def is_wired(self) -> bool:
        return self._wired

    def distance_ahead(self, n: float, e: float, dn: float, de: float) -> float:
        try:
            data = self.reader()
        except Exception as exc:  # never let a sensor read crash the loop
            if not self._warned:
                print(f"Obstacle sensor read failed ({exc}) — treating as blocked")
                self._warned = True
            return 0.0
        if not data:
            if not self._warned:
                print("Obstacle sensor not wired — failing safe (blocked)")
                self._warned = True
            return 0.0
        # Map (dn,de) back to a Direction key
        for direction, (ddn, dde) in DIR_DELTA.items():
            if (ddn, dde) == (dn, de):
                return float(data.get(direction, math.inf))
        return math.inf
