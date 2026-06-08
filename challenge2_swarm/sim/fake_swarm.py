"""
Simulated HULA drones for Challenge 2 dry-run (no pyhulax, no WiFi).

Each fake drone has a UWB tag ID. move() nudges tag position in arena N/E.
"""

from __future__ import annotations

from common.uwb_c2 import SimulatedUWBC2

try:
    from pyhulax.core import Direction
except ImportError:

    class Direction:  # type: ignore[no-redef]
        FORWARD = "FORWARD"
        BACKWARD = "BACKWARD"
        LEFT = "LEFT"
        RIGHT = "RIGHT"


MOVE_SCALE = 0.18  # meters per move() tick at speed 1.0 (sim only; tune for laptop dry-run)


class FakeVideoFrame:
    def __init__(self, bgr):
        self._bgr = bgr

    def to_rgb(self):
        return self._bgr


class FakeVideoStream:
    def __init__(self) -> None:
        self.latest_frame = None

    def start(self) -> None:
        pass


class FakeDroneAPI:
    def __init__(self, tag_id: int, uwb: SimulatedUWBC2, ip: str) -> None:
        self.tag_id = tag_id
        self._uwb = uwb
        self.ip = ip
        self._airborne = False
        self._last_move = (Direction.FORWARD, 0.0)

    def connect(self, ip: str) -> None:
        self.ip = ip

    def set_video_stream(self, enabled: bool) -> None:
        pass

    def create_video_stream(self):
        return FakeVideoStream()

    def takeoff(self) -> None:
        self._airborne = True

    def land(self) -> None:
        self._airborne = False

    def hover(self) -> None:
        self._last_move = (Direction.FORWARD, 0.0)

    def move(self, direction, speed: float) -> None:
        self._last_move = (direction, speed)
        if not self._airborne or speed <= 0:
            return
        dn = de = 0.0
        step = MOVE_SCALE * speed
        if direction == Direction.FORWARD:
            dn = step
        elif direction == Direction.BACKWARD:
            dn = -step
        elif direction == Direction.RIGHT:
            de = step
        elif direction == Direction.LEFT:
            de = -step
        self._uwb.nudge_tag(self.tag_id, dn, de)


def build_fake_swarm(
    uwb: SimulatedUWBC2,
    tag_ids: list[int],
    start_positions: list[tuple[float, float]] | None = None,
) -> dict[str, FakeDroneAPI]:
    """Create fake drones keyed by fake IP."""
    contexts: dict[str, FakeDroneAPI] = {}
    for i, tag_id in enumerate(tag_ids):
        if start_positions and i < len(start_positions):
            n, e = start_positions[i]
            uwb.set_tag_ne(tag_id, n, e)
        else:
            uwb.set_tag_ne(tag_id, 0.0, float(i) * 0.3)
        ip = f"192.168.1.{100 + tag_id}"
        contexts[ip] = FakeDroneAPI(tag_id=tag_id, uwb=uwb, ip=ip)
    return contexts
