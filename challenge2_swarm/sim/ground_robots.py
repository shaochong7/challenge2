"""
Simulated RoboMaster convoy for Challenge 2 dry-run.

Robots loiter (slowly drift) within the arena, like the real convoy that
"loiters for a period of time" per the briefing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class GroundRobot:
    robot_id: int
    n0: float
    e0: float
    drift_radius: float = 0.05
    drift_speed: float = 0.2  # radians per sim second
    _t: float = 0.0

    def step(self, dt: float) -> None:
        self._t += dt

    def position(self) -> tuple[float, float]:
        # Small circular loiter around the spawn point
        n = self.n0 + self.drift_radius * math.sin(self.drift_speed * self._t)
        e = self.e0 + self.drift_radius * math.cos(self.drift_speed * self._t)
        return n, e


def default_convoy() -> list[GroundRobot]:
    """5 ground robots spread across a 1x1 m arena (the convoy)."""
    return [
        GroundRobot(0, 0.25, 0.20),
        GroundRobot(1, 0.50, 0.45),
        GroundRobot(2, 0.75, 0.30),
        GroundRobot(3, 0.40, 0.75),
        GroundRobot(4, 0.80, 0.80),
    ]


@dataclass
class RandomWalkGroundRobot:
    robot_id: int
    n: float
    e: float
    n_min: float
    n_max: float
    e_min: float
    e_max: float
    speed_mps: float = 0.25
    heading_rad: float = 0.0
    turn_interval_s: float = 1.5
    _rng: random.Random | None = None
    _t_until_turn: float = 0.0

    def __post_init__(self) -> None:
        if self._rng is None:
            self._rng = random.Random()
        self.heading_rad = self._rng.uniform(0.0, math.tau)
        self._t_until_turn = self._rng.uniform(0.2, self.turn_interval_s)

    def step(self, dt: float) -> None:
        assert self._rng is not None
        self._t_until_turn -= dt
        if self._t_until_turn <= 0.0:
            self.heading_rad = self._rng.uniform(0.0, math.tau)
            self._t_until_turn = self._rng.uniform(0.4, self.turn_interval_s)

        self.n += math.sin(self.heading_rad) * self.speed_mps * dt
        self.e += math.cos(self.heading_rad) * self.speed_mps * dt

        bounced = False
        if self.n < self.n_min:
            self.n = self.n_min
            bounced = True
        elif self.n > self.n_max:
            self.n = self.n_max
            bounced = True
        if self.e < self.e_min:
            self.e = self.e_min
            bounced = True
        elif self.e > self.e_max:
            self.e = self.e_max
            bounced = True
        if bounced:
            self.heading_rad = self._rng.uniform(0.0, math.tau)

    def position(self) -> tuple[float, float]:
        return self.n, self.e


def random_convoy(
    *,
    count: int = 5,
    n_min: float = 0.5,
    n_max: float = 9.5,
    e_min: float = 0.5,
    e_max: float = 4.5,
    seed: int = 7,
) -> list[RandomWalkGroundRobot]:
    rng = random.Random(seed)
    return [
        RandomWalkGroundRobot(
            robot_id=i,
            n=rng.uniform(n_min, n_max),
            e=rng.uniform(e_min, e_max),
            n_min=n_min,
            n_max=n_max,
            e_min=e_min,
            e_max=e_max,
            speed_mps=rng.uniform(0.12, 0.30),
            _rng=random.Random(rng.randrange(1_000_000)),
        )
        for i in range(count)
    ]
