"""UWB navigation for swarm (C2)."""

from common.uwb_c2 import SimulatedUWBC2, parser_xy_to_ne
from common.velocity_nav import NavGains
from challenge2_swarm.uwb_nav import uwb_nav_tick, velocity_to_direction


def test_parser_xy_to_ne():
    n, e = parser_xy_to_ne(3.0, 5.0)
    assert n == 5.0 and e == 3.0


def test_velocity_to_direction_forward():
    from challenge2_swarm.uwb_nav import Direction

    d, s = velocity_to_direction(0.3, 0.05, max_speed=0.5)
    assert d == Direction.FORWARD
    assert abs(s - 0.3) < 0.01


def test_velocity_to_direction_right():
    from challenge2_swarm.uwb_nav import Direction

    d, s = velocity_to_direction(0.05, 0.4, max_speed=0.5)
    assert d == Direction.RIGHT


def test_uwb_nav_tick_at_goal():
    uwb = SimulatedUWBC2({0: (1.0, 2.0)})
    tick = uwb_nav_tick(uwb, 0, 1.0, 2.0, NavGains(), max_speed=0.5)
    assert tick.at_goal is True


def test_uwb_nav_tick_moves_toward_target():
    uwb = SimulatedUWBC2({0: (0.0, 0.0)})
    tick = uwb_nav_tick(uwb, 0, 1.0, 0.0, NavGains(), max_speed=0.5)
    assert tick.at_goal is False
    assert tick.direction is not None
    assert tick.speed > 0
