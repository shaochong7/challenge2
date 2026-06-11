"""UWB navigation for swarm (C2)."""

from common.geofence import ArenaBounds
from common.uwb_c2 import SimulatedUWBC2, UWBParserThreadC2, parser_xy_to_ne
from common.velocity_nav import NavGains
from challenge2_swarm.uwb_nav import uwb_nav_tick, velocity_to_direction


def test_parser_xy_to_ne():
    n, e = parser_xy_to_ne(3.0, 5.0)
    assert n == 5.0 and e == 3.0


def test_parser_xy_to_ne_applies_cage_origin():
    n, e = parser_xy_to_ne(5.5, 7.5, origin_x=5.5, origin_y=5.5)
    assert n == 2.0 and e == 0.0


def test_c2_parser_keeps_raw_z_and_applies_origin():
    parser = UWBParserThreadC2(origin_x=5.0, origin_y=0.0)
    data = bytearray([0x55, 0x00] + [0xFF] * (896 - 3) + [0xEE])
    offset = 2
    data[offset] = 7
    data[offset + 1] = 0
    data[offset + 2:offset + 5] = int(6000).to_bytes(3, "little", signed=True)
    data[offset + 5:offset + 8] = int(2000).to_bytes(3, "little", signed=True)
    data[offset + 8:offset + 11] = int(900).to_bytes(3, "little", signed=True)

    parser._parse_data(data)

    n, e, ready = parser.get_tag_ne(7)
    assert ready is True
    assert n == 2.0
    assert e == 1.0
    with parser._lock:
        raw_x, raw_y, raw_z, _ = parser._tag_data[7]
    assert raw_x == 6.0
    assert raw_y == 2.0
    assert raw_z == 0.9


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


def test_uwb_nav_tick_geofence_violation():
    uwb = SimulatedUWBC2({0: (20.0, 20.0)})
    geofence = ArenaBounds(0.0, 10.0, 0.0, 10.0)
    tick = uwb_nav_tick(uwb, 0, 5.0, 5.0, NavGains(), max_speed=0.5, geofence=geofence)
    assert tick.geofence_violation is True
