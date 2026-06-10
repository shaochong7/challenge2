"""Position-NED helpers for organiser move_it.py style mapping flight."""

from common.position_nav import world_to_local_ned


def test_world_to_local_ned_uses_home_as_origin():
    target = world_to_local_ned(
        target_n=4.0,
        target_e=1.5,
        target_d=-2.0,
        home_n=1.0,
        home_e=-0.5,
    )
    assert target.north_m == 3.0
    assert target.east_m == 2.0
    assert target.down_m == -2.0
