"""Swarm FSM state ordering for the pad -> search flow."""

import pytest

from challenge2_swarm.obstacle import MapObstacleSensor, ObstacleBox
from challenge2_swarm.swarm_core import (
    DroneContext,
    DroneState,
    clamp_hover_height_m,
    load_landing_zones,
    phase_speed_limits,
    setup_obstacle_sensors,
)


def test_states_present():
    names = {s.name for s in DroneState}
    assert {
        "TAKEOFF",
        "GO_TO_ZONE",
        "LAND",
        "WAIT_FOR_SEARCH",
        "TAKEOFF_SEARCH",
        "SEARCH",
        "SNAPSHOT",
        "FINAL_LAND",
        "DONE",
    } <= names


def test_pad_landing_precedes_search():
    assert DroneState.GO_TO_ZONE < DroneState.LAND < DroneState.WAIT_FOR_SEARCH
    assert DroneState.WAIT_FOR_SEARCH < DroneState.TAKEOFF_SEARCH
    assert DroneState.TAKEOFF_SEARCH < DroneState.SEARCH < DroneState.FINAL_LAND


def _ctx() -> DroneContext:
    return DroneContext(ip="1.1.1.1", api=object(), tag_id=0)


def test_lidar_avoidance_refuses_to_fly_when_unwired():
    # brief: flying over obstacles invalidates the score -> must fail safe.
    contexts = {"a": _ctx()}
    cfg = {"obstacle_avoidance_enabled": True, "obstacle_source": "lidar"}
    with pytest.raises(RuntimeError):
        setup_obstacle_sensors(contexts, cfg)


def test_avoidance_disabled_skips_sensor_setup():
    contexts = {"a": _ctx()}
    setup_obstacle_sensors(contexts, {"obstacle_avoidance_enabled": False})
    assert contexts["a"].obstacle_sensor is None


def test_preset_sensor_is_kept():
    ctx = _ctx()
    ctx.obstacle_sensor = MapObstacleSensor([ObstacleBox(0, 0, 1, 1)])
    contexts = {"a": ctx}
    setup_obstacle_sensors(contexts, {"obstacle_avoidance_enabled": True, "obstacle_source": "lidar"})
    assert isinstance(ctx.obstacle_sensor, MapObstacleSensor)


def test_hover_height_is_capped_to_configured_limit():
    cfg = {"hover_height_m": 1.8, "hover_height_max_m": 0.9}
    assert clamp_hover_height_m(cfg) == 0.9


def test_hover_height_keeps_configured_safe_value():
    cfg = {"hover_height_m": 0.9, "hover_height_max_m": 1.1}
    assert clamp_hover_height_m(cfg) == 0.9


def test_search_speed_is_separate_and_capped_by_transit_speed():
    cfg = {"move_speed": 0.5, "search_move_speed": 0.3}
    assert phase_speed_limits(cfg) == (0.5, 0.3)
    cfg = {"move_speed": 0.4, "search_move_speed": 0.8}
    assert phase_speed_limits(cfg) == (0.4, 0.4)


def test_landing_zones_can_be_selected_from_five(tmp_path):
    report = tmp_path / "landing_pad_report.json"
    report.write_text(
        """
        {
          "valid_landing_zones": [
            {"marker_id": 11, "n": 4.4, "e": 1.35},
            {"marker_id": 45, "n": 7.85, "e": 1.3},
            {"marker_id": 51, "n": 4.4, "e": 4.4},
            {"marker_id": 67, "n": 8.7, "e": 1.95},
            {"marker_id": 101, "n": 7.85, "e": 4.4}
          ]
        }
        """,
        encoding="utf-8",
    )
    zones = load_landing_zones(report, selected_marker_ids=[101, 11, 67])
    assert [zone["marker_id"] for zone in zones] == [101, 11, 67]
