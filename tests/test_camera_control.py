from challenge2_swarm.camera_control import configure_ground_marker_camera


class _Api:
    def __init__(self) -> None:
        self.calls = []

    def set_camera_angle(self, mode, angle):
        self.calls.append((mode.name, angle))
        return "ok"


def test_configure_ground_marker_camera_points_down():
    api = _Api()
    sent = configure_ground_marker_camera(
        api,
        {"camera_pitch_mode": "DOWN_ABSOLUTE", "camera_pitch_angle_deg": 200},
        label="test",
    )
    assert sent is True
    assert api.calls == [("DOWN_ABSOLUTE", 90)]


def test_configure_ground_marker_camera_can_be_disabled():
    api = _Api()
    sent = configure_ground_marker_camera(api, {"camera_pitch_enabled": False})
    assert sent is False
    assert api.calls == []
