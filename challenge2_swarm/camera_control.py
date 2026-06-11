"""Camera setup helpers for HULA ground-marker detection."""

from __future__ import annotations


def _camera_pitch_mode(mode_name: str):
    from pyhulax.core import CameraPitchMode

    normalized = mode_name.strip().upper()
    if not normalized:
        normalized = "DOWN_ABSOLUTE"
    try:
        return getattr(CameraPitchMode, normalized)
    except AttributeError as exc:
        valid = [name for name in dir(CameraPitchMode) if name.isupper()]
        raise ValueError(
            f"Unknown camera pitch mode {mode_name!r}; expected one of {valid}"
        ) from exc


def configure_ground_marker_camera(api, swarm_cfg: dict, *, label: str = "HULA") -> bool:
    """Point the HULA camera down for Challenge 2 ArUco/ground-robot markers.

    Returns True if a command was sent. Camera pitch support can vary by drone
    firmware, so hardware errors are logged and treated as non-fatal.
    """

    if not bool(swarm_cfg.get("camera_pitch_enabled", True)):
        return False

    mode_name = str(swarm_cfg.get("camera_pitch_mode", "DOWN_ABSOLUTE"))
    angle = int(swarm_cfg.get("camera_pitch_angle_deg", 90))
    angle = max(0, min(90, angle))

    try:
        mode = _camera_pitch_mode(mode_name)
        result = api.set_camera_angle(mode, angle)
    except Exception as exc:
        print(f"{label}: camera pitch setup skipped ({exc})")
        return False

    print(f"{label}: camera pitch {mode_name.upper()} {angle} deg -> {result}")
    return True
