"""
Challenge 2 — Swarm HULA drones (Semi-Final).

Based on organizer huladola.py:
  - Dola discovers drones on WiFi
  - pyhulax DroneAPI per IP
  - Per-drone finite state machine in one non-blocking loop
  - YOLO snapshots of RoboMaster convoy targets

States per drone:
  0 IDLE -> 1 TAKEOFF -> 2 MOVE_TO_ZONE -> 3 SEARCH_LOITER -> 4 SNAPSHOT -> 5 DONE
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import cv2

from challenge2_swarm.dola import Dola
from common.config_loader import load_config
from detection.target_detector import TargetDetector

try:
    from pyhulax import DroneAPI
    from pyhulax.core import Direction
except ImportError:
    DroneAPI = None  # type: ignore
    Direction = None  # type: ignore


class DroneState(IntEnum):
    IDLE = 0
    TAKEOFF = 1
    MOVE_TO_ZONE = 2
    SEARCH = 3
    SNAPSHOT = 4
    DONE = 5


@dataclass
class DroneContext:
    ip: str
    api: object
    stream: object | None = None
    state: DroneState = DroneState.IDLE
    state_entered: float = field(default_factory=time.time)
    snapshots_taken: int = 0
    target_n: float = 0.0
    target_e: float = 0.0


def _load_landing_zones() -> list[dict]:
    """Read the 3 valid landing zones (world N/E) produced by Challenge 1."""
    report = Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    if not report.exists():
        return []
    data = json.loads(report.read_text(encoding="utf-8"))
    zones = data.get("valid_landing_zones", [])
    if not zones:  # fall back to raw observations
        zones = [
            {"n": o.get("world_n", 0), "e": o.get("world_e", 0), "marker_id": o.get("marker_id")}
            for o in data.get("observations", [])
            if o.get("valid_landing")
        ]
    return zones[:3]


def _elapsed(ctx: DroneContext) -> float:
    return time.time() - ctx.state_entered


def _set_state(ctx: DroneContext, state: DroneState) -> None:
    ctx.state = state
    ctx.state_entered = time.time()


def discover_and_connect(cfg: dict) -> dict[str, DroneContext]:
    if DroneAPI is None:
        raise ImportError("Install pyhulax on the C2 machine")

    swarm = cfg["swarm"]
    dola = Dola()
    dola.start()
    try:
        if swarm.get("plane_ids"):
            ips = dola.get_ips_by_plane_ids(
                swarm["plane_ids"], listen_seconds=swarm.get("listen_seconds", 5)
            )
        else:
            ips = dola.get_all_ips(listen_seconds=swarm.get("listen_seconds", 5))
    finally:
        dola.stop()

    contexts: dict[str, DroneContext] = {}
    for plane_id, ip in ips.items():
        if not ip:
            print(f"Plane {plane_id}: not found")
            continue
        print(f"Plane {plane_id}: {ip}")
        api = DroneAPI()
        api.connect(ip)
        api.set_video_stream(True)
        stream = api.create_video_stream()
        if stream is not None:
            stream.start()
        contexts[str(ip)] = DroneContext(ip=str(ip), api=api, stream=stream)
    return contexts


def run_swarm_mission(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    swarm_cfg = cfg["swarm"]
    landing_obs = _load_landing_zones()
    snapshot_dir = Path(swarm_cfg.get("snapshot_dir", "output/snapshots"))
    detector = TargetDetector(swarm_cfg.get("yolo_weights"))

    contexts = discover_and_connect(cfg)
    if not contexts:
        print("No drones connected.")
        return

    ips = list(contexts.keys())
    for i, ip in enumerate(ips):
        ctx = contexts[ip]
        _set_state(ctx, DroneState.TAKEOFF)
        if i < len(landing_obs):
            ctx.target_n = float(landing_obs[i].get("n", 0))
            ctx.target_e = float(landing_obs[i].get("e", 0))

    takeoff_wait = float(swarm_cfg.get("takeoff_wait_s", 5))
    move_duration = float(swarm_cfg.get("move_duration_s", 2.0))
    move_speed = float(swarm_cfg.get("move_speed", 0.5))

    print("Swarm mission running — Ctrl+C to stop")
    try:
        while any(c.state != DroneState.DONE for c in contexts.values()):
            for ip, ctx in contexts.items():
                api = ctx.api
                stream = ctx.stream

                if ctx.state == DroneState.TAKEOFF:
                    api.takeoff()
                    if _elapsed(ctx) >= takeoff_wait:
                        _set_state(ctx, DroneState.MOVE_TO_ZONE)

                elif ctx.state == DroneState.MOVE_TO_ZONE:
                    # TODO: replace timed move with UWB position loop when UWB lib available on C2
                    api.move(Direction.FORWARD, move_speed)
                    if _elapsed(ctx) >= move_duration:
                        api.hover()
                        _set_state(ctx, DroneState.SEARCH)

                elif ctx.state == DroneState.SEARCH:
                    api.move(Direction.RIGHT, move_speed * 0.5)
                    frame = stream.latest_frame if stream else None
                    if frame is not None:
                        bgr = frame.to_rgb()
                        dets = detector.detect(bgr)
                        if dets:
                            _set_state(ctx, DroneState.SNAPSHOT)

                elif ctx.state == DroneState.SNAPSHOT:
                    frame = stream.latest_frame if stream else None
                    if frame is not None:
                        bgr = frame.to_rgb()
                        dets = detector.detect(bgr)
                        out = snapshot_dir / f"{ip.replace('.', '_')}_{ctx.snapshots_taken}.jpg"
                        detector.save_snapshot(bgr, out, dets)
                        ctx.snapshots_taken += 1
                        print(f"Snapshot {out} ({len(dets)} detections)")
                    api.hover()
                    _set_state(ctx, DroneState.DONE)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        for ctx in contexts.values():
            try:
                ctx.api.land()
            except Exception:
                pass


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_swarm_mission(path)


if __name__ == "__main__":
    main()
