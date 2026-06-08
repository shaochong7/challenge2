"""Shared swarm state machine — real hardware and dry-run."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from challenge2_swarm.uwb_nav import apply_nav_tick, uwb_nav_tick
from common.config_loader import load_config
from common.uwb_c2 import UWBSource
from common.velocity_nav import NavGains
from detection.target_detector import TargetDetector

try:
    from pyhulax.core import Direction
except ImportError:
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
    tag_id: int
    stream: object | None = None
    state: DroneState = DroneState.IDLE
    state_entered: float = field(default_factory=time.time)
    snapshots_taken: int = 0
    target_n: float = 0.0
    target_e: float = 0.0


def load_landing_zones(report_path: Path | None = None) -> list[dict]:
    report = report_path or (
        Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    )
    if not report.exists():
        return []
    data = json.loads(report.read_text(encoding="utf-8"))
    zones = data.get("valid_landing_zones", [])
    if not zones:
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


def run_swarm_loop(
    contexts: dict[str, DroneContext],
    uwb: UWBSource,
    cfg: dict,
    *,
    simulated: bool = False,
) -> None:
    swarm_cfg = cfg["swarm"]
    nav_cfg = cfg["navigation"]
    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    land_th = float(swarm_cfg.get("landing_threshold_m", 0.05))
    gains.n_threshold = land_th
    gains.e_threshold = land_th

    snapshot_dir = Path(swarm_cfg.get("snapshot_dir", "output/snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    detector = TargetDetector(swarm_cfg.get("yolo_weights"))

    landing_zones = load_landing_zones()
    ips = list(contexts.keys())
    for i, ip in enumerate(ips):
        ctx = contexts[ip]
        _set_state(ctx, DroneState.TAKEOFF)
        if i < len(landing_zones):
            ctx.target_n = float(landing_zones[i].get("n", 0))
            ctx.target_e = float(landing_zones[i].get("e", 0))
            print(
                f"Drone {ip} tag={ctx.tag_id} -> zone N={ctx.target_n:.2f} E={ctx.target_e:.2f}"
            )

    takeoff_wait = float(swarm_cfg.get("takeoff_wait_s", 5))
    move_speed = float(swarm_cfg.get("move_speed", 0.5))
    nav_timeout = float(swarm_cfg.get("uwb_nav_timeout_s", 120))
    search_timeout = float(swarm_cfg.get("search_timeout_s", 15))
    min_move_speed = float(swarm_cfg.get("min_move_speed", 0.05))

    mode = "SIM" if simulated else "LIVE"
    print(f"Swarm mission ({mode}) — Ctrl+C to stop")

    try:
        while any(c.state != DroneState.DONE for c in contexts.values()):
            for ip, ctx in contexts.items():
                api = ctx.api
                stream = ctx.stream

                if ctx.state == DroneState.TAKEOFF:
                    api.takeoff()
                    if _elapsed(ctx) >= (0.5 if simulated else takeoff_wait):
                        _set_state(ctx, DroneState.MOVE_TO_ZONE)

                elif ctx.state == DroneState.MOVE_TO_ZONE:
                    if _elapsed(ctx) > nav_timeout:
                        print(f"{ip}: UWB nav timeout, proceeding to search")
                        api.hover()
                        _set_state(ctx, DroneState.SEARCH)
                        continue
                    tick = uwb_nav_tick(
                        uwb, ctx.tag_id, ctx.target_n, ctx.target_e, gains, move_speed
                    )
                    apply_nav_tick(api, tick, min_speed=min_move_speed)
                    if tick.at_goal:
                        print(f"{ip}: reached landing zone N={ctx.target_n:.2f} E={ctx.target_e:.2f}")
                        _set_state(ctx, DroneState.SEARCH)

                elif ctx.state == DroneState.SEARCH:
                    if Direction is not None:
                        api.move(Direction.RIGHT, move_speed * 0.5)
                    frame = stream.latest_frame if stream else None
                    if frame is not None:
                        bgr = frame.to_rgb()
                        if detector.detect(bgr):
                            _set_state(ctx, DroneState.SNAPSHOT)
                    elif simulated and _elapsed(ctx) >= search_timeout:
                        print(f"{ip}: sim search complete (no camera/YOLO)")
                        _set_state(ctx, DroneState.DONE)

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

            time.sleep(0.05 if simulated else 0.1)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        for ctx in contexts.values():
            try:
                ctx.api.land()
            except Exception:
                pass
