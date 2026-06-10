"""Shared swarm state machine — real hardware and dry-run.

Objectives (Challenge 2):
  - find the ground-robot convoy and snapshot each robot (recon)
  - occupy every valid landing pad from Challenge 1 (deployment)

Flow per drone:

  TAKEOFF
    -> GO_TO_ZONE      fly to its assigned valid landing pad
    -> LAND            land on the pad (deployment score)
    -> TAKEOFF_SEARCH  take off again
    -> SEARCH          lawnmower coverage of its strip, snapshot robots it sees
    -> FINAL_LAND      land after search
    -> DONE
  SNAPSHOT is entered from any moving state and resumes that state afterwards.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from challenge2_swarm.obstacle import (
    HulaObstacleSensor,
    MapObstacleSensor,
    ObstacleBox,
)
from challenge2_swarm.search_pattern import Region, lawnmower_waypoints, split_region
from challenge2_swarm.uwb_nav import apply_nav_tick, uwb_nav_tick
from common.geofence import ArenaBounds, GeofenceViolation
from common.uwb_c2 import UWBSource
from common.velocity_nav import NavGains

try:
    from pyhulax.core import Direction
except ImportError:
    Direction = None  # type: ignore


class DroneState(IntEnum):
    IDLE = 0
    TAKEOFF = 1
    GO_TO_ZONE = 2
    LAND = 3
    TAKEOFF_SEARCH = 4
    SEARCH = 5
    SNAPSHOT = 6
    FINAL_LAND = 7
    DONE = 8


@dataclass
class DroneContext:
    ip: str
    api: object
    tag_id: int
    stream: object | None = None
    state: DroneState = DroneState.IDLE
    state_entered: float = field(default_factory=time.time)
    # landing zone (ambush position from Challenge 1)
    target_n: float = 0.0
    target_e: float = 0.0
    has_zone: bool = False
    pad_landed: bool = False
    landed: bool = False
    # search coverage
    search_waypoints: list[tuple[float, float]] = field(default_factory=list)
    search_idx: int = 0
    # results
    snapshots_taken: int = 0
    found_target_ids: set = field(default_factory=set)
    last_snapshot_t: float = 0.0
    _pending: list = field(default_factory=list)
    resume_state: DroneState = DroneState.SEARCH
    obstacle_sensor: object | None = None
    wp_started: float = 0.0  # when current search waypoint began (stall detection)


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


def load_arena_bounds(report_path: Path | None = None) -> dict | None:
    """Read arena_bounds from the Challenge 1 map so the swarm searches the
    exact area the mapping drone surveyed."""
    report = report_path or (
        Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    )
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    return data.get("arena_bounds")


def load_obstacle_boxes(report_path: Path | None = None) -> list[ObstacleBox]:
    """Read obstacle footprints (arena N/E boxes) from the Challenge 1 map, if the
    mapping drone exported them. Used when obstacle_source == 'map'."""
    report = report_path or (
        Path(__file__).resolve().parents[1] / "output" / "challenge1" / "landing_pad_report.json"
    )
    if not report.exists():
        return []
    data = json.loads(report.read_text(encoding="utf-8"))
    boxes = []
    for o in data.get("obstacles", []):
        try:
            boxes.append(ObstacleBox(o["n0"], o["e0"], o["n1"], o["e1"]))
        except (KeyError, TypeError):
            continue
    return boxes


def setup_obstacle_sensors(contexts: dict, swarm_cfg: dict) -> None:
    """Attach an obstacle sensor to each drone (unless one is already set, e.g. the
    dry-run injects a MapObstacleSensor). FAILS SAFE: if avoidance is enabled but no
    usable sensor is available, raises so the mission refuses to fly (the brief
    forbids flying over obstacles)."""
    if not bool(swarm_cfg.get("obstacle_avoidance_enabled", True)):
        return
    source = str(swarm_cfg.get("obstacle_source", "lidar")).lower()
    clearance = float(swarm_cfg.get("obstacle_clearance_m", 0.3))

    map_boxes = load_obstacle_boxes() if source == "map" else []
    for ctx in contexts.values():
        if ctx.obstacle_sensor is not None:
            continue
        if source == "map":
            if not map_boxes:
                raise RuntimeError(
                    "obstacle_source=map but no obstacles found in the Challenge 1 "
                    "report — re-run mapping or set obstacle_source: lidar"
                )
            ctx.obstacle_sensor = MapObstacleSensor(map_boxes, clearance)
        else:  # lidar
            sensor = HulaObstacleSensor(ctx.api)
            if not sensor.is_wired():
                raise RuntimeError(
                    "obstacle_avoidance_enabled with obstacle_source=lidar, but the "
                    "pyhulax obstacle-sensing reader is not wired (see "
                    "HulaObstacleSensor._default_reader). Refusing to fly — flying "
                    "over obstacles would invalidate the score. Wire the lidar reader "
                    "or set obstacle_avoidance_enabled: false to fly without it."
                )
            ctx.obstacle_sensor = sensor


def _elapsed(ctx: DroneContext) -> float:
    return time.time() - ctx.state_entered


def _set_state(ctx: DroneContext, state: DroneState) -> None:
    ctx.state = state
    ctx.state_entered = time.time()


def _search_area(swarm_cfg: dict, use_map_bounds: bool = True) -> Region:
    # Prefer the bounds the mapping drone actually surveyed (from its map);
    # fall back to the config search_area.
    a = (load_arena_bounds() if use_map_bounds else None) or swarm_cfg.get("search_area", {})
    return Region(
        n_min=float(a.get("n_min", 0.0)),
        n_max=float(a.get("n_max", 1.0)),
        e_min=float(a.get("e_min", 0.0)),
        e_max=float(a.get("e_max", 1.0)),
    )


def _validate_swarm_geofence(
    geofence: ArenaBounds | None,
    contexts: dict[str, DroneContext],
    landing_zones: list[dict],
    search_area: Region,
) -> None:
    if geofence is None:
        return
    geofence.validate_region(
        search_area.n_min, search_area.n_max, search_area.e_min, search_area.e_max, "search area"
    )
    points: list[tuple[float, float, str]] = []
    for i, z in enumerate(landing_zones):
        points.append((float(z["n"]), float(z["e"]), f"landing zone {i}"))
    for ip, ctx in contexts.items():
        for j, (wn, we) in enumerate(ctx.search_waypoints):
            points.append((wn, we, f"{ip} search wp {j}"))
    geofence.validate_ne_points(points)


def assign_search_regions(
    contexts: dict[str, DroneContext], swarm_cfg: dict
) -> Region:
    """Split the search area into per-drone strips and build lawnmower paths."""
    area = _search_area(swarm_cfg, use_map_bounds=bool(swarm_cfg.get("use_map_bounds", True)))
    spacing = float(swarm_cfg.get("search_spacing_m", 0.3))
    ips = list(contexts.keys())
    num = len(ips)
    for i, ip in enumerate(ips):
        region = split_region(area, num, i)
        contexts[ip].search_waypoints = lawnmower_waypoints(region, spacing)
    return area


def run_swarm_loop(
    contexts: dict[str, DroneContext],
    uwb: UWBSource,
    cfg: dict,
    sensor,
    *,
    simulated: bool = False,
    on_tick=None,
) -> None:
    swarm_cfg = cfg["swarm"]
    nav_cfg = cfg["navigation"]
    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    arrive_th = float(swarm_cfg.get("waypoint_threshold_m", 0.12))
    gains.n_threshold = arrive_th
    gains.e_threshold = arrive_th

    geofence = ArenaBounds.from_config(cfg)
    landing_zones = load_landing_zones()
    search_area = assign_search_regions(contexts, swarm_cfg)
    try:
        _validate_swarm_geofence(geofence, contexts, landing_zones, search_area)
    except GeofenceViolation as exc:
        print(f"Geofence preflight failed: {exc}")
        return

    # Obstacle avoidance (brief: strictly no flying over obstacles).
    try:
        setup_obstacle_sensors(contexts, swarm_cfg)
    except RuntimeError as exc:
        print(f"Obstacle-avoidance preflight failed: {exc}")
        return
    avoidance_on = bool(swarm_cfg.get("obstacle_avoidance_enabled", True))
    stop_distance = float(swarm_cfg.get("obstacle_stop_distance_m", 0.6))

    ips = list(contexts.keys())
    for i, ip in enumerate(ips):
        ctx = contexts[ip]
        _set_state(ctx, DroneState.TAKEOFF)
        if i < len(landing_zones):
            ctx.target_n = float(landing_zones[i].get("n", 0))
            ctx.target_e = float(landing_zones[i].get("e", 0))
            ctx.has_zone = True
            print(f"{ip}: assigned landing zone N={ctx.target_n:.2f} E={ctx.target_e:.2f}")
        else:
            print(f"{ip}: no landing zone assigned (will land in place)")

    takeoff_wait = float(swarm_cfg.get("takeoff_wait_s", 5))
    pad_land_wait = float(swarm_cfg.get("pad_land_wait_s", 1.0))
    move_speed = float(swarm_cfg.get("move_speed", 0.5))
    # Decouple the swarm speed cap from the mapping drone (which is limited to
    # 0.3 m/s). The HULA may fly up to its own move_speed (brief: 0.5 m/s).
    gains.max_vel_xy = move_speed
    hover_height_m = float(swarm_cfg.get("hover_height_m", 1.1))
    nav_timeout = float(swarm_cfg.get("uwb_nav_timeout_s", 120))
    # Skip a search waypoint we can't reach in time (e.g. blocked by an obstacle).
    wp_timeout = float(swarm_cfg.get("search_wp_timeout_s", min(30.0, nav_timeout)))
    min_move_speed = float(swarm_cfg.get("min_move_speed", 0.05))
    snapshot_cooldown = float(swarm_cfg.get("snapshot_cooldown_s", 1.0))
    dedup_dist = float(swarm_cfg.get("target_dedup_m", 0.25))
    do_area_search = bool(swarm_cfg.get("do_area_search", True))

    snapshot_dir = Path(swarm_cfg.get("snapshot_dir", "output/snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    mode = "SIM" if simulated else "LIVE"
    print(f"Swarm pad-first search mission ({mode}) - Ctrl+C to stop")

    def _nav_to(ctx, tn, te):
        tick = uwb_nav_tick(
            uwb, ctx.tag_id, tn, te, gains, move_speed, geofence=geofence,
            obstacle_sensor=ctx.obstacle_sensor if avoidance_on else None,
            stop_distance=stop_distance,
        )
        if tick.blocked:
            print(
                f"tag {ctx.tag_id}: obstacle ahead at N={tick.current_n:.2f} "
                f"E={tick.current_e:.2f} — holding (no fly-over)"
            )
        if apply_nav_tick(ctx.api, tick, min_speed=min_move_speed):
            print(
                f"tag {ctx.tag_id}: GEOFENCE — outside UWB anchors at "
                f"N={tick.current_n:.2f} E={tick.current_e:.2f}, EMERGENCY LAND in place"
            )
            try:
                ctx.api.land()
            except Exception:
                pass
            ctx.landed = True
            _set_state(ctx, DroneState.DONE)
        return tick

    def _new_targets(ctx, sensed):
        """Filter out robots already snapshotted (by id or proximity + cooldown)."""
        out = []
        for t in sensed:
            if t.target_id is not None:
                if t.target_id in ctx.found_target_ids:
                    continue
            else:
                if (time.time() - ctx.last_snapshot_t) < snapshot_cooldown:
                    continue
            out.append(t)
        return out

    def _detect_while_moving(ctx) -> bool:
        """Snapshot any new robot the camera sees. Returns True if it switched
        to SNAPSHOT (caller should stop moving this tick)."""
        new = _new_targets(ctx, sensor.sense(ctx))
        if new:
            ctx._pending = new
            ctx.resume_state = ctx.state
            _set_state(ctx, DroneState.SNAPSHOT)
            return True
        return False

    def _takeoff(api, height_m: float) -> None:
        """Take off to the recommended height. pyhulax build may or may not accept a
        height arg; try it, fall back to a plain takeoff (brief height: ~1.1 m)."""
        try:
            api.takeoff(height_m)
        except TypeError:
            api.takeoff()
        except Exception:
            api.takeoff()

    try:
        while any(c.state != DroneState.DONE for c in contexts.values()):
            for ip, ctx in contexts.items():
                api = ctx.api

                if ctx.state == DroneState.TAKEOFF:
                    _takeoff(api, hover_height_m)
                    if _elapsed(ctx) >= (0.5 if simulated else takeoff_wait):
                        if ctx.has_zone:
                            _set_state(ctx, DroneState.GO_TO_ZONE)
                        else:
                            ctx.search_idx = 0
                            ctx.wp_started = time.time()
                            _set_state(
                                ctx,
                                DroneState.SEARCH
                                if do_area_search and ctx.search_waypoints
                                else DroneState.FINAL_LAND,
                            )

                elif ctx.state == DroneState.GO_TO_ZONE:
                    tick = _nav_to(ctx, ctx.target_n, ctx.target_e)
                    if tick.at_goal or _elapsed(ctx) > nav_timeout:
                        _set_state(ctx, DroneState.LAND)

                elif ctx.state == DroneState.LAND:
                    if not ctx.pad_landed:
                        api.land()
                        ctx.pad_landed = True
                        ctx.landed = True
                        print(
                            f"{ip}: LANDED on mapped pad N={ctx.target_n:.2f} "
                            f"E={ctx.target_e:.2f}"
                        )
                    if do_area_search and ctx.search_waypoints:
                        if _elapsed(ctx) >= (0.2 if simulated else pad_land_wait):
                            _set_state(ctx, DroneState.TAKEOFF_SEARCH)
                    else:
                        _set_state(ctx, DroneState.DONE)

                elif ctx.state == DroneState.TAKEOFF_SEARCH:
                    _takeoff(api, hover_height_m)
                    ctx.landed = False
                    if _elapsed(ctx) >= (0.5 if simulated else takeoff_wait):
                        ctx.search_idx = 0
                        ctx.wp_started = time.time()
                        _set_state(ctx, DroneState.SEARCH)

                elif ctx.state == DroneState.SEARCH:
                    # snapshot robots seen during coverage (recon goal)
                    if _detect_while_moving(ctx):
                        continue
                    # advance along lawnmower coverage path
                    if ctx.search_idx >= len(ctx.search_waypoints):
                        _set_state(ctx, DroneState.FINAL_LAND)
                        continue
                    wn, we = ctx.search_waypoints[ctx.search_idx]
                    tick = _nav_to(ctx, wn, we)
                    stalled = (time.time() - ctx.wp_started) > wp_timeout
                    if tick.at_goal or stalled:
                        if stalled and not tick.at_goal:
                            print(
                                f"{ip}: search waypoint N={wn:.2f} E={we:.2f} "
                                f"unreachable (blocked) — skipping"
                            )
                        ctx.search_idx += 1
                        ctx.wp_started = time.time()

                elif ctx.state == DroneState.SNAPSHOT:
                    api.hover()
                    out = snapshot_dir / f"drone{ctx.tag_id}_snap{ctx.snapshots_taken:02d}.jpg"
                    count = sensor.save_snapshot(ctx, ctx._pending, out)
                    ctx.snapshots_taken += 1
                    ctx.last_snapshot_t = time.time()
                    for t in ctx._pending:
                        if t.target_id is not None:
                            ctx.found_target_ids.add(t.target_id)
                    ids = [t.target_id for t in ctx._pending]
                    print(f"{ip}: SNAPSHOT {out.name} targets={ids} ({count} boxes)")
                    ctx._pending = []
                    _set_state(ctx, ctx.resume_state)

                elif ctx.state == DroneState.FINAL_LAND:
                    api.land()
                    ctx.landed = True
                    print(
                        f"{ip}: FINAL LAND - {len(ctx.found_target_ids)} robots found, "
                        f"pad_landed={ctx.pad_landed}"
                    )
                    _set_state(ctx, DroneState.DONE)

                elif ctx.state == DroneState.LAND:
                    # precision-land on the assigned pad (step 8)
                    api.land()
                    ctx.landed = True
                    where = (
                        f"pad N={ctx.target_n:.2f} E={ctx.target_e:.2f}"
                        if ctx.has_zone
                        else "in place"
                    )
                    print(
                        f"{ip}: LANDED {where} — "
                        f"{len(ctx.found_target_ids)} robots found"
                    )
                    _set_state(ctx, DroneState.DONE)

            if on_tick is not None:
                on_tick(contexts)
            time.sleep(0.02 if simulated else 0.1)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        for ctx in contexts.values():
            if not ctx.landed:
                try:
                    ctx.api.land()
                except Exception:
                    pass
