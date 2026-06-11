"""
Laptop dry-run of Challenge 2 swarm — no pyhulax, WiFi, or UWB hardware.

Uses:
  - SimulatedUWBC2 (tag positions updated by fake move())
  - FakeDroneAPI (pyhulax stand-in)
  - landing_pad_report.json from Challenge 1 dry-run (or built-in fallbacks)

Run Challenge 1 dry-run first for best results:
    python scripts/dry_run_challenge1.py --fast
    python scripts/dry_run_challenge2.py --fast
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge2_swarm.obstacle import MapObstacleSensor, ObstacleBox
from challenge2_swarm.sim.fake_swarm import build_fake_swarm
from challenge2_swarm.sim.ground_robots import default_convoy, random_convoy
from challenge2_swarm.swarm_core import DroneContext, load_landing_zones, run_swarm_loop
from challenge2_swarm.target_sensor import SimTargetSensor
from common.config_loader import load_config
from common.uwb_c2 import SimulatedUWBC2

OUTPUT_LOG = ROOT / "output" / "challenge2" / "dry_run_log.txt"


def _n_tunnel_boxes(
    center_n: float,
    center_e: float,
    *,
    outer_x_m: float = 1.0,
    outer_y_m: float = 0.4,
    outer_z_m: float = 1.2,
    hole_x_m: float = 0.8,
    hole_z_m: float = 1.0,
) -> list[ObstacleBox]:
    """Represent a 3D n-shaped tunnel as 2D obstacle footprints.

    The flight sim is top-down, so z is kept in metadata/logs while collision
    avoidance sees the two side posts plus a shallow top lintel footprint.
    """

    post_w = max(0.05, (outer_x_m - hole_x_m) / 2.0)
    lintel_depth = max(0.05, outer_y_m * (outer_z_m - hole_z_m) / outer_z_m)
    n0 = center_n - outer_y_m / 2.0
    n1 = center_n + outer_y_m / 2.0
    e0 = center_e - outer_x_m / 2.0
    e1 = center_e + outer_x_m / 2.0
    return [
        ObstacleBox(n0=n0, e0=e0, n1=n1, e1=e0 + post_w),
        ObstacleBox(n0=n0, e0=e1 - post_w, n1=n1, e1=e1),
        ObstacleBox(n0=n1 - lintel_depth, e0=e0, n1=n1, e1=e1),
    ]


def _random_tunnels(
    *,
    count: int = 8,
    seed: int = 42,
    n_min: float = 0.7,
    n_max: float = 9.3,
    e_min: float = 0.7,
    e_max: float = 4.3,
) -> tuple[list[ObstacleBox], list[dict]]:
    rng = random.Random(seed)
    boxes: list[ObstacleBox] = []
    metadata: list[dict] = []
    attempts = 0
    while len(metadata) < count and attempts < count * 100:
        attempts += 1
        center_n = rng.uniform(n_min, n_max)
        center_e = rng.uniform(e_min, e_max)
        too_close = any(
            abs(center_n - item["center_n_y_m"]) < 1.2
            and abs(center_e - item["center_e_x_m"]) < 1.0
            for item in metadata
        )
        if too_close:
            continue
        tunnel_boxes = _n_tunnel_boxes(center_n, center_e)
        boxes.extend(tunnel_boxes)
        metadata.append(
            {
                "center_n_y_m": round(center_n, 3),
                "center_e_x_m": round(center_e, 3),
                "outer": {"x_m": 1.0, "y_m": 0.4, "z_m": 1.2},
                "hole": {"x_m": 0.8, "y_m": 0.4, "z_m": 1.0},
                "boxes": [box.__dict__ for box in tunnel_boxes],
            }
        )
    return boxes, metadata


def _ensure_landing_report() -> None:
    zones = load_landing_zones()
    if zones:
        print(f"Loaded {len(zones)} landing zones from Challenge 1 report")
        return
    print("No Challenge 1 report — using default zones (0.08,0.08), (0.92,0.08), (0.92,0.92)")
    report_dir = ROOT / "output" / "challenge1"
    report_dir.mkdir(parents=True, exist_ok=True)
    import json

    fallback = {
        "challenge": 1,
        "simulated": True,
        "valid_landing_zones": [
            {"marker_id": 0, "n": 0.08, "e": 0.08},
            {"marker_id": 1, "n": 0.92, "e": 0.08},
            {"marker_id": 2, "n": 0.92, "e": 0.92},
        ],
        "observations": [],
    }
    (report_dir / "landing_pad_report.json").write_text(
        json.dumps(fallback, indent=2), encoding="utf-8"
    )


def run_dry_swarm(
    config_path: str | None = None,
    fast: bool = False,
    full_scale: bool = False,
    seed: int = 42,
) -> None:
    _ensure_landing_report()
    cfg = load_config(config_path)
    swarm_cfg = cfg["swarm"]
    tag_ids = list(swarm_cfg.get("tag_ids", [0, 1, 2]))[:3]

    uwb = SimulatedUWBC2()
    uwb.start()

    # Start all drones near arena origin
    fake_apis = build_fake_swarm(uwb, tag_ids, start_positions=[(0.0, 0.0)] * len(tag_ids))
    contexts: dict[str, DroneContext] = {}
    for ip, api in fake_apis.items():
        contexts[ip] = DroneContext(
            ip=ip, api=api, tag_id=api.tag_id, stream=api.create_video_stream()
        )

    # Default sim runs on a compact 1x1 m arena with a tight footprint so it
    # stays fast. Full-scale mode uses the requested 5 m x 10 m x 3.5 m arena.
    cfg = dict(cfg)
    cfg["arena"] = dict(cfg.get("arena", {}))
    cfg["swarm"] = dict(swarm_cfg)
    cfg["arena"]["geofence_enabled"] = False
    if full_scale:
        cfg["arena"]["uwb_bounds"] = {"n_min": 0.0, "n_max": 10.0, "e_min": 0.0, "e_max": 5.0}
        cfg["swarm"]["search_area"] = {"n_min": 0.5, "n_max": 9.5, "e_min": 0.5, "e_max": 4.5}
        cfg["swarm"]["search_spacing_m"] = 1.0
        cfg["swarm"]["sim_camera_footprint_m"] = 1.0
        cfg["swarm"]["uwb_nav_timeout_s"] = 180
        cfg["swarm"]["search_wp_timeout_s"] = 12.0
    else:
        cfg["swarm"]["search_area"] = {"n_min": 0.0, "n_max": 1.0, "e_min": 0.0, "e_max": 1.0}
        cfg["swarm"]["search_spacing_m"] = 0.3
    cfg["swarm"]["use_map_bounds"] = False
    # Obstacle avoidance tuned for sim (brief: no flying over).
    cfg["swarm"]["obstacle_avoidance_enabled"] = True
    cfg["swarm"]["obstacle_clearance_m"] = 0.10 if full_scale else 0.05
    cfg["swarm"]["obstacle_hard_stop_distance_m"] = 0.05 if full_scale else 0.08
    cfg["swarm"]["auto_release_search_in_sim"] = True
    swarm_cfg = cfg["swarm"]
    if fast:
        cfg["swarm"]["takeoff_wait_s"] = 0.2
        cfg["swarm"]["uwb_nav_timeout_s"] = 60 if full_scale else 12
        cfg["swarm"]["search_wp_timeout_s"] = 4.0 if full_scale else 2.0
        cfg["swarm"]["move_speed"] = 1.0

    if full_scale:
        sim_obstacles, tunnel_metadata = _random_tunnels(seed=seed)
    else:
        # A small obstacle in the middle the swarm must route AROUND (not over).
        sim_obstacles = [ObstacleBox(n0=0.40, e0=0.40, n1=0.52, e1=0.52)]
        tunnel_metadata = []
    clearance = float(cfg["swarm"]["obstacle_clearance_m"])
    for ctx in contexts.values():
        ctx.obstacle_sensor = MapObstacleSensor(sim_obstacles, clearance)

    # Simulated convoy of ground robots + proximity sensor.
    robots = (
        random_convoy(seed=seed + 100, n_min=0.5, n_max=9.5, e_min=0.5, e_max=4.5)
        if full_scale
        else default_convoy()
    )
    sensor = SimTargetSensor(
        uwb, robots,
        camera_footprint_m=float(cfg["swarm"].get("sim_camera_footprint_m", 0.35)),
    )

    def _step_robots(_contexts) -> None:
        for r in robots:
            r.step(0.02)

    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    scenario_path = ROOT / "output" / "challenge2" / "full_scale_scenario.json"
    if full_scale:
        scenario_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "arena": {"x_east_m": 5.0, "y_north_m": 10.0, "z_height_m": 3.5},
                    "tunnel_count": len(tunnel_metadata),
                    "tunnels": tunnel_metadata,
                    "robots_initial": [
                        {"robot_id": r.robot_id, "n_y_m": r.n, "e_x_m": r.e}
                        for r in robots
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    label = "FULL-SCALE" if full_scale else "COMPACT"
    print(f"=== Challenge 2 {label} DRY RUN (simulated UWB + fake HULAs + convoy) ===")
    if full_scale:
        print("Arena: x/East=5m y/North=10m z=3.5m")
        print(f"Generated {len(tunnel_metadata)} random n-shaped tunnel obstacles")
        print(f"Scenario file: {scenario_path}")
    print(f"Convoy: {len(robots)} ground robots to find")
    try:
        run_swarm_loop(contexts, uwb, cfg, sensor, simulated=True, on_tick=_step_robots)
    finally:
        uwb.stop()

    all_found: set = set()
    pad_landed = 0
    final_landed = 0
    lines = []
    for ip, ctx in contexts.items():
        n, e, ok = uwb.get_tag_ne(ctx.tag_id)
        all_found |= ctx.found_target_ids
        pad_landed += 1 if ctx.pad_landed else 0
        final_landed += 1 if ctx.landed else 0
        lines.append(
            f"{ip} tag={ctx.tag_id} state={ctx.state.name} pad_landed={ctx.pad_landed} "
            f"final_landed={ctx.landed} "
            f"final N={n:.2f} E={e:.2f} zone N={ctx.target_n:.2f} E={ctx.target_e:.2f} "
            f"snapshots={ctx.snapshots_taken} robots_found={sorted(ctx.found_target_ids)}"
        )
    summary = (
        f"Total unique robots found: {len(all_found)}/{len(robots)} -> {sorted(all_found)}\n"
        f"Landing pads visited: {pad_landed}/{len(contexts)}\n"
        f"Final landings: {final_landed}/{len(contexts)}"
    )
    OUTPUT_LOG.write_text("\n".join(lines) + "\n" + summary, encoding="utf-8")
    print(f"\nDry run log: {OUTPUT_LOG}")
    for line in lines:
        print(f"  {line}")
    print(f"  {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated Challenge 2 swarm")
    parser.add_argument("config", nargs="?", help="Path to challenge.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full-scale", action="store_true", help="Run 5m x 10m x 3.5m arena")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for obstacles and robots")
    args = parser.parse_args()
    run_dry_swarm(args.config, fast=args.fast, full_scale=args.full_scale, seed=args.seed)


if __name__ == "__main__":
    main()
