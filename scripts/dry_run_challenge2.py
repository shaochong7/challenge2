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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge2_swarm.obstacle import MapObstacleSensor, ObstacleBox
from challenge2_swarm.sim.fake_swarm import build_fake_swarm
from challenge2_swarm.sim.ground_robots import default_convoy
from challenge2_swarm.swarm_core import DroneContext, load_landing_zones, run_swarm_loop
from challenge2_swarm.target_sensor import SimTargetSensor
from common.config_loader import load_config
from common.uwb_c2 import SimulatedUWBC2

OUTPUT_LOG = ROOT / "output" / "challenge2" / "dry_run_log.txt"


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


def run_dry_swarm(config_path: str | None = None, fast: bool = False) -> None:
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

    # Sim runs on a compact 1x1 m arena with a tight footprint so it stays fast,
    # independent of the real (meters-scale) arena/camera values in the config.
    cfg = dict(cfg)
    cfg["arena"] = dict(cfg.get("arena", {}))
    cfg["arena"]["geofence_enabled"] = False
    cfg["swarm"] = dict(swarm_cfg)
    cfg["swarm"]["search_area"] = {"n_min": 0.0, "n_max": 1.0, "e_min": 0.0, "e_max": 1.0}
    cfg["swarm"]["search_spacing_m"] = 0.3
    cfg["swarm"]["use_map_bounds"] = False  # sim uses its own 1x1 m arena
    # Obstacle avoidance tuned for the compact 1x1 m sim (brief: no flying over).
    cfg["swarm"]["obstacle_avoidance_enabled"] = True
    cfg["swarm"]["obstacle_clearance_m"] = 0.05
    cfg["swarm"]["obstacle_stop_distance_m"] = 0.08
    swarm_cfg = cfg["swarm"]
    if fast:
        cfg["swarm"]["takeoff_wait_s"] = 0.2
        cfg["swarm"]["uwb_nav_timeout_s"] = 12
        cfg["swarm"]["search_wp_timeout_s"] = 2.0
        cfg["swarm"]["move_speed"] = 1.0

    # A small obstacle in the middle the swarm must route AROUND (not over).
    sim_obstacles = [ObstacleBox(n0=0.40, e0=0.40, n1=0.52, e1=0.52)]
    clearance = float(cfg["swarm"]["obstacle_clearance_m"])
    for ctx in contexts.values():
        ctx.obstacle_sensor = MapObstacleSensor(sim_obstacles, clearance)

    # Simulated convoy of ground robots + proximity sensor
    robots = default_convoy()
    sensor = SimTargetSensor(
        uwb, robots,
        camera_footprint_m=float(cfg["swarm"].get("sim_camera_footprint_m", 0.35)),
    )

    def _step_robots(_contexts) -> None:
        for r in robots:
            r.step(0.02)

    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    print("=== Challenge 2 DRY RUN (simulated UWB + fake HULAs + convoy) ===")
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
    args = parser.parse_args()
    run_dry_swarm(args.config, fast=args.fast)


if __name__ == "__main__":
    main()
