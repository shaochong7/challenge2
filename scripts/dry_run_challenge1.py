"""
Laptop dry-run of Challenge 1 — no drone, ROS2, MAVSDK, or RealSense.

Simulates UWB navigation through survey waypoints, renders synthetic camera
frames with ArUco pads at known world positions, and writes the same outputs
as the real mission:

  output/challenge1/landing_pad_report.json
  output/challenge1/arena_map.png
  output/challenge1/occupancy_wpNN.png
  output/challenge1/dry_run_preview_wpNN.png  (color frame with detections)

Run:
    python scripts/dry_run_challenge1.py
    python scripts/dry_run_challenge1.py --fast   # skip hover delay
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge1_mapping.arena_map import ArenaMap, ArenaMapConfig
from challenge1_mapping.sim.fake_navigator import FakeVelocityNavigator
from challenge1_mapping.sim.fake_realsense import FakeRealSenseCapture
from challenge1_mapping.survey_core import (
    build_survey_waypoints,
    process_waypoint,
    save_mission_report,
)
from common.config_loader import load_config
from common.geofence import ArenaBounds
from common.uwb_listener import get_uwb_position, set_simulated_position
from common.velocity_nav import NavGains
from detection.aruco_depth import ArucoDepthDetector
from detection.occupancy_grid import GridConfig

OUTPUT_DIR = ROOT / "output" / "challenge1"


async def run_dry_mission(config_path: str | None = None, fast: bool = False) -> None:
    cfg = load_config(config_path)
    m = cfg["mapping_drone"]
    nav_cfg = cfg["navigation"]

    # Use a richer practice arena instead of the real competition survey plan.
    # This keeps the dry-run deterministic while exercising valid/invalid pads,
    # obstacle mapping, and a broader lawnmower-style path.
    cfg = dict(cfg)
    cfg["mapping_drone"] = dict(cfg["mapping_drone"])
    cfg["mapping_drone"]["auto_survey"] = False
    cfg["mapping_drone"]["survey_waypoints"] = [
        {"n": 0.5, "e": 0.5},
        {"n": 0.5, "e": 1.5},
        {"n": 0.5, "e": 2.5},
        {"n": 0.5, "e": 3.5},
        {"n": 1.5, "e": 3.5},
        {"n": 1.5, "e": 2.5},
        {"n": 1.5, "e": 1.5},
        {"n": 1.5, "e": 0.5},
        {"n": 2.5, "e": 0.5},
        {"n": 2.5, "e": 1.5},
        {"n": 2.5, "e": 2.5},
        {"n": 2.5, "e": 3.5},
        {"n": 3.5, "e": 3.5},
        {"n": 3.5, "e": 2.5},
        {"n": 3.5, "e": 1.5},
        {"n": 3.5, "e": 0.5},
    ]

    bounds_raw = cfg.get("arena", {}).get("uwb_bounds", {})
    arena_cfg = ArenaMapConfig(
        n_min=float(bounds_raw.get("n_min", -5.0)),
        n_max=float(bounds_raw.get("n_max", 5.0)),
        e_min=float(bounds_raw.get("e_min", -5.0)),
        e_max=float(bounds_raw.get("e_max", 5.0)),
    )
    geofence = ArenaBounds.from_config(cfg)
    set_simulated_position(0.0, 0.0)

    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    navigator = FakeVelocityNavigator(
        gains,
        sim_dt=0.20 if fast else 0.05,
        geofence=geofence,
        sleep_s=0.0 if fast else None,
    )

    rs = FakeRealSenseCapture(
        arena_cfg=arena_cfg,
        camera_height_m=1.6,
        dictionary_name=m.get("aruco_dictionary", "DICT_7X7_1000"),
    )
    aruco = ArucoDepthDetector(
        fx=rs.intrinsics.fx,
        fy=rs.intrinsics.fy,
        cx=rs.intrinsics.cx,
        cy=rs.intrinsics.cy,
        dictionary_name=m.get("aruco_dictionary", "DICT_7X7_1000"),
        valid_ids=m.get("valid_marker_ids", []),
        invalid_ids=m.get("invalid_marker_ids", []),
    )

    arena = ArenaMap(arena_cfg)
    grid_cfg = GridConfig()
    observations: list[dict] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    takeoff_d = -float(m["takeoff_height_m"])
    hover_s = 0.1 if fast else float(m.get("hover_at_waypoint_s", 2.0))

    print("=== Challenge 1 DRY RUN (simulated) ===")
    home_n, home_e, _ = get_uwb_position()
    print(f"Home UWB N={home_n:.2f} E={home_e:.2f}")
    waypoints = build_survey_waypoints(cfg)
    if geofence is not None:
        geofence.check_position(home_n, home_e)
        geofence.validate_waypoints(waypoints)

    await navigator.start_offboard()
    await navigator.fly_to(
        home_n,
        home_e,
        takeoff_d,
        ignore_height=False,
        validate_target=False,
    )

    for i, wp in enumerate(waypoints):
        tn, te = float(wp["n"]), float(wp["e"])
        print(f"--- Waypoint {i + 1}/{len(waypoints)} -> N={tn:.2f} E={te:.2f} ---")
        await navigator.fly_to(tn, te, takeoff_d, ignore_height=True)
        await navigator.hover(hover_s, ignore_height=True)

        drone_n, drone_e, _ = get_uwb_position()
        frames = rs.get_frames_at(drone_n, drone_e)

        process_waypoint(
            i, drone_n, drone_e, frames, aruco, arena, observations, OUTPUT_DIR, grid_cfg
        )

        preview = frames.color_bgr.copy()
        aruco.detect(preview, frames.depth_mm, draw=True)
        cv2.putText(
            preview,
            f"WP{i} N={drone_n:.2f} E={drone_e:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(OUTPUT_DIR / f"dry_run_preview_wp{i:02d}.png"), preview)

    save_mission_report(arena, observations, OUTPUT_DIR, simulated=True)
    print("\nDry run complete. Open output/challenge1/arena_map.png to review.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated Challenge 1 mission")
    parser.add_argument("config", nargs="?", help="Path to challenge.yaml")
    parser.add_argument("--fast", action="store_true", help="Shorter hover / faster sim")
    args = parser.parse_args()
    asyncio.run(run_dry_mission(args.config, fast=args.fast))


if __name__ == "__main__":
    main()
