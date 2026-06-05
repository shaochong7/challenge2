"""
Challenge 1 — Mapping drone (University teams).

Cooked from the organizer reference code into one mission:
  - kolomee.py        -> UWB + velocity offboard navigation (common/velocity_nav)
  - ArUco sample      -> landing-pad detection + depth (detection/aruco_depth)
  - getSyncDepthColor -> aligned color+depth (detection/realsense_capture)
  - generateTopDown   -> per-waypoint top-down occupancy (detection/occupancy_grid)

Flow:
  1. UWB + MAVSDK connect, arm, offboard
  2. Fly survey waypoints (UWB N/E) using velocity control
  3. At each waypoint: hover, grab aligned frames, detect ArUco landing pads,
     build a top-down occupancy grid, and place everything in a world map
  4. Save: landing_pad_report.json (world N/E + validity), arena_map.png,
     and per-waypoint occupancy grids — the inputs Challenge 2 needs.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from challenge1_mapping.arena_map import ArenaMap, ArenaMapConfig, marker_world_position
from common.config_loader import load_config
from common.uwb_listener import (
    get_uwb_position,
    shutdown_uwb,
    start_uwb_thread,
    wait_for_uwb,
)
from common.velocity_nav import NavGains, VelocityNavigator, run_telemetry_tasks
from detection.aruco_depth import ArucoDepthDetector
from detection.occupancy_grid import GridConfig, build_occupancy_grid
from detection.realsense_capture import RealSenseCapture

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "challenge1"


async def run_mission(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    m = cfg["mapping_drone"]
    nav_cfg = cfg["navigation"]

    start_uwb_thread(m.get("uwb_topic", "uwb_tag"))
    await wait_for_uwb()

    from mavsdk import System

    drone = System()
    print("Connecting mapping drone...")
    await drone.connect(system_address=m["serial_address"])

    state = await run_telemetry_tasks(drone)
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("Local position estimate OK")
            break

    gains = NavGains(**{k: nav_cfg[k] for k in NavGains.__dataclass_fields__})
    navigator = VelocityNavigator(
        drone,
        gains,
        get_height=lambda: state["down_m"],
        get_yaw=lambda: state["yaw"],
        height_ready=lambda: state["height_ready"],
    )

    rs = RealSenseCapture()
    aruco = ArucoDepthDetector(
        fx=rs.intrinsics.fx,
        fy=rs.intrinsics.fy,
        cx=rs.intrinsics.cx,
        cy=rs.intrinsics.cy,
        dictionary_name=m.get("aruco_dictionary", "DICT_6X6_250"),
        valid_ids=m.get("valid_marker_ids", []),
        invalid_ids=m.get("invalid_marker_ids", []),
    )

    arena = ArenaMap(ArenaMapConfig())
    grid_cfg = GridConfig()
    observations: list[dict] = []
    takeoff_d = -float(m["takeoff_height_m"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        home_n, home_e = await wait_for_uwb()
        navigator.takeoff_yaw = state["yaw"]
        print(f"Home UWB N={home_n:.2f} E={home_e:.2f}, yaw={navigator.takeoff_yaw:.1f}")
        print(f"Battery: {state['battery']:.0f}%")

        await drone.action.set_takeoff_altitude(float(m["takeoff_height_m"]))
        await asyncio.sleep(1.0)

        choice = await asyncio.get_running_loop().run_in_executor(
            None, input, "Arm and start mission? (y/n): "
        )
        if choice.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

        await drone.action.arm()
        await navigator.start_offboard()
        await navigator.fly_to(home_n, home_e, takeoff_d, ignore_height=False)

        waypoints = m.get("survey_waypoints", [])
        hover_s = float(m.get("hover_at_waypoint_s", 2.0))

        for i, wp in enumerate(waypoints):
            tn, te = float(wp["n"]), float(wp["e"])
            print(f"--- Waypoint {i + 1}/{len(waypoints)} -> N={tn:.2f} E={te:.2f} ---")
            await navigator.fly_to(tn, te, takeoff_d, ignore_height=True)
            await navigator.hover(hover_s, ignore_height=True)

            drone_n, drone_e, _ = get_uwb_position()
            arena.add_path_point(drone_n, drone_e)

            frames = rs.get_frames()

            # Top-down occupancy grid for this waypoint (organizer generateTopDown)
            depth_m = frames.depth_mm.astype("float32") / 1000.0
            grid = build_occupancy_grid(
                depth_m,
                frames.intrinsics.fx, frames.intrinsics.fy,
                frames.intrinsics.cx, frames.intrinsics.cy,
                grid_cfg,
            )
            cv2.imwrite(str(OUTPUT_DIR / f"occupancy_wp{i:02d}.png"), grid)

            # ArUco landing pads -> world coordinates
            markers = aruco.detect(frames.color_bgr, frames.depth_mm)
            for obs in markers:
                world_n, world_e = marker_world_position(
                    drone_n, drone_e, obs.x_m, obs.y_m, arena.cfg
                )
                arena.add_landing_pad(obs.marker_id, obs.valid_landing, world_n, world_e)
                row = {
                    "waypoint_index": i,
                    "drone_n": drone_n,
                    "drone_e": drone_e,
                    "world_n": world_n,
                    "world_e": world_e,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **asdict(obs),
                }
                observations.append(row)
                status = "VALID" if obs.valid_landing else "INVALID"
                print(
                    f"  ArUco id={obs.marker_id} {status} "
                    f"world N={world_n:.2f} E={world_e:.2f} (z={obs.z_m:.2f}m)"
                )

        await navigator.send_velocity(0.0, 0.0, 0.0)
        await drone.offboard.stop()
        await drone.action.land()
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                break
            await asyncio.sleep(0.3)
        try:
            await drone.action.disarm()
        except Exception:
            pass

    except Exception as exc:
        print(f"Mission error: {exc}")
        try:
            await navigator.send_velocity(0.0, 0.0, 0.0)
            await drone.offboard.stop()
        except Exception:
            pass
        try:
            await drone.action.land()
        except Exception:
            pass
        raise
    finally:
        rs.stop()
        shutdown_uwb()

    _save_report(arena, observations)


def _save_report(arena: ArenaMap, observations: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "arena_map.png"), arena.render_bgr())

    valid_pads = [
        {"marker_id": p.marker_id, "n": p.n, "e": p.e}
        for p in arena.pads
        if p.valid
    ]
    report = {
        "challenge": 1,
        "observations": observations,
        "valid_landing_ids": sorted({p.marker_id for p in arena.pads if p.valid}),
        "valid_landing_zones": valid_pads,  # world N/E for Challenge 2
    }
    out_path = OUTPUT_DIR / "landing_pad_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {out_path}")
    print(f"Arena map saved: {OUTPUT_DIR / 'arena_map.png'}")
    print(f"Valid landing zones: {len(valid_pads)}")


def main() -> None:
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_mission(cfg_arg))


if __name__ == "__main__":
    main()
