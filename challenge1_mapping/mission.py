"""
Challenge 1 — Mapping drone (University teams).

Flow:
  1. UWB + MAVSDK connect, arm, offboard
  2. Survey waypoints (UWB N/E)
  3. At each waypoint: hover, capture RealSense, detect ArUco, classify landing pads
  4. Write report JSON for Challenge 2 landing selection
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from mavsdk import System

from common.config_loader import load_config
from common.uwb_listener import shutdown_uwb, start_uwb_thread, wait_for_uwb
from common.velocity_nav import NavGains, VelocityNavigator, run_telemetry_tasks
from detection.aruco_depth import ArucoDepthDetector
from detection.realsense_capture import RealSenseCapture

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "challenge1"


async def run_mission(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    m = cfg["mapping_drone"]
    nav_cfg = cfg["navigation"]

    start_uwb_thread(m.get("uwb_topic", "uwb_tag"))
    await wait_for_uwb()

    drone = System()
    print("Connecting mapping drone...")
    await drone.connect(system_address=m["serial_address"])

    state = await run_telemetry_tasks(drone)
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
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

    observations: list[dict] = []
    takeoff_d = -float(m["takeoff_height_m"])

    try:
        home_n, home_e = await wait_for_uwb()
        navigator.takeoff_yaw = state["yaw"]
        print(f"Home UWB N={home_n:.2f} E={home_e:.2f}, yaw={navigator.takeoff_yaw:.1f}")

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

        # Climb to survey height using down component (NED: down positive)
        await navigator.fly_to(home_n, home_e, takeoff_d, ignore_height=False)

        waypoints = m.get("survey_waypoints", [])
        hover_s = float(m.get("hover_at_waypoint_s", 2.0))

        for i, wp in enumerate(waypoints):
            tn = float(wp["n"])
            te = float(wp["e"])
            print(f"--- Waypoint {i + 1}/{len(waypoints)} ---")
            await navigator.fly_to(tn, te, takeoff_d, ignore_height=True)
            await navigator.hover(hover_s, ignore_height=True)

            frames = rs.get_frames()
            markers = aruco.detect(frames.color_bgr, frames.depth_mm)
            for obs in markers:
                row = {
                    "waypoint_index": i,
                    "waypoint_n": tn,
                    "waypoint_e": te,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **asdict(obs),
                }
                observations.append(row)
                status = "VALID" if obs.valid_landing else "INVALID"
                print(
                    f"  ArUco id={obs.marker_id} {status} "
                    f"XYZ=({obs.x_m:.2f},{obs.y_m:.2f},{obs.z_m:.2f})"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "challenge": 1,
        "observations": observations,
        "valid_landing_ids": sorted(
            {o["marker_id"] for o in observations if o["valid_landing"]}
        ),
    }
    out_path = OUTPUT_DIR / "landing_pad_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {out_path}")


def main() -> None:
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_mission(cfg_arg))


if __name__ == "__main__":
    main()
