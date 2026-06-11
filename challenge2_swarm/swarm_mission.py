"""
Challenge 2 — Swarm HULA drones (Semi-Final).

  - Dola discovers drones on WiFi (organizer huladola.py)
  - UWBParserThread on C2 USB for position (organizer UWBParserThread.py)
  - pyhulax move() guided by UWB toward Challenge 1 landing zones
  - ArUco robot-ID snapshots during search phase

Laptop dry-run: python scripts/dry_run_challenge2.py
"""

from __future__ import annotations

import sys
import time

from challenge2_swarm.camera_control import configure_ground_marker_camera
from challenge2_swarm.dola import Dola
from challenge2_swarm.swarm_core import DroneContext, run_swarm_loop
from challenge2_swarm.target_sensor import ArucoTargetSensor
from common.config_loader import load_config
from common.emergency import SwarmEmergencyGuard, land_all_hulas
from common.uwb_c2 import UWBParserThreadC2

try:
    from pyhulax import DroneAPI
except ImportError:
    DroneAPI = None  # type: ignore


def discover_and_connect(cfg: dict) -> dict[str, DroneContext]:
    if DroneAPI is None:
        raise ImportError("Install pyhulax on the C2 machine")

    swarm = cfg["swarm"]
    tag_ids: list[int] = swarm.get("tag_ids", [0, 1, 2])

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
    sorted_planes = sorted(ips.items())
    connect_retries = int(swarm.get("connect_retries", 3))
    for idx, (plane_id, ip) in enumerate(sorted_planes):
        if not ip:
            print(f"Plane {plane_id}: not found")
            continue
        tag_id = tag_ids[idx] if idx < len(tag_ids) else int(plane_id)
        print(f"Plane {plane_id}: {ip} (UWB tag {tag_id})")
        api = None
        for attempt in range(1, connect_retries + 1):
            api = DroneAPI()
            try:
                api.connect(ip)
                break
            except Exception as exc:
                print(f"Plane {plane_id}: connect attempt {attempt}/{connect_retries} failed: {exc}")
                try:
                    api.disconnect()
                except Exception:
                    pass
                api = None
                if attempt < connect_retries:
                    time.sleep(2)
        if api is None:
            print(
                f"Plane {plane_id}: failed to connect to {ip}. Check WiFi is connected "
                "to the HULA/drone network, the drone is powered on, and no other "
                "program is already connected to it."
            )
            continue
        configure_ground_marker_camera(api, swarm, label=f"Plane {plane_id}")
        api.set_video_stream(True)
        stream = api.create_video_stream()
        if stream is not None:
            stream.start()
        contexts[str(ip)] = DroneContext(ip=str(ip), api=api, tag_id=tag_id, stream=stream)
    return contexts


def run_swarm_mission(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    swarm_cfg = cfg["swarm"]

    uwb = UWBParserThreadC2(
        serial_port=swarm_cfg.get("uwb_serial_port"),
        baud_rate=int(swarm_cfg.get("uwb_baud_rate", 921600)),
        origin_x=float(swarm_cfg.get("uwb_origin_x", 0.0)),
        origin_y=float(swarm_cfg.get("uwb_origin_y", 0.0)),
    )
    uwb.start()
    if not uwb.serial_port:
        print("Warning: UWB not connected — MOVE_TO_ZONE will not navigate accurately.")

    contexts = discover_and_connect(cfg)
    if not contexts:
        print("No drones connected.")
        uwb.stop()
        return

    sensor = ArucoTargetSensor(
        dictionary_name=cfg["mapping_drone"].get("aruco_dictionary", "DICT_7X7_1000")
    )

    # SwarmEmergencyGuard lands every HULA on Ctrl+C / kill before exiting.
    try:
        with SwarmEmergencyGuard(contexts):
            run_swarm_loop(contexts, uwb, cfg, sensor, simulated=False)
    except Exception as exc:
        print(f"Swarm mission error: {exc}")
        land_all_hulas(contexts)
        raise
    finally:
        uwb.stop()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_swarm_mission(path)


if __name__ == "__main__":
    main()
