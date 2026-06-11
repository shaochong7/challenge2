"""No-flight HULA obstacle/barrier sensor test.

Connects to one HULA and prints pyhulax obstacle flags. It never sends takeoff
or movement commands.

Usage:
    python scripts/test_hula_obstacles.py
    python scripts/test_hula_obstacles.py --seconds 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhulax import DroneAPI

from challenge2_swarm.dola import Dola


def find_drone_ip(plane_id: int, listen_seconds: float) -> str:
    dola = Dola()
    dola.start()
    try:
        ips = dola.get_ips_by_plane_ids([plane_id], listen_seconds=listen_seconds)
    finally:
        dola.stop()
    ip = ips.get(plane_id)
    if not ip:
        raise RuntimeError(f"Plane {plane_id} not found on the HULA WiFi network")
    return str(ip)


def flags_from_obstacles(obs) -> dict[str, bool]:
    return {
        "forward": bool(getattr(obs, "forward", False)),
        "back": bool(getattr(obs, "back", False)),
        "left": bool(getattr(obs, "left", False)),
        "right": bool(getattr(obs, "right", False)),
        "down": bool(getattr(obs, "down", False)),
    }


def flags_from_raw_barrier(raw) -> dict[str, bool] | None:
    if raw is None:
        return None
    raw = int(raw)
    return {
        "forward": (raw & 1) == 1,
        "back": (raw & 2) == 2,
        "left": (raw & 4) == 4,
        "right": (raw & 8) == 8,
        "down": (raw & 16) == 16,
    }


def command_name(result) -> str:
    name = getattr(result, "name", None)
    if name:
        return str(name)
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane-id", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--listen-seconds", type=float, default=5.0)
    parser.add_argument("--no-enable-barrier", action="store_true")
    args = parser.parse_args()

    ip = find_drone_ip(args.plane_id, args.listen_seconds)
    print(f"Plane {args.plane_id}: {ip}")

    api = DroneAPI()
    try:
        api.connect(ip)
        if not args.no_enable_barrier and hasattr(api, "set_barrier_mode"):
            try:
                result = api.set_barrier_mode(True)
                print(f"Barrier mode enabled: {command_name(result)}")
            except Exception as exc:
                print(f"Could not enable barrier mode: {exc}")
        deadline = time.time() + args.seconds
        print("Reading obstacle flags. Press Ctrl+C to stop.")
        while time.time() < deadline:
            obs = api.get_obstacles()
            flags = flags_from_obstacles(obs)
            raw = None
            distance = None
            battery = None
            try:
                flight = api.get_flight_data()
                raw = flight.barrier
                distance = flight.altitude_tof
                battery = flight.battery_percent
            except Exception:
                pass
            print(
                {
                    "flags": flags,
                    "raw_barrier": raw,
                    "tof_cm": distance,
                    "battery": battery,
                }
            )
            time.sleep(0.5)
    finally:
        if not args.no_enable_barrier and hasattr(api, "set_barrier_mode"):
            try:
                api.set_barrier_mode(False)
            except Exception:
                pass
        try:
            api.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
