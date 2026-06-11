"""Print live UWB tag coordinates for arena-origin/orientation checks.

Use this before flight to verify which direction raw UWB x/y increases and
whether each HULA plane is mapped to the correct tag ID.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import load_config
from common.uwb_c2 import UWBParserThreadC2, parser_xy_to_ne


def _tag_ids_from_args(args) -> list[int]:
    if args.tags:
        return [int(part.strip()) for part in args.tags.split(",") if part.strip()]
    cfg = load_config(args.config)
    return [int(tag_id) for tag_id in cfg.get("swarm", {}).get("tag_ids", [0, 1, 2])]


def _raw_tags(uwb: UWBParserThreadC2) -> dict[int, tuple[float, float, float, float]]:
    with uwb._lock:  # internal read-only diagnostic access
        return dict(uwb._tag_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="Optional path to challenge.yaml")
    parser.add_argument("--tags", help="Comma-separated tag IDs, e.g. 0,1,2")
    parser.add_argument("--serial-port", help="UWB COM port, e.g. COM6")
    parser.add_argument("--baud-rate", type=int, default=None)
    parser.add_argument("--origin-x", type=float, default=None)
    parser.add_argument("--origin-y", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    swarm = cfg.get("swarm", {})
    tag_ids = _tag_ids_from_args(args)
    origin_x = float(args.origin_x if args.origin_x is not None else swarm.get("uwb_origin_x", 0.0))
    origin_y = float(args.origin_y if args.origin_y is not None else swarm.get("uwb_origin_y", 0.0))

    uwb = UWBParserThreadC2(
        serial_port=args.serial_port or swarm.get("uwb_serial_port"),
        baud_rate=int(args.baud_rate or swarm.get("uwb_baud_rate", 921600)),
        origin_x=origin_x,
        origin_y=origin_y,
    )
    uwb.start()
    if not uwb.serial_port:
        print("No UWB serial device opened. Check COM port / USB connection.")
        return

    print("UWB coordinate watch")
    print("Move one tag at a time. Output appears here in this terminal.")
    print(f"Using origin_x={origin_x:.2f}, origin_y={origin_y:.2f}")
    print("raw_x/raw_y are parser coordinates; N/E are what the mission uses.")
    print("Press Ctrl+C to stop.")

    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            raw = _raw_tags(uwb)
            rows = []
            for tag_id in tag_ids:
                entry = raw.get(tag_id)
                if entry is None:
                    rows.append(f"tag {tag_id}: no data")
                    continue
                x, y, z, t = entry
                n, e = parser_xy_to_ne(x, y, origin_x=origin_x, origin_y=origin_y)
                age = time.time() - t
                rows.append(
                    f"tag {tag_id}: raw_x={x:+.3f} raw_y={y:+.3f} raw_z={z:+.3f} "
                    f"=> N={n:+.3f} E={e:+.3f} age={age:.1f}s"
                )
            print(" | ".join(rows))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        uwb.stop()


if __name__ == "__main__":
    main()
