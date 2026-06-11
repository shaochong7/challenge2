"""No-flight HULA ArUco detection test.

Connects to one HULA, starts its video stream, detects ArUco markers, and saves
annotated frames. It never sends takeoff or movement commands.

Usage:
    python scripts/test_hula_aruco.py
    python scripts/test_hula_aruco.py --plane-id 1 --seconds 20
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhulax import DroneAPI
from pyhulax.core.exceptions import DroneConnectionError

from challenge2_swarm.camera_control import configure_ground_marker_camera
from challenge2_swarm.dola import Dola
from challenge2_swarm.target_sensor import ArucoTargetSensor


class _Ctx:
    def __init__(self, stream) -> None:
        self.stream = stream


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


def local_ip_for_drone(drone_ip: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((drone_ip, 80))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


def connect_with_retries(api: DroneAPI, ip: str, attempts: int, delay_s: float) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Connect attempt {attempt}/{attempts}")
            api.connect(ip)
            return
        except DroneConnectionError as exc:
            last_exc = exc
            try:
                api.disconnect()
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc


def frame_to_bgr(frame):
    if frame is None:
        return None
    if hasattr(frame, "image"):
        return frame.image
    if hasattr(frame, "to_rgb"):
        return frame.to_rgb()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane-id", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--listen-seconds", type=float, default=5.0)
    parser.add_argument("--dictionary", default="DICT_7X7_1000")
    parser.add_argument("--snapshot-dir", default="output/aruco_hula")
    parser.add_argument("--connect-attempts", type=int, default=5)
    parser.add_argument("--connect-delay", type=float, default=2.0)
    parser.add_argument("--show", action="store_true", help="Show a live OpenCV preview window")
    parser.add_argument("--save-raw-every", type=float, default=2.0)
    parser.add_argument("--camera-angle", type=int, default=90, help="Downward camera pitch angle, 0-90")
    parser.add_argument("--no-camera-pitch", action="store_true", help="Do not send camera pitch command")
    args = parser.parse_args()

    ip = find_drone_ip(args.plane_id, args.listen_seconds)
    print(f"Plane {args.plane_id}: {ip}")
    print(f"Local drone-network IP: {local_ip_for_drone(ip)}")

    api = DroneAPI()
    stream = None
    sensor = ArucoTargetSensor(args.dictionary)
    out_dir = Path(args.snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[int] = set()
    snapshots = 0

    try:
        connect_with_retries(api, ip, args.connect_attempts, args.connect_delay)
        configure_ground_marker_camera(
            api,
            {
                "camera_pitch_enabled": not args.no_camera_pitch,
                "camera_pitch_mode": "DOWN_ABSOLUTE",
                "camera_pitch_angle_deg": args.camera_angle,
            },
            label=f"Plane {args.plane_id}",
        )
        api.set_video_stream(True)
        stream = api.create_video_stream()
        stream.start()
        ctx = _Ctx(stream)

        deadline = time.time() + args.seconds
        last_status_t = 0.0
        last_raw_save_t = 0.0
        raw_count = 0
        print("Looking for ArUco markers. Press Ctrl+C to stop.")
        while time.time() < deadline:
            frame = stream.latest_frame
            bgr = frame_to_bgr(frame)
            targets = sensor.sense(ctx)
            ids = [t.target_id for t in targets]
            new_ids = [marker_id for marker_id in ids if marker_id not in seen_ids]

            now = time.time()
            if now - last_status_t >= 1.0:
                shape = None if bgr is None else getattr(bgr, "shape", None)
                frame_count = getattr(stream, "frame_count", None)
                fps = getattr(stream, "fps", None)
                print(f"frame={shape} stream_frames={frame_count} fps={fps} ids={ids}")
                last_status_t = now

            if bgr is not None and args.save_raw_every > 0 and now - last_raw_save_t >= args.save_raw_every:
                import cv2

                raw_out = out_dir / f"hula{args.plane_id}_raw_{raw_count:02d}.jpg"
                cv2.imwrite(str(raw_out), bgr)
                enhanced = getattr(sensor, "_last_enhanced", None)
                if enhanced is not None:
                    cv2.imwrite(
                        str(out_dir / f"hula{args.plane_id}_enhanced_{raw_count:02d}.jpg"),
                        enhanced,
                    )
                raw_count += 1
                last_raw_save_t = now

            if args.show and bgr is not None:
                import cv2

                cv2.imshow(f"HULA {args.plane_id} ArUco", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if ids:
                print(f"Detected marker IDs: {ids}")
            if new_ids:
                for marker_id in new_ids:
                    seen_ids.add(marker_id)
                out = out_dir / f"hula{args.plane_id}_aruco_{snapshots:02d}.jpg"
                sensor.save_snapshot(ctx, targets, out)
                snapshots += 1
                print(f"Saved {out}")
            time.sleep(0.1)
    finally:
        if args.show:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        try:
            api.set_video_stream(False)
        except Exception:
            pass
        try:
            api.disconnect()
        except Exception:
            pass
    print(f"Unique marker IDs seen: {sorted(seen_ids)}")


if __name__ == "__main__":
    main()
