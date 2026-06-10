"""
RealSense ArUco camera check for Challenge 1.

This script does not arm or move the drone. It only opens the RealSense camera,
detects ArUco markers using config/challenge.yaml, prints marker IDs, and saves
annotated frames.

Run on the mapping drone / NoMachine Ubuntu session:
    python3 scripts/realsense_aruco_check.py
    python3 scripts/realsense_aruco_check.py --frames 30 --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_loader import load_config  # noqa: E402
from detection.aruco_depth import ArucoDepthDetector  # noqa: E402
from detection.realsense_capture import RealSenseCapture  # noqa: E402


def _annotate(frame, observations) -> None:
    for obs in observations:
        status = "VALID" if obs.valid_landing else "UNKNOWN/INVALID"
        color = (0, 220, 0) if obs.valid_landing else (0, 0, 255)
        cv2.circle(frame, (obs.center_u, obs.center_v), 5, color, -1)
        cv2.putText(
            frame,
            f"id={obs.marker_id} {status} z={obs.z_m:.2f}m",
            (obs.center_u + 8, max(18, obs.center_v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RealSense ArUco detection")
    parser.add_argument("config", nargs="?", help="Optional path to challenge.yaml")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to inspect")
    parser.add_argument("--warmup", type=int, default=10, help="Frames to discard first")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "aruco_check"),
        help="Directory for annotated frames",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    m = cfg["mapping_drone"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Opening RealSense...")
    print(
        f"Resolution {m.get('camera_width', 1280)}x{m.get('camera_height', 720)} "
        f"@ {m.get('camera_fps', 30)} fps"
    )
    rs = RealSenseCapture(
        width=int(m.get("camera_width", 1280)),
        height=int(m.get("camera_height", 720)),
        fps=int(m.get("camera_fps", 30)),
    )
    detector = ArucoDepthDetector(
        fx=rs.intrinsics.fx,
        fy=rs.intrinsics.fy,
        cx=rs.intrinsics.cx,
        cy=rs.intrinsics.cy,
        dictionary_name=m.get("aruco_dictionary", "DICT_7X7_1000"),
        valid_ids=m.get("valid_marker_ids", []),
        invalid_ids=m.get("invalid_marker_ids", []),
        marker_size_m=m.get("marker_size_m"),
    )

    print(
        f"Dictionary={m.get('aruco_dictionary', 'DICT_7X7_1000')} "
        f"valid_ids={m.get('valid_marker_ids', [])}"
    )
    seen: set[int] = set()

    try:
        for _ in range(args.warmup):
            rs.get_frames()

        for i in range(args.frames):
            frames = rs.get_frames()
            annotated = frames.color_bgr.copy()
            observations = detector.detect(annotated, frames.depth_mm, draw=True)
            _annotate(annotated, observations)

            if observations:
                print(f"Frame {i:02d}: detected {len(observations)} marker(s)")
            else:
                print(f"Frame {i:02d}: no markers")
            for obs in observations:
                seen.add(obs.marker_id)
                status = "VALID" if obs.valid_landing else "UNKNOWN/INVALID"
                print(
                    f"  id={obs.marker_id} {status} "
                    f"pixel=({obs.center_u},{obs.center_v}) "
                    f"xyz=({obs.x_m:+.2f},{obs.y_m:+.2f},{obs.z_m:.2f})m "
                    f"pose_used={obs.pose_used}"
                )

            out = out_dir / f"aruco_check_{i:02d}.png"
            cv2.imwrite(str(out), annotated)
            if args.show:
                cv2.imshow("RealSense ArUco check", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(0.05)

    finally:
        rs.stop()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\nSaved annotated frames to: {out_dir}")
    print(f"Detected marker IDs across run: {sorted(seen)}")


if __name__ == "__main__":
    main()
