"""
Visual ArUco demo — no drone/camera needed.

Builds a synthetic arena image with several markers (some "valid", some not),
runs the real detector + depth deprojection, prints results, and saves an
annotated image you can eyeball.

Run:
    python scripts/aruco_demo.py
Output:
    output/aruco_demo.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.aruco_depth import ArucoDepthDetector  # noqa: E402

VALID_IDS = [11, 45, 51, 67, 101]
INVALID_IDS = [201, 202]
DICT_NAME = "DICT_7X7_1000"


def build_scene() -> tuple[np.ndarray, np.ndarray]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    canvas = np.full((600, 900, 3), 60, dtype=np.uint8)  # dark "floor"
    depth = np.zeros((600, 900), dtype=np.uint16)

    placements = [
        (11, 80, 80, 1200),
        (45, 360, 80, 1500),
        (51, 640, 80, 1800),
        (201, 200, 350, 1300),
        (202, 520, 350, 1600),
    ]
    size = 160
    for marker_id, x, y, depth_mm in placements:
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        canvas[y:y + size, x:x + size] = marker_bgr
        depth[y:y + size, x:x + size] = depth_mm
    return canvas, depth


def main() -> None:
    canvas, depth = build_scene()
    detector = ArucoDepthDetector(
        fx=600, fy=600, cx=450, cy=300,
        dictionary_name=DICT_NAME,
        valid_ids=VALID_IDS,
        invalid_ids=INVALID_IDS,
    )
    observations = detector.detect(canvas, depth, draw=True)

    print(f"Detected {len(observations)} markers:")
    for o in observations:
        tag = "VALID" if o.valid_landing else "INVALID"
        print(
            f"  id={o.marker_id:>2} {tag:<7} "
            f"pixel=({o.center_u},{o.center_v}) "
            f"XYZ=({o.x_m:+.2f},{o.y_m:+.2f},{o.z_m:.2f}) m"
        )
        color = (0, 200, 0) if o.valid_landing else (0, 0, 255)
        cv2.circle(canvas, (o.center_u, o.center_v), 6, color, -1)
        cv2.putText(
            canvas, f"{o.marker_id}:{tag}", (o.center_u - 30, o.center_v + 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    out = ROOT / "output" / "aruco_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"\nAnnotated image saved: {out}")


if __name__ == "__main__":
    main()
