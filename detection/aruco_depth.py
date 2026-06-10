"""
ArUco detection + depth deprojection — from organizer sample + RealSense.

Returns marker ID, validity, pixel center, and camera-frame XYZ (meters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


@dataclass
class MarkerObservation:
    marker_id: int
    valid_landing: bool
    center_u: int
    center_v: int
    x_m: float
    y_m: float
    z_m: float
    pose_used: bool = False  # True when 3D came from marker-size pose, not depth


class ArucoDepthDetector:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dictionary_name: str = "DICT_6X6_250",
        valid_ids: Iterable[int] | None = None,
        invalid_ids: Iterable[int] | None = None,
        marker_size_m: float | None = None,
    ) -> None:
        if dictionary_name not in ARUCO_DICTS:
            raise ValueError(f"Unknown dictionary: {dictionary_name}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.valid_ids = set(valid_ids or [])
        self.invalid_ids = set(invalid_ids or [])
        self.marker_size_m = marker_size_m
        self._camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self._dist = np.zeros((5, 1), dtype=np.float64)

    def _pose_xyz(self, pts: np.ndarray) -> tuple[float, float, float] | None:
        """Marker-center XYZ (m) in camera frame via solvePnP using known size.

        Works without depth — useful at altitude where flat markers leave depth
        holes. pts are corners in order TL, TR, BR, BL.
        """
        if not self.marker_size_m:
            return None
        h = self.marker_size_m / 2.0
        obj = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float64,
        )
        ok, _rvec, tvec = cv2.solvePnP(
            obj, pts.astype(np.float64), self._camera_matrix, self._dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        t = np.asarray(tvec, dtype=np.float64).ravel()
        return float(t[0]), float(t[1]), float(t[2])

    def _depth_at(self, depth_mm: np.ndarray, v: int, u: int, patch: int = 3) -> float:
        h, w = depth_mm.shape[:2]
        r0 = max(0, v - patch)
        r1 = min(h, v + patch + 1)
        c0 = max(0, u - patch)
        c1 = min(w, u + patch + 1)
        region = depth_mm[r0:r1, c0:c1].astype(np.float32)
        valid = region[region > 0]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid)) / 1000.0

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        draw: bool = False,
    ) -> list[MarkerObservation]:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results: list[MarkerObservation] = []

        if ids is None:
            return results

        if draw:
            cv2.aruco.drawDetectedMarkers(color_bgr, corners, ids)

        for marker_corner, marker_id in zip(corners, ids.flatten()):
            pts = marker_corner.reshape((4, 2))
            top_left, top_right, bottom_right, bottom_left = pts
            c_x = int((top_left[0] + bottom_right[0]) / 2.0)
            c_y = int((top_left[1] + bottom_right[1]) / 2.0)
            depth_m = self._depth_at(depth_mm, c_y, c_x)

            pose_used = False
            if depth_m > 0:
                x_m = (c_x - self.cx) * depth_m / self.fx
                y_m = (c_y - self.cy) * depth_m / self.fy
                z_m = depth_m
            else:
                # Depth hole (common on flat markers at altitude) — fall back to
                # pose from known marker size.
                pose = self._pose_xyz(pts)
                if pose is None:
                    continue
                x_m, y_m, z_m = pose
                pose_used = True

            mid = int(marker_id)
            if mid in self.valid_ids:
                valid = True
            elif mid in self.invalid_ids:
                valid = False
            else:
                valid = False

            results.append(
                MarkerObservation(
                    marker_id=mid,
                    valid_landing=valid,
                    center_u=c_x,
                    center_v=c_y,
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m,
                    pose_used=pose_used,
                )
            )
        return results
