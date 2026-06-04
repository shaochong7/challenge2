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
            if depth_m <= 0:
                continue

            x_m = (c_x - self.cx) * depth_m / self.fx
            y_m = (c_y - self.cy) * depth_m / self.fy
            z_m = depth_m

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
                )
            )
        return results
