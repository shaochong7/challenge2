"""
Target sensing abstraction for the swarm SEARCH phase.

The mission loop doesn't care whether targets come from a real YOLO model on a
HULA camera feed or from a simulated convoy — it just calls sense() each tick
and save_snapshot() when something new is found.

  ArucoTargetSensor -> real: reads robot marker IDs from stream.latest_frame
  YoloTargetSensor  -> optional: runs TargetDetector on stream.latest_frame
  SimTargetSensor   -> dry-run: "sees" ground robots within camera footprint
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SensedTarget:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int] | None = None
    world_n: float | None = None
    world_e: float | None = None
    target_id: int | None = None  # stable id (sim); None for real YOLO


class TargetSensor(Protocol):
    def sense(self, ctx) -> list[SensedTarget]: ...

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int: ...


def _frame_to_bgr(frame):
    """Return an OpenCV BGR image from either pyhulax or legacy frame objects."""
    if hasattr(frame, "image"):
        return frame.image
    if hasattr(frame, "to_rgb"):
        return frame.to_rgb()
    return frame


class YoloTargetSensor:
    """Real sensor: YOLO on the HULA camera frame."""

    def __init__(self, detector, conf: float = 0.4) -> None:
        self.detector = detector
        self.conf = conf
        self._last_frame = None

    def sense(self, ctx) -> list[SensedTarget]:
        stream = ctx.stream
        frame = stream.latest_frame if stream else None
        if frame is None:
            return []
        bgr = _frame_to_bgr(frame)
        self._last_frame = bgr
        dets = self.detector.detect(bgr, conf=self.conf)
        return [
            SensedTarget(confidence=d.confidence, bbox_xyxy=d.bbox_xyxy)
            for d in dets
        ]

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int:
        if self._last_frame is None:
            return 0
        from detection.target_detector import Detection

        dets = [
            Detection("robomaster", t.confidence, t.bbox_xyxy or (0, 0, 0, 0))
            for t in targets
        ]
        self.detector.save_snapshot(self._last_frame, path, dets)
        return len(targets)


class ArucoTargetSensor:
    """Real sensor: detects robot ArUco marker IDs from the HULA camera frame."""

    def __init__(self, dictionary_name: str = "DICT_7X7_1000", enhance: bool = True) -> None:
        import cv2
        from detection.aruco_depth import ARUCO_DICTS

        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 5
        params.minMarkerPerimeterRate = 0.02
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.enhance = enhance
        self._last_frame = None
        self._last_corners = None
        self._last_ids = None
        self._last_enhanced = None

    def _enhance_gray(self, bgr):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if not self.enhance:
            return gray
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 2), -0.7, 0)
        return np.clip(sharp, 0, 255).astype("uint8")

    def sense(self, ctx) -> list[SensedTarget]:
        import cv2

        stream = ctx.stream
        frame = stream.latest_frame if stream else None
        if frame is None:
            return []

        bgr = _frame_to_bgr(frame)
        self._last_frame = bgr

        gray = self._enhance_gray(bgr)
        self._last_enhanced = gray
        corners, ids, _ = self.detector.detectMarkers(gray)

        self._last_corners = corners
        self._last_ids = ids

        if ids is None:
            return []

        seen: list[SensedTarget] = []
        for marker_corner, marker_id in zip(corners, ids.flatten()):
            pts = marker_corner.reshape((4, 2))
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())

            seen.append(
                SensedTarget(
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    target_id=int(marker_id),
                )
            )

        return seen

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int:
        import cv2

        if self._last_frame is None:
            return 0

        img = self._last_frame.copy()

        if self._last_ids is not None and self._last_corners is not None:
            cv2.aruco.drawDetectedMarkers(img, self._last_corners, self._last_ids)

        for t in targets:
            if t.bbox_xyxy:
                x1, y1, _x2, _y2 = t.bbox_xyxy
                cv2.putText(
                    img,
                    f"ID {t.target_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return len(targets)


class SimTargetSensor:
    """
    Dry-run sensor: a ground robot is 'seen' when the drone's UWB position is
    within camera_footprint_m of it. Produces a synthetic snapshot image.
    """

    def __init__(self, uwb, robots, camera_footprint_m: float = 0.35) -> None:
        self.uwb = uwb
        self.robots = robots  # list of GroundRobot
        self.footprint = camera_footprint_m

    def sense(self, ctx) -> list[SensedTarget]:
        n, e, ready = self.uwb.get_tag_ne(ctx.tag_id)
        if not ready:
            return []
        seen: list[SensedTarget] = []
        for robot in self.robots:
            rn, re = robot.position()
            dist = math.hypot(rn - n, re - e)
            if dist <= self.footprint:
                conf = max(0.5, 1.0 - dist / max(self.footprint, 1e-6))
                seen.append(
                    SensedTarget(
                        confidence=round(conf, 2),
                        world_n=rn,
                        world_e=re,
                        target_id=robot.robot_id,
                    )
                )
        return seen

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int:
        import cv2
        import numpy as np

        n, e, _ = self.uwb.get_tag_ne(ctx.tag_id)
        img = np.full((300, 400, 3), 70, dtype=np.uint8)
        cv2.putText(
            img, f"Drone {ctx.tag_id} @ N={n:.2f} E={e:.2f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        for i, t in enumerate(targets):
            cx = 200 + int((t.world_e or 0 - e) * 300)
            cy = 150 - int((t.world_n or 0 - n) * 300)
            cv2.rectangle(img, (cx - 30, cy - 20), (cx + 30, cy + 20), (0, 200, 0), 2)
            cv2.putText(
                img, f"robot{t.target_id} {t.confidence:.2f}", (cx - 35, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return len(targets)
