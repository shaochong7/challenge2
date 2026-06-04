"""
Target detection for Challenge 2 — YOLO (optional) + snapshot helper.

On mapping drone NPU path, export ONNX -> RKNN per organizer scripts on Discord.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class TargetDetector:
    def __init__(self, weights_path: str | None = None, class_names: list[str] | None = None) -> None:
        self.weights_path = weights_path
        self.class_names = class_names or ["robomaster"]
        self._model = None
        if weights_path and Path(weights_path).exists():
            try:
                from ultralytics import YOLO

                self._model = YOLO(weights_path)
            except ImportError:
                print("ultralytics not installed — detection disabled")

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.4) -> list[Detection]:
        if self._model is None:
            return []
        results = self._model.predict(frame_bgr, conf=conf, verbose=False)
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                out.append(
                    Detection(
                        class_name=name,
                        confidence=float(box.conf[0]),
                        bbox_xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    )
                )
        return out

    @staticmethod
    def save_snapshot(frame_bgr: np.ndarray, path: Path, detections: list[Detection]) -> None:
        img = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{d.class_name} {d.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
