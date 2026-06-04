"""Intel RealSense aligned color + depth frames for mapping drone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FramePair:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: Intrinsics


class RealSenseCapture:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        if rs is None:
            raise ImportError("pyrealsense2 not installed")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(cfg)
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy
        )

    def get_frames(self) -> FramePair:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            raise RuntimeError("RealSense frame timeout")
        depth_mm = np.asanyarray(depth.get_data())
        color_bgr = np.asanyarray(color.get_data())
        return FramePair(color_bgr=color_bgr, depth_mm=depth_mm, intrinsics=self.intrinsics)

    def stop(self) -> None:
        self.pipeline.stop()
