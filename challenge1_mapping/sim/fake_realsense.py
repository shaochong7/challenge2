"""
Synthetic down-facing RealSense frames from drone UWB position.

Projects simulated landing pads into the image using the same camera model as
detection/aruco_depth (pinhole, depth in mm). Pads outside FOV are omitted.
"""

from __future__ import annotations

import cv2
import numpy as np

from challenge1_mapping.arena_map import ArenaMapConfig
from challenge1_mapping.sim.arena_world import SimLandingPad, SimObstacle
from detection.realsense_capture import FramePair, Intrinsics

FX = FY = 600.0
CX, CY = 320.0, 240.0
WIDTH, HEIGHT = 640, 480
MARKER_PX = 100
CAMERA_HEIGHT_M = 0.8


def _world_to_camera_offset(
    pad_n: float,
    pad_e: float,
    drone_n: float,
    drone_e: float,
    cfg: ArenaMapConfig,
) -> tuple[float, float]:
    x_m = (pad_e - drone_e) / cfg.east_sign
    y_m = (pad_n - drone_n) / cfg.north_sign
    return x_m, y_m


def _project_to_pixel(x_m: float, y_m: float, z_m: float, intr: Intrinsics) -> tuple[int, int]:
    u = int(intr.cx + x_m * intr.fx / z_m)
    v = int(intr.cy + y_m * intr.fy / z_m)
    return u, v


class FakeRealSenseCapture:
    def __init__(
        self,
        pads: list[SimLandingPad] | None = None,
        obstacles: list[SimObstacle] | None = None,
        camera_height_m: float = CAMERA_HEIGHT_M,
        arena_cfg: ArenaMapConfig | None = None,
        dictionary_name: str = "DICT_7X7_1000",
    ) -> None:
        from challenge1_mapping.sim.arena_world import DEFAULT_OBSTACLES, DEFAULT_PADS

        self.pads = pads if pads is not None else list(DEFAULT_PADS)
        self.obstacles = obstacles if obstacles is not None else list(DEFAULT_OBSTACLES)
        self.camera_height_m = camera_height_m
        self.arena_cfg = arena_cfg or ArenaMapConfig()
        self.intrinsics = Intrinsics(fx=FX, fy=FY, cx=CX, cy=CY)
        self.dictionary_name = dictionary_name

    def get_frames_at(self, drone_n: float, drone_e: float) -> FramePair:
        color = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)
        depth_mm = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)

        z_m = self.camera_height_m
        z_mm = int(z_m * 1000)
        depth_mm[:, :] = z_mm

        for obs in self.obstacles:
            self._stamp_obstacle(color, depth_mm, obs, drone_n, drone_e, z_m)

        aruco_dict_id = getattr(cv2.aruco, self.dictionary_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        for pad in self.pads:
            x_m, y_m = _world_to_camera_offset(
                pad.n, pad.e, drone_n, drone_e, self.arena_cfg
            )
            u, v = _project_to_pixel(x_m, y_m, z_m, self.intrinsics)
            half = MARKER_PX // 2
            if not (half < u < WIDTH - half and half < v < HEIGHT - half):
                continue
            marker = cv2.aruco.generateImageMarker(aruco_dict, pad.marker_id, MARKER_PX)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            x0, y0 = u - MARKER_PX // 2, v - MARKER_PX // 2
            color[y0 : y0 + MARKER_PX, x0 : x0 + MARKER_PX] = marker_bgr
            depth_mm[y0 : y0 + MARKER_PX, x0 : x0 + MARKER_PX] = z_mm

        return FramePair(color_bgr=color, depth_mm=depth_mm, intrinsics=self.intrinsics)

    def _stamp_obstacle(
        self,
        color: np.ndarray,
        depth_mm: np.ndarray,
        obs: SimObstacle,
        drone_n: float,
        drone_e: float,
        z_ground: float,
    ) -> None:
        cfg = self.arena_cfg
        corners = [
            _world_to_camera_offset(obs.n0, obs.e0, drone_n, drone_e, cfg),
            _world_to_camera_offset(obs.n0, obs.e1, drone_n, drone_e, cfg),
            _world_to_camera_offset(obs.n1, obs.e1, drone_n, drone_e, cfg),
            _world_to_camera_offset(obs.n1, obs.e0, drone_n, drone_e, cfg),
        ]
        pts = np.array(
            [_project_to_pixel(x_m, y_m, z_ground, self.intrinsics) for x_m, y_m in corners],
            dtype=np.int32,
        )
        if pts[:, 0].min() < -50 or pts[:, 0].max() > WIDTH + 50:
            return
        cv2.fillPoly(color, [pts], (40, 40, 90))
        z_top_mm = int(max(0.15, z_ground - obs.height_m) * 1000)
        mask = np.zeros(color.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        depth_mm[mask > 0] = z_top_mm

    def get_frames(self) -> FramePair:
        raise RuntimeError("Use get_frames_at(drone_n, drone_e) in simulation")

    def stop(self) -> None:
        pass
