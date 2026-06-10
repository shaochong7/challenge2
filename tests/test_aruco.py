"""
ArUco detection + depth deprojection on a synthetic image.

Generates a real DICT_6X6_250 marker, places it on a canvas, feeds a constant
depth map, and checks the detector finds the right ID and 3D position.
"""

import cv2
import numpy as np

from detection.aruco_depth import ArucoDepthDetector


def _make_scene(marker_id: int = 7, img_size: int = 480, marker_px: int = 200):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    off = (img_size - marker_px) // 2
    canvas[off:off + marker_px, off:off + marker_px] = marker_bgr
    return canvas


def test_detects_known_marker_id():
    img = _make_scene(marker_id=7)
    depth = np.full(img.shape[:2], 1500, dtype=np.uint16)  # 1.5 m everywhere
    det = ArucoDepthDetector(
        fx=600, fy=600, cx=240, cy=240,
        dictionary_name="DICT_6X6_250",
        valid_ids=[7], invalid_ids=[10],
    )
    obs = det.detect(img, depth)
    assert len(obs) == 1
    assert obs[0].marker_id == 7
    assert obs[0].valid_landing is True


def test_detects_organizer_7x7_marker_id():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    marker = cv2.aruco.generateImageMarker(aruco_dict, 45, 200)
    img = np.full((480, 480, 3), 255, dtype=np.uint8)
    img[140:340, 140:340] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    depth = np.full(img.shape[:2], 1500, dtype=np.uint16)
    det = ArucoDepthDetector(
        fx=600,
        fy=600,
        cx=240,
        cy=240,
        dictionary_name="DICT_7X7_1000",
        valid_ids=[11, 45, 51, 67, 101],
    )
    obs = det.detect(img, depth)
    assert len(obs) == 1
    assert obs[0].marker_id == 45
    assert obs[0].valid_landing is True


def test_invalid_id_classified_invalid():
    img = _make_scene(marker_id=10)
    depth = np.full(img.shape[:2], 1200, dtype=np.uint16)
    det = ArucoDepthDetector(
        fx=600, fy=600, cx=240, cy=240,
        valid_ids=[7], invalid_ids=[10],
    )
    obs = det.detect(img, depth)
    assert len(obs) == 1
    assert obs[0].valid_landing is False


def test_depth_deprojection_center_marker():
    # Marker centered, principal point at center -> X,Y ~ 0, Z = depth
    img = _make_scene(marker_id=7)
    depth = np.full(img.shape[:2], 2000, dtype=np.uint16)  # 2.0 m
    det = ArucoDepthDetector(fx=600, fy=600, cx=240, cy=240, valid_ids=[7])
    obs = det.detect(img, depth)[0]
    assert abs(obs.z_m - 2.0) < 1e-6
    assert abs(obs.x_m) < 0.05
    assert abs(obs.y_m) < 0.05


def test_skips_zero_depth():
    img = _make_scene(marker_id=7)
    depth = np.zeros(img.shape[:2], dtype=np.uint16)  # no valid depth
    det = ArucoDepthDetector(fx=600, fy=600, cx=240, cy=240, valid_ids=[7])
    assert det.detect(img, depth) == []


def test_pose_fallback_recovers_marker_without_depth():
    # 200 px marker, fx=600, 0.2 m marker -> z = 600 * 0.2 / 200 = 0.6 m
    img = _make_scene(marker_id=7, marker_px=200)
    depth = np.zeros(img.shape[:2], dtype=np.uint16)  # depth hole everywhere
    det = ArucoDepthDetector(
        fx=600, fy=600, cx=240, cy=240, valid_ids=[7], marker_size_m=0.20
    )
    obs = det.detect(img, depth)
    assert len(obs) == 1
    assert obs[0].pose_used is True
    assert abs(obs[0].z_m - 0.6) < 0.05
    assert abs(obs[0].x_m) < 0.05
    assert abs(obs[0].y_m) < 0.05


def test_no_marker_returns_empty():
    blank = np.full((480, 480, 3), 255, dtype=np.uint8)
    depth = np.full((480, 480), 1500, dtype=np.uint16)
    det = ArucoDepthDetector(fx=600, fy=600, cx=240, cy=240)
    assert det.detect(blank, depth) == []
