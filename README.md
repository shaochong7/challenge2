# RoboVerse Drone Challenge (University)

Competition code for **Challenge 1** (mapping drone + ArUco landing pads) and **Challenge 2** (3× HULA swarm + convoy detection), built from organizer references:

- `kolomee.py` — UWB + velocity offboard navigation
- `huladola.py` + `dola.py` — swarm discovery and video
- ArUco + depth sample — fiducial + RealSense deprojection

## Repository layout

```
config/challenge.yaml      # IDs, waypoints, gains — edit before competition
common/                    # UWB ROS listener, velocity navigator
detection/                 # ArUco+depth, RealSense, occupancy grid, RKNN/YOLO
challenge1_mapping/        # Mapping mission → landing_pad_report.json
challenge2_swarm/          # Swarm FSM + snapshots
reference/organizer_samples/  # Unmodified organizer sample codes (RealSense, RKNN)
run_challenge1.py
run_challenge2.py
scripts/train_yolo.py      # Train detector on laptop
scripts/aruco_demo.py      # Visual ArUco check (no hardware)
scripts/occupancy_demo.py  # Visual occupancy grid check (no hardware)
```

## Detection backends

Two different machines run two different detectors — keep them straight:

| | Mapping drone (Challenge 1) | Swarm C2 (Challenge 2) |
|---|---|---|
| Hardware | Rockchip NPU | Windows/Ubuntu laptop |
| Detector | `detection/rknn_detector.py` (`rknnlite` + `rknn_decoder.py`) | `detection/target_detector.py` (`ultralytics`) |
| Fiducials | `detection/aruco_depth.py` (ArUco/QR/AprilTag, no training) | — |
| 3D position | `rs.rs2_deproject_pixel_to_point` (distortion-aware) | no depth on HULA |
| Mapping | `detection/occupancy_grid.py` (top-down grid) | — |

`detection/rknn_decoder.py` is the **YOLOv11** decoder (applies sigmoid to class
scores — required for RKNN exports). The organizer's single-image
`testrknn_with_display.py` is **YOLOv8** style (no sigmoid); match the decoder to
whatever model you export. Originals kept in `reference/organizer_samples/`.

## Before competition day

1. **Confirm ArUco dictionary** with organizers (`DICT_6X6_250` in config).
2. Fill **`valid_marker_ids`** / **`invalid_marker_ids`** in `config/challenge.yaml`.
3. Set **`survey_waypoints`** to cover all landing pads in UWB N/E coordinates (measure in arena).
4. Train YOLO on RoboMaster targets → save to `models/robomaster_best.pt`.
5. Copy this repo to **C2 Terminal** (Ubuntu VM for mapping drone; Windows for swarm).

## Challenge 1 — Mapping drone

**Where:** Mapping drone onboard computer (NoMachine from C2).

**Dependencies (Ubuntu 22.04):** ROS2 Humble, `mavsdk`, `pyrealsense2`, `opencv-python`, `rclpy`.

```bash
cd roboverse-drone-challenge
python3 run_challenge1.py
```

**What it does** (stitched from the organizer samples):

1. Subscribes to UWB (`uwb_tag` topic) — *kolomee.py*
2. Arms, starts offboard, flies survey waypoints using **velocity control** (not position goto) — *kolomee.py*
3. Hovers at each point, grabs aligned RealSense color+depth — *getSyncDepthColor.py*
4. Builds a **top-down occupancy grid** per waypoint — *generateTopDown.py*
5. Detects ArUco, classifies valid/invalid, converts each pad to **world N/E** coordinates — *ArUco sample*

**Outputs** in `output/challenge1/`:
- `landing_pad_report.json` — observations + `valid_landing_zones` (world N/E) for Challenge 2
- `arena_map.png` — top-down map: survey path + valid (green) / invalid (red) pads
- `occupancy_wpNN.png` — per-waypoint occupancy grids

## Challenge 2 — Swarm

**Where:** C2 laptop on the **same WiFi** as HULA drones.

**Dependencies:** `pyhulax`, `opencv-python`, optional `ultralytics` for YOLO.

```bash
python run_challenge2.py
```

**What it does:**

1. `Dola` discovers drone IPs
2. Connects 3 drones, starts video streams
3. Per-drone **state machine**: takeoff → move → search → snapshot on detection
4. Reads up to 3 valid landing zones from Challenge 1 report (if present)

**SDK docs:** https://pyhulax.xenops.ae

## Practice: object detection

- **ArUco / AprilTag / QR:** no training — `detection/aruco_depth.py`
- **RoboMaster bodies:** label data → `python scripts/train_yolo.py`
- **Mapping drone NPU:** export trained model with organizer ONNX/RKNN scripts on Discord

## Testing

Two tiers. **Tier 1** runs on any laptop (no drone/camera/ROS2):

```bash
pip install opencv-contrib-python numpy PyYAML pytest
python -m pytest tests/ -v          # 28 unit tests: ArUco, depth, Dola, nav, RKNN decoder, occupancy grid, config
python scripts/aruco_demo.py        # visual ArUco check -> output/aruco_demo.png
python scripts/occupancy_demo.py    # visual occupancy grid -> output/occupancy_demo.png
```

**Tier 2** (needs hardware) — bring-up checks on the real machines:

- Mapping drone: confirm UWB topic publishes (`ros2 topic echo /uwb_tag`), MAVSDK connects, RealSense streams, offboard arms.
- Swarm: confirm Dola finds drone IPs on WiFi, `pyhulax` connects, video frames arrive.

## Tuning navigation

Edit `navigation` section in `config/challenge.yaml` (`kp_xy`, `max_vel_xy`, thresholds). Match organizer `kolomee.py` defaults first, then tune in arena.

## Important rules from organizers

- Use **offboard + velocity setpoints**, not MAVSDK position goto for mapping drone.
- Keep sending setpoints while in offboard (see `prime_offboard()`).
- Swarm loop must stay **non-blocking** — use per-drone states, not long `sleep()` for one drone.
