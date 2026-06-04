# RoboVerse Drone Challenge (University)

Competition code for **Challenge 1** (mapping drone + ArUco landing pads) and **Challenge 2** (3× HULA swarm + convoy detection), built from organizer references:

- `kolomee.py` — UWB + velocity offboard navigation
- `huladola.py` + `dola.py` — swarm discovery and video
- ArUco + depth sample — fiducial + RealSense deprojection

## Repository layout

```
config/challenge.yaml      # IDs, waypoints, gains — edit before competition
common/                    # UWB ROS listener, velocity navigator
detection/                 # ArUco+depth, RealSense, YOLO helper
challenge1_mapping/        # Mapping mission → landing_pad_report.json
challenge2_swarm/          # Swarm FSM + snapshots
run_challenge1.py
run_challenge2.py
scripts/train_yolo.py
```

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

**What it does:**

1. Subscribes to UWB (`uwb_tag` topic)
2. Arms, starts offboard, flies survey waypoints using **velocity control** (not position goto)
3. Hovers at each point, grabs aligned RealSense color+depth
4. Detects ArUco, classifies valid/invalid pads, writes `output/challenge1/landing_pad_report.json`

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

## Tuning navigation

Edit `navigation` section in `config/challenge.yaml` (`kp_xy`, `max_vel_xy`, thresholds). Match organizer `kolomee.py` defaults first, then tune in arena.

## Important rules from organizers

- Use **offboard + velocity setpoints**, not MAVSDK position goto for mapping drone.
- Keep sending setpoints while in offboard (see `prime_offboard()`).
- Swarm loop must stay **non-blocking** — use per-drone states, not long `sleep()` for one drone.
