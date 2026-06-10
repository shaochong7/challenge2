# RoboVerse Drone Challenge (University)

Competition code for **Challenge 1** (mapping drone + ArUco landing pads) and **Challenge 2** (3× HULA swarm + convoy detection), built from organizer references:

- `move_it.py` — MAVSDK position-NED offboard navigation
- `huladola.py` + `dola.py` — swarm discovery and video
- ArUco + depth sample — fiducial + RealSense deprojection

## Repository layout

```
config/challenge.yaml      # IDs, waypoints, gains — edit before competition
common/                    # UWB ROS listener, position-NED / velocity nav helpers
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

## Emergency landing (fail-safe)

Drones **land instead of flying off or dropping** whenever something goes wrong
(`common/emergency.py`):

- **Ctrl+C / kill the program** → SIGINT/SIGTERM are intercepted and the drone lands
  (mapping: stop offboard + `action.land()`; swarm: `api.land()` on every HULA)
- **Crash / unhandled error** → emergency land runs before the exception propagates
- **Dangerous location (geofence breach)** → land in place immediately

It never uses `action.kill()` (that cuts motors and makes the drone drop) — always a
controlled land. Verified by `tests/test_emergency.py` with fake drones.

## UWB geofence (stay inside anchor coverage)

UWB position degrades outside the anchor zone and causes erratic velocity-control
flight. `common/geofence.py` enforces:

- **Preflight:** reject waypoints/search targets outside the safe zone (anchor bounds
  minus `arena.safety_margin_m`)
- **In flight:** every nav tick checks live UWB; if outside anchors → zero velocity /
  hover and abort (mapping) or halt that drone (swarm)

Set `arena.uwb_bounds` in `config/challenge.yaml` to the measured anchor coverage
on competition day.

## Speed limits, HULA height & obstacle avoidance (finals brief rules)

The finals brief sets hard rules — these are enforced in `config/challenge.yaml`:

- **Mapping drone max 0.3 m/s** → `navigation.max_vel_xy: 0.3` (and `max_vel_z: 0.3`).
- **HULA max 0.5 m/s** → `swarm.move_speed: 0.5`. The swarm's P-controller cap is set
  to `move_speed` in `swarm_core` so it is *independent* of the mapping drone's 0.3 limit.
  (Assumes pyhulax `move()` speed is m/s — **verify the unit on the test drone**.)
- **HULA recommended height 1.1 m** → `swarm.hover_height_m: 1.1` (applied on takeoff;
  the code tries `takeoff(height)` and falls back to a plain `takeoff()`).
- **Strictly no flying over obstacles** (violation invalidates the score). The HULA flies
  at a fixed low height, so obstacles are avoided **horizontally** — never by climbing.

Obstacle avoidance (`challenge2_swarm/obstacle.py` + nav in `uwb_nav.py`):
- The nav layer asks an `ObstacleSensor` "how far is the nearest obstacle if I move
  N/S/E/W?" and picks the first **clear** direction toward the target; if the straight
  path is blocked it **sidesteps around** the obstacle. If every direction is blocked it
  **holds position** (never climbs over). Unreachable search waypoints are skipped after
  `swarm.search_wp_timeout_s`.
- Two sensor backends (config `swarm.obstacle_source`):
  - `lidar` → `HulaObstacleSensor` reads the HULA's onboard obstacle sensing via pyhulax.
    **The exact SDK call must be confirmed on the unit** and wired into
    `HulaObstacleSensor._default_reader`. Until wired it **fails safe**: the mission
    refuses to fly rather than risk flying over an obstacle.
  - `map` → `MapObstacleSensor` uses obstacle boxes exported by Challenge 1 (UWB-based,
    no lidar needed).
- Tune `swarm.obstacle_stop_distance_m` (how close before avoiding) and
  `swarm.obstacle_clearance_m` (inflation ≈ drone radius + buffer).

## Mission handoff: mapping drone → swarm

The mapping drone (Challenge 1) produces the map; the swarm (Challenge 2) consumes it.
`output/challenge1/landing_pad_report.json` carries:
- `valid_landing_zones` — world N/E of valid pads → swarm flies to / lands on these.
- `arena_bounds` — the surveyed N/E extent → swarm's lawnmower search covers exactly
  this area (`swarm.use_map_bounds: true`). Falls back to `swarm.search_area` if no map.

Markers are **20 cm × 20 cm**. With the size known, the ArUco detector falls back to
**pose estimation (`solvePnP`)** when depth has holes — common on flat markers at 3.5 m —
instead of dropping the detection. At 3.5 m a 20 cm marker is only ~50 px wide (1280 px,
D430), so keep resolution high and verify detection in the arena.
## Cameras & flight height

- **Mounting:** down-facing (matches organizer `generateTopDown.py`).
- **Model:** Intel RealSense **D430 or D450**; **resolution is configurable**. Keep
  `camera_width/height` high (default 1280x720) so marker IDs stay crisp.
- **Mapping height:** current config uses `mapping_drone.takeoff_height_m: 2.0`,
  matching the organiser `move_it.py` sample. Verify onsite that this safely clears
  obstacles and satisfies the safety brief.

Footprint = how much ground one frame covers, from `common/camera_model.py`. At 2.0 m:

| Module | Color footprint | Suggested survey spacing |
|---|---|---|
| D430 | ~2.8 x 1.5 m | ~1.2 m |
| D450 | ~4.0 x 2.5 m | ~2.0 m |

Consequences baked into the code/config:
- **Mapping legs are tighter at 2 m.** Current `mapping_drone.survey_spacing_m` is
  `1.2` for safe overlap with D430.
- A 20 cm marker is easier to read at 2 m than at higher altitude, but coverage
  requires more survey legs.
- A down-facing camera sees the floor everywhere; obstacle extraction uses surfaces
  that are closer than the estimated floor plane.
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

1. **ArUco dictionary:** current organiser update is `DICT_7X7_1000`.
2. **Marker IDs:** current organiser update is `[11, 45, 51, 67, 101]`.
3. Measure the **`arena.uwb_bounds`** (anchor coverage). With `mapping_drone.auto_survey: true`
   the drone auto-generates a full-area lawnmower survey over the safe-zone — tune
   `survey_spacing_m` to the camera footprint. (Set `auto_survey: false` to use manual
   `survey_waypoints` instead.)
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

1. Subscribes to UWB (`uwb_tag` topic) for geofence/map coordinates
2. Arms, starts offboard, climbs to `takeoff_height_m` (≥ 3.5 m)
3. Flies a **full-area lawnmower survey** (auto-generated over the anchor safe-zone) using
   MAVSDK **position-NED offboard setpoints** (`set_position_velocity_ned`) — *move_it.py*
4. Hovers at each point, grabs aligned RealSense color+depth — *getSyncDepthColor.py*
5. Builds a **top-down occupancy grid** per waypoint — *generateTopDown.py*
6. Detects ArUco, classifies valid/invalid, converts each pad to **world N/E** coordinates — *ArUco sample*

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
3. Each drone gets a distinct valid landing pad from the Challenge 1 report
4. Per-drone **state machine**: `TAKEOFF → SEARCH → GO_TO_ZONE → LAND → DONE`
   (`SNAPSHOT` is entered from any moving state and resumes it)

**Two scoring goals, one flow:**
- **Recon — find + snapshot the convoy.** In `SEARCH` each drone flies a **lawnmower
  (boustrophedon) path** over its strip (`challenge2_swarm/search_pattern.py`), running the
  detector (`challenge2_swarm/target_sensor.py`) every tick. A *new* robot → `SNAPSHOT` (annotated
  image to `output/snapshots/`) → resume. Robots are de-duplicated (`target_dedup_m` /
  `snapshot_cooldown_s`). Detection also runs in `GO_TO_ZONE`, so robots seen **while flying to the
  pad** are snapshotted too.
- **Deployment — occupy the pads.** After searching its strip, the drone flies to its assigned
  landing zone (`GO_TO_ZONE`) and **lands on it** (`LAND` → `api.land()`), so all valid pads end up
  occupied. Set `swarm.do_area_search: false` to skip the search and go straight to the pad.

The arena is split into one **vertical strip per drone** so coverage doesn't overlap. The detector
is swappable behind one interface: `YoloTargetSensor` (real YOLO) vs `SimTargetSensor` (dry-run
convoy); the state machine is identical for both.

**SDK docs:** https://pyhulax.xenops.ae

**UWB on C2** (organizer `UWBParserThread.py` → `common/uwb_c2.py`):
- USB serial parser gives each drone tag's `(N, E)` position
- Navigation uses the same P-controller as the mapping drone, mapped to `pyhulax` `move(Direction, speed)`
- Reads ambush/landing targets from Challenge 1 `valid_landing_zones`
- **Not** pyhulax built-in auto-land — optional ArUco near pads is separate visual aid

**Laptop dry-run:**
```bash
python scripts/dry_run_challenge1.py --fast   # produces landing_pad_report.json
python scripts/dry_run_challenge2.py --fast # 3 fake HULAs lawnmower-search a 5-robot convoy
```

The Challenge 2 dry-run spawns a simulated convoy (`challenge2_swarm/sim/ground_robots.py`) and
reports how many of the 5 robots the swarm collectively found, with snapshots in `output/snapshots/`.

## Practice: object detection

- **ArUco / AprilTag / QR:** no training — `detection/aruco_depth.py`
- **RoboMaster bodies:** label data → `python scripts/train_yolo.py`
- **Mapping drone NPU:** export trained model with organizer ONNX/RKNN scripts on Discord

## Testing

Two tiers. **Tier 1** runs on any laptop (no drone/camera/ROS2):

```bash
pip install opencv-contrib-python numpy PyYAML pytest
python -m pytest tests/ -v          # unit tests (incl. full dry-run)
python scripts/aruco_demo.py        # visual ArUco check -> output/aruco_demo.png
python scripts/occupancy_demo.py    # visual occupancy grid -> output/occupancy_demo.png
python scripts/dry_run_challenge1.py --fast   # full simulated mission (~1 min)
```

**RealSense-only ArUco check on the mapping drone** (no arming / no flight):

```bash
python3 scripts/realsense_aruco_check.py
python3 scripts/realsense_aruco_check.py --frames 30 --show
```

Annotated frames are saved to `output/aruco_check/`.

**Dry-run** (`scripts/dry_run_challenge1.py`) fakes UWB navigation + down-facing
camera, runs the same survey loop as the real drone, and writes:

- `output/challenge1/landing_pad_report.json` (`simulated: true`)
- `output/challenge1/arena_map.png`
- `output/challenge1/occupancy_wpNN.png`
- `output/challenge1/dry_run_preview_wpNN.png` (camera view with ArUco boxes)

**Tier 2** (needs hardware) — bring-up checks on the real machines:

- Mapping drone: confirm UWB topic publishes (`ros2 topic echo /uwb_tag`), MAVSDK connects, RealSense streams, offboard arms.
- Swarm: confirm Dola finds drone IPs on WiFi, `pyhulax` connects, video frames arrive.

## Tuning navigation

Edit `navigation` section in `config/challenge.yaml` (`kp_xy`, `max_vel_xy`, thresholds). The mapping drone now uses organiser `move_it.py` style position-NED setpoints, with UWB still used for geofence and arrival checks.

## Important rules from organizers

- Use **offboard + position-NED setpoints** for the mapping drone, following organiser `move_it.py`.
- Keep sending/holding setpoints while in offboard.
- Swarm loop must stay **non-blocking** — use per-drone states, not long `sleep()` for one drone.

