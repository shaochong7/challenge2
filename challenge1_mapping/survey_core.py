"""Shared survey logic used by real mission and laptop dry-run."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from challenge1_mapping.arena_map import ArenaMap, marker_world_position
from challenge2_swarm.search_pattern import Region, lawnmower_waypoints
from detection.aruco_depth import ArucoDepthDetector
from detection.occupancy_grid import (
    GridConfig,
    build_occupancy_grid,
    obstacle_points_from_depth,
)
from detection.realsense_capture import FramePair


def build_survey_waypoints(cfg: dict) -> list[dict]:
    """Survey waypoints in UWB N/E.

    If mapping_drone.auto_survey is true, generate a lawnmower grid that covers
    the whole anchor safe-zone (uwb_bounds inset by safety_margin_m) so the drone
    maps the entire arena. Otherwise use the manual survey_waypoints list.
    """
    m = cfg["mapping_drone"]
    manual = [
        {"n": float(w["n"]), "e": float(w["e"])}
        for w in m.get("survey_waypoints", [])
    ]
    if not m.get("auto_survey", False):
        return manual

    arena = cfg.get("arena", {})
    bounds = arena.get("uwb_bounds", {})
    margin = float(arena.get("safety_margin_m", 0.5))
    region = Region(
        n_min=float(bounds.get("n_min", -5.0)) + margin,
        n_max=float(bounds.get("n_max", 5.0)) - margin,
        e_min=float(bounds.get("e_min", -5.0)) + margin,
        e_max=float(bounds.get("e_max", 5.0)) - margin,
    )
    spacing = float(m.get("survey_spacing_m", 2.0))
    pts = lawnmower_waypoints(region, spacing)
    return [{"n": n, "e": e} for n, e in pts]


def process_waypoint(
    waypoint_index: int,
    drone_n: float,
    drone_e: float,
    frames: FramePair,
    aruco: ArucoDepthDetector,
    arena: ArenaMap,
    observations: list[dict],
    output_dir: Path,
    grid_cfg: GridConfig | None = None,
) -> None:
    """Detect pads, build occupancy grid, update arena map and observations."""
    grid_cfg = grid_cfg or GridConfig()
    arena.add_path_point(drone_n, drone_e)

    depth_m = frames.depth_mm.astype("float32") / 1000.0
    grid = build_occupancy_grid(
        depth_m,
        frames.intrinsics.fx,
        frames.intrinsics.fy,
        frames.intrinsics.cx,
        frames.intrinsics.cy,
        grid_cfg,
    )
    cv2.imwrite(str(output_dir / f"occupancy_wp{waypoint_index:02d}.png"), grid)

    for point in obstacle_points_from_depth(
        depth_m,
        frames.intrinsics.fx,
        frames.intrinsics.fy,
        frames.intrinsics.cx,
        frames.intrinsics.cy,
        grid_cfg,
    ):
        world_n, world_e = marker_world_position(
            drone_n, drone_e, point.x_m, point.y_m, arena.cfg
        )
        arena.stamp_obstacle(
            world_n,
            world_e,
            height_m=point.height_m,
            distance_m=point.distance_m,
            waypoint_index=waypoint_index,
        )

    annotated = frames.color_bgr.copy()
    markers = aruco.detect(annotated, frames.depth_mm, draw=True)
    for obs in markers:
        world_n, world_e = marker_world_position(
            drone_n, drone_e, obs.x_m, obs.y_m, arena.cfg
        )
        arena.add_landing_pad(obs.marker_id, obs.valid_landing, world_n, world_e)
        observations.append(
            {
                "waypoint_index": waypoint_index,
                "drone_n": drone_n,
                "drone_e": drone_e,
                "world_n": world_n,
                "world_e": world_e,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **asdict(obs),
            }
        )
        status = "VALID" if obs.valid_landing else "INVALID"
        color = (0, 220, 0) if obs.valid_landing else (0, 0, 255)
        cv2.putText(
            annotated,
            f"id={obs.marker_id} {status} N={world_n:.2f} E={world_e:.2f}",
            (obs.center_u + 8, max(18, obs.center_v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        print(
            f"  ArUco id={obs.marker_id} {status} "
            f"world N={world_n:.2f} E={world_e:.2f} (z={obs.z_m:.2f}m)"
        )
    if not markers:
        cv2.putText(
            annotated,
            "NO ARUCO",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / f"aruco_wp{waypoint_index:02d}.png"), annotated)


def save_mission_report(
    arena: ArenaMap,
    observations: list[dict],
    output_dir: Path,
    *,
    simulated: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "arena_map.png"), arena.render_bgr())

    valid_pads = [
        {"marker_id": p.marker_id, "n": p.n, "e": p.e}
        for p in arena.pads
        if p.valid
    ]
    all_pads = [
        {
            "marker_id": p.marker_id,
            "valid_landing": p.valid,
            "n": p.n,
            "e": p.e,
        }
        for p in arena.pads
    ]
    report = {
        "challenge": 1,
        "simulated": simulated,
        "arena_bounds": {
            "n_min": arena.cfg.n_min,
            "n_max": arena.cfg.n_max,
            "e_min": arena.cfg.e_min,
            "e_max": arena.cfg.e_max,
        },
        "observations": observations,
        "obstacles": [
            {
                "n": o.n,
                "e": o.e,
                "height_m": o.height_m,
                "distance_m": o.distance_m,
                "waypoint_index": o.waypoint_index,
            }
            for o in sorted(
                arena.obstacles.values(),
                key=lambda item: (item.waypoint_index, item.n, item.e),
            )
        ],
        "detected_marker_ids": sorted({p.marker_id for p in arena.pads}),
        "all_landing_zones": all_pads,
        "valid_landing_ids": sorted({p.marker_id for p in arena.pads if p.valid}),
        "valid_landing_zones": valid_pads,
    }
    out_path = output_dir / "landing_pad_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {out_path}")
    print(f"Arena map saved: {output_dir / 'arena_map.png'}")
    print(f"Mapped obstacle cells: {len(arena.obstacles)}")
    print(f"Detected marker IDs: {report['detected_marker_ids']}")
    print(f"Valid landing zones: {len(valid_pads)}")
