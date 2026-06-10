"""End-to-end dry-run produces expected output files."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dry_run_challenge1 import OUTPUT_DIR, run_dry_mission


def test_dry_run_writes_outputs(tmp_path, monkeypatch):
    out = tmp_path / "challenge1"
    monkeypatch.setattr("scripts.dry_run_challenge1.OUTPUT_DIR", out)

    asyncio.run(run_dry_mission(fast=True))

    assert (out / "landing_pad_report.json").exists()
    assert (out / "arena_map.png").exists()
    assert (out / "occupancy_wp00.png").exists()
    assert (out / "aruco_wp00.png").exists()
    assert (out / "dry_run_preview_wp00.png").exists()

    report = json.loads((out / "landing_pad_report.json").read_text(encoding="utf-8"))
    assert report["simulated"] is True
    assert report["challenge"] == 1
    assert len(report["observations"]) > 0
    assert len(report["obstacles"]) > 0
    assert len(report["detected_marker_ids"]) > 0
    assert len(report["all_landing_zones"]) >= len(report["valid_landing_zones"])
    assert len(report["valid_landing_zones"]) > 0
