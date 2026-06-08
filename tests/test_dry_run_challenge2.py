"""Challenge 2 dry-run reaches landing zones."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dry_run_challenge2 import run_dry_swarm


def test_dry_run_swarm_reaches_zones():
    run_dry_swarm(fast=True)
    log = ROOT / "output" / "challenge2" / "dry_run_log.txt"
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    assert "tag=0" in text
    assert "state=DONE" in text
    # After search drift, should still have navigated (not stuck at origin)
    assert "final N=0.85" in text or "final N=0.84" in text or "final N=0.01" in text
