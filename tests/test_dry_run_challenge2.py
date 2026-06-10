"""Challenge 2 dry-run reaches landing zones."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dry_run_challenge2 import run_dry_swarm


def test_dry_run_swarm_finds_convoy_and_lands():
    run_dry_swarm(fast=True)
    log = ROOT / "output" / "challenge2" / "dry_run_log.txt"
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    assert "state=DONE" in text
    # The swarm should collectively find all 5 ground robots via coverage search
    assert "Total unique robots found: 5/5" in text
    # ...and every drone should land on its assigned pad before searching
    assert "Landing pads visited: 3/3" in text
    assert "Final landings: 3/3" in text
    assert "pad_landed=True" in text
    assert "final_landed=True" in text
