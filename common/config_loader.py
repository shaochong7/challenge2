"""Load challenge.yaml from project root."""

from pathlib import Path
from typing import Any
from copy import deepcopy

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "config" / "challenge.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    try:
        import yaml  # type: ignore
    except ImportError:
        if cfg_path == _DEFAULT_CONFIG:
            from common.default_config import DEFAULT_CONFIG

            return deepcopy(DEFAULT_CONFIG)
        raise ImportError(
            "PyYAML is not installed. Install PyYAML or run with the default config."
        )
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
