"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML/JSON configuration file.

    Args:
        path: Path to config file.

    Returns:
        Parsed configuration dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        if path.suffix == ".json":
            import json
            return json.load(f)
        raise ValueError(f"Unsupported config format: {path.suffix}")
