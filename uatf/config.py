"""Helpers for loading UATF YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration whose top-level value is a mapping."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as source:
        configuration = yaml.safe_load(source)

    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return configuration

