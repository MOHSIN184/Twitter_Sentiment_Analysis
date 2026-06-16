from pathlib import Path
from typing import Any, Optional, Union

import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Optional[Union[Path, str]] = None) -> dict[str, Any]:
    """Load the YAML configuration file as a dictionary."""
    path = Path(config_path) if config_path is not None else get_project_root() / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration at {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")

    return config
