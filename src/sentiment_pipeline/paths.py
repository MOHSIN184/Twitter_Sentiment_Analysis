from pathlib import Path
from typing import Any

from sentiment_pipeline.config import get_project_root


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Build project paths from configuration and create output directories."""
    root = get_project_root()
    configured_paths = config.get("paths", {})

    paths = {
        "project_root": root,
        "raw_data": root / configured_paths["raw_data"],
        "interim_dir": root / configured_paths["interim_dir"],
        "processed_dir": root / configured_paths["processed_dir"],
        "reports_dir": root / configured_paths["reports_dir"],
        "models_dir": root / configured_paths["models_dir"],
        "figures_dir": root / configured_paths["figures_dir"],
        "logs_dir": root / configured_paths["logs_dir"],
    }
    paths["baseline_models_dir"] = paths["models_dir"] / "baseline"
    paths["advanced_models_dir"] = paths["models_dir"] / "advanced"
    paths["predictions_dir"] = root / config.get("inference", {}).get("batch_output_dir", "outputs/predictions")

    for key in (
        "interim_dir",
        "processed_dir",
        "reports_dir",
        "figures_dir",
        "logs_dir",
        "baseline_models_dir",
        "advanced_models_dir",
        "predictions_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths
