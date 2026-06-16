import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("sentiment_pipeline")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not load CSV dataset from {path}: {exc}") from exc

    LOGGER.info("Loaded dataset from %s with %s rows and columns: %s", path, len(df), list(df.columns))
    return df
