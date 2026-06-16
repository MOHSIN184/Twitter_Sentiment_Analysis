import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "13_validate_artifacts.py"
SPEC = importlib.util.spec_from_file_location("validate_artifacts_script", SCRIPT_PATH)
validate_artifacts_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_artifacts_script)


def test_valid_threshold_between_zero_and_one_passes() -> None:
    assert validate_artifacts_script.validate_threshold_value(0.47)


def test_invalid_threshold_below_zero_fails() -> None:
    assert not validate_artifacts_script.validate_threshold_value(-0.01)


def test_invalid_threshold_above_one_fails() -> None:
    assert not validate_artifacts_script.validate_threshold_value(1.01)


def test_missing_model_path_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_artifacts_script.validate_model_path(tmp_path / "missing.joblib")
