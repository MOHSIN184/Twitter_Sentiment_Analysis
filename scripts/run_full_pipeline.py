import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path) -> None:
    """Run one pipeline script and stop on failure."""
    print(f"\nRunning {script_path.name}...")
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parents[1], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline step failed: {script_path.name}")


def main() -> None:
    """Run analysis, cleaning, and splitting in sequence."""
    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"

    steps = [
        scripts_dir / "01_analyze_dataset.py",
        scripts_dir / "02_clean_dataset.py",
        scripts_dir / "03_split_dataset.py",
        scripts_dir / "04_train_baseline.py",
        scripts_dir / "05_evaluate_model.py",
        scripts_dir / "06_compare_models.py",
        scripts_dir / "07_error_analysis.py",
        scripts_dir / "09_tune_threshold.py",
        scripts_dir / "08_evaluate_best_model.py",
    ]

    for step in steps:
        run_step(step)

    print("\nPipeline completed successfully.")
    print("Generated reports, processed datasets, model comparisons, best model, predictions, and evaluation figures.")


if __name__ == "__main__":
    main()
