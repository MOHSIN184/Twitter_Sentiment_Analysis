import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "config.yaml",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    "LICENSE",
    "PROJECT_REPORT.md",
    "src/sentiment_pipeline/api/app.py",
    "scripts/12_run_api.py",
    "scripts/13_validate_artifacts.py",
    "models/baseline/best_model.joblib",
    "models/baseline/decision_threshold.json",
]

REQUIRED_FOLDERS = ["src", "scripts", "tests", "models", "outputs", "data"]


def run_repository_check() -> bool:
    """Run a GitHub submission readiness check."""
    print("Repository checklist")
    print("====================")
    success = True

    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            print(f"PASS file: {relative_path}")
        else:
            print(f"FAIL missing file: {relative_path}")
            success = False

    for relative_path in REQUIRED_FOLDERS:
        path = PROJECT_ROOT / relative_path
        if path.exists() and path.is_dir():
            print(f"PASS folder: {relative_path}")
        else:
            print(f"FAIL missing folder: {relative_path}")
            success = False

    raw_dataset = PROJECT_ROOT / "data/raw/twitter_sentiment_dataset.csv"
    if raw_dataset.exists() and raw_dataset.stat().st_size > 25 * 1024 * 1024:
        print("WARN raw dataset exists and is larger than 25 MB; do not commit it to GitHub")

    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size > 90 * 1024 * 1024:
            print(f"WARN large file over 90 MB: {path.relative_to(PROJECT_ROOT)}")

    if (PROJECT_ROOT / ".env").exists():
        print("WARN .env exists; do not commit secrets")

    for path in PROJECT_ROOT.rglob("__pycache__"):
        if path.is_dir():
            print(f"WARN __pycache__ folder exists: {path.relative_to(PROJECT_ROOT)}")

    if success:
        print("Repository check passed.")
    else:
        print("Repository check failed because required files or folders are missing.")
    return success


def main() -> None:
    """Run repository readiness checks and fail only for missing required items."""
    if not run_repository_check():
        sys.exit(1)


if __name__ == "__main__":
    main()
