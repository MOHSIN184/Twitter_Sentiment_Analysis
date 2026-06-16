from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist() -> None:
    required_files = [
        "README.md",
        "requirements.txt",
        "config.yaml",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "LICENSE",
        "PROJECT_REPORT.md",
    ]
    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_required_source_folders_exist() -> None:
    required_folders = ["src", "scripts", "tests", "models", "outputs", "data"]
    for relative_path in required_folders:
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path


def test_readme_contains_important_sections() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for section in [
        "Overview",
        "Features",
        "Installation",
        "Run API",
        "API Endpoints",
        "Docker Usage",
        "Tests",
        "Artifact Validation",
    ]:
        assert section in readme


def test_env_example_and_gitignore_exist() -> None:
    assert (PROJECT_ROOT / ".env.example").exists()
    assert (PROJECT_ROOT / ".gitignore").exists()
