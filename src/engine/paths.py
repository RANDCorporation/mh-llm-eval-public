from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def data_path(subdir: str = "") -> Path:
    base = REPO_ROOT / "data"
    return base / subdir if subdir else base
