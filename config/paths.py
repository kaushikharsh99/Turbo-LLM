from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"

MODEL_DIR = PROJECT_ROOT / "models"

CACHE_DIR = PROJECT_ROOT / "cache"

LOG_DIR = PROJECT_ROOT / "logs"

PROFILE_DIR = PROJECT_ROOT / "profiles"

TEMP_DIR = PROJECT_ROOT / "temp"

TEST_DIR = PROJECT_ROOT / "tests"

for directory in (
    CACHE_DIR,
    LOG_DIR,
    PROFILE_DIR,
    TEMP_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)