import logging
import sys
from pathlib import Path

from config.paths import LOG_DIR

LOG_FILE = LOG_DIR / "turbo.log"

def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file = logging.FileHandler(LOG_FILE)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger