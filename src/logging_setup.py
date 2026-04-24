"""Logging bootstrap — stderr + rotating file."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_path: str | Path = "logs/trader.log", level: int = logging.INFO) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_h = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(formatter)
    root.addHandler(file_h)
