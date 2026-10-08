"""Logging setup for UATF command-line training runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_MANAGED_HANDLER = "_uatf_managed_handler"


def configure_logging(log_file: str | Path, *, console_stream: TextIO | None = None) -> None:
    """Send readable INFO messages to the console and timestamped DEBUG logs to a file."""
    logger = logging.getLogger("uatf")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stream_handler = logging.StreamHandler(console_stream or sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(stream_handler, _MANAGED_HANDLER, True)

    try:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    except OSError as error:
        stream_handler.close()
        raise OSError(f"Could not create experiment log {log_file}: {error}") from error
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    setattr(file_handler, _MANAGED_HANDLER, True)

    shutdown_logging()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def shutdown_logging() -> None:
    """Close handlers installed by :func:`configure_logging`."""
    logger = logging.getLogger("uatf")
    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()
    if not any(getattr(handler, _MANAGED_HANDLER, False) for handler in logger.handlers):
        logger.propagate = True
