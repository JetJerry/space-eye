"""Structured logging system for VYOM.

Supports console output with readable formatting and rotating file logs
with optional structured JSON format for automated auditing and post-run analysis.
"""

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include structured extra fields attached to the record
        if hasattr(record, "extra_context") and isinstance(record.extra_context, dict):
            log_entry["context"] = record.extra_context

        for standard_key in (
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName", "extra_context"
        ):
            pass

        # Capture any custom extra attributes passed to logger calls
        custom_extras = {
            k: v for k, v in record.__dict__.items()
            if not k.startswith("_") and k not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName", "extra_context"
            )
        }
        if custom_extras:
            log_entry.setdefault("context", {}).update(custom_extras)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class StandardConsoleFormatter(logging.Formatter):
    """Readable console formatter for interactive development."""

    FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path | str] = None,
    log_file: str = "vyom.log",
    console_output: bool = True,
    json_format: bool = False,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Initialize structured application logging.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_dir: Directory where log files are stored. Defaults to 'logs/'.
        log_file: Name of rotating log file.
        console_output: Whether to output to standard error.
        json_format: Whether the file log should use JSON formatting.
        max_bytes: Max size before rotating log file.
        backup_count: Number of rotating backups to retain.

    Returns:
        The configured root 'vyom' logger.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("vyom")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger.handlers.clear()

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(StandardConsoleFormatter())
        root_logger.addHandler(console_handler)

    # File Handler
    if log_dir is not None:
        target_dir = Path(log_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / log_file

        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)

        if json_format:
            file_handler.setFormatter(StructuredJsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] [%(process)d:%(threadName)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        root_logger.addHandler(file_handler)

    root_logger.propagate = False
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a namespaced VYOM logger."""
    if not name:
        return logging.getLogger("vyom")
    if name.startswith("vyom."):
        return logging.getLogger(name)
    return logging.getLogger(f"vyom.{name}")

