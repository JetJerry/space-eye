"""Strict JSON loader and validator for experiment protocols.

Provides schema validation, helpful error messages, and deterministic protocol parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union
from pydantic import ValidationError

from vyom.core.exceptions import ProtocolError
from vyom.core.logging_setup import get_logger
from vyom.protocol.protocol_schema import ProtocolDefinition

logger = get_logger("protocol.loader")


class ProtocolLoadError(ProtocolError):
    """Base error for failures during protocol retrieval and parsing."""


class ProtocolNotFoundError(ProtocolLoadError):
    """Raised when the specified protocol JSON file does not exist on disk."""


class ProtocolValidationError(ProtocolLoadError):
    """Raised when protocol data violates Pydantic schema or procedural consistency rules."""


def load_protocol_from_dict(data: Dict[str, Any], source_desc: str = "dictionary") -> ProtocolDefinition:
    """Validate and instantiate a ProtocolDefinition from an in-memory dictionary."""
    if not isinstance(data, dict):
        raise ProtocolValidationError(
            f"Invalid protocol format in {source_desc}: Expected a JSON object/dict, got {type(data).__name__}"
        )

    try:
        protocol = ProtocolDefinition.model_validate(data)
        logger.info(
            "Successfully validated protocol '%s' (v%s) with %d steps from %s",
            protocol.experiment_name,
            protocol.protocol_version,
            len(protocol.steps),
            source_desc,
        )
        return protocol
    except ValidationError as e:
        error_msgs = []
        for err in e.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            error_msgs.append(f"  - [{loc}]: {msg}")
        formatted_err = "\n".join(error_msgs)
        raise ProtocolValidationError(
            f"Protocol schema validation failed for {source_desc}:\n{formatted_err}"
        ) from e
    except ValueError as e:
        raise ProtocolValidationError(
            f"Protocol procedural integrity error in {source_desc}: {e}"
        ) from e


def load_protocol_from_json(json_str: str, source_desc: str = "JSON string") -> ProtocolDefinition:
    """Parse a raw JSON string and validate against ProtocolDefinition."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ProtocolValidationError(
            f"Malformed JSON syntax in {source_desc} (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e

    return load_protocol_from_dict(data, source_desc=source_desc)


def load_protocol_from_file(file_path: Union[str, Path]) -> ProtocolDefinition:
    """Load a protocol JSON file from disk with strict schema validation."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise ProtocolNotFoundError(f"Protocol file not found: {path}")

    if not path.is_file():
        raise ProtocolLoadError(f"Protocol path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ProtocolLoadError(f"Failed to read protocol file '{path}': {e}") from e

    return load_protocol_from_json(content, source_desc=str(path))
