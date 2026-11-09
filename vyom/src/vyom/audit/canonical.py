"""Deterministic canonical JSON serialization and cryptographic hash utilities.

Ensures byte-level consistency across platforms, architectures, and Python versions.
Part of VYOM Prompt 11: SHA-256 Hash-Chain Audit Logging.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict


def _sanitize_for_canonical_json(obj: Any) -> Any:
    """Recursively enforce deterministic types and reject non-finite floating point values."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Non-finite float value '{obj}' is forbidden in canonical audit records.")
        # Normalize -0.0 to 0.0
        if obj == 0.0:
            return 0.0
        # Round floating points to 6 decimal places to prevent float precision drift
        return round(obj, 6)
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_canonical_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_canonical_json(v) for v in obj]
    return obj


def canonical_dumps(payload: Any) -> bytes:
    """Serialize payload into strict, deterministic canonical UTF-8 bytes.

    Specification:
    - UTF-8 encoding without BOM
    - Lexicographically sorted dictionary keys (sort_keys=True)
    - Compact separators without whitespace: (',', ':')
    - Prohibition of NaN, Infinity, -Infinity
    - Deterministic float normalization
    """
    sanitized = _sanitize_for_canonical_json(payload)
    json_str = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return json_str.encode("utf-8")


def sha256_digest(data: bytes | str) -> str:
    """Compute standard SHA-256 hexadecimal string digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

