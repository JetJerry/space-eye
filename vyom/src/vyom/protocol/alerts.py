"""Alert throttling, cooldown tracking, and deterministic notification generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple
import uuid

from vyom.core.logging_setup import get_logger
from vyom.protocol.protocol_schema import ProcedureViolation, ViolationType

logger = get_logger("protocol.alerts")


@dataclass
class ProtocolAlert:
    """Formatted user-facing compliance alert."""
    alert_id: str
    violation: ProcedureViolation
    formatted_message: str
    voice_message: str
    timestamp_s: float
    severity: str = "warning"


class AlertCooldownTracker:
    """Deterministic throttle tracker preventing alert flooding during live video streams."""

    def __init__(self, default_cooldown_seconds: float = 3.0) -> None:
        self.default_cooldown_seconds = default_cooldown_seconds
        # Key: (violation_type, step_id, object_track_id) -> last_emitted_timestamp_s
        self._last_emission_times: Dict[Tuple[str, Optional[str], Optional[str]], float] = {}

    def should_emit(
        self,
        violation: ProcedureViolation,
        cooldown_seconds: Optional[float] = None,
    ) -> bool:
        """Evaluate if an alert of this type/step/object can be emitted without violating cooldown."""
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown_seconds
        key = (
            violation.violation_type.value,
            violation.step_id,
            violation.object_track_id,
        )

        last_time = self._last_emission_times.get(key)
        if last_time is None:
            return True

        elapsed = violation.timestamp_s - last_time
        return elapsed >= cooldown

    def record_emission(self, violation: ProcedureViolation) -> ProtocolAlert:
        """Record an alert emission timestamp and return a structured ProtocolAlert."""
        key = (
            violation.violation_type.value,
            violation.step_id,
            violation.object_track_id,
        )
        self._last_emission_times[key] = violation.timestamp_s

        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        voice_msg = violation.voice_message or violation.explanation
        formatted = f"[{violation.violation_type.value}] Step '{violation.step_id or '--'}': {violation.explanation}"

        logger.warning("Protocol Compliance Alert [%s]: %s", alert_id, formatted)

        return ProtocolAlert(
            alert_id=alert_id,
            violation=violation,
            formatted_message=formatted,
            voice_message=voice_msg,
            timestamp_s=violation.timestamp_s,
            severity=violation.severity,
        )

    def reset(self) -> None:
        """Clear all cooldown history."""
        self._last_emission_times.clear()

