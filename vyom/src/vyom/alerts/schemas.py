"""Schemas, enums, and data models for offline voice alerts and speech synthesis.

Part of VYOM Prompt 10: Local Voice Alerts and Experiment Guidance System.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlertCategory(str, Enum):
    """Categorical classification of experiment procedure voice notifications."""
    SESSION_STARTED = "SESSION_STARTED"
    STEP_INSTRUCTION = "STEP_INSTRUCTION"
    STEP_COMPLETED = "STEP_COMPLETED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    STEP_TIMEOUT = "STEP_TIMEOUT"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    LOW_CONFIDENCE_WARNING = "LOW_CONFIDENCE_WARNING"
    PROCEDURE_COMPLETED = "PROCEDURE_COMPLETED"
    PROCEDURE_ABORTED = "PROCEDURE_ABORTED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class VoiceAlertPriority(IntEnum):
    """Priority hierarchy for speech queue scheduling. Lower values indicate higher urgency."""
    CRITICAL = 0    # Out of order, step timeout, critical procedural violation, system error
    HIGH = 1        # Calibration required, blocked procedure
    MEDIUM = 2      # Active step instruction, low confidence guidance
    LOW = 3         # Step completion feedback, session started/completed notification


class VoiceEngineStatus(str, Enum):
    """Operational status of the offline speech synthesis worker."""
    READY = "READY"
    SPEAKING = "SPEAKING"
    MUTED = "MUTED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class VoiceInfo(BaseModel):
    """Metadata describing an installed offline speech driver voice."""
    voice_id: str = Field(description="Unique driver token or registry ID")
    name: str = Field(description="Human-readable voice identifier")
    languages: List[str] = Field(default_factory=list, description="Supported IETF BCP 47 or locale codes")
    gender: Optional[str] = Field(default=None, description="Reported voice gender if available")


@dataclass(order=True)
class VoiceAlertItem:
    """Prioritized speech item dispatched to the background VoiceWorker queue."""
    # Priority sorting keys: lowest priority_rank first, then earliest timestamp
    priority_rank: int
    timestamp_s: float

    # Non-comparable alert payload
    alert_id: str = field(compare=False)
    category: AlertCategory = field(compare=False)
    text: str = field(compare=False)
    reason: str = field(compare=False, default="")
    event_id: Optional[str] = field(compare=False, default=None)
    step_id: Optional[str] = field(compare=False, default=None)
    can_interrupt: bool = field(compare=False, default=False)
    severity: str = field(compare=False, default="warning")  # "info", "warning", "critical"

    @classmethod
    def create(
        cls,
        alert_id: str,
        category: AlertCategory,
        text: str,
        priority: VoiceAlertPriority,
        reason: str = "",
        event_id: Optional[str] = None,
        step_id: Optional[str] = None,
        can_interrupt: bool = False,
        severity: str = "warning",
        timestamp_s: Optional[float] = None,
    ) -> VoiceAlertItem:
        """Factory helper creating a properly prioritized VoiceAlertItem."""
        t = timestamp_s if timestamp_s is not None else time.time()
        return cls(
            priority_rank=priority.value,
            timestamp_s=t,
            alert_id=alert_id,
            category=category,
            text=text,
            reason=reason,
            event_id=event_id,
            step_id=step_id,
            can_interrupt=can_interrupt,
            severity=severity,
        )
