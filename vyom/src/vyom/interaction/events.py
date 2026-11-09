"""Typed interaction events, states, and structured observation records.

Defines deterministic, explainable interaction events emitted when human hands
interact with tracked laboratory apparatus over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class InteractionEventType(str, Enum):
    """Categorical interaction event types emitted by the interaction engine."""
    HAND_APPROACH = "HAND_APPROACH"
    HAND_DEPART = "HAND_DEPART"
    CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
    OBJECT_MOVEMENT = "OBJECT_MOVEMENT"
    PICKUP_CANDIDATE = "PICKUP_CANDIDATE"
    PLACEMENT_CANDIDATE = "PLACEMENT_CANDIDATE"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


class InteractionState(str, Enum):
    """Operational state of a specific Hand-Object pair."""
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
    HOLDING_MOVING = "HOLDING_MOVING"
    PLACEMENT_CANDIDATE = "PLACEMENT_CANDIDATE"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


@dataclass
class InteractionEvent:
    """Structured, explainable interaction event record for compliance and audit logging.

    Preserves exact frame ID, timestamps, participating entities, geometric evidence,
    and coordinate system calibration provenance.
    """
    event_id: str
    event_type: InteractionEventType
    frame_id: int
    timestamp_s: float
    hand_id: str                      # "hand_0", "hand_1"
    handedness: str                   # "Right" or "Left"
    object_track_id: str              # e.g., "pipette_track_1"
    object_class: str                 # e.g., "pipette", "beaker"
    detection_confidence: float       # Object detector confidence [0.0, 1.0]
    interaction_confidence: float     # Temporal-geometric evidence confidence [0.0, 1.0]
    supporting_evidence: Dict[str, Any]  # Distances, velocities, duration, etc.
    coordinate_system: str = "IMAGE_PIXEL"  # "IMAGE_PIXEL" or "RIG_METRIC"
    calibration_status: bool = False
    rig_position: Optional[Tuple[float, float, float]] = None
    spatial_uncertainty: str = "UNCALIBRATED_2D"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for JSON protocol validation, timeline logging, and audit records."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "frame_id": self.frame_id,
            "timestamp_s": round(self.timestamp_s, 4),
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "detection_confidence": round(self.detection_confidence, 4),
            "interaction_confidence": round(self.interaction_confidence, 4),
            "coordinate_system": self.coordinate_system,
            "calibration_status": self.calibration_status,
            "spatial_uncertainty": self.spatial_uncertainty,
            "rig_position": [round(v, 4) for v in self.rig_position] if self.rig_position else None,
            "supporting_evidence": self.supporting_evidence,
        }

    def summary_line(self) -> str:
        """Concise one-line summary string for UI timeline displays."""
        calib_tag = "[RIG]" if self.calibration_status else "[2D]"
        return (
            f"{self.handedness} Hand -> {self.object_track_id} ({self.object_class}): "
            f"{self.event_type.value} {calib_tag} (conf: {self.interaction_confidence:.0%})"
        )

