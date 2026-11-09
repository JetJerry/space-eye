"""Pydantic schema definitions, violation models, and validation types for versioned experiment protocols.

Provides deterministic validation models compliant with VYOM Prompt 9.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StepState(str, Enum):
    """Execution status for an individual procedure step."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    PAUSED = "PAUSED"


class ProtocolLifecycleState(str, Enum):
    """Overall execution lifecycle of an experiment procedure."""
    NOT_STARTED = "NOT_STARTED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


# Alias for backward-compatibility
ProtocolState = ProtocolLifecycleState


class ViolationType(str, Enum):
    """Deterministic procedure compliance violation taxonomy."""
    OUT_OF_ORDER = "OUT_OF_ORDER"
    SKIPPED_STEP = "SKIPPED_STEP"
    INVALID_OBJECT = "INVALID_OBJECT"
    MISSING_PREREQUISITE = "MISSING_PREREQUISITE"
    INCOMPLETE_ACTION = "INCOMPLETE_ACTION"
    UNCERTAIN_OBSERVATION = "UNCERTAIN_OBSERVATION"
    TIMEOUT = "TIMEOUT"
    UNAUTHORIZED_INTERACTION = "UNAUTHORIZED_INTERACTION"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    PROCEDURE_BLOCKED = "PROCEDURE_BLOCKED"


class TargetZoneRequirement(BaseModel):
    """Configurable 3D metric rig coordinate bounding box or image-space target zone."""
    zone_id: str = Field(description="Unique zone slug e.g. 'rack_zone_a'")
    name: Optional[str] = Field(default=None, description="Human readable zone name")
    # 3D Rig coordinates [x, y, z] in meters relative to ArUco origin
    min_bounds_m: Optional[List[float]] = Field(
        default=None,
        description="Minimum metric bounds [x_min, y_min, z_min] in meters",
    )
    max_bounds_m: Optional[List[float]] = Field(
        default=None,
        description="Maximum metric bounds [x_max, y_max, z_max] in meters",
    )
    center_m: Optional[List[float]] = Field(
        default=None,
        description="Target center point [x, y, z] in meters",
    )
    radius_m: Optional[float] = Field(
        default=None,
        description="Acceptance radius in meters from center_m",
    )
    # Image space normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1]
    image_bbox_norm: Optional[List[float]] = Field(
        default=None,
        description="Permitted image-space normalized fallback box [ymin, xmin, ymax, xmax]",
    )

    def contains_rig_point(self, pt: Tuple[float, float, float]) -> bool:
        """Check whether 3D rig point pt=(x, y, z) lies within zone boundaries."""
        x, y, z = pt
        if self.min_bounds_m and self.max_bounds_m:
            if len(self.min_bounds_m) >= 3 and len(self.max_bounds_m) >= 3:
                in_x = self.min_bounds_m[0] <= x <= self.max_bounds_m[0]
                in_y = self.min_bounds_m[1] <= y <= self.max_bounds_m[1]
                in_z = self.min_bounds_m[2] <= z <= self.max_bounds_m[2]
                return in_x and in_y and in_z

        if self.center_m and self.radius_m is not None:
            if len(self.center_m) >= 3:
                dx = x - self.center_m[0]
                dy = y - self.center_m[1]
                dz = z - self.center_m[2]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                return dist <= self.radius_m

        return True


class ProtocolStep(BaseModel):
    """Validation schema for a single deterministic procedure step."""
    step_id: str = Field(description="Unique identifier (e.g., 'STEP_01')")
    order: int = Field(ge=1, description="1-indexed sequence order")
    title: str = Field(
        default="",
        alias="step_name",
        description="Short human-readable step title",
    )
    human_instruction: str = Field(
        default="",
        alias="instruction",
        description="Detailed operator instructions",
    )
    expected_event: str = Field(
        description="Target interaction event (e.g., 'PICKUP_CANDIDATE', 'OBJECT_MOVEMENT', 'PLACEMENT_CANDIDATE', 'RELEASE_CANDIDATE')",
    )
    expected_events: Optional[List[str]] = Field(
        default=None,
        description="Optional list of accepted interaction event aliases",
    )
    required_objects: List[str] = Field(
        default_factory=list,
        alias="expected_objects",
        description="Target object classes participating in interaction",
    )
    preconditions: List[str] = Field(
        default_factory=list,
        alias="prerequisites",
        description="Step IDs that must be completed before this step",
    )
    minimum_evidence_duration_ms: int = Field(
        default=500,
        ge=0,
        alias="stable_duration_ms",
        description="Minimum consecutive temporal evidence required for completion",
    )
    min_consecutive_frames: int = Field(
        default=3,
        ge=1,
        description="Minimum consecutive frames interaction must be observed",
    )
    timeout_seconds: Optional[int] = Field(
        default=60,
        ge=1,
        description="Maximum seconds before step is flagged as TIMED_OUT",
    )
    min_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum interaction confidence required to satisfy step",
    )
    optional: bool = Field(
        default=False,
        description="Whether this step is optional (can be bypassed)",
    )
    repeatable: bool = Field(
        default=False,
        description="Whether this step can be repeated multiple times",
    )
    max_repeats: int = Field(
        default=1,
        ge=1,
        description="Maximum repetitions allowed if repeatable=True",
    )
    requires_calibration: bool = Field(
        default=False,
        description="If True, rejects uncalibrated camera frames and requires rig metric coordinates",
    )
    target_zone: Optional[TargetZoneRequirement] = Field(
        default=None,
        description="Metric 3D zone in rig coordinate space required for placement",
    )
    target_zone_name: Optional[str] = Field(
        default=None,
        description="Reference to a named target zone in ProtocolDefinition",
    )
    completion_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional geometric or state criteria",
    )
    severity: Literal["info", "warning", "critical"] = Field(
        default="warning",
        description="Alert severity if violated or timed out",
    )
    voice_message: str = Field(
        default="",
        description="Speech prompt synthesized for alerts or guidance",
    )

    model_config = ConfigDict(populate_by_name=True)

    @property
    def instruction(self) -> str:
        """Backward-compatible alias for human_instruction."""
        return self.human_instruction

    @property
    def stable_duration_ms(self) -> int:
        """Backward-compatible alias for minimum_evidence_duration_ms."""
        return self.minimum_evidence_duration_ms

    @model_validator(mode="after")
    def validate_title_and_instruction(self) -> ProtocolStep:
        """Ensure title has a default fallback."""
        if not self.title:
            self.title = f"Step {self.order}: {self.step_id}"
        if not self.human_instruction:
            self.human_instruction = self.title
        if not self.voice_message:
            self.voice_message = self.human_instruction
        return self

    def matches_event_type(self, event_type_str: str) -> bool:
        """Deterministic check if observed event type satisfies step expectation."""
        if self.expected_event.upper() == event_type_str.upper():
            return True
        if self.expected_events:
            return any(e.upper() == event_type_str.upper() for e in self.expected_events)
        # Compatibility aliases
        aliases = {
            "OBJECT_PICKED_UP": ["PICKUP_CANDIDATE"],
            "OBJECT_PLACED_IN_TARGET": ["PLACEMENT_CANDIDATE"],
            "OBJECT_RELEASED": ["RELEASE_CANDIDATE"],
            "OBJECT_MOVED": ["OBJECT_MOVEMENT"],
        }
        if self.expected_event.upper() in aliases:
            if event_type_str.upper() in aliases[self.expected_event.upper()]:
                return True
        return False


class ProtocolDefinition(BaseModel):
    """Complete versioned JSON protocol definition."""
    protocol_id: str = Field(description="Unique procedure slug")
    protocol_version: str = Field(
        default="1.0.0",
        alias="version",
        description="Semantic version string",
    )
    experiment_name: str = Field(
        default="",
        alias="protocol_name",
        description="Full experiment title",
    )
    description: Optional[str] = Field(default=None, description="Detailed protocol purpose")
    strict_order: bool = Field(default=True, description="Strict linear sequence enforcement")
    alert_cooldown_seconds: float = Field(default=3.0, ge=0.5, description="Alert throttle interval")
    target_zones: Dict[str, TargetZoneRequirement] = Field(
        default_factory=dict,
        description="Map of named physical target zones on the experiment rig",
    )
    steps: List[ProtocolStep] = Field(default_factory=list, description="Ordered step sequence")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_protocol_integrity(self) -> ProtocolDefinition:
        """Validate step continuity, ID uniqueness, and valid prerequisite references."""
        if not self.experiment_name:
            self.experiment_name = self.protocol_id

        if not self.steps:
            return self

        seen_ids = set()
        seen_orders = set()

        for step in self.steps:
            if step.step_id in seen_ids:
                raise ValueError(f"Duplicate step_id '{step.step_id}' found in protocol.")
            seen_ids.add(step.step_id)

            if step.order in seen_orders:
                raise ValueError(f"Duplicate order index '{step.order}' found in protocol.")
            seen_orders.add(step.order)

        # Check contiguous orders: 1, 2, ..., N
        sorted_orders = sorted(seen_orders)
        expected_orders = list(range(1, len(self.steps) + 1))
        if sorted_orders != expected_orders:
            raise ValueError(
                f"Step order must be contiguous starting from 1. Got {sorted_orders}, expected {expected_orders}."
            )

        # Check preconditions refer to valid previous steps
        for step in self.steps:
            for pre in step.preconditions:
                if pre not in seen_ids:
                    raise ValueError(f"Step '{step.step_id}' references unknown precondition '{pre}'.")
                # In strict mode, preconditions must be earlier steps
                pre_step = next(s for s in self.steps if s.step_id == pre)
                if pre_step.order >= step.order:
                    raise ValueError(
                        f"Precondition '{pre}' (order {pre_step.order}) cannot be >= step '{step.step_id}' (order {step.order})."
                    )

        # Resolve named target zones
        for step in self.steps:
            if step.target_zone_name and not step.target_zone:
                if step.target_zone_name in self.target_zones:
                    step.target_zone = self.target_zones[step.target_zone_name]

        return self


class ProcedureViolation(BaseModel):
    """Structured report of a deterministic compliance violation."""
    violation_id: str = Field(description="Unique UUID for this violation event")
    violation_type: ViolationType = Field(description="Category of violation")
    step_id: Optional[str] = Field(default=None, description="Step ID violated or expected")
    expected_action: str = Field(description="What the protocol required")
    observed_action: str = Field(description="What was actually observed")
    event_id: Optional[str] = Field(default=None, description="Related interaction event ID")
    object_track_id: Optional[str] = Field(default=None, description="Related tracked object ID")
    object_class: Optional[str] = Field(default=None, description="Related object class")
    explanation: str = Field(description="Deterministic explainable reason")
    timestamp_s: float = Field(description="System timestamp when detected")
    frame_id: int = Field(default=0, description="Video frame ID")
    severity: Literal["info", "warning", "critical"] = Field(default="warning")
    voice_message: Optional[str] = Field(default=None, description="Guidance message for operator")


class StepProgress(BaseModel):
    """Runtime tracking status for a single procedure step."""
    step_id: str
    order: int
    title: str
    state: StepState = StepState.NOT_STARTED
    start_time_s: Optional[float] = None
    completed_time_s: Optional[float] = None
    elapsed_time_s: float = 0.0
    evidence_duration_ms: float = 0.0
    consecutive_frames: int = 0
    current_event_id: Optional[str] = None
    violations_count: int = 0
    failure_reason: Optional[str] = None


class ProcedureEvaluationResult(BaseModel):
    """Typed deterministic evaluation output returned by the ProcedureEngine."""
    protocol_id: str
    protocol_state: ProtocolLifecycleState
    current_step_id: Optional[str] = None
    current_step_index: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    remaining_steps: int = 0
    progress_pct: int = 0
    last_evaluated_event_id: Optional[str] = None
    new_violations: List[ProcedureViolation] = Field(default_factory=list)
    all_violations: List[ProcedureViolation] = Field(default_factory=list)
    step_states: Dict[str, StepProgress] = Field(default_factory=dict)
    is_success: bool = False
    message: str = ""

    @property
    def status(self) -> ProtocolLifecycleState:
        """Compatibility property for protocol lifecycle state."""
        return self.protocol_state
