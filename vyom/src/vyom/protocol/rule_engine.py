"""Deterministic, explainable rule evaluation engine for procedure steps.

Evaluates structured interaction events against explicit protocol step criteria
without any black-box or non-deterministic logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vyom.core.logging_setup import get_logger
from vyom.interaction.events import InteractionEvent, InteractionEventType
from vyom.protocol.protocol_schema import (
    ProcedureViolation,
    ProtocolDefinition,
    ProtocolStep,
    StepState,
    ViolationType,
)

logger = get_logger("protocol.rule_engine")


@dataclass
class RuleEvaluationOutcome:
    """Outcome of evaluating an InteractionEvent against a ProtocolStep."""
    is_satisfied: bool
    is_candidate_match: bool  # Matches step's object and event type, but evidence or zone pending
    is_out_of_order: bool = False
    violation: Optional[ProcedureViolation] = None
    accumulated_duration_ms: float = 0.0
    consecutive_frames: int = 0
    explanation: str = ""


class ProcedureRuleEngine:
    """Deterministic, explainable rule engine verifying compliance against protocol steps."""

    def __init__(self) -> None:
        pass

    def evaluate_step(
        self,
        step: ProtocolStep,
        event: InteractionEvent,
        step_progress_frames: int,
        step_accumulated_ms: float,
        step_preconditions_met: bool,
    ) -> RuleEvaluationOutcome:
        """Evaluate an InteractionEvent against an active ProtocolStep.

        Inviolable Rules:
        1. HAND_APPROACH and CONTACT_CANDIDATE alone must NEVER complete a procedure step.
        2. Object class must match step.required_objects.
        3. Confidence must meet step.min_confidence.
        4. Rig calibration required if step specifies requires_calibration or target_zone.
        5. Temporal evidence must satisfy minimum_evidence_duration_ms and min_consecutive_frames.
        6. Preconditions must be completed.
        """
        # 1. Preconditions check
        if not step_preconditions_met:
            violation = ProcedureViolation(
                violation_id=f"viol_pre_{event.frame_id}_{step.step_id}",
                violation_type=ViolationType.MISSING_PREREQUISITE,
                step_id=step.step_id,
                expected_action=f"Prerequisites {step.preconditions} must be completed before {step.step_id}",
                observed_action=f"Attempted {step.step_id} with event {event.event_type.value}",
                event_id=event.event_id,
                object_track_id=event.object_track_id,
                object_class=event.object_class,
                explanation=f"Missing prerequisites for step {step.step_id}: {step.preconditions}",
                timestamp_s=event.timestamp_s,
                frame_id=event.frame_id,
                severity=step.severity,
                voice_message=f"Please complete earlier steps before {step.title}.",
            )
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=False,
                violation=violation,
                explanation=f"Step {step.step_id} blocked: missing prerequisites {step.preconditions}",
            )

        # 2. Inviolable Rule: HAND_APPROACH never completes any step
        if event.event_type == InteractionEventType.HAND_APPROACH:
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=False,
                explanation="HAND_APPROACH observation recorded; does not constitute step completion.",
            )

        # 3. Inviolable Rule: CONTACT_CANDIDATE alone must NEVER complete a procedure step
        if event.event_type == InteractionEventType.CONTACT_CANDIDATE:
            # Contact can qualify progress/approach, but does not complete the step
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=False,
                accumulated_duration_ms=step_accumulated_ms,
                consecutive_frames=step_progress_frames,
                explanation="CONTACT_CANDIDATE observed; touch alone does not fulfill procedural step execution.",
            )

        # 4. Check Event Type match
        if not step.matches_event_type(event.event_type.value):
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=False,
                explanation=f"Event type '{event.event_type.value}' does not match expected '{step.expected_event}'.",
            )

        # 5. Check Object Class match
        if step.required_objects:
            if event.object_class not in step.required_objects:
                violation = ProcedureViolation(
                    violation_id=f"viol_obj_{event.frame_id}_{step.step_id}",
                    violation_type=ViolationType.INVALID_OBJECT,
                    step_id=step.step_id,
                    expected_action=f"Expected interaction with {step.required_objects}",
                    observed_action=f"Interacted with {event.object_class} (track: {event.object_track_id})",
                    event_id=event.event_id,
                    object_track_id=event.object_track_id,
                    object_class=event.object_class,
                    explanation=f"Invalid object class '{event.object_class}'. Step {step.step_id} requires: {step.required_objects}",
                    timestamp_s=event.timestamp_s,
                    frame_id=event.frame_id,
                    severity=step.severity,
                    voice_message=f"Wrong object. Please use {', '.join(step.required_objects)}.",
                )
                return RuleEvaluationOutcome(
                    is_satisfied=False,
                    is_candidate_match=False,
                    violation=violation,
                    explanation=f"Invalid object '{event.object_class}' for step {step.step_id}",
                )

        # 6. Check Confidence Threshold
        if event.interaction_confidence < step.min_confidence:
            violation = ProcedureViolation(
                violation_id=f"viol_conf_{event.frame_id}_{step.step_id}",
                violation_type=ViolationType.UNCERTAIN_OBSERVATION,
                step_id=step.step_id,
                expected_action=f"Interaction confidence >= {step.min_confidence:.0%}",
                observed_action=f"Observed confidence {event.interaction_confidence:.0%}",
                event_id=event.event_id,
                object_track_id=event.object_track_id,
                object_class=event.object_class,
                explanation=f"Interaction confidence {event.interaction_confidence:.2f} below required threshold {step.min_confidence:.2f}",
                timestamp_s=event.timestamp_s,
                frame_id=event.frame_id,
                severity="info",
                voice_message=f"Action observation uncertain. Please perform {step.title} clearly.",
            )
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=True,
                violation=violation,
                explanation=f"Interaction confidence {event.interaction_confidence:.2f} < {step.min_confidence:.2f}",
            )

        # 7. Check Rig Coordinate Calibration and Target Zone
        if step.requires_calibration or step.target_zone is not None:
            if not event.calibration_status or event.rig_position is None:
                violation = ProcedureViolation(
                    violation_id=f"viol_calib_{event.frame_id}_{step.step_id}",
                    violation_type=ViolationType.CALIBRATION_REQUIRED,
                    step_id=step.step_id,
                    expected_action="Calibrated 3D rig metric coordinates",
                    observed_action=f"Uncalibrated 2D observation (status: {event.calibration_status})",
                    event_id=event.event_id,
                    object_track_id=event.object_track_id,
                    object_class=event.object_class,
                    explanation=(
                        f"Step {step.step_id} requires calibrated rig coordinates. "
                        "Rejecting uncalibrated metric comparison. MediaPipe hand-relative coordinates cannot substitute."
                    ),
                    timestamp_s=event.timestamp_s,
                    frame_id=event.frame_id,
                    severity="warning",
                    voice_message="Rig calibration required for target zone placement.",
                )
                return RuleEvaluationOutcome(
                    is_satisfied=False,
                    is_candidate_match=True,
                    violation=violation,
                    explanation=f"Step {step.step_id} requires rig calibration.",
                )

            # Calibrated: verify 3D Target Zone
            if step.target_zone is not None:
                if not step.target_zone.contains_rig_point(event.rig_position):
                    pos_str = f"({event.rig_position[0]:.3f}, {event.rig_position[1]:.3f}, {event.rig_position[2]:.3f})m"
                    violation = ProcedureViolation(
                        violation_id=f"viol_zone_{event.frame_id}_{step.step_id}",
                        violation_type=ViolationType.INVALID_OBJECT,
                        step_id=step.step_id,
                        expected_action=f"Action within target zone '{step.target_zone.zone_id}'",
                        observed_action=f"Rig position {pos_str} outside target zone bounds",
                        event_id=event.event_id,
                        object_track_id=event.object_track_id,
                        object_class=event.object_class,
                        explanation=f"Position {pos_str} outside target zone '{step.target_zone.zone_id}'.",
                        timestamp_s=event.timestamp_s,
                        frame_id=event.frame_id,
                        severity=step.severity,
                        voice_message=f"Please place in target zone {step.target_zone.name or step.target_zone.zone_id}.",
                    )
                    return RuleEvaluationOutcome(
                        is_satisfied=False,
                        is_candidate_match=True,
                        violation=violation,
                        explanation=f"Object position {pos_str} outside target zone.",
                    )

        # 8. Check Temporal Evidence Duration and Persistence
        event_duration_ms = float(event.supporting_evidence.get("duration_ms", 0.0))
        event_frames = int(event.supporting_evidence.get("consecutive_frames", 1))

        # Effective duration is max of event evidence or caller tracking
        total_duration_ms = max(event_duration_ms, step_accumulated_ms)
        total_frames = max(event_frames, step_progress_frames)

        if total_duration_ms < step.minimum_evidence_duration_ms or total_frames < step.min_consecutive_frames:
            return RuleEvaluationOutcome(
                is_satisfied=False,
                is_candidate_match=True,
                accumulated_duration_ms=total_duration_ms,
                consecutive_frames=total_frames,
                explanation=(
                    f"Candidate matched for {step.step_id}, but temporal evidence accumulating "
                    f"({total_duration_ms:.0f}ms/{step.minimum_evidence_duration_ms}ms, "
                    f"{total_frames}/{step.min_consecutive_frames} frames)."
                ),
            )

        # All criteria satisfied!
        return RuleEvaluationOutcome(
            is_satisfied=True,
            is_candidate_match=True,
            accumulated_duration_ms=total_duration_ms,
            consecutive_frames=total_frames,
            explanation=f"Step {step.step_id} criteria fully satisfied by {event.event_type.value}.",
        )
