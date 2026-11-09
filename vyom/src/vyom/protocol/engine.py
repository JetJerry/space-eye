"""Deterministic, JSON-driven procedure validation, step progression, and compliance engine.

Implements strict sequence validation, out-of-order detection, prerequisite checking,
temporal evidence thresholds, calibration validation, and session lifecycle management.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

from vyom.core.logging_setup import get_logger
from vyom.interaction.events import InteractionEvent, InteractionEventType
from vyom.protocol.alerts import AlertCooldownTracker, ProtocolAlert
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProcedureViolation,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolStep,
    StepProgress,
    StepState,
    ViolationType,
)
from vyom.protocol.rule_engine import ProcedureRuleEngine

logger = get_logger("protocol.engine")


class ProcedureEngine:
    """Deterministic, explainable procedure validation and compliance state machine.

    Evaluates live InteractionEvents against an active ProtocolDefinition.
    """

    def __init__(
        self,
        protocol: Optional[ProtocolDefinition] = None,
        alert_cooldown_seconds: Optional[float] = None,
    ) -> None:
        self.protocol: Optional[ProtocolDefinition] = protocol
        self.protocol_state: ProtocolLifecycleState = ProtocolLifecycleState.NOT_STARTED

        cooldown = (
            alert_cooldown_seconds
            if alert_cooldown_seconds is not None
            else (protocol.alert_cooldown_seconds if protocol else 3.0)
        )
        self._cooldown_tracker = AlertCooldownTracker(default_cooldown_seconds=cooldown)
        self._rule_engine = ProcedureRuleEngine()

        self.current_step_index: int = 0
        self.step_states: Dict[str, StepProgress] = {}
        self.all_violations: List[ProcedureViolation] = []
        self._processed_event_ids: Set[str] = set()

        # Step runtime state
        self._step_start_time_s: float = 0.0
        self._step_accumulated_evidence_ms: float = 0.0
        self._step_consecutive_frames: int = 0
        self._active_track_id: Optional[str] = None
        self._last_event_time_s: float = 0.0

        if self.protocol:
            self._initialize_step_states()

    def set_protocol(self, protocol: ProtocolDefinition) -> None:
        """Load or replace the active experiment protocol."""
        self.protocol = protocol
        self._cooldown_tracker.default_cooldown_seconds = protocol.alert_cooldown_seconds
        self.reset()
        logger.info(
            "ProcedureEngine loaded protocol '%s' (v%s) with %d steps",
            protocol.experiment_name,
            protocol.protocol_version,
            len(protocol.steps),
        )

    def _initialize_step_states(self) -> None:
        """Initialize all steps to StepState.NOT_STARTED."""
        self.step_states.clear()
        if not self.protocol:
            return
        for step in self.protocol.steps:
            self.step_states[step.step_id] = StepProgress(
                step_id=step.step_id,
                order=step.order,
                title=step.title,
                state=StepState.NOT_STARTED,
            )

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def start(self, timestamp_s: Optional[float] = None) -> ProcedureEvaluationResult:
        """Start the experiment procedure monitoring session."""
        if not self.protocol or not self.protocol.steps:
            logger.warning("Cannot start ProcedureEngine: no protocol steps loaded.")
            return self._build_result(message="No protocol loaded")

        t = timestamp_s if timestamp_s is not None else time.time()
        self.protocol_state = ProtocolLifecycleState.ACTIVE
        self.current_step_index = 0
        self._step_start_time_s = t
        self._step_accumulated_evidence_ms = 0.0
        self._step_consecutive_frames = 0
        self._active_track_id = None

        first_step = self.protocol.steps[0]
        sp = self.step_states[first_step.step_id]
        sp.state = StepState.IN_PROGRESS
        sp.start_time_s = t

        logger.info(
            "ProcedureEngine started: Protocol '%s', Active step: %s",
            self.protocol.experiment_name,
            first_step.step_id,
        )
        return self._build_result(message=f"Procedure started. Active: {first_step.step_id}")

    def pause(self) -> ProcedureEvaluationResult:
        """Pause active procedure evaluation."""
        if self.protocol_state == ProtocolLifecycleState.ACTIVE:
            self.protocol_state = ProtocolLifecycleState.PAUSED
            if self.current_step:
                self.step_states[self.current_step.step_id].state = StepState.PAUSED
            logger.info("ProcedureEngine paused at step %s", self.current_step_id)
        return self._build_result(message="Procedure paused")

    def resume(self) -> ProcedureEvaluationResult:
        """Resume paused procedure evaluation."""
        if self.protocol_state == ProtocolLifecycleState.PAUSED:
            self.protocol_state = ProtocolLifecycleState.ACTIVE
            if self.current_step:
                self.step_states[self.current_step.step_id].state = StepState.IN_PROGRESS
            logger.info("ProcedureEngine resumed at step %s", self.current_step_id)
        return self._build_result(message="Procedure resumed")

    def stop(self) -> ProcedureEvaluationResult:
        """Stop/abort active procedure evaluation."""
        self.protocol_state = ProtocolLifecycleState.ABORTED
        logger.info("ProcedureEngine aborted at step %s", self.current_step_id)
        return self._build_result(message="Procedure aborted")

    def reset(self) -> ProcedureEvaluationResult:
        """Reset procedure evaluation to pristine initial state."""
        self.protocol_state = ProtocolLifecycleState.NOT_STARTED
        self.current_step_index = 0
        self.all_violations.clear()
        self._processed_event_ids.clear()
        self._cooldown_tracker.reset()
        self._step_start_time_s = 0.0
        self._step_accumulated_evidence_ms = 0.0
        self._step_consecutive_frames = 0
        self._active_track_id = None
        self._initialize_step_states()
        logger.info("ProcedureEngine reset to initial state")
        return self._build_result(message="Procedure reset")

    def skip_step(self, step_id: Optional[str] = None) -> ProcedureEvaluationResult:
        """Manually mark a step as SKIPPED and advance progression.

        Distinguishes operator-directed step bypass from missing visual detections.
        """
        if not self.protocol or self.current_step_index >= len(self.protocol.steps):
            return self._build_result(message="No step to skip")

        step = self.current_step
        if not step:
            return self._build_result(message="No current step")

        if step_id and step.step_id != step_id:
            logger.warning("Requested skip for '%s' but active step is '%s'", step_id, step.step_id)

        sp = self.step_states[step.step_id]
        sp.state = StepState.SKIPPED
        logger.info("Step '%s' (%s) explicitly marked SKIPPED", step.step_id, step.title)

        # Advance to next step
        self.current_step_index += 1
        if self.current_step_index >= len(self.protocol.steps):
            self.protocol_state = ProtocolLifecycleState.COMPLETED
            logger.info("Protocol '%s' COMPLETED via final step skip", self.protocol.protocol_id)
        else:
            next_step = self.protocol.steps[self.current_step_index]
            nsp = self.step_states[next_step.step_id]
            nsp.state = StepState.IN_PROGRESS
            nsp.start_time_s = time.time()
            self._step_start_time_s = nsp.start_time_s
            self._step_accumulated_evidence_ms = 0.0
            self._step_consecutive_frames = 0

        return self._build_result(message=f"Step {step.step_id} skipped")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_step(self) -> Optional[ProtocolStep]:
        if not self.protocol or self.protocol_state == ProtocolLifecycleState.NOT_STARTED:
            return None
        if self.current_step_index >= len(self.protocol.steps):
            return None
        return self.protocol.steps[self.current_step_index]

    @property
    def current_step_id(self) -> Optional[str]:
        step = self.current_step
        return step.step_id if step else None

    # -------------------------------------------------------------------------
    # Event Evaluation
    # -------------------------------------------------------------------------

    def evaluate_event(self, event: InteractionEvent) -> ProcedureEvaluationResult:
        """Evaluate an InteractionEvent against the active protocol.

        Enforces:
        - Duplicate event protection (idempotency)
        - Temporal chronology
        - Active step rule matching and evidence accumulation
        - Inviolable rules: HAND_APPROACH and CONTACT_CANDIDATE never complete a step
        - Out-of-order detection across future steps
        - Timeout detection
        """
        # 1. Duplicate Event Protection (Idempotent processing)
        if event.event_id in self._processed_event_ids:
            logger.debug("Duplicate event '%s' ignored (idempotency protection)", event.event_id)
            return self._build_result(
                last_event_id=event.event_id,
                message=f"Duplicate event {event.event_id} ignored",
            )
        self._processed_event_ids.add(event.event_id)

        # 2. Protocol state check
        if self.protocol_state != ProtocolLifecycleState.ACTIVE:
            return self._build_result(
                last_event_id=event.event_id,
                message=f"Protocol not active (current state: {self.protocol_state.value})",
            )

        step = self.current_step
        if not step:
            return self._build_result(
                last_event_id=event.event_id,
                message="All steps completed",
            )

        new_violations: List[ProcedureViolation] = []
        sp = self.step_states[step.step_id]

        # Update elapsed time
        if sp.start_time_s:
            sp.elapsed_time_s = max(0.0, event.timestamp_s - sp.start_time_s)

        # 3. Timeout check for current step
        if step.timeout_seconds is not None and sp.elapsed_time_s > step.timeout_seconds:
            sp.state = StepState.TIMED_OUT
            timeout_violation = ProcedureViolation(
                violation_id=f"viol_timeout_{event.frame_id}_{step.step_id}",
                violation_type=ViolationType.TIMEOUT,
                step_id=step.step_id,
                expected_action=f"Complete {step.step_id} within {step.timeout_seconds}s",
                observed_action=f"Step elapsed time reached {sp.elapsed_time_s:.1f}s",
                event_id=event.event_id,
                object_track_id=event.object_track_id,
                object_class=event.object_class,
                explanation=f"Step {step.step_id} timed out ({sp.elapsed_time_s:.1f}s > {step.timeout_seconds}s)",
                timestamp_s=event.timestamp_s,
                frame_id=event.frame_id,
                severity="warning",
                voice_message=f"Step {step.order} has exceeded time limit.",
            )
            self._record_violation(timeout_violation, new_violations, sp)

        # 4. Check Preconditions
        preconditions_met = True
        for pre in step.preconditions:
            pre_prog = self.step_states.get(pre)
            if not pre_prog or pre_prog.state != StepState.COMPLETED:
                preconditions_met = False
                break

        # 5. Evaluate event against active step
        outcome = self._rule_engine.evaluate_step(
            step=step,
            event=event,
            step_progress_frames=self._step_consecutive_frames,
            step_accumulated_ms=self._step_accumulated_evidence_ms,
            step_preconditions_met=preconditions_met,
        )

        if outcome.violation:
            self._record_violation(outcome.violation, new_violations, sp)

        # Check if active step is satisfied
        if outcome.is_satisfied:
            # Active step completes!
            sp.state = StepState.COMPLETED
            sp.completed_time_s = event.timestamp_s
            sp.evidence_duration_ms = outcome.accumulated_duration_ms
            sp.consecutive_frames = outcome.consecutive_frames
            sp.current_event_id = event.event_id

            logger.info(
                "COMPLIANCE SUCCESS: Step '%s' (%s) COMPLETED by event '%s'",
                step.step_id,
                step.title,
                event.event_type.value,
            )

            # Advance step index
            self.current_step_index += 1
            if self.current_step_index >= len(self.protocol.steps):
                self.protocol_state = ProtocolLifecycleState.COMPLETED
                logger.info(
                    "PROCEDURE COMPLETED: Protocol '%s' fully completed!",
                    self.protocol.protocol_id,
                )
                return self._build_result(
                    last_event_id=event.event_id,
                    new_violations=new_violations,
                    message=f"Procedure completed: {self.protocol.experiment_name}",
                )
            else:
                next_step = self.protocol.steps[self.current_step_index]
                nsp = self.step_states[next_step.step_id]
                nsp.state = StepState.IN_PROGRESS
                nsp.start_time_s = event.timestamp_s
                self._step_start_time_s = event.timestamp_s
                self._step_accumulated_evidence_ms = 0.0
                self._step_consecutive_frames = 0
                self._active_track_id = None
                return self._build_result(
                    last_event_id=event.event_id,
                    new_violations=new_violations,
                    message=f"Step {step.step_id} completed. Advanced to {next_step.step_id}",
                )

        # Update evidence tracking if candidate match in progress
        if outcome.is_candidate_match and not outcome.violation:
            if self._active_track_id == event.object_track_id:
                dt_ms = (event.timestamp_s - self._last_event_time_s) * 1000.0 if self._last_event_time_s > 0 else 33.3
                self._step_accumulated_evidence_ms += max(0.0, dt_ms)
                self._step_consecutive_frames += 1
            else:
                self._active_track_id = event.object_track_id
                self._step_accumulated_evidence_ms = 33.3
                self._step_consecutive_frames = 1

            self._last_event_time_s = event.timestamp_s
            sp.evidence_duration_ms = self._step_accumulated_evidence_ms
            sp.consecutive_frames = self._step_consecutive_frames
            sp.current_event_id = event.event_id

        # 6. Out-of-Order Execution Check across Future Steps
        # If the event did not match the current step, check if it matches a future step
        if not outcome.is_candidate_match:
            self._check_out_of_order(event, step, new_violations)

        return self._build_result(
            last_event_id=event.event_id,
            new_violations=new_violations,
            message=outcome.explanation,
        )

    def _check_out_of_order(
        self,
        event: InteractionEvent,
        active_step: ProtocolStep,
        new_violations: List[ProcedureViolation],
    ) -> None:
        """Inspect if observed event matches any future protocol step, indicating out-of-order execution."""
        # Never flag out-of-order on mere approaches or departed hands
        if event.event_type in (InteractionEventType.HAND_APPROACH, InteractionEventType.HAND_DEPART):
            return

        for future_step in self.protocol.steps[self.current_step_index + 1 :]:
            # Check if event matches future step's expected event and required objects
            if not future_step.matches_event_type(event.event_type.value):
                continue

            if future_step.required_objects and event.object_class not in future_step.required_objects:
                continue

            # Confirmed out-of-order attempt
            violation = ProcedureViolation(
                violation_id=f"viol_ooo_{event.frame_id}_{future_step.step_id}",
                violation_type=ViolationType.OUT_OF_ORDER,
                step_id=active_step.step_id,
                expected_action=f"Step {active_step.order} [{active_step.step_id}]: {active_step.title} ({active_step.expected_event})",
                observed_action=f"Step {future_step.order} [{future_step.step_id}]: {future_step.title} ({event.event_type.value} on {event.object_class})",
                event_id=event.event_id,
                object_track_id=event.object_track_id,
                object_class=event.object_class,
                explanation=(
                    f"Out-of-order action: Attempted Step {future_step.order} ({future_step.step_id}) "
                    f"before completing Step {active_step.order} ({active_step.step_id})."
                ),
                timestamp_s=event.timestamp_s,
                frame_id=event.frame_id,
                severity=active_step.severity,
                voice_message=f"Out of order action. Please perform Step {active_step.order} ({active_step.title}) first.",
            )

            # Preserve active step as IN_PROGRESS and record violation
            active_sp = self.step_states[active_step.step_id]
            self._record_violation(violation, new_violations, active_sp)

            logger.warning(
                "OUT_OF_ORDER DETECTED: Expected %s, observed attempt of future step %s",
                active_step.step_id,
                future_step.step_id,
            )
            break

    def _record_violation(
        self,
        violation: ProcedureViolation,
        new_violations: List[ProcedureViolation],
        sp: StepProgress,
    ) -> None:
        """Register violation subject to cooldown throttling."""
        if self._cooldown_tracker.should_emit(violation):
            self._cooldown_tracker.record_emission(violation)
            new_violations.append(violation)
            self.all_violations.append(violation)
            sp.violations_count += 1
            if violation.violation_type in (ViolationType.TIMEOUT, ViolationType.PROCEDURE_BLOCKED):
                sp.failure_reason = violation.explanation

    # -------------------------------------------------------------------------
    # Result Compilation
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        last_event_id: Optional[str] = None,
        new_violations: Optional[List[ProcedureViolation]] = None,
        message: str = "",
    ) -> ProcedureEvaluationResult:
        """Assemble structured, typed procedure evaluation result."""
        total = len(self.protocol.steps) if self.protocol else 0
        completed = sum(1 for s in self.step_states.values() if s.state == StepState.COMPLETED)
        remaining = max(0, total - completed)
        pct = int((completed / total) * 100) if total > 0 else 0
        is_success = self.protocol_state == ProtocolLifecycleState.COMPLETED

        return ProcedureEvaluationResult(
            protocol_id=self.protocol.protocol_id if self.protocol else "",
            protocol_state=self.protocol_state,
            current_step_id=self.current_step_id,
            current_step_index=self.current_step_index,
            total_steps=total,
            completed_steps=completed,
            remaining_steps=remaining,
            progress_pct=pct,
            last_evaluated_event_id=last_event_id,
            new_violations=new_violations or [],
            all_violations=list(self.all_violations),
            step_states=dict(self.step_states),
            is_success=is_success,
            message=message,
        )

    def get_state(self) -> ProcedureEvaluationResult:
        """Query current evaluation snapshot without ingesting new events."""
        return self._build_result(message="Current state query")
