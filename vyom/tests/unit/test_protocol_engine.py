"""Focused verification of JSON-driven deterministic procedure compliance engine.

Tests:
1. Valid JSON protocol loading and integrity validation.
2. Invalid protocol rejection (duplicate IDs, order gaps, invalid prerequisites, bad JSON).
3. Correct sequential execution (Step 1 -> 2 -> 3 -> 4) to Protocol COMPLETED.
4. Out-of-order action detection and state preservation.
5. Missing prerequisite handling.
6. Uncertain observation handling (approach/contact alone, low confidence, wrong object).
7. Duplicate event protection (idempotency).
8. Calibration-dependent step validation (uncalibrated rejection, target zone boundaries).
9. Timeout detection.
10. Alert cooldown throttling.
"""

from pathlib import Path
import pytest

from vyom.core.config import find_project_root
from vyom.interaction.events import InteractionEvent, InteractionEventType
from vyom.protocol import (
    ProcedureEngine,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolStep,
    ProtocolValidationError,
    StepState,
    TargetZoneRequirement,
    ViolationType,
    load_protocol_from_dict,
    load_protocol_from_file,
    load_protocol_from_json,
)


def _make_synthetic_event(
    event_id: str,
    event_type: InteractionEventType,
    object_class: str = "centrifuge_tube",
    confidence: float = 0.85,
    duration_ms: float = 600.0,
    timestamp_s: float = 100.0,
    frame_id: int = 1,
    calibration_status: bool = False,
    rig_position: tuple | None = None,
) -> InteractionEvent:
    """Helper to produce controlled synthetic interaction events for deterministic testing."""
    return InteractionEvent(
        event_id=event_id,
        event_type=event_type,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        hand_id="hand_0",
        handedness="Right",
        object_track_id=f"{object_class}_track_1",
        object_class=object_class,
        detection_confidence=0.90,
        interaction_confidence=confidence,
        supporting_evidence={
            "duration_ms": duration_ms,
            "consecutive_frames": 5,
            "displacement_px": 30.0,
        },
        coordinate_system="RIG_METRIC" if calibration_status else "IMAGE_PIXEL",
        calibration_status=calibration_status,
        rig_position=rig_position,
        spatial_uncertainty="CALIBRATED" if calibration_status else "UNCALIBRATED_2D",
    )


# ==============================================================================
# Verification 1 & 2: Protocol Loading and Rejection
# ==============================================================================

def test_valid_json_protocol_loading():
    """Verify loading and parsing the demonstration protocol."""
    root = find_project_root()
    proto_path = root / "config" / "protocols" / "sample_experiment.json"
    proto = load_protocol_from_file(proto_path)

    assert proto.protocol_id == "sample-specimen-transfer"
    assert proto.protocol_version == "1.0.0"
    assert len(proto.steps) == 4
    assert proto.steps[0].step_id == "STEP_01"
    assert proto.steps[0].expected_event == "PICKUP_CANDIDATE"
    assert proto.steps[0].instruction == "Carefully pick up the centrifuge tube from the specimen stand."
    assert proto.steps[0].stable_duration_ms == 400


def test_invalid_protocol_rejection():
    """Verify strict rejection of invalid or malformed protocol configurations."""
    # 1. Duplicate step_id
    with pytest.raises(ProtocolValidationError, match="Duplicate step_id"):
        load_protocol_from_dict({
            "protocol_id": "test_dup",
            "experiment_name": "Test",
            "steps": [
                {"step_id": "STEP_01", "order": 1, "title": "S1", "expected_event": "PICKUP_CANDIDATE"},
                {"step_id": "STEP_01", "order": 2, "title": "S2", "expected_event": "RELEASE_CANDIDATE"},
            ],
        })

    # 2. Non-contiguous order
    with pytest.raises(ProtocolValidationError, match="contiguous starting from 1"):
        load_protocol_from_dict({
            "protocol_id": "test_gap",
            "experiment_name": "Test",
            "steps": [
                {"step_id": "STEP_01", "order": 1, "title": "S1", "expected_event": "PICKUP_CANDIDATE"},
                {"step_id": "STEP_02", "order": 3, "title": "S2", "expected_event": "RELEASE_CANDIDATE"},
            ],
        })

    # 3. Unknown precondition
    with pytest.raises(ProtocolValidationError, match="unknown precondition"):
        load_protocol_from_dict({
            "protocol_id": "test_pre",
            "experiment_name": "Test",
            "steps": [
                {"step_id": "STEP_01", "order": 1, "title": "S1", "expected_event": "PICKUP_CANDIDATE", "preconditions": ["STEP_NONEXISTENT"]},
            ],
        })

    # 4. Malformed JSON string
    with pytest.raises(ProtocolValidationError, match="Malformed JSON syntax"):
        load_protocol_from_json("{ invalid json content }")


# ==============================================================================
# Verification 3: Correct Sequential Execution
# ==============================================================================

def test_correct_sequential_execution():
    """Verify 4-step sequence (Pick up -> Move -> Place -> Release) completes procedure."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)

    res = engine.start(timestamp_s=10.0)
    assert res.protocol_state == ProtocolLifecycleState.ACTIVE
    assert res.current_step_id == "STEP_01"
    assert res.progress_pct == 0

    # Step 1: PICKUP_CANDIDATE on centrifuge_tube
    ev1 = _make_synthetic_event("ev_1", InteractionEventType.PICKUP_CANDIDATE, "centrifuge_tube", duration_ms=500, timestamp_s=11.0)
    res = engine.evaluate_event(ev1)
    assert res.step_states["STEP_01"].state == StepState.COMPLETED
    assert res.current_step_id == "STEP_02"
    assert res.progress_pct == 25

    # Step 2: OBJECT_MOVEMENT on centrifuge_tube
    ev2 = _make_synthetic_event("ev_2", InteractionEventType.OBJECT_MOVEMENT, "centrifuge_tube", duration_ms=400, timestamp_s=12.0)
    res = engine.evaluate_event(ev2)
    assert res.step_states["STEP_02"].state == StepState.COMPLETED
    assert res.current_step_id == "STEP_03"
    assert res.progress_pct == 50

    # Step 3: PLACEMENT_CANDIDATE in calibrated target zone
    ev3 = _make_synthetic_event(
        "ev_3",
        InteractionEventType.PLACEMENT_CANDIDATE,
        "centrifuge_tube",
        duration_ms=600,
        timestamp_s=13.0,
        calibration_status=True,
        rig_position=(0.02, 0.01, 0.08),  # Inside rack_zone_a bounds
    )
    res = engine.evaluate_event(ev3)
    assert res.step_states["STEP_03"].state == StepState.COMPLETED
    assert res.current_step_id == "STEP_04"
    assert res.progress_pct == 75

    # Step 4: RELEASE_CANDIDATE on centrifuge_tube
    ev4 = _make_synthetic_event("ev_4", InteractionEventType.RELEASE_CANDIDATE, "centrifuge_tube", duration_ms=400, timestamp_s=14.0)
    res = engine.evaluate_event(ev4)
    assert res.step_states["STEP_04"].state == StepState.COMPLETED
    assert res.protocol_state == ProtocolLifecycleState.COMPLETED
    assert res.is_success is True
    assert res.progress_pct == 100
    assert res.completed_steps == 4


# ==============================================================================
# Verification 4: Out-of-Order Action Detection
# ==============================================================================

def test_out_of_order_detection():
    """Verify that attempting Step 4 before Step 1 is completed flags OUT_OF_ORDER and preserves state."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    assert engine.current_step_id == "STEP_01"

    # Operator prematurely attempts Step 4 (RELEASE_CANDIDATE)
    ev_ooo = _make_synthetic_event(
        "ev_premature",
        InteractionEventType.RELEASE_CANDIDATE,
        "centrifuge_tube",
        duration_ms=400,
        timestamp_s=11.0,
    )
    res = engine.evaluate_event(ev_ooo)

    # Must preserve current procedure state at STEP_01
    assert engine.current_step_id == "STEP_01"
    assert res.step_states["STEP_01"].state == StepState.IN_PROGRESS
    assert res.step_states["STEP_04"].state == StepState.NOT_STARTED
    assert len(res.new_violations) == 1
    assert res.new_violations[0].violation_type == ViolationType.OUT_OF_ORDER
    assert res.new_violations[0].step_id == "STEP_01"
    assert "Step 4" in res.new_violations[0].observed_action
    assert "Attempted Step 4" in res.new_violations[0].explanation


# ==============================================================================
# Verification 5: Inviolable Rules (Approach/Contact never completes step)
# ==============================================================================

def test_inviolable_approach_and_contact_rules():
    """Verify HAND_APPROACH and CONTACT_CANDIDATE alone NEVER mark a step COMPLETED."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    # 1. HAND_APPROACH
    ev_app = _make_synthetic_event("ev_app", InteractionEventType.HAND_APPROACH, "centrifuge_tube", duration_ms=2000, timestamp_s=11.0)
    res = engine.evaluate_event(ev_app)
    assert res.step_states["STEP_01"].state == StepState.IN_PROGRESS
    assert engine.current_step_id == "STEP_01"

    # 2. CONTACT_CANDIDATE
    ev_con = _make_synthetic_event("ev_con", InteractionEventType.CONTACT_CANDIDATE, "centrifuge_tube", duration_ms=2000, timestamp_s=12.0)
    res = engine.evaluate_event(ev_con)
    assert res.step_states["STEP_01"].state == StepState.IN_PROGRESS
    assert engine.current_step_id == "STEP_01"


def test_invalid_object_and_low_confidence_rejection():
    """Verify wrong object class and low confidence are rejected with violations."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    # 1. Wrong object: beaker instead of centrifuge_tube
    ev_wrong_obj = _make_synthetic_event(
        "ev_wrong",
        InteractionEventType.PICKUP_CANDIDATE,
        object_class="beaker",
        timestamp_s=11.0,
    )
    res = engine.evaluate_event(ev_wrong_obj)
    assert engine.current_step_id == "STEP_01"
    assert any(v.violation_type == ViolationType.INVALID_OBJECT for v in res.new_violations)

    # 2. Low confidence: 0.40 < min_confidence (0.75)
    ev_low_conf = _make_synthetic_event(
        "ev_low",
        InteractionEventType.PICKUP_CANDIDATE,
        object_class="centrifuge_tube",
        confidence=0.40,
        timestamp_s=12.0,
    )
    res = engine.evaluate_event(ev_low_conf)
    assert engine.current_step_id == "STEP_01"
    assert any(v.violation_type == ViolationType.UNCERTAIN_OBSERVATION for v in res.new_violations)


# ==============================================================================
# Verification 6: Calibration and Target Zone Requirements
# ==============================================================================

def test_calibration_and_target_zone_enforcement():
    """Verify metric comparison rejection for uncalibrated frames and target zone bounding."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    # Advance to Step 3 (Place tube in tube rack, requires calibrated rack_zone_a)
    engine.evaluate_event(_make_synthetic_event("e1", InteractionEventType.PICKUP_CANDIDATE, "centrifuge_tube", duration_ms=500, timestamp_s=11.0))
    engine.evaluate_event(_make_synthetic_event("e2", InteractionEventType.OBJECT_MOVEMENT, "centrifuge_tube", duration_ms=400, timestamp_s=12.0))
    assert engine.current_step_id == "STEP_03"

    # Attempt Step 3 WITHOUT calibration
    ev_uncalib = _make_synthetic_event(
        "ev_uncalib",
        InteractionEventType.PLACEMENT_CANDIDATE,
        "centrifuge_tube",
        calibration_status=False,
        rig_position=None,
        timestamp_s=13.0,
    )
    res = engine.evaluate_event(ev_uncalib)
    assert engine.current_step_id == "STEP_03"
    assert any(v.violation_type == ViolationType.CALIBRATION_REQUIRED for v in res.new_violations)

    # Attempt Step 3 WITH calibration but OUTSIDE target zone
    ev_out_zone = _make_synthetic_event(
        "ev_out_zone",
        InteractionEventType.PLACEMENT_CANDIDATE,
        "centrifuge_tube",
        calibration_status=True,
        rig_position=(0.90, 0.90, 0.90),  # Way outside target zone bounds
        timestamp_s=14.0,
    )
    res = engine.evaluate_event(ev_out_zone)
    assert engine.current_step_id == "STEP_03"
    assert any(v.violation_type == ViolationType.INVALID_OBJECT for v in res.new_violations)


# ==============================================================================
# Verification 7: Duplicate Event Protection (Idempotency)
# ==============================================================================

def test_duplicate_event_idempotency():
    """Verify duplicate event IDs do not cause steps to advance twice or corrupt state."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    ev = _make_synthetic_event("unique_ev_101", InteractionEventType.PICKUP_CANDIDATE, "centrifuge_tube", duration_ms=500)

    # First emission: satisfies Step 1 and advances to Step 2
    res1 = engine.evaluate_event(ev)
    assert res1.current_step_id == "STEP_02"
    assert res1.completed_steps == 1

    # Second emission: same event ID replayed
    res2 = engine.evaluate_event(ev)
    # Must remain on Step 2 without advancing or counting double
    assert res2.current_step_id == "STEP_02"
    assert res2.completed_steps == 1
    assert "Duplicate event" in res2.message


# ==============================================================================
# Verification 8: Session Controls & Manual Skip
# ==============================================================================

def test_session_controls_and_manual_skip():
    """Verify pause, resume, reset, and manual skip_step lifecycle controls."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=10.0)

    # Pause
    res = engine.pause()
    assert res.protocol_state == ProtocolLifecycleState.PAUSED
    assert res.step_states["STEP_01"].state == StepState.PAUSED

    # Resume
    res = engine.resume()
    assert res.protocol_state == ProtocolLifecycleState.ACTIVE
    assert res.step_states["STEP_01"].state == StepState.IN_PROGRESS

    # Operator-directed manual skip
    res = engine.skip_step("STEP_01")
    assert res.step_states["STEP_01"].state == StepState.SKIPPED
    assert res.current_step_id == "STEP_02"

    # Reset
    res = engine.reset()
    assert res.protocol_state == ProtocolLifecycleState.NOT_STARTED
    assert res.current_step_id is None
    assert res.completed_steps == 0


# ==============================================================================
# Verification 9: Timeout Detection
# ==============================================================================

def test_step_timeout_detection():
    """Verify that exceeding step timeout marks state TIMED_OUT and records a TIMEOUT violation."""
    proto_data = {
        "protocol_id": "test_timeout_proto",
        "experiment_name": "Timeout Test",
        "steps": [
            {
                "step_id": "STEP_01",
                "order": 1,
                "title": "Quick Step",
                "expected_event": "PICKUP_CANDIDATE",
                "required_objects": ["centrifuge_tube"],
                "timeout_seconds": 5,
            }
        ],
    }
    proto = load_protocol_from_dict(proto_data)
    engine = ProcedureEngine(proto)
    engine.start(timestamp_s=100.0)

    # Candidate event arrives after 10 seconds (exceeds 5s timeout)
    ev_late = _make_synthetic_event(
        "ev_late",
        InteractionEventType.HAND_APPROACH,
        "centrifuge_tube",
        timestamp_s=110.0,
    )
    res = engine.evaluate_event(ev_late)

    assert res.step_states["STEP_01"].state == StepState.TIMED_OUT
    assert any(v.violation_type == ViolationType.TIMEOUT for v in res.all_violations)


# ==============================================================================
# Verification 10: Alert Cooldown Throttling
# ==============================================================================

def test_alert_cooldown_throttling():
    """Verify that repeated violations within cooldown period are throttled."""
    root = find_project_root()
    proto = load_protocol_from_file(root / "config" / "protocols" / "sample_experiment.json")
    engine = ProcedureEngine(proto, alert_cooldown_seconds=3.0)
    engine.start(timestamp_s=10.0)

    # Send 5 invalid object events in rapid succession (within 1 second)
    for i in range(5):
        ev = _make_synthetic_event(
            f"ev_spam_{i}",
            InteractionEventType.PICKUP_CANDIDATE,
            object_class="beaker",  # Invalid object for STEP_01
            timestamp_s=10.0 + (i * 0.1),
        )
        engine.evaluate_event(ev)

    # Even though 5 events occurred, cooldown limits to 1 violation
    assert len(engine.all_violations) == 1

    # Send another event after cooldown expires (at t=14.0s > 10.0s + 3.0s)
    ev_after = _make_synthetic_event(
        "ev_after_cooldown",
        InteractionEventType.PICKUP_CANDIDATE,
        object_class="beaker",
        timestamp_s=14.0,
    )
    res = engine.evaluate_event(ev_after)
    assert len(engine.all_violations) == 2
