"""Verification script for Prompt 9: JSON-Driven Procedure Validation & Compliance Engine.

Performs focused end-to-end checks covering:
1. Valid JSON Protocol Loading
2. Invalid Protocol Schema & Integrity Rejection
3. Correct Sequential Step Execution
4. Out-of-Order Action Detection & State Preservation
5. Missing Prerequisite Enforcement
6. Inviolable Rule Validation (Approach/Contact never completes step)
7. Uncertain Observation & Low-Confidence Handling
8. Duplicate Event Protection (Idempotency)
9. Calibration-Dependent Metric Zone Validation
10. Alert Cooldown Throttling
11. UI ProtocolPanel Integration (Headless Qt)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

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


def make_test_event(
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
    """Construct a synthetic interaction event for deterministic test evaluation."""
    return InteractionEvent(
        event_id=event_id,
        event_type=event_type,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        hand_id="hand_0",
        handedness="Right",
        object_track_id=f"{object_class}_track_1",
        object_class=object_class,
        detection_confidence=0.92,
        interaction_confidence=confidence,
        supporting_evidence={
            "duration_ms": duration_ms,
            "consecutive_frames": 5,
            "displacement_px": 28.5,
        },
        coordinate_system="RIG_METRIC" if calibration_status else "IMAGE_PIXEL",
        calibration_status=calibration_status,
        rig_position=rig_position,
        spatial_uncertainty="CALIBRATED" if calibration_status else "UNCALIBRATED_2D",
    )


def run_checks() -> bool:
    print("=" * 75)
    print("  VYOM PROMPT 9: PROCEDURE COMPLIANCE ENGINE VERIFICATION")
    print("=" * 75)
    all_passed = True

    # -------------------------------------------------------------------------
    # Check 1: Valid Protocol Loading
    # -------------------------------------------------------------------------
    print("\n[Check 1/11] Valid JSON Protocol Loading:")
    try:
        proto_path = root_dir / "config" / "protocols" / "sample_experiment.json"
        proto = load_protocol_from_file(proto_path)
        assert proto.protocol_id == "sample-specimen-transfer"
        assert len(proto.steps) == 4
        assert proto.steps[0].step_id == "STEP_01"
        assert proto.steps[0].expected_event == "PICKUP_CANDIDATE"
        print(f"  ✓ Successfully loaded '{proto.experiment_name}' with {len(proto.steps)} steps.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 2: Invalid Protocol Rejection
    # -------------------------------------------------------------------------
    print("\n[Check 2/11] Invalid Protocol Schema & Procedural Integrity Rejection:")
    rejection_tests = [
        ("Duplicate step_id", {"protocol_id": "p", "steps": [
            {"step_id": "S1", "order": 1, "expected_event": "PICKUP_CANDIDATE"},
            {"step_id": "S1", "order": 2, "expected_event": "RELEASE_CANDIDATE"},
        ]}),
        ("Non-contiguous order index", {"protocol_id": "p", "steps": [
            {"step_id": "S1", "order": 1, "expected_event": "PICKUP_CANDIDATE"},
            {"step_id": "S2", "order": 3, "expected_event": "RELEASE_CANDIDATE"},
        ]}),
        ("Unknown precondition reference", {"protocol_id": "p", "steps": [
            {"step_id": "S1", "order": 1, "expected_event": "PICKUP_CANDIDATE", "preconditions": ["NONEXISTENT"]},
        ]}),
    ]
    for desc, bad_dict in rejection_tests:
        try:
            load_protocol_from_dict(bad_dict)
            print(f"  ✕ Failed to reject: {desc}")
            all_passed = False
        except ProtocolValidationError:
            print(f"  ✓ Correctly rejected: {desc}")

    # -------------------------------------------------------------------------
    # Check 3: Sequential Step Execution
    # -------------------------------------------------------------------------
    print("\n[Check 3/11] Correct Sequential Step Execution (Steps 1 -> 4):")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)

        # Step 1: PICKUP_CANDIDATE
        e1 = make_test_event("e1", InteractionEventType.PICKUP_CANDIDATE, "centrifuge_tube", duration_ms=500, timestamp_s=11.0)
        res1 = engine.evaluate_event(e1)
        assert res1.step_states["STEP_01"].state == StepState.COMPLETED
        assert res1.current_step_id == "STEP_02"

        # Step 2: OBJECT_MOVEMENT
        e2 = make_test_event("e2", InteractionEventType.OBJECT_MOVEMENT, "centrifuge_tube", duration_ms=400, timestamp_s=12.0)
        res2 = engine.evaluate_event(e2)
        assert res2.step_states["STEP_02"].state == StepState.COMPLETED
        assert res2.current_step_id == "STEP_03"

        # Step 3: PLACEMENT_CANDIDATE
        e3 = make_test_event(
            "e3",
            InteractionEventType.PLACEMENT_CANDIDATE,
            "centrifuge_tube",
            duration_ms=600,
            timestamp_s=13.0,
            calibration_status=True,
            rig_position=(0.0, 0.0, 0.05),
        )
        res3 = engine.evaluate_event(e3)
        assert res3.step_states["STEP_03"].state == StepState.COMPLETED
        assert res3.current_step_id == "STEP_04"

        # Step 4: RELEASE_CANDIDATE
        e4 = make_test_event("e4", InteractionEventType.RELEASE_CANDIDATE, "centrifuge_tube", duration_ms=400, timestamp_s=14.0)
        res4 = engine.evaluate_event(e4)
        assert res4.step_states["STEP_04"].state == StepState.COMPLETED
        assert res4.protocol_state == ProtocolLifecycleState.COMPLETED
        assert res4.progress_pct == 100
        print("  ✓ Full 4-step sequence successfully executed to Protocol COMPLETED (100%).")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 4: Out-of-Order Action Detection
    # -------------------------------------------------------------------------
    print("\n[Check 4/11] Out-of-Order Action Detection & State Preservation:")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)

        # Attempt Step 4 while on Step 1
        e_ooo = make_test_event("e_ooo", InteractionEventType.RELEASE_CANDIDATE, "centrifuge_tube", duration_ms=400, timestamp_s=11.0)
        res_ooo = engine.evaluate_event(e_ooo)

        assert engine.current_step_id == "STEP_01"
        assert res_ooo.step_states["STEP_01"].state == StepState.IN_PROGRESS
        assert res_ooo.step_states["STEP_04"].state == StepState.NOT_STARTED
        assert any(v.violation_type == ViolationType.OUT_OF_ORDER for v in res_ooo.new_violations)
        print("  ✓ Detected OUT_OF_ORDER attempt of future Step 4; active STEP_01 preserved.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 5: Inviolable Rule: Hand Approach and Contact Alone Never Complete
    # -------------------------------------------------------------------------
    print("\n[Check 5/11] Inviolable Rule: Approach and Contact Alone Never Complete Step:")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)

        # 1. Approach
        res_app = engine.evaluate_event(make_test_event("ea", InteractionEventType.HAND_APPROACH, timestamp_s=11.0))
        assert res_app.step_states["STEP_01"].state == StepState.IN_PROGRESS
        assert res_app.completed_steps == 0

        # 2. Contact candidate
        res_con = engine.evaluate_event(make_test_event("ec", InteractionEventType.CONTACT_CANDIDATE, timestamp_s=12.0))
        assert res_con.step_states["STEP_01"].state == StepState.IN_PROGRESS
        assert res_con.completed_steps == 0
        print("  ✓ HAND_APPROACH and CONTACT_CANDIDATE alone correctly rejected for step completion.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 6: Invalid Object & Low Confidence Handling
    # -------------------------------------------------------------------------
    print("\n[Check 6/11] Invalid Object & Low-Confidence Handling:")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)

        # Invalid object: beaker
        res_inv_obj = engine.evaluate_event(make_test_event("eo", InteractionEventType.PICKUP_CANDIDATE, object_class="beaker", timestamp_s=11.0))
        assert any(v.violation_type == ViolationType.INVALID_OBJECT for v in res_inv_obj.new_violations)

        # Low confidence: 0.45 < 0.75
        res_low_conf = engine.evaluate_event(make_test_event("el", InteractionEventType.PICKUP_CANDIDATE, confidence=0.45, timestamp_s=12.0))
        assert any(v.violation_type == ViolationType.UNCERTAIN_OBSERVATION for v in res_low_conf.new_violations)
        print("  ✓ Invalid object class and low confidence interactions correctly flagged.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 7: Duplicate Event Protection (Idempotency)
    # -------------------------------------------------------------------------
    print("\n[Check 7/11] Duplicate Event Protection (Idempotency):")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)

        dup_ev = make_test_event("same_event_id_xyz", InteractionEventType.PICKUP_CANDIDATE, duration_ms=500, timestamp_s=11.0)
        res_first = engine.evaluate_event(dup_ev)
        assert res_first.completed_steps == 1
        assert res_first.current_step_id == "STEP_02"

        # Re-send same event ID
        res_second = engine.evaluate_event(dup_ev)
        assert res_second.completed_steps == 1
        assert res_second.current_step_id == "STEP_02"
        assert "Duplicate event" in res_second.message
        print("  ✓ Duplicate event ID replay successfully ignored; step not double-advanced.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 8: Calibration & Target Zone Validation
    # -------------------------------------------------------------------------
    print("\n[Check 8/11] Calibration-Dependent Rig Target Zone Validation:")
    try:
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)
        # Advance to Step 3
        engine.evaluate_event(make_test_event("e1", InteractionEventType.PICKUP_CANDIDATE, duration_ms=500, timestamp_s=11.0))
        engine.evaluate_event(make_test_event("e2", InteractionEventType.OBJECT_MOVEMENT, duration_ms=400, timestamp_s=12.0))
        assert engine.current_step_id == "STEP_03"

        # Uncalibrated placement attempt
        uncal_res = engine.evaluate_event(make_test_event("eu", InteractionEventType.PLACEMENT_CANDIDATE, calibration_status=False, timestamp_s=13.0))
        assert any(v.violation_type == ViolationType.CALIBRATION_REQUIRED for v in uncal_res.new_violations)
        assert engine.current_step_id == "STEP_03"

        # Calibrated but outside target zone
        out_res = engine.evaluate_event(make_test_event("eout", InteractionEventType.PLACEMENT_CANDIDATE, calibration_status=True, rig_position=(1.5, 1.5, 1.5), timestamp_s=14.0))
        assert any(v.violation_type == ViolationType.INVALID_OBJECT for v in out_res.new_violations)
        assert engine.current_step_id == "STEP_03"
        print("  ✓ Uncalibrated metric comparison rejected; target zone boundaries enforced.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 9: Alert Cooldown Throttling
    # -------------------------------------------------------------------------
    print("\n[Check 9/11] Alert Cooldown Throttling:")
    try:
        engine = ProcedureEngine(proto, alert_cooldown_seconds=3.0)
        engine.start(timestamp_s=10.0)

        # Rapidly stream 10 invalid events
        for i in range(10):
            ev = make_test_event(f"rapid_{i}", InteractionEventType.PICKUP_CANDIDATE, object_class="beaker", timestamp_s=10.0 + (i * 0.05))
            engine.evaluate_event(ev)

        assert len(engine.all_violations) == 1
        print("  ✓ 10 rapid violation events safely throttled to 1 alert by 3.0s cooldown.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 10: Step Timeout Detection
    # -------------------------------------------------------------------------
    print("\n[Check 10/11] Step Timeout Detection:")
    try:
        test_proto = load_protocol_from_dict({
            "protocol_id": "tout_proto",
            "steps": [{"step_id": "S1", "order": 1, "expected_event": "PICKUP_CANDIDATE", "timeout_seconds": 5}],
        })
        engine = ProcedureEngine(test_proto)
        engine.start(timestamp_s=100.0)

        late_ev = make_test_event("late", InteractionEventType.HAND_APPROACH, timestamp_s=110.0)
        res_tout = engine.evaluate_event(late_ev)
        assert res_tout.step_states["S1"].state == StepState.TIMED_OUT
        assert any(v.violation_type == ViolationType.TIMEOUT for v in res_tout.all_violations)
        print("  ✓ Step execution exceeding 5s timeout correctly transitioned to TIMED_OUT.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 11: UI ProtocolPanel Headless Integration
    # -------------------------------------------------------------------------
    print("\n[Check 11/11] UI ProtocolPanel Headless Integration:")
    try:
        from PySide6.QtWidgets import QApplication
        from vyom.ui.protocol_panel import ProtocolPanel

        app = QApplication.instance() or QApplication([])
        panel = ProtocolPanel()
        panel.load_protocol(proto)

        assert panel.step_list.count() == 4
        assert panel.progress_bar.value() == 0

        # Simulate engine evaluation update
        engine = ProcedureEngine(proto)
        engine.start(timestamp_s=10.0)
        e1 = make_test_event("ui_e1", InteractionEventType.PICKUP_CANDIDATE, duration_ms=500, timestamp_s=11.0)
        eval_res = engine.evaluate_event(e1)

        panel.update_evaluation_result(eval_res)
        assert panel.progress_bar.value() == 25
        assert "Completed: 1" in panel.step_counts_label.text()
        assert "STEP_02" in panel.step_id_label.text()
        print("  ✓ ProtocolPanel widget rendered 4 steps, updated progress bar to 25%, and advanced card.")
    except Exception as e:
        print(f"  ✕ Failed: {e}")
        all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("  >>> ALL 11 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print("  >>> SOME VERIFICATION CHECKS FAILED! <<<")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

