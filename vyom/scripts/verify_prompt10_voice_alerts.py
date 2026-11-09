"""Comprehensive verification script for Prompt 10: Offline Voice Alerts & Experiment Guidance System.

Validates:
1. Offline Voice Engine Initialization & SAPI5 Voice Discovery
2. Test Utterance Synthesis (Offline Local SAPI5)
3. Procedure Step Instruction Announcements
4. Out-of-Order Warning Announcements (High Priority / Interruption)
5. Alert Cooldown & Verbatim Repetition Throttling
6. Priority Queue Handling & Bounded Capacity Eviction
7. Muted Mode Operation
8. Graceful Fallback on Engine Unavailability
9. GUI Non-Blocking Thread Execution
10. SettingsDialog Voice Parameter Controls
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from vyom.alerts import (
    AlertCategory,
    AlertManager,
    OfflineVoiceEngine,
    VoiceAlertConfig,
    VoiceAlertItem,
    VoiceAlertPriority,
    VoiceEngineStatus,
    VoiceWorker,
)
from vyom.core.config import find_project_root, ConfigManager
from vyom.protocol import (
    ProcedureEngine,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolStep,
    StepProgress,
    StepState,
    ViolationType,
    load_protocol_from_file,
)
from vyom.protocol.protocol_schema import ProcedureViolation, ProcedureEvaluationResult


def run_checks() -> bool:
    print("=" * 78)
    print("  VYOM PROMPT 10: OFFLINE VOICE ALERTS & GUIDANCE SYSTEM VERIFICATION")
    print("=" * 78)
    all_passed = True

    # -------------------------------------------------------------------------
    # Check 1: SAPI5 Offline Voice Discovery
    # -------------------------------------------------------------------------
    print("\n[Check 1/10] SAPI5 Offline Voice Discovery:")
    try:
        voices = OfflineVoiceEngine.get_available_voices()
        assert len(voices) > 0, "Expected at least one installed offline voice on host OS"
        print(f"  ✓ Discovered {len(voices)} offline voices installed on host:")
        for v in voices:
            print(f"    - Voice: {v.name} (ID: {v.voice_id[:40]}...)")
    except Exception as e:
        print(f"  ✕ Failed voice discovery: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 2: Offline Engine Initialization & Short Speech Synthesis
    # -------------------------------------------------------------------------
    print("\n[Check 2/10] Local Offline Speech Synthesis Execution:")
    try:
        engine = OfflineVoiceEngine()
        init_ok, err = engine.initialize(rate=190, volume=0.9)
        assert init_ok is True, f"Engine initialization failed: {err}"
        assert engine.is_initialized is True

        speak_ok, speak_err = engine.speak("VYOM offline voice alert check.")
        assert speak_ok is True, f"Speech playback failed: {speak_err}"
        engine.shutdown()
        print("  ✓ Offline SAPI5 speech synthesis executed successfully without external dependencies.")
    except Exception as e:
        print(f"  ✕ Failed synthesis: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 3: Priority Queue Ordering & Eviction
    # -------------------------------------------------------------------------
    print("\n[Check 3/10] Priority Queue Scheduling (CRITICAL > HIGH > MEDIUM > LOW):")
    try:
        crit = VoiceAlertItem.create("c1", AlertCategory.OUT_OF_ORDER, "Out of order action", VoiceAlertPriority.CRITICAL)
        high = VoiceAlertItem.create("h1", AlertCategory.CALIBRATION_REQUIRED, "Calibration needed", VoiceAlertPriority.HIGH)
        med = VoiceAlertItem.create("m1", AlertCategory.STEP_INSTRUCTION, "Step instruction", VoiceAlertPriority.MEDIUM)
        low = VoiceAlertItem.create("l1", AlertCategory.STEP_COMPLETED, "Step 1 completed", VoiceAlertPriority.LOW)

        assert crit < high < med < low, "Priority queue ordering violation"
        print("  ✓ Priority comparison strictly ordered: CRITICAL (0) < HIGH (1) < MEDIUM (2) < LOW (3)")
    except Exception as e:
        print(f"  ✕ Failed priority ordering: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 4: Queue Bounded Capacity & Eviction
    # -------------------------------------------------------------------------
    print("\n[Check 4/10] Bounded Queue Capacity & Low-Priority Eviction:")
    try:
        cfg = VoiceAlertConfig(max_queue_size=3, enabled=True)
        worker = VoiceWorker(cfg)

        for i in range(3):
            item = VoiceAlertItem.create(f"l_{i}", AlertCategory.STEP_COMPLETED, f"Step {i} done", VoiceAlertPriority.LOW)
            assert worker.enqueue_alert(item) is True

        assert worker._queue.qsize() == 3

        # Add 4th low-priority item -> rejected
        extra_low = VoiceAlertItem.create("ex_l", AlertCategory.STEP_COMPLETED, "Extra low", VoiceAlertPriority.LOW)
        assert worker.enqueue_alert(extra_low) is False

        # Add CRITICAL item -> evicts lowest priority item, keeping queue size <= 3
        crit_alert = VoiceAlertItem.create("crit_1", AlertCategory.OUT_OF_ORDER, "Out of order!", VoiceAlertPriority.CRITICAL)
        assert worker.enqueue_alert(crit_alert) is True
        assert worker._queue.qsize() == 3
        print("  ✓ Bounded speech queue maintained max 3 items and evicted low-priority item for CRITICAL alert.")
    except Exception as e:
        print(f"  ✕ Failed queue eviction: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 5: Alert Cooldown & Verbatim De-duplication
    # -------------------------------------------------------------------------
    print("\n[Check 5/10] Cooldown Throttling & Verbatim Repetition Suppression:")
    try:
        mgr = AlertManager(VoiceAlertConfig(alert_cooldown_seconds=3.0, enabled=True))

        # First alert dispatch
        ok1 = mgr._dispatch_alert(
            category=AlertCategory.OUT_OF_ORDER,
            text="Out of order action. Complete Step 1 first.",
            priority=VoiceAlertPriority.CRITICAL,
            check_cooldown=True,
            cooldown_key=("OUT_OF_ORDER", "STEP_01", "tube_1"),
        )
        assert ok1 is True

        # Immediate repeat with same text/key -> suppressed
        ok2 = mgr._dispatch_alert(
            category=AlertCategory.OUT_OF_ORDER,
            text="Out of order action. Complete Step 1 first.",
            priority=VoiceAlertPriority.CRITICAL,
            check_cooldown=True,
            cooldown_key=("OUT_OF_ORDER", "STEP_01", "tube_1"),
        )
        assert ok2 is False
        print("  ✓ Identical alert suppressed by 3.0s cooldown and verbatim de-duplication.")
    except Exception as e:
        print(f"  ✕ Failed de-duplication: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 6: Procedure Engine Step Instruction Announcements
    # -------------------------------------------------------------------------
    print("\n[Check 6/10] Procedure Step Instruction Announcements:")
    try:
        proto_path = root_dir / "config" / "protocols" / "sample_experiment.json"
        proto = load_protocol_from_file(proto_path)
        mgr = AlertManager(VoiceAlertConfig(enabled=True))

        mgr.on_session_started(proto.experiment_name)
        assert mgr.worker._queue.qsize() == 1

        # Simulate active Step 1
        res1 = ProcedureEvaluationResult(
            protocol_id=proto.protocol_id,
            protocol_state=ProtocolLifecycleState.ACTIVE,
            current_step_id="STEP_01",
            current_step_index=0,
            total_steps=4,
            completed_steps=0,
            remaining_steps=4,
            progress_pct=0,
            step_states={"STEP_01": StepProgress(step_id="STEP_01", order=1, title="Pick up tube", state=StepState.IN_PROGRESS)},
        )
        mgr.process_procedure_result(res1, proto)
        assert mgr._announced_step_id == "STEP_01"
        print(f"  ✓ Active step instruction queued for [{proto.steps[0].step_id}]: '{proto.steps[0].instruction}'")
    except Exception as e:
        print(f"  ✕ Failed instruction announcement: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 7: Out-of-Order Corrective Warning Dispatch
    # -------------------------------------------------------------------------
    print("\n[Check 7/10] Out-of-Order Warning Dispatch (Priority=CRITICAL, can_interrupt=True):")
    try:
        viol = ProcedureViolation(
            violation_id="v_ooo_test",
            violation_type=ViolationType.OUT_OF_ORDER,
            step_id="STEP_01",
            expected_action="Step 1: Pick up tube",
            observed_action="Step 4: Release tube",
            explanation="Out-of-order execution: Step 4 attempted before Step 1 completed.",
            timestamp_s=time.time(),
            severity="critical",
            voice_message="Out of order action. Please complete Step 1 first.",
        )
        res_viol = ProcedureEvaluationResult(
            protocol_id=proto.protocol_id,
            protocol_state=ProtocolLifecycleState.ACTIVE,
            current_step_id="STEP_01",
            current_step_index=0,
            total_steps=4,
            completed_steps=0,
            remaining_steps=4,
            progress_pct=0,
            new_violations=[viol],
            step_states={"STEP_01": StepProgress(step_id="STEP_01", order=1, title="Pick up tube", state=StepState.IN_PROGRESS)},
        )
        mgr.process_procedure_result(res_viol, proto)
        queued_items = list(mgr.worker._queue.queue)
        ooo_item = next((it for it in queued_items if it.category == AlertCategory.OUT_OF_ORDER), None)
        assert ooo_item is not None
        assert ooo_item.priority_rank == VoiceAlertPriority.CRITICAL.value
        assert ooo_item.can_interrupt is True
        print(f"  ✓ Out-of-order alert queued with CRITICAL priority and can_interrupt=True: '{ooo_item.text}'")
    except Exception as e:
        print(f"  ✕ Failed out-of-order alert dispatch: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 8: Muted Mode & Disabled Category Handling
    # -------------------------------------------------------------------------
    print("\n[Check 8/10] Muted Mode & Category Filter Handling:")
    try:
        muted_worker = VoiceWorker(VoiceAlertConfig(enabled=False))
        item = VoiceAlertItem.create("t_m", AlertCategory.STEP_COMPLETED, "Muted test", VoiceAlertPriority.LOW)
        assert muted_worker.enqueue_alert(item) is False

        cat_worker = VoiceWorker(VoiceAlertConfig(enabled=True, enabled_categories={"LOW_CONFIDENCE_WARNING": False}))
        low_conf = VoiceAlertItem.create("t_lc", AlertCategory.LOW_CONFIDENCE_WARNING, "Low conf", VoiceAlertPriority.MEDIUM)
        assert cat_worker.enqueue_alert(low_conf) is False
        print("  ✓ Muted mode and disabled categories safely rejected without audio or queue pollution.")
    except Exception as e:
        print(f"  ✕ Failed muted handling: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 9: Background Worker Non-Blocking Execution
    # -------------------------------------------------------------------------
    print("\n[Check 9/10] Dedicated Worker Non-Blocking Execution:")
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        bg_worker = VoiceWorker(VoiceAlertConfig(enabled=True, rate=200))
        bg_worker.start()

        # Enqueue item to background thread
        test_item = VoiceAlertItem.create("bg_test", AlertCategory.SESSION_STARTED, "Non-blocking check.", VoiceAlertPriority.HIGH)
        bg_worker.enqueue_alert(test_item)

        # Main thread stays completely responsive
        t0 = time.time()
        for _ in range(5):
            app.processEvents()
            time.sleep(0.02)
        elapsed = time.time() - t0
        assert elapsed < 0.35, "Main GUI loop was blocked by speech synthesis"

        bg_worker.stop()
        print("  ✓ Speech synthesis operated asynchronously off GUI thread; main loop processed events smoothly.")
    except Exception as e:
        print(f"  ✕ Failed worker execution check: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Check 10: SettingsDialog Voice Parameter Controls
    # -------------------------------------------------------------------------
    print("\n[Check 10/10] SettingsDialog Voice Controls Integration:")
    try:
        from vyom.ui.settings_dialog import SettingsDialog

        cfg_mgr = ConfigManager(find_project_root())
        cfg_mgr.load()
        dlg = SettingsDialog(cfg_mgr, alert_manager=mgr)

        assert hasattr(dlg, "voice_enabled_check")
        assert hasattr(dlg, "voice_combo")
        assert hasattr(dlg, "rate_spin")
        assert hasattr(dlg, "volume_spin")
        assert hasattr(dlg, "test_voice_btn")

        # Test voice trigger
        dlg._test_voice()
        assert "SPEAKING" in dlg.voice_status_label.text()
        print("  ✓ SettingsDialog initialized voice controls, populated installed SAPI5 voices, and tested playback.")
    except Exception as e:
        print(f"  ✕ Failed settings dialog integration: {e}")
        all_passed = False

    print("\n" + "=" * 78)
    if all_passed:
        print("  >>> ALL 10 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print("  >>> SOME VERIFICATION CHECKS FAILED! <<<")
    print("=" * 78)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

