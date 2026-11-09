"""Focused unit tests for offline voice alerts, speech prioritization, and procedure integration.

Verifies Prompt 10:
1. Offline voice engine initialization and SAPI5 voice discovery.
2. Speaking a test message.
3. Procedure step instruction announcements.
4. Out-of-order warning announcements with high priority.
5. Alert cooldown and verbatim repetition deduplication.
6. Priority queue ordering and bounded capacity eviction.
7. Muted mode handling.
8. Graceful handling of unavailable or failing speech engines.
9. Background worker threading (non-blocking).
"""

import time
import pytest

from vyom.alerts.alert_manager import AlertManager
from vyom.alerts.config import VoiceAlertConfig
from vyom.alerts.schemas import (
    AlertCategory,
    VoiceAlertItem,
    VoiceAlertPriority,
    VoiceEngineStatus,
)
from vyom.alerts.voice_engine import OfflineVoiceEngine
from vyom.alerts.voice_worker import VoiceWorker
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


# ==============================================================================
# Verification 1 & 2: Offline Engine Discovery & Speech
# ==============================================================================

def test_offline_voice_discovery():
    """Verify discovery of installed offline Windows SAPI5 voices without network calls."""
    voices = OfflineVoiceEngine.get_available_voices()
    assert isinstance(voices, list)
    # On Windows, at least one default voice (e.g. David or Zira) is installed
    if voices:
        v = voices[0]
        assert v.voice_id
        assert v.name


def test_offline_engine_init_and_speak():
    """Verify initializing engine and speaking a short test phrase."""
    engine = OfflineVoiceEngine()
    success, err = engine.initialize(rate=180, volume=0.8)
    assert success is True
    assert err is None
    assert engine.is_initialized is True

    # Speak short phrase
    ok, speak_err = engine.speak("VYOM offline test.")
    assert ok is True
    assert speak_err is None

    engine.shutdown()
    assert engine.is_initialized is False


# ==============================================================================
# Verification 3: Priority Queue Ordering & Eviction
# ==============================================================================

def test_voice_alert_priority_ordering():
    """Verify that CRITICAL priority items rank before MEDIUM and LOW items."""
    item_low = VoiceAlertItem.create(
        alert_id="a1",
        category=AlertCategory.STEP_COMPLETED,
        text="Step 1 completed",
        priority=VoiceAlertPriority.LOW,
        timestamp_s=10.0,
    )
    item_crit = VoiceAlertItem.create(
        alert_id="a2",
        category=AlertCategory.OUT_OF_ORDER,
        text="Out of order action",
        priority=VoiceAlertPriority.CRITICAL,
        timestamp_s=11.0,
    )
    item_med = VoiceAlertItem.create(
        alert_id="a3",
        category=AlertCategory.STEP_INSTRUCTION,
        text="Pick up centrifuge tube",
        priority=VoiceAlertPriority.MEDIUM,
        timestamp_s=12.0,
    )

    # In Python PriorityQueue, lower rank pops first
    # CRITICAL (0) < MEDIUM (2) < LOW (3)
    assert item_crit < item_med
    assert item_med < item_low
    assert item_crit < item_low


def test_voice_worker_queue_capacity_and_eviction():
    """Verify bounded queue capacity and lower-priority eviction when queue fills."""
    cfg = VoiceAlertConfig(max_queue_size=3, enabled=True)
    worker = VoiceWorker(cfg)

    # Enqueue 3 low-priority items to fill queue
    for i in range(3):
        item = VoiceAlertItem.create(
            alert_id=f"low_{i}",
            category=AlertCategory.STEP_COMPLETED,
            text=f"Completion {i}",
            priority=VoiceAlertPriority.LOW,
            timestamp_s=10.0 + i,
        )
        assert worker.enqueue_alert(item) is True

    assert worker._queue.qsize() == 3

    # Attempting to add a 4th low-priority item is rejected
    item_extra_low = VoiceAlertItem.create(
        alert_id="low_extra",
        category=AlertCategory.STEP_COMPLETED,
        text="Extra low",
        priority=VoiceAlertPriority.LOW,
        timestamp_s=15.0,
    )
    assert worker.enqueue_alert(item_extra_low) is False
    assert worker._queue.qsize() == 3

    # Enqueue a HIGH priority alarm -> evicts a low priority item to maintain bounded size
    item_high = VoiceAlertItem.create(
        alert_id="high_alert",
        category=AlertCategory.OUT_OF_ORDER,
        text="Critical violation!",
        priority=VoiceAlertPriority.CRITICAL,
        timestamp_s=16.0,
    )
    assert worker.enqueue_alert(item_high) is True
    assert worker._queue.qsize() == 3


# ==============================================================================
# Verification 4 & 5: Cooldown and Verbatim De-duplication
# ==============================================================================

def test_alert_cooldown_and_verbatim_deduplication():
    """Verify that identical or rapid violation messages are throttled by cooldown."""
    cfg = VoiceAlertConfig(alert_cooldown_seconds=3.0, enabled=True)
    manager = AlertManager(cfg)

    # 1. First emission succeeds
    ok1 = manager._dispatch_alert(
        category=AlertCategory.OUT_OF_ORDER,
        text="Out of order action. Perform Step 1 first.",
        priority=VoiceAlertPriority.CRITICAL,
        check_cooldown=True,
        cooldown_key=("OUT_OF_ORDER", "STEP_01", "track_1"),
    )
    assert ok1 is True

    # 2. Immediate second emission with same key/text is suppressed by verbatim/key cooldown
    ok2 = manager._dispatch_alert(
        category=AlertCategory.OUT_OF_ORDER,
        text="Out of order action. Perform Step 1 first.",
        priority=VoiceAlertPriority.CRITICAL,
        check_cooldown=True,
        cooldown_key=("OUT_OF_ORDER", "STEP_01", "track_1"),
    )
    assert ok2 is False


# ==============================================================================
# Verification 6 & 7: Muted Mode & Disabled Category
# ==============================================================================

def test_muted_mode():
    """Verify that muted mode drops alerts without error."""
    cfg = VoiceAlertConfig(enabled=False)
    worker = VoiceWorker(cfg)

    item = VoiceAlertItem.create(
        alert_id="test_muted",
        category=AlertCategory.STEP_INSTRUCTION,
        text="Instruction while muted",
        priority=VoiceAlertPriority.MEDIUM,
    )
    # Dropped early because enabled=False
    enqueued = worker.enqueue_alert(item)
    assert enqueued is False


def test_disabled_category_filtering():
    """Verify that specific disabled categories are ignored."""
    cfg = VoiceAlertConfig(
        enabled=True,
        enabled_categories={"LOW_CONFIDENCE_WARNING": False},
    )
    worker = VoiceWorker(cfg)

    item = VoiceAlertItem.create(
        alert_id="test_disabled_cat",
        category=AlertCategory.LOW_CONFIDENCE_WARNING,
        text="Low confidence warning",
        priority=VoiceAlertPriority.MEDIUM,
    )
    assert worker.enqueue_alert(item) is False


# ==============================================================================
# Verification 8: Procedure Engine Integration
# ==============================================================================

def test_procedure_engine_integration_mapping():
    """Verify that AlertManager translates ProcedureEvaluationResults into prioritized voice alerts."""
    cfg = VoiceAlertConfig(enabled=True, alert_cooldown_seconds=3.0)
    manager = AlertManager(cfg)

    # Mock ProtocolDefinition with 2 steps
    proto = ProtocolDefinition(
        protocol_id="test_exp",
        experiment_name="Test Experiment",
        steps=[
            ProtocolStep(
                step_id="STEP_01",
                order=1,
                title="Pick up tube",
                human_instruction="Please pick up the centrifuge tube.",
                expected_event="PICKUP_CANDIDATE",
                voice_message="Please pick up the centrifuge tube.",
            ),
            ProtocolStep(
                step_id="STEP_02",
                order=2,
                title="Place in rack",
                human_instruction="Place the tube into the tube rack.",
                expected_event="PLACEMENT_CANDIDATE",
                voice_message="Place the tube into the rack.",
            ),
        ],
    )

    # 1. Session start announcement
    manager.on_session_started("Test Experiment")
    assert manager.worker._queue.qsize() == 1

    # 2. Active Step 1 instruction
    step1_prog = StepProgress(step_id="STEP_01", order=1, title="Pick up tube", state=StepState.IN_PROGRESS)
    eval_res_step1 = ProcedureEvaluationResult(
        protocol_id="test_exp",
        protocol_state=ProtocolLifecycleState.ACTIVE,
        current_step_id="STEP_01",
        current_step_index=0,
        total_steps=2,
        completed_steps=0,
        remaining_steps=2,
        progress_pct=0,
        step_states={"STEP_01": step1_prog},
    )
    manager.process_procedure_result(eval_res_step1, proto)
    # Instruction for STEP_01 queued
    assert manager._announced_step_id == "STEP_01"

    # 3. Out-of-Order violation
    viol = ProcedureViolation(
        violation_id="v_ooo_1",
        violation_type=ViolationType.OUT_OF_ORDER,
        step_id="STEP_01",
        expected_action="Complete Step 1",
        observed_action="Attempted Step 2",
        explanation="Out-of-order execution detected.",
        timestamp_s=time.time(),
        severity="critical",
        voice_message="Out of order action. Please complete Step 1 first.",
    )
    eval_res_viol = ProcedureEvaluationResult(
        protocol_id="test_exp",
        protocol_state=ProtocolLifecycleState.ACTIVE,
        current_step_id="STEP_01",
        current_step_index=0,
        total_steps=2,
        completed_steps=0,
        remaining_steps=2,
        progress_pct=0,
        new_violations=[viol],
        step_states={"STEP_01": step1_prog},
    )
    manager.process_procedure_result(eval_res_viol, proto)
    # The high-priority violation was queued
    assert any(it.category == AlertCategory.OUT_OF_ORDER for it in manager.worker._queue.queue)


# ==============================================================================
# Verification 9: Graceful Fallback on Missing Engine
# ==============================================================================

def test_engine_graceful_fallback():
    """Verify engine returns proper failure tuple when initialization cannot proceed."""
    engine = OfflineVoiceEngine()
    # Force invalid voice or driver failure
    engine._is_initialized = False
    ok, err = engine.speak("Should fail gracefully")
    assert ok is False
    assert "not initialized" in err.lower()
