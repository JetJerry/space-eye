"""Central alert management, message templating, cooldown tracking, and procedure engine integration.

Part of VYOM Prompt 10: Local Voice Alerts and Experiment Guidance System.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import uuid
from PySide6.QtCore import QObject, Signal, Slot

from vyom.alerts.config import VoiceAlertConfig
from vyom.alerts.schemas import (
    AlertCategory,
    VoiceAlertItem,
    VoiceAlertPriority,
    VoiceEngineStatus,
    VoiceInfo,
)
from vyom.alerts.voice_engine import OfflineVoiceEngine
from vyom.alerts.voice_worker import VoiceWorker
from vyom.core.logging_setup import get_logger
from vyom.protocol.alerts import AlertCooldownTracker
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProcedureViolation,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolStep,
    ViolationType,
)

logger = get_logger("alerts.manager")


class AlertManager(QObject):
    """High-level alert orchestrator converting procedure compliance outputs into spoken audio guidance."""

    # UI forwarding signals
    voice_status_changed = Signal(str, str)     # status_enum_val, status_text
    voice_speech_started = Signal(object)       # VoiceAlertItem
    voice_speech_finished = Signal(object)      # VoiceAlertItem
    voice_speech_error = Signal(object, str)    # VoiceAlertItem, error_msg
    voice_alert_queued = Signal(object)         # VoiceAlertItem
    voice_alert_dropped = Signal(object, str)   # VoiceAlertItem, reason

    def __init__(
        self,
        config: Optional[VoiceAlertConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or VoiceAlertConfig()
        self.cooldown_tracker = AlertCooldownTracker(
            default_cooldown_seconds=self.config.alert_cooldown_seconds,
        )

        # Worker thread
        self.worker = VoiceWorker(self.config, self)
        self._connect_worker_signals()

        # Track history to prevent verbatim repetitive speech
        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self._announced_step_id: Optional[str] = None
        self._announced_completed_steps: set[str] = set()
        self._announced_procedure_completed: bool = False

    def _connect_worker_signals(self) -> None:
        """Forward background thread worker signals to UI listeners."""
        self.worker.status_changed.connect(self.voice_status_changed)
        self.worker.speech_started.connect(self._on_speech_started)
        self.worker.speech_finished.connect(self._on_speech_finished)
        self.worker.speech_error.connect(self.voice_speech_error)
        self.worker.alert_queued.connect(self.voice_alert_queued)
        self.worker.alert_dropped.connect(self.voice_alert_dropped)

    def start(self) -> None:
        """Start the background voice synthesis worker."""
        if not self.worker.isRunning():
            self.worker.start()
            logger.info("AlertManager voice worker thread started")

    def stop(self) -> None:
        """Stop background speech worker and clean up resources."""
        if self.worker.isRunning():
            self.worker.stop()
            logger.info("AlertManager voice worker thread stopped")

    @Slot(object)
    def _on_speech_started(self, item: VoiceAlertItem) -> None:
        self._last_spoken_text = item.text
        self._last_spoken_time = time.time()
        self.voice_speech_started.emit(item)

    @Slot(object)
    def _on_speech_finished(self, item: VoiceAlertItem) -> None:
        self.voice_speech_finished.emit(item)

    # -------------------------------------------------------------------------
    # Procedure Engine Integration
    # -------------------------------------------------------------------------

    def on_session_started(self, experiment_name: str) -> None:
        """Announce experiment session initialization."""
        self.reset_announcement_state()
        text = f"Session started for {experiment_name}. Ready for procedure."
        self._dispatch_alert(
            category=AlertCategory.SESSION_STARTED,
            text=text,
            priority=VoiceAlertPriority.LOW,
            reason="Session initiated by operator",
        )

    def on_session_aborted(self) -> None:
        """Announce session abort."""
        self._dispatch_alert(
            category=AlertCategory.PROCEDURE_ABORTED,
            text="Procedure stopped.",
            priority=VoiceAlertPriority.HIGH,
            reason="Session stopped by operator",
            can_interrupt=True,
        )

    def process_procedure_result(
        self,
        result: ProcedureEvaluationResult,
        protocol: Optional[ProtocolDefinition] = None,
    ) -> None:
        """Evaluate a ProcedureEvaluationResult and dispatch relevant spoken guidance.

        Consumes validated procedure state transitions:
        - When a new step becomes active: speak instruction.
        - When a step completes: speak completion feedback.
        - When an out-of-order violation occurs: speak corrective guidance.
        - When a timeout occurs: speak timeout alert.
        - When calibration is required: speak calibration warning.
        - When the protocol completes: announce completion.
        """
        # 1. Full Procedure Completion
        if result.is_success and not self._announced_procedure_completed:
            self._announced_procedure_completed = True
            text = "Experiment procedure completed successfully. All steps verified."
            self._dispatch_alert(
                category=AlertCategory.PROCEDURE_COMPLETED,
                text=text,
                priority=VoiceAlertPriority.LOW,
                reason="All protocol steps fulfilled",
            )
            return

        # If procedure not active, do not announce step progression
        if result.protocol_state != ProtocolLifecycleState.ACTIVE:
            return

        # 2. Step Completion Feedback
        # Find any steps that recently transitioned to COMPLETED that we haven't spoken yet
        for step_id, sp in result.step_states.items():
            if sp.state.value == "COMPLETED" and step_id not in self._announced_completed_steps:
                self._announced_completed_steps.add(step_id)
                text = f"Step {sp.order} completed."
                self._dispatch_alert(
                    category=AlertCategory.STEP_COMPLETED,
                    text=text,
                    priority=VoiceAlertPriority.LOW,
                    step_id=step_id,
                    reason=f"Step {step_id} completed",
                )

        # 3. New Active Step Instruction
        current_step_id = result.current_step_id
        if current_step_id and current_step_id != self._announced_step_id:
            self._announced_step_id = current_step_id
            instruction_text = self._get_step_instruction_text(current_step_id, protocol)
            if instruction_text:
                self._dispatch_alert(
                    category=AlertCategory.STEP_INSTRUCTION,
                    text=instruction_text,
                    priority=VoiceAlertPriority.MEDIUM,
                    step_id=current_step_id,
                    reason=f"Active step instruction for {current_step_id}",
                )

        # 4. Process Compliance Violations (Prioritized & Throttled)
        for viol in result.new_violations:
            self._process_violation(viol)

    def _get_step_instruction_text(
        self,
        step_id: str,
        protocol: Optional[ProtocolDefinition],
    ) -> Optional[str]:
        """Retrieve concise spoken instruction prompt for a procedure step."""
        if not protocol:
            return None
        for step in protocol.steps:
            if step.step_id == step_id:
                # Use voice_message if specified, otherwise title or human_instruction
                return step.voice_message or step.human_instruction or step.title
        return None

    def _process_violation(self, viol: ProcedureViolation) -> None:
        """Translate a deterministic ProcedureViolation into spoken warning audio."""
        category_map = {
            ViolationType.OUT_OF_ORDER: AlertCategory.OUT_OF_ORDER,
            ViolationType.TIMEOUT: AlertCategory.STEP_TIMEOUT,
            ViolationType.CALIBRATION_REQUIRED: AlertCategory.CALIBRATION_REQUIRED,
            ViolationType.UNCERTAIN_OBSERVATION: AlertCategory.LOW_CONFIDENCE_WARNING,
            ViolationType.INVALID_OBJECT: AlertCategory.OUT_OF_ORDER,
            ViolationType.MISSING_PREREQUISITE: AlertCategory.OUT_OF_ORDER,
            ViolationType.PROCEDURE_BLOCKED: AlertCategory.OUT_OF_ORDER,
        }
        category = category_map.get(viol.violation_type, AlertCategory.OUT_OF_ORDER)

        priority_map = {
            ViolationType.OUT_OF_ORDER: VoiceAlertPriority.CRITICAL,
            ViolationType.TIMEOUT: VoiceAlertPriority.CRITICAL,
            ViolationType.CALIBRATION_REQUIRED: VoiceAlertPriority.HIGH,
            ViolationType.UNCERTAIN_OBSERVATION: VoiceAlertPriority.MEDIUM,
            ViolationType.INVALID_OBJECT: VoiceAlertPriority.HIGH,
            ViolationType.MISSING_PREREQUISITE: VoiceAlertPriority.CRITICAL,
            ViolationType.PROCEDURE_BLOCKED: VoiceAlertPriority.CRITICAL,
        }
        priority = priority_map.get(viol.violation_type, VoiceAlertPriority.HIGH)
        can_interrupt = priority == VoiceAlertPriority.CRITICAL

        # Message text: use explicit voice_message on violation if provided
        text = viol.voice_message or viol.explanation

        self._dispatch_alert(
            category=category,
            text=text,
            priority=priority,
            reason=viol.explanation,
            event_id=viol.event_id,
            step_id=viol.step_id,
            can_interrupt=can_interrupt,
            severity=viol.severity,
            check_cooldown=True,
            cooldown_key=(viol.violation_type.value, viol.step_id, viol.object_track_id),
        )

    # -------------------------------------------------------------------------
    # Dispatch & Flood Control
    # -------------------------------------------------------------------------

    def _dispatch_alert(
        self,
        category: AlertCategory,
        text: str,
        priority: VoiceAlertPriority,
        reason: str = "",
        event_id: Optional[str] = None,
        step_id: Optional[str] = None,
        can_interrupt: bool = False,
        severity: str = "warning",
        check_cooldown: bool = False,
        cooldown_key: Optional[Tuple[str, Optional[str], Optional[str]]] = None,
    ) -> bool:
        """Internal dispatcher checking verbatim de-duplication and cooldown before queueing."""
        now = time.time()

        # 1. Verbatim De-duplication: avoid repeating identical sentence within cooldown window
        if text == self._last_spoken_text and (now - self._last_spoken_time) < self.config.alert_cooldown_seconds:
            logger.debug("Verbatim repetition suppressed for text: '%s'", text)
            return False

        # 2. Keyed Cooldown Check
        if check_cooldown and cooldown_key:
            last_time = self.cooldown_tracker._last_emission_times.get(cooldown_key)
            if last_time and (now - last_time) < self.config.alert_cooldown_seconds:
                logger.debug("Cooldown suppressed alert for key %s", cooldown_key)
                return False
            self.cooldown_tracker._last_emission_times[cooldown_key] = now

        alert_id = f"voice_{uuid.uuid4().hex[:8]}"
        item = VoiceAlertItem.create(
            alert_id=alert_id,
            category=category,
            text=text,
            priority=priority,
            reason=reason,
            event_id=event_id,
            step_id=step_id,
            can_interrupt=can_interrupt,
            severity=severity,
            timestamp_s=now,
        )

        return self.worker.enqueue_alert(item)

    def speak_test_message(self, custom_text: Optional[str] = None) -> None:
        """Play a short verification utterance to test voice synthesis settings."""
        text = custom_text or "VYOM voice alert system test. Offline speech driver operational."
        self._dispatch_alert(
            category=AlertCategory.SESSION_STARTED,
            text=text,
            priority=VoiceAlertPriority.HIGH,
            reason="Operator triggered test utterance",
            can_interrupt=True,
        )

    def set_muted(self, muted: bool) -> None:
        """Set mute state."""
        self.config.enabled = not muted
        self.worker.update_config(enabled=not muted)

    def set_rate(self, rate: int) -> None:
        """Update speech rate in WPM."""
        self.config.rate = rate
        self.worker.update_config(rate=rate)

    def set_volume(self, volume: float) -> None:
        """Update speech volume [0.0, 1.0]."""
        self.config.volume = volume
        self.worker.update_config(volume=volume)

    def set_voice(self, voice_id: str) -> None:
        """Update selected voice ID."""
        self.config.voice_id = voice_id
        self.worker.update_config(voice_id=voice_id)

    def reset_announcement_state(self) -> None:
        """Reset step and completion announcement history for a new session."""
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0
        self._announced_step_id = None
        self._announced_completed_steps.clear()
        self._announced_procedure_completed = False
        self.cooldown_tracker.reset()

