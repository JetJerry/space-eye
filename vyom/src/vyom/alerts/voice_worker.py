"""Dedicated background QThread for offline voice alert playback and priority queueing.

Ensures that text-to-speech audio synthesis never blocks the GUI event loop,
camera acquisition, or procedure compliance evaluation.
"""

from __future__ import annotations

import queue
import time
from typing import Optional
from PySide6.QtCore import QObject, QThread, Signal, Slot

from vyom.alerts.config import VoiceAlertConfig
from vyom.alerts.schemas import (
    AlertCategory,
    VoiceAlertItem,
    VoiceAlertPriority,
    VoiceEngineStatus,
)
from vyom.alerts.voice_engine import OfflineVoiceEngine
from vyom.core.logging_setup import get_logger

logger = get_logger("alerts.voice_worker")


class VoiceWorker(QThread):
    """Off-GUI dedicated background worker managing SAPI5 voice playback and priority queue."""

    speech_started = Signal(object)          # VoiceAlertItem
    speech_finished = Signal(object)         # VoiceAlertItem
    speech_error = Signal(object, str)       # VoiceAlertItem, error_message
    status_changed = Signal(str, str)        # status_value, status_text
    alert_queued = Signal(object)            # VoiceAlertItem
    alert_dropped = Signal(object, str)      # VoiceAlertItem, reason

    def __init__(
        self,
        config: Optional[VoiceAlertConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or VoiceAlertConfig()
        self._queue: queue.PriorityQueue[VoiceAlertItem] = queue.PriorityQueue()
        self._running: bool = False
        self._is_speaking: bool = False
        self._engine: Optional[OfflineVoiceEngine] = None
        self._status: VoiceEngineStatus = VoiceEngineStatus.READY
        self._current_item: Optional[VoiceAlertItem] = None

    @property
    def status(self) -> VoiceEngineStatus:
        return self._status

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def enqueue_alert(self, item: VoiceAlertItem) -> bool:
        """Enqueue an alert item with priority ordering, flood control, and queue bounds."""
        if not self.config.is_category_enabled(item.category.value):
            logger.debug("Category '%s' is disabled; alert dropped", item.category.value)
            self.alert_dropped.emit(item, f"Category '{item.category.value}' is disabled in settings")
            return False

        # If muted, drop early or allow queuing for visual logs
        if not self.config.enabled:
            logger.debug("Voice alerts are muted; dropping item '%s'", item.text)
            self.alert_dropped.emit(item, "Voice alerts are muted")
            return False

        # Check queue capacity
        current_size = self._queue.qsize()
        if current_size >= self.config.max_queue_size:
            # If incoming item is high priority, make room by dropping lowest priority items
            if item.priority_rank <= VoiceAlertPriority.HIGH.value:
                dropped = self._drop_lowest_priority_item()
                if dropped:
                    self.alert_dropped.emit(dropped, "Evicted for higher priority alert")
            else:
                logger.warning(
                    "Speech queue full (%d items); dropping low priority alert '%s'",
                    current_size,
                    item.text,
                )
                self.alert_dropped.emit(item, "Queue capacity limit exceeded")
                return False

        # Check interruption policy: if high priority arrives and engine is speaking lower priority
        if (
            item.can_interrupt
            and self.config.allow_interruptions
            and self._is_speaking
            and self._current_item
            and self._current_item.priority_rank > item.priority_rank
        ):
            logger.info(
                "Interrupting lower priority speech ('%s') for high-priority alert ('%s')",
                self._current_item.text,
                item.text,
            )
            self._interrupt_current_speech()

        self._queue.put(item)
        self.alert_queued.emit(item)
        logger.debug("Queued voice alert [%s]: '%s' (priority=%d)", item.alert_id, item.text, item.priority_rank)
        return True

    def _drop_lowest_priority_item(self) -> Optional[VoiceAlertItem]:
        """Evict the item with the worst (highest rank) priority from the queue."""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not items:
            return None

        # Sort items: items with highest priority_rank are lowest urgency
        items.sort(key=lambda x: (x.priority_rank, -x.timestamp_s), reverse=True)
        evicted = items.pop(0)

        # Re-queue remaining items
        for it in items:
            self._queue.put(it)

        return evicted

    def _interrupt_current_speech(self) -> None:
        """Signal engine to stop ongoing speech."""
        if self._engine:
            self._engine.stop()

    def update_config(
        self,
        enabled: Optional[bool] = None,
        voice_id: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
    ) -> None:
        """Update runtime speech engine configuration."""
        if enabled is not None:
            self.config.enabled = enabled
            if not self.config.enabled:
                self._interrupt_current_speech()
                self._set_status(VoiceEngineStatus.MUTED, "VOICE: MUTED")
            else:
                self._set_status(VoiceEngineStatus.READY, "VOICE: READY")

        if voice_id is not None and voice_id != self.config.voice_id:
            self.config.voice_id = voice_id
            if self._engine:
                self._engine.set_voice(voice_id)

        if rate is not None:
            self.config.rate = rate
            if self._engine:
                self._engine.set_rate(rate)

        if volume is not None:
            self.config.volume = volume
            if self._engine:
                self._engine.set_volume(volume)

    def _set_status(self, status: VoiceEngineStatus, display_text: str) -> None:
        """Internal helper to emit status updates."""
        self._status = status
        self.status_changed.emit(status.value, display_text)

    def run(self) -> None:
        """Worker thread execution loop."""
        self._running = True

        # Initialize COM on this worker thread
        com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
        except Exception as e:
            logger.debug("COM initialization note: %s", e)

        # Initialize OfflineVoiceEngine
        self._engine = OfflineVoiceEngine()
        success, err = self._engine.initialize(
            voice_id=self.config.voice_id,
            rate=self.config.rate,
            volume=self.config.volume,
        )

        if not success:
            self._set_status(VoiceEngineStatus.UNAVAILABLE, "VOICE: UNAVAILABLE")
            logger.error("VoiceWorker could not initialize speech engine: %s", err)
        else:
            if not self.config.enabled:
                self._set_status(VoiceEngineStatus.MUTED, "VOICE: MUTED")
            else:
                self._set_status(VoiceEngineStatus.READY, "VOICE: READY")

        logger.info("VoiceWorker background thread started and ready")

        while self._running:
            try:
                # Wait for next alert item with timeout to allow clean shutdown checks
                item = self._queue.get(timeout=0.20)
            except queue.Empty:
                continue

            if not self._running:
                break

            # 1. Stale-message rejection
            age_s = time.time() - item.timestamp_s
            if age_s > self.config.max_queue_age_s:
                logger.warning(
                    "Stale speech alert dropped (age %.2fs > %.2fs limit): '%s'",
                    age_s,
                    self.config.max_queue_age_s,
                    item.text,
                )
                self.alert_dropped.emit(item, f"Message stale (queued {age_s:.1f}s ago)")
                self._queue.task_done()
                continue

            # 2. Check mute state
            if not self.config.enabled:
                self.alert_dropped.emit(item, "Voice muted")
                self._queue.task_done()
                continue

            # 3. Speak item
            self._current_item = item
            self._is_speaking = True
            self._set_status(VoiceEngineStatus.SPEAKING, "VOICE: SPEAKING")
            self.speech_started.emit(item)

            ok, speak_err = self._engine.speak(item.text) if self._engine else (False, "Engine unavailable")

            self._is_speaking = False
            self._current_item = None

            if ok:
                self.speech_finished.emit(item)
                logger.info("Successfully spoken alert [%s]: '%s'", item.alert_id, item.text)
            else:
                self.speech_error.emit(item, speak_err or "Unknown error")
                logger.warning("Failed speaking alert [%s]: %s", item.alert_id, speak_err)

            # Reset status back to ready or muted
            if not self.config.enabled:
                self._set_status(VoiceEngineStatus.MUTED, "VOICE: MUTED")
            elif not self._engine or not self._engine.is_initialized:
                self._set_status(VoiceEngineStatus.UNAVAILABLE, "VOICE: UNAVAILABLE")
            else:
                self._set_status(VoiceEngineStatus.READY, "VOICE: READY")

            self._queue.task_done()

        # Clean shutdown
        if self._engine:
            self._engine.shutdown()
            self._engine = None

        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

        logger.info("VoiceWorker background thread terminated cleanly")

    def stop(self) -> None:
        """Request worker thread termination and unblock queue."""
        self._running = False
        self._interrupt_current_speech()
        self.wait(1500)

