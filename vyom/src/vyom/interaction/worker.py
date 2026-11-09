"""Dedicated background worker thread for asynchronous hand-object interaction recognition."""

from __future__ import annotations

import collections
import threading
import time
from typing import Deque, Optional, Tuple
from PySide6.QtCore import QObject, QThread, Signal, Slot

from vyom.core.logging_setup import get_logger
from vyom.interaction.engine import (
    InteractionConfig,
    InteractionEngine,
    InteractionFrameResult,
)
from vyom.interaction.events import InteractionEvent
from vyom.vision.observations import PerceptionFrameObservation

logger = get_logger("interaction.worker")


class InteractionWorker(QThread):
    """Executes state-based interaction recognition off the GUI and capture threads."""

    # Emitted when frame interaction recognition completes
    interaction_result_ready = Signal(object)  # Emits InteractionFrameResult

    # Emitted specifically when a discrete new interaction event is extracted
    interaction_event_emitted = Signal(object)  # Emits InteractionEvent

    def __init__(
        self,
        config: Optional[InteractionConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.engine = InteractionEngine(config)

        self._running: bool = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Bounded latest-observation container
        self._pending_obs: Optional[PerceptionFrameObservation] = None

        # Cached latest result: (result, monotonic_completed_at)
        self._latest_result: Optional[Tuple[InteractionFrameResult, float]] = None

        # Diagnostics
        self._total_processed: int = 0
        self._total_dropped: int = 0
        self._total_events: int = 0
        self._latency_history: Deque[float] = collections.deque(maxlen=30)

    def submit_observation(self, obs: PerceptionFrameObservation) -> None:
        """Submit a multi-modal perception frame observation for interaction analysis.

        Uses bounded latest-frame semantics: if the worker thread is busy, the earlier
        pending observation is dropped immediately to prevent pipeline latency buildup.
        """
        with self._condition:
            if self._pending_obs is not None:
                self._total_dropped += 1
            self._pending_obs = obs
            self._condition.notify()

    def get_latest_result(self, max_age_s: float = 0.250) -> Optional[InteractionFrameResult]:
        """Retrieve most recent interaction result if fresher than max_age_s."""
        with self._lock:
            if self._latest_result is None:
                return None
            res, completed_at = self._latest_result
            if time.monotonic() - completed_at > max_age_s:
                return None
            return res

    def run(self) -> None:
        """Main worker loop evaluating hand-object interactions asynchronously."""
        logger.info("InteractionWorker background thread started")
        self._running = True

        while self._running:
            obs: Optional[PerceptionFrameObservation] = None

            with self._condition:
                while self._running and self._pending_obs is None:
                    self._condition.wait(timeout=0.05)

                if not self._running:
                    break

                obs = self._pending_obs
                self._pending_obs = None

            if obs is None:
                continue

            # Execute deterministic interaction reasoning
            result = self.engine.process_observation(obs)
            now = time.monotonic()

            self._total_processed += 1
            self._latency_history.append(result.processing_time_ms)

            with self._lock:
                self._latest_result = (result, now)

            # Emit frame result
            self.interaction_result_ready.emit(result)

            # Emit individual events
            for ev in result.events:
                self._total_events += 1
                self.interaction_event_emitted.emit(ev)

        logger.info(
            "InteractionWorker finished (processed: %d, dropped: %d, events: %d)",
            self._total_processed,
            self._total_dropped,
            self._total_events,
        )

    def reset(self) -> None:
        """Reset state machines and queues."""
        with self._lock:
            self._pending_obs = None
            self._latest_result = None
            self._latency_history.clear()
        self.engine.reset()

    def stop(self) -> None:
        """Stop worker thread and wait for completion."""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self.wait(1500)
        self.engine.reset()

