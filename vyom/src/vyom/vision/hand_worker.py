"""Dedicated background worker thread for asynchronous, non-blocking MediaPipe hand tracking."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from vyom.core.logging_setup import get_logger
from vyom.vision.hand_tracker import (
    HandTracker,
    HandTrackerConfig,
    HandTrackingResult,
    HandTrackingStatus,
)

logger = get_logger("vision.hand_worker")


class HandTrackingWorker(QThread):
    """Executes MediaPipe hand tracking off the capture and GUI threads using latest-frame semantics."""

    # Emitted when a new hand tracking inference finishes
    hand_result_ready = Signal(object)  # Emits HandTrackingResult

    # Emitted on status transition
    status_changed = Signal(str, str)  # (status_value, detailed_message)

    def __init__(
        self,
        config: Optional[HandTrackerConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.tracker = HandTracker(config)

        self._running: bool = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Bounded latest-frame container: (frame_bgr, frame_idx, timestamp_s)
        self._pending_frame: Optional[Tuple[np.ndarray, int, float]] = None

        # Cached latest result: (result, monotonic_time, (width, height))
        self._latest_result: Optional[Tuple[HandTrackingResult, float, Tuple[int, int]]] = None

        # Performance monitoring
        self._total_processed: int = 0
        self._total_dropped: int = 0

    @property
    def is_initialized(self) -> bool:
        return self.tracker.is_initialized

    @property
    def is_ready(self) -> bool:
        return self.tracker.is_initialized

    @property
    def status(self) -> HandTrackingStatus:
        return HandTrackingStatus.SEARCHING if self.tracker.is_initialized else HandTrackingStatus.NOT_INITIALIZED

    def initialize(self) -> Tuple[bool, HandTrackingStatus, Optional[str]]:
        """Initialize underlying MediaPipe HandLandmarker."""
        success, status, err = self.tracker.initialize()
        self.status_changed.emit(status.value, err or "")
        return success, status, err

    def submit_frame(self, frame_bgr: np.ndarray, timestamp_s: float, frame_idx: int) -> None:
        """Submit frame for background hand tracking.

        Uses latest-frame semantics: if the tracker is currently busy processing,
        the previous pending frame is replaced immediately, preventing any queue buildup.
        """
        with self._condition:
            if self._pending_frame is not None:
                self._total_dropped += 1
            # Store shallow copy of frame reference
            self._pending_frame = (frame_bgr, frame_idx, timestamp_s)
            self._condition.notify()

    def get_latest_result(self, max_age_s: float = 0.200) -> Optional[HandTrackingResult]:
        """Retrieve most recent valid result if fresher than max_age_s.

        Prevents rendering stale or mismatched hand landmarks on much newer video frames.
        """
        with self._lock:
            if self._latest_result is None:
                return None
            result, completed_at, _ = self._latest_result
            if time.monotonic() - completed_at > max_age_s:
                return None
            return result

    def run(self) -> None:
        """Main worker loop processing frames asynchronously."""
        logger.info("Hand tracking worker thread started")
        self._running = True

        while self._running:
            frame_tuple: Optional[Tuple[np.ndarray, int, float]] = None

            with self._condition:
                while self._running and self._pending_frame is None:
                    self._condition.wait(timeout=0.05)

                if not self._running:
                    break

                frame_tuple = self._pending_frame
                self._pending_frame = None

            if frame_tuple is None:
                continue

            frame_bgr, frame_idx, timestamp_s = frame_tuple
            timestamp_ms = int(timestamp_s * 1000.0)

            # Execute MediaPipe inference
            result = self.tracker.process_frame(frame_bgr, timestamp_ms, frame_idx)
            self._total_processed += 1

            now = time.monotonic()
            h, w = frame_bgr.shape[:2]

            with self._lock:
                self._latest_result = (result, now, (w, h))

            # Emit signal to GUI
            self.hand_result_ready.emit(result)

        logger.info(
            "Hand tracking worker finished (processed: %d, dropped: %d)",
            self._total_processed,
            self._total_dropped,
        )

    def reset(self) -> None:
        """Clear cached results and reset tracker timestamps."""
        with self._lock:
            self._pending_frame = None
            self._latest_result = None
        self.tracker.reset()

    def stop(self) -> None:
        """Terminate worker thread and release model handles."""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self.wait(1500)
        self.tracker.close()

