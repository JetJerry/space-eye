"""Dedicated background worker thread for asynchronous, non-blocking YOLOv8-OBB inference."""

from __future__ import annotations

import collections
import threading
import time
from typing import Deque, Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from vyom.core.logging_setup import get_logger
from vyom.vision.obb_detector import (
    OBBDetectorConfig,
    OBBDetectorStatus,
    OBBFrameDetections,
    YOLOOBBDetector,
)

logger = get_logger("vision.obb_worker")


class OBBTrackingWorker(QThread):
    """Executes YOLOv8-OBB object detection off the GUI and capture threads using latest-frame semantics."""

    # Emitted when an inference run finishes
    obb_result_ready = Signal(object)  # Emits OBBFrameDetections

    # Emitted on status transition
    status_changed = Signal(str, str)  # (status_value, detailed_message)

    def __init__(
        self,
        config: Optional[OBBDetectorConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.detector = YOLOOBBDetector(config)

        self._running: bool = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Bounded latest-frame container: (frame_bgr, frame_idx, timestamp_s)
        self._pending_frame: Optional[Tuple[np.ndarray, int, float]] = None

        # Cached latest result: (result, monotonic_completed_at)
        self._latest_result: Optional[Tuple[OBBFrameDetections, float]] = None

        # Performance monitoring
        self._total_processed: int = 0
        self._total_dropped: int = 0
        self._latency_history: Deque[float] = collections.deque(maxlen=30)
        self._last_inference_fps: float = 0.0
        self._fps_time_history: Deque[float] = collections.deque(maxlen=30)

    @property
    def is_initialized(self) -> bool:
        return self.detector.is_initialized

    @property
    def active_device(self) -> str:
        return self.detector.active_device

    @property
    def measured_fps(self) -> float:
        return self._last_inference_fps

    def initialize(self) -> Tuple[bool, OBBDetectorStatus, str]:
        """Initialize underlying YOLOv8-OBB model and report status."""
        success, status, msg = self.detector.initialize()
        self.status_changed.emit(status.value, msg)
        return success, status, msg

    def submit_frame(self, frame_bgr: np.ndarray, timestamp_s: float, frame_idx: int) -> None:
        """Submit frame for background YOLO object detection.

        Uses bounded latest-frame semantics: if the inference thread is currently
        busy with an earlier frame, the previous pending frame is replaced immediately,
        preventing any queue latency buildup.
        """
        with self._condition:
            if self._pending_frame is not None:
                self._total_dropped += 1
            # Store frame reference
            self._pending_frame = (frame_bgr, frame_idx, timestamp_s)
            self._condition.notify()

    def get_latest_result(self, max_age_s: float = 0.250) -> Optional[OBBFrameDetections]:
        """Retrieve most recent valid result if fresher than max_age_s.

        Prevents rendering stale or misaligned bounding boxes on much newer video frames.
        """
        with self._lock:
            if self._latest_result is None:
                return None
            result, completed_at = self._latest_result
            if time.monotonic() - completed_at > max_age_s:
                return None
            return result

    def run(self) -> None:
        """Main worker loop executing inference asynchronously."""
        logger.info("YOLOv8-OBB detection worker thread started")
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

            # Execute YOLOv8-OBB detection
            t_start = time.perf_counter()
            result = self.detector.detect(frame_bgr, timestamp_s, frame_idx)
            now = time.monotonic()
            wall_duration = time.perf_counter() - t_start

            self._total_processed += 1
            self._latency_history.append(result.inference_time_ms)
            self._fps_time_history.append(now)

            if len(self._fps_time_history) >= 2:
                dt = self._fps_time_history[-1] - self._fps_time_history[0]
                if dt > 0.1:
                    self._last_inference_fps = (len(self._fps_time_history) - 1) / dt

            with self._lock:
                self._latest_result = (result, now)

            # Emit signal to GUI
            self.obb_result_ready.emit(result)

        logger.info(
            "YOLOv8-OBB worker thread finished (processed: %d, dropped: %d)",
            self._total_processed,
            self._total_dropped,
        )

    def reset(self) -> None:
        """Clear cached results and pending frames."""
        with self._lock:
            self._pending_frame = None
            self._latest_result = None
            self._latency_history.clear()
            self._fps_time_history.clear()
            self._last_inference_fps = 0.0

    def stop(self) -> None:
        """Terminate worker thread and cleanly release detector resources."""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self.wait(1500)
        self.detector.close()

