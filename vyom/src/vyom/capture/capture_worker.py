"""Background QThread worker for non-blocking video frame acquisition."""

from __future__ import annotations

import collections
import time
from typing import Optional
import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from vyom.capture.frame_source import FrameSource
from vyom.capture.recorder import SessionRecorder
from vyom.core.logging_setup import get_logger

logger = get_logger("capture.worker")


class CaptureWorker(QThread):
    """Acquires video frames in a dedicated thread and communicates with the GUI via Qt signals."""

    # UI display frame (RGB888 QImage)
    frame_ready = Signal(QImage, int, float)

    # Raw numpy BGR frame for future CV perception pipeline (YOLO, MediaPipe, ArUco)
    raw_frame_ready = Signal(object, int, float)

    # Performance & Status Signals
    fps_measured = Signal(float, int)
    source_opened = Signal(str, int, int, float)
    source_closed = Signal()
    source_error = Signal(str)

    # Recording Signals
    recording_started = Signal(str)
    recording_stopped = Signal(str, int, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source: Optional[FrameSource] = None
        self._recorder: Optional[SessionRecorder] = None

        self._running: bool = False
        self._paused: bool = False

        # Sliding window for accurate measured FPS calculation
        self._timestamps_window = collections.deque(maxlen=30)
        self._dropped_frames_count: int = 0
        self._last_fps_emit_time: float = 0.0

    def configure(
        self,
        source: FrameSource,
        recorder: Optional[SessionRecorder] = None,
    ) -> None:
        """Assign the active capture source and optional recorder before starting."""
        self._source = source
        self._recorder = recorder

    def run(self) -> None:
        """Main acquisition loop running exclusively inside the QThread."""
        if self._source is None:
            self.source_error.emit("No video source configured.")
            return

        logger.info("Capture worker thread started")

        if not self._source.open():
            meta = self._source.get_metadata()
            msg = f"Failed to open source: {meta.source_label}"
            logger.error(msg)
            self.source_error.emit(msg)
            self.source_closed.emit()
            return

        meta = self._source.get_metadata()
        self.source_opened.emit(meta.source_label, meta.width, meta.height, meta.fps)

        # Start optional recording
        if self._recorder is not None:
            if self._recorder.start():
                if self._recorder.output_path:
                    self.recording_started.emit(str(self._recorder.output_path))
            else:
                self.source_error.emit("Failed to initialize session video recording.")

        self._running = True
        self._paused = False
        self._timestamps_window.clear()
        self._dropped_frames_count = 0
        self._last_fps_emit_time = time.monotonic()

        while self._running:
            if self._paused:
                time.sleep(0.02)
                continue

            success, frame_bgr, timestamp_s, frame_idx = self._source.read()

            if not success or frame_bgr is None:
                # Check whether EOF or disconnected
                if not self._source.is_opened():
                    logger.warning("Capture source was closed or disconnected")
                    self.source_error.emit("Video stream ended or source disconnected.")
                    break
                # EOF reached on video file
                meta = self._source.get_metadata()
                if meta.source_type == "file":
                    logger.info("Video file playback reached end of stream")
                    break
                # Temporary glitch on camera
                time.sleep(0.005)
                continue

            # Record frame if recorder is active
            if self._recorder is not None and self._recorder.is_recording:
                self._recorder.write_frame(frame_bgr)

            # Emit raw BGR numpy array for future perception pipeline (Prompts 4-8)
            self.raw_frame_ready.emit(frame_bgr, frame_idx, timestamp_s)

            # Convert BGR -> RGB QImage for PySide6 GUI display
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Copy data buffer to prevent any race condition with next read
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            self.frame_ready.emit(qimg, frame_idx, timestamp_s)

            # Update FPS calculation
            self._timestamps_window.append(timestamp_s)
            now = time.monotonic()
            if now - self._last_fps_emit_time >= 0.5:
                if len(self._timestamps_window) >= 2:
                    window_duration = self._timestamps_window[-1] - self._timestamps_window[0]
                    if window_duration > 0.001:
                        current_fps = (len(self._timestamps_window) - 1) / window_duration
                        self.fps_measured.emit(current_fps, self._dropped_frames_count)
                self._last_fps_emit_time = now

        # Finalize and clean up resources
        self._cleanup()
        self.source_closed.emit()
        logger.info("Capture worker thread finished")

    def _cleanup(self) -> None:
        """Release capture source and finalize recording."""
        if self._recorder is not None and self._recorder.is_recording:
            rec_summary = self._recorder.release()
            if rec_summary.get("success"):
                self.recording_stopped.emit(
                    rec_summary.get("output_file", ""),
                    rec_summary.get("total_frames", 0),
                    rec_summary.get("duration_seconds", 0.0),
                )

        if self._source is not None:
            try:
                self._source.release()
            except Exception as e:
                logger.warning("Error releasing frame source: %s", e)

    @Slot()
    def pause_capture(self) -> None:
        """Pause capture loop without releasing device handles."""
        self._paused = True
        if self._source:
            self._source.pause()
        logger.info("Capture worker paused")

    @Slot()
    def resume_capture(self) -> None:
        """Resume capture loop."""
        self._paused = False
        if self._source:
            self._source.resume()
        self._timestamps_window.clear()
        self._last_fps_emit_time = time.monotonic()
        logger.info("Capture worker resumed")

    @Slot()
    def stop_capture(self) -> None:
        """Signal thread to stop and wait for clean termination."""
        self._running = False
        self._paused = False
        self.wait(1500)
        logger.info("Capture worker stop requested")

