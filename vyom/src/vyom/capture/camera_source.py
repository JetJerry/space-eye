"""Hardware webcam and USB camera capture source using OpenCV."""

from __future__ import annotations

import platform
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from vyom.capture.frame_source import FrameSource, SourceMetadata
from vyom.core.logging_setup import get_logger

logger = get_logger("capture.camera")


class CameraSource(FrameSource):
    """Acquires video frames from physical webcam or USB video capture device."""

    def __init__(
        self,
        device_index: int = 0,
        target_width: int = 1280,
        target_height: int = 720,
        target_fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._paused: bool = False
        self._metadata = SourceMetadata(
            source_type="webcam",
            source_label=f"Device {device_index}",
            width=target_width,
            height=target_height,
            fps=float(target_fps),
            is_open=False,
        )

    def open(self) -> bool:
        """Open physical camera device with Windows DirectShow backend if available."""
        self.release()
        self._frame_idx = 0

        # On Windows, DirectShow (CAP_DSHOW) or Media Foundation (CAP_MSMF) initializes much faster
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

        logger.info(
            "Opening camera device %d (backend: %s, target: %dx%d @ %d FPS)",
            self.device_index,
            "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "DEFAULT",
            self.target_width,
            self.target_height,
            self.target_fps,
        )

        try:
            self._cap = cv2.VideoCapture(self.device_index, backend)
        except Exception as e:
            logger.error("Exception creating VideoCapture for device %d: %s", self.device_index, e)
            return False

        if not self._cap or not self._cap.isOpened():
            # Retry with default backend if DSHOW failed
            if backend != cv2.CAP_ANY:
                logger.warning("DirectShow open failed for device %d; retrying with default backend", self.device_index)
                self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)

        if not self._cap or not self._cap.isOpened():
            logger.warning("Camera device %d is not available or disconnected", self.device_index)
            self._metadata.is_open = False
            return False

        # Request configured target parameters
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.target_width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.target_height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        # Inspect negotiated hardware properties
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if actual_fps <= 0.0 or actual_fps > 120.0:
            actual_fps = float(self.target_fps)

        self._metadata = SourceMetadata(
            source_type="webcam",
            source_label=f"Device {self.device_index}: Camera",
            width=actual_w,
            height=actual_h,
            fps=actual_fps,
            is_open=True,
        )

        logger.info("Camera %d opened successfully (%dx%d @ %.1f FPS)", self.device_index, actual_w, actual_h, actual_fps)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Read latest frame with monotonic capture timestamp."""
        if not self.is_opened():
            return False, None, time.monotonic(), self._frame_idx

        if self._paused:
            time.sleep(0.02)
            return False, None, time.monotonic(), self._frame_idx

        ret, frame = self._cap.read()
        timestamp_s = time.monotonic()

        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera device %d", self.device_index)
            return False, None, timestamp_s, self._frame_idx

        self._frame_idx += 1
        return True, frame, timestamp_s, self._frame_idx

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def release(self) -> None:
        """Release camera capture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing camera device %d: %s", self.device_index, e)
            self._cap = None
        self._metadata.is_open = False
        self._paused = False
        logger.info("Camera device %d released", self.device_index)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_metadata(self) -> SourceMetadata:
        return self._metadata
