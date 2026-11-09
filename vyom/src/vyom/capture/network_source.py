"""Network RTSP and HTTP IP camera capture source."""

from __future__ import annotations

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from vyom.capture.frame_source import FrameSource, SourceMetadata
from vyom.core.logging_setup import get_logger

logger = get_logger("capture.network")


class NetworkSource(FrameSource):
    """Acquires video frames from an RTSP, RTMP, or HTTP network camera stream."""

    def __init__(self, stream_url: str, target_fps: int = 30) -> None:
        self.stream_url = stream_url
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._paused: bool = False
        self._metadata = SourceMetadata(
            source_type="network",
            source_label=self._sanitize_url(stream_url),
            width=1280,
            height=720,
            fps=float(target_fps),
            is_open=False,
        )

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Strip sensitive credentials from network stream URLs for display."""
        if "@" in url:
            scheme, rest = url.split("://", 1) if "://" in url else ("", url)
            _, host = rest.split("@", 1)
            return f"{scheme}://***@{host}" if scheme else f"***@{host}"
        return url

    def open(self) -> bool:
        """Open network video stream."""
        self.release()
        self._frame_idx = 0

        logger.info("Connecting to network video stream: %s", self._metadata.source_label)
        try:
            self._cap = cv2.VideoCapture(self.stream_url)
        except Exception as e:
            logger.error("Exception opening network stream %s: %s", self._metadata.source_label, e)
            return False

        if not self._cap or not self._cap.isOpened():
            logger.warning("Failed to connect to network stream: %s", self._metadata.source_label)
            self._metadata.is_open = False
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if actual_fps <= 0.0 or actual_fps > 120.0:
            actual_fps = float(self.target_fps)

        self._metadata.width = actual_w or 1280
        self._metadata.height = actual_h or 720
        self._metadata.fps = actual_fps
        self._metadata.is_open = True

        logger.info("Network stream %s opened (%dx%d @ %.1f FPS)", self._metadata.source_label, actual_w, actual_h, actual_fps)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Read frame from network stream."""
        if not self.is_opened():
            return False, None, time.monotonic(), self._frame_idx

        if self._paused:
            time.sleep(0.02)
            return False, None, time.monotonic(), self._frame_idx

        ret, frame = self._cap.read()
        timestamp_s = time.monotonic()

        if not ret or frame is None:
            logger.warning("Dropped frame or stream interrupted: %s", self._metadata.source_label)
            return False, None, timestamp_s, self._frame_idx

        self._frame_idx += 1
        return True, frame, timestamp_s, self._frame_idx

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def release(self) -> None:
        """Release network stream handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing network stream: %s", e)
            self._cap = None
        self._metadata.is_open = False
        self._paused = False
        logger.info("Network stream released: %s", self._metadata.source_label)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_metadata(self) -> SourceMetadata:
        return self._metadata
