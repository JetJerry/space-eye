"""Local video file replay source supporting MP4 and AVI containers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from vyom.capture.frame_source import FrameSource, SourceMetadata
from vyom.core.logging_setup import get_logger

logger = get_logger("capture.video_file")


class VideoFileSource(FrameSource):
    """Replays local video files with real-time FPS pacing and monotonic frame timestamps."""

    def __init__(self, file_path: Path | str, loop: bool = False) -> None:
        self.file_path = Path(file_path).resolve()
        self.loop = loop

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._paused: bool = False
        self._last_frame_time: float = 0.0
        self._frame_interval_s: float = 1.0 / 30.0

        self._metadata = self._inspect_file_metadata()

    def _inspect_file_metadata(self) -> SourceMetadata:
        """Inspect video file without holding handle open."""
        if not self.file_path.exists():
            return SourceMetadata(
                source_type="file",
                source_label=self.file_path.name,
                width=0,
                height=0,
                fps=0.0,
                is_open=False,
            )

        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            return SourceMetadata(
                source_type="file",
                source_label=self.file_path.name,
                width=0,
                height=0,
                fps=0.0,
                is_open=False,
            )

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 and fps > 0 else None
        cap.release()

        return SourceMetadata(
            source_type="file",
            source_label=self.file_path.name,
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            is_open=False,
        )

    def open(self) -> bool:
        """Open video file for frame replay."""
        self.release()
        self._frame_idx = 0

        if not self.file_path.exists():
            logger.error("Video file not found at: %s", self.file_path)
            return False

        logger.info("Opening video file source: %s", self.file_path)
        self._cap = cv2.VideoCapture(str(self.file_path))

        if not self._cap or not self._cap.isOpened():
            logger.error("OpenCV failed to open video file: %s", self.file_path)
            self._metadata.is_open = False
            return False

        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0
        self._frame_interval_s = 1.0 / fps

        self._metadata.is_open = True
        self._last_frame_time = time.monotonic()
        logger.info("Video file %s opened successfully (%s)", self.file_path.name, self._metadata.description())
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Read next video frame with FPS synchronization."""
        if not self.is_opened():
            return False, None, time.monotonic(), self._frame_idx

        if self._paused:
            time.sleep(0.02)
            return False, None, time.monotonic(), self._frame_idx

        # Enforce real-time playback pacing
        now = time.monotonic()
        elapsed = now - self._last_frame_time
        sleep_needed = self._frame_interval_s - elapsed
        if sleep_needed > 0.001:
            time.sleep(sleep_needed)

        ret, frame = self._cap.read()
        self._last_frame_time = time.monotonic()
        timestamp_s = self._last_frame_time

        if not ret or frame is None:
            # End of video file reached
            if self.loop:
                logger.info("Video reached EOF; looping to beginning: %s", self.file_path.name)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None, timestamp_s, self._frame_idx
            else:
                logger.info("Video reached EOF: %s (Total frames: %d)", self.file_path.name, self._frame_idx)
                return False, None, timestamp_s, self._frame_idx

        self._frame_idx += 1
        return True, frame, timestamp_s, self._frame_idx

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._last_frame_time = time.monotonic()

    def release(self) -> None:
        """Close video file handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing video file: %s", e)
            self._cap = None
        self._metadata.is_open = False
        self._paused = False
        logger.info("Video file source released: %s", self.file_path.name)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_metadata(self) -> SourceMetadata:
        return self._metadata

