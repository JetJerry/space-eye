"""Local session video recording utility using OpenCV VideoWriter."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np

from vyom.core.logging_setup import get_logger

logger = get_logger("capture.recorder")


class SessionRecorder:
    """Thread-safe session video recorder generating unique MP4 files with metadata tracking."""

    def __init__(
        self,
        output_dir: Path | str,
        session_id: str,
        width: int,
        height: int,
        fps: float = 30.0,
        preferred_codec: str = "mp4v",
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.session_id = session_id
        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 30.0
        self.preferred_codec = preferred_codec

        self._writer: Optional[cv2.VideoWriter] = None
        self._output_file: Optional[Path] = None
        self._frame_count: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        """Check whether the recorder is currently active."""
        return self._writer is not None and self._writer.isOpened()

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_file

    def start(self) -> bool:
        """Create output file and initialize VideoWriter."""
        self.release()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique timestamped filename preventing any accidental overwriting
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.output_dir / f"session_{self.session_id}_{timestamp_slug}.mp4"

        counter = 1
        while candidate.exists():
            candidate = self.output_dir / f"session_{self.session_id}_{timestamp_slug}_{counter}.mp4"
            counter += 1

        self._output_file = candidate

        # FourCC codec configuration
        codecs_to_try = [
            (self.preferred_codec, cv2.VideoWriter_fourcc(*self.preferred_codec)),
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
            ("XVID", cv2.VideoWriter_fourcc(*"XVID")),
            ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
        ]

        logger.info(
            "Initializing video recording: %s (%dx%d @ %.1f FPS)",
            self._output_file.name,
            self.width,
            self.height,
            self.fps,
        )

        for codec_name, fourcc in codecs_to_try:
            try:
                self._writer = cv2.VideoWriter(
                    str(self._output_file),
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    isColor=True,
                )
                if self._writer and self._writer.isOpened():
                    logger.info("VideoWriter opened with codec '%s'", codec_name)
                    break
            except Exception as e:
                logger.warning("Failed VideoWriter with codec '%s': %s", codec_name, e)

        if not self._writer or not self._writer.isOpened():
            logger.error("Failed to initialize any OpenCV VideoWriter codec for %s", self._output_file)
            self._writer = None
            self._output_file = None
            return False

        self._frame_count = 0
        self._start_time = time.time()
        return True

    def write_frame(self, frame_bgr: np.ndarray) -> bool:
        """Write single BGR video frame to disk."""
        if not self.is_recording or self._writer is None:
            return False

        try:
            # Enforce exact dimensions
            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

            self._writer.write(frame_bgr)
            self._frame_count += 1
            return True
        except Exception as e:
            logger.error("Error writing video frame %d: %s", self._frame_count, e)
            return False

    def release(self) -> Dict[str, Any]:
        """Finalize video file, release resources, and return summary statistics."""
        summary: Dict[str, Any] = {
            "output_file": str(self._output_file) if self._output_file else None,
            "total_frames": self._frame_count,
            "duration_seconds": 0.0,
            "file_size_bytes": 0,
            "success": False,
        }

        if self._writer is not None:
            self._end_time = time.time()
            try:
                self._writer.release()
            except Exception as e:
                logger.warning("Error releasing VideoWriter: %s", e)
            self._writer = None

            if self._start_time and self._end_time:
                summary["duration_seconds"] = round(self._end_time - self._start_time, 2)

            if self._output_file and self._output_file.exists():
                summary["file_size_bytes"] = self._output_file.stat().st_size
                summary["success"] = True
                logger.info(
                    "Recording finalized: %s (%d frames, %.1fs, %.2f MB)",
                    self._output_file.name,
                    self._frame_count,
                    summary["duration_seconds"],
                    summary["file_size_bytes"] / (1024 * 1024),
                )

        self._frame_count = 0
        self._start_time = None
        self._end_time = None
        self._output_file = None
        return summary

