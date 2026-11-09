"""Abstract interface and metadata container for video frame sources."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class SourceMetadata:
    """Detailed metadata for an active or configured video source."""
    source_type: str  # "webcam", "file", "network"
    source_label: str
    width: int
    height: int
    fps: float
    total_frames: Optional[int] = None
    duration_seconds: Optional[float] = None
    is_open: bool = False

    def description(self) -> str:
        """Concise one-line description of the source parameters."""
        dims = f"{self.width}x{self.height} @ {self.fps:.1f} FPS"
        if self.duration_seconds is not None:
            mins = int(self.duration_seconds // 60)
            secs = int(self.duration_seconds % 60)
            return f"{self.source_label} ({dims}, {mins:02d}:{secs:02d})"
        return f"{self.source_label} ({dims})"


class FrameSource(ABC):
    """Abstract base class for thread-safe video frame acquisition."""

    @abstractmethod
    def open(self) -> bool:
        """Initialize and open the underlying video capture device or file."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Acquire the next video frame.

        Returns:
            Tuple of:
            - success: bool (False if stream finished, camera disconnected, or read failed)
            - frame: Optional[np.ndarray] (BGR format uint8 array)
            - monotonic_timestamp_s: float (time.monotonic() timestamp for MediaPipe tracking)
            - frame_idx: int (Monotonically increasing sequence number)
        """

    @abstractmethod
    def pause(self) -> None:
        """Pause frame acquisition without releasing resources."""

    @abstractmethod
    def resume(self) -> None:
        """Resume frame acquisition."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying OpenCV capture handles and system resources."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Check whether the underlying capture source is currently open and valid."""

    @abstractmethod
    def get_metadata(self) -> SourceMetadata:
        """Retrieve current stream/file metadata."""
