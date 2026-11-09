"""MediaPipe Tasks Hand Landmarker engine, typed observations, and coordinate extraction.

Offline-first hand tracking system using Google's pretrained Hand Landmarker task bundle.
Extracts 21 3D landmarks per hand with handedness (Left/Right) and confidence scores.

COORDINATE SYSTEM SPECIFICATION & CAUTION:
- Landmark image coordinates (x, y, z) are normalized to [0.0, 1.0] relative to frame dimensions.
- Pixel coordinates (pixel_x, pixel_y) are computed in frame resolution space.
- MediaPipe hand-relative world landmarks (world_x, world_y, world_z) represent 3D metric
  coordinates (in METERS) relative to the hand's geometric center.
- CRITICAL: These hand-relative coordinates MUST NOT be treated as global rig coordinates.
  They are strictly hand-relative and cannot be combined directly with ArUco rig coordinates
  without extrinsic camera calibration and spatial transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.core.logging_setup import get_logger

logger = get_logger("vision.hand_tracker")


class HandLandmarkIndex(IntEnum):
    """Standard 21 MediaPipe hand landmark anatomical indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Skeletal connections between landmarks for visualization and structural verification
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Palm base & boundary
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20),
]


class HandTrackingStatus(str, Enum):
    """Standardized operational statuses for hand tracking."""
    NOT_INITIALIZED = "HAND TRACKING: NOT INITIALIZED"
    MODEL_MISSING = "HAND TRACKING: MODEL MISSING"
    SEARCHING = "HAND TRACKING: SEARCHING"
    HAND_DETECTED = "HAND TRACKING: HAND DETECTED"
    NO_HAND_DETECTED = "HAND TRACKING: NO HAND DETECTED"
    ERROR = "HAND TRACKING: ERROR"


@dataclass
class HandLandmark:
    """Single hand landmark observation in normalized, pixel, and hand-relative 3D spaces."""
    index: int
    name: str
    x: float             # Normalized image coordinate [0.0, 1.0]
    y: float             # Normalized image coordinate [0.0, 1.0]
    z: float             # Normalized depth relative to wrist
    pixel_x: int         # Discrete horizontal pixel coordinate in frame
    pixel_y: int         # Discrete vertical pixel coordinate in frame
    # MediaPipe Hand-Relative World Coordinates (in METERS, origin at hand geometric center)
    # Caution: These are NOT rig coordinates!
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None
    visibility: Optional[float] = None
    presence: Optional[float] = None


@dataclass
class TrackedHand:
    """Structured observation of a single human hand with all 21 landmarks."""
    hand_index: int
    handedness: str                     # "Left" or "Right"
    confidence: float                   # Handedness classification score [0.0, 1.0]
    landmarks: List[HandLandmark]       # Exactly 21 landmarks
    pixel_bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    wrist_pixel: Tuple[int, int]
    index_tip_pixel: Tuple[int, int]
    thumb_tip_pixel: Tuple[int, int]

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)

    def get_landmark(self, index: HandLandmarkIndex | int) -> Optional[HandLandmark]:
        """Lookup landmark by index."""
        idx = int(index)
        if 0 <= idx < len(self.landmarks):
            return self.landmarks[idx]
        return None


@dataclass
class HandTrackingResult:
    """Complete typed result emitted for each processed frame."""
    status: HandTrackingStatus
    hands: List[TrackedHand]
    hand_count: int
    frame_idx: int
    timestamp_ms: int
    latency_ms: float
    image_width: int
    image_height: int
    error_message: Optional[str] = None

    @property
    def has_hands(self) -> bool:
        return self.hand_count > 0

    @property
    def summary(self) -> str:
        """Concise summary string for UI logs."""
        if not self.has_hands:
            return self.status.value
        labels = [f"{h.handedness} ({h.confidence:.0%})" for h in self.hands]
        return f"{self.hand_count} hand(s): {', '.join(labels)} | {self.latency_ms:.1f}ms"


@dataclass
class HandTrackerConfig:
    """Configuration parameters for MediaPipe Hand Landmarker."""
    model_path: Path | str = "models/hand_landmarker.task"
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    running_mode: str = "VIDEO"  # "VIDEO", "IMAGE", or "LIVE_STREAM"
    delegate: str = "CPU"        # CPU or GPU if supported by platform


class HandTracker:
    """Encapsulates MediaPipe Tasks Hand Landmarker with offline validation and typed output."""

    def __init__(self, config: Optional[HandTrackerConfig] = None) -> None:
        self.config = config or HandTrackerConfig()
        self._landmarker: Any = None
        self._is_initialized: bool = False
        self._model_resolved_path: Optional[Path] = None
        self._last_timestamp_ms: int = -1

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> Tuple[bool, HandTrackingStatus, Optional[str]]:
        """Validate offline model asset and instantiate MediaPipe Hand Landmarker."""
        target_path = Path(self.config.model_path).resolve()

        if not target_path.exists() or target_path.stat().st_size == 0:
            msg = (
                f"Hand landmarker model missing at: {target_path}. "
                "Download official bundle from Google: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
            logger.warning(msg)
            self._is_initialized = False
            return False, HandTrackingStatus.MODEL_MISSING, msg

        self._model_resolved_path = target_path

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(target_path))

            # Select running mode
            mode_map = {
                "IMAGE": vision.RunningMode.IMAGE,
                "VIDEO": vision.RunningMode.VIDEO,
                "LIVE_STREAM": vision.RunningMode.LIVE_STREAM,
            }
            running_mode = mode_map.get(self.config.running_mode.upper(), vision.RunningMode.VIDEO)

            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=self.config.num_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )

            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            self._last_timestamp_ms = -1
            logger.info(
                "Initialized MediaPipe HandLandmarker (mode=%s, max_hands=%d, model=%s)",
                self.config.running_mode,
                self.config.num_hands,
                target_path.name,
            )
            return True, HandTrackingStatus.SEARCHING, None

        except Exception as e:
            err = f"Failed to initialize MediaPipe HandLandmarker: {e}"
            logger.error(err, exc_info=True)
            self._is_initialized = False
            return False, HandTrackingStatus.ERROR, err

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        frame_idx: int = 0,
    ) -> HandTrackingResult:
        """Process incoming BGR video frame and extract typed 21 landmarks per hand.

        Guarantees strictly increasing millisecond timestamps for VIDEO mode.
        Returns typed HandTrackingResult without fabricating landmarks if model missing or no hands found.
        """
        h, w = frame_bgr.shape[:2]

        if not self._is_initialized or self._landmarker is None:
            # Check if model file is simply missing
            status = HandTrackingStatus.MODEL_MISSING if self._model_resolved_path is None else HandTrackingStatus.NOT_INITIALIZED
            return HandTrackingResult(
                status=status,
                hands=[],
                hand_count=0,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                latency_ms=0.0,
                image_width=w,
                image_height=h,
                error_message="MediaPipe HandLandmarker is not initialized",
            )

        # Enforce strictly monotonic timestamp for MediaPipe Tasks VIDEO mode
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        t0 = time.perf_counter()

        try:
            import mediapipe as mp

            # Convert BGR (OpenCV) -> RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform detection according to configured mode
            if self.config.running_mode.upper() == "IMAGE":
                detection_result = self._landmarker.detect(mp_image)
            else:
                detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Extract structured observations
            hands: List[TrackedHand] = []
            raw_landmarks = getattr(detection_result, "hand_landmarks", []) or []
            raw_world_landmarks = getattr(detection_result, "hand_world_landmarks", []) or []
            raw_handedness = getattr(detection_result, "handedness", []) or []

            for i, hand_lms in enumerate(raw_landmarks):
                # Extract handedness label & score
                handedness_label = "Unknown"
                confidence = 0.0
                if i < len(raw_handedness) and raw_handedness[i]:
                    cat = raw_handedness[i][0]
                    handedness_label = cat.category_name or "Unknown"
                    confidence = float(cat.score)

                world_lms = raw_world_landmarks[i] if i < len(raw_world_landmarks) else None

                parsed_landmarks: List[HandLandmark] = []
                xs: List[int] = []
                ys: List[int] = []

                for idx, lm in enumerate(hand_lms):
                    px = int(np.clip(lm.x * w, 0, w - 1))
                    py = int(np.clip(lm.y * h, 0, h - 1))
                    xs.append(px)
                    ys.append(py)

                    try:
                        name = HandLandmarkIndex(idx).name
                    except ValueError:
                        name = f"LANDMARK_{idx}"

                    wx = world_lms[idx].x if world_lms and idx < len(world_lms) else None
                    wy = world_lms[idx].y if world_lms and idx < len(world_lms) else None
                    wz = world_lms[idx].z if world_lms and idx < len(world_lms) else None

                    parsed_landmarks.append(
                        HandLandmark(
                            index=idx,
                            name=name,
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                            pixel_x=px,
                            pixel_y=py,
                            world_x=wx,
                            world_y=wy,
                            world_z=wz,
                            visibility=getattr(lm, "visibility", None),
                            presence=getattr(lm, "presence", None),
                        )
                    )

                # Compute bounding box with padding
                min_x = max(0, min(xs) - 10)
                min_y = max(0, min(ys) - 10)
                max_x = min(w - 1, max(xs) + 10)
                max_y = min(h - 1, max(ys) + 10)

                wrist_px = (xs[0], ys[0])
                index_tip_px = (xs[8], ys[8]) if len(xs) > 8 else (0, 0)
                thumb_tip_px = (xs[4], ys[4]) if len(xs) > 4 else (0, 0)

                hands.append(
                    TrackedHand(
                        hand_index=i,
                        handedness=handedness_label,
                        confidence=confidence,
                        landmarks=parsed_landmarks,
                        pixel_bbox=(min_x, min_y, max_x, max_y),
                        wrist_pixel=wrist_px,
                        index_tip_pixel=index_tip_px,
                        thumb_tip_pixel=thumb_tip_px,
                    )
                )

            status = HandTrackingStatus.HAND_DETECTED if hands else HandTrackingStatus.NO_HAND_DETECTED

            return HandTrackingResult(
                status=status,
                hands=hands,
                hand_count=len(hands),
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                latency_ms=latency_ms,
                image_width=w,
                image_height=h,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Error during HandLandmarker processing on frame %d: %s", frame_idx, e)
            return HandTrackingResult(
                status=HandTrackingStatus.ERROR,
                hands=[],
                hand_count=0,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                latency_ms=latency_ms,
                image_width=w,
                image_height=h,
                error_message=str(e),
            )

    def reset(self) -> None:
        """Reset internal timestamp state between video sessions."""
        self._last_timestamp_ms = -1

    def close(self) -> None:
        """Clean up MediaPipe resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
                logger.info("Closed MediaPipe HandLandmarker")
            except Exception as e:
                logger.warning("Error closing HandLandmarker: %s", e)
            finally:
                self._landmarker = None
                self._is_initialized = False

