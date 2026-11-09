"""ArUco fiducial marker detector and experimental rig reference tracking engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.core.config import get_config_manager
from vyom.core.logging_setup import get_logger
from vyom.core.schemas import RigConfig
from vyom.vision.calibration import CameraCalibration, RigCoordinateSystem

logger = get_logger("vision.aruco")


# Mapping of dictionary name strings to OpenCV ArUco dictionary constants
ARUCO_DICTIONARY_MAP: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class RigTrackingState(str, Enum):
    """High-level tracking status of the ArUco fiducial system."""
    NOT_INITIALIZED = "NOT_INITIALIZED"
    SEARCHING = "SEARCHING"
    MARKERS_DETECTED = "MARKERS_DETECTED"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    RIG_TRACKING = "RIG_TRACKING"


@dataclass
class DetectedMarker:
    """Individual ArUco marker observation in a video frame."""
    marker_id: int
    corners: np.ndarray        # Shape (4, 2) in image pixel coordinates
    center: Tuple[float, float] # Centroid (cx, cy) in pixels
    perimeter: float
    is_configured_rig_marker: bool = True
    rvec: Optional[np.ndarray] = None  # 3x1 rotation vector in camera frame
    tvec: Optional[np.ndarray] = None  # 3x1 translation vector in camera frame (meters)
    rig_position: Optional[Tuple[float, float, float]] = None  # (X, Y, Z) in rig frame (meters)

    @property
    def top_left(self) -> Tuple[float, float]:
        return float(self.corners[0, 0]), float(self.corners[0, 1])

    @property
    def top_right(self) -> Tuple[float, float]:
        return float(self.corners[1, 0]), float(self.corners[1, 1])

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return float(self.corners[2, 0]), float(self.corners[2, 1])

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return float(self.corners[3, 0]), float(self.corners[3, 1])


@dataclass
class ArucoDetectionResult:
    """Aggregated detection result across a single frame."""
    frame_timestamp_s: float
    markers: List[DetectedMarker] = field(default_factory=list)
    state: RigTrackingState = RigTrackingState.SEARCHING
    configured_ids_visible: List[int] = field(default_factory=list)
    origin_marker_visible: bool = False
    is_calibrated: bool = False
    detection_time_ms: float = 0.0

    @property
    def marker_count(self) -> int:
        return len(self.markers)

    @property
    def status_summary(self) -> str:
        """Concise one-line summary string for UI badges."""
        if self.state == RigTrackingState.RIG_TRACKING:
            return f"ARUCO: RIG TRACKING ({len(self.configured_ids_visible)} visible)"
        elif self.state == RigTrackingState.CALIBRATION_REQUIRED:
            return f"ARUCO: {len(self.configured_ids_visible)} DETECTED (CALIBRATION REQUIRED)"
        elif self.state == RigTrackingState.MARKERS_DETECTED:
            return f"ARUCO: {len(self.configured_ids_visible)} DETECTED (SEARCHING ORIGIN)"
        elif self.state == RigTrackingState.SEARCHING:
            return "ARUCO: SEARCHING"
        return "ARUCO: NOT INITIALIZED"


class ArucoTracker:
    """Detects configured ArUco fiducials and derives rig-relative coordinates."""

    def __init__(
        self,
        rig_config: Optional[RigConfig] = None,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        cfg = rig_config or get_config_manager().config.rig
        self.rig_config = cfg
        self.dictionary_name = cfg.dictionary_name
        self.configured_marker_ids = set(cfg.marker_ids)
        self.origin_marker_id = cfg.origin_marker_id
        self.marker_size_m = cfg.marker_size_m

        # Load calibration if specified in config and not passed explicitly
        if calibration is None and cfg.camera_calibration_file:
            calibration = CameraCalibration.load_from_file(cfg.camera_calibration_file)

        self.rig_coords = RigCoordinateSystem(
            origin_marker_id=self.origin_marker_id,
            marker_size_m=self.marker_size_m,
            calibration=calibration,
        )

        # Initialize OpenCV ArUco detector
        self._init_detector()

        logger.info(
            "ArucoTracker initialized: Dictionary=%s, Target IDs=%s, Marker Size=%.1fmm, Calibrated=%s",
            self.dictionary_name,
            list(self.configured_marker_ids),
            self.marker_size_m * 1000,
            self.rig_coords.is_calibrated,
        )

    def _init_detector(self) -> None:
        """Instantiate OpenCV dictionary and ArucoDetector."""
        dict_id = ARUCO_DICTIONARY_MAP.get(self.dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()

        # Optimize detection parameters for microgravity / desktop lab conditions
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def set_calibration(self, calibration: Optional[CameraCalibration]) -> None:
        """Update active camera calibration."""
        self.rig_coords.calibration = calibration

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float = 0.0) -> ArucoDetectionResult:
        """Process incoming video frame and extract ArUco markers and rig reference."""
        t0 = time.perf_counter()

        # Convert to grayscale for contrast extraction
        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        # Execute marker detection
        corners_list, ids_list, _ = self.detector.detectMarkers(gray)

        detection_ms = (time.perf_counter() - t0) * 1000.0

        result = ArucoDetectionResult(
            frame_timestamp_s=timestamp_s,
            is_calibrated=self.rig_coords.is_calibrated,
            detection_time_ms=detection_ms,
        )

        if ids_list is None or len(ids_list) == 0:
            result.state = RigTrackingState.SEARCHING
            self.rig_coords.clear_origin_pose()
            return result

        flattened_ids = ids_list.flatten()
        detected_markers: List[DetectedMarker] = []
        origin_marker_entry: Optional[DetectedMarker] = None

        # 1. Extract geometric corners and centers for detected markers
        for i, mid in enumerate(flattened_ids):
            mid_int = int(mid)
            corners_4x2 = corners_list[i].reshape(4, 2)

            # Centroid calculation
            cx = float(np.mean(corners_4x2[:, 0]))
            cy = float(np.mean(corners_4x2[:, 1]))

            # Perimeter calculation
            perim = float(cv2.arcLength(corners_4x2.reshape(-1, 1, 2).astype(np.float32), True))

            is_configured = mid_int in self.configured_marker_ids

            marker = DetectedMarker(
                marker_id=mid_int,
                corners=corners_4x2,
                center=(cx, cy),
                perimeter=perim,
                is_configured_rig_marker=is_configured,
            )

            # 2. Pose Estimation (when calibrated)
            if self.rig_coords.is_calibrated:
                ok, rvec, tvec = self.rig_coords.estimate_marker_pose(corners_4x2)
                if ok and rvec is not None and tvec is not None:
                    marker.rvec = rvec
                    marker.tvec = tvec

            if is_configured:
                result.configured_ids_visible.append(mid_int)
                if mid_int == self.origin_marker_id:
                    origin_marker_entry = marker

            detected_markers.append(marker)

        result.markers = detected_markers

        # 3. Update Rig Reference Frame State
        if origin_marker_entry is not None:
            result.origin_marker_visible = True
            if origin_marker_entry.rvec is not None and origin_marker_entry.tvec is not None:
                self.rig_coords.update_origin_pose(origin_marker_entry.rvec, origin_marker_entry.tvec)
        else:
            self.rig_coords.clear_origin_pose()

        # 4. Compute 3D Rig-Relative Positions for all markers (if rig tracking active)
        if self.rig_coords.is_tracking_rig:
            for m in detected_markers:
                if m.tvec is not None:
                    rig_pt = self.rig_coords.transform_camera_to_rig(m.tvec)
                    if rig_pt is not None:
                        m.rig_position = (float(rig_pt[0]), float(rig_pt[1]), float(rig_pt[2]))

        # 5. Determine High-Level System State
        if len(result.configured_ids_visible) == 0:
            result.state = RigTrackingState.SEARCHING
        elif not self.rig_coords.is_calibrated:
            result.state = RigTrackingState.CALIBRATION_REQUIRED
        elif self.rig_coords.is_tracking_rig:
            result.state = RigTrackingState.RIG_TRACKING
        else:
            result.state = RigTrackingState.MARKERS_DETECTED

        return result

