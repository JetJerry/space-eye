"""Computer vision perception pipeline, fiducial tracking, and hand tracking package."""

from vyom.vision.aruco_tracker import (
    ArucoDetectionResult,
    ArucoTracker,
    DetectedMarker,
    RigTrackingState,
)
from vyom.vision.calibration import (
    CameraCalibration,
    CheckerboardCalibrator,
    RigCoordinateSystem,
)
from vyom.vision.drawing import (
    OBB_CLASS_COLORS,
    draw_aruco_overlays,
    draw_hand_overlays,
    draw_obb_overlays,
)
from vyom.vision.hand_tracker import (
    HAND_CONNECTIONS,
    HandLandmark,
    HandLandmarkIndex,
    HandTracker,
    HandTrackerConfig,
    HandTrackingResult,
    HandTrackingStatus,
    TrackedHand,
)
from vyom.vision.hand_worker import HandTrackingWorker
from vyom.vision.obb_detector import (
    FINAL_CLASSES,
    OBBDetectorConfig,
    OBBDetectorStatus,
    OBBDetection,
    OBBFrameDetections,
    PROVISIONAL_SYNTHETIC_CLASSES,
    REAL_WORLD_CLASSES,
    YOLOOBBDetector,
)
from vyom.vision.obb_worker import OBBTrackingWorker
from vyom.vision.observations import (
    ObjectObservation,
    PerceptionFrameObservation,
)

__all__ = [
    "ArucoDetectionResult",
    "ArucoTracker",
    "CameraCalibration",
    "CheckerboardCalibrator",
    "DetectedMarker",
    "FINAL_CLASSES",
    "HAND_CONNECTIONS",
    "HandLandmark",
    "HandLandmarkIndex",
    "HandTracker",
    "HandTrackerConfig",
    "HandTrackingResult",
    "HandTrackingStatus",
    "HandTrackingWorker",
    "OBBDetectorConfig",
    "OBBDetectorStatus",
    "OBBDetection",
    "OBBFrameDetections",
    "OBBTrackingWorker",
    "OBB_CLASS_COLORS",
    "ObjectObservation",
    "PerceptionFrameObservation",
    "PROVISIONAL_SYNTHETIC_CLASSES",
    "REAL_WORLD_CLASSES",
    "RigCoordinateSystem",
    "RigTrackingState",
    "TrackedHand",
    "YOLOOBBDetector",
    "draw_aruco_overlays",
    "draw_hand_overlays",
    "draw_obb_overlays",
]
