"""Computer vision perception pipeline and fiducial tracking package."""

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
from vyom.vision.drawing import draw_aruco_overlays

__all__ = [
    "ArucoDetectionResult",
    "ArucoTracker",
    "CameraCalibration",
    "CheckerboardCalibrator",
    "DetectedMarker",
    "RigCoordinateSystem",
    "RigTrackingState",
    "draw_aruco_overlays",
]
