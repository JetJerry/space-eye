"""Perception observation interface uniting ArUco, MediaPipe, and YOLOv8-OBB detections.

Provides explicit, typed structured observation data structures ready to be consumed
by Prompt 8 (Spatial Association & Hand-Object Interaction Recognition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vyom.vision.aruco_tracker import ArucoDetectionResult, DetectedMarker, RigTrackingState
from vyom.vision.hand_tracker import HandTrackingResult, TrackedHand
from vyom.vision.obb_detector import OBBDetection, OBBFrameDetections


@dataclass
class ObjectObservation:
    """Discrete single-object perception observation for spatial interaction reasoning."""
    frame_id: int
    timestamp_s: float
    object_id: str             # e.g., "pipette_0", "beaker_1"
    class_id: int
    class_name: str
    confidence: float
    corners: np.ndarray        # Shape (4, 2) in frame pixel coordinates
    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float
    is_provisional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "angle_deg": round(self.angle_deg, 2),
            "corners": self.corners.round(2).tolist(),
            "is_provisional": self.is_provisional,
        }


@dataclass
class PerceptionFrameObservation:
    """Unified synchronized multi-modal observation container for a single frame.

    Preserves ArUco fiducial rig reference, MediaPipe hand landmarks, and YOLOv8-OBB
    object detections for spatial association in Prompt 8.
    """
    frame_id: int
    timestamp_s: float
    aruco_result: Optional[ArucoDetectionResult] = None
    hand_result: Optional[HandTrackingResult] = None
    obb_result: Optional[OBBFrameDetections] = None

    @property
    def has_rig(self) -> bool:
        return self.aruco_result is not None and self.aruco_result.state == RigTrackingState.RIG_TRACKING

    @property
    def has_hands(self) -> bool:
        return self.hand_result is not None and self.hand_result.has_hands

    @property
    def has_objects(self) -> bool:
        return self.obb_result is not None and self.obb_result.has_detections

    def get_objects(self) -> List[ObjectObservation]:
        """Convert frame OBB detections to typed ObjectObservations with distinct object IDs."""
        if self.obb_result is None or not self.obb_result.has_detections:
            return []

        counts: Dict[str, int] = {}
        observations: List[ObjectObservation] = []
        for det in self.obb_result.detections:
            cname = det.class_name
            idx = counts.get(cname, 0)
            counts[cname] = idx + 1
            obj_id = f"{cname}_{idx}"

            obs = ObjectObservation(
                frame_id=self.frame_id,
                timestamp_s=self.timestamp_s,
                object_id=obj_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                corners=det.corners,
                center=det.center,
                width=det.width,
                height=det.height,
                angle_deg=det.angle_deg,
                is_provisional=det.is_provisional,
            )
            observations.append(obs)
        return observations

    def get_hands(self) -> List[TrackedHand]:
        """Retrieve tracked hands."""
        if self.hand_result is None:
            return []
        return self.hand_result.hands

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete multi-modal frame observation."""
        return {
            "frame_id": self.frame_id,
            "timestamp_s": round(self.timestamp_s, 4),
            "aruco": {
                "state": self.aruco_result.state.value if self.aruco_result else "NONE",
                "marker_count": self.aruco_result.marker_count if self.aruco_result else 0,
            },
            "hands": {
                "count": self.hand_result.hand_count if self.hand_result else 0,
                "hands": [h.handedness for h in self.hand_result.hands] if self.hand_result else [],
            },
            "objects": {
                "count": self.obb_result.count if self.obb_result else 0,
                "device": self.obb_result.device_used if self.obb_result else "NONE",
                "inference_ms": self.obb_result.inference_time_ms if self.obb_result else 0.0,
                "items": [obs.to_dict() for obs in self.get_objects()],
            },
        }

