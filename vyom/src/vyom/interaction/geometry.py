"""Spatial geometry calculations, distance metrics, and coordinate system boundaries.

CRITICAL INVARIANT:
- MediaPipe hand-relative world coordinates (in meters relative to the hand center)
  MUST NEVER be combined directly with ArUco global rig coordinates.
- If camera calibration is unavailable, all spatial proximity and contact reasoning
  operates strictly in image pixel space with explicit spatial uncertainty annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.vision.hand_tracker import HandLandmarkIndex, TrackedHand


@dataclass
class SpatialProximityMetrics:
    """Quantitative geometric measurements between a single hand and an oriented bounding box."""
    hand_id: str
    handedness: str
    min_fingertip_distance_px: float       # Euclidean distance from nearest fingertip to OBB polygon
    closest_fingertip_index: int           # Anatomical index (4=Thumb, 8=Index, 12=Middle)
    closest_fingertip_name: str
    is_any_fingertip_inside: bool          # True if at least one fingertip is geometrically inside OBB
    fingertips_inside_count: int
    palm_to_center_distance_px: float      # Distance between palm centroid and OBB center
    approach_vector_px: Tuple[float, float] # Vector from palm center to object center (dx, dy)
    contact_candidate: bool                # True if fingertips penetrate or touch OBB boundary
    coordinate_system: str
    spatial_uncertainty: str


# Primary grasping fingertips: Thumb (4), Index (8), Middle (12)
PRIMARY_GRASP_FINGERTIPS: List[Tuple[int, str]] = [
    (HandLandmarkIndex.THUMB_TIP, "Thumb"),
    (HandLandmarkIndex.INDEX_FINGER_TIP, "Index"),
    (HandLandmarkIndex.MIDDLE_FINGER_TIP, "Middle"),
    (HandLandmarkIndex.RING_FINGER_TIP, "Ring"),
    (HandLandmarkIndex.PINKY_TIP, "Pinky"),
]


def compute_palm_center(hand: TrackedHand) -> Tuple[float, float]:
    """Compute the 2D geometric center of the palm in pixel coordinates.

    Uses the wrist (0) and MCP base knuckles (5, 9, 13, 17).
    """
    palm_indices = [0, 5, 9, 13, 17]
    xs = []
    ys = []
    for idx in palm_indices:
        lm = hand.get_landmark(idx)
        if lm is not None:
            xs.append(lm.pixel_x)
            ys.append(lm.pixel_y)

    if not xs:
        # Fallback to wrist pixel
        return float(hand.wrist_pixel[0]), float(hand.wrist_pixel[1])

    return float(np.mean(xs)), float(np.mean(ys))


def compute_hand_obb_proximity(
    hand: TrackedHand,
    obb_corners: np.ndarray,
    obb_center: Tuple[float, float],
    contact_threshold_px: float = 12.0,
    has_rig_calibration: bool = False,
) -> SpatialProximityMetrics:
    """Compute exact spatial proximity between hand fingertips/palm and an oriented bounding box.

    Uses cv2.pointPolygonTest for exact signed distance:
    - Positive: inside polygon
    - Zero: exactly on edge
    - Negative: outside polygon (Euclidean distance = |dist|)
    """
    corners_int = np.round(obb_corners).astype(np.int32)
    palm_cx, palm_cy = compute_palm_center(hand)
    obj_cx, obj_cy = obb_center

    # 1. Evaluate distance from each fingertip to OBB polygon
    min_dist_px = float("inf")
    closest_idx = 8
    closest_name = "Index"
    inside_count = 0

    for tip_idx, tip_name in PRIMARY_GRASP_FINGERTIPS:
        lm = hand.get_landmark(tip_idx)
        if lm is None:
            continue

        pt = (float(lm.pixel_x), float(lm.pixel_y))
        signed_dist = cv2.pointPolygonTest(corners_int, pt, measureDist=True)

        if signed_dist >= 0:
            # Point is inside or on edge
            inside_count += 1
            abs_dist = 0.0
        else:
            abs_dist = abs(signed_dist)

        if abs_dist < min_dist_px:
            min_dist_px = abs_dist
            closest_idx = tip_idx
            closest_name = tip_name

    is_any_inside = inside_count > 0
    contact = is_any_inside or (min_dist_px <= contact_threshold_px)

    # 2. Palm to object center distance and vector
    dx = float(obj_cx - palm_cx)
    dy = float(obj_cy - palm_cy)
    palm_to_obj_dist = float(np.hypot(dx, dy))

    # 3. Coordinate system verification
    if has_rig_calibration:
        coord_sys = "RIG_METRIC"
        uncertainty = "CALIBRATED_RIG_METRIC"
    else:
        coord_sys = "IMAGE_PIXEL"
        uncertainty = "UNCALIBRATED_2D (Depth ambiguity present)"

    return SpatialProximityMetrics(
        hand_id=f"hand_{hand.hand_index}",
        handedness=hand.handedness,
        min_fingertip_distance_px=round(min_dist_px, 2),
        closest_fingertip_index=closest_idx,
        closest_fingertip_name=closest_name,
        is_any_fingertip_inside=is_any_inside,
        fingertips_inside_count=inside_count,
        palm_to_center_distance_px=round(palm_to_obj_dist, 2),
        approach_vector_px=(round(dx, 2), round(dy, 2)),
        contact_candidate=contact,
        coordinate_system=coord_sys,
        spatial_uncertainty=uncertainty,
    )


def compute_obb_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Calculate intersection-over-union between two oriented bounding boxes.

    Returns float IoU in range [0.0, 1.0].
    """
    try:
        pts1 = np.round(corners1).astype(np.float32)
        pts2 = np.round(corners2).astype(np.float32)

        area1 = cv2.contourArea(pts1)
        area2 = cv2.contourArea(pts2)

        if area1 <= 1.0 or area2 <= 1.0:
            return 0.0

        # Compute intersection polygon
        ret, inter_pts = cv2.intersectConvexConvex(pts1, pts2)
        if not ret or inter_pts is None or len(inter_pts) < 3:
            return 0.0

        inter_area = cv2.contourArea(inter_pts)
        union_area = area1 + area2 - inter_area
        if union_area <= 0.0:
            return 0.0

        return float(np.clip(inter_area / union_area, 0.0, 1.0))
    except Exception:
        return 0.0

