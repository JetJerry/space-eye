"""Visual overlay rendering utilities for ArUco markers, rig reference frames, and MediaPipe hand tracking."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.vision.aruco_tracker import ArucoDetectionResult, DetectedMarker, RigTrackingState
from vyom.vision.hand_tracker import (
    HAND_CONNECTIONS,
    HandTrackingResult,
    TrackedHand,
)
from vyom.vision.obb_detector import OBBDetection, OBBFrameDetections

# High-contrast mission control overlay colors (BGR format)
COLOR_MARKER_BORDER = (0, 230, 115)       # Green (BGR: 115, 230, 0)
COLOR_MARKER_ORIGIN = (255, 140, 0)       # Cyan/Blue (BGR: 0, 140, 255)
COLOR_CORNER_ORIGIN = (0, 0, 255)         # Red top-left corner indicator
COLOR_TEXT_BG = (20, 26, 36)              # Dark slate background
COLOR_TEXT_FG = (240, 245, 250)           # Off-white text

# MediaPipe Hand Landmark Anatomy Colors (BGR format)
COLOR_PALM = (180, 160, 120)              # Soft slate / light steel
COLOR_THUMB = (0, 165, 255)               # Amber / Orange
COLOR_INDEX = (255, 220, 0)               # Electric Cyan
COLOR_MIDDLE = (0, 230, 100)              # Vivid Green
COLOR_RING = (0, 215, 255)                # Gold / Yellow
COLOR_PINKY = (220, 60, 220)              # Magenta / Violet
COLOR_TIP_CORE = (255, 255, 255)          # Crisp White Core


def draw_aruco_overlays(
    frame_bgr: np.ndarray,
    detection_result: ArucoDetectionResult,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    axis_length_m: float = 0.04,
) -> np.ndarray:
    """Draw fiducial boundaries, ID tags, corner indicators, and 3D coordinate axes onto video frame."""
    if detection_result.marker_count == 0:
        return frame_bgr

    annotated = frame_bgr.copy()

    for marker in detection_result.markers:
        _draw_single_marker(
            annotated,
            marker,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            axis_length_m=axis_length_m,
        )

    return annotated


def _draw_single_marker(
    frame: np.ndarray,
    marker: DetectedMarker,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    axis_length_m: float = 0.04,
) -> None:
    """Render boundary, center, ID, and 3D coordinate axes for an individual marker."""
    corners_int = marker.corners.astype(np.int32)

    # 1. Draw Closed Boundary Polygon
    border_color = (0, 165, 255) if marker.marker_id == 0 else (0, 230, 115)
    cv2.polylines(frame, [corners_int], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Highlight Top-Left Corner (Orientation indicator)
    tl_x, tl_y = int(marker.corners[0, 0]), int(marker.corners[0, 1])
    cv2.circle(frame, (tl_x, tl_y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # 3. Draw Centroid Crosshair
    cx, cy = int(marker.center[0]), int(marker.center[1])
    cross_sz = 5
    cv2.line(frame, (cx - cross_sz, cy), (cx + cross_sz, cy), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx, cy - cross_sz), (cx, cy + cross_sz), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # 4. Draw Marker ID and Status Label Box
    if marker.marker_id == 0:
        label = "ID: 0 [RIG ORIGIN]"
    else:
        label = f"ID: {marker.marker_id}"

    if marker.rig_position is not None:
        rx, ry, rz = marker.rig_position
        label += f" ({rx*100:+.0f},{ry*100:+.0f},{rz*100:+.0f}cm)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    lx = max(4, tl_x - 10)
    ly = max(th + 6, tl_y - 8)

    cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + baseline), COLOR_TEXT_BG, -1)
    cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + baseline), border_color, 1)
    cv2.putText(frame, label, (lx, ly - 2), font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)

    # 5. Draw 3D Coordinate Frame Axes
    if (
        marker.rvec is not None
        and marker.tvec is not None
        and camera_matrix is not None
        and dist_coeffs is not None
    ):
        try:
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                marker.rvec,
                marker.tvec,
                axis_length_m,
                thickness=2,
            )
        except Exception:
            pass


# ==============================================================================
# MediaPipe Hand Landmark Drawing
# ==============================================================================

def _get_finger_color(idx1: int, idx2: int) -> Tuple[int, int, int]:
    """Return distinct color based on anatomical finger bone."""
    pair = {idx1, idx2}
    if pair.issubset({1, 2, 3, 4}):
        return COLOR_THUMB
    elif pair.issubset({5, 6, 7, 8}):
        return COLOR_INDEX
    elif pair.issubset({9, 10, 11, 12}):
        return COLOR_MIDDLE
    elif pair.issubset({13, 14, 15, 16}):
        return COLOR_RING
    elif pair.issubset({17, 18, 19, 20}):
        return COLOR_PINKY
    return COLOR_PALM


def _get_joint_color(idx: int) -> Tuple[int, int, int]:
    """Return distinct color for joint landmark."""
    if idx in (1, 2, 3, 4):
        return COLOR_THUMB
    elif idx in (5, 6, 7, 8):
        return COLOR_INDEX
    elif idx in (9, 10, 11, 12):
        return COLOR_MIDDLE
    elif idx in (13, 14, 15, 16):
        return COLOR_RING
    elif idx in (17, 18, 19, 20):
        return COLOR_PINKY
    return COLOR_PALM


def draw_hand_overlays(
    frame_bgr: np.ndarray,
    hand_result: Optional[HandTrackingResult],
    draw_connections: bool = True,
    draw_landmarks: bool = True,
    draw_handedness_pill: bool = True,
    draw_bbox: bool = False,
) -> np.ndarray:
    """Draw 21 hand landmarks, finger connections, and handedness badges onto video frame.

    Coexists seamlessly with ArUco overlays without overwriting or interfering.
    Uses identical frame coordinates and dimensions.
    """
    if hand_result is None or not hand_result.has_hands:
        return frame_bgr

    annotated = frame_bgr.copy()

    for hand in hand_result.hands:
        _draw_single_hand(
            annotated,
            hand,
            draw_connections=draw_connections,
            draw_landmarks=draw_landmarks,
            draw_handedness_pill=draw_handedness_pill,
            draw_bbox=draw_bbox,
        )

    return annotated


def _draw_single_hand(
    frame: np.ndarray,
    hand: TrackedHand,
    draw_connections: bool = True,
    draw_landmarks: bool = True,
    draw_handedness_pill: bool = True,
    draw_bbox: bool = False,
) -> None:
    """Render skeletal connections, landmarks, and handedness pill for one hand."""
    h, w = frame.shape[:2]
    lms = hand.landmarks

    # 1. Optional Bounding Box
    if draw_bbox:
        min_x, min_y, max_x, max_y = hand.pixel_bbox
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (80, 100, 120), 1, cv2.LINE_AA)

    # 2. Draw Skeletal Connections (Bones)
    if draw_connections:
        for idx1, idx2 in HAND_CONNECTIONS:
            if idx1 < len(lms) and idx2 < len(lms):
                pt1 = (lms[idx1].pixel_x, lms[idx1].pixel_y)
                pt2 = (lms[idx2].pixel_x, lms[idx2].pixel_y)
                color = _get_finger_color(idx1, idx2)
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # 3. Draw Landmark Points
    if draw_landmarks:
        fingertips = {4, 8, 12, 16, 20}
        for lm in lms:
            pt = (lm.pixel_x, lm.pixel_y)
            color = _get_joint_color(lm.index)

            if lm.index == 0:
                # Wrist anchor: distinct concentric target
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, COLOR_TIP_CORE, -1, cv2.LINE_AA)
            elif lm.index in fingertips:
                # Fingertip: larger indicator with bright center core
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, COLOR_TIP_CORE, -1, cv2.LINE_AA)
            else:
                # Joint: solid circle with thin dark rim
                cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, (20, 20, 20), 1, cv2.LINE_AA)

    # 4. Draw Handedness Label Pill
    if draw_handedness_pill:
        label = f"{hand.handedness} {hand.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position pill below the wrist or below bounding box
        wx, wy = hand.wrist_pixel
        px = int(np.clip(wx - tw // 2, 8, w - tw - 12))
        py = wy + 24
        if py + baseline > h - 8:
            # If wrist is near bottom edge, flip label above wrist
            py = max(th + 8, wy - 16)

        # Draw dark slate pill
        pill_color = COLOR_INDEX if hand.handedness.lower() == "right" else COLOR_THUMB
        cv2.rectangle(frame, (px - 5, py - th - 4), (px + tw + 5, py + baseline + 2), COLOR_TEXT_BG, -1)
        cv2.rectangle(frame, (px - 5, py - th - 4), (px + tw + 5, py + baseline + 2), pill_color, 1)
        cv2.putText(frame, label, (px, py - 1), font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)


# ==============================================================================
# YOLOv8-OBB Oriented Object Detection Drawing
# ==============================================================================

# Distinct high-contrast color palette for all 10 classes (BGR format)
OBB_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "centrifuge_tube": (220, 200, 0),       # Cyan / Aqua
    "tube_rack": (0, 140, 255),             # Warm Orange / Amber
    "rack_hole": (255, 120, 160),           # Lavender / Slate Blue
    "tube_cap": (200, 50, 255),             # Vivid Magenta / Pink
    "pipette": (0, 230, 180),               # Neon Mint / Spring Green
    "beaker": (255, 100, 50),               # Bright Royal Blue
    "conical_flask": (50, 200, 80),         # Emerald / Grass Green
    "sample_vial": (140, 80, 255),          # Coral Pink (Provisional)
    "multiwell_plate": (80, 255, 120),      # Acid Lime (Provisional)
    "rig_slot": (0, 215, 255),              # Gold / Sunflower Yellow (Provisional)
}

DEFAULT_OBB_COLOR: Tuple[int, int, int] = (0, 230, 115)


def draw_obb_overlays(
    frame_bgr: np.ndarray,
    obb_result: Optional[OBBFrameDetections],
    draw_labels: bool = True,
    draw_centroid: bool = True,
    alpha_fill: float = 0.16,
) -> np.ndarray:
    """Draw oriented bounding box polygons, confidence tags, and centroids onto video frame.

    Coexists seamlessly with ArUco markers and MediaPipe hand tracking overlays.
    Preserves frame resolution and coordinate alignments.
    """
    if obb_result is None or not obb_result.has_detections:
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Optional semi-transparent polygon fill layer
    if alpha_fill > 0.0:
        overlay = annotated.copy()
        for det in obb_result.detections:
            color = OBB_CLASS_COLORS.get(det.class_name, DEFAULT_OBB_COLOR)
            corners = det.corners_int
            cv2.fillPoly(overlay, [corners], color)
        cv2.addWeighted(overlay, alpha_fill, annotated, 1.0 - alpha_fill, 0, annotated)

    # Crisp boundary lines, centroids, and labels
    for det in obb_result.detections:
        color = OBB_CLASS_COLORS.get(det.class_name, DEFAULT_OBB_COLOR)
        corners = det.corners_int

        # 1. Closed Oriented Polygon Boundary
        cv2.polylines(annotated, [corners], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # 2. Centroid Target
        if draw_centroid:
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(annotated, (cx, cy), 3, color, -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Label Pill
        if draw_labels:
            prov_tag = "*" if det.is_provisional else ""
            label = f"{det.class_name}{prov_tag} {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Anchor pill near top-most corner to ensure readability
            top_corner_idx = int(np.argmin(corners[:, 1]))
            lx = int(np.clip(corners[top_corner_idx, 0] - tw // 2, 4, w - tw - 8))
            ly = int(corners[top_corner_idx, 1] - 6)
            if ly - th - 4 < 4:
                # If too close to top edge, flip below corner
                ly = int(corners[top_corner_idx, 1] + th + 14)

            cv2.rectangle(annotated, (lx - 4, ly - th - 3), (lx + tw + 4, ly + baseline), COLOR_TEXT_BG, -1)
            cv2.rectangle(annotated, (lx - 4, ly - th - 3), (lx + tw + 4, ly + baseline), color, 1)
            cv2.putText(annotated, label, (lx, ly - 1), font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)

    return annotated

