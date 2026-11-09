"""Visual overlay rendering for hand-object relationships and interaction candidate states."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.interaction.engine import HandObjectRelationship, InteractionFrameResult
from vyom.interaction.events import InteractionState

# Interaction state colors (BGR format)
COLOR_APPROACH = (0, 180, 255)       # Amber / Orange
COLOR_CONTACT = (0, 230, 115)        # Vivid Green
COLOR_HOLDING = (0, 215, 255)        # Pure Gold / Yellow
COLOR_PLACEMENT = (220, 60, 220)     # Magenta / Violet
COLOR_TEXT_BG = (20, 26, 36)
COLOR_TEXT_FG = (240, 245, 250)


def draw_interaction_overlays(
    frame_bgr: np.ndarray,
    interaction_result: Optional[InteractionFrameResult],
    draw_connecting_lines: bool = True,
    draw_badges: bool = True,
) -> np.ndarray:
    """Render dynamic interaction vectors, state pills, and spatial uncertainty tags onto video frame.

    Coexists seamlessly with ArUco, MediaPipe, and YOLOv8-OBB overlays without overwriting.
    """
    if interaction_result is None or not interaction_result.active_relationships:
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Map tracked objects by track_id for fast coordinate lookup
    tracks_by_id = {t.track_id: t for t in interaction_result.tracked_objects}

    # Render active relationships that are in non-IDLE state
    for rel in interaction_result.active_relationships:
        if rel.current_state == InteractionState.IDLE:
            continue

        obj = tracks_by_id.get(rel.object_track_id)
        if obj is None:
            continue

        obj_cx, obj_cy = int(round(obj.center[0])), int(round(obj.center[1]))

        # Select color based on state
        if rel.current_state == InteractionState.APPROACHING:
            color = COLOR_APPROACH
        elif rel.current_state == InteractionState.CONTACT_CANDIDATE:
            color = COLOR_CONTACT
        elif rel.current_state == InteractionState.HOLDING_MOVING:
            color = COLOR_HOLDING
        elif rel.current_state == InteractionState.PLACEMENT_CANDIDATE:
            color = COLOR_PLACEMENT
        else:
            color = (180, 180, 180)

        # 1. Connecting line from hand to object center
        if draw_connecting_lines and rel.last_metrics is not None:
            dx, dy = rel.last_metrics.approach_vector_px
            # Palm center is (obj_center - approach_vector)
            hx = int(round(obj.center[0] - dx))
            hy = int(round(obj.center[1] - dy))

            # Draw dashed or solid association line
            cv2.line(annotated, (hx, hy), (obj_cx, obj_cy), color, 1, cv2.LINE_AA)
            mid_x = (hx + obj_cx) // 2
            mid_y = (hy + obj_cy) // 2
        else:
            mid_x, mid_y = obj_cx, max(20, obj_cy - 20)

        # 2. State pill label
        if draw_badges:
            state_label = f"{rel.handedness} -> {rel.object_track_id}: {rel.current_state.value}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(state_label, font, font_scale, thickness)

            bx = int(np.clip(mid_x - tw // 2, 8, w - tw - 12))
            by = int(np.clip(mid_y - 8, th + 8, h - baseline - 8))

            cv2.rectangle(annotated, (bx - 4, by - th - 3), (bx + tw + 4, by + baseline + 1), COLOR_TEXT_BG, -1)
            cv2.rectangle(annotated, (bx - 4, by - th - 3), (bx + tw + 4, by + baseline + 1), color, 1)
            cv2.putText(annotated, state_label, (bx, by - 1), font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)

    # 3. Spatial calibration uncertainty indicator in top-left banner
    calib_text = "SPATIAL: RIG METRIC CALIBRATED" if interaction_result.has_rig_calibration else "SPATIAL: UNCALIBRATED 2D PROXIMITY"
    calib_color = COLOR_CONTACT if interaction_result.has_rig_calibration else COLOR_APPROACH
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(calib_text, font, 0.36, 1)
    cv2.rectangle(annotated, (8, 8), (14 + tw, 14 + th + baseline), COLOR_TEXT_BG, -1)
    cv2.rectangle(annotated, (8, 8), (14 + tw, 14 + th + baseline), calib_color, 1)
    cv2.putText(annotated, calib_text, (11, 11 + th), font, 0.36, calib_color, 1, cv2.LINE_AA)

    return annotated

