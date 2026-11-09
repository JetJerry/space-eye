"""Visual overlay rendering utility for ArUco markers and rig reference frames."""

from typing import Optional, Tuple
import cv2
import numpy as np

from vyom.vision.aruco_tracker import ArucoDetectionResult, DetectedMarker, RigTrackingState


# High-contrast mission control overlay colors (BGR format)
COLOR_MARKER_BORDER = (0, 230, 115)       # Green (BGR: 115, 230, 0)
COLOR_MARKER_ORIGIN = (255, 140, 0)       # Cyan/Blue (BGR: 0, 140, 255)
COLOR_CORNER_ORIGIN = (0, 0, 255)         # Red top-left corner indicator
COLOR_TEXT_BG = (20, 26, 36)              # Dark slate background
COLOR_TEXT_FG = (240, 245, 250)           # Off-white text


def draw_aruco_overlays(
    frame_bgr: np.ndarray,
    detection_result: ArucoDetectionResult,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    axis_length_m: float = 0.04,
) -> np.ndarray:
    """Draw fiducial boundaries, ID tags, corner indicators, and 3D coordinate axes onto video frame.

    Args:
        frame_bgr: Source video frame (BGR).
        detection_result: Detected markers and rig reference data from ArucoTracker.
        camera_matrix: Optional 3x3 camera matrix for 3D axis projection.
        dist_coeffs: Optional lens distortion coefficients.
        axis_length_m: Physical length of coordinate axes in meters (default: 40 mm).

    Returns:
        Annotated BGR frame (in-place modified or new array).
    """
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
    border_color = (0, 165, 255) if marker.marker_id == 0 else (0, 230, 115)  # Orange for Origin (ID 0), Green for others
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

    # Append 3D metric coordinates if available
    if marker.rig_position is not None:
        rx, ry, rz = marker.rig_position
        label += f" ({rx*100:+.0f},{ry*100:+.0f},{rz*100:+.0f}cm)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Position label above the top-left corner
    lx = max(4, tl_x - 10)
    ly = max(th + 6, tl_y - 8)

    # Background pill
    cv2.rectangle(
        frame,
        (lx - 4, ly - th - 4),
        (lx + tw + 4, ly + baseline),
        COLOR_TEXT_BG,
        -1,
    )
    cv2.rectangle(
        frame,
        (lx - 4, ly - th - 4),
        (lx + tw + 4, ly + baseline),
        border_color,
        1,
    )
    cv2.putText(frame, label, (lx, ly - 2), font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)

    # 5. Draw 3D Coordinate Frame Axes (when calibrated pose estimation is available)
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

