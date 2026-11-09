"""Camera calibration parameters, checkerboard calibration utility, and rig coordinate transformations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from vyom.core.logging_setup import get_logger

logger = get_logger("vision.calibration")


@dataclass
class CameraCalibration:
    """Intrinsic camera calibration parameters."""
    camera_matrix: np.ndarray  # 3x3 float64 intrinsic matrix
    dist_coeffs: np.ndarray    # 1x5 or 1x8 distortion coefficients
    image_width: int
    image_height: int
    reprojection_error: Optional[float] = None
    calibration_date: Optional[str] = None

    def is_valid(self) -> bool:
        """Validate intrinsic matrix dimensions and values."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        if self.camera_matrix.shape != (3, 3):
            return False
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return fx > 0 and fy > 0 and 0 < cx < self.image_width and 0 < cy < self.image_height

    def save_to_file(self, file_path: Path | str) -> bool:
        """Save calibration data to JSON format."""
        target = Path(file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reprojection_error": self.reprojection_error,
            "calibration_date": self.calibration_date,
        }
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved camera calibration to: %s", target)
            return True
        except Exception as e:
            logger.error("Failed to save camera calibration to %s: %s", target, e)
            return False

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> Optional[CameraCalibration]:
        """Load calibration from JSON or NumPy .npz file."""
        target = Path(file_path).resolve()
        if not target.exists():
            logger.warning("Calibration file not found at: %s", target)
            return None

        if target.suffix == ".npz":
            try:
                npz = np.load(str(target))
                return cls(
                    camera_matrix=npz["camera_matrix"],
                    dist_coeffs=npz["dist_coeffs"],
                    image_width=int(npz.get("image_width", 1280)),
                    image_height=int(npz.get("image_height", 720)),
                )
            except Exception as e:
                logger.error("Failed to load NPZ calibration %s: %s", target, e)
                return None

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            calib = cls(
                camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
                image_width=int(data["image_width"]),
                image_height=int(data["image_height"]),
                reprojection_error=data.get("reprojection_error"),
                calibration_date=data.get("calibration_date"),
            )
            if calib.is_valid():
                logger.info("Successfully loaded camera calibration from: %s", target.name)
                return calib
            logger.warning("Loaded calibration parameters from %s are invalid", target)
            return None
        except Exception as e:
            logger.error("Failed to parse JSON calibration %s: %s", target, e)
            return None


class CheckerboardCalibrator:
    """Calibrates camera intrinsics using standard planar checkerboard pattern images."""

    def __init__(
        self,
        board_rows: int = 6,
        board_cols: int = 9,
        square_size_m: float = 0.025,  # 25 mm square size
    ) -> None:
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.square_size_m = square_size_m
        self.pattern_size = (board_cols, board_rows)

        # Prepare object points in 3D world coordinates (z = 0 plane)
        self.objp = np.zeros((board_rows * board_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_m

        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

    def add_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect checkerboard corners in frame and record if valid."""
        h, w = frame_bgr.shape[:2]
        self._image_size = (w, h)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )

        if ret and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.object_points.append(self.objp)
            self.image_points.append(corners_subpix)
            logger.info("Added calibration frame (%d captured total)", len(self.image_points))
            return True, corners_subpix

        return False, None

    def compute_calibration(self) -> Optional[CameraCalibration]:
        """Compute camera matrix and distortion coefficients from accumulated points."""
        if len(self.image_points) < 8 or self._image_size is None:
            logger.warning("Need at least 8 calibration views with detected corners (have %d)", len(self.image_points))
            return None

        logger.info("Computing camera calibration from %d views...", len(self.image_points))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self._image_size,
            None,
            None,
        )

        calib = CameraCalibration(
            camera_matrix=mtx,
            dist_coeffs=dist,
            image_width=self._image_size[0],
            image_height=self._image_size[1],
            reprojection_error=float(ret),
        )
        logger.info("Calibration computed: RMS error = %.4f", ret)
        return calib


class RigCoordinateSystem:
    """Manages coordinate transformations between Camera, ArUco Marker, and Rig Reference Frame.

    Assumptions & Reference Geometry:
    - Origin marker (default ID 0) defines the origin (0, 0, 0) of the experimental workspace.
    - The rig reference plane is defined by the physical mounting surface of the fiducials.
    - In microgravity, screen-up or room gravity has no physical meaning; all relative
      vectors are computed rigidly with respect to the rig coordinate frame.
    - When camera calibration is NOT available:
      Returns UNCALIBRATED and does NOT invent or fabricate 3D coordinates.
    """

    def __init__(
        self,
        origin_marker_id: int = 0,
        marker_size_m: float = 0.05,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        self.origin_marker_id = origin_marker_id
        self.marker_size_m = marker_size_m
        self.calibration = calibration

        # Rigid transform from camera to rig frame: T_rig_cam
        self._rvec_origin: Optional[np.ndarray] = None
        self._tvec_origin: Optional[np.ndarray] = None
        self._rig_origin_visible: bool = False

    @property
    def is_calibrated(self) -> bool:
        """True if camera calibration parameters are valid."""
        return self.calibration is not None and self.calibration.is_valid()

    @property
    def is_tracking_rig(self) -> bool:
        """True if camera is calibrated AND the origin marker is currently established."""
        return self.is_calibrated and self._rig_origin_visible and self._tvec_origin is not None

    def update_origin_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Set camera-to-origin transform from detected origin marker pose."""
        self._rvec_origin = rvec.copy()
        self._tvec_origin = tvec.copy()
        self._rig_origin_visible = True

    def clear_origin_pose(self) -> None:
        """Clear origin tracking state when origin marker is occluded."""
        self._rig_origin_visible = False

    def transform_camera_to_rig(self, point_camera: np.ndarray) -> Optional[np.ndarray]:
        """Transform a 3D point in camera frame [X_c, Y_c, Z_c] into the Rig frame [X_r, Y_r, Z_r].

        Returns:
            np.ndarray of shape (3,) in meters, or None if uncalibrated or origin missing.
        """
        if not self.is_tracking_rig or self._rvec_origin is None or self._tvec_origin is None:
            return None

        # Rotation matrix from camera to origin: R_cam_origin
        R_orig, _ = cv2.Rodrigues(self._rvec_origin)
        # Point relative to origin in camera frame:
        pt_diff = point_camera.reshape(3, 1) - self._tvec_origin.reshape(3, 1)
        # Rotate into origin/rig coordinate system: R^T * (P_c - t_c)
        pt_rig = R_orig.T @ pt_diff
        return pt_rig.flatten()

    def estimate_marker_pose(
        self,
        corners: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute 3D rotation and translation vectors for a single 4-corner marker.

        Returns:
            Tuple of (success, rvec, tvec)
        """
        if not self.is_calibrated or self.calibration is None:
            return False, None, None

        # 3D coordinates of marker corners in marker's local frame
        s = self.marker_size_m / 2.0
        obj_pts = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        img_pts = corners.reshape(-1, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        return bool(success), rvec, tvec

