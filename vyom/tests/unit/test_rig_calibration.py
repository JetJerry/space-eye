"""Unit tests for camera calibration, ArUco tracking math, and rig coordinate transformations."""

import numpy as np
import pytest

from vyom.vision.calibration import CameraCalibration, RigCoordinateSystem


def test_camera_calibration_validation():
    """Verify intrinsic matrix dimension and optical center sanity checks."""
    # Invalid calibration (None values)
    invalid_calib = CameraCalibration(
        camera_matrix=None,
        dist_coeffs=None,
        image_width=1280,
        image_height=720,
    )
    assert not invalid_calib.is_valid()

    # Valid calibration
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((1, 5), dtype=np.float64)

    valid_calib = CameraCalibration(
        camera_matrix=K,
        dist_coeffs=dist,
        image_width=1280,
        image_height=720,
    )
    assert valid_calib.is_valid()


def test_rig_coordinate_system_uncalibrated_behavior():
    """Verify that uncalibrated rig never fabricates 3D coordinates."""
    rig = RigCoordinateSystem(origin_marker_id=0, marker_size_m=0.05, calibration=None)
    assert not rig.is_calibrated
    assert not rig.is_tracking_rig

    # Attempting to transform a point must return None
    pt = np.array([0.1, 0.2, 0.5])
    assert rig.transform_camera_to_rig(pt) is None


def test_rig_coordinate_system_known_point_transform():
    """Verify known-point coordinate transform math with synthetic pose.

    Origin marker is at camera position [0.0, 0.0, 0.5] with zero rotation.
    Point at camera [0.05, 0.02, 0.6] should transform to rig [0.05, 0.02, 0.1].
    """
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((1, 5), dtype=np.float64)
    calib = CameraCalibration(K, dist, 1280, 720)

    rig = RigCoordinateSystem(origin_marker_id=0, marker_size_m=0.05, calibration=calib)
    assert rig.is_calibrated

    # Set origin pose (zero rotation, 0.5m in front of camera along Z)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float64)
    rig.update_origin_pose(rvec, tvec)
    assert rig.is_tracking_rig

    # Test point in camera coordinates
    pt_cam = np.array([0.05, 0.02, 0.6], dtype=np.float64)
    pt_rig = rig.transform_camera_to_rig(pt_cam)

    assert pt_rig is not None
    # Rig X = 0.05 - 0.0 = 0.05
    # Rig Y = 0.02 - 0.0 = 0.02
    # Rig Z = 0.60 - 0.5 = 0.10
    np.testing.assert_allclose(pt_rig, [0.05, 0.02, 0.10], atol=1e-5)

    # Origin itself in camera coords should map to [0, 0, 0] in rig coords
    origin_cam = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    origin_rig = rig.transform_camera_to_rig(origin_cam)
    np.testing.assert_allclose(origin_rig, [0.0, 0.0, 0.0], atol=1e-5)

