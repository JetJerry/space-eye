"""Focused unit tests for MediaPipe Hand Landmarker integration and landmark observations."""

from pathlib import Path
import numpy as np
import pytest

from vyom.vision.aruco_tracker import ArucoDetectionResult
from vyom.vision.drawing import draw_aruco_overlays, draw_hand_overlays
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


def test_hand_landmark_index_completeness():
    """Verify that all 21 anatomical landmarks are mapped."""
    assert len(HandLandmarkIndex) == 21
    assert HandLandmarkIndex.WRIST == 0
    assert HandLandmarkIndex.THUMB_TIP == 4
    assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
    assert HandLandmarkIndex.MIDDLE_FINGER_TIP == 12
    assert HandLandmarkIndex.RING_FINGER_TIP == 16
    assert HandLandmarkIndex.PINKY_TIP == 20
    assert len(HAND_CONNECTIONS) > 15


def test_hand_tracker_missing_model_graceful_handling(tmp_path: Path):
    """Verify system shows MODEL_MISSING and does not fabricate landmarks when asset is missing."""
    bogus_path = tmp_path / "non_existent_hand_model.task"
    config = HandTrackerConfig(model_path=bogus_path)
    tracker = HandTracker(config)

    success, status, err = tracker.initialize()
    assert not success
    assert status == HandTrackingStatus.MODEL_MISSING
    assert err is not None

    # Processing frame without model must not crash and never fabricate landmarks
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(dummy_frame, timestamp_ms=100)
    assert res.status == HandTrackingStatus.MODEL_MISSING
    assert res.hand_count == 0
    assert len(res.hands) == 0


def test_hand_tracker_real_model_initialization_and_inference():
    """Verify that real hand_landmarker.task initializes and processes frames."""
    candidates = [
        Path("vyom/models/hand_landmarker.task"),
        Path("models/hand_landmarker.task"),
        Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task",
    ]
    model_path = None
    for cand in candidates:
        if cand.exists():
            model_path = cand.resolve()
            break

    if model_path is None:
        pytest.skip("hand_landmarker.task not found for test")

    config = HandTrackerConfig(model_path=model_path, num_hands=2, running_mode="VIDEO")
    tracker = HandTracker(config)

    success, status, err = tracker.initialize()
    assert success, f"Failed initialization: {err}"
    assert tracker.is_initialized

    # Process blank frame (should detect 0 hands, NO_HAND_DETECTED, no error)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(blank_frame, timestamp_ms=100, frame_idx=1)

    assert res.status == HandTrackingStatus.NO_HAND_DETECTED
    assert res.hand_count == 0
    assert res.latency_ms > 0.0
    assert res.image_width == 640
    assert res.image_height == 480

    # Test strictly increasing timestamp enforcement
    # Sending timestamp_ms=50 (which is <= 100) should be automatically bumped to 101 without throwing
    res2 = tracker.process_frame(blank_frame, timestamp_ms=50, frame_idx=2)
    assert res2.timestamp_ms == 101
    assert res2.status == HandTrackingStatus.NO_HAND_DETECTED

    tracker.close()
    assert not tracker.is_initialized


def test_hand_overlays_drawing_and_aruco_coexistence():
    """Verify hand landmarks overlay drawing and coexistence with ArUco."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Simulate ArUco empty/present result
    aruco_res = ArucoDetectionResult(frame_timestamp_s=1.0, markers=[])
    frame_with_aruco = draw_aruco_overlays(frame, aruco_res)

    # 2. Simulate synthetic hand with 21 landmarks
    lms = []
    for i in range(21):
        lms.append(
            HandLandmark(
                index=i,
                name=HandLandmarkIndex(i).name,
                x=0.5 + 0.01 * (i % 5),
                y=0.5 + 0.01 * (i // 5),
                z=0.0,
                pixel_x=320 + 5 * (i % 5),
                pixel_y=240 + 5 * (i // 5),
            )
        )

    hand = TrackedHand(
        hand_index=0,
        handedness="Right",
        confidence=0.95,
        landmarks=lms,
        pixel_bbox=(300, 220, 360, 280),
        wrist_pixel=(320, 240),
        index_tip_pixel=(335, 255),
        thumb_tip_pixel=(325, 245),
    )

    hand_result = HandTrackingResult(
        status=HandTrackingStatus.HAND_DETECTED,
        hands=[hand],
        hand_count=1,
        frame_idx=1,
        timestamp_ms=100,
        latency_ms=12.5,
        image_width=640,
        image_height=480,
    )

    # Draw hand overlays on top of ArUco frame
    final_frame = draw_hand_overlays(frame_with_aruco, hand_result)

    assert final_frame.shape == frame.shape
    assert final_frame.dtype == frame.dtype
    # The frame should now have non-zero pixels from drawn landmarks
    assert np.count_nonzero(final_frame) > 0
