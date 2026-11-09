"""Unit tests for hand-object interaction recognition, tracking, and temporal event extraction."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from vyom.interaction.engine import (
    InteractionConfig,
    InteractionEngine,
    InteractionFrameResult,
)
from vyom.interaction.events import (
    InteractionEvent,
    InteractionEventType,
    InteractionState,
)
from vyom.interaction.geometry import (
    compute_hand_obb_proximity,
    compute_obb_iou,
    compute_palm_center,
)
from vyom.interaction.tracker import MultiObjectTracker, TrackedObject
from vyom.vision.hand_tracker import HandLandmark, HandLandmarkIndex, TrackedHand
from vyom.vision.obb_detector import OBBDetection, OBBFrameDetections
from vyom.vision.observations import PerceptionFrameObservation


def _create_mock_hand(
    wrist_px: Tuple[int, int] = (100, 200),
    index_tip_px: Tuple[int, int] = (150, 150),
    thumb_tip_px: Tuple[int, int] = (140, 160),
    handedness: str = "Right",
) -> TrackedHand:
    """Helper creating a valid mock TrackedHand with 21 landmarks."""
    landmarks = []
    for i in range(21):
        if i == HandLandmarkIndex.WRIST:
            px, py = wrist_px
        elif i == HandLandmarkIndex.INDEX_FINGER_TIP:
            px, py = index_tip_px
        elif i == HandLandmarkIndex.THUMB_TIP:
            px, py = thumb_tip_px
        else:
            # Intermediate joints around wrist
            px = int(wrist_px[0] + (index_tip_px[0] - wrist_px[0]) * (i / 20.0))
            py = int(wrist_px[1] + (index_tip_px[1] - wrist_px[1]) * (i / 20.0))

        landmarks.append(
            HandLandmark(
                index=i,
                name=f"landmark_{i}",
                x=px / 640.0,
                y=py / 480.0,
                z=0.0,
                pixel_x=px,
                pixel_y=py,
                # Hand-relative coordinates in meters
                world_x=0.01 * (i - 10),
                world_y=0.01 * (i - 10),
                world_z=0.0,
            )
        )

    return TrackedHand(
        hand_index=0,
        handedness=handedness,
        confidence=0.92,
        landmarks=landmarks,
        pixel_bbox=(min(wrist_px[0], index_tip_px[0]) - 10, min(wrist_px[1], index_tip_px[1]) - 10, 200, 250),
        wrist_pixel=wrist_px,
        index_tip_pixel=index_tip_px,
        thumb_tip_pixel=thumb_tip_px,
    )


def _create_mock_obb_detection(
    center: Tuple[float, float] = (160.0, 150.0),
    width: float = 30.0,
    height: float = 80.0,
    class_name: str = "pipette",
) -> OBBDetection:
    """Helper creating an OBBDetection with rectangle corners."""
    cx, cy = center
    hw, hh = width / 2.0, height / 2.0
    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    return OBBDetection(
        class_id=4,
        class_name=class_name,
        confidence=0.88,
        corners=corners,
        center=center,
        width=width,
        height=height,
        angle_deg=0.0,
        frame_id=1,
        timestamp_s=0.033,
        is_provisional=False,
    )


def test_object_tracker_identity_and_smoothing():
    """Verify MultiObjectTracker maintains persistent identity across consecutive frames."""
    tracker = MultiObjectTracker(max_matching_distance_px=50.0)

    # Frame 1: Initial detection
    det1 = _create_mock_obb_detection(center=(100.0, 100.0), class_name="pipette")
    tracks1 = tracker.update([det1], frame_id=1, timestamp_s=0.033)
    assert len(tracker.all_tracks) == 1
    t = tracker.all_tracks[0]
    assert t.track_id == "pipette_1"
    assert t.consecutive_hits == 1

    # Frame 2: Slight movement (within matching distance)
    det2 = _create_mock_obb_detection(center=(104.0, 102.0), class_name="pipette")
    tracks2 = tracker.update([det2], frame_id=2, timestamp_s=0.066)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == "pipette_1"
    assert tracks2[0].consecutive_hits == 2
    assert tracks2[0].is_confirmed
    assert tracks2[0].total_displacement_px > 0.0

    # Frame 3: Occlusion / dropout (no detections)
    tracks3 = tracker.update([], frame_id=3, timestamp_s=0.099)
    assert len(tracks3) == 0  # Not actively detected
    all_t = tracker.all_tracks
    assert len(all_t) == 1
    assert all_t[0].state == "OCCLUDED"
    assert all_t[0].lost_frames == 1

    # Frame 4: Reappearance
    det4 = _create_mock_obb_detection(center=(105.0, 103.0), class_name="pipette")
    tracks4 = tracker.update([det4], frame_id=4, timestamp_s=0.133)
    assert len(tracks4) == 1
    assert tracks4[0].track_id == "pipette_1"  # Identity preserved!
    assert tracks4[0].state == "ACTIVE"


def test_hand_obb_proximity_inside_and_outside():
    """Verify exact pointPolygonTest distance and inside/outside detection."""
    # OBB box centered at (200, 200), width=40, height=60 -> X: [180, 220], Y: [170, 230]
    obb_corners = np.array([
        [180, 170],
        [220, 170],
        [220, 230],
        [180, 230],
    ], dtype=np.float32)
    obb_center = (200.0, 200.0)

    # 1. Fingertip clearly inside OBB at (200, 200)
    hand_inside = _create_mock_hand(
        wrist_px=(100, 200),
        index_tip_px=(200, 200),
        thumb_tip_px=(195, 205),
    )
    metrics_in = compute_hand_obb_proximity(hand_inside, obb_corners, obb_center, contact_threshold_px=10.0)
    assert metrics_in.is_any_fingertip_inside is True
    assert metrics_in.min_fingertip_distance_px == 0.0
    assert metrics_in.contact_candidate is True
    assert "UNCALIBRATED_2D" in metrics_in.spatial_uncertainty

    # 2. Fingertip outside OBB at (250, 200) -> distance to right edge (220, y) is 30px
    hand_outside = _create_mock_hand(
        wrist_px=(300, 200),
        index_tip_px=(250, 200),
        thumb_tip_px=(260, 200),
    )
    metrics_out = compute_hand_obb_proximity(hand_outside, obb_corners, obb_center, contact_threshold_px=10.0)
    assert metrics_out.is_any_fingertip_inside is False
    assert abs(metrics_out.min_fingertip_distance_px - 30.0) <= 1.0
    assert metrics_out.contact_candidate is False


def test_interaction_temporal_state_machine():
    """Verify temporal state progression: APPROACH -> CONTACT -> PICKUP -> RELEASE."""
    cfg = InteractionConfig(
        approach_distance_threshold_px=60.0,
        contact_distance_threshold_px=15.0,
        min_approach_frames=2,
        min_contact_frames=2,
        min_pickup_frames=2,
        pickup_displacement_threshold_px=20.0,
    )
    engine = InteractionEngine(cfg)

    # Object stationary at (200, 200)
    obb_det = _create_mock_obb_detection(center=(200.0, 200.0), width=30.0, height=60.0)

    # Step 1: Hand far away at (350, 200) -> IDLE
    h1 = _create_mock_hand(wrist_px=(400, 200), index_tip_px=(350, 200))
    obs1 = PerceptionFrameObservation(frame_id=1, timestamp_s=0.033, obb_result=OBBFrameDetections(1, 0.033, [obb_det], 5.0, "cpu", (480, 640)))
    obs1.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h1]})()
    res1 = engine.process_observation(obs1)
    assert len(res1.events) == 0

    # Step 2: Hand approaches to (245, 200) -> distance ~ 30px
    h2 = _create_mock_hand(wrist_px=(300, 200), index_tip_px=(245, 200))
    obs2 = PerceptionFrameObservation(frame_id=2, timestamp_s=0.066, obb_result=OBBFrameDetections(2, 0.066, [obb_det], 5.0, "cpu", (480, 640)))
    obs2.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h2]})()
    res2 = engine.process_observation(obs2)

    # Step 3: Hand continues approaching to (230, 200) -> HAND_APPROACH event
    h3 = _create_mock_hand(wrist_px=(280, 200), index_tip_px=(230, 200))
    obs3 = PerceptionFrameObservation(frame_id=3, timestamp_s=0.099, obb_result=OBBFrameDetections(3, 0.099, [obb_det], 5.0, "cpu", (480, 640)))
    obs3.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h3]})()
    res3 = engine.process_observation(obs3)
    ev_types = [e.event_type for e in res3.events]
    assert InteractionEventType.HAND_APPROACH in ev_types

    # Step 4: Hand touches OBB for 2 consecutive frames -> CONTACT_CANDIDATE
    for f in [4, 5]:
        h_touch = _create_mock_hand(wrist_px=(250, 200), index_tip_px=(200, 200))  # Inside OBB!
        obs_touch = PerceptionFrameObservation(f, f * 0.033, obb_result=OBBFrameDetections(f, f * 0.033, [obb_det], 5.0, "cpu", (480, 640)))
        obs_touch.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h_touch]})()
        res_touch = engine.process_observation(obs_touch)

    contact_evs = [e for e in engine.event_history if e.event_type == InteractionEventType.CONTACT_CANDIDATE]
    assert len(contact_evs) >= 1
    assert contact_evs[0].object_class == "pipette"

    # Step 5: Object moves significantly with hand -> PICKUP_CANDIDATE
    for f in [6, 7]:
        # Object displaced to (250, 200) -> 50px displacement!
        obb_moved = _create_mock_obb_detection(center=(250.0, 200.0), width=30.0, height=60.0)
        h_move = _create_mock_hand(wrist_px=(300, 200), index_tip_px=(250, 200))
        obs_move = PerceptionFrameObservation(f, f * 0.033, obb_result=OBBFrameDetections(f, f * 0.033, [obb_moved], 5.0, "cpu", (480, 640)))
        obs_move.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h_move]})()
        engine.process_observation(obs_move)

    pickup_evs = [e for e in engine.event_history if e.event_type == InteractionEventType.PICKUP_CANDIDATE]
    assert len(pickup_evs) >= 1
    assert pickup_evs[0].supporting_evidence["displacement_from_contact_px"] >= 20.0

    # Step 6: Hand pulls away to (350, 200) -> RELEASE_CANDIDATE
    obb_stopped = _create_mock_obb_detection(center=(250.0, 200.0), width=30.0, height=60.0)
    h_away = _create_mock_hand(wrist_px=(400, 200), index_tip_px=(350, 200))
    obs_away = PerceptionFrameObservation(8, 0.264, obb_result=OBBFrameDetections(8, 0.264, [obb_stopped], 5.0, "cpu", (480, 640)))
    obs_away.hand_result = type("DummyHandResult", (), {"has_hands": True, "hands": [h_away]})()
    res_away = engine.process_observation(obs_away)

    release_evs = [e for e in res_away.events if e.event_type == InteractionEventType.RELEASE_CANDIDATE]
    assert len(release_evs) == 1
    assert release_evs[0].event_type == InteractionEventType.RELEASE_CANDIDATE


def test_interaction_event_serialization():
    """Verify InteractionEvent serializes all required metadata for Prompt 9 procedure validation."""
    ev = InteractionEvent(
        event_id="evt_00001_pickup_candidate",
        event_type=InteractionEventType.PICKUP_CANDIDATE,
        frame_id=42,
        timestamp_s=1.400,
        hand_id="hand_0",
        handedness="Right",
        object_track_id="pipette_1",
        object_class="pipette",
        detection_confidence=0.91,
        interaction_confidence=0.84,
        supporting_evidence={"displacement_px": 28.5, "sustained_frames": 5},
        coordinate_system="IMAGE_PIXEL",
        calibration_status=False,
        spatial_uncertainty="UNCALIBRATED_2D (Depth ambiguity present)",
    )

    d = ev.to_dict()
    assert d["event_id"] == "evt_00001_pickup_candidate"
    assert d["event_type"] == "PICKUP_CANDIDATE"
    assert d["frame_id"] == 42
    assert d["hand_id"] == "hand_0"
    assert d["handedness"] == "Right"
    assert d["object_track_id"] == "pipette_1"
    assert d["object_class"] == "pipette"
    assert d["detection_confidence"] == 0.91
    assert d["interaction_confidence"] == 0.84
    assert d["coordinate_system"] == "IMAGE_PIXEL"
    assert d["calibration_status"] is False
    assert d["supporting_evidence"]["displacement_px"] == 28.5
