"""Deterministic, explainable state-based hand-object interaction recognition engine.

Combines MediaPipe hand landmarks, tracked YOLOv8-OBB objects, and ArUco rig references
into structured, temporally validated interaction events.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np

from vyom.core.logging_setup import get_logger
from vyom.interaction.events import (
    InteractionEvent,
    InteractionEventType,
    InteractionState,
)
from vyom.interaction.geometry import (
    SpatialProximityMetrics,
    compute_hand_obb_proximity,
)
from vyom.interaction.tracker import MultiObjectTracker, TrackedObject
from vyom.vision.aruco_tracker import RigTrackingState
from vyom.vision.hand_tracker import TrackedHand
from vyom.vision.observations import PerceptionFrameObservation

logger = get_logger("interaction.engine")


@dataclass
class InteractionConfig:
    """Configurable geometric and temporal thresholds for interaction reasoning."""
    # Distance thresholds (in image pixels)
    approach_distance_threshold_px: float = 80.0
    approach_exit_threshold_px: float = 110.0   # Hysteresis exit margin
    contact_distance_threshold_px: float = 12.0
    contact_exit_threshold_px: float = 24.0     # Hysteresis exit margin
    release_distance_threshold_px: float = 30.0

    # Displacement thresholds for movement and pickup
    movement_displacement_threshold_px: float = 12.0
    pickup_displacement_threshold_px: float = 25.0
    placement_velocity_threshold_px_s: float = 20.0

    # Temporal persistence requirements (consecutive frames)
    min_approach_frames: int = 3
    min_contact_frames: int = 3
    min_pickup_frames: int = 5
    min_placement_frames: int = 4

    # Observation freshness
    max_stale_observation_age_s: float = 0.350

    # Maximum events retained in memory
    max_event_history: int = 200


@dataclass
class HandObjectRelationship:
    """Stateful interaction tracker for a specific (Hand, TrackedObject) pair."""
    hand_id: str                      # "hand_0", "hand_1"
    handedness: str                   # "Right" or "Left"
    object_track_id: str
    object_class: str
    current_state: InteractionState = InteractionState.IDLE
    consecutive_state_frames: int = 0
    consecutive_contact_frames: int = 0
    distance_history: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=10))
    initial_contact_center: Optional[Tuple[float, float]] = None
    last_event_emitted: Optional[InteractionEventType] = None
    last_event_time_s: float = 0.0
    last_metrics: Optional[SpatialProximityMetrics] = None


@dataclass
class InteractionFrameResult:
    """Container holding interaction recognition output for a single processed frame."""
    frame_id: int
    timestamp_s: float
    active_relationships: List[HandObjectRelationship]
    tracked_objects: List[TrackedObject]
    events: List[InteractionEvent]
    processing_time_ms: float
    has_rig_calibration: bool = False

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0

    @property
    def active_contacts(self) -> List[HandObjectRelationship]:
        return [r for r in self.active_relationships if r.current_state in (
            InteractionState.CONTACT_CANDIDATE,
            InteractionState.HOLDING_MOVING,
            InteractionState.PLACEMENT_CANDIDATE,
        )]


class InteractionEngine:
    """Deterministic, explainable temporal interaction recognition engine.

    Converts raw multi-modal perception observations into typed interaction events.
    Enforces temporal persistence, debouncing, and hysteresis to prevent false alarms.
    """

    def __init__(self, config: Optional[InteractionConfig] = None) -> None:
        self.config = config or InteractionConfig()
        self.tracker = MultiObjectTracker()

        # Key: (hand_id, object_track_id)
        self._relationships: Dict[Tuple[str, str], HandObjectRelationship] = {}

        # Event storage
        self._event_history: Deque[InteractionEvent] = collections.deque(maxlen=self.config.max_event_history)
        self._event_counter: int = 0
        self._last_processed_timestamp_s: float = -1.0

    @property
    def event_history(self) -> List[InteractionEvent]:
        return list(self._event_history)

    def process_observation(
        self,
        obs: PerceptionFrameObservation,
    ) -> InteractionFrameResult:
        """Process a unified multi-modal frame observation and extract interaction events."""
        t0 = time.perf_counter()
        frame_id = obs.frame_id
        timestamp_s = obs.timestamp_s

        # 1. Stale-result check
        if self._last_processed_timestamp_s > 0:
            dt = timestamp_s - self._last_processed_timestamp_s
            if dt > self.config.max_stale_observation_age_s:
                logger.warning(
                    "Stale observation gap detected (%.3fs > %.3fs limit). Resetting interaction temporal histories.",
                    dt,
                    self.config.max_stale_observation_age_s,
                )
                self._reset_transient_states()

        self._last_processed_timestamp_s = timestamp_s

        # 2. Update MultiObjectTracker
        raw_detections = obs.obb_result.detections if obs.obb_result else []
        active_objects = self.tracker.update(raw_detections, frame_id, timestamp_s)

        # 3. Retrieve hands
        tracked_hands = obs.get_hands()

        # 4. Calibration status
        is_calibrated = (
            obs.aruco_result is not None
            and obs.aruco_result.state == RigTrackingState.RIG_TRACKING
        )

        emitted_events: List[InteractionEvent] = []
        current_pair_keys: Set[Tuple[str, str]] = set()

        # 5. Evaluate all (Hand, Object) combinations
        for hand in tracked_hands:
            hand_key = f"hand_{hand.hand_index}"

            for obj in active_objects:
                pair_key = (hand_key, obj.track_id)
                current_pair_keys.add(pair_key)

                # Retrieve or initialize stateful relationship
                rel = self._relationships.get(pair_key)
                if rel is None:
                    rel = HandObjectRelationship(
                        hand_id=hand_key,
                        handedness=hand.handedness,
                        object_track_id=obj.track_id,
                        object_class=obj.class_name,
                    )
                    self._relationships[pair_key] = rel

                # Compute spatial metrics
                metrics = compute_hand_obb_proximity(
                    hand=hand,
                    obb_corners=obj.corners,
                    obb_center=obj.center,
                    contact_threshold_px=self.config.contact_distance_threshold_px,
                    has_rig_calibration=is_calibrated,
                )
                rel.last_metrics = metrics
                rel.distance_history.append(metrics.min_fingertip_distance_px)

                # State machine transition
                new_event = self._update_relationship_state(
                    rel=rel,
                    metrics=metrics,
                    obj=obj,
                    hand=hand,
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )
                if new_event is not None:
                    emitted_events.append(new_event)
                    self._event_history.append(new_event)

        # 6. Cleanup relationships for objects or hands no longer present
        dead_keys = set(self._relationships.keys()) - current_pair_keys
        for k in dead_keys:
            rel = self._relationships[k]
            # If hand separated from an actively held object, emit RELEASE_CANDIDATE
            if rel.current_state in (InteractionState.CONTACT_CANDIDATE, InteractionState.HOLDING_MOVING):
                rel_event = self._create_event(
                    event_type=InteractionEventType.RELEASE_CANDIDATE,
                    rel=rel,
                    obj_confidence=0.5,
                    interaction_conf=0.70,
                    evidence={"reason": "hand_or_object_exited_view"},
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )
                emitted_events.append(rel_event)
                self._event_history.append(rel_event)
            del self._relationships[k]

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return InteractionFrameResult(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            active_relationships=list(self._relationships.values()),
            tracked_objects=active_objects,
            events=emitted_events,
            processing_time_ms=round(duration_ms, 2),
            has_rig_calibration=is_calibrated,
        )

    def _update_relationship_state(
        self,
        rel: HandObjectRelationship,
        metrics: SpatialProximityMetrics,
        obj: TrackedObject,
        hand: TrackedHand,
        frame_id: int,
        timestamp_s: float,
        is_calibrated: bool,
    ) -> Optional[InteractionEvent]:
        """State machine evaluating deterministic state transitions and temporal evidence."""
        dist = metrics.min_fingertip_distance_px
        is_contact = metrics.contact_candidate
        state = rel.current_state

        event: Optional[InteractionEvent] = None

        # ----------------------------------------------------------------------
        # State: IDLE -> APPROACHING
        # ----------------------------------------------------------------------
        if state == InteractionState.IDLE:
            if dist <= self.config.approach_distance_threshold_px:
                rel.consecutive_state_frames += 1
                # Check distance trend (decreasing)
                if (
                    rel.consecutive_state_frames >= self.config.min_approach_frames
                    and self._is_distance_decreasing(rel.distance_history)
                ):
                    rel.current_state = InteractionState.APPROACHING
                    rel.consecutive_state_frames = 0
                    event = self._create_event(
                        event_type=InteractionEventType.HAND_APPROACH,
                        rel=rel,
                        obj_confidence=obj.confidence,
                        interaction_conf=self._compute_confidence(metrics, hand, obj, 0.65),
                        evidence={
                            "min_distance_px": dist,
                            "closest_fingertip": metrics.closest_fingertip_name,
                            "approach_vector": metrics.approach_vector_px,
                        },
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        is_calibrated=is_calibrated,
                    )
            else:
                rel.consecutive_state_frames = 0

        # ----------------------------------------------------------------------
        # State: APPROACHING -> CONTACT_CANDIDATE or IDLE
        # ----------------------------------------------------------------------
        elif state == InteractionState.APPROACHING:
            if is_contact:
                rel.consecutive_contact_frames += 1
                if rel.consecutive_contact_frames >= self.config.min_contact_frames:
                    rel.current_state = InteractionState.CONTACT_CANDIDATE
                    rel.consecutive_state_frames = 0
                    rel.initial_contact_center = obj.center
                    event = self._create_event(
                        event_type=InteractionEventType.CONTACT_CANDIDATE,
                        rel=rel,
                        obj_confidence=obj.confidence,
                        interaction_conf=self._compute_confidence(metrics, hand, obj, 0.80),
                        evidence={
                            "min_distance_px": dist,
                            "fingertips_inside": metrics.fingertips_inside_count,
                            "closest_fingertip": metrics.closest_fingertip_name,
                            "consecutive_contact_frames": rel.consecutive_contact_frames,
                        },
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        is_calibrated=is_calibrated,
                    )
            elif dist > self.config.approach_exit_threshold_px:
                # Hand retreated without contact
                rel.current_state = InteractionState.IDLE
                rel.consecutive_state_frames = 0
                rel.consecutive_contact_frames = 0
                event = self._create_event(
                    event_type=InteractionEventType.HAND_DEPART,
                    rel=rel,
                    obj_confidence=obj.confidence,
                    interaction_conf=0.60,
                    evidence={"distance_px": dist},
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )
            else:
                rel.consecutive_contact_frames = 0

        # ----------------------------------------------------------------------
        # State: CONTACT_CANDIDATE -> HOLDING_MOVING or RELEASE_CANDIDATE
        # ----------------------------------------------------------------------
        elif state == InteractionState.CONTACT_CANDIDATE:
            if is_contact or dist <= self.config.contact_exit_threshold_px:
                # Contact persists: check if object begins moving
                disp_from_contact = 0.0
                if rel.initial_contact_center is not None:
                    disp_from_contact = float(np.hypot(
                        obj.center[0] - rel.initial_contact_center[0],
                        obj.center[1] - rel.initial_contact_center[1],
                    ))

                if disp_from_contact >= self.config.pickup_displacement_threshold_px:
                    rel.consecutive_state_frames += 1
                    if rel.consecutive_state_frames >= self.config.min_pickup_frames:
                        rel.current_state = InteractionState.HOLDING_MOVING
                        rel.consecutive_state_frames = 0
                        event = self._create_event(
                            event_type=InteractionEventType.PICKUP_CANDIDATE,
                            rel=rel,
                            obj_confidence=obj.confidence,
                            interaction_conf=self._compute_confidence(metrics, hand, obj, 0.85),
                            evidence={
                                "displacement_from_contact_px": round(disp_from_contact, 2),
                                "object_velocity_px_s": round(float(np.hypot(*obj.velocity_px_per_s)), 2),
                                "sustained_contact_frames": rel.consecutive_contact_frames,
                            },
                            frame_id=frame_id,
                            timestamp_s=timestamp_s,
                            is_calibrated=is_calibrated,
                        )
                elif obj.frame_displacement_px >= self.config.movement_displacement_threshold_px:
                    # Minor object motion without full pickup
                    event = self._create_event(
                        event_type=InteractionEventType.OBJECT_MOVEMENT,
                        rel=rel,
                        obj_confidence=obj.confidence,
                        interaction_conf=0.75,
                        evidence={"displacement_px": round(obj.frame_displacement_px, 2)},
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        is_calibrated=is_calibrated,
                    )
            elif dist > self.config.release_distance_threshold_px:
                # Hand pulled away without picking up
                rel.current_state = InteractionState.RELEASE_CANDIDATE
                rel.consecutive_state_frames = 0
                rel.consecutive_contact_frames = 0
                event = self._create_event(
                    event_type=InteractionEventType.RELEASE_CANDIDATE,
                    rel=rel,
                    obj_confidence=obj.confidence,
                    interaction_conf=0.80,
                    evidence={"release_distance_px": dist},
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )

        # ----------------------------------------------------------------------
        # State: HOLDING_MOVING -> PLACEMENT_CANDIDATE or RELEASE_CANDIDATE
        # ----------------------------------------------------------------------
        elif state == InteractionState.HOLDING_MOVING:
            speed = float(np.hypot(*obj.velocity_px_per_s))

            if dist > self.config.release_distance_threshold_px:
                # Hand released object while/after moving
                rel.current_state = InteractionState.RELEASE_CANDIDATE
                rel.consecutive_state_frames = 0
                event = self._create_event(
                    event_type=InteractionEventType.RELEASE_CANDIDATE,
                    rel=rel,
                    obj_confidence=obj.confidence,
                    interaction_conf=0.85,
                    evidence={"release_distance_px": dist, "total_displacement_px": obj.total_displacement_px},
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )
            elif speed <= self.config.placement_velocity_threshold_px_s:
                # Object motion has arrested while hand is still near/contacting -> candidate placement
                rel.consecutive_state_frames += 1
                if rel.consecutive_state_frames >= self.config.min_placement_frames:
                    rel.current_state = InteractionState.PLACEMENT_CANDIDATE
                    rel.consecutive_state_frames = 0
                    event = self._create_event(
                        event_type=InteractionEventType.PLACEMENT_CANDIDATE,
                        rel=rel,
                        obj_confidence=obj.confidence,
                        interaction_conf=0.82,
                        evidence={
                            "final_speed_px_s": round(speed, 2),
                            "total_displacement_px": round(obj.total_displacement_px, 2),
                            "stationary_frames": rel.consecutive_state_frames,
                        },
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        is_calibrated=is_calibrated,
                    )

        # ----------------------------------------------------------------------
        # State: PLACEMENT_CANDIDATE -> RELEASE_CANDIDATE
        # ----------------------------------------------------------------------
        elif state == InteractionState.PLACEMENT_CANDIDATE:
            if dist > self.config.release_distance_threshold_px:
                rel.current_state = InteractionState.RELEASE_CANDIDATE
                rel.consecutive_state_frames = 0
                event = self._create_event(
                    event_type=InteractionEventType.RELEASE_CANDIDATE,
                    rel=rel,
                    obj_confidence=obj.confidence,
                    interaction_conf=0.90,
                    evidence={"release_distance_px": dist},
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    is_calibrated=is_calibrated,
                )

        # ----------------------------------------------------------------------
        # State: RELEASE_CANDIDATE -> IDLE
        # ----------------------------------------------------------------------
        elif state == InteractionState.RELEASE_CANDIDATE:
            if dist > self.config.approach_exit_threshold_px:
                rel.current_state = InteractionState.IDLE
                rel.consecutive_state_frames = 0
                rel.consecutive_contact_frames = 0
                rel.initial_contact_center = None

        return event

    def _create_event(
        self,
        event_type: InteractionEventType,
        rel: HandObjectRelationship,
        obj_confidence: float,
        interaction_conf: float,
        evidence: Dict[str, Any],
        frame_id: int,
        timestamp_s: float,
        is_calibrated: bool,
    ) -> InteractionEvent:
        """Construct typed InteractionEvent with unique ID and calibration context."""
        self._event_counter += 1
        event_id = f"evt_{self._event_counter:05d}_{event_type.value.lower()}"

        rel.last_event_emitted = event_type
        rel.last_event_time_s = timestamp_s

        coord_sys = "RIG_METRIC" if is_calibrated else "IMAGE_PIXEL"
        uncertainty = "CALIBRATED_RIG_METRIC" if is_calibrated else "UNCALIBRATED_2D (Depth ambiguity present)"

        return InteractionEvent(
            event_id=event_id,
            event_type=event_type,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            hand_id=rel.hand_id,
            handedness=rel.handedness,
            object_track_id=rel.object_track_id,
            object_class=rel.object_class,
            detection_confidence=obj_confidence,
            interaction_confidence=interaction_conf,
            supporting_evidence=evidence,
            coordinate_system=coord_sys,
            calibration_status=is_calibrated,
            spatial_uncertainty=uncertainty,
        )

    def _compute_confidence(
        self,
        metrics: SpatialProximityMetrics,
        hand: TrackedHand,
        obj: TrackedObject,
        base_conf: float,
    ) -> float:
        """Calculate grounded interaction confidence combining detector scores and geometric tightness."""
        # Tightness factor: higher when fingertips penetrate or are within 5px
        tightness = 1.0 if metrics.is_any_fingertip_inside else max(0.0, 1.0 - (metrics.min_fingertip_distance_px / 30.0))
        # Combined score
        conf = base_conf * 0.4 + hand.confidence * 0.2 + obj.confidence * 0.2 + tightness * 0.2
        return float(np.clip(conf, 0.10, 0.98))

    def _is_distance_decreasing(self, history: Deque[float]) -> bool:
        """Verify downward-trending distance towards object."""
        if len(history) < 2:
            return False
        if len(history) == 2:
            return history[0] > history[1]
        recent = list(history)[-3:]
        return (recent[0] >= recent[1] > recent[2]) or (recent[0] > recent[1] >= recent[2])

    def _reset_transient_states(self) -> None:
        """Reset relationship counters upon temporal discontinuities."""
        for rel in self._relationships.values():
            rel.consecutive_state_frames = 0
            rel.consecutive_contact_frames = 0
            rel.current_state = InteractionState.IDLE

    def reset(self) -> None:
        """Reset all state machines, tracker identities, and event histories."""
        self.tracker.reset()
        self._relationships.clear()
        self._event_history.clear()
        self._event_counter = 0
        self._last_processed_timestamp_s = -1.0
