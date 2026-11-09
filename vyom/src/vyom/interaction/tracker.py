"""Multi-object temporal association and persistent identity tracking for laboratory apparatus."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from vyom.core.logging_setup import get_logger
from vyom.interaction.geometry import compute_obb_iou
from vyom.vision.obb_detector import OBBDetection

logger = get_logger("interaction.tracker")


@dataclass
class TrackedObject:
    """Persistent tracked laboratory object identity across consecutive video frames."""
    track_id: str                      # Unique persistent identifier (e.g., "pipette_1")
    class_name: str
    class_id: int
    confidence: float
    corners: np.ndarray                # Shape (4, 2) float32 pixel coordinates
    center: Tuple[float, float]
    initial_center: Tuple[float, float] # Recorded at initial track confirmation
    smoothed_center: Tuple[float, float]
    velocity_px_per_s: Tuple[float, float] = (0.0, 0.0)
    frame_displacement_px: float = 0.0  # Displacement in current frame
    total_displacement_px: float = 0.0  # Cumulative distance from initial_center
    first_frame_id: int = 0
    last_frame_id: int = 0
    last_timestamp_s: float = 0.0
    consecutive_hits: int = 1
    lost_frames: int = 0
    is_provisional: bool = False
    state: str = "ACTIVE"              # "ACTIVE", "OCCLUDED", "REMOVED"

    @property
    def is_active(self) -> bool:
        return self.state == "ACTIVE"

    @property
    def is_confirmed(self) -> bool:
        return self.consecutive_hits >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "smoothed_center": [round(self.smoothed_center[0], 2), round(self.smoothed_center[1], 2)],
            "velocity_px_per_s": [round(self.velocity_px_per_s[0], 2), round(self.velocity_px_per_s[1], 2)],
            "frame_displacement_px": round(self.frame_displacement_px, 2),
            "total_displacement_px": round(self.total_displacement_px, 2),
            "state": self.state,
            "consecutive_hits": self.consecutive_hits,
            "lost_frames": self.lost_frames,
            "is_provisional": self.is_provisional,
        }


class MultiObjectTracker:
    """Lightweight deterministic object tracker associating OBB detections across frames.

    Uses class matching, centroid gating, IoU, and exponential smoothing to maintain
    consistent object identities despite temporary occlusion or detection dropouts.
    """

    def __init__(
        self,
        max_matching_distance_px: float = 90.0,
        min_iou_threshold: float = 0.15,
        max_lost_frames: int = 15,
        ema_alpha: float = 0.65,
    ) -> None:
        self.max_matching_distance_px = max_matching_distance_px
        self.min_iou_threshold = min_iou_threshold
        self.max_lost_frames = max_lost_frames
        self.ema_alpha = ema_alpha

        self._tracks: Dict[str, TrackedObject] = {}
        self._class_counters: Dict[str, int] = {}
        self._next_uid: int = 1

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Return all currently confirmed active tracks."""
        return [t for t in self._tracks.values() if t.is_active and t.is_confirmed]

    @property
    def all_tracks(self) -> List[TrackedObject]:
        """Return all tracked objects that are not permanently removed."""
        return [t for t in self._tracks.values() if t.state != "REMOVED"]

    def update(
        self,
        detections: List[OBBDetection],
        frame_id: int,
        timestamp_s: float,
    ) -> List[TrackedObject]:
        """Update tracker state with new frame detections.

        Matches incoming detections against existing tracks, updates motion statistics,
        handles lost/occluded tracks, and spawns new persistent identities.
        """
        # 1. Match detections to existing tracks
        matched_tracks, matched_dets = self._associate_detections_to_tracks(detections)

        # 2. Update matched tracks
        for track_id, det_idx in matched_tracks.items():
            det = detections[det_idx]
            track = self._tracks[track_id]

            # Time delta for velocity
            dt = max(0.001, timestamp_s - track.last_timestamp_s) if track.last_timestamp_s > 0 else 0.033

            prev_cx, prev_cy = track.center
            cur_cx, cur_cy = det.center

            # Displacement
            dx = float(cur_cx - prev_cx)
            dy = float(cur_cy - prev_cy)
            disp = float(np.hypot(dx, dy))

            # Exponential Moving Average for position smoothing
            smooth_cx = self.ema_alpha * cur_cx + (1.0 - self.ema_alpha) * track.smoothed_center[0]
            smooth_cy = self.ema_alpha * cur_cy + (1.0 - self.ema_alpha) * track.smoothed_center[1]

            # Velocity in px/s
            vx = dx / dt
            vy = dy / dt

            # Total displacement from original stationary position
            tot_disp = float(np.hypot(cur_cx - track.initial_center[0], cur_cy - track.initial_center[1]))

            track.corners = det.corners
            track.center = det.center
            track.smoothed_center = (smooth_cx, smooth_cy)
            track.velocity_px_per_s = (vx, vy)
            track.frame_displacement_px = disp
            track.total_displacement_px = tot_disp
            track.confidence = det.confidence
            track.last_frame_id = frame_id
            track.last_timestamp_s = timestamp_s
            track.consecutive_hits += 1
            track.lost_frames = 0
            track.state = "ACTIVE"

        # 3. Handle unmatched existing tracks (mark lost/occluded)
        all_existing_ids = set(self._tracks.keys())
        unmatched_track_ids = all_existing_ids - set(matched_tracks.keys())

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.lost_frames += 1
            track.frame_displacement_px = 0.0

            if track.lost_frames > self.max_lost_frames:
                track.state = "REMOVED"
            else:
                track.state = "OCCLUDED"

        # 4. Spawn new tracks for unmatched detections
        all_det_indices = set(range(len(detections)))
        unmatched_det_indices = all_det_indices - set(matched_dets)

        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            cname = det.class_name

            # Generate persistent track ID (e.g., "pipette_1")
            idx = self._class_counters.get(cname, 0) + 1
            self._class_counters[cname] = idx
            track_id = f"{cname}_{idx}"

            new_track = TrackedObject(
                track_id=track_id,
                class_name=cname,
                class_id=det.class_id,
                confidence=det.confidence,
                corners=det.corners,
                center=det.center,
                initial_center=det.center,
                smoothed_center=det.center,
                velocity_px_per_s=(0.0, 0.0),
                frame_displacement_px=0.0,
                total_displacement_px=0.0,
                first_frame_id=frame_id,
                last_frame_id=frame_id,
                last_timestamp_s=timestamp_s,
                consecutive_hits=1,
                lost_frames=0,
                is_provisional=det.is_provisional,
                state="ACTIVE",
            )
            self._tracks[track_id] = new_track

        # 5. Clean up permanently removed tracks to keep memory bounded
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.state != "REMOVED"}

        return self.active_tracks

    def _associate_detections_to_tracks(
        self,
        detections: List[OBBDetection],
    ) -> Tuple[Dict[str, int], Set[int]]:
        """Greedy bipartite association based on class identity, centroid distance, and IoU."""
        matched_tracks: Dict[str, int] = {}
        matched_dets: Set[int] = set()

        if not self._tracks or not detections:
            return matched_tracks, matched_dets

        # Build candidate pairing scores: (cost, track_id, det_idx)
        candidates = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(detections):
                # Strict class matching constraint
                if track.class_name != det.class_name:
                    continue

                # Centroid Euclidean distance
                dist = float(np.hypot(track.smoothed_center[0] - det.center[0], track.smoothed_center[1] - det.center[1]))
                if dist > self.max_matching_distance_px:
                    continue

                # Compute IoU if close
                iou = compute_obb_iou(track.corners, det.corners)

                # Cost metric: weighted combination of normalized distance and inverse IoU
                cost = (dist / self.max_matching_distance_px) * 0.7 + (1.0 - iou) * 0.3
                candidates.append((cost, track_id, det_idx))

        # Sort candidate associations by ascending cost
        candidates.sort(key=lambda x: x[0])

        assigned_tracks: Set[str] = set()
        for cost, track_id, det_idx in candidates:
            if track_id not in assigned_tracks and det_idx not in matched_dets:
                matched_tracks[track_id] = det_idx
                assigned_tracks.add(track_id)
                matched_dets.add(det_idx)

        return matched_tracks, matched_dets

    def reset(self) -> None:
        """Clear all active tracks and counters."""
        self._tracks.clear()
        self._class_counters.clear()
        self._next_uid = 1

