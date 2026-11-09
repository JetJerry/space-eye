"""Visual observation fusion and temporal interaction event extraction package."""

from vyom.interaction.drawing import (
    COLOR_APPROACH,
    COLOR_CONTACT,
    COLOR_HOLDING,
    COLOR_PLACEMENT,
    draw_interaction_overlays,
)
from vyom.interaction.engine import (
    HandObjectRelationship,
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
    PRIMARY_GRASP_FINGERTIPS,
    SpatialProximityMetrics,
    compute_hand_obb_proximity,
    compute_obb_iou,
    compute_palm_center,
)
from vyom.interaction.tracker import (
    MultiObjectTracker,
    TrackedObject,
)
from vyom.interaction.worker import InteractionWorker

__all__ = [
    "COLOR_APPROACH",
    "COLOR_CONTACT",
    "COLOR_HOLDING",
    "COLOR_PLACEMENT",
    "HandObjectRelationship",
    "InteractionConfig",
    "InteractionEngine",
    "InteractionEvent",
    "InteractionEventType",
    "InteractionFrameResult",
    "InteractionState",
    "InteractionWorker",
    "MultiObjectTracker",
    "PRIMARY_GRASP_FINGERTIPS",
    "SpatialProximityMetrics",
    "TrackedObject",
    "compute_hand_obb_proximity",
    "compute_obb_iou",
    "compute_palm_center",
    "draw_interaction_overlays",
]
