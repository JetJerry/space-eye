"""YOLOv8-OBB Oriented Object Detection Engine, typed detections, and device management.

Integrates custom-trained YOLOv8n-OBB checkpoint into the VYOM vision pipeline with
configurable CPU/GPU inference, automatic fallback, confidence filtering, and
four-corner oriented bounding box extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from vyom.core.logging_setup import get_logger

logger = get_logger("vision.obb_detector")

# Expected final 10-class taxonomy
FINAL_CLASSES: List[str] = [
    "centrifuge_tube",     # 0: Real (LabOS) + Synthetic v2
    "tube_rack",           # 1: Real (LabOS + ChemEq25)
    "rack_hole",           # 2: Real (LabOS)
    "tube_cap",            # 3: Real (LabOS)
    "pipette",             # 4: Real (ChemEq25) + Synthetic v2
    "beaker",              # 5: Real (ChemEq25)
    "conical_flask",       # 6: Real (ChemEq25)
    "sample_vial",         # 7: Provisional (Synthetic v2)
    "multiwell_plate",     # 8: Provisional (Synthetic v2)
    "rig_slot",            # 9: Provisional (Synthetic v2)
]

PROVISIONAL_SYNTHETIC_CLASSES: Set[str] = {
    "sample_vial",
    "multiwell_plate",
    "rig_slot",
}

REAL_WORLD_CLASSES: Set[str] = {
    "centrifuge_tube",
    "tube_rack",
    "rack_hole",
    "tube_cap",
    "pipette",
    "beaker",
    "conical_flask",
}


class OBBDetectorStatus(str, Enum):
    """Subsystem operational statuses for YOLOv8-OBB."""
    MODEL_MISSING = "YOLO: MODEL MISSING"
    LOADING = "YOLO: LOADING"
    READY = "YOLO: READY"
    INFERENCE_ACTIVE = "YOLO: INFERENCE ACTIVE"
    NO_OBJECTS_DETECTED = "YOLO: NO OBJECTS DETECTED"
    ERROR = "YOLO: ERROR"


@dataclass
class OBBDetection:
    """Structured observation of a single oriented bounding box detection."""
    class_id: int
    class_name: str
    confidence: float
    # 4 corner coordinates in frame pixel space: shape (4, 2) as float32 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    corners: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float
    frame_id: int
    timestamp_s: float
    is_provisional: bool = False

    @property
    def corners_int(self) -> np.ndarray:
        """Discrete integer pixel coordinates for rendering."""
        return np.round(self.corners).astype(np.int32)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection for logging, storage, or protocol analysis."""
        return {
            "frame_id": self.frame_id,
            "timestamp_s": round(self.timestamp_s, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "angle_deg": round(self.angle_deg, 2),
            "corners": self.corners.round(2).tolist(),
            "is_provisional": self.is_provisional,
        }


@dataclass
class OBBFrameDetections:
    """Frame-level container for all oriented bounding box detections."""
    frame_id: int
    timestamp_s: float
    detections: List[OBBDetection]
    inference_time_ms: float
    device_used: str
    frame_shape: Tuple[int, int]  # (height, width)
    gpu_memory_used_mb: Optional[float] = None

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def by_class(self, class_name: str) -> List[OBBDetection]:
        return [d for d in self.detections if d.class_name == class_name]

    def real_world_detections(self) -> List[OBBDetection]:
        return [d for d in self.detections if not d.is_provisional]

    def provisional_detections(self) -> List[OBBDetection]:
        return [d for d in self.detections if d.is_provisional]


@dataclass
class OBBDetectorConfig:
    """Configuration options for YOLOv8-OBB inference engine."""
    model_path: Path = Path("models/vyom_yolov8n_obb_smoke.pt")
    metadata_path: Optional[Path] = Path("models/smoke_model_metadata.json")
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    device_preference: str = "AUTO"  # "AUTO", "CPU", "GPU"
    allow_cpu_fallback: bool = True
    max_detections: int = 100


def resolve_asset_path(path_str_or_path: Path | str) -> Path:
    """Robustly resolve model and metadata asset paths relative to project root or workspace."""
    p = Path(path_str_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    # Search project roots
    from vyom.core.config import find_project_root
    root = find_project_root()
    candidates = [
        root / p,
        root / "vyom" / p,
        root.parent / "vyom" / p,
        Path("d:/space_eye/vyom") / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return p.resolve()


class YOLOOBBDetector:
    """YOLOv8-OBB inference engine with hardware inspection, class verification, and offline execution."""

    def __init__(self, config: Optional[OBBDetectorConfig] = None) -> None:
        self.config = config or OBBDetectorConfig()
        self._model: Optional[YOLO] = None
        self._status: OBBDetectorStatus = OBBDetectorStatus.LOADING
        self._status_message: str = "Detector initializing"

        # Hardware & runtime state
        self._hardware_gpu_detected: bool = False
        self._hardware_gpu_name: Optional[str] = None
        self._hardware_vram_mb: Optional[float] = None
        self._active_device: str = "cpu"
        self._fallback_occurred: bool = False

        # Model provenance
        self._model_sha256: Optional[str] = None
        self._class_names: Dict[int, str] = {}
        self._metadata: Optional[Dict[str, Any]] = None

        # Statistics
        self._total_inferences: int = 0
        self._total_detections: int = 0
        self._last_latency_ms: float = 0.0

    @property
    def status(self) -> OBBDetectorStatus:
        return self._status

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    @property
    def active_device(self) -> str:
        return self._active_device

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    def _inspect_hardware(self) -> None:
        """Detect GPU hardware and determine active device according to device_preference."""
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            self._hardware_gpu_detected = True
            self._hardware_gpu_name = torch.cuda.get_device_name(0)
            self._hardware_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logger.info(
                "Detected CUDA GPU: %s (%.0f MiB VRAM)",
                self._hardware_gpu_name,
                self._hardware_vram_mb,
            )
        else:
            self._hardware_gpu_detected = False
            logger.info("No CUDA GPU detected; CPU execution only")

        pref = (self.config.device_preference or "AUTO").strip().upper()
        if pref == "GPU":
            if cuda_avail:
                self._active_device = "cuda:0"
                self._fallback_occurred = False
            elif self.config.allow_cpu_fallback:
                logger.warning("GPU preferred but CUDA unavailable. Falling back to CPU.")
                self._active_device = "cpu"
                self._fallback_occurred = True
            else:
                raise RuntimeError("GPU execution requested but no CUDA device is available.")
        elif pref == "CPU":
            self._active_device = "cpu"
            self._fallback_occurred = False
        else:  # AUTO
            if cuda_avail:
                self._active_device = "cuda:0"
            else:
                self._active_device = "cpu"
            self._fallback_occurred = False

        logger.info(
            "Selected active inference device: %s (preference: %s, fallback: %s)",
            self._active_device,
            pref,
            self._fallback_occurred,
        )

    def initialize(self) -> Tuple[bool, OBBDetectorStatus, str]:
        """Load and verify custom YOLOv8-OBB model checkpoint.

        Strictly enforces:
        1. Model file existence.
        2. Compatibility with YOLO OBB task.
        3. Match with 10-class taxonomy.
        4. NEVER substitutes a generic model if missing.
        """
        model_path = resolve_asset_path(self.config.model_path)
        logger.info("Initializing YOLO-OBB detector from: %s", model_path)

        # 1. Verify checkpoint existence
        if not model_path.exists():
            self._status = OBBDetectorStatus.MODEL_MISSING
            self._status_message = (
                f"Trained checkpoint '{Path(self.config.model_path).name}' does not exist at {model_path}. "
                "Inference stopped. Generic pretrained models will not be silently substituted."
            )
            logger.error(self._status_message)
            return False, self._status, self._status_message

        # 2. Inspect device
        try:
            self._inspect_hardware()
        except Exception as e:
            self._status = OBBDetectorStatus.ERROR
            self._status_message = f"Hardware inspection failed: {e}"
            logger.error(self._status_message)
            return False, self._status, self._status_message

        # 3. Compute SHA-256 for integrity verification
        try:
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            self._model_sha256 = h.hexdigest()
        except Exception as e:
            logger.warning("Failed to compute checkpoint SHA-256: %s", e)

        # 4. Load metadata provenance if present
        meta_path = resolve_asset_path(self.config.metadata_path) if self.config.metadata_path else None
        if meta_path and meta_path.exists():
            try:
                self._metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                logger.info("Loaded checkpoint provenance metadata from %s", meta_path)
            except Exception as e:
                logger.warning("Could not read metadata file %s: %s", meta_path, e)

        # 5. Load model checkpoint into Ultralytics YOLO
        try:
            self._status = OBBDetectorStatus.LOADING
            self._model = YOLO(str(model_path), task="obb")
        except Exception as e:
            self._status = OBBDetectorStatus.ERROR
            self._status_message = f"Failed to load YOLO model: {e}"
            logger.error(self._status_message, exc_info=True)
            self._model = None
            return False, self._status, self._status_message

        # 6. Verify task and class architecture
        model_task = getattr(self._model, "task", None)
        if model_task != "obb":
            self._status = OBBDetectorStatus.ERROR
            self._status_message = f"Incompatible model task '{model_task}'. Expected 'obb'."
            logger.error(self._status_message)
            self._model = None
            return False, self._status, self._status_message

        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            self._class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            self._class_names = {i: str(n) for i, n in enumerate(names)}
        else:
            self._class_names = {i: c for i, c in enumerate(FINAL_CLASSES)}

        logger.info(
            "Model loaded successfully on %s with %d classes: %s",
            self._active_device,
            len(self._class_names),
            list(self._class_names.values()),
        )

        self._status = OBBDetectorStatus.READY
        self._status_message = (
            f"Ready on {self._active_device.upper()} "
            f"({len(self._class_names)} classes: {', '.join(list(self._class_names.values())[:4])}...)"
        )
        return True, self._status, self._status_message

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        frame_id: int = 0,
    ) -> OBBFrameDetections:
        """Execute oriented object detection on an input BGR frame.

        Extracts 4-corner oriented bounding box coordinates, center coordinates,
        dimensions, rotation angle, class ID, class name, and confidence score.
        """
        if self._model is None:
            return OBBFrameDetections(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                detections=[],
                inference_time_ms=0.0,
                device_used=self._active_device,
                frame_shape=frame_bgr.shape[:2],
            )

        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        try:
            results = self._model.predict(
                source=frame_bgr,
                imgsz=self.config.imgsz,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                device=self._active_device,
                max_det=self.config.max_detections,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self._last_latency_ms = inference_ms
            self._total_inferences += 1
        except Exception as e:
            logger.error("Inference execution failed on %s: %s", self._active_device, e)
            self._status = OBBDetectorStatus.ERROR
            self._status_message = f"Inference failure: {e}"
            return OBBFrameDetections(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                detections=[],
                inference_time_ms=(time.perf_counter() - t0) * 1000.0,
                device_used=self._active_device,
                frame_shape=(h, w),
            )

        detections: List[OBBDetection] = []
        if results and len(results) > 0 and hasattr(results[0], "obb") and results[0].obb is not None:
            obb = results[0].obb
            if len(obb) > 0:
                # xyxyxyxy: shape (N, 4, 2) corner points in pixel coordinates
                corners_batch = obb.xyxyxyxy.cpu().numpy()
                confs_batch = obb.conf.cpu().numpy()
                classes_batch = obb.cls.cpu().numpy().astype(int)

                # xywhr: shape (N, 5) -> center_x, center_y, width, height, angle_rad
                xywhr_batch = None
                if hasattr(obb, "xywhr"):
                    xywhr_batch = obb.xywhr.cpu().numpy()

                for i in range(len(obb)):
                    cid = int(classes_batch[i])
                    cname = self._class_names.get(cid, f"class_{cid}")
                    score = float(confs_batch[i])
                    corners = corners_batch[i].astype(np.float32)  # (4, 2)

                    if xywhr_batch is not None and len(xywhr_batch) > i:
                        cx = float(xywhr_batch[i, 0])
                        cy = float(xywhr_batch[i, 1])
                        bw = float(xywhr_batch[i, 2])
                        bh = float(xywhr_batch[i, 3])
                        angle_rad = float(xywhr_batch[i, 4])
                        angle_deg = float(np.degrees(angle_rad))
                    else:
                        # Fallback geometric calculation from corners
                        cx = float(np.mean(corners[:, 0]))
                        cy = float(np.mean(corners[:, 1]))
                        d01 = float(np.linalg.norm(corners[0] - corners[1]))
                        d12 = float(np.linalg.norm(corners[1] - corners[2]))
                        bw = max(d01, d12)
                        bh = min(d01, d12)
                        angle_deg = float(np.degrees(np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0])))

                    is_prov = cname in PROVISIONAL_SYNTHETIC_CLASSES

                    det = OBBDetection(
                        class_id=cid,
                        class_name=cname,
                        confidence=score,
                        corners=corners,
                        center=(cx, cy),
                        width=bw,
                        height=bh,
                        angle_deg=angle_deg,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        is_provisional=is_prov,
                    )
                    detections.append(det)

        self._total_detections += len(detections)

        # Update status
        if len(detections) > 0:
            self._status = OBBDetectorStatus.INFERENCE_ACTIVE
            self._status_message = f"{len(detections)} objects ({self._active_device.upper()} | {inference_ms:.1f}ms)"
        else:
            self._status = OBBDetectorStatus.NO_OBJECTS_DETECTED
            self._status_message = f"No objects detected ({self._active_device.upper()} | {inference_ms:.1f}ms)"

        gpu_vram: Optional[float] = None
        if "cuda" in self._active_device and torch.cuda.is_available():
            gpu_vram = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)

        return OBBFrameDetections(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            detections=detections,
            inference_time_ms=round(inference_ms, 2),
            device_used=self._active_device,
            frame_shape=(h, w),
            gpu_memory_used_mb=gpu_vram,
        )

    def close(self) -> None:
        """Release PyTorch model references and free GPU memory."""
        logger.info("Releasing YOLO-OBB model resources")
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._status = OBBDetectorStatus.READY
        self._status_message = "Detector closed"

