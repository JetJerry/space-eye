"""Pydantic configuration schemas and validation rules for VYOM."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class InferenceDevice(str, Enum):
    """Allowed inference compute device targets."""
    AUTO = "AUTO"
    CPU = "CPU"
    GPU = "GPU"


class GPUBackend(str, Enum):
    """Supported GPU execution providers on Windows."""
    CUDA = "CUDA"
    DIRECTML = "DIRECTML"
    WINML = "WINML"


class InferenceConfig(BaseModel):
    """Configuration for inference device selection and GPU fallback policies."""
    device_preference: InferenceDevice = InferenceDevice.AUTO
    gpu_backend: GPUBackend = GPUBackend.CUDA
    allow_cpu_fallback: bool = True
    preferred_gpu_index: int = Field(default=0, ge=0)
    min_vram_mb: int = Field(default=2048, ge=512, description="Minimum free VRAM required for GPU selection")
    fp16_enabled: bool = Field(default=False, description="Enable FP16 half-precision on compatible GPUs")
    cpu_num_threads: int = Field(default=4, ge=1, le=64, description="Threads allocated for CPU inference")

    @field_validator("device_preference", mode="before")
    @classmethod
    def normalize_device_preference(cls, v: str | InferenceDevice) -> InferenceDevice:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in InferenceDevice.__members__:
                return InferenceDevice[v_upper]
            raise ValueError(f"Invalid device_preference '{v}'. Must be one of: AUTO, CPU, GPU")
        return v

    @field_validator("gpu_backend", mode="before")
    @classmethod
    def normalize_gpu_backend(cls, v: str | GPUBackend) -> GPUBackend:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in GPUBackend.__members__:
                return GPUBackend[v_upper]
            raise ValueError(f"Invalid gpu_backend '{v}'. Must be one of: CUDA, DIRECTML, WINML")
        return v


class ModelPathsConfig(BaseModel):
    """Configuration of local offline model asset paths."""
    models_dir: Path = Field(default=Path("models"))
    object_detector_path: Optional[Path] = Field(default=Path("models/object_detector.onnx"))
    yolo_pt_path: Optional[Path] = Field(default=Path("models/yolov8_obb.pt"))
    hand_landmarker_path: Optional[Path] = Field(default=Path("models/hand_landmarker.task"))
    verify_checksums: bool = True


class DatasetPathsConfig(BaseModel):
    """Configuration of dataset paths including the external ENIGMA-360 archive."""
    data_dir: Path = Field(default=Path("data"))
    enigma360_path: Path = Field(
        default=Path("D:/space_eye/datasets/ENIGMA-360"),
        description="External directory for ENIGMA-360 dataset. Kept read-only."
    )
    enigma360_subset: str = Field(default="exo", description="Selected subset: exocentric ('exo')")
    enigma360_read_only: bool = Field(default=True, description="Strict protection against modifying source")
    raw_dir: Path = Field(default=Path("data/raw"))
    annotations_dir: Path = Field(default=Path("data/annotations"))
    yolo_obb_dir: Path = Field(default=Path("data/yolo_obb"))


class CameraRecordingConfig(BaseModel):
    """Local video recording configuration."""
    enabled: bool = False
    output_dir: Path = Field(default=Path("sessions/recordings"))
    codec: str = "mp4v"
    container: str = "mp4"


class CameraConfig(BaseModel):
    """Camera device, stream, and frame acquisition parameters."""
    source_type: Literal["device_index", "video_file", "rtsp_stream"] = "device_index"
    device_index: int = Field(default=0, ge=0)
    video_file_path: Optional[Path] = None
    rtsp_url: Optional[str] = None
    width: int = Field(default=1280, ge=320)
    height: int = Field(default=720, ge=240)
    target_fps: int = Field(default=30, ge=1, le=120)
    auto_reconnect: bool = True
    reconnect_timeout_seconds: float = Field(default=5.0, ge=0.5)
    recording: CameraRecordingConfig = Field(default_factory=CameraRecordingConfig)


class RigLandmark(BaseModel):
    """Pre-defined reference position relative to a marker."""
    relative_to_marker: int
    offset_xyz_m: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class RigConfig(BaseModel):
    """ArUco marker dictionary, physical dimensions, and rig reference frame parameters."""
    dictionary_name: str = "DICT_4X4_50"
    marker_size_m: float = Field(default=0.05, gt=0.0, description="Physical marker side length in meters")
    origin_marker_id: int = 0
    marker_ids: List[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    camera_calibration_file: Optional[Path] = None
    reference_frame: str = "rig_relative"
    rig_landmarks: Dict[str, RigLandmark] = Field(default_factory=dict)
    min_markers_for_calibration: int = Field(default=2, ge=1)
    microgravity_mode: bool = True


class LoggingConfig(BaseModel):
    """Structured application logging configuration."""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: Path = Field(default=Path("logs"))
    log_file: str = "vyom.log"
    console_output: bool = True
    json_format: bool = False
    max_bytes: int = Field(default=10485760, ge=1024)
    backup_count: int = Field(default=5, ge=1)


class StorageConfig(BaseModel):
    """Session storage and database parameters."""
    sessions_dir: Path = Field(default=Path("sessions"))
    database_path: Path = Field(default=Path("sessions/vyom_sessions.db"))


class ProtocolStepSchema(BaseModel):
    """Validation schema for an individual procedure step."""
    step_id: str
    order: int = Field(ge=1)
    title: str
    instruction: str
    expected_event: str
    required_objects: List[str] = Field(default_factory=list)
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    stable_duration_ms: int = Field(default=500, ge=0)
    timeout_seconds: int = Field(default=60, ge=1)
    severity: Literal["info", "warning", "critical"] = "warning"
    voice_message: str


class ProtocolSchema(BaseModel):
    """Validation schema for a complete JSON experiment procedure."""
    protocol_id: str
    protocol_version: str
    experiment_name: str
    description: Optional[str] = None
    strict_order: bool = True
    alert_cooldown_seconds: float = Field(default=3.0, ge=0.5)
    steps: List[ProtocolStepSchema] = Field(default_factory=list)


class VyomConfig(BaseModel):
    """Complete root application configuration for VYOM."""
    app_name: str = "VYOM"
    version: str = "0.1.0"
    environment: str = "development"
    offline_mode: bool = True
    debug: bool = False

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    models: ModelPathsConfig = Field(default_factory=ModelPathsConfig)
    datasets: DatasetPathsConfig = Field(default_factory=DatasetPathsConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    rig: RigConfig = Field(default_factory=RigConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    protocols_dir: Path = Field(default=Path("config/protocols"))
    default_protocol: str = "sample_experiment.json"
