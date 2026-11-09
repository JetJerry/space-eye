"""Core configuration, schema validation, structured logging, and exceptions."""

from vyom.core.config import (
    ConfigManager,
    HardwareDetector,
    find_project_root,
    get_config_manager,
    load_config,
)
from vyom.core.exceptions import (
    ArUcoCalibrationError,
    CameraError,
    ConfigurationError,
    ConfigurationValidationError,
    DatasetNotFoundError,
    DeviceFallbackWarning,
    HardwareInferenceError,
    ModelNotFoundError,
    ProtocolError,
    VyomError,
)
from vyom.core.logging_setup import get_logger, setup_logging
from vyom.core.schemas import (
    CameraConfig,
    DatasetPathsConfig,
    InferenceConfig,
    InferenceDevice,
    LoggingConfig,
    ModelPathsConfig,
    ProtocolSchema,
    ProtocolStepSchema,
    RigConfig,
    VyomConfig,
)

__all__ = [
    "ArUcoCalibrationError",
    "CameraConfig",
    "CameraError",
    "ConfigManager",
    "ConfigurationError",
    "ConfigurationValidationError",
    "DatasetNotFoundError",
    "DatasetPathsConfig",
    "DeviceFallbackWarning",
    "HardwareDetector",
    "HardwareInferenceError",
    "InferenceConfig",
    "InferenceDevice",
    "LoggingConfig",
    "ModelNotFoundError",
    "ModelPathsConfig",
    "ProtocolError",
    "ProtocolSchema",
    "ProtocolStepSchema",
    "RigConfig",
    "VyomConfig",
    "find_project_root",
    "get_config_manager",
    "get_logger",
    "load_config",
    "setup_logging",
]

