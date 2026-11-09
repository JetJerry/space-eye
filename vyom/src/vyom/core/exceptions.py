"""Domain-specific exceptions for VYOM."""


class VyomError(Exception):
    """Base exception for all VYOM errors."""


class ConfigurationError(VyomError):
    """Raised when application or subsystem configuration is missing or corrupt."""


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration values fail schema validation."""


class HardwareInferenceError(VyomError):
    """Raised when requested inference hardware (e.g. CUDA GPU) cannot be initialized."""


class DeviceFallbackWarning(UserWarning):
    """Issued when execution falls back from GPU to CPU."""


class DatasetNotFoundError(VyomError):
    """Raised when a configured dataset directory or subset cannot be found."""


class ModelNotFoundError(VyomError):
    """Raised when an expected model weight or asset file is missing."""


class ProtocolError(VyomError):
    """Raised when experiment protocol definition is invalid or corrupted."""


class CameraError(VyomError):
    """Raised when camera capture source cannot be opened or stream fails."""


class ArUcoCalibrationError(VyomError):
    """Raised when ArUco fiducials or camera calibration parameters are invalid."""

