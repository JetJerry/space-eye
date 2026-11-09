"""Configuration management and validation for VYOM.

Handles JSON configuration file loading, environment overrides, schema validation,
dataset path resolution, and inference device selection with graceful CPU fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vyom.core.exceptions import (
    ConfigurationError,
    ConfigurationValidationError,
    DatasetNotFoundError,
    DeviceFallbackWarning,
    HardwareInferenceError,
    ProtocolError,
)
from vyom.core.logging_setup import get_logger
from vyom.core.schemas import (
    CameraConfig,
    InferenceConfig,
    InferenceDevice,
    ProtocolSchema,
    RigConfig,
    VyomConfig,
)

logger = get_logger("core.config")


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Detect the VYOM repository root directory by locating pyproject.toml or config/."""
    current = (start_path or Path.cwd()).resolve()

    # Search current directory and parent directories
    for path in [current] + list(current.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "vyom").exists():
            return path
        if (path / "config" / "app.json").exists():
            return path

    # If script is inside src/vyom/core
    module_root = Path(__file__).resolve().parents[3]
    if (module_root / "pyproject.toml").exists():
        return module_root

    # Default to current directory
    return current


class HardwareDetector:
    """Inspects local compute hardware (CPU, NVIDIA GPU) safely without heavy dependencies."""

    @staticmethod
    def is_cuda_available() -> Tuple[bool, str]:
        """Detect whether NVIDIA CUDA acceleration is genuinely usable.

        Returns:
            Tuple of (is_available, description_or_reason)
        """
        # First check via torch if it is installed
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "CUDA Device"
                return True, f"PyTorch CUDA active: {device_name} ({device_count} device(s))"
            return False, "PyTorch installed but CUDA is not available on this system"
        except ImportError:
            pass

        # Check via onnxruntime if installed
        try:
            import onnxruntime as ort  # type: ignore

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return True, "ONNX Runtime CUDAExecutionProvider is available"
        except ImportError:
            pass

        # Fallback inspection: check nvidia-smi on Windows
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                res = subprocess.run(
                    [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    gpu_info = res.stdout.strip().split("\n")[0]
                    return True, f"NVIDIA GPU detected via nvidia-smi: {gpu_info} (PyTorch/CUDA runtime not yet loaded)"
            except Exception:
                pass

        return False, "No active NVIDIA CUDA GPU detected; CPU execution provider recommended"


class ConfigManager:
    """Manages loading, validating, and querying VYOM application configurations."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = (project_root or find_project_root()).resolve()
        self.config_dir = self.project_root / "config"
        self._config: Optional[VyomConfig] = None

    @property
    def config(self) -> VyomConfig:
        """Access the loaded configuration singleton."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(
        self,
        config_dir: Optional[Path | str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> VyomConfig:
        """Load and merge configuration from JSON files, environment variables, and overrides."""
        target_dir = Path(config_dir).resolve() if config_dir else self.config_dir

        app_json_path = target_dir / "app.json"
        camera_json_path = target_dir / "camera.json"
        rig_json_path = target_dir / "rig.json"

        # 1. Base App Config
        app_data: Dict[str, Any] = {}
        if app_json_path.exists():
            try:
                with open(app_json_path, "r", encoding="utf-8") as f:
                    app_data = json.load(f)
            except Exception as e:
                raise ConfigurationError(f"Failed to read app configuration file {app_json_path}: {e}") from e

        # 2. Camera Config
        if camera_json_path.exists():
            try:
                with open(camera_json_path, "r", encoding="utf-8") as f:
                    app_data["camera"] = json.load(f)
            except Exception as e:
                raise ConfigurationError(f"Failed to read camera configuration {camera_json_path}: {e}") from e

        # 3. Rig Config
        if rig_json_path.exists():
            try:
                with open(rig_json_path, "r", encoding="utf-8") as f:
                    app_data["rig"] = json.load(f)
            except Exception as e:
                raise ConfigurationError(f"Failed to read rig configuration {rig_json_path}: {e}") from e

        # 4. Environment Variable Overrides (Prefix: VYOM_)
        self._apply_env_overrides(app_data)

        # 5. Programmatic Overrides
        if overrides:
            self._deep_update(app_data, overrides)

        # 6. Validate via Pydantic
        try:
            validated_config = VyomConfig(**app_data)
        except Exception as e:
            raise ConfigurationValidationError(f"Configuration schema validation failed: {e}") from e

        # 7. Normalize relative paths against project root
        self._resolve_paths(validated_config)

        self._config = validated_config
        logger.info(
            "Configuration loaded successfully for %s v%s (environment: %s)",
            validated_config.app_name,
            validated_config.version,
            validated_config.environment,
        )
        return validated_config

    def resolve_inference_device(self, config: Optional[InferenceConfig] = None) -> Tuple[str, str]:
        """Determine the actual execution device and provider based on preference and hardware.

        Returns:
            Tuple of (effective_device: "CPU" | "GPU", provider_description: str)
        """
        inf = config or self.config.inference
        cuda_ok, cuda_msg = HardwareDetector.is_cuda_available()

        if inf.device_preference == InferenceDevice.CPU:
            return "CPU", "Explicitly configured for CPU execution"

        if inf.device_preference == InferenceDevice.GPU:
            if cuda_ok:
                return "GPU", f"Using GPU backend ({inf.gpu_backend}): {cuda_msg}"
            if inf.allow_cpu_fallback:
                warnings.warn(
                    f"GPU preference requested but unavailable ({cuda_msg}). Falling back to CPU.",
                    category=DeviceFallbackWarning,
                    stacklevel=2,
                )
                logger.warning("GPU requested but unavailable (%s). Safe fallback to CPU active.", cuda_msg)
                return "CPU", f"Safe CPU fallback (GPU unavailable: {cuda_msg})"
            raise HardwareInferenceError(f"GPU requested but unavailable and CPU fallback is disabled: {cuda_msg}")

        # AUTO Preference: select GPU only if genuinely verified, otherwise safe CPU
        if cuda_ok:
            logger.info("AUTO device preference selected GPU: %s", cuda_msg)
            return "GPU", f"AUTO selected GPU ({inf.gpu_backend}): {cuda_msg}"

        logger.info("AUTO device preference selected CPU: %s", cuda_msg)
        return "CPU", f"AUTO selected CPU: {cuda_msg}"

    def validate_enigma360_dataset(self) -> Dict[str, Any]:
        """Inspect the external ENIGMA-360 dataset path without modifying files."""
        dataset_path = self.config.datasets.enigma360_path

        info: Dict[str, Any] = {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "subset": self.config.datasets.enigma360_subset,
            "read_only": self.config.datasets.enigma360_read_only,
            "annotations_found": False,
            "video_dir_found": False,
            "status": "NOT_FOUND",
        }

        if not dataset_path.exists():
            logger.warning("ENIGMA-360 dataset directory does not exist at: %s", dataset_path)
            return info

        # Inspect non-destructively
        ann_dir = dataset_path / "annotations"
        data_dir = dataset_path / "data"
        info["annotations_found"] = ann_dir.is_dir()
        info["video_dir_found"] = data_dir.is_dir()
        info["status"] = "CONFIGURED_AND_AVAILABLE"

        logger.info(
            "Validated external dataset ENIGMA-360 at %s (annotations: %s, data: %s)",
            dataset_path,
            info["annotations_found"],
            info["video_dir_found"],
        )
        return info

    def load_protocol(self, protocol_name_or_path: str | Path) -> ProtocolSchema:
        """Load and validate an experiment protocol JSON file against ProtocolSchema."""
        target_path = Path(protocol_name_or_path)
        if not target_path.is_absolute():
            # Check relative to protocols_dir
            candidate = self.config.protocols_dir / target_path
            if candidate.exists():
                target_path = candidate
            else:
                target_path = self.project_root / target_path

        if not target_path.exists():
            raise ProtocolError(f"Protocol file not found at: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                protocol_data = json.load(f)
            protocol = ProtocolSchema(**protocol_data)
            logger.info("Loaded protocol '%s' (v%s) with %d steps", protocol.experiment_name, protocol.protocol_version, len(protocol.steps))
            return protocol
        except Exception as e:
            raise ProtocolError(f"Protocol validation failed for {target_path}: {e}") from e

    def _resolve_paths(self, cfg: VyomConfig) -> None:
        """Resolve paths relative to project root if they are not already absolute."""
        root = self.project_root

        def _to_abs(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        cfg.models.models_dir = _to_abs(cfg.models.models_dir)
        if cfg.models.object_detector_path:
            cfg.models.object_detector_path = _to_abs(cfg.models.object_detector_path)
        if cfg.models.yolo_pt_path:
            cfg.models.yolo_pt_path = _to_abs(cfg.models.yolo_pt_path)
        if cfg.models.hand_landmarker_path:
            cfg.models.hand_landmarker_path = _to_abs(cfg.models.hand_landmarker_path)

        cfg.datasets.data_dir = _to_abs(cfg.datasets.data_dir)
        cfg.datasets.raw_dir = _to_abs(cfg.datasets.raw_dir)
        cfg.datasets.annotations_dir = _to_abs(cfg.datasets.annotations_dir)
        cfg.datasets.yolo_obb_dir = _to_abs(cfg.datasets.yolo_obb_dir)

        cfg.logging.log_dir = _to_abs(cfg.logging.log_dir)
        cfg.storage.sessions_dir = _to_abs(cfg.storage.sessions_dir)
        cfg.storage.database_path = _to_abs(cfg.storage.database_path)
        cfg.protocols_dir = _to_abs(cfg.protocols_dir)

    def _apply_env_overrides(self, data: Dict[str, Any]) -> None:
        """Override configuration with VYOM_ environment variables."""
        if "VYOM_ENV" in os.environ:
            data["environment"] = os.environ["VYOM_ENV"]

        if "VYOM_LOG_LEVEL" in os.environ:
            data.setdefault("logging", {})["log_level"] = os.environ["VYOM_LOG_LEVEL"]

        if "VYOM_DEVICE_PREFERENCE" in os.environ:
            data.setdefault("inference", {})["device_preference"] = os.environ["VYOM_DEVICE_PREFERENCE"]

        if "VYOM_GPU_BACKEND" in os.environ:
            data.setdefault("inference", {})["gpu_backend"] = os.environ["VYOM_GPU_BACKEND"]

        if "VYOM_ALLOW_CPU_FALLBACK" in os.environ:
            val = os.environ["VYOM_ALLOW_CPU_FALLBACK"].strip().lower()
            data.setdefault("inference", {})["allow_cpu_fallback"] = val in ("1", "true", "yes")

        if "VYOM_ENIGMA360_PATH" in os.environ:
            data.setdefault("datasets", {})["enigma360_path"] = os.environ["VYOM_ENIGMA360_PATH"]

        if "VYOM_ENIGMA360_SUBSET" in os.environ:
            data.setdefault("datasets", {})["enigma360_subset"] = os.environ["VYOM_ENIGMA360_SUBSET"]

    @staticmethod
    def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                ConfigManager._deep_update(source[key], value)
            else:
                source[key] = value
        return source


# Global module singleton helper
_default_manager: Optional[ConfigManager] = None


def get_config_manager(project_root: Optional[Path] = None) -> ConfigManager:
    """Retrieve the global ConfigManager singleton."""
    global _default_manager
    if _default_manager is None or project_root is not None:
        _default_manager = ConfigManager(project_root=project_root)
    return _default_manager


def load_config(config_dir: Optional[Path | str] = None) -> VyomConfig:
    """Convenience function to load and validate configuration."""
    return get_config_manager().load(config_dir=config_dir)

