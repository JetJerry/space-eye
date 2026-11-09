"""Configuration management and hardware inspection for VYOM.

Handles JSON configuration loading, environment overrides, schema validation,
dataset path resolution, and multi-tier hardware inspection distinguishing
physical GPU detection from actual inference provider availability.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
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
    GPUBackend,
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


@dataclass
class HardwareReport:
    """Comprehensive multi-tier hardware and inference runtime inspection report.

    Distinguishes physical GPU hardware detection from inference runtime installation
    and execution-provider registration.
    """
    gpu_hardware_detected: bool
    gpu_name: Optional[str]
    gpu_vram_mb: Optional[int]
    inference_runtime_installed: str
    gpu_execution_provider_available: bool
    gpu_execution_provider_name: Optional[str]
    active_device_selected: str  # "CPU" or "GPU"
    device_selection_reason: str

    def summary_line(self) -> str:
        """One-line concise hardware summary for status bars."""
        hw = self.gpu_name if self.gpu_hardware_detected else "None"
        prov = self.gpu_execution_provider_name if self.gpu_execution_provider_available else "None"
        return f"Hardware: {hw} | Runtime: {self.inference_runtime_installed} | Provider: {prov} | Active: {self.active_device_selected}"


class HardwareDetector:
    """Inspects compute hardware and distinguishes physical devices from active inference providers."""

    @staticmethod
    def detect_physical_gpu() -> Tuple[bool, Optional[str], Optional[int]]:
        """Inspect physical GPU hardware via nvidia-smi on Windows.

        Returns:
            Tuple of (detected: bool, gpu_name: Optional[str], vram_mb: Optional[int])
        """
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                res = subprocess.run(
                    [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    first_line = res.stdout.strip().split("\n")[0]
                    parts = [p.strip() for p in first_line.split(",")]
                    name = parts[0]
                    vram = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                    return True, name, vram
            except Exception:
                pass
        return False, None, None

    @staticmethod
    def inspect_inference_runtime() -> Tuple[str, bool, Optional[str]]:
        """Check installed inference runtimes and actual CUDA execution provider registration.

        Returns:
            Tuple of (runtime_installed: str, cuda_provider_available: bool, provider_name: Optional[str])
        """
        runtime_desc = "None (No inference runtime loaded)"
        cuda_provider_available = False
        provider_name = None

        # 1. Check ONNX Runtime (Deployment Inference Engine)
        try:
            import onnxruntime as ort  # type: ignore

            ver = getattr(ort, "__version__", "unknown")
            runtime_desc = f"ONNX Runtime {ver}"
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                cuda_provider_available = True
                provider_name = "CUDAExecutionProvider"
            elif "DmlExecutionProvider" in providers:
                provider_name = "DmlExecutionProvider"
            return runtime_desc, cuda_provider_available, provider_name
        except ImportError:
            pass

        # 2. Check PyTorch (Training / Validation Engine)
        try:
            import torch  # type: ignore

            ver = getattr(torch, "__version__", "unknown")
            runtime_desc = f"PyTorch {ver}"
            if torch.cuda.is_available():
                cuda_provider_available = True
                provider_name = "torch.cuda"
            return runtime_desc, cuda_provider_available, provider_name
        except ImportError:
            pass

        return runtime_desc, False, None

    @classmethod
    def generate_report(
        cls,
        preference: InferenceDevice = InferenceDevice.AUTO,
        backend: GPUBackend = GPUBackend.CUDA,
        allow_cpu_fallback: bool = True,
    ) -> HardwareReport:
        """Generate a complete multi-tier hardware inspection report with safe device selection."""
        has_gpu, gpu_name, vram_mb = cls.detect_physical_gpu()
        runtime_desc, provider_available, provider_name = cls.inspect_inference_runtime()

        # Decision Logic:
        # A detected physical GPU must NEVER automatically be reported as an active CUDA device
        # unless an actual execution provider is confirmed operational.
        if preference == InferenceDevice.CPU:
            selected_device = "CPU"
            reason = "Explicitly configured for CPU inference."
        elif preference == InferenceDevice.GPU:
            if provider_available:
                selected_device = "GPU"
                reason = f"GPU execution provider active ({provider_name}) on {gpu_name}."
            elif allow_cpu_fallback:
                selected_device = "CPU"
                gpu_msg = f" (Physical GPU: {gpu_name})" if has_gpu else ""
                reason = f"GPU preference requested but CUDA execution provider not installed{gpu_msg}. Safe CPU fallback active."
                warnings.warn(reason, category=DeviceFallbackWarning, stacklevel=2)
            else:
                raise HardwareInferenceError(
                    f"GPU preference requested but no CUDA execution provider is available (Physical GPU: {gpu_name}). CPU fallback disabled."
                )
        else:  # AUTO Preference
            if provider_available:
                selected_device = "GPU"
                reason = f"AUTO selected verified GPU execution provider ({provider_name}) on {gpu_name}."
            else:
                selected_device = "CPU"
                if has_gpu:
                    reason = f"AUTO selected CPU fallback. Physical GPU ({gpu_name}) detected, but CUDA execution provider is not installed."
                else:
                    reason = "AUTO selected CPU (no compatible GPU hardware detected)."

        return HardwareReport(
            gpu_hardware_detected=has_gpu,
            gpu_name=gpu_name,
            gpu_vram_mb=vram_mb,
            inference_runtime_installed=runtime_desc,
            gpu_execution_provider_available=provider_available,
            gpu_execution_provider_name=provider_name,
            active_device_selected=selected_device,
            device_selection_reason=reason,
        )


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

    def get_hardware_report(self, config: Optional[InferenceConfig] = None) -> HardwareReport:
        """Inspect hardware and generate the 4-tier hardware report."""
        inf = config or self.config.inference
        return HardwareDetector.generate_report(
            preference=inf.device_preference,
            backend=inf.gpu_backend,
            allow_cpu_fallback=inf.allow_cpu_fallback,
        )

    def resolve_inference_device(self, config: Optional[InferenceConfig] = None) -> Tuple[str, str]:
        """Determine effective execution device and reason string."""
        report = self.get_hardware_report(config)
        return report.active_device_selected, report.device_selection_reason

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
