"""Focused verification of configuration loading, validation, and dataset resolution."""

from pathlib import Path
import pytest

from vyom.core.config import ConfigManager, find_project_root
from vyom.core.exceptions import ConfigurationValidationError, ProtocolError
from vyom.core.schemas import InferenceDevice, VyomConfig


def test_find_project_root():
    root = find_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "vyom").exists()


def test_config_manager_load():
    root = find_project_root()
    manager = ConfigManager(project_root=root)
    cfg = manager.load()

    assert isinstance(cfg, VyomConfig)
    assert cfg.app_name == "VYOM"
    assert cfg.version == "0.1.0"
    assert cfg.offline_mode is True
    assert cfg.inference.device_preference in (InferenceDevice.AUTO, InferenceDevice.CPU, InferenceDevice.GPU)
    assert cfg.camera.target_fps == 30
    assert cfg.rig.dictionary_name == "DICT_4X4_50"


def test_enigma360_dataset_configuration():
    root = find_project_root()
    manager = ConfigManager(project_root=root)
    cfg = manager.load()

    # Verify external ENIGMA-360 dataset configuration
    assert "ENIGMA-360" in str(cfg.datasets.enigma360_path)
    assert cfg.datasets.enigma360_subset == "exo"
    assert cfg.datasets.enigma360_read_only is True

    # Validate dataset inspection (non-destructive)
    info = manager.validate_enigma360_dataset()
    assert "status" in info
    assert info["read_only"] is True


def test_inference_device_resolution():
    root = find_project_root()
    manager = ConfigManager(project_root=root)
    device, desc = manager.resolve_inference_device()

    assert device in ("CPU", "GPU")
    assert isinstance(desc, str) and len(desc) > 0


def test_sample_protocol_loading():
    root = find_project_root()
    manager = ConfigManager(project_root=root)
    cfg = manager.load()

    protocol = manager.load_protocol(cfg.default_protocol)
    assert protocol.protocol_id == "sample-specimen-transfer"
    assert len(protocol.steps) >= 2
    assert protocol.steps[0].step_id == "STEP_01"
    assert protocol.steps[0].expected_event == "OBJECT_PICKED_UP"
    assert protocol.steps[1].step_id == "STEP_02"
    assert protocol.steps[1].expected_event == "OBJECT_PLACED_IN_TARGET"


def test_invalid_device_preference():
    with pytest.raises(Exception):
        VyomConfig(inference={"device_preference": "QUANTUM_ACCELERATOR"})

