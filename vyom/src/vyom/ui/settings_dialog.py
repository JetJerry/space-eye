"""Settings and configuration inspection dialog for VYOM."""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vyom.core.config import ConfigManager
from vyom.core.schemas import InferenceDevice, VyomConfig
from vyom.ui.styles import APPLICATION_STYLESHEET, BG_PRIMARY, TEXT_MUTED


class SettingsDialog(QDialog):
    """Configuration inspector and runtime settings dialog."""

    def __init__(self, config_manager: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_manager = config_manager
        self.config: VyomConfig = config_manager.config

        self.setWindowTitle("VYOM — System Configuration & Hardware Preferences")
        self.setMinimumSize(540, 480)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # 1. Hardware & Compute Group
        hw_group = QGroupBox("Inference & Compute Hardware")
        hw_layout = QFormLayout(hw_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["AUTO", "CPU", "GPU"])
        self.device_combo.setCurrentText(self.config.inference.device_preference.value)
        hw_layout.addRow("Device Preference:", self.device_combo)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["CUDA", "DIRECTML", "WINML"])
        self.backend_combo.setCurrentText(self.config.inference.gpu_backend.value)
        hw_layout.addRow("GPU Backend:", self.backend_combo)

        self.fallback_check = QCheckBox("Allow automatic CPU fallback if GPU fails")
        self.fallback_check.setChecked(self.config.inference.allow_cpu_fallback)
        hw_layout.addRow("CPU Fallback:", self.fallback_check)

        layout.addWidget(hw_group)

        # 2. Camera Configuration Group
        cam_group = QGroupBox("Camera Capture Parameters")
        cam_layout = QFormLayout(cam_group)

        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 9)
        self.cam_index_spin.setValue(self.config.camera.device_index)
        cam_layout.addRow("Device Index:", self.cam_index_spin)

        res_box = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(self.config.camera.width)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(self.config.camera.height)
        res_box.addWidget(self.width_spin)
        res_box.addWidget(QLabel("x"))
        res_box.addWidget(self.height_spin)
        cam_layout.addRow("Resolution (WxH):", res_box)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(self.config.camera.target_fps)
        cam_layout.addRow("Target FPS:", self.fps_spin)

        layout.addWidget(cam_group)

        # 3. ArUco Rig & Fiducial Group
        rig_group = QGroupBox("ArUco Rig Reference")
        rig_layout = QFormLayout(rig_group)

        self.dict_line = QLineEdit(self.config.rig.dictionary_name)
        self.dict_line.setReadOnly(True)
        rig_layout.addRow("Fiducial Dictionary:", self.dict_line)

        self.marker_size_line = QLineEdit(f"{self.config.rig.marker_size_m * 1000:.1f} mm")
        self.marker_size_line.setReadOnly(True)
        rig_layout.addRow("Physical Marker Size:", self.marker_size_line)

        self.marker_ids_line = QLineEdit(str(self.config.rig.marker_ids))
        self.marker_ids_line.setReadOnly(True)
        rig_layout.addRow("Active Marker IDs:", self.marker_ids_line)

        layout.addWidget(rig_group)

        # 4. External Dataset Path Group
        ds_group = QGroupBox("Configured External Dataset")
        ds_layout = QFormLayout(ds_group)
        self.dataset_path_line = QLineEdit(str(self.config.datasets.enigma360_path))
        self.dataset_path_line.setReadOnly(True)
        ds_layout.addRow("ENIGMA-360 Path:", self.dataset_path_line)
        layout.addWidget(ds_group)

        layout.addStretch()

        # Dialog Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._apply_and_close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _apply_and_close(self) -> None:
        """Apply modified preferences back to configuration."""
        dev_pref = self.device_combo.currentText()
        self.config.inference.device_preference = InferenceDevice(dev_pref)
        self.config.inference.allow_cpu_fallback = self.fallback_check.isChecked()
        self.config.camera.device_index = self.cam_index_spin.value()
        self.config.camera.width = self.width_spin.value()
        self.config.camera.height = self.height_spin.value()
        self.config.camera.target_fps = self.fps_spin.value()
        self.accept()

