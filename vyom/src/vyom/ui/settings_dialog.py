"""Settings and configuration inspection dialog for VYOM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional
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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vyom.alerts.voice_engine import OfflineVoiceEngine
from vyom.core.config import ConfigManager
from vyom.core.schemas import InferenceDevice, VyomConfig
from vyom.ui.styles import APPLICATION_STYLESHEET, BG_PRIMARY, COLOR_ACCENT, TEXT_MUTED


class SettingsDialog(QDialog):
    """Configuration inspector and runtime settings dialog."""

    def __init__(
        self,
        config_manager: ConfigManager,
        parent: QWidget | None = None,
        alert_manager: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self.config_manager = config_manager
        self.config: VyomConfig = config_manager.config
        self.alert_manager = alert_manager

        self.setWindowTitle("VYOM — System Configuration & Hardware Preferences")
        self.setMinimumSize(560, 600)
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

        # 3. Offline Voice Alerts & Guidance Group (Prompt 10)
        voice_group = QGroupBox("Offline Voice Alerts & Guidance (SAPI5)")
        voice_layout = QFormLayout(voice_group)

        self.voice_enabled_check = QCheckBox("Enable offline spoken alerts & guidance")
        self.voice_enabled_check.setChecked(self.config.voice_alerts.enabled)
        voice_layout.addRow("Voice Alerts:", self.voice_enabled_check)

        self.voice_combo = QComboBox()
        available_voices = OfflineVoiceEngine.get_available_voices()
        current_voice_id = self.config.voice_alerts.voice_id

        if available_voices:
            for v in available_voices:
                self.voice_combo.addItem(f"{v.name}", v.voice_id)
            if current_voice_id:
                idx = self.voice_combo.findData(current_voice_id)
                if idx >= 0:
                    self.voice_combo.setCurrentIndex(idx)
        else:
            self.voice_combo.addItem("No offline SAPI5 voices detected", None)
            self.voice_combo.setEnabled(False)

        voice_layout.addRow("Installed Voice:", self.voice_combo)

        # Speed and Volume
        speech_row = QHBoxLayout()
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(100, 300)
        self.rate_spin.setValue(self.config.voice_alerts.rate)
        self.rate_spin.setSuffix(" WPM")

        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(0, 100)
        self.volume_spin.setValue(int(self.config.voice_alerts.volume * 100))
        self.volume_spin.setSuffix(" %")

        speech_row.addWidget(QLabel("Speed:"))
        speech_row.addWidget(self.rate_spin)
        speech_row.addSpacing(16)
        speech_row.addWidget(QLabel("Volume:"))
        speech_row.addWidget(self.volume_spin)
        voice_layout.addRow("Speech Rate & Vol:", speech_row)

        # Test Voice Button & Status
        test_row = QHBoxLayout()
        self.test_voice_btn = QPushButton("🔊 Test Voice")
        self.test_voice_btn.setStyleSheet("padding: 4px 12px; font-weight: 600;")
        self.test_voice_btn.clicked.connect(self._test_voice)
        self.voice_status_label = QLabel("Status: VOICE: READY")
        self.voice_status_label.setStyleSheet("color: #8fa1b6; font-size: 11px;")
        test_row.addWidget(self.test_voice_btn)
        test_row.addWidget(self.voice_status_label)
        test_row.addStretch()
        voice_layout.addRow("Verification:", test_row)

        layout.addWidget(voice_group)

        # 4. ArUco Rig & External Dataset Group
        rig_group = QGroupBox("Rig & Dataset Reference")
        rig_layout = QFormLayout(rig_group)
        self.dict_line = QLineEdit(f"{self.config.rig.dictionary_name} ({self.config.rig.marker_size_m * 1000:.0f} mm)")
        self.dict_line.setReadOnly(True)
        rig_layout.addRow("Fiducials:", self.dict_line)

        self.dataset_path_line = QLineEdit(str(self.config.datasets.enigma360_path))
        self.dataset_path_line.setReadOnly(True)
        rig_layout.addRow("ENIGMA-360:", self.dataset_path_line)
        layout.addWidget(rig_group)

        layout.addStretch()

        # Dialog Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._apply_and_close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _test_voice(self) -> None:
        """Trigger a test utterance using current settings."""
        if hasattr(self.alert_manager, "speak_test_message"):
            # Update temporary settings for test
            sel_voice = self.voice_combo.currentData()
            rate = self.rate_spin.value()
            vol = self.volume_spin.value() / 100.0
            if sel_voice and hasattr(self.alert_manager, "set_voice"):
                self.alert_manager.set_voice(sel_voice)
            if hasattr(self.alert_manager, "set_rate"):
                self.alert_manager.set_rate(rate)
            if hasattr(self.alert_manager, "set_volume"):
                self.alert_manager.set_volume(vol)
            self.alert_manager.speak_test_message()
            self.voice_status_label.setText("Status: VOICE: SPEAKING (Test)")
        else:
            self.voice_status_label.setText("Status: VOICE: TEST (Simulated)")

    def _apply_and_close(self) -> None:
        """Apply modified preferences back to configuration."""
        dev_pref = self.device_combo.currentText()
        self.config.inference.device_preference = InferenceDevice(dev_pref)
        self.config.inference.allow_cpu_fallback = self.fallback_check.isChecked()
        self.config.camera.device_index = self.cam_index_spin.value()
        self.config.camera.width = self.width_spin.value()
        self.config.camera.height = self.height_spin.value()
        self.config.camera.target_fps = self.fps_spin.value()

        # Apply voice alert settings
        self.config.voice_alerts.enabled = self.voice_enabled_check.isChecked()
        sel_voice = self.voice_combo.currentData()
        if sel_voice:
            self.config.voice_alerts.voice_id = sel_voice
        self.config.voice_alerts.rate = self.rate_spin.value()
        self.config.voice_alerts.volume = self.volume_spin.value() / 100.0

        if self.alert_manager and hasattr(self.alert_manager, "worker"):
            self.alert_manager.worker.update_config(
                enabled=self.config.voice_alerts.enabled,
                voice_id=self.config.voice_alerts.voice_id,
                rate=self.config.voice_alerts.rate,
                volume=self.config.voice_alerts.volume,
            )

        self.accept()
