"""Left-side session controls and video source management panel."""

from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vyom.ui.styles import (
    BG_CARD,
    BORDER_SUBTLE,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    StatusBadge,
    TEXT_MUTED,
    TEXT_SECONDARY,
)


class SessionPanel(QFrame):
    """Left navigation panel managing video sources, session lifecycle, and protocols."""

    start_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()
    source_changed = Signal(str, object)
    protocol_changed = Signal(str)
    recording_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # 1. Video Source Selection Group
        source_group = QGroupBox("Camera / Video Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Webcam (Device Index)", "Local Video File", "Network Stream (RTSP)"])
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        source_layout.addWidget(self.source_type_combo)

        # Device index sub-row
        self.device_row = QWidget()
        dev_layout = QHBoxLayout(self.device_row)
        dev_layout.setContentsMargins(0, 0, 0, 0)
        dev_layout.addWidget(QLabel("Device Index:"))
        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 9)
        self.device_spin.setValue(0)
        dev_layout.addWidget(self.device_spin)
        source_layout.addWidget(self.device_row)

        # Video file sub-row
        self.file_row = QWidget()
        file_layout = QHBoxLayout(self.file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #65768d; font-size: 11px;")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(self.browse_btn)
        self.file_row.setVisible(False)
        source_layout.addWidget(self.file_row)

        # Recording Checkbox
        self.record_checkbox = QCheckBox("Record Session Locally")
        self.record_checkbox.toggled.connect(self.recording_toggled.emit)
        source_layout.addWidget(self.record_checkbox)

        layout.addWidget(source_group)

        # 2. Session Lifecycle Group
        session_group = QGroupBox("Session Controls")
        session_layout = QVBoxLayout(session_group)
        session_layout.setSpacing(8)

        self.start_btn = QPushButton("▶  Start Session")
        self.start_btn.setObjectName("btnSuccess")
        self.start_btn.clicked.connect(self.start_requested.emit)
        session_layout.addWidget(self.start_btn)

        control_row = QHBoxLayout()
        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setObjectName("btnWarning")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_requested.emit)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("btnDanger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        control_row.addWidget(self.pause_btn)
        control_row.addWidget(self.stop_btn)
        session_layout.addLayout(control_row)

        self.reset_btn = QPushButton("↺  Reset Session")
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        session_layout.addWidget(self.reset_btn)

        layout.addWidget(session_group)

        # 3. Protocol Selection Group
        protocol_group = QGroupBox("Experiment Protocol")
        protocol_layout = QVBoxLayout(protocol_group)
        protocol_layout.setSpacing(6)

        self.protocol_combo = QComboBox()
        self.protocol_combo.currentTextChanged.connect(self.protocol_changed.emit)
        protocol_layout.addWidget(self.protocol_combo)

        self.protocol_status_label = QLabel("Status: PROTOCOL NOT SELECTED")
        self.protocol_status_label.setStyleSheet("color: #d29922; font-size: 11px; font-weight: 600;")
        protocol_layout.addWidget(self.protocol_status_label)

        layout.addWidget(protocol_group)

        layout.addStretch()

        # 4. Explicit Subsystem Diagnostics Summary Card
        status_box = QFrame()
        status_box.setObjectName("cardFrame")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(6)

        status_title = QLabel("SYSTEM STATUS")
        status_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #9fb0c3; letter-spacing: 0.5px;")
        status_layout.addWidget(status_title)

        self.cam_status_badge = StatusBadge("CAMERA NOT CONNECTED", color=COLOR_DANGER)
        self.model_status_badge = StatusBadge("MODEL NOT LOADED", color=COLOR_WARNING)
        status_layout.addWidget(self.cam_status_badge)
        status_layout.addWidget(self.model_status_badge)

        layout.addWidget(status_box)

    def set_session_state(self, state: str) -> None:
        """Update button enabled states according to active session lifecycle."""
        st = state.upper()
        if st in ("ACTIVE", "RUNNING"):
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(True)
            self.source_type_combo.setEnabled(False)
            self.cam_status_badge.set_badge_color(COLOR_SUCCESS, "CAMERA ACTIVE")
        elif st == "PAUSED":
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("▶  Resume")
            self.stop_btn.setEnabled(True)
            self.source_type_combo.setEnabled(False)
            self.cam_status_badge.set_badge_color(COLOR_WARNING, "CAMERA PAUSED")
        else:  # IDLE / STOPPED
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(False)
            self.source_type_combo.setEnabled(True)
            self.cam_status_badge.set_badge_color(COLOR_DANGER, "CAMERA NOT CONNECTED")

    def set_available_protocols(self, protocol_names: list[str], default_selected: str | None = None) -> None:
        """Populate the protocol selector dropdown."""
        self.protocol_combo.clear()
        self.protocol_combo.addItems(protocol_names)
        if default_selected and default_selected in protocol_names:
            self.protocol_combo.setCurrentText(default_selected)
            self.protocol_status_label.setText(f"Status: LOADED ({default_selected})")
            self.protocol_status_label.setStyleSheet("color: #2ea043; font-size: 11px; font-weight: 600;")
        elif not protocol_names:
            self.protocol_status_label.setText("Status: PROTOCOL NOT SELECTED")
            self.protocol_status_label.setStyleSheet("color: #d29922; font-size: 11px; font-weight: 600;")

    def _on_source_type_changed(self, index: int) -> None:
        is_device = (index == 0)
        is_file = (index == 1)
        self.device_row.setVisible(is_device)
        self.file_row.setVisible(is_file)

        if is_device:
            self.source_changed.emit("device_index", self.device_spin.value())
        elif is_file:
            self.source_changed.emit("video_file", self.file_path_label.text())
        else:
            self.source_changed.emit("rtsp_stream", "rtsp://localhost:8554/live")

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experiment Video Recording",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)",
        )
        if path:
            filename = Path(path).name
            self.file_path_label.setText(filename)
            self.file_path_label.setToolTip(path)
            self.source_changed.emit("video_file", path)
