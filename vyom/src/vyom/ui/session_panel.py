"""Left-side session controls, video source selector, and file metadata display."""

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

from vyom.capture.video_file_source import VideoFileSource
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
    history_requested = Signal()
    source_changed = Signal(str, object)
    protocol_changed = Signal(str)
    recording_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        self.setMinimumWidth(270)
        self.setMaximumWidth(330)

        self._selected_video_file: Path | None = None
        self._default_dataset_dir = Path("D:/space_eye/datasets/ENIGMA-360")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 1. Video Source Selection Group
        source_group = QGroupBox("Camera / Video Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems([
            "Webcam (Device Index)",
            "Local Video File (MP4/AVI)",
            "Network Stream (RTSP)",
        ])
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
        self.device_spin.valueChanged.connect(self._on_device_changed)
        dev_layout.addWidget(self.device_spin)
        source_layout.addWidget(self.device_row)

        # Video file sub-row with metadata
        self.file_row = QWidget()
        file_layout = QVBoxLayout(self.file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(4)

        browse_row = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #8fa1b6; font-size: 11px;")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse_file)
        browse_row.addWidget(self.file_path_label, stretch=1)
        browse_row.addWidget(self.browse_btn)
        file_layout.addLayout(browse_row)

        # Video metadata display
        self.file_meta_label = QLabel("Dimensions: -- | FPS: -- | Duration: --")
        self.file_meta_label.setStyleSheet("color: #65768d; font-size: 10px; font-family: monospace;")
        file_layout.addWidget(self.file_meta_label)

        # Loop checkbox for file playback
        self.loop_checkbox = QCheckBox("Loop Video on End of File")
        self.loop_checkbox.setChecked(True)
        file_layout.addWidget(self.loop_checkbox)

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

        self.history_btn = QPushButton("📜  Session Archive & Reports")
        self.history_btn.setStyleSheet(
            f"background-color: {BG_CARD}; border: 1px solid {BORDER_SUBTLE}; "
            f"padding: 6px 10px; font-weight: 600; border-radius: 4px;"
        )
        self.history_btn.clicked.connect(self.history_requested.emit)
        session_layout.addWidget(self.history_btn)

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

        status_title = QLabel("SUBSYSTEM STATUS")
        status_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #9fb0c3; letter-spacing: 0.5px;")
        status_layout.addWidget(status_title)

        self.cam_status_badge = StatusBadge("CAMERA NOT CONNECTED", color=COLOR_DANGER)
        self.model_status_badge = StatusBadge("MODEL NOT CONFIGURED", color=COLOR_WARNING)
        status_layout.addWidget(self.cam_status_badge)
        status_layout.addWidget(self.model_status_badge)

        layout.addWidget(status_box)

    @property
    def selected_video_file(self) -> Path | None:
        return self._selected_video_file

    @property
    def should_loop_video(self) -> bool:
        return self.loop_checkbox.isChecked()

    @property
    def is_recording_enabled(self) -> bool:
        return self.record_checkbox.isChecked()

    def set_session_state(self, state: str) -> None:
        """Update button enabled states according to active session lifecycle."""
        st = state.upper()
        if st in ("ACTIVE", "RUNNING"):
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(True)
            self.source_type_combo.setEnabled(False)
            self.device_spin.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.record_checkbox.setEnabled(False)
            self.cam_status_badge.set_badge_color(COLOR_SUCCESS, "CAPTURE ACTIVE")
        elif st == "PAUSED":
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("▶  Resume")
            self.stop_btn.setEnabled(True)
            self.source_type_combo.setEnabled(False)
            self.device_spin.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.record_checkbox.setEnabled(False)
            self.cam_status_badge.set_badge_color(COLOR_WARNING, "CAPTURE PAUSED")
        else:  # IDLE / STOPPED
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(False)
            self.source_type_combo.setEnabled(True)
            self.device_spin.setEnabled(True)
            self.browse_btn.setEnabled(True)
            self.record_checkbox.setEnabled(True)
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
            self.source_changed.emit("video_file", str(self._selected_video_file) if self._selected_video_file else "")
        else:
            self.source_changed.emit("rtsp_stream", "rtsp://localhost:8554/live")

    def _on_device_changed(self, val: int) -> None:
        if self.source_type_combo.currentIndex() == 0:
            self.source_changed.emit("device_index", val)

    def _on_browse_file(self) -> None:
        # Default start directory: ENIGMA-360 videos if available
        start_dir = ""
        enigma_videos = self._default_dataset_dir / "data" / "videos"
        if enigma_videos.exists():
            start_dir = str(enigma_videos)
        elif self._default_dataset_dir.exists():
            start_dir = str(self._default_dataset_dir)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experiment Video File",
            start_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)",
        )
        if path:
            self.set_selected_video_file(Path(path))

    def set_selected_video_file(self, file_path: Path) -> None:
        """Assign video file and display its inspected metadata."""
        self._selected_video_file = file_path
        self.file_path_label.setText(file_path.name)
        self.file_path_label.setToolTip(str(file_path))

        # Inspect video metadata using VideoFileSource helper
        source = VideoFileSource(file_path)
        meta = source.get_metadata()
        dur_str = f"{int(meta.duration_seconds // 60):02d}:{int(meta.duration_seconds % 60):02d}" if meta.duration_seconds else "--:--"
        self.file_meta_label.setText(
            f"{meta.width}x{meta.height} | {meta.fps:.1f} FPS | {meta.total_frames or '--'} frames ({dur_str})"
        )

        self.source_changed.emit("video_file", str(file_path))
