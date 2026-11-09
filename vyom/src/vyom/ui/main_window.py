"""Primary desktop application window for VYOM."""

from pathlib import Path
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vyom.core.config import ConfigManager, get_config_manager
from vyom.core.logging_setup import get_logger
from vyom.ui.event_panel import EventPanel
from vyom.ui.live_monitor import LiveMonitor
from vyom.ui.protocol_panel import ProtocolPanel
from vyom.ui.session_panel import SessionPanel
from vyom.ui.settings_dialog import SettingsDialog
from vyom.ui.styles import APPLICATION_STYLESHEET
from vyom.ui.top_bar import TopBar

logger = get_logger("ui.main_window")


class VyomMainWindow(QMainWindow):
    """Main desktop monitoring workstation window."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        super().__init__()
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.config

        self.setWindowTitle("VYOM — Experiment Activity & Procedure Compliance Workstation")
        self.setMinimumSize(1200, 750)
        self.resize(1360, 840)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        self._session_state: str = "IDLE"

        self._init_ui()
        self._init_data()
        self._connect_signals()

        logger.info("VyomMainWindow initialized successfully")

    def _init_ui(self) -> None:
        """Construct the mission control desktop layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Bar
        self.top_bar = TopBar(self)
        root_layout.addWidget(self.top_bar)

        # 2. Main Work Area (Vertical Splitter: Top Splitter + Event Panel)
        v_splitter = QSplitter(Qt.Vertical)

        # Horizontal Splitter: Left Sidebar + Center Monitor + Right Protocol Panel
        h_splitter = QSplitter(Qt.Horizontal)

        # Left Sidebar (Session Controls, Camera/File Selection, Protocols)
        self.session_panel = SessionPanel(self)
        h_splitter.addWidget(self.session_panel)

        # Central Live Monitor (Video Display & Overlay Zones)
        self.live_monitor = LiveMonitor(self)
        h_splitter.addWidget(self.live_monitor)

        # Right Procedure Panel (Checklist, Current Step, Next Step, Progress)
        self.protocol_panel = ProtocolPanel(self)
        h_splitter.addWidget(self.protocol_panel)

        # Proportions: 280px left, flex center, 340px right
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setStretchFactor(2, 0)
        h_splitter.setSizes([280, 740, 340])

        v_splitter.addWidget(h_splitter)

        # Bottom Event & Subsystem Status Panel
        self.event_panel = EventPanel(self)
        v_splitter.addWidget(self.event_panel)

        # Proportions: monitor top flex, bottom fixed/compact
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([580, 220])

        root_layout.addWidget(v_splitter, stretch=1)

    def _init_data(self) -> None:
        """Populate initial configurations, 4-tier hardware report, and protocols."""
        # 1. Multi-tier Hardware Inspection
        # Distinguishes physical GPU, inference runtime, execution provider, and active selection
        hw_report = self.config_manager.get_hardware_report()
        self.top_bar.set_compute_status(hw_report.active_device_selected, hw_report.device_selection_reason)
        self.event_panel.set_hardware_report(hw_report)

        # 2. Protocols Discovery
        protocols_dir = self.config.protocols_dir
        protocol_files = []
        if protocols_dir.exists():
            protocol_files = [p.name for p in protocols_dir.glob("*.json")]

        self.session_panel.set_available_protocols(
            protocol_files,
            default_selected=self.config.default_protocol,
        )

        # Load default protocol into right panel if present
        if self.config.default_protocol in protocol_files:
            try:
                proto = self.config_manager.load_protocol(self.config.default_protocol)
                self.protocol_panel.load_protocol(proto)
            except Exception as e:
                logger.error("Failed to load initial protocol: %s", e)

        # 3. Initial Subsystem States
        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.event_panel.set_subsystem_statuses(camera_connected=False)

    def _connect_signals(self) -> None:
        """Wire UI events across panels using Qt signals and slots."""
        self.top_bar.settings_clicked.connect(self._open_settings_dialog)

        self.session_panel.start_requested.connect(self._handle_start_session)
        self.session_panel.pause_requested.connect(self._handle_pause_session)
        self.session_panel.stop_requested.connect(self._handle_stop_session)
        self.session_panel.reset_requested.connect(self._handle_reset_session)
        self.session_panel.protocol_changed.connect(self._handle_protocol_selected)
        self.session_panel.source_changed.connect(self._handle_source_changed)

    @Slot()
    def _open_settings_dialog(self) -> None:
        """Open settings dialog modal."""
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            hw_report = self.config_manager.get_hardware_report()
            self.top_bar.set_compute_status(hw_report.active_device_selected, hw_report.device_selection_reason)
            self.event_panel.set_hardware_report(hw_report)
            self.event_panel.add_event(
                severity="INFO",
                source="CONFIG",
                description=f"Configuration updated. Active inference device: {hw_report.active_device_selected}",
            )

    @Slot()
    def _handle_start_session(self) -> None:
        """Handle operator Start Session action."""
        self._session_state = "ACTIVE"
        source_info = self._get_current_source_info()

        self.session_panel.set_session_state("ACTIVE")
        self.top_bar.set_session_status("ACTIVE")
        self.live_monitor.set_camera_status(True, source_info)
        self.event_panel.set_subsystem_statuses(camera_connected=True, camera_desc=source_info)

        self.event_panel.add_event(
            severity="INFO",
            source="SESSION",
            description=f"Session started by operator. Source: {source_info}",
            hash_digest="session:start",
        )
        logger.info("Session started with source: %s", source_info)

    @Slot()
    def _handle_pause_session(self) -> None:
        """Handle operator Pause/Resume toggle."""
        if self._session_state == "ACTIVE":
            self._session_state = "PAUSED"
            self.session_panel.set_session_state("PAUSED")
            self.top_bar.set_session_status("PAUSED")
            self.event_panel.add_event(
                severity="WARNING",
                source="SESSION",
                description="Session paused by operator. Monitoring suspended.",
            )
        else:
            self._session_state = "ACTIVE"
            self.session_panel.set_session_state("ACTIVE")
            self.top_bar.set_session_status("ACTIVE")
            self.event_panel.add_event(
                severity="INFO",
                source="SESSION",
                description="Session resumed by operator.",
            )

    @Slot()
    def _handle_stop_session(self) -> None:
        """Handle operator Stop Session action."""
        self._session_state = "IDLE"
        source_info = self._get_current_source_info()

        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.live_monitor.set_camera_status(False, source_info)
        self.event_panel.set_subsystem_statuses(camera_connected=False)

        self.event_panel.add_event(
            severity="INFO",
            source="SESSION",
            description="Session stopped by operator. Awaiting audit finalization.",
            hash_digest="session:stop",
        )
        logger.info("Session stopped")

    @Slot()
    def _handle_reset_session(self) -> None:
        """Handle operator Reset Session action."""
        self._session_state = "IDLE"
        source_info = self._get_current_source_info()

        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.protocol_panel.set_current_step(0)
        self.live_monitor.set_camera_status(False, source_info)
        self.event_panel.set_subsystem_statuses(camera_connected=False)
        self.event_panel.clear_timeline()
        self.event_panel.add_event(
            severity="INFO",
            source="SESSION",
            description="Session reset. Compliance progress returned to STEP_01.",
        )

    @Slot(str)
    def _handle_protocol_selected(self, protocol_name: str) -> None:
        """Load newly chosen protocol file."""
        if not protocol_name:
            return
        try:
            proto = self.config_manager.load_protocol(protocol_name)
            self.protocol_panel.load_protocol(proto)
            self.event_panel.add_event(
                severity="INFO",
                source="PROTOCOL",
                description=f"Loaded protocol '{proto.experiment_name}' (v{proto.protocol_version}) with {len(proto.steps)} steps.",
            )
        except Exception as e:
            self.event_panel.add_event(
                severity="CRITICAL",
                source="PROTOCOL",
                description=f"Failed to load protocol {protocol_name}: {e}",
            )

    @Slot(str, object)
    def _handle_source_changed(self, source_type: str, source_val: object) -> None:
        """Handle video source selection change."""
        self.config.camera.source_type = source_type  # type: ignore
        info = self._get_current_source_info()
        self.live_monitor.set_camera_status(self._session_state == "ACTIVE", info)
        self.event_panel.add_event(
            severity="INFO",
            source="CAPTURE",
            description=f"Video capture source updated to: {info}",
        )

    def _get_current_source_info(self) -> str:
        st = self.session_panel.source_type_combo.currentText()
        if "Webcam" in st:
            return f"Device {self.session_panel.device_spin.value()} (1280x720 @ 30 FPS)"
        if "File" in st:
            return f"File: {self.session_panel.file_path_label.text()}"
        return "RTSP Stream"

    def closeEvent(self, event: QCloseEvent) -> None:
        """Clean shutdown handler."""
        logger.info("Application window closing")
        event.accept()
