"""Bottom panel rendering timestamped event timelines and subsystem status."""

from __future__ import annotations

from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vyom.core.config import HardwareReport
from vyom.ui.styles import (
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    StatusBadge,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from vyom.vision.aruco_tracker import ArucoDetectionResult, RigTrackingState
from vyom.vision.hand_tracker import HandTrackingResult, HandTrackingStatus


class EventPanel(QFrame):
    """Bottom panel presenting an append-only event stream and comprehensive subsystem status indicators."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        self.setMinimumHeight(190)
        self.setMaximumHeight(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # 1. Header & Timeline Filter Controls
        header_layout = QHBoxLayout()
        title_label = QLabel("EVENT & ALERT AUDIT TIMELINE")
        title_label.setStyleSheet("font-size: 11px; font-weight: 700; color: #9fb0c3; letter-spacing: 0.5px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self.filter_all_btn = QPushButton("All")
        self.filter_events_btn = QPushButton("Events")
        self.filter_alerts_btn = QPushButton("Warnings")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_timeline)

        header_layout.addWidget(self.filter_all_btn)
        header_layout.addWidget(self.filter_events_btn)
        header_layout.addWidget(self.filter_alerts_btn)
        header_layout.addWidget(self.clear_btn)
        layout.addLayout(header_layout)

        # 2. Timeline Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Severity", "Source", "Event / Alert Description", "Audit Hash"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, stretch=1)

        # 3. Subsystem Diagnostic Status Strip
        # Distinctly reports: Camera, YOLO Detection, Hand Tracking, ArUco, FPS, and Hardware/Inference
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(0, 2, 0, 0)
        status_bar.setSpacing(10)

        self.cam_status_lbl = QLabel("Camera: DISCONNECTED")
        self.cam_status_lbl.setStyleSheet("font-size: 11px; color: #f85149; font-weight: 600;")

        self.yolo_status_lbl = QLabel("YOLO-OBB: NOT CONFIGURED")
        self.yolo_status_lbl.setStyleSheet("font-size: 11px; color: #d29922; font-weight: 600;")

        self.hands_status_lbl = QLabel("MediaPipe: NOT LOADED")
        self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #d29922; font-weight: 600;")

        self.aruco_status_lbl = QLabel("ArUco: NOT INITIALIZED")
        self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #65768d; font-weight: 600;")

        self.interaction_status_lbl = QLabel("Interactions: IDLE")
        self.interaction_status_lbl.setStyleSheet("font-size: 11px; color: #8fa1b6; font-weight: 600;")

        self.fps_status_lbl = QLabel("FPS: -- (0 dropped)")
        self.fps_status_lbl.setStyleSheet("font-size: 11px; color: #65768d;")

        status_bar.addWidget(self.cam_status_lbl)
        status_bar.addWidget(self.yolo_status_lbl)
        status_bar.addWidget(self.hands_status_lbl)
        status_bar.addWidget(self.aruco_status_lbl)
        status_bar.addWidget(self.interaction_status_lbl)
        status_bar.addStretch()
        status_bar.addWidget(self.fps_status_lbl)

        layout.addLayout(status_bar)

        # 4. Multi-tier Hardware and Inference Status Line
        self.hw_status_label = QLabel("Hardware: Detecting...")
        self.hw_status_label.setStyleSheet("font-size: 10px; color: #8fa1b6; font-family: monospace;")
        layout.addWidget(self.hw_status_label)

        # Add initial system event
        self.add_event(
            severity="INFO",
            source="SYSTEM",
            description="VYOM desktop monitoring shell initialized in OFFLINE mode.",
            hash_digest="genesis:00000000",
        )

    def add_event(
        self,
        severity: str,
        source: str,
        description: str,
        timestamp_str: str | None = None,
        hash_digest: str = "—",
    ) -> None:
        """Append an event record to the timeline table."""
        ts = timestamp_str or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        row = self.table.rowCount()
        self.table.insertRow(row)

        ts_item = QTableWidgetItem(ts)
        ts_item.setTextAlignment(Qt.AlignCenter)

        sev_item = QTableWidgetItem(severity.upper())
        sev_item.setTextAlignment(Qt.AlignCenter)
        if severity.upper() == "CRITICAL":
            sev_item.setForeground(Qt.GlobalColor.red)
        elif severity.upper() == "WARNING":
            sev_item.setForeground(Qt.GlobalColor.yellow)
        else:
            sev_item.setForeground(Qt.GlobalColor.cyan)

        src_item = QTableWidgetItem(source)
        desc_item = QTableWidgetItem(description)
        hash_item = QTableWidgetItem(hash_digest[:16] if hash_digest != "—" else "—")
        hash_item.setTextAlignment(Qt.AlignCenter)
        hash_item.setToolTip(hash_digest)

        for col, item in enumerate([ts_item, sev_item, src_item, desc_item, hash_item]):
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()

    def clear_timeline(self) -> None:
        """Clear the timeline view."""
        self.table.setRowCount(0)

    def set_subsystem_statuses(
        self,
        camera_connected: bool,
        camera_desc: str = "DISCONNECTED",
        fps: float = 0.0,
        dropped: int = 0,
    ) -> None:
        """Update camera and FPS diagnostics."""
        if camera_connected:
            self.cam_status_lbl.setText(f"Camera: CONNECTED ({camera_desc})")
            self.cam_status_lbl.setStyleSheet("font-size: 11px; color: #2ea043; font-weight: 600;")
        else:
            self.cam_status_lbl.setText("Camera: DISCONNECTED")
            self.cam_status_lbl.setStyleSheet("font-size: 11px; color: #f85149; font-weight: 600;")
            self.aruco_status_lbl.setText("ArUco: NOT INITIALIZED")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #65768d; font-weight: 600;")
        self.fps_status_lbl.setText(f"FPS: {fps:.1f} ({dropped} dropped)")

    def set_aruco_status(self, result: ArucoDetectionResult) -> None:
        """Update ArUco status label in the subsystem diagnostics strip."""
        if result.state == RigTrackingState.RIG_TRACKING:
            self.aruco_status_lbl.setText(f"ArUco: RIG TRACKING (IDs: {result.configured_ids_visible})")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #2ea043; font-weight: 600;")
        elif result.state == RigTrackingState.CALIBRATION_REQUIRED:
            self.aruco_status_lbl.setText(f"ArUco: {len(result.configured_ids_visible)} DETECTED (CALIBRATION REQUIRED)")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #d29922; font-weight: 600;")
        elif result.state == RigTrackingState.MARKERS_DETECTED:
            self.aruco_status_lbl.setText(f"ArUco: {len(result.configured_ids_visible)} DETECTED (SEARCHING ORIGIN)")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #2f81f7; font-weight: 600;")
        elif result.state == RigTrackingState.SEARCHING:
            self.aruco_status_lbl.setText("ArUco: SEARCHING (DICT_4X4_50)")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #8fa1b6; font-weight: 600;")
        else:
            self.aruco_status_lbl.setText("ArUco: NOT INITIALIZED")
            self.aruco_status_lbl.setStyleSheet("font-size: 11px; color: #65768d; font-weight: 600;")

    def set_hand_tracking_status(self, result: HandTrackingResult) -> None:
        """Update MediaPipe status label in the subsystem diagnostics strip."""
        if result.status == HandTrackingStatus.MODEL_MISSING:
            self.hands_status_lbl.setText("MediaPipe: MODEL MISSING")
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #f85149; font-weight: 600;")
        elif result.status == HandTrackingStatus.ERROR:
            self.hands_status_lbl.setText(f"MediaPipe: ERROR ({result.error_message or 'inference'})")
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #f85149; font-weight: 600;")
        elif result.has_hands:
            labels = [f"{h.handedness} ({h.confidence:.0%})" for h in result.hands]
            total_lms = sum(len(h.landmarks) for h in result.hands)
            txt = f"MediaPipe: {result.hand_count} Hand(s) [{', '.join(labels)} | {total_lms} LMs | {result.latency_ms:.1f}ms]"
            self.hands_status_lbl.setText(txt)
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #2ea043; font-weight: 600;")
        elif result.status == HandTrackingStatus.NO_HAND_DETECTED:
            self.hands_status_lbl.setText(f"MediaPipe: NO HANDS [{result.latency_ms:.1f}ms]")
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #8fa1b6; font-weight: 600;")
        elif result.status == HandTrackingStatus.SEARCHING:
            self.hands_status_lbl.setText("MediaPipe: SEARCHING")
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #8fa1b6; font-weight: 600;")
        else:
            self.hands_status_lbl.setText("MediaPipe: NOT INITIALIZED")
            self.hands_status_lbl.setStyleSheet("font-size: 11px; color: #65768d; font-weight: 600;")

    def set_interaction_status(self, text: str, active: bool = False) -> None:
        """Update Interaction status label in diagnostics strip."""
        self.interaction_status_lbl.setText(f"Interactions: {text}")
        if active:
            self.interaction_status_lbl.setStyleSheet("font-size: 11px; color: #2ea043; font-weight: 600;")
        else:
            self.interaction_status_lbl.setStyleSheet("font-size: 11px; color: #8fa1b6; font-weight: 600;")

    def set_hardware_report(self, report: HardwareReport) -> None:
        """Render the 4-tier hardware status distinguishing physical GPU from execution provider."""
        hw_desc = f"{report.gpu_name} ({report.gpu_vram_mb} MiB)" if report.gpu_hardware_detected else "None"
        prov_desc = report.gpu_execution_provider_name if report.gpu_execution_provider_available else "None"

        summary = (
            f"[1] Hardware: {hw_desc}  |  "
            f"[2] Runtime: {report.inference_runtime_installed}  |  "
            f"[3] Provider: {prov_desc}  |  "
            f"[4] Active Device: {report.active_device_selected}"
        )
        self.hw_status_label.setText(summary)
        self.hw_status_label.setToolTip(report.device_selection_reason)
