"""Pre-session readiness inspection, subsystem validation, and safety interlocks.

Part of VYOM Prompt 12: Pre-Session Validation and Calibration Interlocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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

from vyom.alerts.alert_manager import AlertManager
from vyom.core.config import ConfigManager
from vyom.protocol.protocol_schema import ProtocolDefinition
from vyom.ui.styles import (
    APPLICATION_STYLESHEET,
    BG_CARD,
    BG_PRIMARY,
    BG_ROOT,
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from vyom.vision.aruco_tracker import ArucoTracker


@dataclass
class ReadinessItem:
    """Status report for a single pre-session requirement."""
    category: str
    name: str
    status: str       # "PASS", "FAIL", "WARNING", "INFO"
    is_mandatory: bool
    details: str
    remediation: Optional[str] = None


class SessionReadinessChecker:
    """Evaluates whether all perception, calibration, model, and storage prerequisites are met."""

    def __init__(
        self,
        config_manager: ConfigManager,
        aruco_tracker: ArucoTracker,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.config_manager = config_manager
        self.config = config_manager.config
        self.aruco_tracker = aruco_tracker
        self.alert_manager = alert_manager

    def perform_check(
        self,
        protocol: Optional[ProtocolDefinition],
        source_type: str,
        source_param: Any,
        selected_video_file: Optional[Path] = None,
        yolo_ready: bool = True,
        yolo_status_msg: str = "",
        hand_ready: bool = True,
        hand_status_msg: str = "",
    ) -> Tuple[bool, List[ReadinessItem]]:
        """Run complete pre-session inspection and return (can_start, items)."""
        items: List[ReadinessItem] = []
        mandatory_passed = True

        # 1. Protocol Validation
        if protocol is None:
            items.append(
                ReadinessItem(
                    category="Protocol",
                    name="Protocol Definition",
                    status="FAIL",
                    is_mandatory=True,
                    details="No experiment protocol loaded.",
                    remediation="Select a valid JSON protocol in the left sidebar dropdown.",
                )
            )
            mandatory_passed = False
        else:
            items.append(
                ReadinessItem(
                    category="Protocol",
                    name=f"Protocol: {protocol.experiment_name}",
                    status="PASS",
                    is_mandatory=True,
                    details=f"Loaded ID: {protocol.protocol_id} v{protocol.protocol_version} ({len(protocol.steps)} steps).",
                )
            )

        # 2. Camera / Video Source
        if source_type == "video_file":
            if selected_video_file and selected_video_file.is_file() and selected_video_file.stat().st_size > 1000:
                items.append(
                    ReadinessItem(
                        category="Capture",
                        name="Video File Source",
                        status="PASS",
                        is_mandatory=True,
                        details=f"File verified: {selected_video_file.name} ({selected_video_file.stat().st_size / (1024*1024):.1f} MB).",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        category="Capture",
                        name="Video File Source",
                        status="FAIL",
                        is_mandatory=True,
                        details="No valid local experiment video selected or file is empty.",
                        remediation="Click 'Browse...' in Camera Source panel to select an MP4/AVI experiment recording.",
                    )
                )
                mandatory_passed = False
        elif source_type == "device_index":
            items.append(
                ReadinessItem(
                    category="Capture",
                    name=f"Live Camera (Index {source_param})",
                    status="PASS",
                    is_mandatory=True,
                    details=f"Configured device index {source_param} (Target: {self.config.camera.width}x{self.config.camera.height} @ {self.config.camera.target_fps} FPS).",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    category="Capture",
                    name="Network Stream",
                    status="WARNING",
                    is_mandatory=False,
                    details=f"Configured RTSP stream endpoint: {source_param}.",
                    remediation="Ensure RTSP server is active on designated IP/port.",
                )
            )

        # 3. YOLOv8-OBB Model Checkpoint
        model_path = Path("models/vyom_yolov8n_obb_smoke.pt")
        onnx_path = Path("models/vyom_yolov8n_obb_smoke.onnx")
        if model_path.is_file() or onnx_path.is_file():
            active_name = model_path.name if model_path.is_file() else onnx_path.name
            items.append(
                ReadinessItem(
                    category="Perception",
                    name="YOLOv8-OBB Checkpoint",
                    status="PASS" if yolo_ready else "WARNING",
                    is_mandatory=True,
                    details=f"Verified model asset: {active_name} ({yolo_status_msg or 'Operational'}).",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    category="Perception",
                    name="YOLOv8-OBB Checkpoint",
                    status="FAIL",
                    is_mandatory=True,
                    details="Neither PyTorch (.pt) nor ONNX (.onnx) model checkpoint found in models/.",
                    remediation="Train or place vyom_yolov8n_obb_smoke.pt in models/ directory.",
                )
            )
            mandatory_passed = False

        # 4. MediaPipe Hand Landmarker Asset
        mp_task_path = Path("models/hand_landmarker.task")
        if mp_task_path.is_file() and mp_task_path.stat().st_size > 100000:
            items.append(
                ReadinessItem(
                    category="Perception",
                    name="MediaPipe Hand Landmarker",
                    status="PASS" if hand_ready else "WARNING",
                    is_mandatory=True,
                    details=f"Asset verified: hand_landmarker.task ({mp_task_path.stat().st_size / (1024*1024):.1f} MB, CPU-only on Windows).",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    category="Perception",
                    name="MediaPipe Hand Landmarker",
                    status="FAIL",
                    is_mandatory=True,
                    details="Local task asset models/hand_landmarker.task is missing or corrupted.",
                    remediation="Ensure hand_landmarker.task is downloaded to models/ folder.",
                )
            )
            mandatory_passed = False

        # 5. ArUco Rig & Metric Target Zone Spatial Requirements
        protocol_requires_metric_zones = False
        if protocol and (protocol.target_zones or any(getattr(s, "target_zone_name", None) for s in protocol.steps)):
            protocol_requires_metric_zones = True

        is_calibrated = self.aruco_tracker.rig_coords.is_calibrated
        if protocol_requires_metric_zones:
            if is_calibrated:
                items.append(
                    ReadinessItem(
                        category="Spatial Rig",
                        name="Rig Metric Calibration",
                        status="PASS",
                        is_mandatory=True,
                        details="Camera matrix & distortion coefficients established. 3D metric coordinate transformations active.",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        category="Spatial Rig",
                        name="Rig Metric Calibration",
                        status="FAIL",
                        is_mandatory=True,
                        details="Protocol requires metric target zones, but camera intrinsics/rig coordinates are uncalibrated.",
                        remediation="Calibrate camera intrinsics or use image-space bounding zones.",
                    )
                )
                mandatory_passed = False
        else:
            items.append(
                ReadinessItem(
                    category="Spatial Rig",
                    name="Rig Metric Calibration",
                    status="PASS" if is_calibrated else "INFO",
                    is_mandatory=False,
                    details=(
                        "Calibrated 3D metric coordinates active."
                        if is_calibrated
                        else "Uncalibrated (Protocol uses pixel-space bounding zones; metric calibration optional)."
                    ),
                )
            )

        # 6. Compute & Hardware Provider
        hw_report = self.config_manager.get_hardware_report()
        items.append(
            ReadinessItem(
                category="Hardware",
                name=f"Inference Device ({hw_report.active_device_selected})",
                status="PASS",
                is_mandatory=True,
                details=f"{hw_report.device_selection_reason} Runtime: {hw_report.inference_runtime_installed}.",
            )
        )

        # 7. Available Storage Space
        storage_dir = Path("sessions")
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            total, used, free = shutil.disk_usage(storage_dir)
            free_mb = free / (1024 * 1024)
            if free_mb >= 500:
                items.append(
                    ReadinessItem(
                        category="Storage",
                        name="Disk Storage Capacity",
                        status="PASS",
                        is_mandatory=True,
                        details=f"Available storage: {free_mb / 1024:.1f} GB free for session recordings and SQLite audit database.",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        category="Storage",
                        name="Disk Storage Capacity",
                        status="FAIL",
                        is_mandatory=True,
                        details=f"Low disk space: {free_mb:.0f} MB free. At least 500 MB required for session recording and database.",
                        remediation="Free disk space on volume before starting experiment.",
                    )
                )
                mandatory_passed = False
        except Exception as e:
            items.append(
                ReadinessItem(
                    category="Storage",
                    name="Disk Storage Capacity",
                    status="WARNING",
                    is_mandatory=False,
                    details=f"Could not inspect disk free space: {e}",
                )
            )

        # 8. Offline Voice Alert Engine
        if self.alert_manager and self.alert_manager.is_engine_available():
            items.append(
                ReadinessItem(
                    category="Audio",
                    name="Offline Voice Alerts",
                    status="PASS",
                    is_mandatory=False,
                    details=f"Windows SAPI5 offline speech synthesizer operational (Active voice: {self.alert_manager.get_active_voice_name()}).",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    category="Audio",
                    name="Offline Voice Alerts",
                    status="WARNING",
                    is_mandatory=False,
                    details="Offline voice alerts unavailable or muted. Visual cues and event logging will continue normally.",
                    remediation="Check Windows SAPI5 voice settings or unmute voice in settings.",
                )
            )

        return mandatory_passed, items


class SessionReadinessDialog(QDialog):
    """UI Dialog displaying the pre-flight readiness checklist and blocking invalid sessions."""

    def __init__(
        self,
        can_start: bool,
        items: List[ReadinessItem],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.can_start = can_start
        self.items = items

        self.setWindowTitle("VYOM — Pre-Flight Experiment Readiness Checklist")
        self.setMinimumSize(850, 520)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header Banner
        header = QFrame()
        header.setObjectName("cardFrame")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)

        icon_label = QLabel("✓" if can_start else "✕")
        icon_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {COLOR_SUCCESS if can_start else COLOR_DANGER};"
        )
        h_layout.addWidget(icon_label)

        title_layout = QVBoxLayout()
        title = QLabel(
            "EXPERIMENT READINESS: READY TO PROCEED" if can_start else "EXPERIMENT READINESS: CHECKS FAILED"
        )
        title.setStyleSheet("font-size: 15px; font-weight: bold; letter-spacing: 0.5px;")
        subtitle = QLabel(
            "All mandatory hardware, perception, calibration, and storage prerequisites satisfied."
            if can_start
            else "Mandatory prerequisites are missing or uncalibrated. Session start is blocked."
        )
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        h_layout.addLayout(title_layout, stretch=1)
        layout.addWidget(header)

        # Table of Checklist Items
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Subsystem", "Item", "Status", "Requirement", "Details & Remediation"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setRowCount(len(items))

        for row, it in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(it.category))
            self.table.setItem(row, 1, QTableWidgetItem(it.name))

            status_item = QTableWidgetItem(it.status)
            if it.status == "PASS":
                status_item.setForeground(QColor(COLOR_SUCCESS))
            elif it.status == "FAIL":
                status_item.setForeground(QColor(COLOR_DANGER))
            elif it.status == "WARNING":
                status_item.setForeground(QColor(COLOR_WARNING))
            else:
                status_item.setForeground(QColor(TEXT_MUTED))
            self.table.setItem(row, 2, status_item)

            req_item = QTableWidgetItem("Mandatory" if it.is_mandatory else "Optional")
            if it.is_mandatory:
                req_item.setForeground(QColor("#f87171"))
            else:
                req_item.setForeground(QColor(TEXT_MUTED))
            self.table.setItem(row, 3, req_item)

            detail_msg = it.details
            if it.remediation:
                detail_msg += f" &bull; Remediation: {it.remediation}"
            self.table.setItem(row, 4, QTableWidgetItem(detail_msg))

        layout.addWidget(self.table)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if can_start:
            self.btn_proceed = QPushButton("▶  Proceed & Launch Session")
            self.btn_proceed.setStyleSheet(
                f"background-color: {COLOR_SUCCESS}; color: white; font-weight: bold; padding: 8px 18px; border-radius: 4px;"
            )
            self.btn_proceed.clicked.connect(self.accept)
            btn_layout.addWidget(self.btn_proceed)

        self.btn_cancel = QPushButton("Close" if not can_start else "Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

