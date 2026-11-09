"""Runtime diagnostics, latency profiling, and hardware telemetry dialog.

Part of VYOM Prompt 12: Performance Profiling and System Diagnostics.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vyom.core.profiler import SystemPerformanceProfiler, get_performance_profiler
from vyom.ui.styles import APPLICATION_STYLESHEET, BG_CARD, BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY


class DiagnosticsDialog(QDialog):
    """Real-time performance inspector displaying latencies, frame rates, CPU, RAM, and GPU telemetry."""

    def __init__(
        self,
        profiler: Optional[SystemPerformanceProfiler] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.profiler = profiler or get_performance_profiler()

        self.setWindowTitle("VYOM — Real-Time Performance & Hardware Telemetry")
        self.setMinimumSize(680, 520)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("SUBSYSTEM LATENCY & HARDWARE PROFILER")
        header.setStyleSheet("font-size: 15px; font-weight: bold; letter-spacing: 0.5px;")
        layout.addWidget(header)

        sub = QLabel(
            "Live moving averages across capture, inference, tracking, interaction, and rule evaluation stages."
        )
        sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(sub)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            f"background-color: {BG_CARD}; border: 1px solid {BORDER_SUBTLE}; "
            f"font-family: 'Consolas', monospace; font-size: 12px; padding: 10px; border-radius: 6px;"
        )
        layout.addWidget(self.text_area, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_copy = QPushButton("📋 Copy Telemetry")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_row.addWidget(self.btn_copy)

        btn_row.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)

        layout.addLayout(btn_row)

        # Refresh immediately
        self._refresh()

        # Timer to auto-update every 1.5 seconds
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        report = self.profiler.format_diagnostics_report()
        self.text_area.setPlainText(report)

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_area.toPlainText())
        self.btn_copy.setText("✓ Copied!")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("📋 Copy Telemetry"))

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)

