"""Live monitor component for camera feed and perception overlays."""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vyom.ui.styles import (
    BORDER_STRONG,
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    StatusBadge,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


class VideoViewport(QWidget):
    """Viewport rendering active video frames while maintaining aspect ratio, or a mission reticle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(540, 320)
        self._pixmap: QPixmap | None = None
        self._title: str = "CAMERA NOT CONNECTED"
        self._subtitle: str = "Select a video source and click 'Start Session' to begin frame acquisition."

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        """Update the displayed video frame."""
        self._pixmap = pixmap
        self.update()

    def set_status_message(self, title: str, subtitle: str) -> None:
        """Update placeholder text."""
        self._title = title
        self._subtitle = subtitle
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint video frame or mission placeholder grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        if self._pixmap and not self._pixmap.isNull():
            # Maintain aspect ratio cleanly
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            return

        # 1. Background Fill
        painter.fillRect(0, 0, w, h, QColor("#06090e"))

        # 2. Subtle Engineering Grid Lines
        grid_pen = QPen(QColor("#131b26"), 1, Qt.DotLine)
        painter.setPen(grid_pen)
        grid_size = 36
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)

        # 3. Center Crosshairs / Target Reticle
        cx, cy = w / 2.0, h / 2.0
        reticle_pen = QPen(QColor("#253449"), 1.5, Qt.SolidLine)
        painter.setPen(reticle_pen)
        painter.drawEllipse(QRectF(cx - 40, cy - 40, 80, 80))
        painter.drawLine(int(cx - 55), int(cy), int(cx - 15), int(cy))
        painter.drawLine(int(cx + 15), int(cy), int(cx + 55), int(cy))
        painter.drawLine(int(cx), int(cy - 55), int(cx), int(cy - 15))
        painter.drawLine(int(cx), int(cy + 15), int(cx), int(cy + 55))

        # 4. Status Title
        painter.setPen(QColor(TEXT_PRIMARY))
        title_font = QFont("Segoe UI", 14, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(
            QRectF(20, cy + 55, w - 40, 30),
            Qt.AlignHCenter | Qt.AlignTop,
            self._title,
        )

        # 5. Subtitle Instruction
        painter.setPen(QColor(TEXT_MUTED))
        sub_font = QFont("Segoe UI", 10, QFont.Normal)
        painter.setFont(sub_font)
        painter.drawText(
            QRectF(20, cy + 90, w - 40, 40),
            Qt.AlignHCenter | Qt.AlignTop,
            self._subtitle,
        )


class LiveMonitor(QFrame):
    """Main video monitoring container with status header and reserved overlay zones."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("monitorFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top Header Bar inside monitor
        top_bar = QHBoxLayout()
        self.source_label = QLabel("Source: [Device 0] USB Camera (1280x720)")
        self.source_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #8fa1b6;")

        self.fps_badge = StatusBadge("-- FPS", color=COLOR_ACCENT)
        self.latency_label = QLabel("Latency: -- ms")
        self.latency_label.setStyleSheet("font-size: 11px; color: #65768d;")

        top_bar.addWidget(self.source_label)
        top_bar.addStretch()
        top_bar.addWidget(self.latency_label)
        top_bar.addWidget(self.fps_badge)
        layout.addLayout(top_bar)

        # Main Video Viewport
        self.viewport = VideoViewport(self)
        layout.addWidget(self.viewport, stretch=1)

        # Bottom Reserved Perception Overlay Strip
        # Explicitly reserves visual space for YOLO, MediaPipe, and ArUco overlays
        overlay_strip = QHBoxLayout()
        overlay_strip.setSpacing(8)

        self.yolo_overlay_badge = StatusBadge("YOLO-OBB: NOT CONFIGURED", color=COLOR_WARNING)
        self.mediapipe_overlay_badge = StatusBadge("HANDS: NOT LOADED", color=COLOR_WARNING)
        self.aruco_overlay_badge = StatusBadge("ARUCO: UNCALIBRATED", color=COLOR_ACCENT)

        self.frame_counter_label = QLabel("Frame: 0 | Monotonic: 0.000s")
        self.frame_counter_label.setStyleSheet("font-size: 11px; color: #65768d; font-family: monospace;")

        overlay_strip.addWidget(self.yolo_overlay_badge)
        overlay_strip.addWidget(self.mediapipe_overlay_badge)
        overlay_strip.addWidget(self.aruco_overlay_badge)
        overlay_strip.addStretch()
        overlay_strip.addWidget(self.frame_counter_label)

        layout.addLayout(overlay_strip)

    def set_camera_status(self, connected: bool, source_info: str) -> None:
        """Update camera connection state."""
        self.source_label.setText(f"Source: {source_info}")
        if connected:
            self.viewport.set_status_message(
                "CAMERA CONNECTED",
                "Acquiring video stream...",
            )
        else:
            self.viewport.set_status_message(
                "CAMERA NOT CONNECTED",
                "Select a video source and click 'Start Session' to begin frame acquisition.",
            )

    def update_frame(self, pixmap: QPixmap, frame_idx: int = 0, timestamp_s: float = 0.0) -> None:
        """Display newly acquired frame."""
        self.viewport.set_pixmap(pixmap)
        self.frame_counter_label.setText(f"Frame: {frame_idx} | Monotonic: {timestamp_s:.3f}s")
