"""Top bar component for the VYOM desktop monitoring shell."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vyom import __version__
from vyom.ui.styles import (
    BG_PRIMARY,
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_OFFLINE,
    COLOR_SUCCESS,
    COLOR_WARNING,
    StatusBadge,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


class TopBar(QFrame):
    """Header bar displaying system branding, security badges, hardware compute mode, and settings."""

    settings_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setStyleSheet(
            f"""
            QFrame#topBar {{
                background-color: {BG_PRIMARY};
                border-bottom: 1px solid {BORDER_SUBTLE};
                padding: 4px 12px;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(16)

        # Title / Branding
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(1)

        title_label = QLabel("VYOM")
        title_label.setStyleSheet(
            f"font-size: 16px; font-weight: 800; color: {TEXT_PRIMARY}; letter-spacing: 1.5px;"
        )

        subtitle_label = QLabel(f"Procedure Compliance & Activity Monitor  v{__version__}")
        subtitle_label.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED}; font-weight: 500;")

        brand_layout.addWidget(title_label)
        brand_layout.addWidget(subtitle_label)
        layout.addLayout(brand_layout)

        layout.addStretch()

        # Status Badges
        # 1. Offline Mode Indicator
        self.offline_badge = StatusBadge("OFFLINE — LOCAL BY DESIGN", color=COLOR_OFFLINE)
        layout.addWidget(self.offline_badge)

        # 2. Session Status Badge
        self.session_badge = StatusBadge("SESSION: IDLE", color=COLOR_WARNING)
        layout.addWidget(self.session_badge)

        # 3. Compute Hardware Badge
        self.compute_badge = StatusBadge("COMPUTE: DETECTING...", color=COLOR_ACCENT)
        layout.addWidget(self.compute_badge)

        # 4. Voice Alert Status Badge (Prompt 10)
        self.voice_badge = StatusBadge("VOICE: READY", color=COLOR_SUCCESS)
        layout.addWidget(self.voice_badge)

        # Settings Button
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)

    def set_session_status(self, status: str) -> None:
        """Update session state display."""
        status_upper = status.upper()
        if status_upper in ("ACTIVE", "RUNNING", "RECORDING"):
            self.session_badge.set_badge_color(COLOR_SUCCESS, f"SESSION: {status_upper}")
        elif status_upper in ("PAUSED", "IDLE"):
            self.session_badge.set_badge_color(COLOR_WARNING, f"SESSION: {status_upper}")
        else:
            self.session_badge.set_badge_color(COLOR_DANGER, f"SESSION: {status_upper}")

    def set_compute_status(self, device: str, detail: str) -> None:
        """Update active compute hardware provider display."""
        color = COLOR_SUCCESS if device == "GPU" else COLOR_ACCENT
        self.compute_badge.set_badge_color(color, f"COMPUTE: {device}")
        self.compute_badge.setToolTip(detail)

    def set_voice_status(self, status: str, color: str | None = None) -> None:
        """Update offline speech synthesis badge status."""
        status_upper = status.upper()
        if color:
            badge_color = color
        elif "SPEAKING" in status_upper:
            badge_color = COLOR_ACCENT
        elif "MUTED" in status_upper:
            badge_color = COLOR_WARNING
        elif "UNAVAILABLE" in status_upper or "ERROR" in status_upper:
            badge_color = COLOR_DANGER
        else:
            badge_color = COLOR_SUCCESS
        self.voice_badge.set_badge_color(badge_color, status_upper)

