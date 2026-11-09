"""Styling, theme constants, and reusable visual components for VYOM.

Follows a high-contrast engineering/mission-monitoring aesthetic
suited for microgravity payload monitoring and laboratory operations.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QFrame


# ==============================================================================
# Theme Color Palette (Mission Control Dark)
# ==============================================================================
BG_ROOT = "#0a0d12"
BG_PRIMARY = "#0f141c"
BG_SECONDARY = "#161d27"
BG_CARD = "#1a222e"
BG_CARD_HOVER = "#202a38"
BORDER_SUBTLE = "#283344"
BORDER_STRONG = "#3a4960"

TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#9fb0c3"
TEXT_MUTED = "#65768d"

COLOR_ACCENT = "#2f81f7"       # Primary blue
COLOR_SUCCESS = "#2ea043"      # Operation green
COLOR_WARNING = "#d29922"      # Alert amber
COLOR_DANGER = "#f85149"       # Violation red
COLOR_OFFLINE = "#8957e5"      # Offline purple / secure badge


# ==============================================================================
# Global Qt Application Stylesheet
# ==============================================================================
APPLICATION_STYLESHEET = f"""
QMainWindow, QDialog {{
    background-color: {BG_ROOT};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}}

QWidget {{
    color: {TEXT_PRIMARY};
}}

QFrame#cardFrame {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}

QFrame#panelFrame {{
    background-color: {BG_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
}}

QFrame#monitorFrame {{
    background-color: #05070a;
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    border: none;
    background: {BG_PRIMARY};
    width: 8px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    min-height: 20px;
    border-radius: 4px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Push Buttons */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {BG_CARD_HOVER};
    border-color: {COLOR_ACCENT};
}}
QPushButton:pressed {{
    background-color: {BORDER_SUBTLE};
}}
QPushButton:disabled {{
    background-color: {BG_SECONDARY};
    color: {TEXT_MUTED};
    border-color: {BORDER_SUBTLE};
}}

QPushButton#btnPrimary {{
    background-color: #1f6feb;
    color: #ffffff;
    border: 1px solid #388bfd;
    font-weight: 600;
}}
QPushButton#btnPrimary:hover {{
    background-color: #388bfd;
}}
QPushButton#btnPrimary:disabled {{
    background-color: {BG_CARD};
    color: {TEXT_MUTED};
    border-color: {BORDER_SUBTLE};
}}

QPushButton#btnSuccess {{
    background-color: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    font-weight: 600;
}}
QPushButton#btnSuccess:hover {{
    background-color: #2ea043;
}}
QPushButton#btnSuccess:disabled {{
    background-color: {BG_CARD};
    color: {TEXT_MUTED};
    border-color: {BORDER_SUBTLE};
}}

QPushButton#btnWarning {{
    background-color: #9e6a03;
    color: #ffffff;
    border: 1px solid #bb8009;
    font-weight: 600;
}}
QPushButton#btnWarning:hover {{
    background-color: #bb8009;
}}

QPushButton#btnDanger {{
    background-color: #da3633;
    color: #ffffff;
    border: 1px solid #f85149;
    font-weight: 600;
}}
QPushButton#btnDanger:hover {{
    background-color: #f85149;
}}

/* Inputs & Combos */
QComboBox, QLineEdit, QSpinBox {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 5px 10px;
    selection-background-color: {COLOR_ACCENT};
}}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus {{
    border-color: {COLOR_ACCENT};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {BORDER_SUBTLE};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_STRONG};
    color: {TEXT_PRIMARY};
    selection-background-color: {COLOR_ACCENT};
    outline: none;
}}

/* Group Boxes */
QGroupBox {{
    font-weight: 600;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    background-color: {BG_PRIMARY};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {COLOR_SUCCESS};
    border-radius: 3px;
}}

/* Splitters */
QSplitter::handle {{
    background-color: {BORDER_SUBTLE};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* Tables / Tree Views */
QTableWidget, QTreeWidget, QListWidget {{
    background-color: {BG_PRIMARY};
    alternate-background-color: {BG_SECONDARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    gridline-color: {BORDER_SUBTLE};
    selection-background-color: {BG_CARD_HOVER};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    font-weight: 600;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {BORDER_STRONG};
    border-right: 1px solid {BORDER_SUBTLE};
}}
"""


class StatusBadge(QLabel):
    """Configurable pill/badge widget displaying component state."""

    def __init__(
        self,
        text: str,
        color: str = COLOR_ACCENT,
        parent=None,
    ) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_badge_color(color)

    def set_badge_color(self, color: str, text: str | None = None) -> None:
        """Update badge background and border color."""
        if text is not None:
            self.setText(text)
        self.setStyleSheet(
            f"""
            QLabel {{
                background-color: {color}22;
                color: {color};
                border: 1px solid {color}66;
                border-radius: 10px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            """
        )
