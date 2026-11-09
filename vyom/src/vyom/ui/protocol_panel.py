"""Right-side procedure compliance and step monitoring panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from vyom.core.schemas import ProtocolSchema
from vyom.ui.styles import (
    BG_CARD,
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    StatusBadge,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class ProtocolPanel(QFrame):
    """Right panel presenting procedural instructions, checklist, and progress tracking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        self.setMinimumWidth(310)
        self.setMaximumWidth(390)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Header: Experiment Name & Version
        self.header_title = QLabel("PROTOCOL NOT SELECTED")
        self.header_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #e6edf3;")
        self.header_desc = QLabel("Load an experiment protocol from config to monitor compliance.")
        self.header_desc.setWordWrap(True)
        self.header_desc.setStyleSheet("font-size: 11px; color: #8fa1b6;")

        layout.addWidget(self.header_title)
        layout.addWidget(self.header_desc)

        # 2. Step Progress & Metrics Strip
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)

        metrics_row = QHBoxLayout()
        self.progress_pct_label = QLabel("Progress: 0%")
        self.progress_pct_label.setStyleSheet("font-size: 11px; font-weight: 700; color: #e6edf3;")

        self.step_counts_label = QLabel("Completed: 0 | Remaining: 0")
        self.step_counts_label.setStyleSheet("font-size: 11px; color: #8fa1b6;")

        metrics_row.addWidget(self.progress_pct_label)
        metrics_row.addStretch()
        metrics_row.addWidget(self.step_counts_label)
        metrics_layout.addLayout(metrics_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        metrics_layout.addWidget(self.progress_bar)
        layout.addLayout(metrics_layout)

        # 3. Current Active Step Detailed Card
        active_box = QGroupBox("Current Expected Step")
        active_layout = QVBoxLayout(active_box)
        active_layout.setSpacing(6)

        step_header = QHBoxLayout()
        self.step_id_label = QLabel("STEP —")
        self.step_id_label.setStyleSheet("font-size: 13px; font-weight: 800; color: #388bfd;")
        self.step_event_badge = StatusBadge("AWAITING START", color=COLOR_WARNING)
        step_header.addWidget(self.step_id_label)
        step_header.addStretch()
        step_header.addWidget(self.step_event_badge)
        active_layout.addLayout(step_header)

        self.step_title_label = QLabel("No active procedure step")
        self.step_title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #e6edf3;")
        self.step_instruction = QLabel("Start a session to initiate procedure compliance monitoring.")
        self.step_instruction.setWordWrap(True)
        self.step_instruction.setStyleSheet("font-size: 11px; color: #9fb0c3; line-height: 1.4;")
        active_layout.addWidget(self.step_title_label)
        active_layout.addWidget(self.step_instruction)

        self.objects_label = QLabel("Required Objects: None")
        self.objects_label.setStyleSheet("font-size: 11px; color: #65768d;")
        self.timeout_label = QLabel("Timeout: --s | Min Confidence: --%")
        self.timeout_label.setStyleSheet("font-size: 11px; color: #65768d;")
        active_layout.addWidget(self.objects_label)
        active_layout.addWidget(self.timeout_label)

        layout.addWidget(active_box)

        # 4. Procedure Checklist (Completed, Current, Remaining)
        steps_group = QGroupBox("Procedure Steps")
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setSpacing(6)

        self.step_list = QListWidget()
        self.step_list.setSelectionMode(QListWidget.NoSelection)
        steps_layout.addWidget(self.step_list)

        layout.addWidget(steps_group, stretch=1)

        # 5. Next Expected Step Guidance Box
        self.next_step_frame = QFrame()
        self.next_step_frame.setObjectName("cardFrame")
        next_layout = QVBoxLayout(self.next_step_frame)
        next_layout.setContentsMargins(8, 8, 8, 8)
        next_layout.setSpacing(2)

        next_title = QLabel("NEXT EXPECTED STEP")
        next_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #65768d; letter-spacing: 0.5px;")
        self.next_step_text = QLabel("Awaiting initialization...")
        self.next_step_text.setWordWrap(True)
        self.next_step_text.setStyleSheet("font-size: 11px; color: #9fb0c3;")
        next_layout.addWidget(next_title)
        next_layout.addWidget(self.next_step_text)

        layout.addWidget(self.next_step_frame)

        self._protocol: ProtocolSchema | None = None
        self._current_step_idx: int = 0

    def load_protocol(self, protocol: ProtocolSchema) -> None:
        """Load and render an experiment procedure protocol."""
        self._protocol = protocol
        self.header_title.setText(f"{protocol.experiment_name} (v{protocol.protocol_version})")
        self.header_desc.setText(protocol.description or "Experiment procedure protocol loaded.")

        self.step_list.clear()
        for step in protocol.steps:
            item_text = f"○ [{step.step_id}] {step.title}"
            item = QListWidgetItem(item_text)
            item.setToolTip(step.instruction)
            self.step_list.addItem(item)

        self.set_current_step(0)

    def set_current_step(self, step_idx: int) -> None:
        """Set active procedure step and update completed, remaining, and next steps."""
        if not self._protocol or not self._protocol.steps:
            return

        total = len(self._protocol.steps)
        self._current_step_idx = min(max(0, step_idx), total - 1)
        step = self._protocol.steps[self._current_step_idx]

        completed_count = self._current_step_idx
        remaining_count = total - completed_count - 1
        pct = int((completed_count / total) * 100)

        # Metrics
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"Progress: {pct}%")
        self.step_counts_label.setText(
            f"Completed: {completed_count} | Active: 1 | Remaining: {remaining_count}"
        )

        # Active Card
        self.step_id_label.setText(f"{step.step_id} (Step {step.order}/{total})")
        self.step_title_label.setText(step.title)
        self.step_instruction.setText(step.instruction)
        self.step_event_badge.set_badge_color(COLOR_ACCENT, step.expected_event)
        objs = ", ".join(step.required_objects) if step.required_objects else "None"
        self.objects_label.setText(f"Required Objects: {objs}")
        self.timeout_label.setText(
            f"Timeout: {step.timeout_seconds}s | Stable Duration: {step.stable_duration_ms}ms | Min Conf: {int(step.min_confidence * 100)}%"
        )

        # Checklist Items
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            st = self._protocol.steps[i]
            if i < self._current_step_idx:
                item.setText(f"✓ [{st.step_id}] {st.title}")
                item.setForeground(Qt.GlobalColor.darkGreen)
            elif i == self._current_step_idx:
                item.setText(f"▶ [{st.step_id}] {st.title} (Active)")
                item.setForeground(Qt.GlobalColor.cyan)
            else:
                item.setText(f"○ [{st.step_id}] {st.title} (Pending)")
                item.setForeground(Qt.GlobalColor.gray)

        # Next Step
        if self._current_step_idx + 1 < total:
            next_st = self._protocol.steps[self._current_step_idx + 1]
            self.next_step_text.setText(f"Next: [{next_st.step_id}] {next_st.title}")
        else:
            self.next_step_text.setText("Final procedure step in progress. Next: Protocol completion.")
