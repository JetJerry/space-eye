"""Right-side procedure compliance, step progression, and protocol monitoring panel."""

from __future__ import annotations

from typing import Optional, Union
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
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
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProtocolDefinition,
    ProtocolLifecycleState,
    StepProgress,
    StepState,
)
from vyom.ui.styles import (
    BG_CARD,
    BORDER_SUBTLE,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
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
        self.setMinimumWidth(320)
        self.setMaximumWidth(410)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Header: Experiment Name, Version & Lifecycle Badge
        header_top = QHBoxLayout()
        self.header_title = QLabel("PROTOCOL NOT SELECTED")
        self.header_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #e6edf3;")
        self.state_badge = StatusBadge("IDLE", color=COLOR_MUTED)
        header_top.addWidget(self.header_title)
        header_top.addStretch()
        header_top.addWidget(self.state_badge)
        layout.addLayout(header_top)

        self.header_desc = QLabel("Load an experiment protocol from config to monitor compliance.")
        self.header_desc.setWordWrap(True)
        self.header_desc.setStyleSheet("font-size: 11px; color: #8fa1b6;")
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
        self.timeout_label = QLabel("Timeout: --s | Stable Duration: --ms | Min Conf: --%")
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
        self.step_list.setStyleSheet(
            "QListWidget { background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 4px; }"
            "QListWidget::item { padding: 5px; margin-bottom: 2px; border-radius: 3px; font-size: 11px; }"
        )
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

        self._protocol: Optional[Union[ProtocolDefinition, ProtocolSchema]] = None
        self._current_step_idx: int = 0

    def load_protocol(self, protocol: Union[ProtocolDefinition, ProtocolSchema]) -> None:
        """Load and render an experiment procedure protocol."""
        self._protocol = protocol
        exp_name = getattr(protocol, "experiment_name", "Protocol")
        ver = getattr(protocol, "protocol_version", "1.0.0")
        desc = getattr(protocol, "description", None)

        self.header_title.setText(f"{exp_name} (v{ver})")
        self.header_desc.setText(desc or "Experiment procedure protocol loaded.")
        self.state_badge.set_badge_color(COLOR_MUTED, "READY")

        self.step_list.clear()
        for step in protocol.steps:
            title = getattr(step, "title", step.step_id)
            item_text = f"○ [{step.step_id}] {title}"
            item = QListWidgetItem(item_text)
            inst = getattr(step, "human_instruction", getattr(step, "instruction", ""))
            item.setToolTip(inst)
            item.setForeground(QColor("#8b949e"))
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
        remaining_count = max(0, total - completed_count - 1)
        pct = int((completed_count / total) * 100)

        # Metrics
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"Progress: {pct}%")
        self.step_counts_label.setText(
            f"Completed: {completed_count} | Active: 1 | Remaining: {remaining_count}"
        )

        # Active Card
        title = getattr(step, "title", step.step_id)
        inst = getattr(step, "human_instruction", getattr(step, "instruction", ""))
        self.step_id_label.setText(f"{step.step_id} (Step {step.order}/{total})")
        self.step_title_label.setText(title)
        self.step_instruction.setText(inst)
        self.step_event_badge.set_badge_color(COLOR_ACCENT, step.expected_event)

        req_objs = getattr(step, "required_objects", [])
        objs = ", ".join(req_objs) if req_objs else "None"
        self.objects_label.setText(f"Required Objects: {objs}")

        dur = getattr(step, "minimum_evidence_duration_ms", getattr(step, "stable_duration_ms", 500))
        tout = getattr(step, "timeout_seconds", 60)
        conf = getattr(step, "min_confidence", 0.70)
        self.timeout_label.setText(
            f"Timeout: {tout}s | Stable: {dur}ms | Min Conf: {int(conf * 100)}%"
        )

        # Checklist Items
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            st = self._protocol.steps[i]
            st_title = getattr(st, "title", st.step_id)
            if i < self._current_step_idx:
                item.setText(f"✓ [{st.step_id}] {st_title}")
                item.setForeground(QColor("#3fb950"))  # Green
            elif i == self._current_step_idx:
                item.setText(f"▶ [{st.step_id}] {st_title} (Active)")
                item.setForeground(QColor("#58a6ff"))  # Bright Cyan/Blue
            else:
                item.setText(f"○ [{st.step_id}] {st_title} (Pending)")
                item.setForeground(QColor("#8b949e"))  # Gray

        # Next Step
        if self._current_step_idx + 1 < total:
            next_st = self._protocol.steps[self._current_step_idx + 1]
            next_title = getattr(next_st, "title", next_st.step_id)
            self.next_step_text.setText(f"Next: [{next_st.step_id}] {next_title}")
        else:
            self.next_step_text.setText("Final procedure step in progress. Next: Protocol completion.")

    def update_evaluation_result(self, result: ProcedureEvaluationResult) -> None:
        """Update dashboard panel live from a ProcedureEvaluationResult."""
        # 1. Update State Badge
        state = result.protocol_state
        if state == ProtocolLifecycleState.ACTIVE:
            self.state_badge.set_badge_color(COLOR_SUCCESS, "ACTIVE")
        elif state == ProtocolLifecycleState.PAUSED:
            self.state_badge.set_badge_color(COLOR_WARNING, "PAUSED")
        elif state == ProtocolLifecycleState.COMPLETED:
            self.state_badge.set_badge_color(COLOR_SUCCESS, "COMPLETED")
        elif state == ProtocolLifecycleState.FAILED:
            self.state_badge.set_badge_color(COLOR_DANGER, "FAILED")
        elif state == ProtocolLifecycleState.ABORTED:
            self.state_badge.set_badge_color(COLOR_DANGER, "ABORTED")
        else:
            self.state_badge.set_badge_color(COLOR_MUTED, "READY")

        # 2. Update Progress Bar & Metrics
        self.progress_bar.setValue(result.progress_pct)
        self.progress_pct_label.setText(f"Progress: {result.progress_pct}%")
        self.step_counts_label.setText(
            f"Completed: {result.completed_steps} | Remaining: {result.remaining_steps}"
        )

        # 3. Update Active Step Card
        if not self._protocol or not self._protocol.steps:
            return

        total = len(self._protocol.steps)
        curr_idx = result.current_step_index

        if curr_idx < total and result.protocol_state != ProtocolLifecycleState.COMPLETED:
            step = self._protocol.steps[curr_idx]
            sp = result.step_states.get(step.step_id)
            title = getattr(step, "title", step.step_id)
            inst = getattr(step, "human_instruction", getattr(step, "instruction", ""))

            self.step_id_label.setText(f"{step.step_id} (Step {step.order}/{total})")
            self.step_title_label.setText(title)
            self.step_instruction.setText(inst)

            # Badge shows current candidate or expected
            badge_color = COLOR_ACCENT
            badge_txt = step.expected_event
            if sp and sp.state == StepState.TIMED_OUT:
                badge_color = COLOR_DANGER
                badge_txt = "TIMED OUT"
            elif sp and sp.state == StepState.OUT_OF_ORDER:
                badge_color = COLOR_WARNING
                badge_txt = "OUT OF ORDER"
            self.step_event_badge.set_badge_color(badge_color, badge_txt)

            req_objs = getattr(step, "required_objects", [])
            objs = ", ".join(req_objs) if req_objs else "None"
            self.objects_label.setText(f"Required Objects: {objs}")

            dur = getattr(step, "minimum_evidence_duration_ms", getattr(step, "stable_duration_ms", 500))
            tout = getattr(step, "timeout_seconds", 60)
            conf = getattr(step, "min_confidence", 0.70)

            # Show live accumulating evidence duration if active
            ev_str = f"{sp.evidence_duration_ms:.0f}ms / {dur}ms" if sp else f"-- / {dur}ms"
            self.timeout_label.setText(
                f"Timeout: {tout}s | Evidence: {ev_str} | Min Conf: {int(conf * 100)}%"
            )
        else:
            self.step_id_label.setText("COMPLETED")
            self.step_title_label.setText("Procedure Execution Successfully Completed")
            self.step_instruction.setText("All required experimental procedure steps have been satisfied.")
            self.step_event_badge.set_badge_color(COLOR_SUCCESS, "ALL DONE")
            self.objects_label.setText("Required Objects: Complete")
            self.timeout_label.setText("All compliance requirements satisfied.")

        # 4. Update Checklist Items with rich states
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            st = self._protocol.steps[i]
            st_title = getattr(st, "title", st.step_id)
            sp = result.step_states.get(st.step_id)

            if sp and sp.state == StepState.COMPLETED:
                item.setText(f"✓ [{st.step_id}] {st_title}")
                item.setForeground(QColor("#3fb950"))  # Green
            elif sp and sp.state == StepState.IN_PROGRESS:
                item.setText(f"▶ [{st.step_id}] {st_title} (Active)")
                item.setForeground(QColor("#58a6ff"))  # Cyan
            elif sp and sp.state == StepState.OUT_OF_ORDER:
                item.setText(f"⚠ [{st.step_id}] {st_title} (Out of Order Attempt)")
                item.setForeground(QColor("#d29922"))  # Orange
            elif sp and sp.state == StepState.TIMED_OUT:
                item.setText(f"✕ [{st.step_id}] {st_title} (Timed Out)")
                item.setForeground(QColor("#f85149"))  # Red
            elif sp and sp.state == StepState.SKIPPED:
                item.setText(f"⊘ [{st.step_id}] {st_title} (Skipped)")
                item.setForeground(QColor("#d29922"))  # Yellow
            else:
                item.setText(f"○ [{st.step_id}] {st_title} (Pending)")
                item.setForeground(QColor("#8b949e"))  # Gray

        # 5. Next Step Guidance
        if curr_idx + 1 < total and result.protocol_state != ProtocolLifecycleState.COMPLETED:
            next_st = self._protocol.steps[curr_idx + 1]
            next_title = getattr(next_st, "title", next_st.step_id)
            self.next_step_text.setText(f"Next: [{next_st.step_id}] {next_title}")
        elif result.protocol_state == ProtocolLifecycleState.COMPLETED:
            self.next_step_text.setText("All steps completed. Ready for procedure audit verification.")
        else:
            self.next_step_text.setText("Final procedure step in progress.")
