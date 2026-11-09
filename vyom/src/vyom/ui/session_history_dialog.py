"""Session history, cryptographic audit verification, and report export dialog.

Part of VYOM Prompt 11: Persistent Session Management and Audit Verification UI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vyom.audit.hash_chain import AuditRecord
from vyom.audit.reporter import ExperimentReporter
from vyom.audit.verifier import (
    AuditVerificationResult,
    AuditVerificationState,
    load_audit_records_from_jsonl,
    verify_audit_chain,
)
from vyom.core.logging_setup import get_logger
from vyom.storage.models import SessionRecord
from vyom.storage.session_manager import SessionManager
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

logger = get_logger("ui.session_history")


class SessionHistoryDialog(QDialog):
    """Dialog displaying historical experiment sessions with one-click audit verification and reporting."""

    def __init__(
        self,
        session_manager: SessionManager,
        reporter: ExperimentReporter,
        parent: Optional[QWidget] = None,
        audit_dir: Path | str = "sessions",
    ) -> None:
        super().__init__(parent)
        self.session_manager = session_manager
        self.reporter = reporter
        self.audit_dir = Path(audit_dir).resolve()
        self._selected_session: Optional[SessionRecord] = None

        self.setWindowTitle("VYOM — Experiment Sessions & Cryptographic Audit History")
        self.setMinimumSize(960, 640)
        self.resize(1080, 700)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        self._init_ui()
        self._load_sessions()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header title
        header = QLabel("MISSION EXPERIMENT ARCHIVE & AUDIT TRAIL")
        header.setStyleSheet("font-size: 16px; font-weight: bold; letter-spacing: 0.5px;")
        main_layout.addWidget(header)

        # Splitter: Left table, Right details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Session list table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(8)

        table_label = QLabel("Recorded Experiment Sessions")
        table_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 600; font-size: 12px;")
        table_layout.addWidget(table_label)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Session ID", "Protocol", "Status", "Duration", "Steps", "Violations", "Audit Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_session_selected)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_container)

        # Right panel: Details & Verification Panel
        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(10)

        details_label = QLabel("Session Cryptographic Seal & Provenance")
        details_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 600; font-size: 12px;")
        details_layout.addWidget(details_label)

        # Details text / info box
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet(
            f"background-color: {BG_CARD}; border: 1px solid {BORDER_SUBTLE}; "
            f"border-radius: 6px; font-family: 'Consolas', monospace; font-size: 12px; padding: 8px;"
        )
        self.details_text.setPlaceholderText("Select an experiment session to inspect cryptographic chain and evidence.")
        details_layout.addWidget(self.details_text)

        # Actions group
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        self.btn_verify = QPushButton("🔍 Verify Audit Chain")
        self.btn_verify.setEnabled(False)
        self.btn_verify.setStyleSheet(
            f"background-color: {COLOR_ACCENT}; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px;"
        )
        self.btn_verify.clicked.connect(self._verify_selected_session)
        actions_layout.addWidget(self.btn_verify)

        self.btn_reports = QPushButton("📊 Export / Open Reports")
        self.btn_reports.setEnabled(False)
        self.btn_reports.setStyleSheet(
            f"background-color: #238636; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px;"
        )
        self.btn_reports.clicked.connect(self._export_and_open_reports)
        actions_layout.addWidget(self.btn_reports)

        self.btn_manifest = QPushButton("📜 View Manifest")
        self.btn_manifest.setEnabled(False)
        self.btn_manifest.setStyleSheet(
            f"background-color: {BG_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_SUBTLE}; padding: 8px 12px; border-radius: 4px;"
        )
        self.btn_manifest.clicked.connect(self._view_manifest)
        actions_layout.addWidget(self.btn_manifest)

        details_layout.addLayout(actions_layout)
        splitter.addWidget(details_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Dialog bottom close button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def _load_sessions(self) -> None:
        """Fetch and populate past sessions from SQLite database."""
        sessions = self.session_manager.list_sessions(limit=100)
        self.table.setRowCount(len(sessions))

        for row_idx, s in enumerate(sessions):
            self.table.setItem(row_idx, 0, QTableWidgetItem(s.session_id))
            self.table.setItem(row_idx, 1, QTableWidgetItem(s.protocol_name))

            item_status = QTableWidgetItem(s.status)
            if s.status == "COMPLETED":
                item_status.setForeground(QColor(COLOR_SUCCESS))
            elif s.status == "TERMINATED":
                item_status.setForeground(QColor(COLOR_DANGER))
            self.table.setItem(row_idx, 2, item_status)

            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{s.duration_s:.1f}s"))
            self.table.setItem(row_idx, 4, QTableWidgetItem(f"{s.completed_steps}/{s.total_steps}"))

            item_viol = QTableWidgetItem(str(s.violations_count))
            if s.violations_count > 0:
                item_viol.setForeground(QColor(COLOR_DANGER))
            else:
                item_viol.setForeground(QColor(COLOR_SUCCESS))
            self.table.setItem(row_idx, 5, item_viol)

            item_audit = QTableWidgetItem(s.audit_status)
            if s.audit_status == AuditVerificationState.INTEGRITY_VERIFIED.value:
                item_audit.setForeground(QColor(COLOR_SUCCESS))
            elif s.audit_status in (AuditVerificationState.HASH_MISMATCH.value, AuditVerificationState.SEQUENCE_ERROR.value):
                item_audit.setForeground(QColor(COLOR_DANGER))
            else:
                item_audit.setForeground(QColor(TEXT_MUTED))
            self.table.setItem(row_idx, 6, item_audit)

        if sessions:
            self.table.selectRow(0)

    def _on_session_selected(self) -> None:
        """Handle selection change in the session table."""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            self._selected_session = None
            self.btn_verify.setEnabled(False)
            self.btn_reports.setEnabled(False)
            self.btn_manifest.setEnabled(False)
            self.details_text.clear()
            return

        row = self.table.currentRow()
        sid_item = self.table.item(row, 0)
        if not sid_item:
            return

        session_id = sid_item.text()
        session = self.session_manager.get_session(session_id)
        if not session:
            return

        self._selected_session = session
        self.btn_verify.setEnabled(True)
        self.btn_reports.setEnabled(True)
        self.btn_manifest.setEnabled(True)

        self._render_session_details(session)

    def _render_session_details(
        self,
        session: SessionRecord,
        ver_res: Optional[AuditVerificationResult] = None,
    ) -> None:
        """Format and display session provenance and verification details."""
        steps = self.session_manager.get_step_results(session.session_id)
        violations = self.session_manager.get_violations(session.session_id)
        recordings = self.session_manager.get_recordings(session.session_id)

        status_color = COLOR_SUCCESS if session.status == "COMPLETED" else COLOR_WARNING

        lines = [
            f"=== EXPERIMENT SESSION: {session.session_id} ===",
            f"Protocol:       {session.protocol_name} ({session.protocol_id} v{session.protocol_version})",
            f"State:          {session.status} (Duration: {session.duration_s:.1f}s)",
            f"Start UTC:      {session.start_time_utc}",
            f"End UTC:        {session.end_time_utc or 'In Progress'}",
            f"Hardware:       {session.hardware_device or 'CPU'} ({session.hardware_backend or 'Default'})",
            f"Model Version:  {session.model_version or 'N/A'}",
            f"Steps:          {session.completed_steps} completed / {session.total_steps} total",
            f"Violations:     {session.violations_count} detected",
            "",
            "=== CRYPTOGRAPHIC AUDIT STATUS ===",
            f"Audit Status:   {ver_res.status.value if ver_res else session.audit_status}",
            f"Final Digest:   {ver_res.final_hash if ver_res else (session.final_chain_digest or 'Not Verified')}",
        ]

        if ver_res:
            lines.extend([
                f"Genesis Hash:   {ver_res.genesis_hash}",
                f"Events Checked: {ver_res.total_records} sequential records",
                f"Verified At:    {ver_res.verification_timestamp_utc}",
            ])
            if ver_res.error_message:
                lines.append(f"VERIFICATION FAILURE: {ver_res.error_message}")
        else:
            lines.append("Click 'Verify Audit Chain' to independently replay all hash records.")

        if recordings:
            lines.append("")
            lines.append("=== VIDEO RECORDING EVIDENCE ===")
            for r in recordings:
                lines.append(
                    f"File:   {Path(r.file_path).name} ({r.width}x{r.height} @ {r.fps:.1f}fps, {r.duration_s:.1f}s)"
                )
                lines.append(f"SHA-256: {r.sha256_checksum or 'Pending'}")

        if violations:
            lines.append("")
            lines.append(f"=== COMPLIANCE VIOLATIONS ({len(violations)}) ===")
            for v in violations[:5]:
                lines.append(
                    f"[{v.get('timestamp_s', 0.0):.1f}s] {v.get('severity', 'MED')} | "
                    f"Step: {v.get('step_id')} | {v.get('explanation')}"
                )
            if len(violations) > 5:
                lines.append(f"... and {len(violations) - 5} more violations.")

        self.details_text.setPlainText("\n".join(lines))

    def _get_audit_records_for_session(self, session_id: str) -> List[AuditRecord]:
        """Fetch audit records from SQLite or fallback to JSONL file."""
        raw_db_records = self.session_manager.get_audit_records(session_id)
        if raw_db_records:
            records = []
            for r in raw_db_records:
                payload = json.loads(r["payload_json"]) if "payload_json" in r else {}
                records.append(
                    AuditRecord(
                        sequence_num=r["sequence_num"],
                        event_id=r["event_id"],
                        session_id=r["session_id"],
                        timestamp_utc=r["timestamp_utc"],
                        timestamp_monotonic=r["timestamp_monotonic"],
                        event_type=r["event_type"],
                        payload=payload,
                        previous_hash=r["previous_hash"],
                        current_hash=r["current_hash"],
                    )
                )
            return records

        # Try JSONL
        jsonl_candidates = [
            self.audit_dir / session_id / f"{session_id}_audit.jsonl",
            self.audit_dir / f"{session_id}_audit.jsonl",
        ]
        for p in jsonl_candidates:
            if p.is_file():
                return load_audit_records_from_jsonl(p)

        return []

    def _verify_selected_session(self) -> None:
        """Run independent cryptographic replay verification on selected session."""
        if not self._selected_session:
            return

        session = self._selected_session
        records = self._get_audit_records_for_session(session.session_id)

        res = verify_audit_chain(records)

        # Update SQLite audit status
        self.session_manager.update_audit_status(
            session_id=session.session_id,
            status=res.status.value,
            final_digest=res.final_hash or None,
        )

        # Refresh details display
        self._render_session_details(session, ver_res=res)

        # Refresh table item
        row = self.table.currentRow()
        item = self.table.item(row, 6)
        if item:
            item.setText(res.status.value)
            if res.is_valid:
                item.setForeground(QColor(COLOR_SUCCESS))
            else:
                item.setForeground(QColor(COLOR_DANGER))

        # Show notification dialog
        if res.is_valid:
            QMessageBox.information(
                self,
                "Audit Verification Passed",
                f"✓ Cryptographic SHA-256 Audit Chain Verified!\n\n"
                f"Session: {session.session_id}\n"
                f"Events Checked: {res.total_records}\n"
                f"Genesis Hash: {res.genesis_hash[:16]}...\n"
                f"Final Digest: {res.final_hash}\n\n"
                f"Status: {res.status.value}\n"
                f"All hashes mathematically verified against canonical serialization.",
            )
        else:
            QMessageBox.warning(
                self,
                "Audit Verification Failed",
                f"✗ Tamper or Discontinuity Detected!\n\n"
                f"Session: {session.session_id}\n"
                f"Status: {res.status.value}\n"
                f"Sequence: #{res.error_record_sequence}\n"
                f"Error: {res.error_message}",
            )

    def _export_and_open_reports(self) -> None:
        """Generate full suite of session reports and open folder."""
        if not self._selected_session:
            return

        session = self._selected_session
        records = self._get_audit_records_for_session(session.session_id)
        ver_res = verify_audit_chain(records)

        steps = self.session_manager.get_step_results(session.session_id)
        violations = self.session_manager.get_violations(session.session_id)
        voice_alerts = self.session_manager.get_voice_alerts(session.session_id)
        recordings = self.session_manager.get_recordings(session.session_id)

        generated = self.reporter.export_all_reports(
            session=session,
            step_results=steps,
            violations=violations,
            voice_alerts=voice_alerts,
            recordings=recordings,
            verification_result=ver_res,
        )

        html_path = generated["html"]
        report_dir = html_path.parent

        # Open in Explorer / default application
        try:
            if sys.platform == "win32":
                os.startfile(html_path)
            else:
                subprocess.Popen(["xdg-open", str(html_path)])
        except Exception as e:
            logger.warning("Could not launch default viewer: %s", e)

        QMessageBox.information(
            self,
            "Reports Exported",
            f"Successfully exported experiment certification package:\n\n"
            f"HTML Report: {html_path.name}\n"
            f"JSON Summary: {generated['json'].name}\n"
            f"Step CSV: {generated['csv'].name}\n"
            f"Cryptographic Manifest: {generated['manifest'].name}\n\n"
            f"Location: {report_dir}",
        )

    def _view_manifest(self) -> None:
        """Display the exported JSON manifest if available, or generate it."""
        if not self._selected_session:
            return

        session = self._selected_session
        rep_dir = self.reporter.get_session_report_dir(session.session_id)
        manifest_path = rep_dir / f"{session.session_id}_manifest.json"

        if not manifest_path.is_file():
            # Generate reports first
            self._export_and_open_reports()

        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read()

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Cryptographic Audit Manifest — {manifest_path.name}")
            dialog.resize(700, 500)
            dialog.setStyleSheet(APPLICATION_STYLESHEET)
            d_layout = QVBoxLayout(dialog)

            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(content)
            text.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
            d_layout.addWidget(text)

            btn = QPushButton("Close")
            btn.clicked.connect(dialog.accept)
            d_layout.addWidget(btn)
            dialog.exec()

