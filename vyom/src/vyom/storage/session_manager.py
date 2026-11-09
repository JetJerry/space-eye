"""High-level persistent session lifecycle management and database repository.

Part of VYOM Prompt 11: Persistent Session Management, SQLite Storage, and Audit Logging.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from vyom.core.logging_setup import get_logger
from vyom.interaction.events import InteractionEvent
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProcedureViolation,
    ProtocolDefinition,
    StepProgress,
)
from vyom.storage.database import DatabaseManager
from vyom.storage.models import RecordingRecord, SessionRecord

logger = get_logger("storage.session_manager")


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> Optional[str]:
    """Calculate the cryptographic SHA-256 checksum of an on-disk file."""
    path = Path(file_path)
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning("Failed to compute SHA-256 for file '%s': %s", path, e)
        return None


class SessionManager:
    """Manages experiment session lifecycle, state persistence, and relation recording in SQLite."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def generate_session_id(self) -> str:
        """Create a unique, chronologically ordered session slug."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        return f"SES_{now_str}_{rand_suffix}"

    def create_session(
        self,
        protocol: ProtocolDefinition,
        camera_source: Optional[str] = None,
        hardware_device: Optional[str] = None,
        hardware_backend: Optional[str] = None,
        model_version: Optional[str] = None,
        model_checksum: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SessionRecord:
        """Create and persist a new experiment session record in IDLE state."""
        sid = session_id or self.generate_session_id()
        now_utc = datetime.now(timezone.utc).isoformat()

        record = SessionRecord(
            session_id=sid,
            protocol_id=protocol.protocol_id,
            protocol_version=protocol.protocol_version,
            protocol_name=protocol.experiment_name,
            start_time_utc=now_utc,
            status="IDLE",
            camera_source=camera_source,
            hardware_device=hardware_device,
            hardware_backend=hardware_backend,
            model_version=model_version,
            model_checksum=model_checksum,
            rig_calibrated=False,
            total_steps=len(protocol.steps),
            completed_steps=0,
            violations_count=0,
            created_at=now_utc,
            updated_at=now_utc,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, protocol_id, protocol_version, protocol_name,
                    start_time_utc, status, camera_source, hardware_device,
                    hardware_backend, model_version, model_checksum, rig_calibrated,
                    total_steps, completed_steps, violations_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.session_id, record.protocol_id, record.protocol_version, record.protocol_name,
                    record.start_time_utc, record.status, record.camera_source, record.hardware_device,
                    record.hardware_backend, record.model_version, record.model_checksum, int(record.rig_calibrated),
                    record.total_steps, record.completed_steps, record.violations_count, record.created_at, record.updated_at
                ),
            )

            # Insert protocol execution record
            conn.execute(
                """
                INSERT INTO protocol_executions (
                    execution_id, session_id, protocol_id, protocol_version,
                    strict_order, total_steps, completed_steps, is_success, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    f"exec_{sid}", sid, protocol.protocol_id, protocol.protocol_version,
                    int(protocol.strict_order), len(protocol.steps), 0, 0, now_utc
                ),
            )

            # Pre-populate step results
            for step in protocol.steps:
                conn.execute(
                    """
                    INSERT INTO step_results (
                        step_result_id, session_id, step_id, step_order, title, state
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        f"step_{sid}_{step.step_id}", sid, step.step_id, step.order,
                        step.title, "NOT_STARTED"
                    ),
                )

        logger.info("Persisted new session record '%s' for protocol '%s'", sid, protocol.experiment_name)
        return record

    def start_session(self, session_id: str, rig_calibrated: bool = False) -> bool:
        """Transition session to ACTIVE status."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                UPDATE sessions
                SET status = 'ACTIVE', start_time_utc = ?, rig_calibrated = ?, updated_at = ?
                WHERE session_id = ?;
                """,
                (now_utc, int(rig_calibrated), now_utc, session_id),
            )
            return cur.rowcount > 0

    def pause_session(self, session_id: str) -> bool:
        """Transition session to PAUSED status."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = 'PAUSED', updated_at = ? WHERE session_id = ?;",
                (now_utc, session_id),
            )
            return cur.rowcount > 0

    def resume_session(self, session_id: str) -> bool:
        """Transition session back to ACTIVE status."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = 'ACTIVE', updated_at = ? WHERE session_id = ?;",
                (now_utc, session_id),
            )
            return cur.rowcount > 0

    def stop_session(
        self,
        session_id: str,
        is_completed: bool = False,
        final_digest: Optional[str] = None,
        duration_s: Optional[float] = None,
    ) -> bool:
        """Finalize an experiment session with completed or aborted status."""
        now_utc = datetime.now(timezone.utc).isoformat()
        status = "COMPLETED" if is_completed else "ABORTED"

        with self.db.connection() as conn:
            # Query session start time to calculate duration if not passed
            cur = conn.execute("SELECT start_time_utc FROM sessions WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            dur = duration_s or 0.0
            if row and not duration_s:
                try:
                    st = datetime.fromisoformat(row["start_time_utc"])
                    et = datetime.fromisoformat(now_utc)
                    dur = max(0.0, (et - st).total_seconds())
                except Exception:
                    pass

            cur = conn.execute(
                """
                UPDATE sessions
                SET status = ?, end_time_utc = ?, duration_s = ?,
                    final_chain_digest = COALESCE(?, final_chain_digest),
                    updated_at = ?
                WHERE session_id = ?;
                """,
                (status, now_utc, dur, final_digest, now_utc, session_id),
            )

            # Update protocol execution is_success
            conn.execute(
                """
                UPDATE protocol_executions
                SET is_success = ?, completed_at = ?
                WHERE session_id = ?;
                """,
                (int(is_completed), now_utc, session_id),
            )
            return cur.rowcount > 0

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Query full metadata record for a session."""
        with self.db.connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return SessionRecord(
                session_id=row["session_id"],
                protocol_id=row["protocol_id"],
                protocol_version=row["protocol_version"],
                protocol_name=row["protocol_name"],
                start_time_utc=row["start_time_utc"],
                end_time_utc=row["end_time_utc"],
                duration_s=float(row["duration_s"] or 0.0),
                status=row["status"],
                camera_source=row["camera_source"],
                hardware_device=row["hardware_device"],
                hardware_backend=row["hardware_backend"],
                model_version=row["model_version"],
                model_checksum=row["model_checksum"],
                rig_calibrated=bool(row["rig_calibrated"]),
                total_steps=int(row["total_steps"] or 0),
                completed_steps=int(row["completed_steps"] or 0),
                violations_count=int(row["violations_count"] or 0),
                final_chain_digest=row["final_chain_digest"],
                audit_status=row["audit_status"] or "PENDING",
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_sessions(self, limit: int = 50) -> List[SessionRecord]:
        """List past experiment sessions in descending order of creation time."""
        records: List[SessionRecord] = []
        with self.db.connection() as conn:
            cur = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?;", (limit,))
            for row in cur.fetchall():
                records.append(
                    SessionRecord(
                        session_id=row["session_id"],
                        protocol_id=row["protocol_id"],
                        protocol_version=row["protocol_version"],
                        protocol_name=row["protocol_name"],
                        start_time_utc=row["start_time_utc"],
                        end_time_utc=row["end_time_utc"],
                        duration_s=float(row["duration_s"] or 0.0),
                        status=row["status"],
                        camera_source=row["camera_source"],
                        hardware_device=row["hardware_device"],
                        hardware_backend=row["hardware_backend"],
                        model_version=row["model_version"],
                        model_checksum=row["model_checksum"],
                        rig_calibrated=bool(row["rig_calibrated"]),
                        total_steps=int(row["total_steps"] or 0),
                        completed_steps=int(row["completed_steps"] or 0),
                        violations_count=int(row["violations_count"] or 0),
                        final_chain_digest=row["final_chain_digest"],
                        audit_status=row["audit_status"] or "PENDING",
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
        return records

    def sync_evaluation_result(
        self,
        session_id: str,
        result: ProcedureEvaluationResult,
    ) -> None:
        """Update session step states, completion metrics, and counts in database."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET completed_steps = ?, violations_count = ?, updated_at = ?
                WHERE session_id = ?;
                """,
                (result.completed_steps, len(result.all_violations), now_utc, session_id),
            )

            conn.execute(
                "UPDATE protocol_executions SET completed_steps = ? WHERE session_id = ?;",
                (result.completed_steps, session_id),
            )

            for step_id, sp in result.step_states.items():
                conn.execute(
                    """
                    UPDATE step_results
                    SET state = ?, start_time_s = ?, completed_time_s = ?,
                        elapsed_s = ?, evidence_duration_ms = ?, current_event_id = ?,
                        violations_count = ?, failure_reason = ?
                    WHERE session_id = ? AND step_id = ?;
                    """,
                    (
                        sp.state.value, sp.start_time_s, sp.completed_time_s,
                        sp.elapsed_time_s, sp.evidence_duration_ms, sp.current_event_id,
                        sp.violations_count, sp.failure_reason, session_id, step_id
                    ),
                )

    def record_interaction_event(
        self,
        session_id: str,
        event: InteractionEvent,
    ) -> None:
        """Persist an accepted or candidate interaction event."""
        now_utc = datetime.now(timezone.utc).isoformat()
        rig_json = json.dumps(list(event.rig_position)) if event.rig_position else None
        ev_json = json.dumps(event.supporting_evidence)

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO interaction_events (
                    event_id, session_id, frame_id, timestamp_s, event_type,
                    hand_id, handedness, object_track_id, object_class,
                    detection_confidence, interaction_confidence, coordinate_system,
                    calibration_status, rig_position_json, supporting_evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id, session_id, event.frame_id, event.timestamp_s,
                    event.event_type.value, event.hand_id, event.handedness,
                    event.object_track_id, event.object_class, event.detection_confidence,
                    event.interaction_confidence, event.coordinate_system,
                    int(event.calibration_status), rig_json, ev_json, now_utc
                ),
            )

    def record_violation(
        self,
        session_id: str,
        viol: ProcedureViolation,
    ) -> None:
        """Persist a procedure compliance violation."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO compliance_violations (
                    violation_id, session_id, violation_type, step_id,
                    expected_action, observed_action, event_id, object_track_id,
                    object_class, explanation, timestamp_s, frame_id, severity,
                    voice_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    viol.violation_id, session_id, viol.violation_type.value, viol.step_id,
                    viol.expected_action, viol.observed_action, viol.event_id,
                    viol.object_track_id, viol.object_class, viol.explanation,
                    viol.timestamp_s, viol.frame_id, viol.severity, viol.voice_message, now_utc
                ),
            )

    def record_voice_alert(
        self,
        session_id: str,
        alert_id: str,
        category: str,
        priority: int,
        text: str,
        reason: str = "",
        event_id: Optional[str] = None,
        step_id: Optional[str] = None,
        status: str = "QUEUED",
        err: Optional[str] = None,
    ) -> None:
        """Persist a spoken voice alert lifecycle row."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO voice_alerts (
                    alert_id, session_id, category, priority, text, reason,
                    event_id, step_id, timestamp_s, status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    alert_id, session_id, category, priority, text, reason,
                    event_id, step_id, time.time(), status, err, now_utc
                ),
            )

    def finalize_recording(
        self,
        session_id: str,
        file_path: str,
        total_frames: int,
        duration_s: float,
        fps: float,
        width: int,
        height: int,
    ) -> RecordingRecord:
        """Calculate SHA-256 for a finalized recording file and persist metadata."""
        path = Path(file_path).resolve()
        size_bytes = path.stat().st_size if path.exists() else 0
        checksum = compute_file_sha256(path)
        rec_id = f"rec_{session_id}_{uuid.uuid4().hex[:6]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        rec = RecordingRecord(
            recording_id=rec_id,
            session_id=session_id,
            file_path=str(path),
            file_size_bytes=size_bytes,
            sha256_checksum=checksum,
            total_frames=total_frames,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
            created_at=now_utc,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO session_recordings (
                    recording_id, session_id, file_path, file_size_bytes,
                    sha256_checksum, total_frames, duration_s, fps, width, height, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    rec.recording_id, rec.session_id, rec.file_path, rec.file_size_bytes,
                    rec.sha256_checksum, rec.total_frames, rec.duration_s, rec.fps,
                    rec.width, rec.height, rec.created_at
                ),
            )

        logger.info("Finalized recording for '%s': %s (SHA-256: %s)", session_id, path.name, checksum)
        return rec

    def update_audit_status(
        self,
        session_id: str,
        status: str,
        final_digest: Optional[str] = None,
    ) -> None:
        """Record audit verification result in SQLite."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET audit_status = ?, final_chain_digest = COALESCE(?, final_chain_digest), updated_at = ?
                WHERE session_id = ?;
                """,
                (status, final_digest, now_utc, session_id),
            )

    def get_step_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all recorded step outcomes for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM step_results
                WHERE session_id = ?
                ORDER BY step_order ASC;
                """,
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_violations(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all compliance violations recorded for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM compliance_violations
                WHERE session_id = ?
                ORDER BY timestamp_s ASC, rowid ASC;
                """,
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_voice_alerts(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all voice alerts recorded for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM voice_alerts
                WHERE session_id = ?
                ORDER BY timestamp_s ASC, rowid ASC;
                """,
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_interaction_events(self, session_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """Retrieve interaction events for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM interaction_events
                WHERE session_id = ?
                ORDER BY timestamp_s ASC, rowid ASC
                LIMIT ?;
                """,
                (session_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_recordings(self, session_id: str) -> List[RecordingRecord]:
        """Retrieve recorded video files for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM session_recordings
                WHERE session_id = ?
                ORDER BY created_at ASC;
                """,
                (session_id,),
            )
            records = []
            for row in cur.fetchall():
                records.append(
                    RecordingRecord(
                        recording_id=row["recording_id"],
                        session_id=row["session_id"],
                        file_path=row["file_path"],
                        file_size_bytes=int(row["file_size_bytes"] or 0),
                        sha256_checksum=row["sha256_checksum"],
                        total_frames=int(row["total_frames"] or 0),
                        duration_s=float(row["duration_s"] or 0.0),
                        fps=float(row["fps"] or 0.0),
                        width=int(row["width"] or 0),
                        height=int(row["height"] or 0),
                        created_at=row["created_at"],
                    )
                )
            return records

    def get_audit_records(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit chain records stored in SQLite for a session."""
        with self.db.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_chain_records
                WHERE session_id = ?
                ORDER BY sequence_num ASC;
                """,
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]
