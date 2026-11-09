"""SQLite database initialization, schema versioning, foreign-key enforcement, and connection management.

Part of VYOM Prompt 11: Persistent Session Management, SQLite Storage, and Audit Logging.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, Optional

from vyom.core.logging_setup import get_logger

logger = get_logger("storage.database")

CURRENT_SCHEMA_VERSION = 1

SCHEMA_SQL_V1 = """
-- Schema migration metadata
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL,
    description TEXT NOT NULL
);

-- Experiment sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    protocol_id TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    protocol_name TEXT NOT NULL,
    start_time_utc TEXT NOT NULL,
    end_time_utc TEXT,
    duration_s REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'IDLE',
    camera_source TEXT,
    hardware_device TEXT,
    hardware_backend TEXT,
    model_version TEXT,
    model_checksum TEXT,
    rig_calibrated INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,
    violations_count INTEGER DEFAULT 0,
    final_chain_digest TEXT,
    audit_status TEXT DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Protocol executions
CREATE TABLE IF NOT EXISTS protocol_executions (
    execution_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    protocol_id TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    strict_order INTEGER DEFAULT 1,
    total_steps INTEGER NOT NULL,
    completed_steps INTEGER DEFAULT 0,
    is_success INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

-- Procedure step results
CREATE TABLE IF NOT EXISTS step_results (
    step_result_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    step_id TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    title TEXT NOT NULL,
    state TEXT NOT NULL,
    start_time_s REAL,
    completed_time_s REAL,
    elapsed_s REAL DEFAULT 0.0,
    evidence_duration_ms REAL DEFAULT 0.0,
    current_event_id TEXT,
    violations_count INTEGER DEFAULT 0,
    failure_reason TEXT
);

-- Interaction events
CREATE TABLE IF NOT EXISTS interaction_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL,
    timestamp_s REAL NOT NULL,
    event_type TEXT NOT NULL,
    hand_id TEXT NOT NULL,
    handedness TEXT NOT NULL,
    object_track_id TEXT NOT NULL,
    object_class TEXT NOT NULL,
    detection_confidence REAL NOT NULL,
    interaction_confidence REAL NOT NULL,
    coordinate_system TEXT NOT NULL,
    calibration_status INTEGER NOT NULL,
    rig_position_json TEXT,
    supporting_evidence_json TEXT,
    created_at TEXT NOT NULL
);

-- Compliance violations
CREATE TABLE IF NOT EXISTS compliance_violations (
    violation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    violation_type TEXT NOT NULL,
    step_id TEXT,
    expected_action TEXT NOT NULL,
    observed_action TEXT NOT NULL,
    event_id TEXT,
    object_track_id TEXT,
    object_class TEXT,
    explanation TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    frame_id INTEGER DEFAULT 0,
    severity TEXT NOT NULL,
    voice_message TEXT,
    created_at TEXT NOT NULL
);

-- Spoken voice alerts
CREATE TABLE IF NOT EXISTS voice_alerts (
    alert_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    priority INTEGER NOT NULL,
    text TEXT NOT NULL,
    reason TEXT,
    event_id TEXT,
    step_id TEXT,
    timestamp_s REAL NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

-- Video recording references and file hashes
CREATE TABLE IF NOT EXISTS session_recordings (
    recording_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    sha256_checksum TEXT,
    total_frames INTEGER DEFAULT 0,
    duration_s REAL DEFAULT 0.0,
    fps REAL DEFAULT 30.0,
    width INTEGER DEFAULT 1280,
    height INTEGER DEFAULT 720,
    created_at TEXT NOT NULL
);

-- Cryptographic SHA-256 audit chain records
CREATE TABLE IF NOT EXISTS audit_chain_records (
    sequence_num INTEGER NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    timestamp_monotonic REAL NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    current_hash TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence_num)
);

-- Performance indices
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_step_results_session ON step_results(session_id, step_order);
CREATE INDEX IF NOT EXISTS idx_interaction_events_session ON interaction_events(session_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_compliance_violations_session ON compliance_violations(session_id);
CREATE INDEX IF NOT EXISTS idx_voice_alerts_session ON voice_alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_chain_session_seq ON audit_chain_records(session_id, sequence_num);
"""


class DatabaseManager:
    """Manages SQLite database lifecycle, schema initialization, and transactional connections."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        # Enable Write-Ahead Logging for high concurrency and crash resilience
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active SQLite connection with automatic commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Explicitly run schema migrations and ensure database is ready."""
        self._initialize_database()

    def get_schema_version(self) -> int:
        """Query current applied schema version."""
        with self.connection() as conn:
            cur = conn.execute("SELECT MAX(version) FROM schema_version;")
            row = cur.fetchone()
            return int(row[0]) if (row and row[0] is not None) else 0

    def _initialize_database(self) -> None:
        """Execute migrations and ensure schema is up-to-date."""
        with self.connection() as conn:
            cursor = conn.cursor()
            # Check existing schema version
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';"
            )
            table_exists = cursor.fetchone() is not None

            current_ver = 0
            if table_exists:
                cursor.execute("SELECT MAX(version) FROM schema_version;")
                row = cursor.fetchone()
                if row and row[0] is not None:
                    current_ver = row[0]

            if current_ver < CURRENT_SCHEMA_VERSION:
                logger.info(
                    "Migrating SQLite schema at '%s' from version %d to %d",
                    self.db_path,
                    current_ver,
                    CURRENT_SCHEMA_VERSION,
                )
                cursor.executescript(SCHEMA_SQL_V1)
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO schema_version (version, applied_at_utc, description) VALUES (?, ?, ?);",
                    (CURRENT_SCHEMA_VERSION, now_utc, "Initial Prompt 11 SQLite schema"),
                )
                logger.info("Database schema migration to v%d completed successfully", CURRENT_SCHEMA_VERSION)
            else:
                logger.debug("Database schema at '%s' is up-to-date (v%d)", self.db_path, current_ver)
