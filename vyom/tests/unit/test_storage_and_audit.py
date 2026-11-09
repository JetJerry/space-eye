"""Comprehensive unit tests for persistent SQLite storage, SHA-256 hash chaining, and reporting.

Part of VYOM Prompt 11: Persistent Storage, Audit Logging, and Experiment Reports.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Dict
import pytest

from vyom.audit.canonical import canonical_dumps, sha256_digest
from vyom.audit.hash_chain import (
    GENESIS_PREV_HASH,
    AuditChainLogger,
    AuditRecord,
)
from vyom.audit.reporter import ExperimentReporter
from vyom.audit.verifier import (
    AuditVerificationResult,
    AuditVerificationState,
    export_audit_manifest,
    load_audit_records_from_jsonl,
    verify_audit_chain,
)
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProcedureViolation,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolStep,
    StepProgress,
    StepState,
    ViolationType,
)
from vyom.storage.database import CURRENT_SCHEMA_VERSION, DatabaseManager
from vyom.storage.models import RecordingRecord, SessionRecord
from vyom.storage.session_manager import SessionManager, compute_file_sha256


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_protocol() -> ProtocolDefinition:
    step1 = ProtocolStep(
        step_id="STEP_01",
        order=1,
        title="Pick up Sample Vial",
        human_instruction="Carefully grasp the sample vial.",
        expected_event="PICKUP_CANDIDATE",
        required_objects=["sample_vial"],
        optional=False,
    )
    step2 = ProtocolStep(
        step_id="STEP_02",
        order=2,
        title="Insert Vial into Test Rig Slot",
        human_instruction="Place the sample vial securely into rig slot A.",
        expected_event="PLACEMENT_CANDIDATE",
        required_objects=["sample_vial", "rig_slot"],
        optional=False,
    )
    return ProtocolDefinition(
        protocol_id="PROTO_TEST_01",
        protocol_version="1.0.0",
        experiment_name="Sample Vial Placement Test",
        description="Test protocol for storage and audit unit tests.",
        steps=[step1, step2],
    )


# ==============================================================================
# 1. SQLite Database Schema & Manager Tests
# ==============================================================================


def test_database_initialization(temp_dir: Path):
    db_path = temp_dir / "test_vyom.db"
    db = DatabaseManager(db_path)
    db.initialize()

    assert db_path.is_file()

    # Verify tables exist
    with db.connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = {row["name"] for row in tables}

    expected_tables = {
        "schema_version",
        "sessions",
        "protocol_executions",
        "step_results",
        "interaction_events",
        "compliance_violations",
        "voice_alerts",
        "session_recordings",
        "audit_chain_records",
    }
    assert expected_tables.issubset(table_names)

    # Verify schema version
    assert db.get_schema_version() == CURRENT_SCHEMA_VERSION

    # Calling initialize again should be idempotent
    db.initialize()
    assert db.get_schema_version() == CURRENT_SCHEMA_VERSION


# ==============================================================================
# 2. SessionManager Lifecycle & Persistence Tests
# ==============================================================================


def test_session_manager_lifecycle(temp_dir: Path, sample_protocol: ProtocolDefinition):
    db_path = temp_dir / "lifecycle.db"
    db = DatabaseManager(db_path)
    db.initialize()
    mgr = SessionManager(db)

    # 1. Create Session
    session = mgr.create_session(
        protocol=sample_protocol,
        camera_source="Webcam Device 0",
        hardware_device="GPU",
        hardware_backend="CUDA",
        model_version="smoke_v1",
    )
    assert session.session_id.startswith("SES_")
    assert session.status == "IDLE"
    assert session.total_steps == 2
    assert session.completed_steps == 0

    # 2. Start Session
    started = mgr.start_session(session.session_id)
    assert started is True
    s_active = mgr.get_session(session.session_id)
    assert s_active is not None
    assert s_active.status == "ACTIVE"

    # 3. Pause & Resume
    assert mgr.pause_session(session.session_id) is True
    assert mgr.get_session(session.session_id).status == "PAUSED"
    assert mgr.resume_session(session.session_id) is True
    assert mgr.get_session(session.session_id).status == "ACTIVE"

    # 4. Record Compliance Violation
    viol = ProcedureViolation(
        violation_id="viol_001",
        violation_type=ViolationType.OUT_OF_ORDER,
        step_id="STEP_02",
        expected_action="STEP_01",
        observed_action="STEP_02 attempted before STEP_01",
        severity="critical",
        explanation="Step 2 attempted before Step 1 was completed.",
        voice_message="Warning: Please complete Step 1 first.",
        timestamp_s=15.0,
    )
    mgr.record_violation(session.session_id, viol)
    viols = mgr.get_violations(session.session_id)
    assert len(viols) == 1
    assert viols[0]["violation_id"] == "viol_001"
    assert viols[0]["severity"] == "critical"

    # 5. Record Voice Alert
    mgr.record_voice_alert(
        session_id=session.session_id,
        alert_id="alert_001",
        category="VIOLATION",
        priority=1,
        text="Warning: Please complete Step 1 first.",
        reason="OUT_OF_ORDER",
        status="SPOKEN",
    )
    alerts = mgr.get_voice_alerts(session.session_id)
    assert len(alerts) == 1
    assert alerts[0]["status"] == "SPOKEN"

    # 6. Finalize Recording
    fake_rec_file = temp_dir / "session_rec.mp4"
    fake_rec_file.write_bytes(b"dummy video bytes 1234567890")
    rec = mgr.finalize_recording(
        session_id=session.session_id,
        file_path=str(fake_rec_file),
        total_frames=300,
        duration_s=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )
    assert rec.sha256_checksum is not None
    assert len(rec.sha256_checksum) == 64

    # 7. Stop Session
    stopped = mgr.stop_session(
        session_id=session.session_id,
        is_completed=True,
        final_digest="fedcba9876543210" * 4,
    )
    assert stopped is True
    s_done = mgr.get_session(session.session_id)
    assert s_done.status == "COMPLETED"
    assert s_done.final_chain_digest == "fedcba9876543210" * 4

    # 8. Reconnect test: Open new DB connection and verify persistence
    db2 = DatabaseManager(db_path)
    db2.initialize()
    mgr2 = SessionManager(db2)
    s_reloaded = mgr2.get_session(session.session_id)
    assert s_reloaded is not None
    assert s_reloaded.session_id == session.session_id
    assert s_reloaded.status == "COMPLETED"
    assert s_reloaded.final_chain_digest == "fedcba9876543210" * 4
    assert len(mgr2.get_violations(session.session_id)) == 1
    assert len(mgr2.get_recordings(session.session_id)) == 1


# ==============================================================================
# 3. Canonical JSON Byte Determinism & Rejection Tests
# ==============================================================================


def test_canonical_json_determinism():
    dict_a = {"z": 100, "a": [3, 2, 1], "m": {"inner_b": "val", "inner_a": True}}
    dict_b = {"m": {"inner_a": True, "inner_b": "val"}, "a": [3, 2, 1], "z": 100}

    bytes_a = canonical_dumps(dict_a)
    bytes_b = canonical_dumps(dict_b)

    assert bytes_a == bytes_b
    assert sha256_digest(bytes_a) == sha256_digest(bytes_b)


def test_canonical_json_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        canonical_dumps({"bad_val": float("nan")})

    with pytest.raises(ValueError):
        canonical_dumps({"bad_val": float("inf")})

    with pytest.raises(ValueError):
        canonical_dumps({"bad_val": float("-inf")})


# ==============================================================================
# 4. SHA-256 Hash Chaining & Append-Only Logger Tests
# ==============================================================================


def test_audit_chain_logger_and_dual_write(temp_dir: Path):
    db_path = temp_dir / "audit_chain.db"
    db = DatabaseManager(db_path)
    db.initialize()

    session_id = "SES_TEST_CHAIN_001"
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, protocol_id, protocol_version, protocol_name, start_time_utc, status, created_at, updated_at) "
            "VALUES (?, 'PROTO_101', '1.0', 'Microgravity Fluid Mixing', '2026-11-09T00:00:00Z', 'ACTIVE', '2026-11-09T00:00:00Z', '2026-11-09T00:00:00Z');",
            (session_id,),
        )

    audit_dir = temp_dir / "audit"
    logger = AuditChainLogger(session_id=session_id, audit_dir=audit_dir, db_manager=db)

    # 1. Initialize Genesis Record
    genesis = logger.initialize_genesis(
        protocol_id="PROTO_101",
        protocol_version="1.0",
        experiment_name="Microgravity Fluid Mixing",
    )
    assert genesis.sequence_num == 0
    assert genesis.previous_hash == GENESIS_PREV_HASH
    assert len(genesis.current_hash) == 64
    assert genesis.current_hash == genesis.calculate_expected_hash()

    # 2. Append Event #1
    evt1 = logger.append_event(
        event_type="ARUCO_CALIBRATION",
        payload={"origin_marker_id": 0, "status": "CALIBRATED"},
    )
    assert evt1.sequence_num == 1
    assert evt1.previous_hash == genesis.current_hash
    assert evt1.current_hash == evt1.calculate_expected_hash()

    # 3. Append Event #2
    evt2 = logger.append_event(
        event_type="INTERACTION_EVENT",
        payload={"event_type": "pickup", "object": "sample_vial", "confidence": 0.95},
    )
    assert evt2.sequence_num == 2
    assert evt2.previous_hash == evt1.current_hash
    assert evt2.current_hash == evt2.calculate_expected_hash()

    # 4. Verify Dual-Write to JSONL
    jsonl_file = audit_dir / f"{session_id}_audit.jsonl"
    assert jsonl_file.is_file()
    records_from_file = load_audit_records_from_jsonl(jsonl_file)
    assert len(records_from_file) == 3
    assert records_from_file[0].current_hash == genesis.current_hash
    assert records_from_file[1].current_hash == evt1.current_hash
    assert records_from_file[2].current_hash == evt2.current_hash

    # 5. Verify Dual-Write to SQLite
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT sequence_num, event_type, current_hash FROM audit_chain_records WHERE session_id = ? ORDER BY sequence_num ASC;",
            (session_id,),
        ).fetchall()
    assert len(rows) == 3
    assert rows[0]["sequence_num"] == 0
    assert rows[1]["sequence_num"] == 1
    assert rows[2]["sequence_num"] == 2
    assert rows[2]["current_hash"] == evt2.current_hash


# ==============================================================================
# 5. Independent Audit Chain Verifier & Tamper Detection Tests
# ==============================================================================


def test_audit_verifier_valid_chain(temp_dir: Path):
    audit_dir = temp_dir / "audit_valid"
    logger = AuditChainLogger(session_id="SES_VALID", audit_dir=audit_dir)
    logger.initialize_genesis("P1", "1.0", "Experiment")
    for i in range(1, 6):
        logger.append_event(f"EVENT_{i}", {"index": i, "data": f"payload_{i}"})

    res = verify_audit_chain(logger.records)
    assert res.is_valid is True
    assert res.status == AuditVerificationState.INTEGRITY_VERIFIED
    assert res.total_records == 6
    assert res.error_message is None
    assert len(res.final_hash) == 64


def test_audit_verifier_detects_tampered_payload(temp_dir: Path):
    audit_dir = temp_dir / "audit_tamper"
    logger = AuditChainLogger(session_id="SES_TAMPER", audit_dir=audit_dir)
    logger.initialize_genesis("P1", "1.0", "Experiment")
    logger.append_event("STEP_1", {"object": "beaker"})
    logger.append_event("STEP_2", {"object": "flask"})

    records = logger.records
    assert len(records) == 3

    # Tamper with payload of record #1
    records[1].payload["object"] = "TAMPERED_OBJECT"

    res = verify_audit_chain(records)
    assert res.is_valid is False
    assert res.status == AuditVerificationState.HASH_MISMATCH
    assert res.error_record_sequence == 1
    assert "Payload tampering" in res.error_message or "mismatch" in res.error_message.lower()


def test_audit_verifier_detects_missing_record(temp_dir: Path):
    audit_dir = temp_dir / "audit_missing"
    logger = AuditChainLogger(session_id="SES_MISSING", audit_dir=audit_dir)
    logger.initialize_genesis("P1", "1.0", "Experiment")
    logger.append_event("EVT_1", {"k": 1})
    logger.append_event("EVT_2", {"k": 2})
    logger.append_event("EVT_3", {"k": 3})

    records = logger.records
    # Delete record #2 (so sequence is 0, 1, 3)
    corrupted = [records[0], records[1], records[3]]

    res = verify_audit_chain(corrupted)
    assert res.is_valid is False
    assert res.status == AuditVerificationState.MISSING_RECORD
    assert res.error_record_sequence == 3


def test_audit_verifier_detects_reordered_records(temp_dir: Path):
    audit_dir = temp_dir / "audit_reorder"
    logger = AuditChainLogger(session_id="SES_REORDER", audit_dir=audit_dir)
    logger.initialize_genesis("P1", "1.0", "Experiment")
    logger.append_event("EVT_1", {"k": 1})
    logger.append_event("EVT_2", {"k": 2})

    records = logger.records
    # Swap record 1 and record 2
    swapped = [records[0], records[2], records[1]]

    res = verify_audit_chain(swapped)
    assert res.is_valid is False
    assert res.status in (
        AuditVerificationState.SEQUENCE_ERROR,
        AuditVerificationState.MISSING_RECORD,
        AuditVerificationState.HASH_MISMATCH,
    )


def test_audit_verifier_detects_corrupt_genesis(temp_dir: Path):
    rec = AuditRecord(
        sequence_num=1,  # Not 0!
        event_id="e1",
        session_id="s1",
        timestamp_utc="",
        timestamp_monotonic=0.0,
        event_type="INIT",
        payload={},
        previous_hash=GENESIS_PREV_HASH,
        current_hash="abc",
    )
    res = verify_audit_chain([rec])
    assert res.is_valid is False
    assert res.status == AuditVerificationState.INVALID_GENESIS


# ==============================================================================
# 6. Automatic Experiment Reporting & Manifest Export Tests
# ==============================================================================


def test_experiment_reporting_package(temp_dir: Path, sample_protocol: ProtocolDefinition):
    reporter = ExperimentReporter(output_root=temp_dir / "reports_root")

    session = SessionRecord(
        session_id="SES_REPORT_TEST",
        protocol_id=sample_protocol.protocol_id,
        protocol_version=sample_protocol.protocol_version,
        protocol_name=sample_protocol.experiment_name,
        start_time_utc="2026-11-09T12:00:00Z",
        end_time_utc="2026-11-09T12:05:00Z",
        duration_s=300.0,
        status="COMPLETED",
        camera_source="Webcam Device 0",
        hardware_device="CPU",
        hardware_backend="OpenCV",
        total_steps=2,
        completed_steps=2,
        violations_count=0,
        final_chain_digest="1234567890abcdef" * 4,
    )

    step_results = [
        {
            "step_order": 1,
            "step_id": "STEP_01",
            "step_name": "Pick up Sample Vial",
            "state": "COMPLETED",
            "is_required": 1,
            "elapsed_s": 45.2,
            "failure_reason": None,
        },
        {
            "step_order": 2,
            "step_id": "STEP_02",
            "step_name": "Insert Vial into Test Rig Slot",
            "state": "COMPLETED",
            "is_required": 1,
            "elapsed_s": 62.1,
            "failure_reason": None,
        },
    ]

    violations = []
    voice_alerts = [
        {
            "alert_id": "a1",
            "category": "GUIDANCE",
            "priority": 3,
            "text": "Procedure completed successfully.",
            "timestamp_s": 107.3,
            "status": "SPOKEN",
        }
    ]

    rec_file = temp_dir / "test_session_video.mp4"
    rec_file.write_bytes(b"dummy mp4 video data for report test")
    checksum = compute_file_sha256(rec_file)

    recordings = [
        RecordingRecord(
            recording_id="rec_01",
            session_id=session.session_id,
            file_path=str(rec_file),
            file_size_bytes=rec_file.stat().st_size,
            sha256_checksum=checksum,
            total_frames=1500,
            duration_s=50.0,
            fps=30.0,
            width=1280,
            height=720,
        )
    ]

    ver_res = AuditVerificationResult(
        status=AuditVerificationState.INTEGRITY_VERIFIED,
        is_valid=True,
        total_records=12,
        genesis_hash="genesis_hash_val" * 4,
        final_hash=session.final_chain_digest,
        verification_timestamp_utc="2026-11-09T12:05:01Z",
    )

    exported = reporter.export_all_reports(
        session=session,
        step_results=step_results,
        violations=violations,
        voice_alerts=voice_alerts,
        recordings=recordings,
        verification_result=ver_res,
        operator_id="ASTRONAUT_EX_1",
    )

    # 1. Verify HTML Report
    html_path = exported["html"]
    assert html_path.is_file()
    html_content = html_path.read_text(encoding="utf-8")
    assert "VYOM Experiment Compliance Certification" in html_content
    assert "INTEGRITY VERIFIED" in html_content
    assert "Zero compliance violations recorded" in html_content
    assert session.session_id in html_content

    # 2. Verify JSON Report
    json_path = exported["json"]
    assert json_path.is_file()
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["session"]["session_id"] == session.session_id
    assert json_data["verification"]["is_valid"] is True
    assert json_data["metrics"]["completion_rate_pct"] == 100.0

    # 3. Verify CSV Summary
    csv_path = exported["csv"]
    assert csv_path.is_file()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["step_id"] == "STEP_01"
    assert rows[0]["state"] == "COMPLETED"

    # 4. Verify Cryptographic Manifest
    manifest_path = exported["manifest"]
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["session_id"] == session.session_id
    assert manifest_data["verification"]["is_valid"] is True
    assert manifest_data["verification"]["final_hash"] == session.final_chain_digest
    assert len(manifest_data["recordings"]) == 1
    assert manifest_data["recordings"][0]["sha256"] == checksum
    assert "cryptographic_guarantee_notice" in manifest_data
