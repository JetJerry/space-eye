"""Comprehensive verification script for Prompt 11: Persistent Storage, Hash-Chain Audit Logging & Reports.

Validates:
1. SQLite Database Schema & Relational Architecture (WAL mode, Foreign Keys, 8 tables)
2. Persistent Experiment Session Lifecycle (IDLE -> ACTIVE -> PAUSED -> ACTIVE -> COMPLETED)
3. Relational Event Persistence (Violations, Spoken Alerts, Recordings with SHA-256)
4. Offline Database Persistence Across Reconnect (Fresh connection validation)
5. Canonical JSON Determinism & Float Safety (UTF-8, sorted keys, NaN/Inf rejection)
6. Cryptographic SHA-256 Hash Chaining & Genesis Anchoring (H_0 with 64 zero hex chars)
7. Dual-Write Synchronization (Append-only JSONL file and SQLite audit_chain_records)
8. Independent Cryptographic Replay Verification (Sequence 0..N replay & seal verification)
9. Tamper Detection & Discontinuity Rejection (Payload tampering, deletion, reordering)
10. Automated Multi-Format Certification Package (HTML report, JSON summary, CSV, detached manifest)
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from vyom.audit import (
    GENESIS_PREV_HASH,
    AuditChainLogger,
    AuditRecord,
    AuditVerificationResult,
    AuditVerificationState,
    ExperimentReporter,
    canonical_dumps,
    export_audit_manifest,
    load_audit_records_from_jsonl,
    sha256_digest,
    verify_audit_chain,
)
from vyom.protocol import (
    ProtocolDefinition,
    ProtocolStep,
    ViolationType,
)
from vyom.protocol.protocol_schema import ProcedureViolation
from vyom.storage import (
    CURRENT_SCHEMA_VERSION,
    DatabaseManager,
    RecordingRecord,
    SessionManager,
    SessionRecord,
    compute_file_sha256,
)


def run_checks() -> bool:
    print("=" * 78)
    print("  VYOM PROMPT 11: STORAGE, SHA-256 HASH-CHAIN AUDIT & REPORTING VERIFICATION")
    print("=" * 78)
    all_passed = True

    work_dir = Path(tempfile.mkdtemp(prefix="vyom_prompt11_verify_"))
    try:
        # ---------------------------------------------------------------------
        # Check 1: SQLite Schema & Relational Architecture
        # ---------------------------------------------------------------------
        print("\n[Check 1/10] SQLite Schema & Relational Architecture:")
        db_path = work_dir / "vyom_verify.db"
        db = DatabaseManager(db_path)
        db.initialize()

        with db.connection() as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            pragmas = {
                "foreign_keys": conn.execute("PRAGMA foreign_keys;").fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode;").fetchone()[0].upper(),
            }

        required_tables = [
            "schema_version", "sessions", "protocol_executions", "step_results",
            "interaction_events", "compliance_violations", "voice_alerts",
            "session_recordings", "audit_chain_records",
        ]
        missing = [t for t in required_tables if t not in tables]
        assert not missing, f"Missing tables: {missing}"
        assert pragmas["foreign_keys"] == 1, "Foreign keys must be enabled"
        assert pragmas["journal_mode"] == "WAL", f"Journal mode must be WAL, got {pragmas['journal_mode']}"
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION

        print(f"  ✓ Database created at: {db_path.name}")
        print(f"  ✓ All {len(required_tables)} relational tables verified in schema (Version: {CURRENT_SCHEMA_VERSION})")
        print(f"  ✓ Pragmas enforced: Foreign Keys = {pragmas['foreign_keys']}, Journal Mode = {pragmas['journal_mode']}")

        # ---------------------------------------------------------------------
        # Check 2: Persistent Experiment Session Lifecycle
        # ---------------------------------------------------------------------
        print("\n[Check 2/10] Persistent Experiment Session Lifecycle:")
        mgr = SessionManager(db)

        proto = ProtocolDefinition(
            protocol_id="PROTO_EXP_01",
            protocol_version="1.2.0",
            experiment_name="Microgravity Centrifuge Calibration",
            description="Testing automated session management and persistence.",
            steps=[
                ProtocolStep(
                    step_id="STEP_01",
                    order=1,
                    title="Grasp Calibration Weight",
                    human_instruction="Pick up weight from rig slot 1.",
                    expected_event="PICKUP_CANDIDATE",
                    required_objects=["sample_vial"],
                ),
                ProtocolStep(
                    step_id="STEP_02",
                    order=2,
                    title="Position Weight on Balance",
                    human_instruction="Place weight into slot 2.",
                    expected_event="PLACEMENT_CANDIDATE",
                    required_objects=["sample_vial", "rig_slot"],
                ),
            ],
        )

        session = mgr.create_session(
            protocol=proto,
            camera_source="Webcam Device 0",
            hardware_device="GPU",
            hardware_backend="CUDA",
            model_version="smoke_v1",
        )
        assert session.status == "IDLE"
        assert session.total_steps == 2

        mgr.start_session(session.session_id)
        assert mgr.get_session(session.session_id).status == "ACTIVE"

        mgr.pause_session(session.session_id)
        assert mgr.get_session(session.session_id).status == "PAUSED"

        mgr.resume_session(session.session_id)
        assert mgr.get_session(session.session_id).status == "ACTIVE"

        mgr.stop_session(session.session_id, is_completed=True, final_digest="0123456789abcdef" * 4)
        stopped_s = mgr.get_session(session.session_id)
        assert stopped_s.status == "COMPLETED"
        assert stopped_s.final_chain_digest == "0123456789abcdef" * 4

        print(f"  ✓ Created session: {session.session_id}")
        print("  ✓ Full lifecycle verified: IDLE -> ACTIVE -> PAUSED -> ACTIVE -> COMPLETED")
        print(f"  ✓ Session metrics recorded: {stopped_s.completed_steps}/{stopped_s.total_steps} steps, duration: {stopped_s.duration_s:.2f}s")

        # ---------------------------------------------------------------------
        # Check 3: Relational Event Persistence
        # ---------------------------------------------------------------------
        print("\n[Check 3/10] Relational Event Persistence (Violations, Alerts, Media):")
        # Record Violation
        viol = ProcedureViolation(
            violation_id="viol_audit_01",
            violation_type=ViolationType.OUT_OF_ORDER,
            step_id="STEP_02",
            expected_action="STEP_01",
            observed_action="STEP_02 attempted before STEP_01",
            severity="critical",
            explanation="Safety interlock violated: Weight moved before balance calibrated.",
            voice_message="Warning: Please complete balance calibration first.",
            timestamp_s=42.5,
        )
        mgr.record_violation(session.session_id, viol)
        db_viols = mgr.get_violations(session.session_id)
        assert len(db_viols) == 1
        assert db_viols[0]["violation_id"] == "viol_audit_01"

        # Record Voice Alert
        mgr.record_voice_alert(
            session_id=session.session_id,
            alert_id="alert_audit_01",
            category="VIOLATION",
            priority=1,
            text="Warning: Please complete balance calibration first.",
            reason="OUT_OF_ORDER",
            status="SPOKEN",
        )
        db_alerts = mgr.get_voice_alerts(session.session_id)
        assert len(db_alerts) == 1
        assert db_alerts[0]["status"] == "SPOKEN"

        # Record Video File with Checksum
        fake_video = work_dir / "flight_test.mp4"
        fake_video.write_bytes(b"Simulated experiment exocentric video footage content 12345")
        rec_row = mgr.finalize_recording(
            session_id=session.session_id,
            file_path=str(fake_video),
            total_frames=900,
            duration_s=30.0,
            fps=30.0,
            width=1280,
            height=720,
        )
        assert rec_row.sha256_checksum is not None
        assert len(rec_row.sha256_checksum) == 64
        db_recs = mgr.get_recordings(session.session_id)
        assert len(db_recs) == 1
        assert db_recs[0].sha256_checksum == rec_row.sha256_checksum

        print(f"  ✓ Recorded 1 compliance violation: [{db_viols[0]['severity'].upper()}] {db_viols[0]['explanation']}")
        print(f"  ✓ Recorded 1 voice alert: '{db_alerts[0]['text']}' (status: {db_alerts[0]['status']})")
        print(f"  ✓ Recorded 1 video recording: {Path(rec_row.file_path).name} (SHA-256: {rec_row.sha256_checksum[:16]}...)")

        # ---------------------------------------------------------------------
        # Check 4: Offline Database Persistence Across Reconnect
        # ---------------------------------------------------------------------
        print("\n[Check 4/10] Offline Database Persistence Across Reconnect:")
        # Close old handles, instantiate brand-new DatabaseManager and SessionManager on same file
        del mgr
        del db

        db_reconnected = DatabaseManager(db_path)
        db_reconnected.initialize()
        mgr_reconnected = SessionManager(db_reconnected)

        re_session = mgr_reconnected.get_session(session.session_id)
        assert re_session is not None
        assert re_session.session_id == session.session_id
        assert re_session.status == "COMPLETED"
        assert re_session.protocol_id == "PROTO_EXP_01"
        assert len(mgr_reconnected.get_violations(session.session_id)) == 1
        assert len(mgr_reconnected.get_voice_alerts(session.session_id)) == 1
        assert len(mgr_reconnected.get_recordings(session.session_id)) == 1

        print("  ✓ Reopened fresh database connection on existing disk storage")
        print(f"  ✓ Verified session record '{re_session.session_id}' and all child relations match exactly")

        # ---------------------------------------------------------------------
        # Check 5: Canonical JSON Determinism & Float Safety
        # ---------------------------------------------------------------------
        print("\n[Check 5/10] Canonical JSON Determinism & Float Safety:")
        d1 = {"zebra": 1, "alpha": [3, 2, 1], "nested": {"b": True, "a": False}}
        d2 = {"alpha": [3, 2, 1], "nested": {"a": False, "b": True}, "zebra": 1}

        b1 = canonical_dumps(d1)
        b2 = canonical_dumps(d2)
        assert b1 == b2, "Canonical byte streams must be identical regardless of key order"
        assert sha256_digest(b1) == sha256_digest(b2)

        # Reject NaN and Infinity
        nan_caught = False
        try:
            canonical_dumps({"bad": float("nan")})
        except ValueError:
            nan_caught = True
        assert nan_caught, "NaN must be rejected"

        inf_caught = False
        try:
            canonical_dumps({"bad": float("inf")})
        except ValueError:
            inf_caught = True
        assert inf_caught, "Infinity must be rejected"

        print("  ✓ Canonical byte serialization is strictly deterministic across key permutations")
        print("  ✓ Rejection of non-standard IEEE 754 NaN and Inf confirmed (ValueError enforced)")

        # ---------------------------------------------------------------------
        # Check 6: Cryptographic SHA-256 Hash Chaining & Genesis Anchoring
        # ---------------------------------------------------------------------
        print("\n[Check 6/10] Cryptographic SHA-256 Hash Chaining & Genesis Anchoring:")
        chain_session_id = "SES_CHAIN_VERIFY_001"
        audit_folder = work_dir / "audit_chain"
        audit_folder.mkdir(parents=True, exist_ok=True)

        chain_session = mgr_reconnected.create_session(
            protocol=proto,
            camera_source="File: demo.mp4",
            session_id=chain_session_id,
        )

        chain_logger = AuditChainLogger(
            session_id=chain_session_id,
            audit_dir=audit_folder,
            db_manager=db_reconnected,
        )

        genesis = chain_logger.initialize_genesis(
            protocol_id=proto.protocol_id,
            protocol_version=proto.protocol_version,
            experiment_name=proto.experiment_name,
        )
        assert genesis.sequence_num == 0
        assert genesis.previous_hash == GENESIS_PREV_HASH
        assert genesis.current_hash == genesis.calculate_expected_hash()

        e1 = chain_logger.append_event("STEP_STARTED", {"step_id": "STEP_01"})
        assert e1.sequence_num == 1
        assert e1.previous_hash == genesis.current_hash

        e2 = chain_logger.append_event("INTERACTION_EVENT", {"type": "pickup", "object": "sample_vial"})
        assert e2.sequence_num == 2
        assert e2.previous_hash == e1.current_hash

        e3 = chain_logger.append_event("STEP_COMPLETED", {"step_id": "STEP_01"})
        assert e3.sequence_num == 3
        assert e3.previous_hash == e2.current_hash

        print(f"  ✓ Genesis record sequence 0 initialized with documented invariant: {GENESIS_PREV_HASH[:16]}...")
        print(f"  ✓ Genesis hash computed: {genesis.current_hash}")
        print(f"  ✓ Sequential chaining verified for events 1..3; latest hash: {e3.current_hash[:16]}...")

        # ---------------------------------------------------------------------
        # Check 7: Dual-Write Synchronization (JSONL + SQLite)
        # ---------------------------------------------------------------------
        print("\n[Check 7/10] Dual-Write Synchronization (JSONL + SQLite):")
        jsonl_path = audit_folder / f"{chain_session_id}_audit.jsonl"
        assert jsonl_path.is_file(), "JSONL file must exist on disk"

        jsonl_records = load_audit_records_from_jsonl(jsonl_path)
        assert len(jsonl_records) == 4

        db_records = mgr_reconnected.get_audit_records(chain_session_id)
        assert len(db_records) == 4

        for i in range(4):
            assert jsonl_records[i].current_hash == db_records[i]["current_hash"]
            assert jsonl_records[i].previous_hash == db_records[i]["previous_hash"]

        print(f"  ✓ JSONL file written to disk: {jsonl_path.name} (4 lines)")
        print(f"  ✓ SQLite table 'audit_chain_records' contains 4 matching rows in lockstep")

        # ---------------------------------------------------------------------
        # Check 8: Independent Cryptographic Replay Verification
        # ---------------------------------------------------------------------
        print("\n[Check 8/10] Independent Cryptographic Replay Verification:")
        ver_res = verify_audit_chain(jsonl_records)
        assert ver_res.is_valid is True
        assert ver_res.status == AuditVerificationState.INTEGRITY_VERIFIED
        assert ver_res.total_records == 4
        assert ver_res.genesis_hash == genesis.current_hash
        assert ver_res.final_hash == e3.current_hash

        print(f"  ✓ Verification status: {ver_res.status.value} (is_valid: {ver_res.is_valid})")
        print(f"  ✓ Replayed 4 canonical records from sequence 0 to 3 without error")
        print(f"  ✓ Final Root Chain Digest: {ver_res.final_hash}")

        # ---------------------------------------------------------------------
        # Check 9: Tamper Detection & Discontinuity Rejection
        # ---------------------------------------------------------------------
        print("\n[Check 9/10] Tamper Detection & Discontinuity Rejection:")
        # Test A: Altered payload
        tampered_records = [AuditRecord.from_dict(asdict(r)) for r in jsonl_records]
        tampered_records[2].payload["object"] = "UNAUTHORIZED_OBJECT"
        res_tamper = verify_audit_chain(tampered_records)
        assert res_tamper.is_valid is False
        assert res_tamper.status == AuditVerificationState.HASH_MISMATCH
        assert res_tamper.error_record_sequence == 2

        # Test B: Deleted record
        missing_records = [tampered_records[0], tampered_records[1], tampered_records[3]]
        res_missing = verify_audit_chain(missing_records)
        assert res_missing.is_valid is False
        assert res_missing.status == AuditVerificationState.MISSING_RECORD

        # Test C: Swapped records
        swapped_records = [tampered_records[0], tampered_records[2], tampered_records[1], tampered_records[3]]
        res_swapped = verify_audit_chain(swapped_records)
        assert res_swapped.is_valid is False
        assert res_swapped.status in (AuditVerificationState.SEQUENCE_ERROR, AuditVerificationState.MISSING_RECORD)

        print(f"  ✓ Payload tampering detection: PASS -> Caught {res_tamper.status.value} at sequence #{res_tamper.error_record_sequence}")
        print(f"  ✓ Record deletion detection:   PASS -> Caught {res_missing.status.value}")
        print(f"  ✓ Record reordering detection:  PASS -> Caught {res_swapped.status.value}")

        # ---------------------------------------------------------------------
        # Check 10: Automated Multi-Format Certification Package
        # ---------------------------------------------------------------------
        print("\n[Check 10/10] Automated Multi-Format Certification Package:")
        reports_dir = work_dir / "sessions"
        reporter = ExperimentReporter(output_root=reports_dir)

        step_results = [
            {
                "step_order": 1,
                "step_id": "STEP_01",
                "step_name": "Grasp Calibration Weight",
                "state": "COMPLETED",
                "is_required": 1,
                "elapsed_s": 14.2,
                "failure_reason": None,
            },
            {
                "step_order": 2,
                "step_id": "STEP_02",
                "step_name": "Position Weight on Balance",
                "state": "COMPLETED",
                "is_required": 1,
                "elapsed_s": 18.7,
                "failure_reason": None,
            },
        ]

        artifacts = reporter.export_all_reports(
            session=stopped_s,
            step_results=step_results,
            violations=mgr_reconnected.get_violations(session.session_id),
            voice_alerts=mgr_reconnected.get_voice_alerts(session.session_id),
            recordings=mgr_reconnected.get_recordings(session.session_id),
            verification_result=ver_res,
            operator_id="CREW_COMMANDER_01",
        )

        for key in ["html", "json", "csv", "manifest"]:
            path = artifacts[key]
            assert path.is_file(), f"Artifact '{key}' missing at {path}"
            size = path.stat().st_size
            assert size > 50, f"Artifact '{key}' file unexpectedly small: {size} bytes"
            print(f"  ✓ Exported {key.upper():8s} -> {path.name} ({size:,} bytes)")

        # Verify Manifest integrity note
        manifest_data = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
        assert "cryptographic_guarantee_notice" in manifest_data
        assert manifest_data["verification"]["is_valid"] is True
        print(f"  ✓ Detached Manifest verified with honest cryptographic guarantee notice")

    except Exception as e:
        print(f"\n✕ VERIFICATION ENCOUNTERED AN EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print("\n" + "=" * 78)
    if all_passed:
        print("  ✓ ALL 10 PROMPT 11 STORAGE, AUDIT & REPORTING CHECKS PASSED")
    else:
        print("  ✕ PROMPT 11 STORAGE & AUDIT VERIFICATION FAILED")
    print("=" * 78)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
