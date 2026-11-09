"""Independent cryptographic audit chain verifier and manifest generator.

Part of VYOM Prompt 11: SHA-256 Hash-Chain Audit Logging and Verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from vyom.audit.canonical import canonical_dumps
from vyom.audit.hash_chain import GENESIS_PREV_HASH, AuditRecord
from vyom.core.logging_setup import get_logger

logger = get_logger("audit.verifier")


class AuditVerificationState(str, Enum):
    """Possible outcomes of cryptographic audit verification."""
    INTEGRITY_VERIFIED = "INTEGRITY_VERIFIED"
    HASH_MISMATCH = "HASH_MISMATCH"
    MISSING_RECORD = "MISSING_RECORD"
    SEQUENCE_ERROR = "SEQUENCE_ERROR"
    INVALID_GENESIS = "INVALID_GENESIS"
    INCOMPLETE_AUDIT = "INCOMPLETE_AUDIT"


@dataclass
class AuditVerificationResult:
    """Detailed cryptographic audit verification report."""
    status: AuditVerificationState
    is_valid: bool
    total_records: int
    genesis_hash: str = ""
    final_hash: str = ""
    first_timestamp_utc: Optional[str] = None
    last_timestamp_utc: Optional[str] = None
    error_message: Optional[str] = None
    error_record_sequence: Optional[int] = None
    verification_timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "genesis_hash": self.genesis_hash,
            "final_hash": self.final_hash,
            "first_timestamp_utc": self.first_timestamp_utc,
            "last_timestamp_utc": self.last_timestamp_utc,
            "error_message": self.error_message,
            "error_record_sequence": self.error_record_sequence,
            "verification_timestamp_utc": self.verification_timestamp_utc,
        }


def load_audit_records_from_jsonl(file_path: Path | str) -> List[AuditRecord]:
    """Parse an append-only JSONL audit file into a sequence of AuditRecord objects."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audit log file not found: {path}")

    records: List[AuditRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                records.append(AuditRecord.from_dict(data))
            except Exception as e:
                logger.error("Failed to parse audit JSONL line %d in '%s': %s", line_num, path.name, e)
                raise ValueError(f"Corrupt JSONL line {line_num} in {path.name}: {e}") from e
    return records


def verify_audit_chain(
    records: Sequence[Union[AuditRecord, Dict[str, Any]]],
) -> AuditVerificationResult:
    """Verify mathematical integrity of an SHA-256 append-only hash chain.

    Replays every record from sequence 0 to end:
    1. Checks genesis record format and initial previous hash.
    2. Confirms strictly sequential sequence numbers (0, 1, 2, ...).
    3. Confirms previous_hash matches preceding record's current_hash.
    4. Recomputes SHA-256 over canonical record bytes and compares against current_hash.

    Args:
        records: List of AuditRecord instances or raw dictionaries.

    Returns:
        AuditVerificationResult summarizing pass/fail state, hashes, and failure diagnostics.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    if not records:
        return AuditVerificationResult(
            status=AuditVerificationState.INCOMPLETE_AUDIT,
            is_valid=False,
            total_records=0,
            error_message="Audit chain is empty: no records found to verify.",
            verification_timestamp_utc=now_utc,
        )

    # Convert dicts if necessary
    parsed_records: List[AuditRecord] = []
    for item in records:
        if isinstance(item, AuditRecord):
            parsed_records.append(item)
        elif isinstance(item, dict):
            # Parse payload_json if coming from SQLite
            d = dict(item)
            if "payload_json" in d and "payload" not in d:
                try:
                    d["payload"] = json.loads(d["payload_json"])
                except Exception:
                    d["payload"] = {}
            parsed_records.append(AuditRecord.from_dict(d))
        else:
            return AuditVerificationResult(
                status=AuditVerificationState.SEQUENCE_ERROR,
                is_valid=False,
                total_records=len(records),
                error_message=f"Unsupported audit record type: {type(item).__name__}",
                verification_timestamp_utc=now_utc,
            )

    total = len(parsed_records)
    first_ts = parsed_records[0].timestamp_utc
    last_ts = parsed_records[-1].timestamp_utc

    # Step 1: Genesis verification
    genesis = parsed_records[0]
    if genesis.sequence_num != 0:
        return AuditVerificationResult(
            status=AuditVerificationState.INVALID_GENESIS,
            is_valid=False,
            total_records=total,
            error_record_sequence=genesis.sequence_num,
            error_message=(
                f"Genesis record must have sequence_num 0, found {genesis.sequence_num}."
            ),
            verification_timestamp_utc=now_utc,
        )

    if genesis.previous_hash != GENESIS_PREV_HASH:
        return AuditVerificationResult(
            status=AuditVerificationState.INVALID_GENESIS,
            is_valid=False,
            total_records=total,
            error_record_sequence=0,
            error_message=(
                f"Genesis previous_hash must be {GENESIS_PREV_HASH}, found '{genesis.previous_hash}'."
            ),
            verification_timestamp_utc=now_utc,
        )

    expected_genesis_hash = genesis.calculate_expected_hash()
    if genesis.current_hash != expected_genesis_hash:
        return AuditVerificationResult(
            status=AuditVerificationState.HASH_MISMATCH,
            is_valid=False,
            total_records=total,
            genesis_hash=genesis.current_hash,
            error_record_sequence=0,
            error_message=(
                f"Genesis record hash mismatch. Recorded: '{genesis.current_hash}', "
                f"Calculated: '{expected_genesis_hash}'."
            ),
            verification_timestamp_utc=now_utc,
        )

    genesis_hash = genesis.current_hash
    previous_hash = genesis_hash

    # Step 2..N: Sequential chain verification
    for i in range(1, total):
        rec = parsed_records[i]

        # Check sequence continuity
        if rec.sequence_num != i:
            status = (
                AuditVerificationState.MISSING_RECORD
                if rec.sequence_num > i
                else AuditVerificationState.SEQUENCE_ERROR
            )
            return AuditVerificationResult(
                status=status,
                is_valid=False,
                total_records=total,
                genesis_hash=genesis_hash,
                error_record_sequence=rec.sequence_num,
                error_message=(
                    f"Sequence discontinuity at index {i}: expected sequence {i}, "
                    f"found {rec.sequence_num}."
                ),
                verification_timestamp_utc=now_utc,
            )

        # Check previous_hash link
        if rec.previous_hash != previous_hash:
            return AuditVerificationResult(
                status=AuditVerificationState.HASH_MISMATCH,
                is_valid=False,
                total_records=total,
                genesis_hash=genesis_hash,
                error_record_sequence=rec.sequence_num,
                error_message=(
                    f"Chain link broken at record #{rec.sequence_num}: "
                    f"recorded previous_hash '{rec.previous_hash}' does not match "
                    f"prior record's current_hash '{previous_hash}'."
                ),
                verification_timestamp_utc=now_utc,
            )

        # Check current_hash computation from canonical bytes
        expected_hash = rec.calculate_expected_hash()
        if rec.current_hash != expected_hash:
            return AuditVerificationResult(
                status=AuditVerificationState.HASH_MISMATCH,
                is_valid=False,
                total_records=total,
                genesis_hash=genesis_hash,
                error_record_sequence=rec.sequence_num,
                error_message=(
                    f"Payload tampering or hash mismatch at record #{rec.sequence_num}: "
                    f"recorded hash '{rec.current_hash}' != calculated hash '{expected_hash}'."
                ),
                verification_timestamp_utc=now_utc,
            )

        previous_hash = rec.current_hash

    final_hash = parsed_records[-1].current_hash
    logger.info(
        "Audit chain verified successfully: %d records, genesis=%s, final=%s",
        total,
        genesis_hash[:16],
        final_hash[:16],
    )

    return AuditVerificationResult(
        status=AuditVerificationState.INTEGRITY_VERIFIED,
        is_valid=True,
        total_records=total,
        genesis_hash=genesis_hash,
        final_hash=final_hash,
        first_timestamp_utc=first_ts,
        last_timestamp_utc=last_ts,
        error_message=None,
        error_record_sequence=None,
        verification_timestamp_utc=now_utc,
    )


def export_audit_manifest(
    session_id: str,
    protocol_id: str,
    protocol_version: str,
    verification_result: AuditVerificationResult,
    output_path: Path | str,
    operator_id: Optional[str] = None,
    recording_checksums: Optional[List[Dict[str, Any]]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate and write a detached cryptographic audit manifest JSON document.

    This document anchors the session's final hash digest so that any retroactive
    database alterations or re-generation can be detected by comparing the final digest.
    """
    manifest_path = Path(output_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "manifest_schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "protocol": {
            "protocol_id": protocol_id,
            "protocol_version": protocol_version,
        },
        "operator_id": operator_id or "ANONYMOUS_OPERATOR",
        "verification": verification_result.to_dict(),
        "recordings": recording_checksums or [],
        "metadata": additional_metadata or {},
        "cryptographic_guarantee_notice": (
            "This manifest anchors the cryptographic SHA-256 hash chain for the designated "
            "VYOM experiment session. Hash chaining guarantees mathematical tamper-evidence "
            "against unauthorized modifications, deletions, or reordering of internal event "
            "records. Verifiable integrity relies on comparing the final_hash against this "
            "independently exported or archived manifest digest."
        ),
    }

    canonical_manifest_bytes = canonical_dumps(manifest_data)
    manifest_data["manifest_digest_sha256"] = AuditRecord(
        sequence_num=0,
        event_id="manifest",
        session_id=session_id,
        timestamp_utc="",
        timestamp_monotonic=0.0,
        event_type="MANIFEST",
        payload=manifest_data,
        previous_hash="",
        current_hash="",
    ).calculate_expected_hash()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info("Exported audit manifest for session '%s' to '%s'", session_id, manifest_path)
    return manifest_data

