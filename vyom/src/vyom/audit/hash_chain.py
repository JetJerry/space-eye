"""Cryptographic SHA-256 tamper-evident hash chaining and append-only audit logging.

Part of VYOM Prompt 11: SHA-256 Hash-Chain Audit Logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from vyom.audit.canonical import canonical_dumps, sha256_digest
from vyom.core.logging_setup import get_logger
from vyom.storage.database import DatabaseManager

logger = get_logger("audit.hash_chain")

# Documented constant genesis previous-hash: 64 zero hex characters
GENESIS_PREV_HASH = "0" * 64
AUDIT_SCHEMA_VERSION = "1.0.0"


@dataclass
class AuditRecord:
    """Immutable, cryptographically linked event audit record."""
    sequence_num: int
    event_id: str
    session_id: str
    timestamp_utc: str
    timestamp_monotonic: float
    event_type: str
    payload: Dict[str, Any]
    previous_hash: str
    current_hash: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Produce deterministic canonical dictionary for cryptographic hashing."""
        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "sequence_num": self.sequence_num,
            "session_id": self.session_id,
            "timestamp_monotonic": round(self.timestamp_monotonic, 4),
            "timestamp_utc": self.timestamp_utc,
        }

    def to_canonical_bytes(self) -> bytes:
        """Serialize record content into canonical UTF-8 bytes."""
        return canonical_dumps(self.to_canonical_dict())

    def calculate_expected_hash(self) -> str:
        """Compute the expected SHA-256 hash digest from canonical record bytes."""
        return sha256_digest(self.to_canonical_bytes())

    def to_json_line(self) -> str:
        """Format as a single-line JSON string for append-only JSONL files."""
        d = {
            "sequence_num": self.sequence_num,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_monotonic": self.timestamp_monotonic,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
        }
        return json.dumps(d, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditRecord:
        """Deserialize from dictionary."""
        return cls(
            sequence_num=int(data["sequence_num"]),
            event_id=str(data["event_id"]),
            session_id=str(data["session_id"]),
            timestamp_utc=str(data["timestamp_utc"]),
            timestamp_monotonic=float(data.get("timestamp_monotonic", 0.0)),
            event_type=str(data["event_type"]),
            payload=dict(data.get("payload", {})),
            previous_hash=str(data["previous_hash"]),
            current_hash=str(data["current_hash"]),
        )


class AuditChainLogger:
    """Thread-safe append-only SHA-256 hash-chain logger writing to JSONL and SQLite."""

    def __init__(
        self,
        session_id: str,
        audit_dir: Path | str,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.session_id = session_id
        self.audit_dir = Path(audit_dir).resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.audit_dir / f"{session_id}_audit.jsonl"
        self.db = db_manager

        self._lock = threading.Lock()
        self._records: List[AuditRecord] = []
        self._current_sequence_num: int = 0
        self._latest_hash: str = GENESIS_PREV_HASH

    @property
    def latest_hash(self) -> str:
        with self._lock:
            return self._latest_hash

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def records(self) -> List[AuditRecord]:
        with self._lock:
            return list(self._records)

    def initialize_genesis(
        self,
        protocol_id: str,
        protocol_version: str,
        experiment_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Create the immutable genesis record anchoring the cryptographic audit chain."""
        with self._lock:
            if self._records:
                logger.warning("Genesis record already initialized for session '%s'", self.session_id)
                return self._records[0]

            now_utc = datetime.now(timezone.utc).isoformat()
            now_mono = time.monotonic()
            genesis_payload = {
                "genesis_type": "SESSION_GENESIS",
                "session_id": self.session_id,
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "experiment_name": experiment_name,
                "metadata": metadata or {},
            }

            rec = AuditRecord(
                sequence_num=0,
                event_id=f"genesis_{self.session_id}",
                session_id=self.session_id,
                timestamp_utc=now_utc,
                timestamp_monotonic=now_mono,
                event_type="SESSION_GENESIS",
                payload=genesis_payload,
                previous_hash=GENESIS_PREV_HASH,
                current_hash="",
            )
            rec.current_hash = rec.calculate_expected_hash()

            self._records.append(rec)
            self._current_sequence_num = 1
            self._latest_hash = rec.current_hash

            self._persist_record(rec)
            logger.info(
                "Audit chain initialized for session '%s' with genesis hash: %s",
                self.session_id,
                rec.current_hash,
            )
            return rec

    def append_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        event_id: Optional[str] = None,
        timestamp_s: Optional[float] = None,
    ) -> AuditRecord:
        """Append an auditable event record to the SHA-256 chain and persist immediately."""
        with self._lock:
            if not self._records:
                raise RuntimeError(
                    f"Cannot append event to audit chain: Genesis record not initialized for session '{self.session_id}'"
                )

            seq = self._current_sequence_num
            eid = event_id or f"evt_{uuid.uuid4().hex[:10]}"
            now_utc = datetime.now(timezone.utc).isoformat()
            now_mono = time.monotonic() if timestamp_s is None else timestamp_s

            rec = AuditRecord(
                sequence_num=seq,
                event_id=eid,
                session_id=self.session_id,
                timestamp_utc=now_utc,
                timestamp_monotonic=now_mono,
                event_type=event_type,
                payload=payload,
                previous_hash=self._latest_hash,
                current_hash="",
            )
            rec.current_hash = rec.calculate_expected_hash()

            self._records.append(rec)
            self._current_sequence_num += 1
            self._latest_hash = rec.current_hash

            self._persist_record(rec)
            logger.debug(
                "Appended audit event [%d] '%s' -> Hash: %s",
                rec.sequence_num,
                rec.event_type,
                rec.current_hash[:16],
            )
            return rec

    def _persist_record(self, record: AuditRecord) -> None:
        """Write record atomically to append-only JSONL file and SQLite database."""
        # 1. Append to JSONL file
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(record.to_json_line())
        except Exception as e:
            logger.error("Failed to append audit record to JSONL '%s': %s", self.jsonl_path, e)
            raise IOError(f"Audit log write failure: {e}") from e

        # 2. Insert into SQLite table if DatabaseManager provided
        if self.db:
            try:
                with self.db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO audit_chain_records (
                            sequence_num, session_id, event_id, timestamp_utc,
                            timestamp_monotonic, event_type, payload_json,
                            previous_hash, current_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            record.sequence_num, record.session_id, record.event_id,
                            record.timestamp_utc, record.timestamp_monotonic, record.event_type,
                            json.dumps(record.payload), record.previous_hash, record.current_hash
                        ),
                    )
            except Exception as e:
                logger.error("Failed to persist audit record to SQLite: %s", e)
                # JSONL is primary on-disk audit trail; raise to ensure caller detects write failure
                raise

