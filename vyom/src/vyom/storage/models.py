"""Data structures for session persistence and database records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SessionRecord:
    """Represents a persisted experiment session row."""
    session_id: str
    protocol_id: str
    protocol_version: str
    protocol_name: str
    start_time_utc: str
    end_time_utc: Optional[str] = None
    duration_s: float = 0.0
    status: str = "IDLE"              # "IDLE", "ACTIVE", "PAUSED", "COMPLETED", "ABORTED"
    camera_source: Optional[str] = None
    hardware_device: Optional[str] = None
    hardware_backend: Optional[str] = None
    model_version: Optional[str] = None
    model_checksum: Optional[str] = None
    rig_calibrated: bool = False
    total_steps: int = 0
    completed_steps: int = 0
    violations_count: int = 0
    final_chain_digest: Optional[str] = None
    audit_status: str = "PENDING"     # "PENDING", "INTEGRITY_VERIFIED", "HASH_MISMATCH", etc.
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_name": self.protocol_name,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "duration_s": round(self.duration_s, 2),
            "status": self.status,
            "camera_source": self.camera_source,
            "hardware_device": self.hardware_device,
            "hardware_backend": self.hardware_backend,
            "model_version": self.model_version,
            "model_checksum": self.model_checksum,
            "rig_calibrated": self.rig_calibrated,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "violations_count": self.violations_count,
            "final_chain_digest": self.final_chain_digest,
            "audit_status": self.audit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RecordingRecord:
    """Represents a finalized session video recording and checksum."""
    recording_id: str
    session_id: str
    file_path: str
    file_size_bytes: int = 0
    sha256_checksum: Optional[str] = None
    total_frames: int = 0
    duration_s: float = 0.0
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "session_id": self.session_id,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "sha256_checksum": self.sha256_checksum,
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
        }
