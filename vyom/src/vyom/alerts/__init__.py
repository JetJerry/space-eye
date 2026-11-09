"""Offline local voice alerts and experiment guidance subsystem for VYOM.

Components:
- schemas.py: Data models for voice items, categories, priorities, and engine statuses.
- config.py: Pydantic configuration for speech rates, volumes, cooldowns, and queue size.
- voice_engine.py: Windows SAPI5 / pyttsx3 offline driver wrapper.
- voice_worker.py: Dedicated background QThread managing priority queueing and audio synthesis.
- alert_manager.py: High-level procedure engine integration, flood control, and templating.
"""

from vyom.alerts.alert_manager import AlertManager
from vyom.alerts.config import VoiceAlertConfig
from vyom.alerts.schemas import (
    AlertCategory,
    VoiceAlertItem,
    VoiceAlertPriority,
    VoiceEngineStatus,
    VoiceInfo,
)
from vyom.alerts.voice_engine import OfflineVoiceEngine
from vyom.alerts.voice_worker import VoiceWorker

__all__ = [
    "AlertManager",
    "VoiceAlertConfig",
    "VoiceWorker",
    "OfflineVoiceEngine",
    "AlertCategory",
    "VoiceAlertPriority",
    "VoiceEngineStatus",
    "VoiceInfo",
    "VoiceAlertItem",
]
