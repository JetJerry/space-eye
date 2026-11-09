"""Configuration schema for offline voice notifications and SAPI5 speech parameters."""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


from vyom.core.schemas import VoiceAlertConfig

__all__ = ["VoiceAlertConfig"]
