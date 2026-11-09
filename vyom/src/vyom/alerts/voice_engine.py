"""Local offline speech engine wrapper using pyttsx3 with Windows SAPI5 driver.

Completely offline by design. No cloud TTS services, external network calls,
or telemetry dependencies are permitted.
"""

from __future__ import annotations

import platform
from typing import List, Optional, Tuple

from vyom.alerts.schemas import VoiceInfo
from vyom.core.logging_setup import get_logger

logger = get_logger("alerts.voice_engine")


class OfflineVoiceEngine:
    """Offline speech synthesis driver interface for Windows SAPI5."""

    def __init__(self) -> None:
        self._engine: Optional[object] = None
        self._is_initialized: bool = False
        self._current_voice_id: Optional[str] = None
        self._current_rate: int = 175
        self._current_volume: float = 1.0

    @classmethod
    def get_available_voices(cls) -> List[VoiceInfo]:
        """Inspect and enumerate offline speech synthesis voices installed on the host OS."""
        voices_list: List[VoiceInfo] = []
        try:
            import pyttsx3
            # Use sapi5 driver on Windows, or default fallback
            driver = "sapi5" if platform.system() == "Windows" else None
            engine = pyttsx3.init(driver)
            raw_voices = engine.getProperty("voices") or []

            for v in raw_voices:
                langs = []
                raw_langs = getattr(v, "languages", [])
                if isinstance(raw_langs, (list, tuple)):
                    langs = [str(l) for l in raw_langs]
                elif raw_langs:
                    langs = [str(raw_langs)]

                v_info = VoiceInfo(
                    voice_id=str(v.id),
                    name=str(v.name or v.id),
                    languages=langs,
                    gender=str(getattr(v, "gender", "")) or None,
                )
                voices_list.append(v_info)

            engine.stop()
            del engine
            logger.info("Discovered %d installed offline Windows voices", len(voices_list))
        except Exception as e:
            logger.error("Failed to query offline speech voices: %s", e)

        return voices_list

    def initialize(
        self,
        voice_id: Optional[str] = None,
        rate: int = 175,
        volume: float = 1.0,
    ) -> Tuple[bool, Optional[str]]:
        """Initialize the local speech synthesis engine on the current thread."""
        try:
            import pyttsx3
            driver = "sapi5" if platform.system() == "Windows" else None
            self._engine = pyttsx3.init(driver)

            # 1. Verify available voices
            voices = self._engine.getProperty("voices")
            if not voices:
                self._is_initialized = False
                err_msg = "No offline Windows SAPI5 voices found installed on system."
                logger.error(err_msg)
                return False, err_msg

            # 2. Select voice
            selected_voice = None
            if voice_id:
                for v in voices:
                    if v.id == voice_id or v.name == voice_id:
                        selected_voice = v.id
                        break

            if not selected_voice and voices:
                selected_voice = voices[0].id

            if selected_voice:
                self._engine.setProperty("voice", selected_voice)
                self._current_voice_id = selected_voice

            # 3. Configure speech rate & volume
            self._current_rate = int(rate)
            self._current_volume = float(volume)
            self._engine.setProperty("rate", self._current_rate)
            self._engine.setProperty("volume", self._current_volume)

            self._is_initialized = True
            logger.info(
                "OfflineVoiceEngine initialized successfully (Voice: %s, Rate: %d WPM, Vol: %.2f)",
                self._current_voice_id,
                self._current_rate,
                self._current_volume,
            )
            return True, None
        except Exception as e:
            self._is_initialized = False
            self._engine = None
            err_msg = f"Offline speech engine initialization error: {e}"
            logger.error(err_msg)
            return False, err_msg

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized and self._engine is not None

    @property
    def current_voice_id(self) -> Optional[str]:
        return self._current_voice_id

    def set_voice(self, voice_id: str) -> bool:
        """Update active voice ID."""
        if not self.is_initialized or not self._engine:
            return False
        try:
            self._engine.setProperty("voice", voice_id)
            self._current_voice_id = voice_id
            return True
        except Exception as e:
            logger.warning("Failed to set speech voice '%s': %s", voice_id, e)
            return False

    def set_rate(self, rate: int) -> bool:
        """Update speech rate in words per minute."""
        if not self.is_initialized or not self._engine:
            return False
        try:
            self._current_rate = int(rate)
            self._engine.setProperty("rate", self._current_rate)
            return True
        except Exception as e:
            logger.warning("Failed to set speech rate to %d: %s", rate, e)
            return False

    def set_volume(self, volume: float) -> bool:
        """Update speech volume between 0.0 and 1.0."""
        if not self.is_initialized or not self._engine:
            return False
        try:
            self._current_volume = max(0.0, min(1.0, float(volume)))
            self._engine.setProperty("volume", self._current_volume)
            return True
        except Exception as e:
            logger.warning("Failed to set speech volume to %.2f: %s", volume, e)
            return False

    def speak(self, text: str) -> Tuple[bool, Optional[str]]:
        """Speak a text string synchronously on the caller's thread."""
        if not self.is_initialized or not self._engine:
            return False, "Speech engine is not initialized or unavailable."

        clean_text = text.strip()
        if not clean_text:
            return True, None

        try:
            self._engine.say(clean_text)
            self._engine.runAndWait()
            return True, None
        except Exception as e:
            err_msg = f"Speech playback failed: {e}"
            logger.error(err_msg)
            return False, err_msg

    def stop(self) -> None:
        """Immediately stop currently playing utterance."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception as e:
                logger.debug("Exception while stopping speech engine: %s", e)

    def shutdown(self) -> None:
        """Cleanly tear down the speech engine instance."""
        if self._engine:
            try:
                self.stop()
            except Exception:
                pass
            self._engine = None
        self._is_initialized = False
        logger.info("OfflineVoiceEngine shut down cleanly.")
