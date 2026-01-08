import os
import io
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

# ------------------------------------------------------------------
# 🔧 FORCE ffmpeg + ffprobe into PATH (CRITICAL FOR WINDOWS)
# ------------------------------------------------------------------
FFMPEG_BIN = r"C:\ffmpeg-8.0.1-full_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment
from groq import Groq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


class AudioService:
    """Service for audio transcription using Groq Whisper API (FREE)."""

    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)

        logger.info("🎵 AudioService initialized")
        logger.info(f"PATH includes ffmpeg: {FFMPEG_BIN in os.environ['PATH']}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        reraise=True,
    )
    async def transcribe(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        logger.info(f"🎧 Transcribing audio: {filename}")

        file_ext = Path(filename).suffix.lower()

        audio_bytes, duration = await self._prepare_audio(audio_bytes, file_ext)

        max_duration = self.settings.MAX_AUDIO_DURATION_MIN * 60
        if duration > max_duration:
            raise ValueError(
                f"Audio duration {duration:.2f}s exceeds limit {max_duration}s"
            )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.settings.WHISPER_MODEL,
                    language="en",
                )

            return {
                "text": transcription.text.strip(),
                "duration": round(duration, 2),
                "language": "en",
                "confidence": 0.90,
            }

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def _prepare_audio(
        self, audio_bytes: bytes, file_ext: str
    ) -> Tuple[bytes, float]:
        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=file_ext.lstrip("."),
            )

            duration = len(audio) / 1000.0

            output = io.BytesIO()
            audio.export(output, format="mp3", bitrate="128k")

            return output.getvalue(), duration

        except Exception as e:
            logger.exception("❌ Audio preprocessing failed")
            raise RuntimeError("Unsupported or corrupted audio file") from e
