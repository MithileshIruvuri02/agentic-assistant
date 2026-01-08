from typing import Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


class YouTubeService:
    """Safe YouTube transcript fetcher (never crashes the app)."""

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        reraise=False,  # ❗ NEVER bubble up
    )
    async def get_transcript(self, video_id: str) -> Dict[str, Any]:
        logger.info(f"🎬 Fetching transcript for YouTube video: {video_id}")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # 1️⃣ Try manually created English captions
            for transcript in transcript_list:
                logger.info(
                    f"Trying transcript | lang={transcript.language_code} | generated={transcript.is_generated}"
                )
                if transcript.language_code.startswith("en") and not transcript.is_generated:
                    data = transcript.fetch()
                    return self._success(data)

            # 2️⃣ Try auto-generated English captions
            for transcript in transcript_list:
                logger.info(
                    f"Trying transcript | lang={transcript.language_code} | generated={transcript.is_generated}"
                )
                if transcript.language_code.startswith("en"):
                    data = transcript.fetch()
                    return self._success(data)

            raise NoTranscriptFound(video_id)

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"🚫 Captions unavailable: {e}")
            return self._failure("Captions unavailable")

        except Exception as e:
            # Handles XML parse errors, broken captions, YouTube weirdness
            logger.error(f"❌ Transcript parsing failed: {e}")
            return self._failure("Transcript fetch failed")

    # ---------- helpers ----------

    def _success(self, transcript_data):
        full_text = " ".join(seg["text"] for seg in transcript_data)
        duration = (
            transcript_data[-1]["start"] + transcript_data[-1]["duration"]
            if transcript_data else 0
        )

        return {
            "success": True,
            "text": full_text.strip(),
            "duration": round(duration, 2),
            "language": "en",
            "segment_count": len(transcript_data),
        }

    def _failure(self, reason: str):
        return {
            "success": False,
            "text": "",
            "duration": 0,
            "language": "unknown",
            "segment_count": 0,
            "error": reason,
        }
