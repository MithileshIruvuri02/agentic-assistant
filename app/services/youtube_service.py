from typing import Dict, Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


class YouTubeService:
    """Service for fetching YouTube video transcripts."""

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,  # IMPORTANT: let RetryError bubble up
    )
    async def get_transcript(self, video_id: str) -> Dict[str, Any]:
        logger.info(f"Fetching transcript for YouTube video: {video_id}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            full_text = " ".join(entry["text"] for entry in transcript_list)

            duration = (
                transcript_list[-1]["start"] + transcript_list[-1]["duration"]
                if transcript_list
                else 0
            )

            return {
                "text": full_text.strip(),
                "duration": round(duration, 2),
                "language": "en",
                "segment_count": len(transcript_list),
            }

        except TranscriptsDisabled:
            logger.error("YouTube transcripts disabled")
            raise RuntimeError("Transcripts disabled")

        except NoTranscriptFound:
            logger.error("No transcript found")
            raise RuntimeError("No transcript found")

        except Exception as e:
            logger.error(f"YouTube transcript fetch failed: {e}")
            raise RuntimeError(str(e))
