import re
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import magic
from loguru import logger

from app.models.schemas import InputType, ExtractedContent
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.youtube_service import YouTubeService
from app.config import get_settings


class InputProcessor:
    """Processes different input types and extracts usable content."""

    def __init__(self):
        self.settings = get_settings()
        self.ocr_service = OCRService()
        self.pdf_service = PDFService()
        self.audio_service = AudioService()
        self.youtube_service = YouTubeService()

        self.youtube_patterns = [
            r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^\s&]+)'
        ]

    async def process(
        self,
        text: Optional[str] = None,
        file: Optional[UploadFile] = None,
    ) -> ExtractedContent:
        logger.info(f"Processing input | text={bool(text)} file={bool(file)}")

        if text and self._is_youtube_url(text):
            return await self._process_youtube(text)

        if file:
            return await self._process_file(file)

        if text:
            return ExtractedContent(
                text=text.strip(),
                input_type=InputType.TEXT,
                extraction_method="direct",
                metadata={"length": len(text)},
            )

        raise ValueError("No valid input provided")

    # ---------- YouTube ----------

    def _is_youtube_url(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.youtube_patterns)

    async def _process_youtube(self, text: str) -> ExtractedContent:
        logger.info("🎬 Processing YouTube URL")

        video_id = None
        for pattern in self.youtube_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                video_id = match.group(1).split("&")[0]
                break

        if not video_id:
            raise ValueError("Could not extract YouTube video ID")

        transcript = await self.youtube_service.get_transcript(video_id)

        # ✅ SUCCESS
        if transcript.get("success"):
            return ExtractedContent(
                text=transcript["text"],
                input_type=InputType.TEXT,
                extraction_method="youtube_transcript",
                confidence=0.9,
                metadata={
                    "video_id": video_id,
                    "duration_seconds": transcript["duration"],
                    "language": transcript["language"],
                },
            )

        # ✅ GRACEFUL FALLBACK (NO CRASH)
        return ExtractedContent(
            text=(
                "🤔 I couldn’t access captions for this YouTube video. "
                "Please upload the transcript, enable captions, or upload the audio/video file."
            ),
            input_type=InputType.TEXT,
            extraction_method="youtube_failed",
            confidence=0.0,
            metadata={
                "video_id": video_id,
                "error": transcript.get("error"),
            },
        )

    # ---------- File handling ----------

    async def _process_file(self, file: UploadFile) -> ExtractedContent:
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        mime_type = magic.from_buffer(content, mime=True)

        logger.info(f"Processing file | {file.filename} | {mime_type}")

        if mime_type.startswith("image/"):
            result = await self.ocr_service.extract_text(content)
            return ExtractedContent(
                text=result["text"],
                input_type=InputType.IMAGE,
                confidence=result["confidence"],
                extraction_method="ocr",
            )

        if mime_type == "application/pdf":
            result = await self.pdf_service.extract_text(content)
            return ExtractedContent(
                text=result["text"],
                input_type=InputType.PDF,
                confidence=result.get("confidence"),
                extraction_method=result["method"],
            )

        if mime_type.startswith("audio/"):
            result = await self.audio_service.transcribe(content, file.filename)
            return ExtractedContent(
                text=result["text"],
                input_type=InputType.AUDIO,
                confidence=result.get("confidence"),
                extraction_method="whisper_api",
            )

        raise ValueError(f"Unsupported file type: {mime_type}")
