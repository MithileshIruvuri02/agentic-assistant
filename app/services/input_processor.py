import re
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import magic
from loguru import logger
from tenacity import RetryError

from app.models.schemas import InputType, ExtractedContent
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.youtube_service import YouTubeService
from app.config import get_settings


class InputProcessor:
    """Processes different types of input and extracts content."""

    def __init__(self):
        self.settings = get_settings()
        self.ocr_service = OCRService()
        self.pdf_service = PDFService()
        self.audio_service = AudioService()
        self.youtube_service = YouTubeService()

        self.youtube_patterns = [
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^\s&]+)",
            r"(?:https?:\/\/)?youtu\.be\/([^\s&]+)",
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

    def _is_youtube_url(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.youtube_patterns)

    async def _process_youtube(self, text: str) -> ExtractedContent:
        logger.info("Processing YouTube URL")

        video_id = None
        for pattern in self.youtube_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                video_id = match.group(1)
                break

        if not video_id:
            return ExtractedContent(
                text=text,
                input_type=InputType.TEXT,
                extraction_method="youtube_failed",
                metadata={"error": "Invalid YouTube URL"},
            )

        try:
            transcript = await self.youtube_service.get_transcript(video_id)

            return ExtractedContent(
                text=transcript["text"],
                input_type=InputType.TEXT,
                extraction_method="youtube_transcript",
                metadata={
                    "video_id": video_id,
                    "duration_seconds": transcript.get("duration"),
                    "language": transcript.get("language", "unknown"),
                },
            )

        except (RetryError, Exception) as e:
            logger.warning(f"YouTube transcript unavailable: {e}")

            return ExtractedContent(
                text=text,
                input_type=InputType.TEXT,
                extraction_method="youtube_failed",
                metadata={
                    "video_id": video_id,
                    "error": "Transcript unavailable",
                },
            )

    async def _process_file(self, file: UploadFile) -> ExtractedContent:
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        mime_type = magic.from_buffer(content, mime=True)

        if mime_type.startswith("image/"):
            return await self._process_image(content, file.filename)

        if mime_type == "application/pdf":
            return await self._process_pdf(content, file.filename)

        if mime_type.startswith("audio/"):
            return await self._process_audio(content, file.filename)

        raise ValueError(f"Unsupported file type: {mime_type}")

    async def _process_image(self, content: bytes, filename: str) -> ExtractedContent:
        result = await self.ocr_service.extract_text(content)

        return ExtractedContent(
            text=result["text"],
            input_type=InputType.IMAGE,
            confidence=result.get("confidence"),
            extraction_method="ocr",
            metadata={"filename": filename},
        )

    async def _process_pdf(self, content: bytes, filename: str) -> ExtractedContent:
        result = await self.pdf_service.extract_text(content)

        return ExtractedContent(
            text=result["text"],
            input_type=InputType.PDF,
            extraction_method=result["method"],
            metadata={"filename": filename},
        )

    async def _process_audio(self, content: bytes, filename: str) -> ExtractedContent:
        result = await self.audio_service.transcribe(content, filename)

        return ExtractedContent(
            text=result["text"],
            input_type=InputType.AUDIO,
            extraction_method="whisper",
            metadata={"filename": filename},
        )
