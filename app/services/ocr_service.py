import io
import re
import os
from typing import Dict, Any

from PIL import Image
import pytesseract
import easyocr
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()
if hasattr(settings, "TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    logger.info(f"✅ Tesseract forced path: {settings.TESSERACT_CMD}")


class OCRService:
    """OCR text extraction service."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = self.settings.OCR_ENGINE.lower()
        self.easyocr_reader = None

        if self.engine == "easyocr":
            self._init_easyocr()

    def _init_easyocr(self):
        try:
            self.easyocr_reader = easyocr.Reader(
                self.settings.OCR_LANGUAGES,
                gpu=False
            )
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR failed, fallback to Tesseract: {e}")
            self.engine = "tesseract"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def extract_text(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            logger.info(f"OCR using engine: {self.engine}")

            if self.engine == "easyocr" and self.easyocr_reader:
                return await self._extract_easyocr(image)

            return await self._extract_tesseract(image)

        except Exception as e:
            logger.exception("OCR extraction failed")
            raise RuntimeError("OCR extraction failed") from e

    async def _extract_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        try:
            config = "--oem 3 --psm 6"

            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=config
            )

            text = pytesseract.image_to_string(image, config=config)

            confidences = [
                conf for conf in data.get("conf", [])
                if isinstance(conf, int) and conf >= 0
            ]

            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

            cleaned = self._clean_text(text)

            return {
                "text": cleaned,
                "confidence": round(avg_conf, 2),
                "engine": "tesseract"
            }

        except Exception as e:
            logger.exception("Tesseract OCR failed")
            raise RuntimeError("Tesseract OCR failed") from e

    async def _extract_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        image_np = np.array(image)
        results = self.easyocr_reader.readtext(image_np)

        texts, confs = [], []

        for _, text, conf in results:
            texts.append(text)
            confs.append(conf)

        cleaned = self._clean_text(" ".join(texts))
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return {
            "text": cleaned,
            "confidence": round(avg_conf, 2),
            "engine": "easyocr"
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"[^\w\s\-.,!?;:()\[\]{}\"'@#$%&*+=/<>]",
            "",
            text
        )
        return text.strip()
