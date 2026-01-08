import io
from typing import Dict, Any
import PyPDF2
import pdfplumber
from pdf2image import convert_from_bytes
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.services.ocr_service import OCRService


class PDFService:
    """Service for extracting text from PDF files."""
    
    def __init__(self):
        self.ocr_service = OCRService()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_text(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text from PDF. Try text extraction first, fallback to OCR if needed.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info("Starting PDF text extraction")
        
        # Try pdfplumber first (best for text PDFs)
        try:
            result = await self._extract_with_pdfplumber(pdf_bytes)
            if result["text"].strip() and len(result["text"]) > 50:
                logger.info("Successfully extracted text with pdfplumber")
                return result
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Try PyPDF2 as fallback
        try:
            result = await self._extract_with_pypdf2(pdf_bytes)
            if result["text"].strip() and len(result["text"]) > 50:
                logger.info("Successfully extracted text with PyPDF2")
                return result
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Fallback to OCR for scanned PDFs
        logger.info("Text extraction failed, falling back to OCR")
        return await self._extract_with_ocr(pdf_bytes)
    
    async def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using pdfplumber."""
        text_parts = []
        page_count = 0
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        
        return {
            "text": full_text.strip(),
            "pages": page_count,
            "method": "pdfplumber",
            "confidence": 0.95  # High confidence for text-based extraction
        }
    
    async def _extract_with_pypdf2(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using PyPDF2."""
        text_parts = []
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(pdf_reader.pages)
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        
        return {
            "text": full_text.strip(),
            "pages": page_count,
            "method": "pypdf2",
            "confidence": 0.90
        }
    
    async def _extract_with_ocr(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Convert PDF to images and extract text using OCR."""
        logger.info("Converting PDF to images for OCR")
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=300)
        page_count = len(images)
        
        text_parts = []
        confidences = []
        
        for i, image in enumerate(images):
            logger.info(f"OCR processing page {i+1}/{page_count}")
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Extract text from image
            ocr_result = await self.ocr_service.extract_text(img_bytes)
            
            if ocr_result["text"].strip():
                text_parts.append(ocr_result["text"])
                confidences.append(ocr_result["confidence"])
        
        full_text = "\n\n".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "text": full_text.strip(),
            "pages": page_count,
            "method": "ocr_fallback",
            "confidence": round(avg_confidence, 2)
        }