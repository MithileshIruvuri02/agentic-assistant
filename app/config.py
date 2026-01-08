from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys - Now using FREE APIs
    GROQ_API_KEY: str  # FREE - Get from https://console.groq.com
    ANTHROPIC_API_KEY: Optional[str] = None  # Optional, can use Groq for everything
    
    ENABLE_CACHE: bool = False
    REDIS_URL: Optional[str] = None
    # Model Configuration - FREE models
    GROQ_MODEL: str = "mixtral-8x7b-32768"  # Fast, free, 32k context
    # Alternative free models:
    # "llama-3.3-70b-versatile" - Most capable
    # "llama-3.1-8b-instant" - Fastest
    # "gemma2-9b-it" - Good balance
    
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"  # Only if you have key
    
    # For audio transcription - we'll use Groq's Whisper (FREE)
    WHISPER_MODEL: str = "whisper-large-v3"
    
    # OCR Configuration
    OCR_ENGINE: str = "tesseract"  # tesseract or easyocr
    OCR_LANGUAGES: list[str] = ["en"]
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    
    # File Upload Limits
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_AUDIO_FORMATS: list[str] = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    ALLOWED_IMAGE_FORMATS: list[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    ALLOWED_PDF_FORMATS: list[str] = [".pdf"]
    
    # Processing Configuration
    MAX_AUDIO_DURATION_MIN: int = 30
    CHUNK_SIZE_TOKENS: int = 4000
    
    # Cost Estimation (Groq is FREE, but we track for analytics)
    GROQ_INPUT_COST: float = 0.0  # FREE!
    GROQ_OUTPUT_COST: float = 0.0  # FREE!
    CLAUDE_INPUT_COST: float = 0.003
    CLAUDE_OUTPUT_COST: float = 0.015
    WHISPER_COST_PER_MIN: float = 0.0  # Groq Whisper is FREE!
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Use Groq for all tasks (since it's free)
    USE_GROQ_FOR_ALL: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Token cost multipliers (Groq is FREE!)
TOKEN_COSTS = {
    "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
    "gemma2-9b-it": {"input": 0.0, "output": 0.0},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}