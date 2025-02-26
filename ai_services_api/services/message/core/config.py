# ai_services_api/services/message/config.py
from pydantic import BaseSettings, validator
from functools import lru_cache
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Required settings
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY')
    DATABASE_URL: str = "postgresql://postgres:p0stgres@postgres:5432/aphrc"
    
    # Application settings
    APP_NAME: Optional[str] = "Expert Recommendation System"
    DEBUG: Optional[bool] = True
    API_V1_STR: Optional[str] = "/api/v1"
    
    # Database settings
    POSTGRES_HOST: Optional[str] = "postgres"
    POSTGRES_DB: Optional[str] = "aphrc"
    POSTGRES_USER: Optional[str] = "postgres"
    POSTGRES_PASSWORD: Optional[str] = "p0stgres"

    class Config:
        env_file = ".env"
        case_sensitive = True
        use_enum_values = True
        allow_extra = True  # This replaces `extra = "allow"` in Pydantic v1

    # In Pydantic v1, we use validator instead of property
    @validator('GEMINI_API_KEY')
    def validate_gemini_key(cls, v: Optional[str]) -> str:
        if not v:
            logger.error("GEMINI_API_KEY not found")
            raise ValueError("GEMINI_API_KEY is not configured")
        return v

@lru_cache()
def get_settings() -> Settings:
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        raise
