from pydantic_settings import BaseSettings
from functools import lru_cache
import logging

class Settings(BaseSettings):
    """Configuration settings for the application"""
    # Application Settings
    APP_NAME: str = "Expert Recommendation System"
    DEBUG: bool = False
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    REDIS_URL: str = "redis://redis:6379"
    
    # Neo4j Settings
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123456789!"
    
    # PostgreSQL Settings
    POSTGRES_HOST: str = "167.86.85.127"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str | None = None
    
    # PGAdmin Settings
    PGADMIN_EMAIL: str
    PGADMIN_PASSWORD: str
    
    # API Settings
    OPENALEX_API_URL: str = "https://api.openalex.org"
    GEMINI_API_KEY: str

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_database_url(self) -> str:
        """Generate database URL if not provided"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    
    # Configure logging based on settings
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return settings