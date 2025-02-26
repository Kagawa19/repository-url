from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "AI-Enhanced Search"
    VERSION: str = "1.0.0"
    APP_NAME: str = "Expert Recommendation System"

    # Database settings
    POSTGRES_DB: str = "aphrcdb"
    POSTGRES_USER: str = "aphrcuser"
    POSTGRES_PASSWORD: str = "kimu"
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres:5432/{POSTGRES_DB}"
    DSPACE_DB_URL: str = "postgresql://aphrcuser:kimu@postgres:5432/dspace"

    # PGAdmin settings
    PGADMIN_EMAIL: str = "briankimu97@gmail.com"
    PGADMIN_PASSWORD: str = "kimu"

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: str = "6379"
    REDIS_DB: str = "1"
    REDIS_URL: str = "redis://redis:6379"
    REDIS_GRAPH_URL: str = "redis://redis-graph:6380"
    REDIS_GRAPH_NAME: str = "reco_graph"
    REDIS_CONNECT_TIMEOUT: str = "5"
    REDIS_SOCKET_TIMEOUT: str = "5"
    REDIS_HEALTH_CHECK_INTERVAL: str = "30"

    # API settings
    OPENALEX_API_URL: str = "https://api.openalex.org"
    WEBSITE_API_URL: str = "https://aphrc.org/api"
    DSPACE_API_URL: str = "http://knowhub.aphrc.org/rest"
    ORCID_API_URL: str = "https://api.orcid.org/v3.0"
    
    # API keys
    GEMINI_API_KEY: str = "AIzaSyAh8cGbwmKLK5k7aXUqMnlEiqp2a4L5Ur0"
    ORCID_API_KEY: str = "your_api_key"

    # Model settings
    MODEL_PATH: str = "all-MiniLM-L6-v2"
    
    # File paths
    PDF_FOLDER: str = "ai_services_api/services/search/pdf"
    INDEX_PATH: str = "ai_services_api/services/search/models/faiss_index.idx"  
    CHUNK_MAPPING_PATH: str = "ai_services_api/services/search/models/chunk_mapping.pkl"  
    DATA_PATH: str = "data/test.csv"
    EXPERTS_DB_PATH: str = "data/experts.csv"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    

@lru_cache()
def get_settings():
    return Settings()