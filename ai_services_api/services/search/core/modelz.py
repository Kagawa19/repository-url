# File: ai_services_api/services/search/core/models.py
from sentence_transformers import SentenceTransformer
import os
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = "/app/models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedding_model():
    """Load embedding model with local cache fallback"""
    try:
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        
        # Try local cached model first
        if os.path.exists(model_path):
            logger.info(f"Loading local model from {model_path}")
            return SentenceTransformer(model_path)
            
        # Download and cache if missing
        logger.info(f"Downloading and caching model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        model.save(model_path)
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError("Embedding model unavailable") from e