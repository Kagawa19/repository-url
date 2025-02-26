import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
import hashlib
from ai_services_api.services.search.core.cache_manager import CacheManager  # Add this import

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        try:
            logger.info("Initializing embedding model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.cache_manager = cache_manager
            logger.info(f"Model initialized successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"emb_{hashlib.md5(text.encode()).hexdigest()}"

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            # Preprocess text
            text = " ".join(text.split())
            
            # Check cache if available
            if self.cache_manager:
                cache_key = self._get_cache_key(text)
                cached_embedding = self.cache_manager.get(cache_key)
                if cached_embedding is not None:
                    return np.array(cached_embedding)

            # Generate embedding
            embeddings = self.model.encode([text], convert_to_numpy=True)
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.set(
                    cache_key, 
                    embeddings.tolist(),  # Convert to list for JSON serialization
                    expire=3600  # Cache for 1 hour
                )
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise