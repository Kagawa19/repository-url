import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import os
import json

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Handles text embedding generation using SentenceTransformer models.
    Supports batching and model caching, aligned with ExpertRedisIndexManager.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name or path of the SentenceTransformer model to use
        """
        self.model_name = model_name or os.getenv('MODEL_NAME', '/app/models/sentence-transformers/all-MiniLM-L6-v2')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '512'))
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '50'))  # Align with WebContentProcessor
        self.cache_dir = os.getenv('MODEL_CACHE_DIR', '.model_cache')
        
        # Device is always CPU, matching ExpertRedisIndexManager
        self.device = 'cpu'
        
        self.setup_model()
        logger.info(f"EmbeddingModel initialized on {self.device} using {self.model_name}")

    def setup_model(self):
        """Set up the SentenceTransformer model."""
        try:
            # Create cache directory if needed
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Initialize model
            logger.info(f"Loading SentenceTransformer model from: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                local_files_only=True,  # Match ExpertRedisIndexManager
                cache_folder=self.cache_dir
            )
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_name}: {e}")
            logger.warning("Falling back to None; manual embedding will be used.")
            self.model = None

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing, matching ExpertRedisIndexManager
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Truncate if too long (rough character estimate)
        max_chars = self.max_tokens * 4  # Approximate chars per token
        if len(text) > max_chars:
            text = text[:max_chars]
            
        return text

    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        logger.info("Creating fallback embedding")
        # Create a deterministic embedding based on character values, matching ExpertRedisIndexManager
        embedding = np.zeros(384)  # Standard dimension for all-MiniLM-L6-v2
        for i, char in enumerate(text):
            embedding[i % len(embedding)] += ord(char) / 1000
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            # Preprocess text
            text = self.preprocess_text(text)
            
            if self.model is None:
                logger.warning("No model available, using fallback embedding")
                return self._create_fallback_embedding(text)
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return self._create_fallback_embedding(text)

    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        embeddings = []
        
        try:
            if self.model is None:
                logger.warning("No model available, using fallback embeddings")
                return [self._create_fallback_embedding(text) for text in texts]
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Preprocess batch
                batch_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    device=self.device
                )
                
                embeddings.extend(batch_embeddings)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            return [self._create_fallback_embedding(text) for text in texts]

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Calculate cosine similarity
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns:
            Dict: Model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'max_tokens': self.max_tokens,
            'batch_size': self.batch_size,
            'embedding_dim': 384,  # Fixed for all-MiniLM-L6-v2
            'cache_dir': self.cache_dir
        }

    def cleanup(self):
        """Clean up model resources."""
        try:
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()