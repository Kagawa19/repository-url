# services/embeddings/model_handler.py

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Optional, Dict
import logging
from datetime import datetime
import os
from tqdm import tqdm
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Handles text embedding generation using transformer models.
    Supports batching and model caching.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '512'))
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
        self.cache_dir = os.getenv('MODEL_CACHE_DIR', '.model_cache')
        
        # Set up device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.setup_model()
        logger.info(f"EmbeddingModel initialized on {self.device} using {self.model_name}")

    def setup_model(self):
        """Set up the model and tokenizer"""
        try:
            # Create cache directory if needed
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Truncate if too long (rough character estimate)
        max_chars = self.max_tokens * 4  # Approximate chars per token
        if len(text) > max_chars:
            text = text[:max_chars]
            
        return text

    @torch.no_grad()
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
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.max_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Use mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Create attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Calculate mean
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Get final embedding
            embedding = sum_embeddings / sum_mask
            
            return embedding.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    @torch.no_grad()
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
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Preprocess batch
                batch_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=self.max_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Create attention mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                
                # Calculate mean
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Get batch embeddings
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                embeddings.extend(batch_embeddings)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            raise

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
            'embedding_dim': self.model.config.hidden_size,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'cache_dir': self.cache_dir
        }

    def cleanup(self):
        """Clean up model resources"""
        try:
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()