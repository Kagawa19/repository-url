import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union
import logging
from ..config.settings import MODEL_NAME, MAX_TOKENS

class EmbeddingModel:
    def __init__(self):
        self.setup_logging()
        self.setup_model()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def setup_model(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    @torch.no_grad()
    def create_embedding(self, text: str) -> np.ndarray:
        try:
            # Tokenize and truncate
            inputs = self.tokenizer(
                text,
                max_length=MAX_TOKENS,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Use mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            return embedding.cpu().numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            raise

    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        for text in texts:
            try:
                embedding = self.create_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error processing text: {e}")
                continue
        return embeddings

    def __enter__(self):
        return self