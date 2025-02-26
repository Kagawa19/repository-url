import PyPDF2
import redis
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import Dict, List, Union
import os
from tqdm import tqdm
from datetime import datetime
from enum import Enum
from typing import Optional


class RedisKeyTypes(Enum):
    """Enum for different types of Redis keys"""
    PDF_TEXT = "pdf:text"           # For storing PDF text content
    PDF_EMBEDDING = "pdf:emb"       # For storing PDF embeddings
    WEBPAGE_TEXT = "web:text"       # For storing webpage text content
    WEBPAGE_EMBEDDING = "web:emb"   # For storing webpage embeddings
    METADATA = "meta"               # For storing metadata


class RedisKeyManager:
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        # Redis connection for text data
        self.redis_text = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Redis connection for binary data (embeddings)
        self.redis_binary = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )

    def generate_key(self, key_type: RedisKeyTypes, identifier: str, chunk_id: Optional[int] = None) -> str:
        """Generate Redis key based on type and identifier"""
        base_key = f"{key_type.value}:{identifier}"
        if chunk_id is not None:
            return f"{base_key}:chunk_{chunk_id}"
        return base_key

    def store_pdf_content(self, pdf_name: str, text: str, metadata: Dict):
        """Store PDF text content and metadata"""
        # Store text content
        text_key = self.generate_key(RedisKeyTypes.PDF_TEXT, pdf_name)
        self.redis_text.set(text_key, text)
        
        # Store metadata
        metadata_key = self.generate_key(RedisKeyTypes.METADATA, f"pdf:{pdf_name}")
        self.redis_text.hset(metadata_key, mapping=metadata)  # Updated to use hset()

    def store_pdf_embedding(self, pdf_name: str, chunk_id: int, embedding: np.ndarray):
        """Store PDF chunk embedding"""
        emb_key = self.generate_key(RedisKeyTypes.PDF_EMBEDDING, pdf_name, chunk_id)
        self.redis_binary.set(emb_key, embedding.tobytes())

    def store_webpage_content(self, url: str, text: str, metadata: Dict):
        """Store webpage text content and metadata"""
        # Store text content
        text_key = self.generate_key(RedisKeyTypes.WEBPAGE_TEXT, url)
        self.redis_text.set(text_key, text)
        
        # Store metadata
        metadata_key = self.generate_key(RedisKeyTypes.METADATA, f"web:{url}")
        self.redis_text.hset(metadata_key, mapping=metadata)  # Updated to use hset()

    def store_webpage_embedding(self, url: str, chunk_id: int, embedding: np.ndarray):
        """Store webpage chunk embedding"""
        emb_key = self.generate_key(RedisKeyTypes.WEBPAGE_EMBEDDING, url, chunk_id)
        self.redis_binary.set(emb_key, embedding.tobytes())


class TextProcessingPipeline:
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        self.key_manager = RedisKeyManager(redis_host, redis_port, redis_db)
        self.model = SentenceTransformer(model_name)

    def process_pdf(self, pdf_path: str):
        """Process PDF file and store in Redis"""
        # Extract PDF text (using previous implementation)
        text = self.extract_pdf_text(pdf_path)
        
        # Create metadata
        pdf_name = os.path.basename(pdf_path)
        metadata = {
            'filename': pdf_name,
            'source_type': 'pdf',
            'processed_date': datetime.now().isoformat()
        }
        
        # Store text and metadata
        self.key_manager.store_pdf_content(pdf_name, text, metadata)
        
        # Process and store embeddings for chunks
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = self.create_embedding(chunk)
            self.key_manager.store_pdf_embedding(pdf_name, i, embedding)
        
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return ""

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for given text."""
        return self.model.encode(text)

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximate size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def store_text_and_embedding(
        self,
        key: str,
        text: str,
        source_type: str,
        metadata: Dict = None
    ):
        """Store text and its embedding in Redis."""
        # Store the original text with metadata
        text_data = {
            'text': text,
            'source_type': source_type,
            'metadata': json.dumps(metadata) if metadata else '{}',  # Serialize metadata to JSON string
        }
        
        # Store text using key_manager's redis_text
        self.key_manager.redis_text.hset(
            f"text:{key}",
            mapping=text_data  # Corrected part
        )
        
        # Create and store embeddings for chunks
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = self.create_embedding(chunk)
            
            # Store embeddings using key_manager's redis_binary
            self.key_manager.redis_binary.hset(
                f"embedding:{key}",
                f"chunk_{i}",
                embedding.tobytes()
            )


    def process_scraped_data(self, data: List[Dict], parent_url=""):
        """Process all scraped data from a list of page data."""
        def process_page(page_data, parent_url=""):
            if not page_data:
                return

            # Process main page content
            url = page_data.get('url', '')
            content = page_data.get('content', '')

            if content:
                self.store_text_and_embedding(
                    key=f"page:{url}",
                    text=content,
                    source_type='webpage',
                    metadata={
                        'url': url,
                        'parent_url': parent_url,
                        'depth': page_data.get('depth', 0)
                    }
                )

            # Process child pages recursively
            children = page_data.get('children', [])
            for child in children:
                process_page(child, url)

        for page_data in data:
            process_page(page_data)

    def process_all_pdfs(self, pdf_folder: str):
        """Process all PDFs in the specified folder."""
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = self.extract_pdf_text(pdf_path)
            if text:
                self.store_text_and_embedding(
                    key=f"pdf:{pdf_file}",
                    text=text,
                    source_type='pdf',
                    metadata={'filename': pdf_file}
                )


def main():
    # Example usage
    pipeline = TextProcessingPipeline()
    
    # Process PDFs
    pdf_folder = 'pdf_files'
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pipeline.process_pdf(pdf_path)
    
    # Process scraped data
    with open('scraped_data.json', 'r') as f:
        data = json.load(f)
        pipeline.process_scraped_data(data)


if __name__ == "__main__":
    main()
