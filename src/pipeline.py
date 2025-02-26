# src/pipeline.py
from typing import List, Dict
import hashlib
from datetime import datetime
from .scrapers.web_scraper import WebsiteScraper
from .scrapers.pdf_processor import PDFProcessor
from .embeddings.model_handler import EmbeddingModel
from .utils.redis_handler import RedisHandler

class ContentPipeline:
    def __init__(self):
        self.web_scraper = WebsiteScraper()
        self.pdf_processor = PDFProcessor()
        self.embedding_model = EmbeddingModel()
        self.redis_handler = RedisHandler()

    def process_webpage(self, page_data: Dict):
        """Process a single webpage"""
        # Generate key
        key = hashlib.md5(page_data['url'].encode()).hexdigest()
        
        # Create embedding
        embedding = self.embedding_model.create_embedding(page_data['content'])
        
        # Store in Redis
        self.redis_handler.store_text(
            key=key,
            text=page_data['content'],
            metadata={
                'url': page_data['url'],
                'title': page_data['title'],
                'type': 'webpage',
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self.redis_handler.store_embedding(
            key=key,
            embedding=embedding,
            metadata={
                'url': page_data['url'],
                'type': 'webpage',
                'timestamp': datetime.now().isoformat()
            }
        )

    def process_pdf(self, pdf_data: Dict):
        """Process a single PDF"""
        for i, chunk in enumerate(pdf_data['chunks']):
            # Generate key
            key = f"{hashlib.md5(pdf_data['url'].encode()).hexdigest()}_chunk_{i}"
            
            # Create embedding
            embedding = self.embedding_model.create_embedding(chunk)
            
            # Store in Redis
            self.redis_handler.store_text(
                key=key,
                text=chunk,
                metadata={
                    'url': pdf_data['url'],
                    'chunk_index': i,
                    'total_chunks': len(pdf_data['chunks']),
                    'type': 'pdf',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.redis_handler.store_embedding(
                key=key,
                embedding=embedding,
                metadata={
                    'url': pdf_data['url'],
                    'chunk_index': i,
                    'total_chunks': len(pdf_data['chunks']),
                    'type': 'pdf',
                    'timestamp': datetime.now().isoformat()
                }
            )

    def run(self):
        """Run the complete pipeline"""
        try:
            # 1. Scrape website
            pages_data = self.web_scraper.scrape_site()
            
            # 2. Process webpages
            for page in pages_data:
                self.process_webpage(page)
                
            # 3. Process PDFs
            pdf_results = self.pdf_processor.process_pdfs(self.web_scraper.pdf_links)
            for pdf in pdf_results:
                self.process_pdf(pdf)
                
        finally:
            self.web_scraper.close()