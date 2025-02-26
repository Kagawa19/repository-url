# src/scrapers/pdf_processor.py
import os
import logging
import requests
import PyPDF2
from typing import Dict, List, Optional
import io
from urllib.parse import urlparse
import hashlib
from datetime import datetime
from ..config.settings import PDF_FOLDER, PDF_CHUNK_SIZE
from ..utils.text_cleaner import TextCleaner

class PDFProcessor:
    def __init__(self):
        self.setup_logging()
        os.makedirs(PDF_FOLDER, exist_ok=True)

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def download_pdf(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = os.path.join(
                    PDF_FOLDER, 
                    hashlib.md5(url.encode()).hexdigest() + '.pdf'
                )
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {e}")
        return None

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    cleaned_text = TextCleaner.clean_pdf_text(page_text)
                    text.append(cleaned_text)
                return '\n'.join(text)
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > PDF_CHUNK_SIZE:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_pdf(self, pdf_url: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Processing PDF: {pdf_url}")
            
            # Download PDF
            pdf_path = self.download_pdf(pdf_url)
            if not pdf_path:
                return None
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return None
            
            # Create chunks
            chunks = self.chunk_text(text)
            
            return {
                'url': pdf_url,
                'file_path': pdf_path,
                'chunks': chunks,
                'num_chunks': len(chunks),
                'total_length': len(text),
                'timestamp': datetime.now().isoformat(),
                'hash': hashlib.sha256(text.encode()).hexdigest()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_url}: {e}")
            return None

    def process_pdfs(self, pdf_urls: List[str]) -> List[Dict]:
        results = []
        for url in pdf_urls:
            result = self.process_pdf(url)
            if result:
                results.append(result)
        return results