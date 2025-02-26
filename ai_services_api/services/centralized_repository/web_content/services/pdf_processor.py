# services/web_content/pdf_processor.py

import os
import logging
import requests
import PyPDF2
from typing import Dict, List, Optional, Tuple
import io
from urllib.parse import urlparse
import hashlib
from datetime import datetime
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import re
from ai_services_api.services.centralized_repository.web_content.config.settings import PDF_CHUNK_SIZE

from ai_services_api.services.centralized_repository.web_content.utils.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF downloading, text extraction, and content chunking.
    Supports both local and remote PDFs.
    """
    
    def __init__(self, chunk_size: int = PDF_CHUNK_SIZE):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Maximum size of text chunks
        """
        self.chunk_size = chunk_size
        self.setup_storage()
        self.text_cleaner = TextCleaner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Track processed PDFs
        self.processed_files: Dict[str, str] = {}  # url -> file_path
        logger.info("PDFProcessor initialized successfully")

    def setup_storage(self):
        """Set up storage directory for PDFs"""
        try:
            self.pdf_dir = os.getenv('PDF_FOLDER', 'data/pdf_files')
            os.makedirs(self.pdf_dir, exist_ok=True)
            logger.info(f"PDF storage directory set up at: {self.pdf_dir}")
        except Exception as e:
            logger.error(f"Failed to set up PDF storage: {str(e)}")
            raise

    def download_pdf(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Download PDF from URL.
        
        Args:
            url: URL of the PDF
            timeout: Request timeout in seconds
            
        Returns:
            Optional[str]: Path to downloaded file
        """
        try:
            # Check if already downloaded
            if url in self.processed_files:
                if os.path.exists(self.processed_files[url]):
                    logger.info(f"Using cached PDF for: {url}")
                    return self.processed_files[url]
            
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=timeout, stream=True)
            
            if response.status_code == 200:
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type:
                    logger.warning(f"URL does not point to a PDF: {url}")
                    return None
                
                # Generate filename
                filename = os.path.join(
                    self.pdf_dir,
                    f"{hashlib.md5(url.encode()).hexdigest()}.pdf"
                )
                
                # Save file
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                self.processed_files[url] = filename
                logger.info(f"PDF downloaded to: {filename}")
                return filename
                
            logger.error(f"Failed to download PDF. Status code: {response.status_code}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {str(e)}")
            return None

    def extract_text_with_pymupdf(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            doc = fitz.open(file_path)
            text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text.append(page.get_text())
                
            doc.close()
            return "\n".join(text)
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
            return None

    def extract_text_with_pdfplumber(self, file_path: str) -> Optional[str]:
        """
        Fallback text extraction using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text() or '')
                return "\n".join(text)
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {str(e)}")
            return None

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF using multiple methods.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            # Try PyMuPDF first
            text = self.extract_text_with_pymupdf(file_path)
            
            # Fallback to pdfplumber if needed
            if not text or not text.strip():
                text = self.extract_text_with_pdfplumber(file_path)
                
            if text:
                # Clean the extracted text
                cleaned_text = self.text_cleaner.clean_pdf_text(text)
                return cleaned_text
                
            logger.warning(f"No text extracted from: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into manageable chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            
            for paragraph in paragraphs:
                # Split long paragraphs if needed
                words = paragraph.split()
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    
                    if current_length + word_length > self.chunk_size:
                        if current_chunk:  # Only append if there's content
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = word_length
                    else:
                        current_chunk.append(word)
                        current_length += word_length
                
                # Add paragraph break
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add any remaining content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return [chunk for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []

    def get_pdf_metadata(self, file_path: str) -> Dict:
        """
        Extract PDF metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dict: PDF metadata
        """
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            
            # Add additional metadata
            metadata.update({
                'page_count': len(doc),
                'file_size': os.path.getsize(file_path),
                'extracted_at': datetime.now().isoformat()
            })
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {}

    def process_pdf(self, pdf_url: str) -> Optional[Dict]:
        """
        Process a single PDF document.
        
        Args:
            pdf_url: URL of the PDF
            
        Returns:
            Optional[Dict]: Processed PDF data
        """
        try:
            logger.info(f"Processing PDF: {pdf_url}")
            
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
            if not chunks:
                return None
            
            # Get metadata
            metadata = self.get_pdf_metadata(pdf_path)
            
            return {
                'url': pdf_url,
                'file_path': pdf_path,
                'chunks': chunks,
                'num_chunks': len(chunks),
                'total_length': len(text),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'hash': hashlib.sha256(text.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_url}: {str(e)}")
            return None

    def process_pdfs(self, pdf_urls: List[str]) -> List[Dict]:
        """
        Process multiple PDF documents.
        
        Args:
            pdf_urls: List of PDF URLs
            
        Returns:
            List[Dict]: List of processed PDF data
        """
        results = []
        for url in pdf_urls:
            try:
                result = self.process_pdf(url)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing PDF {url}: {str(e)}")
                continue
        return results

    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Remove temporary directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            # Optionally remove downloaded PDFs
            if os.getenv('CLEANUP_PDFS', 'false').lower() == 'true':
                for file_path in self.processed_files.values():
                    try:
                        os.remove(file_path)
                    except:
                        pass
                        
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()