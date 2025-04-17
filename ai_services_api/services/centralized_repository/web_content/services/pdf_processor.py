"""
PDF processor for extracting content from PDFs.

WARNING: Do not import ContentPipeline or WebContentProcessor in this module to avoid circular imports.
"""
import logging
from typing import Dict, List
import requests
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def process_pdfs(self, urls: List[str]) -> List[Dict]:
        """Process PDFs from a list of URLs"""
        results = []
        for url in urls:
            try:
                if url.endswith('.pdf'):
                    response = self.session.get(url)
                    response.raise_for_status()
                    pdf_file = BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text (simplified example)
                    chunks = []
                    chunk_ids = []
                    for page_num in range(len(pdf_reader.pages)):
                        text = pdf_reader.pages[page_num].extract_text()
                        chunks.append(text)
                        chunk_ids.append(page_num + 1)  # Simplified ID
                    
                    results.append({
                        'url': url,
                        'chunks': chunks,
                        'chunk_ids': chunk_ids,
                        'content_type': 'pdf'
                    })
            except Exception as e:
                logger.error(f"Error processing PDF {url}: {str(e)}")
        return results
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.session.close()
            logger.info("PDFProcessor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during PDFProcessor cleanup: {str(e)}")