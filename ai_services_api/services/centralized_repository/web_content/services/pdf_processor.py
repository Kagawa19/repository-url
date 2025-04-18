import logging
from typing import Dict, List
import requests
import PyPDF2
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
                if not url.lower().endswith('.pdf'):
                    logger.debug(f"Skipping non-PDF URL: {url}")
                    continue
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                pdf_file = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                content = ""
                for page_num in range(len(pdf_reader.pages)):
                    text = pdf_reader.pages[page_num].extract_text()
                    content += text + "\n"
                
                results.append({
                    'doi': url,
                    'title': 'PDF Document',
                    'summary': content[:1000],
                    'type': 'pdf',
                    'source': 'website'
                })
                logger.info(f"Processed PDF: {url}")
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