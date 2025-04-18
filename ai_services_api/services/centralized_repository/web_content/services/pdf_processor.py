import hashlib
import logging
import os
import re
from io import BytesIO
from typing import Dict, List

import PyPDF2
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF content extraction with advanced features"""
    
    def __init__(self):
        self.session = requests.Session()
        self.doi_pattern = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.I)
    
    def process_pdfs(self, urls: List[str]) -> List[Dict]:
        """Process PDFs from a list of URLs with deduplication and smart extraction"""
        results = []
        seen_hashes = set()
        
        for url in urls:
            try:
                if not url.lower().endswith('.pdf'):
                    logger.debug(f"Skipping non-PDF URL: {url}")
                    continue

                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                # Content-based deduplication
                content_hash = hashlib.md5(response.content).hexdigest()
                if content_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate PDF: {url}")
                    continue
                seen_hashes.add(content_hash)

                pdf_file = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract metadata and content
                title = self._extract_title(pdf_reader)
                content, summary = self._process_content(pdf_reader)
                doi = self._extract_doi(content) or url

                results.append({
                    'doi': doi,
                    'title': title or os.path.basename(url).replace('.pdf', ''),
                    'summary': summary,
                    'full_content': content,
                    'type': 'pdf',
                    'source': 'website',
                    'page_count': len(pdf_reader.pages),
                    'content_hash': content_hash,
                    'url': url
                })
                logger.info(f"Processed PDF: {url} ({len(pdf_reader.pages)} pages)")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {url}: {str(e)}")
            except PyPDF2.errors.PdfReadError as e:
                logger.error(f"Invalid PDF format {url}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")

        return results

    def _extract_title(self, reader: PyPDF2.PdfReader) -> str:
        """Extract title from metadata or content"""
        try:
            # Try metadata first
            meta_title = reader.metadata.get('/Title', '')
            if meta_title:
                return meta_title.strip()
            
            # Fallback to content analysis
            first_page = reader.pages[0].extract_text()
            lines = [line.strip() for line in first_page.split('\n') if line.strip()]
            return lines[0] if lines else ''
        except Exception as e:
            logger.debug(f"Title extraction failed: {str(e)}")
            return ''

    def _process_content(self, reader: PyPDF2.PdfReader) -> tuple:
        """Smart content extraction with summary optimization"""
        full_text = []
        summary_pages = []
        
        try:
            # Process important pages for summary
            summary_indices = [0, 1, 2, -1]
            for idx in sorted(set(summary_indices)):
                if 0 <= idx < len(reader.pages):
                    text = self._safe_extract_page(reader.pages[idx])
                    summary_pages.append(text)
                    full_text.append(text)

            # Sample remaining content
            for idx in range(3, len(reader.pages) - 1):
                if idx % 5 == 0:  # Sample every 5th page
                    text = self._safe_extract_page(reader.pages[idx])
                    full_text.append(text)

        except Exception as e:
            logger.debug(f"Content processing error: {str(e)}")

        full_content = '\n'.join(full_text)
        summary = '\n'.join(summary_pages)[:1500]  # Truncate to 1500 chars
        return full_content, summary

    def _safe_extract_page(self, page) -> str:
        """Safe text extraction with encoding handling"""
        try:
            text = page.extract_text()
            return text.encode('latin-1', 'replace').decode('utf-8', 'ignore')
        except Exception as e:
            logger.debug(f"Page extraction failed: {str(e)}")
            return ''

    def _extract_doi(self, content: str) -> str:
        """Extract DOI from text content"""
        try:
            match = self.doi_pattern.search(content)
            return match.group(1) if match else ''
        except Exception as e:
            logger.debug(f"DOI extraction failed: {str(e)}")
            return ''

    def cleanup(self):
        """Clean up resources"""
        try:
            self.session.close()
            logger.info("PDFProcessor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")