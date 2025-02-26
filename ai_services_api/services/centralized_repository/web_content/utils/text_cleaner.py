# utils/text_cleaner.py

import re
import unicodedata
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import html

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Handles text cleaning and normalization for web content and PDFs.
    """
    
    def __init__(self):
        # Common patterns to clean
        self.patterns = {
            'whitespace': r'\s+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'special_chars': r'[^\w\s.,!?-]',
            'multiple_dots': r'\.{2,}',
            'multiple_spaces': r' {2,}',
            'page_numbers': r'\b(?:page|pg\.?) \d+\b',
            'header_footer': r'^.*(header|footer).*$',
        }
        
        # Initialize compiled regex patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content and extract text.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            str: Cleaned text
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'head', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ')
            
            # Clean the text
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            return html_content

    def clean_pdf_text(self, text: str) -> str:
        """
        Clean text extracted from PDF.
        
        Args:
            text: Raw PDF text
            
        Returns:
            str: Cleaned text
        """
        try:
            # Decode HTML entities
            text = html.unescape(text)
            
            # Remove page numbers
            text = self.compiled_patterns['page_numbers'].sub('', text)
            
            # Remove headers and footers
            text = self.compiled_patterns['header_footer'].sub('', text)
            
            # Apply general text cleaning
            text = self.clean_text(text)
            
            # Remove excessive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning PDF text: {str(e)}")
            return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            str: Cleaned text
        """
        try:
            # Convert to string if not already
            text = str(text)
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Replace HTML entities
            text = html.unescape(text)
            
            # Remove URLs
            text = self.compiled_patterns['urls'].sub('', text)
            
            # Remove email addresses
            text = self.compiled_patterns['emails'].sub('', text)
            
            # Remove special characters
            text = self.compiled_patterns['special_chars'].sub('', text)
            
            # Normalize whitespace
            text = self.compiled_patterns['whitespace'].sub(' ', text)
            
            # Remove multiple periods
            text = self.compiled_patterns['multiple_dots'].sub('.', text)
            
            # Remove multiple spaces
            text = self.compiled_patterns['multiple_spaces'].sub(' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistency.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Normalize unicode
            text = unicodedata.normalize('NFKC', text)
            
            # Replace newlines with spaces
            text = text.replace('\n', ' ')
            
            # Remove multiple spaces
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        try:
            # First clean the text
            text = self.clean_text(text)
            
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Filter out empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            return sentences
            
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            return [text]