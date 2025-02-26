import logging
from typing import Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def safe_str(value: Any) -> str:
    """
    Safely convert any value to string.
    
    Args:
        value: Any value to convert to string
        
    Returns:
        str: String representation of the value or "N/A" if None/empty
    """
    try:
        if value is None:
            return "N/A"
        
        if isinstance(value, (list, dict)):
            return str(value)
        
        string_value = str(value).strip()
        return string_value if string_value else "N/A"
        
    except Exception as e:
        logger.error(f"Error converting value to string: {e}")
        return "N/A"

def convert_inverted_index_to_text(inverted_index: Optional[Dict[str, list]]) -> str:
    """
    Convert an inverted index dictionary to readable text.
    
    Args:
        inverted_index: Dictionary mapping words to their positions
        
    Returns:
        str: Reconstructed text or "N/A" if conversion fails
    """
    if not inverted_index:
        logger.debug("Empty inverted index received")
        return "N/A"
    
    try:
        # Create list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            if not isinstance(positions, list):
                logger.warning(f"Invalid positions format for word '{word}': {positions}")
                continue
                
            for pos in positions:
                if not isinstance(pos, int):
                    logger.warning(f"Invalid position value for word '{word}': {pos}")
                    continue
                word_positions.append((pos, word))
        
        if not word_positions:
            logger.warning("No valid word positions found in inverted index")
            return "N/A"
        
        # Sort by position and join words
        sorted_words = [word for _, word in sorted(word_positions)]
        text = ' '.join(sorted_words)
        
        # Clean up any artifacts
        text = clean_text(text)
        
        logger.debug(f"Successfully converted inverted index to text of length {len(text)}")
        return text
        
    except Exception as e:
        logger.error(f"Error converting inverted index: {e}")
        return "N/A"

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    try:
        if not text or text == "N/A":
            return ""
            
        # Basic cleaning operations
        cleaned = text.strip()
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove multiple periods
        cleaned = re.sub(r'\.{2,}', '.', cleaned)
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
        
        # Remove URL artifacts
        cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
        
        # Remove HTML tags if any
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        logger.debug(f"Cleaned text from length {len(text)} to {len(cleaned)}")
        return cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def truncate_text(text: str, max_length: int = 5000) -> str:
    """
    Truncate text to specified maximum length while preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        
    Returns:
        str: Truncated text
    """
    try:
        if not text or len(text) <= max_length:
            return text
            
        truncated = text[:max_length]
        
        # Find last complete sentence
        last_sentence = truncated.rfind('.')
        if last_sentence > max_length * 0.8:  # Only use if reasonably close to end
            truncated = truncated[:last_sentence + 1]
        else:
            # Fall back to last complete word
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            
        logger.debug(f"Truncated text from length {len(text)} to {len(truncated)}")
        return truncated.strip()
        
    except Exception as e:
        logger.error(f"Error truncating text: {e}")
        return text[:max_length] if text else ""

def normalize_field_name(field: str) -> str:
    """
    Normalize field name by removing special characters and standardizing format.
    
    Args:
        field: Field name to normalize
        
    Returns:
        str: Normalized field name
    """
    try:
        if not field:
            return ""
            
        # Convert to lowercase and strip whitespace
        normalized = field.lower().strip()
        
        # Remove special characters except spaces and alphanumeric
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = ' '.join(normalized.split())
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing field name: {e}")
        return field