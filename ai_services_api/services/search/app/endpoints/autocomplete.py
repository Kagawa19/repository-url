from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import httpx
import logging
from dotenv import load_dotenv
import os
import json
from functools import lru_cache

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration class to centralize settings
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    FALLBACK_PROVIDERS = [
        "openai",  # Placeholder for future implementation
        "anthropic"  # Placeholder for future implementation
    ]
    MAX_SUGGESTIONS = 5
    CACHE_EXPIRY = 3600  # 1 hour cache for suggestions

# Custom exceptions
class AutocompleteError(Exception):
    """Base exception for autocomplete errors"""
    pass

class APIProviderError(AutocompleteError):
    """Exception for API provider failures"""
    pass

# Utility function for API key validation
def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate the API key with basic checks.
    
    Args:
        api_key (str): API key to validate
    
    Returns:
        bool: Whether the API key passes basic validation
    """
    return bool(api_key and len(api_key) > 10 and not api_key.isspace())

# Async HTTP client for making requests
async def make_http_request(
    url: str, 
    method: str = 'POST', 
    json_data: Optional[dict] = None, 
    headers: Optional[dict] = None
) -> dict:
    """
    Make an async HTTP request with error handling.
    
    Args:
        url (str): Request URL
        method (str): HTTP method (default: POST)
        json_data (dict, optional): JSON payload
        headers (dict, optional): Request headers
    
    Returns:
        dict: Parsed JSON response
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                method, 
                url, 
                json=json_data, 
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"HTTP Request Error: {e}")
        raise APIProviderError(f"Network error: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {e}")
        raise APIProviderError(f"API returned error: {e.response.status_code}")

# Cached autocomplete suggestions
@lru_cache(maxsize=100)
def _cached_suggestions(query: str) -> List[str]:
    """
    Provide cached default suggestions as a fallback.
    
    Args:
        query (str): Input query
    
    Returns:
        List[str]: Cached suggestions
    """
    # Default fallback suggestions based on query
    default_map = {
        "search": ["search publications", "search papers", "search journals", "search research", "search articles"],
        "find": ["find experts", "find publications", "find research", "find authors", "find papers"],
        "": []
    }
    
    # Find best matching category
    matched_category = next(
        (cat for cat in default_map if cat and query.lower().startswith(cat)), 
        ""
    )
    
    return default_map[matched_category][:Config.MAX_SUGGESTIONS]

async def get_gemini_autocomplete(query: str) -> List[str]:
    """
    Get autocomplete suggestions using Gemini API.
    
    Args:
        query (str): Input query to generate suggestions for
    
    Returns:
        List[str]: List of autocomplete suggestions
    """
    # Validate API key
    if not validate_api_key(Config.GEMINI_API_KEY):
        logger.warning("Invalid Gemini API key")
        return _cached_suggestions(query)
    
    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={Config.GEMINI_API_KEY}"
    
    # Payload for Gemini API
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Generate {Config.MAX_SUGGESTIONS} unique, concise autocomplete suggestions for the search query: '{query}'. Return only suggestions, one per line, without numbering or extra text."
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 100
        }
    }
    
    try:
        # Make request to Gemini API
        response = await make_http_request(url, json_data=payload)
        
        # Extract suggestions
        if (candidates := response.get("candidates")):
            text_response = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Process suggestions
            suggestions = [
                suggestion.strip() 
                for suggestion in text_response.split("\n") 
                if suggestion.strip() and suggestion.strip().lower() != query.lower()
            ]
            
            # Return up to MAX_SUGGESTIONS, fallback to cached if empty
            return suggestions[:Config.MAX_SUGGESTIONS] or _cached_suggestions(query)
        
        # Fallback if no suggestions
        return _cached_suggestions(query)
    
    except APIProviderError:
        # Fallback to cached suggestions on API errors
        return _cached_suggestions(query)

# FastAPI Router
router = APIRouter()

@router.get("/experts/autocomplete", response_model=List[str])
async def autocomplete(
    q: str = Query(..., min_length=1, max_length=100, description="Query to generate autocomplete suggestions")
):
    """
    Generate autocomplete suggestions for a given query.
    
    - **q**: Input query string (required, 1-100 characters)
    - Returns a list of autocomplete suggestions
    """
    logger.info(f"Received autocomplete request for query: {q}")
    
    try:
        # Generate suggestions
        results = await get_gemini_autocomplete(q)
        
        logger.info(f"Generated {len(results)} autocomplete suggestions")
        return results
    
    except Exception as e:
        logger.error(f"Unexpected error in autocomplete: {e}")
        # Return cached suggestions or empty list as final fallback
        return _cached_suggestions(q)

@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        dict: Health status of the service
    """
    return {
        "status": "healthy",
        "service": "autocomplete",
        "dependencies": {
            "gemini_api": validate_api_key(Config.GEMINI_API_KEY)
        }
    }