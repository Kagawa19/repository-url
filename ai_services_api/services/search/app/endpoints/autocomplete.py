from fastapi import APIRouter, HTTPException, Query, Depends
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

# Check and log if API key is missing
if not Config.GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set. Autocomplete will use fallback suggestions.")

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
        logger.error(f"HTTP Status Error: {e.response.status_code}: {e.response.text}")
        raise APIProviderError(f"API returned error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error during HTTP request: {str(e)}")
        raise APIProviderError(f"Unexpected error: {str(e)}")

# Define default suggestions function
def get_default_suggestions(query: str) -> List[str]:
    """
    Provide default suggestions as a fallback.
    
    Args:
        query (str): Input query
    
    Returns:
        List[str]: Default suggestions
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
    
    return default_map.get(matched_category, [])[:Config.MAX_SUGGESTIONS]

# Cache wrapper for suggestions
@lru_cache(maxsize=100)
def cached_suggestions(query: str) -> List[str]:
    """Cache layer for default suggestions"""
    return get_default_suggestions(query)

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
        logger.warning("Invalid or missing Gemini API key")
        return cached_suggestions(query)
    
    # Gemini API endpoint - using updated model name
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-001:generateContent?key={Config.GEMINI_API_KEY}"
    
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
        
        # Extract suggestions with better error handling
        candidates = response.get("candidates", [])
        if not candidates:
            logger.warning("No candidates in Gemini API response")
            return cached_suggestions(query)
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            logger.warning("No parts in Gemini API response content")
            return cached_suggestions(query)
            
        text_response = parts[0].get("text", "")
        
        if not text_response:
            logger.warning("Empty text in Gemini API response")
            return cached_suggestions(query)
        
        # Process suggestions
        suggestions = [
            suggestion.strip() 
            for suggestion in text_response.split("\n") 
            if suggestion.strip() and suggestion.strip().lower() != query.lower()
        ]
        
        # Return up to MAX_SUGGESTIONS, fallback to cached if empty
        if not suggestions:
            logger.info("No valid suggestions from Gemini API, using fallback")
            return cached_suggestions(query)
            
        return suggestions[:Config.MAX_SUGGESTIONS]
    
    except APIProviderError as e:
        # Log the specific error
        logger.error(f"Gemini API error: {str(e)}")
        return cached_suggestions(query)
    except Exception as e:
        # Catch all unexpected errors
        logger.error(f"Unexpected error in get_gemini_autocomplete: {str(e)}")
        return cached_suggestions(query)

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
        logger.error(f"Unexpected error in autocomplete endpoint: {str(e)}")
        # Return cached suggestions as final fallback
        return cached_suggestions(q)

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