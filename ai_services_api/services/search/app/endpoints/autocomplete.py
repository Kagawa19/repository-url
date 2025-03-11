from fastapi import APIRouter, HTTPException, Query
from typing import List
import requests
import logging
from dotenv import load_dotenv
import os

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API key from .env file
load_dotenv()

# Validate API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ ERROR: Missing GEMINI_API_KEY in .env file")

# Gemini API Endpoint
URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"

router = APIRouter()

async def get_autocomplete(query: str) -> List[str]:
    """
    Get autocomplete suggestions for a given query.
    
    Args:
        query (str): Input query to generate suggestions for
    
    Returns:
        List[str]: List of autocomplete suggestions
    """
    try:
        # Updated prompt to clearly request a list format
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Generate 5 autocomplete suggestions for the query: '{query}'. Return only the suggestions, one per line, with no numbering or additional text."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 100
            }
        }
        
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"API Response: {data}")
        
        # Extract suggestions from the correct path in Gemini's response structure
        suggestions = []
        if (
            "candidates" in data and 
            len(data["candidates"]) > 0 and 
            "content" in data["candidates"][0] and
            "parts" in data["candidates"][0]["content"] and
            len(data["candidates"][0]["content"]["parts"]) > 0 and
            "text" in data["candidates"][0]["content"]["parts"][0]
        ):
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            # Split by newline and clean up each suggestion
            suggestions = [
                suggestion.strip()
                for suggestion in text_response.split("\n")
                if suggestion.strip() and suggestion.strip().lower() != query.lower()
            ]
        
        # Ensure we have at most 5 suggestions
        return suggestions[:5]
    
    except requests.RequestException as e:
        logger.error(f"Error fetching autocomplete suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching autocomplete suggestions: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/experts/autocomplete", response_model=List[str])
async def autocomplete(
    q: str = Query(..., min_length=1, description="Query to generate autocomplete suggestions")
):
    """
    Generate autocomplete suggestions for a given query.
    
    - **q**: Input query string (required)
    - Returns a list of autocomplete suggestions
    """
    logger.info(f"Received autocomplete request for query: {q}")
    results = await get_autocomplete(q)
    logger.info(f"Generated {len(results)} autocomplete suggestions")
    return results

@router.get("/test/experts/autocomplete", response_model=List[str])
async def test_autocomplete(
    q: str = Query(..., min_length=1, description="Query to generate autocomplete suggestions")
):
    """
    Test endpoint for generating autocomplete suggestions.
    
    - **q**: Input query string (required)
    - Returns a list of autocomplete suggestions
    """
    logger.info(f"Received test autocomplete request for query: {q}")
    results = await get_autocomplete(q)
    logger.info(f"Generated {len(results)} test autocomplete suggestions")
    return results