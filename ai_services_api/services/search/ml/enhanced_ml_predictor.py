import logging
from typing import List, Dict, Optional, Tuple, Any
from redis import Redis
from datetime import datetime
import json
import os
import httpx
import asyncio

# Import your original MLPredictor
from ai_services_api.services.search.ml.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)

class DynamicRefinementGenerator:
    """Handles generating dynamic refinements using Gemini API"""
    
    def __init__(self, redis_client: Redis = None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour cache for refinements
        
        logger.info("Initialized DynamicRefinementGenerator with Gemini API integration")
    
    async def get_refinements(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate dynamic refinements based on query using Gemini API
        
        Args:
            query: The search query
            context: Optional context like user history and previous results
            
        Returns:
            Dictionary of refinement suggestions
        """
        try:
            # Check cache first for performance
            cache_key = f"gemini:refinements:{query}"
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for refinements: {query}")
                    return json.loads(cached)
            
            # Construct prompt for Gemini
            prompt = self._build_refinement_prompt(query, context)
            
            # Call Gemini API
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.api_url}?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.2,
                            "topP": 0.8,
                            "topK": 40
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Gemini API error: {response.status_code} {response.text}")
                    return self._get_fallback_refinements(query)
                
                result = response.json()
                
                # Extract and parse the model's response
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_content = result['candidates'][0]['content']['parts'][0]['text']
                    refinements = self._parse_gemini_response(text_content)
                    
                    # Cache the result
                    if self.redis_client:
                        self.redis_client.setex(
                            cache_key, 
                            self.cache_ttl, 
                            json.dumps(refinements)
                        )
                    
                    return refinements
                
                return self._get_fallback_refinements(query)
                
        except Exception as e:
            logger.error(f"Error generating refinements with Gemini: {str(e)}")
            return self._get_fallback_refinements(query)
    
    def _build_refinement_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """Build a prompt for Gemini to generate refinements"""
        context_str = ""
        if context:
            if 'user_history' in context and context['user_history']:
                context_str += f"User's recent searches: {', '.join(context['user_history'][:5])}\n"
            if 'predictions' in context and context['predictions']:
                context_str += f"Related search predictions: {', '.join(context['predictions'])}\n"
                
        prompt = f"""
        You are an expert system that helps refine expert search queries in a professional organization.
        
        Current partial query: "{query}"
        
        {context_str}
        
        Generate search refinement suggestions in the following JSON format:
        {{
            "expertise_areas": [list of 5-7 relevant expertise areas that would refine this search],
            "related_queries": [list of 5-7 alternative ways to phrase this search query],
            "filters": [
                {{
                    "type": "filter_type",
                    "label": "Human-readable label",
                    "values": [possible values for this filter]
                }}
            ],
            "broader_terms": [2-3 broader/more general terms related to this query],
            "narrower_terms": [2-3 more specific terms related to this query],
            "related_terms": [2-3 related but different terms]
        }}
        
        Ensure all suggestions are directly relevant to finding experts based on the query.
        Respond ONLY with the JSON object and nothing else.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from Gemini"""
        try:
            # Extract JSON from the response (in case there's surrounding text)
            import re
            json_match = re.search(r'({.*})', response_text.replace('\n', ''), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no valid JSON found, try to parse the whole response
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return self._get_fallback_refinements("")
    
    def _get_fallback_refinements(self, query: str) -> Dict[str, Any]:
        """Provide basic fallback refinements when API fails"""
        # This is a minimal fallback, not hardcoded domain-specific content
        return {
            "expertise_areas": [],
            "related_queries": [],
            "filters": [
                {
                    "type": "department",
                    "label": "Department",
                    "values": []
                },
                {
                    "type": "active_status",
                    "label": "Availability",
                    "values": ["Active Experts", "All Experts"]
                }
            ],
            "broader_terms": [],
            "narrower_terms": [],
            "related_terms": []
        }


class EnhancedMLPredictor(MLPredictor):
    """Enhanced predictor with real-time refinement capabilities"""
    
    def __init__(self):
        super().__init__()
        self.refinement_generator = DynamicRefinementGenerator(self.redis_client)
        
    async def predict_with_refinements(self, partial_query: str, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get search suggestions with dynamic refinements
        
        Args:
            partial_query: The partial search query
            user_id: User identifier for personalization
            limit: Maximum number of suggestions to return
            
        Returns:
            Dictionary with predictions and refinements
        """
        try:
            # Check cache first
            cache_key = f"enhanced_prediction:{user_id}:{partial_query}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Get base predictions
            base_result = self.predict(partial_query, user_id, limit)
            
            if isinstance(base_result, list):
                predictions = base_result
                confidence_scores = [1.0 - (i * 0.1) for i in range(len(predictions))]
            else:
                predictions = base_result.get("predictions", [])
                confidence_scores = base_result.get("confidence_scores", [])
            
            # If query is too short, return basic predictions without refinements
            if len(partial_query) < 3:
                result = {
                    "predictions": predictions,
                    "confidence_scores": confidence_scores,
                    "refinements": {},
                    "user_id": user_id
                }
                return result
            
            # Build context for refinement generation
            context = {
                "user_history": self._get_user_search_history(user_id),
                "predictions": predictions
            }
            
            # Get dynamic refinements
            refinements = await self.refinement_generator.get_refinements(partial_query, context)
            
            # Combine results
            result = {
                "predictions": predictions,
                "confidence_scores": confidence_scores,
                "refinements": refinements,
                "user_id": user_id
            }
            
            # Cache the result (short TTL for real-time experience)
            self.redis_client.setex(cache_key, 60, json.dumps(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_with_refinements: {str(e)}")
            return {
                "predictions": [],
                "confidence_scores": [],
                "refinements": {},
                "user_id": user_id
            }
    
    def _get_user_search_history(self, user_id: str) -> List[str]:
        """Get recent search history for a user"""
        try:
            user_key = f"suggestions:user:{user_id}"
            recent_searches = self.redis_client.zrevrange(user_key, 0, 5)
            return recent_searches
        except Exception as e:
            logger.error(f"Error getting user history: {str(e)}")
            return []