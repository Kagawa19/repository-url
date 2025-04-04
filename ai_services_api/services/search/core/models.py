# models.py
# Contains all Pydantic models for the search API responses

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ExpertSearchResult(BaseModel):
    """Model representing a single expert search result."""
    id: str
    first_name: str
    last_name: str
    designation: str
    theme: str
    unit: str
    contact: str
    is_active: bool
    score: Optional[float] = None
    bio: Optional[str] = None  
    knowledge_expertise: List[str] = []
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "id": "12345",
                "first_name": "Jane",
                "last_name": "Smith",
                "designation": "Research Scientist",
                "theme": "Neuroscience",
                "unit": "Cognitive Research Unit",
                "contact": "jane.smith@example.com",
                "is_active": True,
                "score": 0.92,
                "bio": "Dr. Jane Smith is a leading researcher in cognitive neuroscience.",
                "knowledge_expertise": ["Neuroscience", "Cognitive Psychology", "MRI Imaging"]
            }
        }

class SearchResponse(BaseModel):
    """Model representing the full search response."""
    total_results: int
    experts: List[ExpertSearchResult]
    user_id: str
    session_id: str
    refinements: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "total_results": 2,
                "experts": [
                    {
                        "id": "12345",
                        "first_name": "Jane",
                        "last_name": "Smith",
                        "designation": "Research Scientist",
                        "theme": "Neuroscience",
                        "unit": "Cognitive Research Unit",
                        "contact": "jane.smith@example.com",
                        "is_active": True,
                        "score": 0.92,
                        "bio": "Dr. Jane Smith is a leading researcher in cognitive neuroscience.",
                        "knowledge_expertise": ["Neuroscience", "Cognitive Psychology", "MRI Imaging"]
                    },
                    {
                        "id": "67890",
                        "first_name": "John",
                        "last_name": "Doe",
                        "designation": "Senior Researcher",
                        "theme": "Neuroscience",
                        "unit": "Brain Imaging Department",
                        "contact": "john.doe@example.com",
                        "is_active": True,
                        "score": 0.85,
                        "bio": "Dr. John Doe specializes in brain imaging techniques.",
                        "knowledge_expertise": ["Brain Imaging", "fMRI", "Neural Networks"]
                    }
                ],
                "user_id": "user123",
                "session_id": "sess456",
                "refinements": {
                    "filters": [
                        {
                            "type": "expertise",
                            "label": "Expertise Area",
                            "values": ["Neuroscience", "Brain Imaging", "Cognitive Psychology"]
                        }
                    ],
                    "related_queries": [
                        "brain research",
                        "cognitive neuroscience",
                        "neural imaging" 
                    ],
                    "expertise_areas": [
                        "Neuroscience",
                        "Cognitive",
                        "Imaging",
                        "Brain",
                        "Psychology"
                    ]
                }
            }
        }


class PredictionResponse(BaseModel):
    predictions: List[str]
    total_suggestions: int
    search_context: Optional[str] = None
    confidence_scores: List[float]
    refinements: List[str] = []
    user_id: str
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "predictions": [
                    "neuroscience",
                    "neurology",
                    "neural networks",
                    "neurological disorders",
                    "neurosurgery"
                ],
                "confidence_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
                "user_id": "user123",
                "refinements": {
                    "filters": [
                        {
                            "type": "query_type",
                            "label": "Query Suggestions",
                            "values": ["neuroscience", "neurology", "neural networks"]
                        }
                    ],
                    "related_queries": [
                        "neuroscience research",
                        "expert in neuroscience",
                        "neuroscience specialist",
                        "studies on neuroscience",
                        "neuroscience analysis"
                    ],
                    "expertise_areas": [
                        "Neuroscience",
                        "Research",
                        "Expert",
                        "Specialist",
                        "Neural"
                    ]
                }
            }
        }

# Additional models for input validation

class CacheInvalidationRequest(BaseModel):
    """Model for cache invalidation requests."""
    user_id: Optional[str] = None
    pattern: Optional[str] = None
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "user_id": "user123",
                "pattern": "neuro"
            }
        }

class SearchHistoryEntry(BaseModel):
    """Model representing a single search history entry."""
    query: str
    timestamp: str
    result_count: int
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "query": "neuroscience",
                "timestamp": "2023-09-15T14:23:45Z",
                "result_count": 5
            }
        }

class UserSearchHistory(BaseModel):
    """Model representing a user's search history."""
    user_id: str
    searches: List[SearchHistoryEntry]
    total_searches: int
    
    class Config:
        """Pydantic configuration for model."""
        schema_extra = {
            "example": {
                "user_id": "user123",
                "searches": [
                    {
                        "query": "neuroscience",
                        "timestamp": "2023-09-15T14:23:45Z",
                        "result_count": 5
                    },
                    {
                        "query": "cognitive psychology",
                        "timestamp": "2023-09-14T10:15:30Z",
                        "result_count": 3
                    }
                ],
                "total_searches": 2
            }
        }