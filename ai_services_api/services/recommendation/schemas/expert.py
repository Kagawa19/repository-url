from pydantic import BaseModel, Field
from typing import List, Optional

class SimilarExpert(BaseModel):
    """Model for similar expert response"""
    orcid: str = Field(..., description="Expert's ORCID identifier")
    name: str = Field(..., description="Expert's full name")
    shared_domain_count: int = Field(0, description="Number of shared domains")
    shared_domains: List[str] = Field(default_factory=list, description="List of shared domains")
    similarity_score: float = Field(0.0, description="Similarity score based on domain overlap")

class DomainInfo(BaseModel):
    """Model for domain information"""
    domain: str = Field(..., description="Domain name")
    field: str = Field(..., description="Field name")
    subfield: Optional[str] = Field(None, description="Subfield name")

class ExpertBase(BaseModel):
    """Base model for expert data"""
    orcid: str = Field(..., description="Expert's ORCID identifier")

class ExpertCreate(ExpertBase):
    """Model for creating a new expert"""
    pass

class ExpertResponse(BaseModel):
    """Model for expert response data"""
    orcid: str = Field(..., description="Expert's ORCID identifier")
    name: str = Field(..., description="Expert's full name")
    domains_fields_subfields: List[DomainInfo] = Field(
        default_factory=list,
        description="Expert's domains, fields, and subfields"
    )
    similar_experts: List[SimilarExpert] = Field(
        default_factory=list,
        description="List of similar experts"
    )