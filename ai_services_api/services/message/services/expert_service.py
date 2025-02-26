from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Optional, Dict, Any
from ai_services_api.services.message.models.expert import Expert
from ai_services_api.services.message.schemas.expert import ExpertCreate, ExpertUpdate
from fastapi import HTTPException, status

class ExpertService:
    def __init__(self, db: Session):
        self.db = db

    async def get_expert(self, expert_id: int) -> Optional[Expert]:
        """Get expert by ID"""
        return self.db.query(Expert)\
            .filter(Expert.id == expert_id)\
            .filter(Expert.is_active == True)\
            .first()

    async def get_expert_by_email(self, email: str) -> Optional[Expert]:
        """Get expert by email"""
        return self.db.query(Expert)\
            .filter(Expert.email == email)\
            .filter(Expert.is_active == True)\
            .first()

    async def list_experts(
        self,
        skip: int = 0,
        limit: int = 10,
        search: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Expert]:
        """List experts with filtering options"""
        query = self.db.query(Expert).filter(Expert.is_active == True)

        if search:
            search_filter = or_(
                Expert.first_name.ilike(f"%{search}%"),
                Expert.last_name.ilike(f"%{search}%"),
                Expert.email.ilike(f"%{search}%"),
                Expert.designation.ilike(f"%{search}%"),
                Expert.theme.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)

        if filters:
            if filters.get("designation"):
                query = query.filter(Expert.designation == filters["designation"])
            if filters.get("theme"):
                query = query.filter(Expert.theme == filters["theme"])
            if filters.get("unit"):
                query = query.filter(Expert.unit == filters["unit"])
            if filters.get("domains"):
                query = query.filter(Expert.domains.contains(filters["domains"]))
            if filters.get("fields"):
                query = query.filter(Expert.fields.contains(filters["fields"]))

        return query.offset(skip).limit(limit).all()

    async def find_experts_by_expertise(
        self,
        domains: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        theme: Optional[str] = None
    ) -> List[Expert]:
        """Find experts based on their expertise"""
        query = self.db.query(Expert).filter(Expert.is_active == True)

        filters = []
        if domains:
            filters.append(Expert.domains.overlap(domains))
        if fields:
            filters.append(Expert.fields.overlap(fields))
        if theme:
            filters.append(Expert.theme == theme)

        if filters:
            query = query.filter(and_(*filters))

        return query.all()

    async def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about experts"""
        total_experts = self.db.query(Expert)\
            .filter(Expert.is_active == True)\
            .count()
        
        themes = self.db.query(Expert.theme)\
            .filter(Expert.is_active == True)\
            .filter(Expert.theme.isnot(None))\
            .distinct()\
            .all()
        themes = [theme[0] for theme in themes if theme[0]]

        domains = self.db.query(Expert.domains)\
            .filter(Expert.is_active == True)\
            .distinct()\
            .all()
        unique_domains = set()
        for domain_list in domains:
            if domain_list[0]:
                unique_domains.update(domain_list[0])

        return {
            "total_experts": total_experts,
            "themes": themes,
            "domains": list(unique_domains)
        }
