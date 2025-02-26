import aiohttp
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from ai_services_api.services.recommendation.config import get_settings

class OpenAlexService:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.OPENALEX_API_URL or 'https://api.openalex.org'
        self.logger = logging.getLogger(__name__)

    async def _fetch_data(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """
        Helper method to fetch data from OpenAlex API
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
        Returns:
            Optional[Dict]: Response data if successful
        """
        url = f"{self.base_url}/{endpoint}"
        self.logger.debug(f"Fetching data from {url} with params: {params}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        self.logger.error(f"Failed to fetch data. Status: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error in _fetch_data: {e}")
            return None

    async def get_expert_data(self, orcid: str) -> Optional[Dict[str, Any]]:
        """
        Fetch expert data from OpenAlex using ORCID
        Args:
            orcid (str): The ORCID identifier
        Returns:
            Optional[Dict[str, Any]]: Expert data if found
        """
        self.logger.info(f"Fetching expert data for ORCID: {orcid}")
        
        # Format ORCID for OpenAlex API
        if not orcid.startswith('https://orcid.org/'):
            formatted_orcid = f"https://orcid.org/{orcid}"
        else:
            formatted_orcid = orcid

        params = {"filter": f"orcid:{formatted_orcid}"}
        data = await self._fetch_data('authors', params=params)

        if data and 'results' in data:
            results = data['results']
            if results:
                expert_data = results[0]
                self.logger.info(f"Successfully fetched data for {orcid}")
                self.logger.debug(f"Expert data: {expert_data}")
                return expert_data

        self.logger.warning(f"No expert found for ORCID: {orcid}")
        return None

    async def get_expert_domains(self, orcid: str) -> List[Dict[str, str]]:
        """
        Fetch expert's domains, fields, and subfields from their works
        Args:
            orcid (str): Expert's ORCID
        Returns:
            List[Dict[str, str]]: List of domain-field-subfield mappings
        """
        self.logger.info(f"Fetching domains for ORCID: {orcid}")

        # Get expert data first
        expert_data = await self.get_expert_data(orcid)
        if not expert_data:
            self.logger.warning(f"No expert data found for ORCID: {orcid}")
            return []

        # Extract OpenAlex ID
        openalex_id = expert_data['id']
        self.logger.debug(f"Found OpenAlex ID: {openalex_id}")

        # Get the expert's works
        params = {
            'filter': f"author.id:{openalex_id}",
            'per-page': 50
        }
        works_data = await self._fetch_data('works', params=params)

        if not works_data or 'results' not in works_data:
            self.logger.warning(f"No works found for {orcid}")
            return []

        # Process domains, fields, and subfields
        domains_fields_subfields = []
        unique_combinations = set()

        for work in works_data['results']:
            for topic in work.get('topics', []):
                domain = topic.get('domain', {}).get('display_name', 'Unknown Domain')
                field = topic.get('field', {}).get('display_name', 'Unknown Field')
                subfield = topic.get('subfield', {}).get('display_name', 'Unknown Subfield')

                # Create unique combination key
                combo = (domain, field, subfield)
                
                if combo not in unique_combinations:
                    unique_combinations.add(combo)
                    domains_fields_subfields.append({
                        'domain': domain,
                        'field': field,
                        'subfield': subfield
                    })

        self.logger.info(f"Found {len(domains_fields_subfields)} unique topic combinations for {orcid}")
        return domains_fields_subfields

    async def get_expert_works(self, orcid: str) -> Optional[Dict]:
        """
        Fetch all works by an expert
        Args:
            orcid (str): Expert's ORCID
        Returns:
            Optional[Dict]: Works data if successful
        """
        self.logger.info(f"Fetching works for ORCID: {orcid}")

        expert_data = await self.get_expert_data(orcid)
        if not expert_data:
            self.logger.warning(f"No expert data found for ORCID: {orcid}")
            return None

        openalex_id = expert_data['id']
        params = {
            'filter': f"author.id:{openalex_id}",
            'per-page': 100  # Increased to get more works
        }

        works_data = await self._fetch_data('works', params=params)
        if works_data:
            work_count = len(works_data.get('results', []))
            self.logger.info(f"Retrieved {work_count} works for {orcid}")
            return works_data

        self.logger.warning(f"No works found for {orcid}")
        return None

    async def get_expert_detailed_data(self, orcid: str) -> Optional[Dict[str, Any]]:
    """Get detailed expert data with analytics metadata."""
    expert_data = await self.get_expert_data(orcid)
    if not expert_data:
        return None

    try:
        # Get additional metadata
        works_data = await self.get_expert_works(orcid)
        domains_data = await self.get_expert_domains(orcid)
        
        # Calculate metadata metrics
        metadata = {
            'total_works': len(works_data.get('results', [])) if works_data else 0,
            'unique_domains': len(set(d['domain'] for d in domains_data)),
            'unique_fields': len(set(d['field'] for d in domains_data)),
            'expertise_breadth': len(domains_data),
            'data_completeness': self._calculate_completeness(expert_data),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        expert_data['analytics_metadata'] = metadata
        return expert_data
        
    except Exception as e:
        self.logger.error(f"Error getting detailed expert data: {e}")
        return expert_data  # Return basic data if enhanced fetch fails

def _calculate_completeness(self, expert_data: Dict) -> float:
    """Calculate data completeness score."""
    required_fields = ['id', 'display_name', 'works_count', 'cited_by_count']
    optional_fields = ['last_known_institution', 'x_concepts', 'counts_by_year']
    
    score = 0
    total_fields = len(required_fields) + len(optional_fields)
    
    # Check required fields
    for field in required_fields:
        if expert_data.get(field):
            score += 1
            
    # Check optional fields
    for field in optional_fields:
        if expert_data.get(field):
            score += 0.5
            
    return score / total_fields

async def get_expert_domains(self, orcid: str) -> List[Dict[str, str]]:
    """Enhanced domain fetching with confidence scores."""
    domains_fields_subfields = await super().get_expert_domains(orcid)
    
    # Add confidence scores and metadata
    enhanced_domains = []
    for item in domains_fields_subfields:
        # Calculate confidence based on work counts and citations
        confidence = await self._calculate_domain_confidence(
            orcid, 
            item['domain'], 
            item['field']
        )
        
        enhanced_domains.append({
            **item,
            'confidence_score': confidence,
            'metadata': {
                'frequency': 1,  # Will be updated by aggregation
                'last_seen': datetime.utcnow().isoformat()
            }
        })
    
    # Aggregate and calculate frequencies
    domain_frequencies = {}
    for domain in enhanced_domains:
        key = (domain['domain'], domain['field'])
        if key in domain_frequencies:
            domain_frequencies[key]['frequency'] += 1
        else:
            domain_frequencies[key] = domain
            
    return list(domain_frequencies.values())

async def _calculate_domain_confidence(
    self, 
    orcid: str, 
    domain: str, 
    field: str
) -> float:
    """Calculate confidence score for domain attribution."""
    try:
        # Get works in this domain
        works = await self.get_works_by_topic(domain)
        if not works:
            return 0.5  # Default score
            
        relevant_works = [
            w for w in works 
            if any(
                t.get('field', {}).get('display_name') == field 
                for t in w.get('topics', [])
            )
        ]
        
        if not relevant_works:
            return 0.5
            
        # Calculate confidence based on:
        # 1. Number of works in domain
        # 2. Citations of works in domain
        # 3. Recency of works
        work_count_score = min(len(relevant_works) / 10, 1.0)
        
        citations = sum(w.get('cited_by_count', 0) for w in relevant_works)
        citation_score = min(citations / 100, 1.0)
        
        # Calculate recency score
        current_year = datetime.utcnow().year
        years = [int(w.get('publication_year', current_year)) for w in relevant_works]
        avg_year = sum(years) / len(years)
        recency_score = (avg_year - (current_year - 10)) / 10
        
        # Combine scores with weights
        confidence = (
            work_count_score * 0.4 +
            citation_score * 0.4 +
            recency_score * 0.2
        )
        
        return min(max(confidence, 0.1), 1.0)
        
    except Exception as e:
        self.logger.error(f"Error calculating domain confidence: {e}")
        return 0.5  # Default score on error

    async def get_works_by_topic(self, topic: str, limit: int = 50) -> List[Dict]:
        """
        Fetch works related to a specific topic
        Args:
            topic (str): Topic to search for
            limit (int): Maximum number of works to return
        Returns:
            List[Dict]: List of works
        """
        self.logger.info(f"Fetching works for topic: {topic}")

        params = {
            'filter': f"topics.id:{topic}",
            'per-page': min(limit, 100)
        }

        data = await self._fetch_data('works', params=params)
        if data and 'results' in data:
            works = data['results']
            self.logger.info(f"Found {len(works)} works for topic {topic}")
            return works

        self.logger.warning(f"No works found for topic {topic}")
        return []
