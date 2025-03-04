import os
import logging
import asyncio
import aiohttp
import requests
from typing import List, Dict, Optional
import json
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OrcidProcessor:
    def __init__(self, db: DatabaseManager = None, summarizer: TextSummarizer = None):
        """
        Initialize ORCID Processor with database and summarizer.
        
        Args:
            db (DatabaseManager, optional): Database manager instance
            summarizer (TextSummarizer, optional): Summarizer instance
        """
        self.db = db or DatabaseManager()
        self.summarizer = summarizer or TextSummarizer()
        self.base_url = "https://pub.orcid.org/v3.0"
        
        # Get ORCID API credentials
        self.client_id = os.getenv('ORCID_CLIENT_ID')
        self.client_secret = os.getenv('ORCID_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("ORCID API credentials not found")
        
        # Get access token
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """
        Retrieve access token for ORCID API.
        
        Returns:
            str: Access token for API requests
        """
        token_url = "https://orcid.org/oauth/token"
        response = requests.post(
            token_url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "/read-public"
            },
            headers={"Accept": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get ORCID access token: {response.text}")
        
        return response.json()["access_token"]

    def _get_experts_with_orcid(self) -> List[Dict]:
        """
        Retrieve experts with ORCID identifiers from the database.
        
        Returns:
            List[Dict]: List of experts with ORCID
        """
        try:
            experts = self.db.execute("""
                SELECT id, first_name, last_name, orcid
                FROM experts_expert
                WHERE orcid IS NOT NULL 
                  AND orcid != '' 
                  AND orcid != 'Unknown'
                  AND first_name != 'Unknown' 
                  AND last_name != 'Unknown'
            """)
            
            return [
                {
                    'id': expert[0],
                    'first_name': expert[1],
                    'last_name': expert[2],
                    'orcid': expert[3]
                } for expert in experts
            ]
        except Exception as e:
            logger.error(f"Error retrieving experts with ORCID: {e}")
            return []

    async def process_publications(self, pub_processor: PublicationProcessor, source: str = 'orcid') -> None:
        # Get experts with ORCID
        experts = self._get_experts_with_orcid()
        
        if not experts:
            logger.info("No experts with ORCID found")
            return
        
        logger.info(f"Processing publications for {len(experts)} experts")
        
        publication_count = 0
        max_publications = 1000
        
        async with aiohttp.ClientSession() as session:
            for expert in experts:
                try:
                    if publication_count >= max_publications:
                        logger.info(f"Reached maximum total publication limit ({max_publications})")
                        break
                    
                    logger.info(f"Fetching publications for {expert['first_name']} {expert['last_name']}")
                    fetched_works = await self._fetch_expert_publications(
                        session, 
                        expert['orcid'],
                        expert,
                        per_page=min(5, max_publications - publication_count)
                    )
                    
                    for work_summary in fetched_works:
                        try:
                            if publication_count >= max_publications:
                                break
                                
                            work = self._convert_orcid_to_standard_format(work_summary, expert)
                            if not work:
                                continue
                                
                            self.db.execute("BEGIN")
                            try:
                                processed = pub_processor.process_single_work(work, source=source)
                                if processed:
                                    publication_count += 1
                                    logger.info(
                                        f"Processed publication {publication_count}/{max_publications}: "
                                        f"{work.get('title', 'Unknown Title')}"
                                    )
                                    self.db.execute("COMMIT")
                                else:
                                    self.db.execute("ROLLBACK")
                                    
                            except Exception as e:
                                self.db.execute("ROLLBACK")
                                logger.error(f"Error in transaction: {e}")
                                continue
                                
                        except Exception as e:
                            logger.error(f"Error processing work: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(
                        f"Error processing publications for {expert['first_name']} {expert['last_name']}: {e}"
                    )
                    continue
        
        logger.info(f"ORCID publications processing completed. Total processed: {publication_count}")

    async def _fetch_expert_publications(
        self, 
        session: aiohttp.ClientSession, 
        orcid: str,
        expert: Dict,
        per_page: int = 5
    ) -> List[Dict]:
        try:
            clean_orcid = orcid.replace('https://orcid.org/', '')
            url = f"{self.base_url}/{clean_orcid}/works"
            headers = {
                "Accept": "application/json", 
                "Authorization": f"Bearer {self.access_token}"
            }
            params = {'per-page': per_page}
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if not isinstance(data, dict):
                            logger.error(f"Unexpected response format: {type(data)}")
                            return []
                            
                        works = []
                        for group in data.get('group', []):
                            if not isinstance(group, dict):
                                continue
                                
                            work_summaries = group.get('work-summary', [])
                            if not work_summaries or not isinstance(work_summaries, list):
                                continue
                            
                            summary = work_summaries[0]
                            if not isinstance(summary, dict):
                                continue
                                
                            works.append(summary)
                        
                        return works
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON response: {e}")
                        return []
                else:
                    logger.error(f"Failed to fetch works: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching ORCID publications: {e}")
            return []

    def _convert_orcid_to_standard_format(self, work: Dict, expert: Dict) -> Optional[Dict]:
        """
        Convert ORCID work format to standard format.
        
        Args:
            work: Dictionary containing work data from ORCID
            expert: Dictionary containing expert information
            
        Returns:
            Optional[Dict]: Standardized work data or None if conversion fails
        """
        try:
            if not isinstance(work, dict) or not isinstance(expert, dict):
                logger.error(f"Invalid input types: work={type(work)}, expert={type(expert)}")
                return None

            # Extract title
            title_container = work.get('title', {})
            title = title_container.get('title', {}).get('value', '') if isinstance(title_container, dict) else ''

            # Extract publication date
            pub_date = work.get('publication-date', {})
            year = pub_date.get('year', {}).get('value', '') if isinstance(pub_date, dict) else ''

            # Get the URL from multiple possible sources
            url = None
            
            # First try the URL field
            url_container = work.get('url', {})
            if isinstance(url_container, dict):
                url = url_container.get('value', '')

            # If no URL, try DOI
            if not url:
                doi = self._get_identifier(work, 'doi')
                if doi:
                    url = f"https://doi.org/{doi}"

            # If still no URL, try other identifiers
            if not url:
                for id_type in ['handle', 'uri']:
                    identifier = self._get_identifier(work, id_type)
                    if identifier:
                        url = identifier
                        break

            # If still no URL, create one from ORCID work
            if not url:
                url = f"https://orcid.org/{expert['orcid']}/work/{work.get('put-code', '')}"

            # Process contributors
            contributors = []
            contributors_container = work.get('contributors', {})
            if isinstance(contributors_container, dict):
                for contributor in contributors_container.get('contributor', []):
                    if not isinstance(contributor, dict):
                        continue
                    credit_name = contributor.get('credit-name', {})
                    if isinstance(credit_name, dict):
                        name = credit_name.get('value', '')
                        if name:
                            contributors.append({
                                'name': name,
                                'role': contributor.get('contributor-attributes', {}).get('contributor-role', '')
                            })

            # Build response with URL as DOI
            response_data = {
                'id': str(uuid.uuid4()),
                'source': 'orcid',
                'expert_first_name': str(expert.get('first_name', '')),
                'expert_last_name': str(expert.get('last_name', '')),
                'expert_orcid': str(expert.get('orcid', '')),
                'title': title,
                'type': work.get('type', ''),
                'doi': url,  # Store URL in the doi field
                'contributors': contributors,
                'publication_year': year,
                'journal': work.get('journal-title', {}).get('value', '')
            }

            if not response_data['title']:
                logger.warning("Work missing required title field")
                return None

            logger.info(f"Successfully converted work: {response_data['title']}")
            return response_data

        except Exception as e:
            logger.error(f"Error converting work to standard format: {e}")
            return None

    def _get_identifier(self, work_summary: Dict, id_type: str) -> str:
        """
        Extract a specific identifier from work summary.
        """
        try:
            external_ids = work_summary.get('external-ids', {}).get('external-id', [])
            for ext_id in external_ids:
                if ext_id.get('external-id-type') == id_type:
                    return ext_id.get('external-id-value', '')
        except Exception as e:
            logger.error(f"Error getting {id_type} identifier: {e}")
        return ''

    def close(self):
        """Close database connection and cleanup resources."""
        try:
            if hasattr(self, 'db'):
                self.db.close()
            logger.info("OrcidProcessor resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")