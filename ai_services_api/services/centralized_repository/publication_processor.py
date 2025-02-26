import logging
import json  # Add at the top of both files
from typing import Dict, Optional, List, Any
import uuid

from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.text_processor import (
    safe_str, 
    convert_inverted_index_to_text, 
    truncate_text
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PublicationProcessor:
    def __init__(self, db: DatabaseManager, summarizer: TextSummarizer):
        """Initialize PublicationProcessor."""
        self.db = db
        self.summarizer = summarizer
        self._setup_database_indexes()
    def _setup_database_indexes(self) -> None:
        """Create necessary database indexes if they don't exist."""
        try:
            indexes = [
                """
                CREATE INDEX IF NOT EXISTS idx_resources_doi 
                ON resources_resource(doi);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_experts_name 
                ON experts_expert(first_name, last_name);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_resource_topics 
                ON resources_resource USING gin(topics);
                """
            ]
            
            for index_sql in indexes:
                try:
                    self.db.execute(index_sql)
                except Exception as e:
                    logger.error(f"Error creating index: {e}")
                    continue
                    
            logger.info("Database indexes verified/created successfully")
        except Exception as e:
            logger.error(f"Error setting up database indexes: {e}")
            raise

    def _check_publication_exists(self, title: str, doi: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Check if publication exists and get its summary.
        
        Args:
            title: Publication title
            doi: Optional DOI
            
        Returns:
            tuple: (exists, summary)
        """
        try:
            if doi:
                result = self.db.execute("""
                    SELECT EXISTS(SELECT 1 FROM resources_resource WHERE doi = %s),
                           (SELECT summary FROM resources_resource WHERE doi = %s)
                    """, (doi, doi))
                if result and result[0][0]:
                    return True, result[0][1]

            # If no DOI or DOI not found, check by title
            result = self.db.execute("""
                SELECT EXISTS(SELECT 1 FROM resources_resource WHERE title = %s),
                       (SELECT summary FROM resources_resource WHERE title = %s)
                """, (title, title))
            if result:
                return result[0][0], result[0][1]
            return False, None
        except Exception as e:
            logger.error(f"Error checking publication existence: {e}")
            return False, None

    def _clean_and_validate_work(self, work: Dict) -> tuple[Optional[str], Optional[str]]:
        """Clean and validate work data."""
        try:
            # Get DOI safely
            doi = None
            if isinstance(work.get('doi'), str):
                doi = work.get('doi')
            
            # Get title safely with defensive chaining
            title = None
            title_data = work.get('title', {}) if isinstance(work.get('title'), dict) else work.get('title', '')
            if isinstance(title_data, dict):
                title = ((title_data.get('title', {}) or {}).get('value', ''))
            elif isinstance(title_data, str):
                title = title_data.strip()
                
            return doi, title if title else None
            
        except Exception as e:
            logger.error(f"Error in clean and validate: {e}")
            return None, None

    def process_orcid_work(self, work: Dict, source: str = 'orcid') -> bool:
        """Process a single publication work."""
        try:
            # Clean and validate work
            doi, title = self._clean_and_validate_work(work)
            if not title:  # Title is required, DOI is optional
                logger.warning("Missing required title, skipping work")
                return False

            # Process authors with defensive handling
            authors = []
            for authorship in (work.get('authorships', []) or []):
                author = (authorship or {}).get('author', {})
                if author and isinstance(author, dict):
                    display_name = author.get('display_name')
                    if display_name and isinstance(display_name, str):
                        authors.append(display_name.strip())

            # Process domains/concepts with defensive handling
            domains = []
            for concept in (work.get('concepts', []) or []):
                concept = concept or {}
                if isinstance(concept, dict):
                    display_name = concept.get('display_name')
                    if display_name and isinstance(display_name, str):
                        domains.append(display_name.strip())

            # Get publication year safely
            publication_year = None
            if isinstance(work.get('publication_year'), (int, str)):
                try:
                    publication_year = int(work.get('publication_year'))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid publication year format for {title}")

            # Process abstract for summary generation
            abstract = work.get('abstract', '')
            if not abstract and title:
                logger.info("No abstract available, generating description from title")
                abstract = f"Publication about {title}"

            # Generate summary
            summary = None
            try:
                if abstract:
                    logger.info(f"Generating summary for: {title}")
                    summary = self.summarizer.summarize(title, abstract)
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = abstract[:500] if abstract else None

            # Add to database
            success = self.db.add_publication(
                title=title,
                doi=doi,
                authors=authors,
                domains=domains,
                publication_year=publication_year,
                summary=summary
            )

            if success:
                logger.info(f"Successfully processed publication: {title}")
                return True
            else:
                logger.error(f"Failed to add publication to database: {title}")
                return False

        except Exception as e:
            logger.error(f"Error processing work: {e}")
            return False

   

    def process_single_work(self, work: Dict, source: str = 'openalex') -> bool:
        """Process a single publication work."""
        try:
            # Clean and validate work
            doi, title = self._clean_and_validate_work(work)
            if not title:  # Title is required, DOI is optional
                logger.info("No title found for work, skipping")
                return False
            
            # Check if publication exists and has summary
            exists, existing_summary = self._check_publication_exists(title, doi)
            if exists and existing_summary:
                logger.info(f"Publication already exists with summary. Skipping.")
                return False
            
            # Process abstract for summary generation
            abstract = work.get('abstract', '')
            if not abstract:
                logger.info("No abstract available, using title")
                abstract = title

            # Generate summary
            try:
                summary, content_type = self.summarizer.summarize(title, abstract)
                if not summary:
                    summary = abstract[:500]
                    content_type = 'publications'
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = abstract[:500]
                content_type = 'publications'
            
            # Process authors
            authors = []
            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                if author.get('display_name'):
                    authors.append(author['display_name'])
            
            # Process domains/concepts
            domains = []
            for concept in work.get('concepts', []):
                if concept.get('display_name'):
                    domains.append(concept['display_name'])

            # Set type based on source
            if source == 'openalex':
                content_type = 'publications'
            else:
                content_type = work.get('type', 'publications')
                    
            # Add main publication record
            self.db.add_publication(
                title=title,
                doi=doi,  
                summary=summary,
                source=source,
                type=content_type,
                authors=authors,
                domains=domains,
                publication_year=work.get('publication_year')
            )
            
            logger.info(f"Successfully processed publication: {title} with type: {content_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing work: {e}")
            return False
    def _extract_metadata(self, work: Dict) -> Dict[str, Any]:
            """
            Extract additional metadata from work.
            
            Args:
                work: Publication work dictionary
                
            Returns:
                dict: Extracted metadata
            """
            try:
                return {
                    'type': safe_str(work.get('type')),
                    'publication_year': work.get('publication_year'),
                    'citation_count': work.get('cited_by_count'),
                    'language': safe_str(work.get('language')),
                    'publisher': safe_str(work.get('publisher')),
                    'journal': safe_str(work.get('host_venue', {}).get('display_name')),
                    'fields_of_study': work.get('fields_of_study', [])
                }
            except Exception as e:
                logger.error(f"Error extracting metadata: {e}")
                return {}

    def _process_authors(self, authorships: List[Dict], doi: str) -> None:
        """
        Process all authors from a work and add them as author tags.
        
        Args:
            authorships: List of authorship information
            doi: Publication DOI
        """
        try:
            for authorship in authorships:
                author = authorship.get('author', {})
                if not author:
                    continue

                author_name = author.get('display_name')
                if not author_name:
                    continue

                tag_info = {
                    'name': author_name,
                    'type': 'author',
                    'metadata': {
                        'orcid': author.get('orcid'),
                        'openalex_id': author.get('id'),
                        'affiliations': [
                            aff.get('display_name') 
                            for aff in authorship.get('institutions', [])
                        ],
                        'is_corresponding': authorship.get('is_corresponding', False)
                    }
                }

                try:
                    tag_id = self.db.add_tag(tag_info)
                    self.db.link_publication_tag(doi, tag_id)
                    logger.debug(f"Processed author tag: {author_name}")
                except Exception as e:
                    logger.error(f"Error adding author tag: {e}")

        except Exception as e:
            logger.error(f"Error processing authors: {e}")

    def _process_domains(self, work: Dict, doi: str) -> None:
        """
        Process domain information from work and add as domain tags.
        
        Args:
            work: Work dictionary with domain information
            doi: Publication DOI
        """
        try:
            # Process topics/domains
            for topic in work.get('topics', []):
                # Process domain
                domain = topic.get('domain', {}).get('display_name')
                if domain:
                    tag_info = {
                        'name': domain,
                        'type': 'domain',
                        'metadata': {
                            'score': topic.get('score'),
                            'level': topic.get('level'),
                            'field': topic.get('field', {}).get('display_name'),
                            'subfields': [
                                sf.get('display_name') 
                                for sf in topic.get('subfields', [])
                            ]
                        }
                    }
                    try:
                        tag_id = self.db.add_tag(tag_info)
                        self.db.link_publication_tag(doi, tag_id)
                        logger.debug(f"Processed domain tag: {domain}")
                    except Exception as e:
                        logger.error(f"Error adding domain tag: {e}")

        except Exception as e:
            logger.error(f"Error processing domains: {e}")

   

    def process_batch(self, works: List[Dict], source: str = 'openalex') -> int:
        """
        Process a batch of works.
        
        Args:
            works: List of publication work dictionaries
            source: Source of the publications (default: 'openalex')
            
        Returns:
            int: Number of successfully processed works
        """
        successful = 0
        for work in works:
            try:
                if self.process_single_work(work, source):
                    successful += 1
            except Exception as e:
                logger.error(f"Error processing work in batch: {e}")
                continue
        return successful

    def close(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'db'):
                self.db.close()
            logger.info("PublicationProcessor resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
