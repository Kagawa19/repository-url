import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import redis
from dotenv import load_dotenv
import os
import time
import json
from src.utils.db_utils import DatabaseConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ExpertRedisIndexManager:
    def __init__(self):
        """Initialize Redis index manager for experts with robust model loading."""
        try:
            self.db = DatabaseConnector()  # Initialize the database connector
            load_dotenv()

            # Define the path to the pre-downloaded model directory
            model_path = '/app/models/sentence-transformers/all-MiniLM-L6-v2'

            try:
                logger.info(f"Loading SentenceTransformer model from: {model_path}")
                self.embedding_model = SentenceTransformer(
                    model_path,
                    local_files_only=True  # Ensures only local files are used
                )
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                logger.warning("Falling back to None; manual embedding will be used.")
                self.embedding_model = None

            # Setup Redis connections
            self.setup_redis_connections()

            logger.info("ExpertRedisIndexManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ExpertRedisIndexManager: {e}")
            raise


    def setup_redis_connections(self):
        """Setup Redis connections with retry logic."""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                
                # Initialize Redis connections
                self.redis_text = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=True,
                    db=0
                )
                self.redis_binary = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=False,
                    db=0
                )
                
                # Test connections
                self.redis_text.ping()
                self.redis_binary.ping()
                
                logger.info("Redis connections established successfully")
                return
                
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error("Failed to connect to Redis after maximum retries")
                    raise
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)

    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        logger.info("Creating fallback embedding")
        # Create a deterministic embedding based on character values
        embedding = np.zeros(384)  # Standard dimension for simple embeddings
        for i, char in enumerate(text):
            embedding[i % len(embedding)] += ord(char) / 1000
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    def _parse_jsonb(self, data):
        """Parse JSONB data safely."""
        if not data:
            return {}
        try:
            if isinstance(data, str):
                return json.loads(data)
            return data
        except:
            return {}

    

    def _create_resource_text_content(self, resource: Dict[str, Any]) -> str:
        """Create combined text content for resource embedding with additional safeguards."""
        try:
            # Start with title as the most important information
            text_parts = []
            if resource.get('title'):
                text_parts.append(f"Title: {str(resource['title']).strip()}")
            else:
                text_parts.append("Title: Unknown Resource")

            # Add publication details
            if resource.get('publication_year'):
                text_parts.append(f"Year: {resource['publication_year']}")
                
            if resource.get('doi'):
                text_parts.append(f"DOI: {resource['doi']}")
                
            # Add type and source
            if resource.get('type'):
                text_parts.append(f"Type: {resource['type']}")
                
            if resource.get('source'):
                text_parts.append(f"Source: {resource['source']}")
                
            # Add field and subfield
            if resource.get('field'):
                field_text = f"Field: {resource['field']}"
                if resource.get('subfield'):
                    field_text += f", Subfield: {resource['subfield']}"
                text_parts.append(field_text)
            
            # Handle authors
            authors = resource.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    authors_text = ", ".join([str(author).strip() for author in authors if author])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
                elif isinstance(authors, str):
                    text_parts.append(f"Authors: {authors}")
                elif isinstance(authors, dict):
                    # Handle case where authors might be a dictionary
                    authors_text = ", ".join([str(value).strip() for value in authors.values() if value])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
            
            # Add domains as keywords
            domains = resource.get('domains', [])
            if domains and isinstance(domains, list):
                domains_text = ", ".join([str(domain).strip() for domain in domains if domain])
                if domains_text:
                    text_parts.append(f"Domains: {domains_text}")
            
            # Add summary - most important for semantic search
            if resource.get('summary'):
                summary = str(resource['summary']).strip()
                if summary:
                    text_parts.append(f"Summary: {summary}")
            
            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Resource"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for resource {resource.get('id', 'Unknown')}: {e}")
            return "Error Processing Resource"

    def _fetch_resource_details(self, resource_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch details for multiple resources by their IDs."""
        if not resource_ids:
            return {}
            
        resource_details = {}
        try:
            conn = self.db.get_connection()
            with conn.cursor() as cur:
                # Prepare placeholders for the IN clause
                placeholders = ', '.join(['%s'] * len(resource_ids))
                
                query = f"""
                    SELECT 
                        id, 
                        title, 
                        doi, 
                        abstract, 
                        publication_year,
                        topics
                    FROM resources_resource
                    WHERE id IN ({placeholders})
                """
                
                cur.execute(query, resource_ids)
                
                for row in cur.fetchall():
                    resource_id = row[0]
                    resource_details[resource_id] = {
                        'title': row[1] if row[1] else '',
                        'doi': row[2] if row[2] else '',
                        'abstract': row[3] if row[3] else '',
                        'publication_year': row[4] if row[4] else '',
                        'topics': self._parse_jsonb(row[5]) if row[5] else {}
                    }
                    
            return resource_details
            
        except Exception as e:
            logger.error(f"Error fetching resource details: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def _store_expert_data(self, expert: Dict[str, Any], text_content: str, embedding: np.ndarray) -> None:
        """
        Stores comprehensive expert data with APHRC classification.
        All experts from experts_expert table are considered APHRC experts.
        """
        # All experts are APHRC (from experts_expert table)
        base_key = f"aphrc_expert:{expert['id']}"
        
        pipeline = self.redis_text.pipeline()
        try:
            # Store text content
            pipeline.set(f"text:{base_key}", text_content)
            
            # Store embedding
            self.redis_binary.set(
                f"emb:{base_key}",
                embedding.astype(np.float32).tobytes()
            )
            
            # Enhanced metadata with more comprehensive fields
            metadata = {
                'id': str(expert['id']),
                'first_name': str(expert.get('first_name', '')),
                'last_name': str(expert.get('last_name', '')),
                'middle_name': str(expert.get('middle_name', '')),
                'email': str(expert.get('email', '')),
                'designation': str(expert.get('designation', '')),
                'theme': str(expert.get('theme', '')),
                'unit': str(expert.get('unit', '')),
                'expertise': json.dumps(expert.get('knowledge_expertise', {})),
                'bio': str(expert.get('bio', '')),
                'orcid': str(expert.get('orcid', '')),
                'domains': json.dumps(expert.get('domains', [])),
                'fields': json.dumps(expert.get('fields', [])),
                'subfields': json.dumps(expert.get('subfields', [])),
                'normalized_domains': json.dumps(expert.get('normalized_domains', [])),
                'normalized_fields': json.dumps(expert.get('normalized_fields', [])),
                'normalized_skills': json.dumps(expert.get('normalized_skills', [])),
                'keywords': json.dumps(expert.get('keywords', [])),
                'contact_details': str(expert.get('contact_details', '')),
                'photo_url': str(expert.get('photo', '')),
                'is_aphrc': 'true',  # Explicit marker
            }
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            
            # Store resource links (both APHRC and global publications)
            if expert.get('linked_resources'):
                links_key = f"links:expert:{expert['id']}:resources"
                pipeline.delete(links_key)  # Clear existing
                
                for link in expert['linked_resources']:
                    if link.get('resource_id') and link.get('confidence'):
                        pipeline.zadd(
                            links_key,
                            {str(link['resource_id']): float(link['confidence'])}
                        )
                        
                        # Reverse link (resource knows its APHRC experts)
                        res_key = f"links:resource:{link['resource_id']}:experts"
                        pipeline.zadd(
                            res_key,
                            {str(expert['id']): float(link['confidence'])}
                        )
            
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            logger.error(f"Error storing expert {expert.get('id')}: {e}")
            raise

    def _create_text_content(self, expert: Dict[str, Any]) -> str:
        """Create comprehensive text content for embedding with expert profile and publication data."""
        try:
            # Start with basic identity section
            text_parts = ["EXPERT PROFILE"]
            
            # Name section
            name_parts = []
            if expert.get('first_name'):
                name_parts.append(str(expert['first_name']).strip())
            if expert.get('middle_name'):
                name_parts.append(str(expert['middle_name']).strip())
            if expert.get('last_name'):
                name_parts.append(str(expert['last_name']).strip())
            
            if name_parts:
                text_parts.append(f"Name: {' '.join(name_parts)}")
            else:
                text_parts.append("Name: Unknown Expert")
                
            # Professional information - ENHANCED to include designation
            professional_parts = []
            if expert.get('designation'):
                professional_parts.append(f"Position: {expert['designation']}")
            if expert.get('unit'):
                professional_parts.append(f"Unit: {expert['unit']}")
            if expert.get('theme'):
                professional_parts.append(f"Theme: {expert['theme']}")
                
            if professional_parts:
                text_parts.append("PROFESSIONAL INFORMATION")
                text_parts.extend(professional_parts)
                
            # Contact information - MODIFIED to include ORCID more prominently
            contact_parts = []
            if expert.get('email'):
                contact_parts.append(f"Email: {expert['email']}")
            if expert.get('contact_details'):
                contact_parts.append(f"Contact: {expert['contact_details']}")
            if expert.get('orcid'):
                contact_parts.append(f"ORCID: {expert['orcid']}")
                
            if contact_parts:
                text_parts.append("CONTACT INFORMATION")
                text_parts.extend(contact_parts)
                
            # Biography - ENHANCED to include more context
            if expert.get('bio'):
                text_parts.append("BIOGRAPHY")
                text_parts.append(expert['bio'])
                
            # Expertise and research areas section - ENHANCED for better matching
            expertise_parts = []
            
            # Handle knowledge expertise
            expertise = expert.get('knowledge_expertise', {})
            if expertise and isinstance(expertise, dict):
                for key, value in expertise.items():
                    if value:
                        if isinstance(value, list):
                            # Clean list values
                            clean_values = [str(v).strip() for v in value if v is not None]
                            clean_values = [v for v in clean_values if v]  # Remove empty strings
                            if clean_values:
                                expertise_parts.append(f"{key.title()}: {' | '.join(clean_values)}")
                        elif isinstance(value, (str, int, float)):
                            # Handle single values
                            clean_value = str(value).strip()
                            if clean_value:
                                expertise_parts.append(f"{key.title()}: {clean_value}")
            elif expertise and isinstance(expertise, list):
                # Handle case where knowledge_expertise is a list
                clean_values = [str(v).strip() for v in expertise if v is not None]
                clean_values = [v for v in clean_values if v]  # Remove empty strings
                if clean_values:
                    expertise_parts.append(f"Expertise: {' | '.join(clean_values)}")
                    
            # Add domains, fields, and subfields - ENHANCED to include normalized versions
            if expert.get('domains'):
                expertise_parts.append(f"Research Domains: {' | '.join(str(d) for d in expert['domains'])}")
            if expert.get('fields'):
                expertise_parts.append(f"Research Fields: {' | '.join(str(f) for f in expert['fields'])}")
            if expert.get('subfields'):
                expertise_parts.append(f"Research Subfields: {' | '.join(str(s) for s in expert['subfields'])}")
                
            # Add normalized data - NEW section for better matching
            if expert.get('normalized_domains'):
                expertise_parts.append(f"Normalized Domains: {' | '.join(str(d) for d in expert['normalized_domains'])}")
            if expert.get('normalized_fields'):
                expertise_parts.append(f"Normalized Fields: {' | '.join(str(f) for f in expert['normalized_fields'])}")
            if expert.get('normalized_skills'):
                expertise_parts.append(f"Skills: {' | '.join(str(s) for s in expert['normalized_skills'])}")
                
            # Add keywords and search_text - NEW section for improved searchability
            if expert.get('keywords'):
                expertise_parts.append(f"Keywords: {' | '.join(str(k) for k in expert['keywords'])}")
            if expert.get('search_text'):
                expertise_parts.append(f"Search Terms: {expert['search_text']}")
                
            if expertise_parts:
                text_parts.append("EXPERTISE AND RESEARCH AREAS")
                text_parts.extend(expertise_parts)
                
            # Add publication metadata - keep unchanged
            linked_resources = expert.get('linked_resources', [])
            if linked_resources and isinstance(linked_resources, list):
                # Sort resources by confidence score (highest first)
                sorted_resources = sorted(
                    linked_resources, 
                    key=lambda x: float(x.get('confidence', 0)), 
                    reverse=True
                )
                
                # Get top publications (limit to 5 most confident matches)
                top_resources = sorted_resources[:5]
                
                if top_resources:
                    # Fetch publication details from database
                    resource_details = self._fetch_resource_details([r.get('resource_id') for r in top_resources])
                    
                    if resource_details:
                        text_parts.append("KEY PUBLICATIONS")
                        for resource_id, details in resource_details.items():
                            # Format each publication with its metadata
                            pub_parts = []
                            
                            # Add title
                            if details.get('title'):
                                pub_parts.append(f"Title: {details['title']}")
                            
                            # Add year
                            if details.get('publication_year'):
                                pub_parts.append(f"Year: {details['publication_year']}")
                            
                            # Add DOI if available
                            if details.get('doi'):
                                pub_parts.append(f"DOI: {details['doi']}")
                            
                            # Add authors if available
                            if details.get('authors'):
                                authors = details['authors']
                                if isinstance(authors, list):
                                    author_text = ', '.join(str(a) for a in authors[:5])
                                    if len(authors) > 5:
                                        author_text += " et al."
                                else:
                                    author_text = str(authors)
                                pub_parts.append(f"Authors: {author_text}")
                                
                            # Add abstract snippet if available
                            if details.get('abstract'):
                                abstract = details['abstract']
                                # Take first 150 characters of abstract
                                abstract_snippet = abstract[:150] + ('...' if len(abstract) > 150 else '')
                                pub_parts.append(f"Abstract: {abstract_snippet}")
                            
                            # Add topics if available
                            if details.get('topics'):
                                topics = details['topics']
                                if isinstance(topics, list):
                                    topic_text = ' | '.join(str(t) for t in topics[:5])
                                elif isinstance(topics, dict):
                                    topic_text = ' | '.join(f"{k}: {v}" for k, v in list(topics.items())[:3])
                                else:
                                    topic_text = str(topics)
                                pub_parts.append(f"Topics: {topic_text}")
                                
                            # Join publication parts with spaces
                            text_parts.append(' '.join(pub_parts))
            
            # Join all parts with double newlines for good separation
            final_text = '\n\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Expert Profile"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for expert {expert.get('id', 'Unknown')}: {e}")
            return "Error Processing Expert Profile"

    def _create_resource_text_content(self, resource: Dict[str, Any]) -> str:
        """Create comprehensive text content for resource embedding."""
        try:
            # Create a comprehensive text representation that captures all significant details
            text_parts = []
            
            # Essential fields
            if resource.get('title'):
                text_parts.append(f"Title: {str(resource['title']).strip()}")
            
            # Abstract or summary (prioritize abstract)
            if resource.get('abstract'):
                text_parts.append(f"Abstract: {str(resource['abstract']).strip()}")
            elif resource.get('summary'):
                text_parts.append(f"Summary: {str(resource['summary']).strip()}")
            
            # Bibliographic details
            if resource.get('publication_year'):
                text_parts.append(f"Year: {resource['publication_year']}")
            
            if resource.get('doi'):
                text_parts.append(f"DOI: {resource['doi']}")
            
            # Authors
            authors = resource.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    authors_text = ", ".join([str(author).strip() for author in authors if author])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
                elif isinstance(authors, str):
                    text_parts.append(f"Authors: {authors}")
                elif isinstance(authors, dict):
                    # Handle dictionary representation of authors
                    authors_text = ", ".join([str(value).strip() for value in authors.values() if value])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
            
            # Additional contextual information
            if resource.get('type'):
                text_parts.append(f"Type: {resource['type']}")
            
            if resource.get('source'):
                text_parts.append(f"Source: {resource['source']}")
            
            # Domains and topics
            domains = resource.get('domains', [])
            if domains and isinstance(domains, list):
                domains_text = ", ".join([str(domain).strip() for domain in domains if domain])
                if domains_text:
                    text_parts.append(f"Domains: {domains_text}")
            
            topics = resource.get('topics', {})
            if topics and isinstance(topics, dict):
                topics_text = ", ".join([f"{k}: {v}" for k, v in topics.items() if v])
                if topics_text:
                    text_parts.append(f"Topics: {topics_text}")
            
            # Optional detailed fields
            if resource.get('description'):
                text_parts.append(f"Description: {str(resource['description']).strip()}")
            
            # Identifiers
            identifiers = resource.get('identifiers', {})
            if identifiers and isinstance(identifiers, dict):
                identifiers_text = ", ".join([f"{k}: {v}" for k, v in identifiers.items() if v])
                if identifiers_text:
                    text_parts.append(f"Identifiers: {identifiers_text}")
            
            # Publishers and collection
            publishers = resource.get('publishers', {})
            if publishers and isinstance(publishers, dict):
                publishers_text = ", ".join([str(value).strip() for value in publishers.values() if value])
                if publishers_text:
                    text_parts.append(f"Publishers: {publishers_text}")
            
            if resource.get('collection'):
                text_parts.append(f"Collection: {resource['collection']}")
            
            # Language and citation
            if resource.get('language'):
                text_parts.append(f"Language: {resource['language']}")
            
            if resource.get('citation'):
                text_parts.append(f"Citation: {resource['citation']}")
            
            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Work"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for work {resource.get('id', 'Unknown')}: {e}")
            return "Error Processing Work"
    

    def clear_redis_indexes(self) -> bool:
        """Clear all Redis indexes for both experts and publications."""
        try:
            # Clear both old and new patterns to ensure complete cleanup
            
            # Clear expert indexes - old patterns
            old_expert_patterns = ['text:expert:*', 'emb:expert:*', 'meta:expert:*']
            for pattern in old_expert_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear expert indexes - new patterns
            new_expert_patterns = ['text:aphrc_expert:*', 'emb:aphrc_expert:*', 'meta:aphrc_expert:*']
            for pattern in new_expert_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear publication/resource indexes - old patterns
            old_resource_patterns = [
                'text:resource:*', 'emb:resource:*', 'meta:resource:*',
                'text:publication:*', 'emb:publication:*', 'meta:publication:*'
            ]
            for pattern in old_resource_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear publication/resource indexes - new patterns
            new_resource_patterns = [
                'text:expert_resource:*', 'emb:expert_resource:*', 'meta:expert_resource:*'
            ]
            for pattern in new_resource_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear resource links
            link_patterns = ['links:expert:*', 'links:resource:*', 'expert:*:resources']
            for pattern in link_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            logger.info("Cleared all expert and publication Redis indexes (both old and new patterns)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis indexes: {e}")
            return False

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch comprehensive expert data from database including their resource links."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if tables exist
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'experts_expert'
                        ) AS expert_table_exists,
                        EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'expert_resource_links'
                        ) AS links_table_exists;
                    """)
                    exists = cur.fetchone()
                    if not exists[0]:  # experts_expert table doesn't exist
                        logger.warning("experts_expert table does not exist yet")
                        return []
                    
                    # Fetch experts with their linked resources in one query
                    # Expanded to include many more fields from experts_expert table
                    query = """
                        SELECT 
                            e.id,
                            e.first_name,
                            e.last_name,
                            e.middle_name,
                            e.knowledge_expertise,
                            e.email,
                            e.designation,
                            e.theme,
                            e.unit,
                            e.bio,
                            e.orcid,
                            e.domains,
                            e.fields,
                            e.subfields,
                            e.normalized_domains,
                            e.normalized_fields,
                            e.normalized_skills,
                            e.keywords,
                            e.contact_details,
                            e.photo,
                            COALESCE(
                                json_agg(
                                    json_build_object(
                                        'resource_id', erl.resource_id,
                                        'confidence', erl.confidence_score,
                                        'author_position', erl.author_position
                                    ) 
                                    ORDER BY erl.confidence_score DESC
                                ) FILTER (WHERE erl.resource_id IS NOT NULL),
                                '[]'::json
                            ) AS linked_resources
                        FROM experts_expert e
                        LEFT JOIN expert_resource_links erl ON e.id = erl.expert_id
                        WHERE e.id IS NOT NULL AND e.is_active = TRUE
                        GROUP BY 
                            e.id, e.first_name, e.last_name, e.middle_name, e.knowledge_expertise,
                            e.email, e.designation, e.theme, e.unit, e.bio, e.orcid,
                            e.domains, e.fields, e.subfields, e.normalized_domains,
                            e.normalized_fields, e.normalized_skills, e.keywords,
                            e.contact_details, e.photo
                    """ if exists[1] else """
                        SELECT 
                            id,
                            first_name,
                            last_name,
                            middle_name,
                            knowledge_expertise,
                            email,
                            designation,
                            theme,
                            unit,
                            bio,
                            orcid,
                            domains,
                            fields,
                            subfields,
                            normalized_domains,
                            normalized_fields,
                            normalized_skills,
                            keywords,
                            contact_details,
                            photo
                        FROM experts_expert
                        WHERE id IS NOT NULL AND is_active = TRUE
                    """
                    
                    cur.execute(query)
                    experts = []
                    for row in cur.fetchall():
                        # Create expert dictionary with many more fields
                        expert = {
                            'id': row[0],
                            'first_name': row[1] or '',
                            'last_name': row[2] or '',
                            'middle_name': row[3] or '',
                            'knowledge_expertise': self._parse_jsonb(row[4]),
                            'email': row[5] or '',
                            'designation': row[6] or '',
                            'theme': row[7] or '',
                            'unit': row[8] or '',
                            'bio': row[9] or '',
                            'orcid': row[10] or '',
                            'domains': row[11] if row[11] else [],
                            'fields': row[12] if row[12] else [],
                            'subfields': row[13] if row[13] else [],
                            'normalized_domains': row[14] if row[14] else [],
                            'normalized_fields': row[15] if row[15] else [],
                            'normalized_skills': row[16] if row[16] else [],
                            'keywords': row[17] if row[17] else [],
                            'contact_details': row[18] or '',
                            'photo': row[19] or '',
                            'linked_resources': row[20] if len(row) > 20 else []
                        }
                        experts.append(expert)
                    
                    logger.info(f"Fetched {len(experts)} experts with comprehensive profiles and resource links")
                    return experts
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    raise
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()
        

    # 1. Add this method to ExpertRedisIndexManager class:

    def create_expert_redis_index(self) -> bool:
        """Create Redis indexes for experts including their resource links."""
        try:
            logger.info("Starting expert indexing including resource links...")
            
            experts = self.fetch_experts()
            
            if not experts:
                logger.warning("No experts found to index")
                return False
            
            success_count = 0
            error_count = 0
            
            for expert in experts:
                try:
                    expert_id = expert.get('id', 'Unknown')
                    logger.info(f"Processing expert {expert_id} with {len(expert.get('linked_resources', []))} resources")
                    
                    # Create text content
                    text_content = self._create_text_content(expert)
                    if not text_content or text_content.isspace():
                        logger.warning(f"Empty text content for expert {expert_id}")
                        continue

                    # Generate embedding
                    try:
                        embedding = (
                            self.embedding_model.encode(text_content) 
                            if self.embedding_model is not None 
                            else self._create_fallback_embedding(text_content)
                        )
                    except Exception as embed_err:
                        logger.error(f"Embedding failed for expert {expert_id}: {embed_err}")
                        embedding = self._create_fallback_embedding(text_content)
                    
                    # Store all data including links
                    self._store_expert_data(expert, text_content, embedding)
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error indexing expert {expert_id}: {e}")
                    continue
            
            logger.info(
                f"Expert indexing complete. Success: {success_count}, "
                f"Failures: {error_count}, "
                f"Total resource links processed: {sum(len(e.get('linked_resources', [])) for e in experts)}"
            )
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Fatal error in expert indexing: {e}")
            return False

    
    def create_redis_index(self) -> bool:
        """
        Create Redis indexes for experts, publications, and their relationships.
        
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            logger.info("Starting comprehensive Redis indexing process...")
            
            # Track overall success
            overall_success = True
            
            # Step 1: Clear existing indexes (optional, but can help prevent stale data)
            self.clear_redis_indexes()
            
            # Step 2: Index experts
            experts_success = self.create_expert_redis_index()
            
            # Step 3: Index publications/resources
            publications_success = self.create_publications_redis_index()
            
            # Determine final indexing status
            if experts_success and publications_success:
                logger.info("Successfully indexed experts and publications in Redis")
                return True
            else:
                logger.error("Failed to completely index experts and publications")
                return False
            
        except Exception as final_err:
            logger.error(f"Catastrophic error during Redis indexing: {final_err}")
            return False

            
    def fetch_resources(self) -> List[Dict[str, Any]]:
        """Fetch all resources from the resources_resource table."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if required table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'resources_resource'
                        ) AS resources_exists;
                    """)
                    exists = cur.fetchone()
                    
                    if not exists[0]:  # resources_resource table doesn't exist
                        logger.warning("resources_resource table does not exist yet")
                        return []
                    
                    # Simplified query to get all resources
                    query = """
                        SELECT * FROM resources_resource
                        ORDER BY id
                    """
                    
                    cur.execute(query)
                    resources = []
                    skipped_count = 0
                    
                    for row in cur.fetchall():
                        try:
                            resource = {
                                'id': row[0],
                                'doi': str(row[1]) if row[1] is not None else '',
                                'title': str(row[2]) if row[2] is not None else '',
                                'abstract': str(row[3]) if row[3] is not None else '',
                                'summary': str(row[4]) if row[4] is not None else '',
                                'domains': self._parse_jsonb(row[5]) if row[5] else [],
                                'topics': self._parse_jsonb(row[6]) if row[6] else {},
                                'description': str(row[7]) if row[7] is not None else '',
                                'expert_id': str(row[8]) if row[8] is not None else '',
                                'type': str(row[9]) if row[9] is not None else 'publication',
                                'subtitles': self._parse_jsonb(row[10]) if row[10] else {},
                                'publishers': self._parse_jsonb(row[11]) if row[11] else {},
                                'collection': str(row[12]) if row[12] is not None else '',
                                'date_issue': str(row[13]) if row[13] is not None else '',
                                'citation': str(row[14]) if row[14] is not None else '',
                                'language': str(row[15]) if row[15] is not None else '',
                                'identifiers': self._parse_jsonb(row[16]) if row[16] else {},
                                'created_at': row[17].isoformat() if row[17] else None,
                                'updated_at': row[18].isoformat() if row[18] else None,
                                'source': str(row[19]) if row[19] is not None else 'unknown',
                                'authors': self._parse_jsonb(row[20]) if row[20] else [],
                                'publication_year': str(row[21]) if row[21] is not None else ''
                            }
                            resources.append(resource)
                        except Exception as row_error:
                            skipped_count += 1
                            logger.error(f"Error processing resource row: {row_error}")
                            continue
                    
                    logger.info(
                        f"Fetched {len(resources)} resources from database "
                        f"(skipped {skipped_count} invalid records)"
                    )
                    return resources
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    raise
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()
    
    def create_publications_redis_index(self) -> bool:
        """Create Redis indexes for publications/resources."""
        try:
            logger.info("Starting publications indexing...")
            
            resources = self.fetch_resources()
            
            if not resources:
                logger.warning("No resources found to index")
                return False
            
            logger.info(f"Processing {len(resources)} resources for indexing")
            success_count = 0
            error_count = 0
            
            for resource in resources:
                try:
                    resource_id = resource.get('id', 'Unknown')
                    logger.info(f"Processing resource {resource_id}")
                    
                    # Create text content
                    text_content = self._create_resource_text_content(resource)
                    if not text_content or text_content.isspace():
                        logger.warning(f"Empty text content generated for resource {resource_id}")
                        continue
                    
                    # Generate embedding with robust fallback
                    try:
                        embedding = (
                            self.embedding_model.encode(text_content) 
                            if self.embedding_model is not None 
                            else self._create_fallback_embedding(text_content)
                        )
                        
                        if embedding is None or not isinstance(embedding, np.ndarray):
                            raise ValueError("Invalid embedding generated")
                    except Exception as embed_err:
                        logger.error(f"Embedding generation failed for resource {resource_id}: {embed_err}")
                        # Mandatory fallback embedding
                        embedding = self._create_fallback_embedding(text_content)
                    
                    # Store in Redis
                    self._store_resource_data(resource, text_content, embedding)
                    success_count += 1
                    logger.info(f"Successfully indexed resource {resource_id}")
                    
                except Exception as resource_index_err:
                    error_count += 1
                    logger.error(f"Error indexing resource {resource.get('id', 'Unknown')}: {resource_index_err}")
            
            logger.info(f"Resource indexing complete. Successes: {success_count}, Failures: {error_count}")
            return success_count > 0
            
        except Exception as resources_fetch_err:
            logger.error(f"Failed to fetch or process resources: {resources_fetch_err}")
            return False

    
    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison"""
        if not name or not isinstance(name, str):
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = ' '.join(name.lower().split())
        
        # Remove common suffixes and prefixes
        prefixes = ['dr.', 'dr ', 'prof.', 'prof ', 'professor ', 'mr.', 'mr ', 'mrs.', 'mrs ', 'ms.', 'ms ']
        suffixes = [' phd', ' md', ' jr', ' sr', ' jr.', ' sr.', ' ii', ' iii', ' iv']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        return normalized

    def _store_resource_data(self, resource: Dict[str, Any], text_content: str, embedding: np.ndarray) -> None:
        """Stores resources in Redis using work-based key patterns."""
        pipeline = None
        
        try:
            # Get expert_id directly from the resource
            expert_id = resource.get('expert_id')
            
            # Use new work-based key pattern
            resource_id = resource['id']
            pipeline = self.redis_text.pipeline()
            
            # Store text content
            pipeline.set(f"text:work:{resource_id}", text_content)
            
            # Store embedding
            self.redis_binary.set(f"emb:work:{resource_id}", embedding.astype(np.float32).tobytes())
            
            # Store metadata
            metadata = {
                'id': str(resource_id),
                'title': str(resource.get('title', '')),
                'abstract': str(resource.get('abstract', '')),
                'authors': json.dumps(resource.get('authors', [])),
                'publication_year': str(resource.get('publication_year', '')),
                'expert_id': str(expert_id) if expert_id else '',
                'doi': str(resource.get('doi', ''))
            }
            pipeline.hset(f"meta:work:{resource_id}", mapping=metadata)
            
            # If there's an expert_id, create a link from author to work
            if expert_id and str(expert_id).strip().lower() != 'none':
                author_works_key = f"author:{expert_id}:works"
                pipeline.sadd(author_works_key, str(resource_id))
            
            pipeline.execute()
            logger.info(f"Successfully stored work {resource_id}")
            
        except Exception as e:
            if pipeline:
                pipeline.reset()
                logger.debug("Reset Redis pipeline due to error")
            logger.error(f"Error storing work {resource.get('id')}: {e}")
            raise
            

    

    
    def close(self):
        """Close Redis connections."""
        try:
            if hasattr(self, 'redis_text'):
                self.redis_text.close()
            if hasattr(self, 'redis_binary'):
                self.redis_binary.close()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

    def __del__(self):
        """Ensure connections are closed on deletion."""
        self.close()

    

    
    