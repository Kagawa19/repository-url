"""
Graph database initialization and semantic processing module.
"""
import os
import logging
import psycopg2
import google.generativeai as genai
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import json
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import socket
import requests

# Load environment variables
load_dotenv()

# Configure HTTP proxy if needed
if os.getenv('HTTP_PROXY'):
    os.environ['HTTPS_PROXY'] = os.getenv('HTTP_PROXY')
    os.environ['https_proxy'] = os.getenv('HTTP_PROXY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_connectivity():
    """Diagnose connectivity issues with Gemini API"""
    try:
        # Test DNS resolution
        try:
            ip = socket.gethostbyname('generativelanguage.googleapis.com')
            logger.info(f"DNS resolution for Gemini API: {ip}")
        except Exception as e:
            logger.error(f"DNS resolution failed: {e}")
        
        # Test HTTP connectivity
        try:
            resp = requests.get('https://generativelanguage.googleapis.com/healthz', timeout=10)
            logger.info(f"HTTP connectivity check: Status {resp.status_code}")
        except Exception as e:
            logger.error(f"HTTP connectivity test failed: {e}")
    except Exception as e:
        logger.error(f"Connectivity diagnostics failed: {e}")

# Run connectivity check at startup
check_connectivity()

# Configure Gemini
model = None
try:
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        logger.info("Gemini API key found, configuring client")
        genai.configure(api_key=gemini_api_key)
        
        # Test the configuration
        try:
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            model = None
    else:
        logger.warning("No Gemini API key found in environment variables")
        model = None
except ImportError as e:
    logger.warning(f"Could not import Google Generative AI package: {e}")
    model = None
except Exception as e:
    logger.error(f"Unexpected error configuring Gemini: {e}")
    model = None

class DatabaseConnectionManager:
    """Manages database connections and configuration"""
    
    @staticmethod
    def get_postgres_connection():
        """Create a connection to PostgreSQL database."""
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            parsed_url = urlparse(database_url)
            conn_params = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        else:
            conn_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }

        try:
            conn = psycopg2.connect(**conn_params)
            logger.info(f"Successfully connected to database: {conn_params['dbname']}")
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

class GraphDatabaseInitializer:
    def __init__(self):
        """Initialize GraphDatabaseInitializer."""
        self._neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        logger.info("Neo4j driver initialized")
        
        # Initialize circuit breaker for Gemini API
        self._gemini_failures = 0
        self._gemini_disabled_until = None

    def _should_use_gemini(self):
        """Determine if Gemini should be used based on failure history"""
        if not model:
            return False
            
        if self._gemini_disabled_until:
            if datetime.now() < self._gemini_disabled_until:
                logger.info(f"Gemini API temporarily disabled until {self._gemini_disabled_until}")
                return False
            # Reset circuit breaker
            self._gemini_disabled_until = None
            self._gemini_failures = 0
            logger.info("Gemini API circuit breaker reset, will try again")
        
        return True

    def _call_gemini_with_retry(self, prompt, max_retries=2, timeout=30):
        """Call Gemini API with retry logic"""
        if not self._should_use_gemini():
            return None
            
        retries = 0
        while retries <= max_retries:
            try:
                logger.info(f"Calling Gemini API (attempt {retries+1}/{max_retries+1})")
                # Remove the timeout parameter since it's not supported
                response = model.generate_content(prompt)
                # Success, reset failure counter
                self._gemini_failures = 0
                return response
            except Exception as e:
                retries += 1
                logger.warning(f"Gemini API call failed (attempt {retries}/{max_retries+1}): {e}")
                if retries > max_retries:
                    # Track failures for circuit breaker
                    self._gemini_failures += 1
                    if self._gemini_failures >= 5:
                        # Disable for 30 minutes after 5 consecutive failures
                        self._gemini_disabled_until = datetime.now() + timedelta(minutes=30)
                        logger.warning(f"Temporarily disabled Gemini API until {self._gemini_disabled_until}")
                    raise
                time.sleep(2 * retries)  # Exponential backoff
        return None
    def _create_indexes(self):
        """Create enhanced indexes in Neo4j"""
        index_queries = [
            # Basic indexes
            "CREATE INDEX expert_id IF NOT EXISTS FOR (e:Expert) ON (e.id)",
            "CREATE INDEX expert_name IF NOT EXISTS FOR (e:Expert) ON (e.name)",
            "CREATE INDEX expert_orcid IF NOT EXISTS FOR (e:Expert) ON (e.orcid)",
            "CREATE INDEX theme_name IF NOT EXISTS FOR (t:Theme) ON (t.name)",
            "CREATE INDEX unit_name IF NOT EXISTS FOR (u:Unit) ON (u.name)",
            
            # New indexes for fields and domains
            "CREATE INDEX field_name IF NOT EXISTS FOR (f:Field) ON (f.name)",
            "CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name)",
            
            # Semantic indexes
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX area_name IF NOT EXISTS FOR (ra:ResearchArea) ON (ra.name)",
            "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX related_name IF NOT EXISTS FOR (r:RelatedArea) ON (r.name)",
            "CREATE INDEX interest_related IF NOT EXISTS FOR ()-[r:INTEREST_RELATED]-() ON (r.weight)",
            
            # Fulltext indexes
            """CREATE FULLTEXT INDEX expert_fulltext IF NOT EXISTS 
               FOR (e:Expert) ON EACH [e.name, e.designation]""",
            """CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS 
               FOR (c:Concept) ON EACH [c.name]"""
        ]
        
        with self._neo4j_driver.session() as session:
            for query in index_queries:
                try:
                    session.run(query)
                    logger.info(f"Index created: {query}")
                except Exception as e:
                    logger.warning(f"Error creating index: {e}")

    def _fetch_experts_data(self):
        """Fetch experts data from PostgreSQL"""
        conn = None
        try:
            conn = DatabaseConnectionManager.get_postgres_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    id,
                    first_name, 
                    last_name,
                    knowledge_expertise,
                    designation,
                    domains,
                    fields,
                    theme,
                    unit,
                    orcid,
                    is_active
                FROM experts_expert
                WHERE id IS NOT NULL
            """)
            
            experts_data = cur.fetchall()
            logger.info(f"Fetched {len(experts_data)} experts from database")
            return experts_data
        except Exception as e:
            logger.error(f"Error fetching experts data: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _process_historical_data(self):
        """Process existing search matches and messages for adaptive patterns"""
        conn = None
        try:
            conn = DatabaseConnectionManager.get_postgres_connection()
            cur = conn.cursor()
            
            # Get experts who appear together in search results
            try:
                cur.execute("""
                    SELECT 
                        e1.expert_id as expert1_id,
                        e2.expert_id as expert2_id,
                        COUNT(*) as co_occurrence,
                        AVG(e1.similarity_score + e2.similarity_score) / 2 as avg_similarity
                    FROM expert_search_matches e1
                    JOIN expert_search_matches e2 
                        ON e1.search_id = e2.search_id 
                        AND e1.expert_id < e2.expert_id
                    GROUP BY e1.expert_id, e2.expert_id
                    HAVING COUNT(*) > 1
                """)
                search_patterns = cur.fetchall()
                logger.info(f"Fetched {len(search_patterns)} search co-occurrence patterns")
            except Exception as e:
                logger.warning(f"Could not fetch search patterns: {e}")
                search_patterns = []
            
            # Get direct communication patterns
            try:
                cur.execute("""
                    SELECT 
                        sender_id,
                        receiver_id,
                        COUNT(*) as message_count,
                        MAX(created_at) as last_interaction
                    FROM expert_messages
                    GROUP BY sender_id, receiver_id
                    HAVING COUNT(*) > 0
                """)
                message_patterns = cur.fetchall()
                logger.info(f"Fetched {len(message_patterns)} message interaction patterns")
            except Exception as e:
                logger.warning(f"Could not fetch message patterns: {e}")
                message_patterns = []
            
            return search_patterns, message_patterns
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            return [], []
        finally:
            if conn:
                conn.close()

    def _create_adaptive_relationships(self, session, search_patterns, message_patterns):
        """Create/update relationship weights based on actual usage"""
        try:
            # Add weights from search co-occurrence
            for expert1_id, expert2_id, co_occurrence, avg_similarity in search_patterns:
                weight = min(co_occurrence * 0.1 * avg_similarity, 1.0)
                session.run("""
                    MATCH (e1:Expert {id: $expert1_id})
                    MATCH (e2:Expert {id: $expert2_id})
                    MERGE (e1)-[r:FREQUENTLY_SEARCHED_WITH]-(e2)
                    SET r.weight = $weight,
                        r.co_occurrence = $co_occurrence,
                        r.last_updated = datetime()
                """, {
                    "expert1_id": str(expert1_id),
                    "expert2_id": str(expert2_id),
                    "weight": weight,
                    "co_occurrence": co_occurrence
                })

            # Add weights from message interactions
            for sender_id, receiver_id, message_count, last_interaction in message_patterns:
                weight = min(message_count * 0.15, 1.0)
                session.run("""
                    MATCH (e1:Expert {id: $sender_id})
                    MATCH (e2:Expert {id: $receiver_id})
                    MERGE (e1)-[r:INTERACTS_WITH]->(e2)
                    SET r.weight = $weight,
                        r.message_count = $message_count,
                        r.last_updated = datetime()
                """, {
                    "sender_id": str(sender_id),
                    "receiver_id": str(receiver_id),
                    "weight": weight,
                    "message_count": message_count
                })
                
            logger.info("Created adaptive relationships based on historical patterns")
        except Exception as e:
            logger.error(f"Error creating adaptive relationships: {e}")

    def _process_expertise(self, expertise_list: List[str]) -> Dict[str, List[str]]:
        """Process expertise list and return standardized format"""
        try:
            if not expertise_list:
                return {
                    'concepts': [],
                    'areas': [],
                    'methods': [],
                    'related': []
                }

            # Check if we should use Gemini for enhanced processing
            if self._should_use_gemini():
                prompt = f"""Return only a raw JSON object with these keys for this expertise list: {expertise_list}
                {{
                    "standardized_concepts": [],
                    "research_areas": [],
                    "methods": [],
                    "related_areas": []
                }}
                Return the JSON object only, no markdown formatting, no code fences, no additional text."""
                
                try:
                    response = self._call_gemini_with_retry(prompt, max_retries=2, timeout=45)
                        
                    if not response or not response.text or not response.text.strip():
                        logger.warning("Received empty response from Gemini, using direct mapping")
                    else:
                        cleaned_response = (response.text
                                        .replace('```json', '')
                                        .replace('```JSON', '')
                                        .replace('```', '')
                                        .strip())
                        
                        try:
                            parsed = json.loads(cleaned_response)
                            logger.info("Successfully processed expertise with Gemini")
                            return {
                                'concepts': parsed.get('standardized_concepts', expertise_list),
                                'areas': parsed.get('research_areas', []),
                                'methods': parsed.get('methods', []),
                                'related': parsed.get('related_areas', [])
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
                except Exception as e:
                    logger.warning(f"Error in Gemini API call: {e}, falling back to direct mapping")
            
            # Fallback to heuristic method 
            logger.info("Using heuristic expertise mapping")
            
            # Simple heuristic categorization
            methods = []
            areas = []
            related = []
            
            method_keywords = ["method", "analysis", "technique", "approach", "framework", "assessment", "evaluation"]
            
            for expertise in expertise_list:
                if not expertise:
                    continue
                    
                lower_exp = expertise.lower()
                
                # Check if it might be a method
                if any(keyword in lower_exp for keyword in method_keywords):
                    methods.append(expertise)
                # Otherwise consider it a research area
                else:
                    areas.append(expertise)
            
            return {
                'concepts': expertise_list,
                'areas': areas,
                'methods': methods,
                'related': related
            }

        except Exception as e:
            logger.error(f"Error in expertise processing: {e}")
            return {
                'concepts': expertise_list,
                'areas': [],
                'methods': [],
                'related': []
            }

    def create_expert_node(self, session, expert_data: tuple):
        """Create expert node with semantic relationships"""
        try:
            # Unpack expert data
            (expert_id, first_name, last_name, knowledge_expertise, designation, 
            domains, fields, theme, unit, orcid, is_active) = expert_data
            
            expert_name = f"{first_name} {last_name}" if first_name and last_name else "Unknown Expert"

            # Create basic expert node
            session.run("""
                MERGE (e:Expert {id: $id})
                SET e.name = $name,
                    e.designation = $designation,
                    e.theme = $theme,
                    e.unit = $unit,
                    e.orcid = $orcid,
                    e.is_active = $is_active,
                    e.updated_at = datetime()
            """, {
                "id": str(expert_id),
                "name": expert_name,
                "designation": designation or "",
                "theme": theme or "",
                "unit": unit or "",
                "orcid": orcid or "",
                "is_active": is_active if is_active is not None else False
            })

            # Process expertise if available
            if knowledge_expertise:
                semantic_data = self._process_expertise(knowledge_expertise)
                self._create_semantic_relationships(session, str(expert_id), semantic_data)

            # Create field relationships if available
            if fields:
                for field in fields:
                    if not field:
                        continue
                        
                    session.run("""
                        MERGE (f:Field {name: $field})
                        MERGE (e:Expert {id: $expert_id})-[r:SPECIALIZES_IN]->(f)
                        SET r.weight = 0.9,
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "field": field
                    })

            # Create domain relationships if available
            if domains:
                for domain in domains:
                    if not domain:
                        continue
                        
                    session.run("""
                        MERGE (d:Domain {name: $domain})
                        MERGE (e:Expert {id: $expert_id})-[r:WORKS_IN_DOMAIN]->(d)
                        SET r.weight = 0.85,
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "domain": domain
                    })

            # Always create organizational relationships for reliable fallback matching
            # Theme relationship
            if theme:
                session.run("""
                    MERGE (t:Theme {name: $theme})
                    MERGE (e:Expert {id: $expert_id})-[r:BELONGS_TO_THEME]->(t)
                    SET r.last_updated = datetime()
                """, {
                    "expert_id": str(expert_id),
                    "theme": theme
                })

            # Unit relationship
            if unit:
                session.run("""
                    MERGE (u:Unit {name: $unit})
                    MERGE (e:Expert {id: $expert_id})-[r:BELONGS_TO_UNIT]->(u)
                    SET r.last_updated = datetime()
                """, {
                    "expert_id": str(expert_id),
                    "unit": unit
                })
                
            logger.info(f"Successfully created/updated expert node: {expert_name}")
                
        except Exception as e:
            logger.error(f"Error creating expert node for {expert_id}: {e}")
            raise


    def _create_semantic_relationships(self, session, expert_id: str, semantic_data: Dict[str, List[str]]):
        """Create semantic relationships for an expert"""
        try:
            # Create concept relationships (unchanged)
            for concept in semantic_data['concepts']:
                if not concept:
                    continue
                    
                session.run("""
                    MERGE (c:Concept {name: $concept})
                    MERGE (e:Expert {id: $expert_id})-[r:HAS_CONCEPT]->(c)
                    SET r.weight = 1.0,
                        r.last_updated = datetime()
                """, {
                    "expert_id": expert_id,
                    "concept": concept
                })

            # Create research area relationships (unchanged)
            for area in semantic_data['areas']:
                if not area:
                    continue
                    
                session.run("""
                    MERGE (ra:ResearchArea {name: $area})
                    MERGE (e:Expert {id: $expert_id})-[r:RESEARCHES_IN]->(ra)
                    SET r.weight = 0.8,
                        r.last_updated = datetime()
                """, {
                    "expert_id": expert_id,
                    "area": area
                })

            # Create method relationships (unchanged)
            for method in semantic_data['methods']:
                if not method:
                    continue
                    
                session.run("""
                    MERGE (m:Method {name: $method})
                    MERGE (e:Expert {id: $expert_id})-[r:USES_METHOD]->(m)
                    SET r.weight = 0.7,
                        r.last_updated = datetime()
                """, {
                    "expert_id": expert_id,
                    "method": method
                })

            # Create related area relationships - FIXED VERSION
            for related in semantic_data['related']:
                if not related:
                    continue
                    
                session.run("""
                    MERGE (ra:RelatedArea {name: $related})
                    MERGE (e:Expert {id: $expert_id})-[rel:RELATED_TO]->(ra)
                    SET rel.weight = 0.5,
                        rel.last_updated = datetime()
                """, {
                    "expert_id": expert_id,
                    "related": related
                })

        except Exception as e:
            logger.error(f"Error creating semantic relationships for expert {expert_id}: {e}")
            raise

    # Modified method to be asynchronous
    async def initialize_graph(self):
        """Initialize the graph with experts and their relationships"""
        try:
            # Create indexes first (unchanged)
            self._create_indexes()
            
            # Fetch experts data (unchanged)
            experts_data = self._fetch_experts_data()
            
            if not experts_data:
                logger.warning("No experts data found to process")
                return False

            # Process each expert (unchanged)
            with self._neo4j_driver.session() as session:
                for expert_data in experts_data:
                    try:
                        self.create_expert_node(session, expert_data)
                    except Exception as e:
                        logger.error(f"Error processing expert data: {e}")
                        continue

                # Add adaptive relationships based on historical data (unchanged)
                search_patterns, message_patterns = self._process_historical_data()
                self._create_adaptive_relationships(session, search_patterns, message_patterns)
                
                # NEW: Create interest-based relationships from user_topic_interests
                try:
                    interest_relationships = self._fetch_user_interest_relationships()
                    self._create_interest_relationships(session, interest_relationships)
                    logger.info("Added interest-based relationships from user topic interactions")
                except Exception as e:
                    logger.error(f"Error creating interest relationships: {e}")

            logger.info("Graph initialization complete with adaptive and interest-based relationships!")
            return True

        except Exception as e:
            logger.error(f"Graph initialization failed: {e}")
            return False
                

    def _fetch_user_interest_relationships(self):
        """
        Fetch interest relationship patterns from user_topic_interests table.
        This helps build connections between experts based on user interest patterns.
        """
        conn = None
        try:
            conn = DatabaseConnectionManager.get_postgres_connection()
            cur = conn.cursor()
            
            # Query to find experts that share similar interests based on user interaction patterns
            cur.execute("""
                WITH user_topics AS (
                    -- Get users with their top topics by interaction count
                    SELECT 
                        user_id, 
                        topic_key, 
                        topic_type,
                        interaction_count,
                        ROW_NUMBER() OVER(PARTITION BY user_id, topic_type ORDER BY interaction_count DESC) as rank
                    FROM user_topic_interests
                    WHERE topic_type IN ('expert_expertise', 'publication_domain')
                ),
                user_interests AS (
                    -- Select only top 5 interests per user and type
                    SELECT * FROM user_topics WHERE rank <= 5
                ),
                expert_interactions AS (
                    -- Get experts that users have interacted with
                    SELECT 
                        user_id, 
                        content_id as expert_id,
                        COUNT(*) as interaction_count
                    FROM user_interest_logs
                    WHERE interaction_type = 'expert' AND content_id IS NOT NULL
                    GROUP BY user_id, content_id
                ),
                co_occurrence AS (
                    -- Find experts that appear together in user interests
                    SELECT 
                        e1.expert_id as expert1_id,
                        e2.expert_id as expert2_id,
                        COUNT(DISTINCT e1.user_id) as shared_users,
                        SUM(e1.interaction_count + e2.interaction_count) as total_interactions
                    FROM expert_interactions e1
                    JOIN expert_interactions e2 
                        ON e1.user_id = e2.user_id 
                        AND e1.expert_id < e2.expert_id
                    GROUP BY e1.expert_id, e2.expert_id
                    HAVING COUNT(DISTINCT e1.user_id) > 1
                )
                -- Return the final relationship data
                SELECT 
                    expert1_id,
                    expert2_id,
                    shared_users,
                    total_interactions,
                    total_interactions::float / shared_users as avg_interaction_strength
                FROM co_occurrence
                ORDER BY shared_users DESC, total_interactions DESC
                LIMIT 500
            """)
            
            interest_relationships = cur.fetchall()
            logger.info(f"Fetched {len(interest_relationships)} interest-based relationships")
            return interest_relationships
            
        except Exception as e:
            logger.error(f"Error fetching interest relationships: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _create_interest_relationships(self, session, relationships):
        """
        Create relationships between experts based on user interest patterns.
        """
        if not relationships:
            logger.info("No interest relationships to create")
            return
            
        try:
            for expert1_id, expert2_id, shared_users, total_interactions, avg_strength in relationships:
                # Calculate relationship weight - normalize to 0-1 range
                weight = min(avg_strength / 10.0, 1.0)
                
                # Create bidirectional relationship
                session.run("""
                    MATCH (e1:Expert {id: $expert1_id})
                    MATCH (e2:Expert {id: $expert2_id})
                    MERGE (e1)-[r:INTEREST_RELATED]-(e2)
                    SET r.weight = $weight,
                        r.shared_users = $shared_users,
                        r.total_interactions = $total_interactions,
                        r.last_updated = datetime()
                """, {
                    "expert1_id": str(expert1_id),
                    "expert2_id": str(expert2_id),
                    "weight": weight,
                    "shared_users": shared_users,
                    "total_interactions": total_interactions
                })
            
            logger.info(f"Created {len(relationships)} interest-based relationships")
        except Exception as e:
            logger.error(f"Error creating interest relationships: {e}")

    # Non-async version for direct calls
    # Add this to your GraphDatabaseInitializer class to integrate message domains and fields
    def _fetch_message_domain_field_patterns(self):
        """
        Fetch domain and field relationships based on messaging patterns.
        This helps build connections between domains and fields that are frequently mentioned together.
        """
        conn = None
        try:
            # Create connection parameters
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                parsed_url = urlparse(database_url)
                conn_params = {
                    'host': parsed_url.hostname,
                    'port': parsed_url.port,
                    'dbname': parsed_url.path[1:],
                    'user': parsed_url.username,
                    'password': parsed_url.password
                }
            else:
                conn_params = {
                    'host': os.getenv('POSTGRES_HOST', 'postgres'),
                    'port': os.getenv('POSTGRES_PORT', '5432'),
                    'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                    'user': os.getenv('POSTGRES_USER', 'postgres'),
                    'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
                }
            
            conn = psycopg2.connect(**conn_params)
            with conn.cursor() as cur:
                # Find domains that frequently appear together in messages
                cur.execute("""
                    WITH message_domains AS (
                        -- Expand the domains array into rows
                        SELECT 
                            id, 
                            user_id, 
                            expert_id,
                            UNNEST(domains) as domain
                        FROM message_expert_interactions
                        WHERE domains IS NOT NULL AND ARRAY_LENGTH(domains, 1) > 0
                    ),
                    domain_pairs AS (
                        -- Create pairs of domains that appear in the same message
                        SELECT 
                            md1.domain as domain1,
                            md2.domain as domain2,
                            COUNT(*) as co_occurrence
                        FROM message_domains md1
                        JOIN message_domains md2 
                            ON md1.id = md2.id AND md1.domain < md2.domain
                        GROUP BY md1.domain, md2.domain
                        HAVING COUNT(*) > 1
                    )
                    -- Return the domain relationships
                    SELECT 
                        domain1,
                        domain2,
                        co_occurrence,
                        co_occurrence / 
                            (SELECT COUNT(*) FROM message_expert_interactions WHERE domains @> ARRAY[domain1]) as confidence
                    FROM domain_pairs
                    ORDER BY co_occurrence DESC
                    LIMIT 500
                """)
                
                domain_relationships = cur.fetchall()
                logger.info(f"Fetched {len(domain_relationships)} domain co-occurrence relationships")
                
                # Find fields that frequently appear together in messages
                cur.execute("""
                    WITH message_fields AS (
                        -- Expand the fields array into rows
                        SELECT 
                            id, 
                            user_id, 
                            expert_id,
                            UNNEST(fields) as field
                        FROM message_expert_interactions
                        WHERE fields IS NOT NULL AND ARRAY_LENGTH(fields, 1) > 0
                    ),
                    field_pairs AS (
                        -- Create pairs of fields that appear in the same message
                        SELECT 
                            mf1.field as field1,
                            mf2.field as field2,
                            COUNT(*) as co_occurrence
                        FROM message_fields mf1
                        JOIN message_fields mf2 
                            ON mf1.id = mf2.id AND mf1.field < mf2.field
                        GROUP BY mf1.field, mf2.field
                        HAVING COUNT(*) > 1
                    )
                    -- Return the field relationships
                    SELECT 
                        field1,
                        field2,
                        co_occurrence,
                        co_occurrence / 
                            (SELECT COUNT(*) FROM message_expert_interactions WHERE fields @> ARRAY[field1]) as confidence
                    FROM field_pairs
                    ORDER BY co_occurrence DESC
                    LIMIT 500
                """)
                
                field_relationships = cur.fetchall()
                logger.info(f"Fetched {len(field_relationships)} field co-occurrence relationships")
                
                # Find domain-field connections
                cur.execute("""
                    WITH domain_field_pairs AS (
                        -- Create domain-field pairs from messages
                        SELECT 
                            UNNEST(domains) as domain,
                            UNNEST(fields) as field,
                            COUNT(*) as co_occurrence
                        FROM message_expert_interactions
                        WHERE domains IS NOT NULL AND ARRAY_LENGTH(domains, 1) > 0
                        AND fields IS NOT NULL AND ARRAY_LENGTH(fields, 1) > 0
                        GROUP BY domain, field
                        HAVING COUNT(*) > 1
                    )
                    -- Return the domain-field relationships
                    SELECT 
                        domain,
                        field,
                        co_occurrence,
                        co_occurrence / 
                            (SELECT COUNT(*) FROM message_expert_interactions WHERE domains @> ARRAY[domain]) as confidence
                    FROM domain_field_pairs
                    ORDER BY co_occurrence DESC
                    LIMIT 500
                """)
                
                domain_field_relationships = cur.fetchall()
                logger.info(f"Fetched {len(domain_field_relationships)} domain-field relationships")
                
                return domain_relationships, field_relationships, domain_field_relationships
                
        except Exception as e:
            logger.error(f"Error fetching message domain/field patterns: {e}")
            return [], [], []
        finally:
            if conn:
                conn.close()

    def _create_message_based_relationships(self, session, domain_relationships, field_relationships, domain_field_relationships):
        """
        Create relationships in Neo4j based on messaging domain and field patterns.
        
        Args:
            session: Neo4j session
            domain_relationships: List of domain co-occurrence tuples
            field_relationships: List of field co-occurrence tuples
            domain_field_relationships: List of domain-field relationship tuples
        """
        try:
            # Create domain co-occurrence relationships
            for domain1, domain2, co_occurrence, confidence in domain_relationships:
                # Skip empty domains
                if not domain1 or not domain2:
                    continue
                    
                # Calculate weight based on co-occurrence and confidence
                weight = min(confidence * 0.8, 0.9)  # Cap at 0.9 to avoid overriding direct relationships
                
                # Create bidirectional relationship
                session.run("""
                    MERGE (d1:Domain {name: $domain1})
                    MERGE (d2:Domain {name: $domain2})
                    MERGE (d1)-[r:RELATED_TO]-(d2)
                    SET r.weight = $weight,
                        r.co_occurrence = $co_occurrence,
                        r.confidence = $confidence,
                        r.source = 'messaging',
                        r.last_updated = datetime()
                """, {
                    "domain1": domain1,
                    "domain2": domain2,
                    "weight": weight,
                    "co_occurrence": co_occurrence,
                    "confidence": confidence
                })
            
            logger.info(f"Created {len(domain_relationships)} domain co-occurrence relationships")
            
            # Create field co-occurrence relationships
            for field1, field2, co_occurrence, confidence in field_relationships:
                # Skip empty fields
                if not field1 or not field2:
                    continue
                    
                # Calculate weight based on co-occurrence and confidence
                weight = min(confidence * 0.8, 0.9)  # Cap at 0.9 to avoid overriding direct relationships
                
                # Create bidirectional relationship
                session.run("""
                    MERGE (f1:Field {name: $field1})
                    MERGE (f2:Field {name: $field2})
                    MERGE (f1)-[r:RELATED_TO]-(f2)
                    SET r.weight = $weight,
                        r.co_occurrence = $co_occurrence,
                        r.confidence = $confidence,
                        r.source = 'messaging',
                        r.last_updated = datetime()
                """, {
                    "field1": field1,
                    "field2": field2,
                    "weight": weight,
                    "co_occurrence": co_occurrence,
                    "confidence": confidence
                })
            
            logger.info(f"Created {len(field_relationships)} field co-occurrence relationships")
            
            # Create domain-field relationships
            for domain, field, co_occurrence, confidence in domain_field_relationships:
                # Skip empty values
                if not domain or not field:
                    continue
                    
                # Calculate weight based on co-occurrence and confidence
                weight = min(confidence * 0.7, 0.85)  # Cap at 0.85 to avoid overriding direct relationships
                
                # Create relationship
                session.run("""
                    MERGE (d:Domain {name: $domain})
                    MERGE (f:Field {name: $field})
                    MERGE (d)-[r:RELATED_TO_FIELD]->(f)
                    SET r.weight = $weight,
                        r.co_occurrence = $co_occurrence,
                        r.confidence = $confidence,
                        r.source = 'messaging',
                        r.last_updated = datetime()
                """, {
                    "domain": domain,
                    "field": field,
                    "weight": weight,
                    "co_occurrence": co_occurrence,
                    "confidence": confidence
                })
            
            logger.info(f"Created {len(domain_field_relationships)} domain-field relationships")
        
        except Exception as e:
            logger.error(f"Error creating message-based relationships: {e}")
            raise

    # Update the initialize_graph method to include message domain/field pattern analysis
    async def initialize_graph(self):
        """Initialize the graph with experts and their relationships"""
        try:
            # Create indexes first (unchanged)
            self._create_indexes()
            
            # Fetch experts data (unchanged)
            experts_data = self._fetch_experts_data()
            
            if not experts_data:
                logger.warning("No experts data found to process")
                return False

            # Process each expert (unchanged)
            with self._neo4j_driver.session() as session:
                for expert_data in experts_data:
                    try:
                        self.create_expert_node(session, expert_data)
                    except Exception as e:
                        logger.error(f"Error processing expert data: {e}")
                        continue

                # Add adaptive relationships based on historical data (unchanged)
                search_patterns, message_patterns = self._process_historical_data()
                self._create_adaptive_relationships(session, search_patterns, message_patterns)
                
                # NEW: Create interest-based relationships from user_topic_interests
                try:
                    interest_relationships = self._fetch_user_interest_relationships()
                    self._create_interest_relationships(session, interest_relationships)
                    logger.info("Added interest-based relationships from user topic interactions")
                except Exception as e:
                    logger.error(f"Error creating interest relationships: {e}")
                
                # NEW: Create relationships based on message domain/field patterns
                try:
                    domain_relationships, field_relationships, domain_field_relationships = self._fetch_message_domain_field_patterns()
                    self._create_message_based_relationships(
                        session, 
                        domain_relationships, 
                        field_relationships, 
                        domain_field_relationships
                    )
                    logger.info("Added message-based domain and field relationships")
                except Exception as e:
                    logger.error(f"Error creating message-based relationships: {e}")

            logger.info("Graph initialization complete with adaptive, interest-based, and message-based relationships!")
            return True

        except Exception as e:
            logger.error(f"Graph initialization failed: {e}")
            return False

    def close(self):
        """Close the Neo4j driver"""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            logger.info("Neo4j driver closed")