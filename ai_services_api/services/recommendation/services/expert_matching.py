import os
import logging
import json
import time
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

import psycopg2
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('expert_matching.log', encoding='utf-8')  # File logging
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseConnectionManager:
    @staticmethod
    def get_neo4j_driver():
        """Create a connection to Neo4j database with enhanced logging and error handling."""
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        
        try:
            logger.info(f"Attempting Neo4j connection to {neo4j_uri}")
            
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(
                    neo4j_user,
                    os.getenv('NEO4J_PASSWORD')
                )
            )
            
            # Verify connection
            with driver.session() as session:
                session.run("MATCH (n) RETURN 1 LIMIT 1")
            
            logger.info(f"Neo4j connection established successfully for user: {neo4j_user}")
            return driver
        
        except Exception as e:
            logger.error(f"Neo4j Connection Error: Unable to connect to {neo4j_uri}", exc_info=True)
            raise

class ExpertMatchingService:
    def __init__(self, driver=None):
        """
        Initialize ExpertMatchingService with comprehensive logging and optional driver
        
        :param driver: Optional pre-existing Neo4j driver
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # Use provided driver or create a new one
            self._neo4j_driver = driver or DatabaseConnectionManager.get_neo4j_driver()
            self.logger.info("ExpertMatchingService initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize ExpertMatchingService", exc_info=True)
            raise

    async def _log_recommendation_interaction(self, user_id: str, expert_ids: List[str]) -> None:
        """
        Log when experts are recommended to a user for future reference.
        
        Args:
            user_id: The user's identifier
            expert_ids: List of expert IDs that were recommended
        """
        if not expert_ids:
            return
            
        try:
            # Define connection parameters
            conn_params = {}
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
            
            # Connect to database and log interactions
            conn = None
            try:
                conn = psycopg2.connect(**conn_params)
                with conn.cursor() as cur:
                    # Use interaction_type 'expert' for recommendation interactions
                    for i, expert_id in enumerate(expert_ids):
                        # Log each recommendation with position information
                        position = i + 1  # 1-based position
                        
                        cur.execute("""
                            INSERT INTO user_interest_logs 
                                (user_id, session_id, query, interaction_type, content_id, response_quality, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """, (
                            user_id,
                            f"recommendation_{int(time.time())}",  # Generate a session ID
                            f"recommendation_position_{position}",  # Store position information
                            'expert',
                            expert_id,
                            0.5  # Neutral initial quality score
                        ))
                    
                    conn.commit()
                    self.logger.debug(f"Logged {len(expert_ids)} recommendation interactions for user {user_id}")
                    
            except Exception as db_error:
                self.logger.error(f"Database error logging recommendation interactions: {db_error}")
            finally:
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error logging recommendation interactions: {e}")

    async def _get_user_interests(self, user_id: str, limit: int = 5) -> Dict[str, List[str]]:
        """
        Retrieve a user's top interests across different categories.
        
        Args:
            user_id: The user's identifier
            limit: Maximum number of interests per category
            
        Returns:
            Dict mapping interest types to lists of interest topics
        """
        try:
            # Define connection parameters
            conn_params = {}
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
            
            # Initialize empty result
            interests = {
                'publication_topic': [],
                'publication_domain': [],
                'expert_expertise': []
            }
            
            # Connect to database and fetch interests
            conn = None
            try:
                conn = psycopg2.connect(**conn_params)
                with conn.cursor() as cur:
                    # Time-weighted query - more recent and higher engagement scores rank higher
                    query = """
                    SELECT 
                        topic_key, 
                        topic_type,
                        engagement_score * (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - last_interaction)) / 86400.0)) 
                            AS recency_score
                    FROM user_topic_interests
                    WHERE user_id = %s AND topic_type = %s
                    ORDER BY recency_score DESC, interaction_count DESC
                    LIMIT %s
                    """
                    
                    # Execute for each interest type
                    for topic_type in interests.keys():
                        cur.execute(query, (user_id, topic_type, limit))
                        results = cur.fetchall()
                        
                        # Extract topic keys
                        interests[topic_type] = [row[0] for row in results]
                        
                        self.logger.debug(
                            f"Retrieved {len(interests[topic_type])} {topic_type} interests for user {user_id}"
                        )
                
                return interests
                
            except Exception as db_error:
                self.logger.error(f"Database error retrieving user interests: {db_error}")
                return interests
            finally:
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error retrieving user interests: {e}")
            return {
                'publication_topic': [],
                'publication_domain': [],
                'expert_expertise': []
            }

    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top similar experts using balanced multi-factor matching with adaptive patterns
        based on search history, interactions, and user interests
        
        :param user_id: ID of the expert to find recommendations for
        :param limit: Maximum number of recommendations to return
        :return: List of recommended experts
        """
        start_time = datetime.utcnow()
        
        # Comprehensive query logging
        self.logger.info(f"Generating recommendations for user ID: {user_id}, limit: {limit}")
        
        try:
            # ---- NEW ADDITION: Get user interests from user_topic_interests table ----
            user_interests = await self._get_user_interests(user_id)
            self.logger.info(f"Retrieved {len(user_interests)} interests for user {user_id}")
            
            # ---- UNCHANGED: Session creation and debug info ----
            with self._neo4j_driver.session() as session:
                # Verify expert exists and get baseline information
                debug_query = """
                MATCH (e:Expert {id: $expert_id})
                OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c)
                OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f)
                OPTIONAL MATCH (e)-[:WORKS_IN_DOMAIN]->(d)
                RETURN 
                    e.name as name, 
                    e.designation as designation, 
                    e.theme as theme, 
                    e.unit as unit,
                    COUNT(DISTINCT c) as concept_count,
                    COUNT(DISTINCT f) as field_count,
                    COUNT(DISTINCT d) as domain_count
                """
                debug_result = session.run(debug_query, {"expert_id": user_id})
                debug_record = debug_result.single()
                
                if not debug_record:
                    self.logger.warning(f"No expert found with ID: {user_id}")
                    return []
                
                # Log expert details for debugging
                self.logger.info(f"Expert Debug Info: {dict(debug_record)}")
                
                # Check if we have any relationship data to work with
                has_concepts = debug_record.get("concept_count", 0) > 0
                has_fields = debug_record.get("field_count", 0) > 0
                has_domains = debug_record.get("domain_count", 0) > 0
                
                # ---- MODIFIED: Dynamic weights with user interest integration ----
                # Initialize weights with default values
                concept_weight = 0.0
                domain_weight = 0.0
                field_weight = 0.0
                org_weight = 0.3  # Reduced slightly to make room for other weights
                search_weight = 0.2  # Add weight for search patterns
                interest_weight = 0.3  # New weight for user interests
                
                # Dynamic weights based on data availability
                if has_concepts:
                    concept_weight = 0.2
                if has_domains:
                    domain_weight = 0.2
                if has_fields:
                    field_weight = 0.2
                
                # Calculate adjustment to ensure weights sum to a reasonable value
                available_features = sum([1 if x else 0 for x in [has_concepts, has_domains, has_fields]])
                
                # If few features available, increase weight of organizational and behavioral factors
                if available_features <= 1:
                    org_weight = 0.3
                    search_weight = 0.2
                    interest_weight = 0.3
                else:
                    org_weight = 0.1
                    search_weight = 0.1
                    interest_weight = 0.2
                
                # ---- MODIFIED: Enhanced query with user interest integration ----
                interest_clause = ""
                interest_match = ""
                interests_param = {}
                
                # Add interest-based matching if user has interests
                if user_interests:
                    # Build clause for publication topics
                    if user_interests.get('publication_topic'):
                        interest_match += """
                        // Match with user's publication topic interests
                        OPTIONAL MATCH (e2)-[:HAS_CONCEPT]->(tc:Concept)
                        WHERE tc.name IN $publication_topics
                        """
                        interests_param['publication_topics'] = user_interests.get('publication_topic', [])
                        interest_clause += """
                        // Calculate interest overlap with publication topics
                        COUNT(DISTINCT CASE WHEN tc.name IN $publication_topics THEN tc ELSE NULL END) 
                            * $interest_weight as publication_topic_score,
                        """
                    
                    # Build clause for domain interests
                    if user_interests.get('publication_domain'):
                        interest_match += """
                        // Match with user's domain interests
                        OPTIONAL MATCH (e2)-[:WORKS_IN_DOMAIN]->(td:Domain)
                        WHERE td.name IN $publication_domains
                        """
                        interests_param['publication_domains'] = user_interests.get('publication_domain', [])
                        interest_clause += """
                        // Calculate interest overlap with publication domains
                        COUNT(DISTINCT CASE WHEN td.name IN $publication_domains THEN td ELSE NULL END) 
                            * $interest_weight as publication_domain_score,
                        """
                    
                    # Build clause for expertise interests
                    if user_interests.get('expert_expertise'):
                        interest_match += """
                        // Match with user's expertise interests
                        OPTIONAL MATCH (e2)-[:SPECIALIZES_IN]->(te:Field)
                        WHERE te.name IN $expert_expertise
                        """
                        interests_param['expert_expertise'] = user_interests.get('expert_expertise', [])
                        interest_clause += """
                        // Calculate interest overlap with expertise
                        COUNT(DISTINCT CASE WHEN te.name IN $expert_expertise THEN te ELSE NULL END) 
                            * $interest_weight as expertise_score,
                        """
                
                # Combine interest score into final similarity calculation
                interest_total_score = ""
                if interest_clause:
                    interest_total_score = """
                    // Add interest-based scores to total
                    + publication_topic_score 
                    + publication_domain_score
                    + expertise_score
                    """
                
                # Main recommendation query with interest integration
                query = f"""
                MATCH (e1:Expert {{id: $expert_id}})
                MATCH (e2:Expert)
                WHERE e1 <> e2
                AND e2.is_active = true  // Only return active experts
                
                // Core similarity measures
                OPTIONAL MATCH (e1)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(e2)
                OPTIONAL MATCH (e1)-[:WORKS_IN_DOMAIN]->(d:Domain)<-[:WORKS_IN_DOMAIN]-(e2)
                OPTIONAL MATCH (e1)-[:SPECIALIZES_IN]->(f:Field)<-[:SPECIALIZES_IN]-(e2)
                
                // Related concepts
                OPTIONAL MATCH (e1)-[:RESEARCHES_IN]->(ra:ResearchArea)<-[:RESEARCHES_IN]-(e2)
                OPTIONAL MATCH (e1)-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(e2)
                
                // Organizational proximity
                OPTIONAL MATCH (e1)-[:BELONGS_TO_THEME]->(t:Theme)<-[:BELONGS_TO_THEME]-(e2)
                OPTIONAL MATCH (e1)-[:BELONGS_TO_UNIT]->(u:Unit)<-[:BELONGS_TO_UNIT]-(e2)
                
                // Search patterns
                OPTIONAL MATCH (e1)-[fsc:FREQUENTLY_SEARCHED_WITH]-(e2)
                
                // Direct interactions
                OPTIONAL MATCH (e1)-[int:INTERACTS_WITH]->(e2)
                
                {interest_match}
                
                WITH e1, e2, c, d, f, ra, m, fsc, int,
                    CASE WHEN e1.theme IS NOT NULL AND e1.theme = e2.theme THEN 1.0 ELSE 0.0 END as same_theme,
                    CASE WHEN e1.unit IS NOT NULL AND e1.unit = e2.unit THEN 1.0 ELSE 0.0 END as same_unit
                    {", tc, td, te" if interest_match else ""}
                
                // Collect all relevant data
                WITH e1, e2, 
                    // Semantic overlap
                    COUNT(DISTINCT c) as concept_count,
                    COUNT(DISTINCT d) as domain_count,
                    COUNT(DISTINCT f) as field_count,
                    COUNT(DISTINCT ra) as area_count,
                    COUNT(DISTINCT m) as method_count,
                    
                    // Organizational factors
                    same_theme, same_unit,
                    
                    // Search patterns - weight adjusted based on frequency
                    COALESCE(fsc.weight, 0.0) * $search_weight as search_similarity,
                    
                    // Interaction patterns
                    COALESCE(int.weight, 0.0) * $search_weight as interaction_similarity,
                    
                    {interest_clause if interest_clause else ""}
                    
                    // Collect details for explanation
                    COLLECT(DISTINCT COALESCE(c.name, '')) as shared_concepts,
                    COLLECT(DISTINCT COALESCE(d.name, '')) as shared_domains,
                    COLLECT(DISTINCT COALESCE(f.name, '')) as shared_fields,
                    COLLECT(DISTINCT COALESCE(ra.name, '')) as shared_areas,
                    COLLECT(DISTINCT COALESCE(m.name, '')) as shared_methods
                
                // Calculate overall score with all factors
                WITH e2, 
                    (concept_count * $concept_weight + 
                    domain_count * $domain_weight + 
                    field_count * $field_weight + 
                    area_count * 0.1 +
                    method_count * 0.1 +
                    same_theme * $org_weight + 
                    same_unit * $org_weight +
                    search_similarity +
                    interaction_similarity
                    {interest_total_score}) as similarity_score,
                    
                    // Include search pattern strength
                    search_similarity,
                    interaction_similarity,
                    
                    // Collect for explanation
                    shared_concepts,
                    shared_domains,
                    shared_fields,
                    shared_areas,
                    shared_methods,
                    same_theme,
                    same_unit
                
                // Calculate result data for each expert
                WITH e2, similarity_score, search_similarity, interaction_similarity,
                    shared_concepts, shared_domains, shared_fields, shared_areas, shared_methods, 
                    same_theme, same_unit
                
                // Fix to ensure uniqueness by ID
                WITH DISTINCT e2.id as expert_id, 
                    e2.name as expert_name,
                    e2.designation as expert_designation,
                    e2.theme as expert_theme,
                    e2.unit as expert_unit,
                    similarity_score,
                    search_similarity, interaction_similarity,
                    shared_concepts, shared_domains, shared_fields, shared_areas, shared_methods,
                    same_theme, same_unit
                
                // Put the results in the expected format with ordering
                RETURN {{
                    id: expert_id,
                    name: expert_name,
                    designation: expert_designation,
                    theme: expert_theme,
                    unit: expert_unit,
                    match_details: {{
                        shared_concepts: [c IN shared_concepts WHERE c <> ''],
                        shared_domains: [d IN shared_domains WHERE d <> ''],
                        shared_fields: [f IN shared_fields WHERE f <> ''],
                        shared_research_areas: [ra IN shared_areas WHERE ra <> ''],
                        shared_methods: [m IN shared_methods WHERE m <> '']
                    }},
                    match_reason: CASE 
                        WHEN search_similarity > 0 THEN 'Frequently searched together'
                        WHEN interaction_similarity > 0 THEN 'Previous interaction'
                        WHEN size([c IN shared_concepts WHERE c <> '']) > 0 THEN 'Shared expertise'
                        WHEN size([d IN shared_domains WHERE d <> '']) > 0 THEN 'Shared domains'
                        WHEN size([f IN shared_fields WHERE f <> '']) > 0 THEN 'Shared fields'
                        WHEN same_theme = 1.0 AND same_unit = 1.0 THEN 'Same theme and unit'
                        WHEN same_theme = 1.0 THEN 'Same theme'
                        WHEN same_unit = 1.0 THEN 'Same unit'
                        ELSE 'Potential collaboration'
                    END,
                    similarity_score: similarity_score
                }} as result
                ORDER BY similarity_score DESC
                LIMIT $limit
                """
                
                # Parameters for this specific query
                params = {
                    "expert_id": user_id,
                    "limit": limit,
                    "concept_weight": concept_weight,
                    "domain_weight": domain_weight,
                    "field_weight": field_weight,
                    "org_weight": org_weight,
                    "search_weight": search_weight,
                    "interest_weight": interest_weight,
                    **interests_param  # Add the interest parameters if they exist
                }
                
                # Run recommendations with the appropriate parameters
                result = session.run(query, params)
                similar_experts = [record["result"] for record in result]
                
                # Additional deduplication step - ensure unique expert IDs in the result
                unique_experts = []
                seen_ids = set()
                
                for expert in similar_experts:
                    if expert['id'] not in seen_ids:
                        seen_ids.add(expert['id'])
                        unique_experts.append(expert)
                
                # Log if we had to remove any duplicates
                if len(similar_experts) != len(unique_experts):
                    self.logger.warning(
                        f"Removed {len(similar_experts) - len(unique_experts)} duplicate experts "
                        f"that weren't caught by the DISTINCT clause"
                    )
                
                # Limit to the requested number if we have more after deduplication
                unique_experts = unique_experts[:limit]
                
                # ---- NEW ADDITION: Log interactions after getting recommendations ----
                # Log this recommendation interaction for future reference
                await self._log_recommendation_interaction(user_id, [expert['id'] for expert in unique_experts])
                
                # Performance and result logging
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                self.logger.info(
                    f"Recommendation generation for user {user_id}: "
                    f"Found {len(unique_experts)} unique experts, "
                    f"Process time: {process_time:.2f} seconds"
                )
                
                # Log weights used for diagnosis
                self.logger.debug(
                    f"Match weights: concepts={concept_weight}, domains={domain_weight}, "
                    f"fields={field_weight}, org={org_weight}, search={search_weight}, "
                    f"interest={interest_weight}, "
                    f"available_features={available_features}"
                )
                
                return unique_experts
                    
        except Exception as e:
            self.logger.error(
                f"Error finding similar experts for user {user_id}: {str(e)}", 
                exc_info=True
            )
            return []

    def close(self):
        """Close database connections with logging"""
        try:
            if self._neo4j_driver:
                self._neo4j_driver.close()
                self.logger.info("Neo4j connection closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)