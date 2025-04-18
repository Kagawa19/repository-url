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

    async def _get_additional_recommendations(
        self,
        session,
        user_id: str,
        count: int,
        exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get additional recommendations when primary methods don't return enough.
        Uses a broader matching strategy with organizational factors and popularity.
        
        Args:
            session: Neo4j session
            user_id: User ID to get recommendations for
            count: Number of additional recommendations needed
            exclude_ids: List of expert IDs to exclude
            
        Returns:
            List of additional recommended experts
        """
        if count <= 0:
            return []
            
        exclude_ids = exclude_ids or []
        self.logger.info(f"Finding {count} additional recommendations for user {user_id}")
        
        try:
            # Get the user's theme and unit if available
            user_query = """
            MATCH (e:Expert {id: $user_id})
            RETURN e.theme as theme, e.unit as unit
            """
            
            user_result = session.run(user_query, {"user_id": user_id})
            user_record = user_result.single()
            
            theme = user_record.get("theme") if user_record else None
            unit = user_record.get("unit") if user_record else None
            
            # Build a query that prioritizes organizational similarity and popularity
            query = """
            MATCH (e:Expert)
            WHERE e.is_active = true
            AND NOT e.id IN $exclude_ids
            AND e.id <> $user_id
            
            // Calculate organizational similarity
            WITH e, 
                CASE 
                    WHEN e.theme = $theme AND e.unit = $unit THEN 1.0
                    WHEN e.theme = $theme THEN 0.7
                    WHEN e.unit = $unit THEN 0.6
                    ELSE 0.0
                END as org_similarity
            
            // Get connection count as popularity measure
            OPTIONAL MATCH (e)-[r]-(other)
            WITH e, org_similarity, COUNT(r) as connection_count
            
            // Get expertise for diversity
            OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c:Concept)
            OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f:Field)
            OPTIONAL MATCH (e)-[:WORKS_IN_DOMAIN]->(d:Domain)
            
            // Collect expertise fields
            WITH e, org_similarity, connection_count,
                COLLECT(DISTINCT c.name) as concepts,
                COLLECT(DISTINCT f.name) as fields,
                COLLECT(DISTINCT d.name) as domains
            
            // Calculate final score combining organization, popularity and expertise diversity
            WITH 
                e.id as expert_id,
                e.name as expert_name,
                e.designation as expert_designation,
                e.theme as expert_theme,
                e.unit as expert_unit,
                org_similarity * 0.5 + 
                (CASE WHEN connection_count > 20 THEN 0.5 ELSE connection_count / 40.0 END) * 0.3 +
                (SIZE(concepts) + SIZE(fields) + SIZE(domains)) / 20.0 * 0.2 as score,
                concepts, fields, domains
                
            // Return in standard format
            RETURN {
                id: expert_id,
                name: expert_name,
                designation: expert_designation,
                theme: expert_theme,
                unit: expert_unit,
                match_details: {
                    shared_concepts: concepts,
                    shared_fields: fields,
                    shared_domains: domains
                },
                match_reason: CASE
                    WHEN expert_theme = $theme AND expert_unit = $unit THEN 'Same theme and unit'
                    WHEN expert_theme = $theme THEN 'Same theme'
                    WHEN expert_unit = $unit THEN 'Same unit'
                    ELSE 'Additional recommendation'
                END,
                similarity_score: score
            } as result
            ORDER BY score DESC
            LIMIT $count
            """
            
            params = {
                "user_id": user_id,
                "exclude_ids": exclude_ids,
                "theme": theme,
                "unit": unit,
                "count": count
            }
            
            result = session.run(query, params)
            additional_experts = [record["result"] for record in result]
            
            self.logger.info(f"Found {len(additional_experts)} additional recommendations")
            return additional_experts
            
        except Exception as e:
            self.logger.error(f"Error finding additional recommendations: {str(e)}", exc_info=True)
            return []

    async def _get_diverse_experts(
        self,
        session,
        count: int,
        exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find diverse experts from different fields and themes to provide variety.
        
        Args:
            session: Neo4j session
            count: Number of diverse experts needed
            exclude_ids: List of expert IDs to exclude
            
        Returns:
            List of diverse recommended experts
        """
        if count <= 0:
            return []
            
        exclude_ids = exclude_ids or []
        self.logger.info(f"Finding {count} diverse experts")
        
        try:
            # This query focuses on finding experts from diverse fields
            # It groups experts by their themes and fields, then selects representatives
            query = """
            // Match active experts not already recommended
            MATCH (e:Expert)
            WHERE e.is_active = true
            AND NOT e.id IN $exclude_ids
            
            // Get their themes and fields for diversity
            WITH e, e.theme as theme
            OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f:Field)
            OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c:Concept)
            
            // Group by theme and collect expertise
            WITH e, theme, 
                COLLECT(DISTINCT f.name) as fields,
                COLLECT(DISTINCT c.name) as concepts
                
            // Calculate a diversity score based on field coverage
            WITH e, theme, fields, concepts,
                SIZE(fields) + SIZE(concepts) as expertise_count
            
            // Find experts with most diverse expertise
            ORDER BY expertise_count DESC
            
            // Get complete expert data for return
            WITH e.id as expert_id,
                e.name as expert_name,
                e.designation as expert_designation,
                theme as expert_theme,
                e.unit as expert_unit,
                fields, concepts,
                expertise_count
                
            // Return in standard format
            RETURN {
                id: expert_id,
                name: expert_name,
                designation: expert_designation,
                theme: expert_theme,
                unit: expert_unit,
                match_details: {
                    shared_concepts: concepts,
                    shared_fields: fields
                },
                match_reason: 'Diverse expertise',
                similarity_score: expertise_count / 20.0
            } as result
            LIMIT $count
            """
            
            params = {
                "exclude_ids": exclude_ids,
                "count": count
            }
            
            result = session.run(query, params)
            diverse_experts = [record["result"] for record in result]
            
            self.logger.info(f"Found {len(diverse_experts)} diverse experts")
            return diverse_experts
            
        except Exception as e:
            self.logger.error(f"Error finding diverse experts: {str(e)}", exc_info=True)
            return []

    async def _get_fallback_recommendations(
        self,
        session,
        count: int,
        exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fallback recommendations when all other methods fail.
        Simply returns the most connected active experts.
        
        Args:
            session: Neo4j session
            count: Number of fallback recommendations needed
            exclude_ids: List of expert IDs to exclude
            
        Returns:
            List of fallback recommended experts
        """
        if count <= 0:
            return []
            
        exclude_ids = exclude_ids or []
        self.logger.info(f"Using fallback to find {count} experts")
        
        try:
            # The simplest possible query - just get active experts
            # sorted by their connection count
            query = """
            // Match active experts
            MATCH (e:Expert)
            WHERE e.is_active = true
            AND NOT e.id IN $exclude_ids
            
            // Count their connections as a basic popularity metric
            OPTIONAL MATCH (e)-[r]-()
            WITH e, COUNT(r) as connection_count
            
            // Get expert information
            WITH e.id as expert_id,
                e.name as expert_name,
                e.designation as expert_designation,
                e.theme as expert_theme,
                e.unit as expert_unit,
                connection_count
                
            // Sort by connection count
            ORDER BY connection_count DESC
            
            // Return in standard format
            RETURN {
                id: expert_id,
                name: expert_name,
                designation: expert_designation,
                theme: expert_theme,
                unit: expert_unit,
                match_details: {},
                match_reason: 'Popular expert',
                similarity_score: 0.1
            } as result
            LIMIT $count
            """
            
            params = {
                "exclude_ids": exclude_ids,
                "count": count
            }
            
            result = session.run(query, params)
            fallback_experts = [record["result"] for record in result]
            
            self.logger.info(f"Found {len(fallback_experts)} fallback experts")
            
            # If we still don't have enough (which would be very unlikely),
            # create synthetic placeholder experts
            if len(fallback_experts) < count:
                self.logger.warning(f"Unable to find enough real experts, creating placeholders")
                
                placeholder_count = count - len(fallback_experts)
                placeholder_experts = []
                
                for i in range(placeholder_count):
                    placeholder_experts.append({
                        "id": f"placeholder_{i}",
                        "name": f"Expert {i+1}",
                        "designation": "Recommendation Unavailable",
                        "theme": "",
                        "unit": "",
                        "match_details": {},
                        "match_reason": "Placeholder recommendation",
                        "similarity_score": 0.01
                    })
                
                fallback_experts.extend(placeholder_experts)
            
            return fallback_experts
            
        except Exception as e:
            self.logger.error(f"Error in fallback recommendations: {str(e)}", exc_info=True)
            
            # If even the database query fails, create complete placeholder recommendations
            self.logger.warning("Creating emergency placeholder recommendations")
            
            placeholders = []
            for i in range(count):
                placeholders.append({
                    "id": f"emergency_{i}",
                    "name": f"Expert {i+1}",
                    "designation": "Recommendation Unavailable",
                    "theme": "",
                    "unit": "",
                    "match_details": {},
                    "match_reason": "System is currently unavailable",
                    "similarity_score": 0.0
                })
                
            return placeholders

    async def _get_cold_start_recommendations(
        self, 
        session, 
        limit: int = 5, 
        theme: str = None, 
        unit: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for new users with no history or minimal data.
        Uses a combination of organizational structure and popular experts.
        Always returns exactly 5 recommendations.
        
        Args:
            session: Neo4j session
            limit: Maximum number of recommendations to return (fixed at 5)
            theme: User's theme if available
            unit: User's unit if available
            
        Returns:
            List of exactly 5 recommended experts
        """
        self.logger.info("Generating cold-start recommendations")
        limit = 5  # Always use exactly 5
        
        try:
            # Build the match clause based on available organizational data
            org_match = ""
            params = {"limit": limit * 2}  # Request more than needed to ensure we have enough
            
            if theme:
                org_match += "MATCH (t:Theme {name: $theme}) "
                org_match += "MATCH (e:Expert)-[:BELONGS_TO_THEME]->(t) "
                params["theme"] = theme
            
            if unit:
                if theme:
                    org_match += "OPTIONAL "
                else:
                    org_match += "MATCH "
                org_match += "MATCH (u:Unit {name: $unit}) "
                org_match += "MATCH (e:Expert)-[:BELONGS_TO_UNIT]->(u) "
                params["unit"] = unit
            
            # If no organizational info is available, use a general approach
            if not org_match:
                # Find experts based on popularity and diversity
                query = """
                // Match all active experts
                MATCH (e:Expert)
                WHERE e.is_active = true
                
                // Find how connected they are as a proxy for popularity/relevance
                OPTIONAL MATCH (e)-[r]-(other)
                
                // Group and count their connections
                WITH e, COUNT(r) as connection_count
                
                // Get their expertise areas for diversity
                OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c:Concept)
                OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f:Field)
                
                // Get data to return
                WITH 
                    e.id as expert_id,
                    e.name as expert_name,
                    e.designation as expert_designation,
                    e.theme as expert_theme,
                    e.unit as expert_unit,
                    connection_count,
                    COLLECT(DISTINCT c.name) as concepts,
                    COLLECT(DISTINCT f.name) as fields
                
                // Calculate a diversity-weighted popularity score
                WITH 
                    expert_id, expert_name, expert_designation, expert_theme, expert_unit,
                    connection_count * (0.3 + 0.7 * (SIZE(concepts) + SIZE(fields)) / 10.0) as popularity_score,
                    concepts, fields
                
                // Return in the standard format
                RETURN {
                    id: expert_id,
                    name: expert_name,
                    designation: expert_designation,
                    theme: expert_theme,
                    unit: expert_unit,
                    match_details: {
                        shared_concepts: concepts,
                        shared_fields: fields
                    },
                    match_reason: 'Popular expert',
                    similarity_score: popularity_score
                } as result
                ORDER BY popularity_score DESC
                LIMIT $limit
                """
            else:
                # With organizational info, find experts in the same org units
                query = f"""
                // Start with organizational matches
                {org_match}
                WHERE e.is_active = true
                
                // Avoid self-matching
                WITH DISTINCT e
                
                // Find how connected they are as a proxy for popularity/relevance
                OPTIONAL MATCH (e)-[r]-(other)
                
                // Group and count their connections
                WITH e, COUNT(r) as connection_count
                
                // Get their expertise areas for later explanation
                OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c:Concept)
                OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f:Field)
                
                // Get data to return
                WITH 
                    e.id as expert_id,
                    e.name as expert_name,
                    e.designation as expert_designation,
                    e.theme as expert_theme,
                    e.unit as expert_unit,
                    connection_count,
                    COLLECT(DISTINCT c.name) as concepts,
                    COLLECT(DISTINCT f.name) as fields
                
                // Calculate an organizational relevance score
                WITH 
                    expert_id, expert_name, expert_designation, expert_theme, expert_unit,
                    CASE 
                        WHEN expert_theme = $theme AND expert_unit = $unit THEN 1.0
                        WHEN expert_theme = $theme THEN 0.8
                        WHEN expert_unit = $unit THEN 0.7
                        ELSE 0.5
                    END * (0.5 + 0.5 * connection_count / 10.0) as org_score,
                    concepts, fields
                
                // Return in the standard format
                RETURN {
                    id: expert_id,
                    name: expert_name,
                    designation: expert_designation,
                    theme: expert_theme,
                    unit: expert_unit,
                    match_details: {
                        shared_concepts: concepts,
                        shared_fields: fields
                    },
                    match_reason: CASE
                        WHEN expert_theme = $theme AND expert_unit = $unit THEN 'Same theme and unit'
                        WHEN expert_theme = $theme THEN 'Same theme'
                        WHEN expert_unit = $unit THEN 'Same unit'
                        ELSE 'Organizational connection'
                    END,
                    similarity_score: org_score
                } as result
                ORDER BY org_score DESC
                LIMIT $limit
                """
            
            # Execute the query
            result = session.run(query, params)
            recommended_experts = [record["result"] for record in result]
            
            # Ensure we have unique experts
            unique_experts = []
            seen_ids = set()
            
            for expert in recommended_experts:
                if expert['id'] not in seen_ids:
                    seen_ids.add(expert['id'])
                    unique_experts.append(expert)
            
            # NEW: If we don't have enough recommendations, try another approach
            if len(unique_experts) < 5:
                self.logger.info(f"Cold start found only {len(unique_experts)} experts, trying diversity approach")
                
                # Get diverse experts from different fields to supplement
                diversity_experts = await self._get_diverse_experts(
                    session, 
                    5 - len(unique_experts),
                    [expert['id'] for expert in unique_experts]  # Exclude already recommended experts
                )
                
                if diversity_experts:
                    unique_experts.extend(diversity_experts)
            
            # NEW: If still not enough, use the fallback approach
            if len(unique_experts) < 5:
                self.logger.info(f"Still only have {len(unique_experts)} experts, using fallback approach")
                
                fallback_experts = await self._get_fallback_recommendations(
                    session,
                    5 - len(unique_experts),
                    [expert['id'] for expert in unique_experts]  # Exclude already recommended experts
                )
                
                if fallback_experts:
                    unique_experts.extend(fallback_experts)
            
            # Always return exactly 5 experts (or fewer if absolutely impossible)
            final_experts = unique_experts[:5]
            
            self.logger.info(f"Generated {len(final_experts)} cold-start recommendations")
            return final_experts
            
        except Exception as e:
            self.logger.error(f"Error generating cold-start recommendations: {str(e)}", exc_info=True)
            # On error, try the fallback method
            try:
                return await self._get_fallback_recommendations(session, 5)
            except Exception:
                return []

    # This code would be added to the ExpertMatchingService class to enhance the recommendation
# system with expert domains and fields data from message interactions

    async def _analyze_messaging_interactions(self, user_id: str) -> Dict[str, List[str]]:
        """
        Analyze messaging interactions to extract domain and field preferences.
        
        Args:
            user_id: The user's identifier
                
        Returns:
            Dict containing preferred domains and fields based on messaging patterns
        """
        self.logger.info(f"Analyzing messaging interactions for user {user_id}")
        
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
            
            # Initialize results
            messaging_preferences = {
                'preferred_domains': [],
                'preferred_fields': []
            }
            
            # Connect to database
            conn = None
            try:
                conn = psycopg2.connect(**conn_params)
                with conn.cursor() as cur:
                    # Find experts the user has messaged recently
                    cur.execute("""
                        SELECT DISTINCT receiver_id 
                        FROM expert_messages
                        WHERE sender_id = %s
                        AND created_at > NOW() - INTERVAL '30 days'
                        ORDER BY MAX(created_at) DESC
                        LIMIT 10
                    """, (user_id,))
                    
                    messaged_experts = [str(row[0]) for row in cur.fetchall()]
                    
                    if not messaged_experts:
                        self.logger.info(f"No recent messaging activity found for user {user_id}")
                        return messaging_preferences
                    
                    # Get the domains and fields of these experts
                    placeholder = ','.join(['%s'] * len(messaged_experts))
                    cur.execute(f"""
                        SELECT domains, fields 
                        FROM experts_expert
                        WHERE id IN ({placeholder})
                        AND is_active = true
                    """, messaged_experts)
                    
                    # Collect all domains and fields
                    all_domains = []
                    all_fields = []
                    
                    for row in cur.fetchall():
                        domains = row[0] if row[0] else []
                        fields = row[1] if row[1] else []
                        
                        all_domains.extend(domains)
                        all_fields.extend(fields)
                    
                    # Count frequencies
                    domain_frequency = {}
                    for domain in all_domains:
                        domain_frequency[domain] = domain_frequency.get(domain, 0) + 1
                    
                    field_frequency = {}
                    for field in all_fields:
                        field_frequency[field] = field_frequency.get(field, 0) + 1
                    
                    # Sort by frequency and take top items
                    preferred_domains = sorted(domain_frequency.items(), key=lambda x: x[1], reverse=True)
                    preferred_fields = sorted(field_frequency.items(), key=lambda x: x[1], reverse=True)
                    
                    # Return top 5 domains and fields
                    messaging_preferences['preferred_domains'] = [domain for domain, _ in preferred_domains[:5]]
                    messaging_preferences['preferred_fields'] = [field for field, _ in preferred_fields[:5]]
                    
                    self.logger.info(
                        f"Extracted {len(messaging_preferences['preferred_domains'])} domains and "
                        f"{len(messaging_preferences['preferred_fields'])} fields from messaging patterns"
                    )
                    
                    return messaging_preferences
                    
            except Exception as db_error:
                self.logger.error(f"Database error analyzing messaging patterns: {str(db_error)}")
                return messaging_preferences
            finally:
                if conn:
                    conn.close()
                        
        except Exception as e:
            self.logger.error(f"Error analyzing messaging patterns: {str(e)}")
            return {
                'preferred_domains': [],
                'preferred_fields': []
            }

    # Modify the get_recommendations_for_user method to include messaging preferences
    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top similar experts using balanced multi-factor matching with adaptive patterns
        based on search history, interactions, user interests, and messaging patterns
        
        :param user_id: ID of the expert to find recommendations for
        :param limit: Maximum number of recommendations to return
        :return: List of recommended experts (always 5)
        """
        start_time = datetime.utcnow()
        
        # Comprehensive query logging
        self.logger.info(f"Generating recommendations for user ID: {user_id}, limit: {limit}")
        
        try:
            # Get user interests from user_topic_interests table
            user_interests = await self._get_user_interests(user_id)
            self.logger.info(f"Retrieved {sum(len(v) for v in user_interests.values())} interests for user {user_id}")
            
            # Get messaging preferences - THIS IS NEW
            messaging_preferences = await self._analyze_messaging_interactions(user_id)
            self.logger.info(
                f"Retrieved {len(messaging_preferences['preferred_domains'])} domains and "
                f"{len(messaging_preferences['preferred_fields'])} fields from messaging patterns"
            )
            
            # Check if this is a new user (no interests and likely no history)
            is_new_user = all(len(interests) == 0 for interests in user_interests.values())
            has_messaging_preferences = (
                len(messaging_preferences['preferred_domains']) > 0 or 
                len(messaging_preferences['preferred_fields']) > 0
            )
            
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
                    # If user isn't found as an expert, use cold-start recommendations
                    return await self._get_cold_start_recommendations(session, 5)
                
                # Log expert details for debugging
                self.logger.info(f"Expert Debug Info: {dict(debug_record)}")
                
                # Check if we have any relationship data to work with
                has_concepts = debug_record.get("concept_count", 0) > 0
                has_fields = debug_record.get("field_count", 0) > 0
                has_domains = debug_record.get("domain_count", 0) > 0
                
                # For new users or users with minimal data, use the cold start approach,
                # UNLESS they have messaging preferences
                if is_new_user and not (has_concepts or has_fields or has_domains) and not has_messaging_preferences:
                    self.logger.info(f"New user with minimal data detected: {user_id}")
                    return await self._get_cold_start_recommendations(session, 5, theme=debug_record.get("theme"), unit=debug_record.get("unit"))
                
                # Dynamic weights based on data availability
                # Initialize weights with default values
                concept_weight = 0.0
                domain_weight = 0.0
                field_weight = 0.0
                org_weight = 0.25  # Adjusted to make room for messaging weight
                search_weight = 0.15  # Adjusted to make room for messaging weight
                interest_weight = 0.2  # Adjusted to make room for messaging weight
                messaging_weight = 0.2  # NEW: Weight for messaging preferences
                
                # Dynamic weights based on data availability
                if has_concepts:
                    concept_weight = 0.15
                if has_domains:
                    domain_weight = 0.15
                if has_fields:
                    field_weight = 0.15
                
                # Adjust weights if no messaging preferences
                if not has_messaging_preferences:
                    messaging_weight = 0.0
                    # Redistribute weights
                    concept_weight = concept_weight * 1.2 if has_concepts else 0.0
                    domain_weight = domain_weight * 1.2 if has_domains else 0.0
                    field_weight = field_weight * 1.2 if has_fields else 0.0
                    org_weight = 0.3
                    search_weight = 0.2
                    interest_weight = 0.25
                
                # Interest-based matching clauses
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
                
                # NEW: Messaging-based preferences matching
                messaging_clause = ""
                messaging_match = ""
                messaging_param = {}
                
                if has_messaging_preferences:
                    # Build clause for messaging domain preferences
                    if messaging_preferences.get('preferred_domains'):
                        messaging_match += """
                        // Match with user's messaging domain preferences
                        OPTIONAL MATCH (e2)-[:WORKS_IN_DOMAIN]->(md:Domain)
                        WHERE md.name IN $messaging_domains
                        """
                        messaging_param['messaging_domains'] = messaging_preferences.get('preferred_domains', [])
                        messaging_clause += """
                        // Calculate overlap with messaging domain preferences
                        COUNT(DISTINCT CASE WHEN md.name IN $messaging_domains THEN md ELSE NULL END) 
                            * $messaging_weight as messaging_domain_score,
                        """
                    
                    # Build clause for messaging field preferences
                    if messaging_preferences.get('preferred_fields'):
                        messaging_match += """
                        // Match with user's messaging field preferences
                        OPTIONAL MATCH (e2)-[:SPECIALIZES_IN]->(mf:Field)
                        WHERE mf.name IN $messaging_fields
                        """
                        messaging_param['messaging_fields'] = messaging_preferences.get('preferred_fields', [])
                        messaging_clause += """
                        // Calculate overlap with messaging field preferences
                        COUNT(DISTINCT CASE WHEN mf.name IN $messaging_fields THEN mf ELSE NULL END) 
                            * $messaging_weight as messaging_field_score,
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
                
                # NEW: Combine messaging score into final similarity calculation
                messaging_total_score = ""
                if messaging_clause:
                    messaging_total_score = """
                    // Add messaging-based scores to total
                    + messaging_domain_score
                    + messaging_field_score
                    """
                
                # Main recommendation query with interest and messaging integration
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
                
                {messaging_match}
                
                WITH e1, e2, c, d, f, ra, m, fsc, int,
                    CASE WHEN e1.theme IS NOT NULL AND e1.theme = e2.theme THEN 1.0 ELSE 0.0 END as same_theme,
                    CASE WHEN e1.unit IS NOT NULL AND e1.unit = e2.unit THEN 1.0 ELSE 0.0 END as same_unit
                    {", tc, td, te" if interest_match else ""}
                    {", md, mf" if messaging_match else ""}
                
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
                    
                    {messaging_clause if messaging_clause else ""}
                    
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
                    {interest_total_score}
                    {messaging_total_score}) as similarity_score,
                    
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
                
                # Get a higher limit to ensure we have enough experts after deduplication
                adjusted_limit = max(limit * 2, 10)
                
                # Parameters for this specific query
                params = {
                    "expert_id": user_id,
                    "limit": adjusted_limit,  # Using a higher limit to ensure enough results after filtering
                    "concept_weight": concept_weight,
                    "domain_weight": domain_weight,
                    "field_weight": field_weight,
                    "org_weight": org_weight,
                    "search_weight": search_weight,
                    "interest_weight": interest_weight,
                    "messaging_weight": messaging_weight,  # NEW: Added messaging weight parameter
                    **interests_param,  # Add the interest parameters if they exist
                    **messaging_param  # NEW: Add the messaging parameters if they exist
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
                
                # NEW: Check if we have enough recommendations and add more if needed
                if len(unique_experts) < 5:
                    self.logger.info(f"Only found {len(unique_experts)} recommendations, adding more to reach 5")
                    
                    # Get additional experts to reach exactly 5
                    additional_experts = await self._get_additional_recommendations(
                        session, 
                        user_id, 
                        5 - len(unique_experts),
                        [expert['id'] for expert in unique_experts]  # Exclude already recommended experts
                    )
                    
                    if additional_experts:
                        unique_experts.extend(additional_experts)
                        self.logger.info(f"Added {len(additional_experts)} additional experts")
                
                # NEW: Always limit to exactly 5 results
                final_experts = unique_experts[:5]
                
                # Log this recommendation interaction for future reference
                await self._log_recommendation_interaction(user_id, [expert['id'] for expert in final_experts])
                
                # Performance and result logging
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                self.logger.info(
                    f"Recommendation generation for user {user_id}: "
                    f"Returning exactly {len(final_experts)} experts, "
                    f"Process time: {process_time:.2f} seconds"
                )
                
                # Log weights used for diagnosis
                self.logger.debug(
                    f"Match weights: concepts={concept_weight}, domains={domain_weight}, "
                    f"fields={field_weight}, org={org_weight}, search={search_weight}, "
                    f"interest={interest_weight}, messaging={messaging_weight}"
                )
                
                return final_experts
                    
        except Exception as e:
            self.logger.error(
                f"Error finding similar experts for user {user_id}: {str(e)}", 
                exc_info=True
            )
            # NEW: On error, return 5 fallback recommendations
            try:
                with self._neo4j_driver.session() as session:
                    return await self._get_fallback_recommendations(session, 5)
            except Exception as fallback_error:
                self.logger.error(f"Error getting fallback recommendations: {fallback_error}")
                return []

    def close(self):
        """Close database connections with logging"""
        try:
            if self._neo4j_driver:
                self._neo4j_driver.close()
                self.logger.info("Neo4j connection closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)