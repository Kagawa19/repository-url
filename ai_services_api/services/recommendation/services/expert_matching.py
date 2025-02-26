import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime
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
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
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

    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top similar experts using enhanced semantic matching
        
        :param user_id: ID of the expert to find recommendations for
        :param limit: Maximum number of recommendations to return
        :return: List of recommended experts
        """
        start_time = datetime.utcnow()
        
        # Comprehensive query logging
        self.logger.info(f"Generating recommendations for user ID: {user_id}, limit: {limit}")
        
        try:
            with self._neo4j_driver.session() as session:
                # Comprehensive expert existence and debug query
                debug_query = """
                MATCH (e:Expert {id: $expert_id})
                RETURN 
                    e.name as name, 
                    e.designation as designation, 
                    e.theme as theme, 
                    e.unit as unit,
                    [x in labels(e) | x] as labels
                """
                debug_result = session.run(debug_query, {"expert_id": user_id})
                debug_record = debug_result.single()
                
                if not debug_record:
                    self.logger.warning(f"No expert found with ID: {user_id}")
                    return []
                
                # Log expert details for debugging
                self.logger.info(f"Expert Debug Info: {dict(debug_record)}")
                
                # Enhanced recommendation query with more sophisticated scoring
                query = """
                MATCH (e1:Expert {id: $expert_id})
                MATCH (e2:Expert)
                WHERE e1 <> e2
                
                // Enhanced similarity calculation
                OPTIONAL MATCH (e1)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(e2)
                OPTIONAL MATCH (e1)-[:WORKS_IN_DOMAIN]->(d:Domain)<-[:WORKS_IN_DOMAIN]-(e2)
                OPTIONAL MATCH (e1)-[:SPECIALIZES_IN]->(f:Field)<-[:SPECIALIZES_IN]-(e2)
                OPTIONAL MATCH (e1)-[:BELONGS_TO_THEME]->(t:Theme)<-[:BELONGS_TO_THEME]-(e2)
                OPTIONAL MATCH (e1)-[:BELONGS_TO_UNIT]->(u:Unit)<-[:BELONGS_TO_UNIT]-(e2)
                OPTIONAL MATCH (e1)-[:RESEARCHES_IN]->(ra:ResearchArea)<-[:RESEARCHES_IN]-(e2)
                OPTIONAL MATCH (e1)-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(e2)
                OPTIONAL MATCH (e1)-[:FREQUENTLY_SEARCHED_WITH]-(e2)
                OPTIONAL MATCH (e1)-[:INTERACTS_WITH]->(e2)
                
                WITH e1, e2, 
                    COUNT(DISTINCT c) as concept_count,
                    COUNT(DISTINCT d) as domain_count,
                    COUNT(DISTINCT f) as field_count,
                    COUNT(DISTINCT t) as theme_count,
                    COUNT(DISTINCT u) as unit_count,
                    COUNT(DISTINCT ra) as area_count,
                    COUNT(DISTINCT m) as method_count,
                    COLLECT(DISTINCT COALESCE(c.name, '')) as shared_concepts,
                    COLLECT(DISTINCT COALESCE(d.name, '')) as shared_domains,
                    COLLECT(DISTINCT COALESCE(f.name, '')) as shared_fields,
                    COLLECT(DISTINCT COALESCE(ra.name, '')) as shared_areas,
                    COLLECT(DISTINCT COALESCE(m.name, '')) as shared_methods
                
                // Weighted similarity calculation with more nuanced scoring
                WITH e2, 
                    (concept_count * 0.4 + 
                    domain_count * 0.3 + 
                    field_count * 0.2 + 
                    theme_count * 0.05 + 
                    unit_count * 0.05 +
                    area_count * 0.1 +
                    method_count * 0.1) as similarity_score,
                    shared_concepts,
                    shared_domains,
                    shared_fields,
                    shared_areas,
                    shared_methods
                
                RETURN {
                    id: e2.id,
                    name: e2.name,
                    designation: e2.designation,
                    theme: e2.theme,
                    unit: e2.unit,
                    match_details: {
                        shared_concepts: shared_concepts,
                        shared_domains: shared_domains,
                        shared_fields: shared_fields,
                        shared_research_areas: shared_areas,
                        shared_methods: shared_methods
                    },
                    similarity_score: similarity_score
                } as result
                ORDER BY similarity_score DESC
                LIMIT $limit
                """
                
                # Run recommendations with enhanced parameters
                result = session.run(query, {
                    "expert_id": user_id,
                    "limit": limit
                })
                
                similar_experts = [record["result"] for record in result]
                
                # Performance and result logging
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                self.logger.info(
                    f"Recommendation generation for user {user_id}: "
                    f"Found {len(similar_experts)} experts, "
                    f"Process time: {process_time:.2f} seconds"
                )
                
                return similar_experts
                
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