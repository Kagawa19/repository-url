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

    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top similar experts using balanced multi-factor matching with adaptive patterns
        based on search history and interactions
        
        :param user_id: ID of the expert to find recommendations for
        :param limit: Maximum number of recommendations to return
        :return: List of recommended experts
        """
        start_time = datetime.utcnow()
        
        # Comprehensive query logging
        self.logger.info(f"Generating recommendations for user ID: {user_id}, limit: {limit}")
        
        try:
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
                
                # Initialize weights with default values
                concept_weight = 0.0
                domain_weight = 0.0
                field_weight = 0.0
                org_weight = 0.3  # Reduced slightly to make room for search weight
                search_weight = 0.2  # Add weight for search patterns
                
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
                    search_weight = 0.3
                else:
                    org_weight = 0.1
                    search_weight = 0.2
                
                # Modified query to ensure uniqueness of recommendations
                query = """
                MATCH (e1:Expert {id: $expert_id})
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
                
                WITH e1, e2, c, d, f, ra, m, fsc, int,
                    CASE WHEN e1.theme IS NOT NULL AND e1.theme = e2.theme THEN 1.0 ELSE 0.0 END as same_theme,
                    CASE WHEN e1.unit IS NOT NULL AND e1.unit = e2.unit THEN 1.0 ELSE 0.0 END as same_unit
                
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
                    interaction_similarity) as similarity_score,
                    
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
                
                // Always return top results regardless of score, ensuring unique experts by ID
                RETURN DISTINCT {
                    id: e2.id,
                    name: e2.name,
                    designation: e2.designation,
                    theme: e2.theme,
                    unit: e2.unit,
                    match_details: {
                        shared_concepts: [c IN shared_concepts WHERE c <> ''],
                        shared_domains: [d IN shared_domains WHERE d <> ''],
                        shared_fields: [f IN shared_fields WHERE f <> ''],
                        shared_research_areas: [ra IN shared_areas WHERE ra <> ''],
                        shared_methods: [m IN shared_methods WHERE m <> '']
                    },
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
                } as result
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
                    "search_weight": search_weight
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
                
                # Limit to the requested number if we have more after deduplication
                unique_experts = unique_experts[:limit]
                
                # Performance and result logging
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                self.logger.info(
                    f"Recommendation generation for user {user_id}: "
                    f"Found {len(unique_experts)} unique experts from {len(similar_experts)} initial matches, "
                    f"Process time: {process_time:.2f} seconds"
                )
                
                # Log weights used for diagnosis
                self.logger.debug(
                    f"Match weights: concepts={concept_weight}, domains={domain_weight}, "
                    f"fields={field_weight}, org={org_weight}, search={search_weight}, "
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