import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime
from neo4j import AsyncGraphDatabase  # Changed to AsyncGraphDatabase
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

class AsyncDatabaseConnectionManager:
    @staticmethod
    async def get_neo4j_driver():
        """Create an async connection to Neo4j database with enhanced logging and error handling."""
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        
        try:
            logger.info(f"Attempting Neo4j async connection to {neo4j_uri}")
            
            driver = AsyncGraphDatabase.driver(
                neo4j_uri,
                auth=(
                    neo4j_user,
                    os.getenv('NEO4J_PASSWORD')
                )
            )
            
            # Verify connection
            async with driver.session() as session:
                await session.run("MATCH (n) RETURN 1 LIMIT 1")
            
            logger.info(f"Neo4j async connection established successfully for user: {neo4j_user}")
            return driver
        
        except Exception as e:
            logger.error(f"Neo4j Async Connection Error: Unable to connect to {neo4j_uri}", exc_info=True)
            raise

class ExpertMatchingService:
    def __init__(self, driver=None):
        """
        Initialize ExpertMatchingService with comprehensive logging and optional driver
        
        :param driver: Optional pre-existing Neo4j driver
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # Use provided driver or create a new one (the driver creation will be async but init isn't)
            # We'll handle actual connection in the get_recommendations_for_user method
            self._neo4j_driver = driver
            self.driver_provided = driver is not None
            self.logger.info("ExpertMatchingService initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize ExpertMatchingService", exc_info=True)
            raise

    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        limit: int = 5,
        min_score: float = 0.01  # Lower threshold to ensure recommendations
    ) -> List[Dict[str, Any]]:
        """
        Find top similar experts using a more lenient matching approach to ensure 
        results even with sparse data
        
        :param user_id: ID of the expert to find recommendations for
        :param limit: Maximum number of recommendations to return
        :param min_score: Minimum similarity score (set very low to ensure matches)
        :return: List of recommended experts
        """
        start_time = datetime.utcnow()
        
        # Comprehensive query logging
        self.logger.info(f"Generating recommendations for user ID: {user_id}, limit: {limit}")
        
        # If driver wasn't provided, create it now asynchronously
        if not self.driver_provided:
            self._neo4j_driver = await AsyncDatabaseConnectionManager.get_neo4j_driver()
        
        try:
            async with self._neo4j_driver.session() as session:
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
                debug_result = await session.run(debug_query, {"expert_id": user_id})
                debug_record = await debug_result.single()
                
                if not debug_record:
                    self.logger.warning(f"No expert found with ID: {user_id}")
                    # Instead of returning empty, fall back to finding any experts
                    fallback_query = """
                    MATCH (e:Expert)
                    WHERE e.id <> $expert_id AND e.is_active = true
                    RETURN {
                        id: e.id,
                        name: e.name,
                        designation: e.designation,
                        theme: e.theme,
                        unit: e.unit,
                        match_details: {
                            shared_concepts: [],
                            shared_domains: [],
                            shared_fields: [],
                            shared_research_areas: [],
                            shared_methods: [],
                            search_strength: false,
                            interaction_strength: false
                        },
                        match_reason: 'Random match - improve user profile for better matches',
                        similarity_score: 0.01
                    } as result
                    LIMIT $limit
                    """
                    fallback_result = await session.run(fallback_query, {"expert_id": user_id, "limit": limit})
                    fallback_records = await fallback_result.data()
                    fallback_experts = [record["result"] for record in fallback_records]
                    
                    self.logger.info(f"Fallback recommendations used for user {user_id}. Found {len(fallback_experts)} random experts.")
                    return fallback_experts
                
                # Log expert details for debugging
                self.logger.info(f"Expert Debug Info: {dict(debug_record)}")
                
                # Check if we have any relationship data to work with
                has_concepts = debug_record.get("concept_count", 0) > 0
                has_fields = debug_record.get("field_count", 0) > 0
                has_domains = debug_record.get("domain_count", 0) > 0
                
                # More lenient weights - ensure all weights are always positive
                # Even with no data, we'll still get theme/unit matching
                concept_weight = 0.15 if has_concepts else 0.05
                domain_weight = 0.15 if has_domains else 0.05
                field_weight = 0.15 if has_fields else 0.05
                org_weight = 0.2  # Always give decent weight to organizational factors
                search_weight = 0.1  # Reduced slightly from original
                
                # Query is mostly the same, but now with more lenient weights
                query = """
                MATCH (e1:Expert {id: $expert_id})
                MATCH (e2:Expert)
                WHERE e1 <> e2 AND e2.is_active = true
                
                // Basic similarity - just being active is worth something
                WITH e1, e2, 0.001 as base_similarity
                
                // Core similarity measures
                OPTIONAL MATCH (e1)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(e2)
                OPTIONAL MATCH (e1)-[:WORKS_IN_DOMAIN]->(d:Domain)<-[:WORKS_IN_DOMAIN]-(e2)
                OPTIONAL MATCH (e1)-[:SPECIALIZES_IN]->(f:Field)<-[:SPECIALIZES_IN]-(e2)
                
                // Related concepts
                OPTIONAL MATCH (e1)-[:RESEARCHES_IN]->(ra:ResearchArea)<-[:RESEARCHES_IN]-(e2)
                OPTIONAL MATCH (e1)-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(e2)
                
                // Organizational proximity - crucial for sparse data
                OPTIONAL MATCH (e1)-[:BELONGS_TO_THEME]->(t:Theme)<-[:BELONGS_TO_THEME]-(e2)
                OPTIONAL MATCH (e1)-[:BELONGS_TO_UNIT]->(u:Unit)<-[:BELONGS_TO_UNIT]-(e2)
                
                // Search patterns
                OPTIONAL MATCH (e1)-[fsc:FREQUENTLY_SEARCHED_WITH]-(e2)
                
                // Direct interactions
                OPTIONAL MATCH (e1)-[int:INTERACTS_WITH]->(e2)
                
                WITH e1, e2, c, d, f, ra, m, fsc, int, base_similarity,
                    CASE WHEN e1.theme IS NOT NULL AND e1.theme = e2.theme THEN 1.0 ELSE 0.0 END as same_theme,
                    CASE WHEN e1.unit IS NOT NULL AND e1.unit = e2.unit THEN 1.0 ELSE 0.0 END as same_unit,
                    // Add designation similarity - people with same roles might collaborate
                    CASE WHEN e1.designation IS NOT NULL AND e1.designation = e2.designation THEN 0.5 ELSE 0.0 END as same_designation
                
                // Collect all relevant data
                WITH e1, e2, base_similarity,
                    // Semantic overlap
                    COUNT(DISTINCT c) as concept_count,
                    COUNT(DISTINCT d) as domain_count,
                    COUNT(DISTINCT f) as field_count,
                    COUNT(DISTINCT ra) as area_count,
                    COUNT(DISTINCT m) as method_count,
                    
                    // Organizational factors
                    same_theme, same_unit, same_designation,
                    
                    // Search patterns - weight adjusted
                    COALESCE(fsc.weight, 0.0) * $search_weight as search_similarity,
                    
                    // Interaction patterns
                    COALESCE(int.weight, 0.0) * $search_weight as interaction_similarity,
                    
                    // Collect details for explanation
                    COLLECT(DISTINCT COALESCE(c.name, '')) as shared_concepts,
                    COLLECT(DISTINCT COALESCE(d.name, '')) as shared_domains,
                    COLLECT(DISTINCT COALESCE(f.name, '')) as shared_fields,
                    COLLECT(DISTINCT COALESCE(ra.name, '')) as shared_areas,
                    COLLECT(DISTINCT COALESCE(m.name, '')) as shared_methods
                
                // Calculate overall score with all factors - ensure base similarity keeps score positive
                WITH e2, 
                    (base_similarity +
                    concept_count * $concept_weight + 
                    domain_count * $domain_weight + 
                    field_count * $field_weight + 
                    area_count * 0.05 +
                    method_count * 0.05 +
                    same_theme * $org_weight + 
                    same_unit * $org_weight +
                    same_designation * 0.1 +
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
                
                // Return results that meet minimum threshold
                WHERE similarity_score >= $min_score
                
                RETURN {
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
                        shared_methods: [m IN shared_methods WHERE m <> ''],
                        search_strength: search_similarity > 0,
                        interaction_strength: interaction_similarity > 0
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
                        ELSE 'Potential collaboration opportunity'
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
                    "search_weight": search_weight,
                    "min_score": min_score
                }
                
                # Run recommendations with the appropriate parameters
                result = await session.run(query, params)
                
                # Process results for async
                records = await result.data()
                similar_experts = [record["result"] for record in records]
                
                # If we still got no results despite lenient matching, use fallback to random experts
                if not similar_experts:
                    self.logger.warning(f"No matches found even with lenient criteria for user {user_id}. Using fallback.")
                    
                    fallback_query = """
                    MATCH (e:Expert)
                    WHERE e.id <> $expert_id AND e.is_active = true
                    RETURN {
                        id: e.id,
                        name: e.name,
                        designation: e.designation,
                        theme: e.theme,
                        unit: e.unit,
                        match_details: {
                            shared_concepts: [],
                            shared_domains: [],
                            shared_fields: [],
                            shared_research_areas: [],
                            shared_methods: [],
                            search_strength: false,
                            interaction_strength: false
                        },
                        match_reason: 'Random match - improve user profile for better matches',
                        similarity_score: 0.01
                    } as result
                    LIMIT $limit
                    """
                    fallback_result = await session.run(fallback_query, {"expert_id": user_id, "limit": limit})
                    fallback_records = await fallback_result.data()
                    similar_experts = [record["result"] for record in fallback_records]
                
                # Performance and result logging
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                self.logger.info(
                    f"Recommendation generation for user {user_id}: "
                    f"Found {len(similar_experts)} experts, "
                    f"Process time: {process_time:.2f} seconds"
                )
                
                # Log weights used for diagnosis
                self.logger.debug(
                    f"Match weights: concepts={concept_weight}, domains={domain_weight}, "
                    f"fields={field_weight}, org={org_weight}, search={search_weight}"
                )
                
                return similar_experts
                
        except Exception as e:
            self.logger.error(
                f"Error finding similar experts for user {user_id}: {str(e)}", 
                exc_info=True
            )
            # Even on error, try to return some results
            try:
                async with self._neo4j_driver.session() as session:
                    fallback_query = """
                    MATCH (e:Expert) 
                    WHERE e.id <> $expert_id AND e.is_active = true
                    RETURN {
                        id: e.id,
                        name: e.name,
                        designation: e.designation,
                        theme: e.theme,
                        unit: e.unit,
                        match_details: {
                            shared_concepts: [],
                            shared_domains: [],
                            shared_fields: [],
                            shared_research_areas: [],
                            shared_methods: [],
                            search_strength: false,
                            interaction_strength: false
                        },
                        match_reason: 'Error occurred - showing random experts',
                        similarity_score: 0.01
                    } as result
                    LIMIT $limit
                    """
                    fallback_result = await session.run(fallback_query, {"expert_id": user_id, "limit": limit})
                    fallback_records = await fallback_result.data()
                    error_fallback = [record["result"] for record in fallback_records]
                    
                    self.logger.info(f"Returned error fallback recommendations for user {user_id}")
                    return error_fallback
            except Exception:
                self.logger.error("Critical error: Could not provide any recommendations", exc_info=True)
                return []
    async def close(self):
        """Close database connections with logging (async version)"""
        try:
            if not self.driver_provided and self._neo4j_driver:
                await self._neo4j_driver.close()
                self.logger.info("Neo4j connection closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)