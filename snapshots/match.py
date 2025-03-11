import os
import psycopg2
import json
import logging
from typing import List, Dict, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

class ExpertResourceMatcher:
    def __init__(self, db_params: Dict[str, str]):
        """
        Initialize the matcher with database connection parameters.
        
        Args:
            db_params (Dict[str, str]): Database connection parameters
        """
        self.db_params = db_params
        self.connection = None
        self.cursor = None

    def _connect(self):
        """Establish a database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_params)
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to the database")
        except psycopg2.Error as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

    def _close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name for consistent matching.
        
        Args:
            name (str): Name to normalize
        
        Returns:
            str: Normalized name
        """
        # Remove extra spaces, convert to lowercase
        normalized = re.sub(r'\s+', ' ', name.lower().strip())
        
        # Remove common titles and suffixes
        titles = ['dr', 'mr', 'mrs', 'ms', 'prof', 'phd']
        for title in titles:
            normalized = normalized.replace(f"{title} ", '')
        
        return normalized

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """
        Fetch active experts from the database.
        
        Returns:
            List[Dict[str, Any]]: List of experts
        """
        try:
            self.cursor.execute("""
                SELECT 
                    id, 
                    first_name, 
                    last_name, 
                    middle_name,
                    COALESCE(first_name || ' ' || COALESCE(middle_name, '') || ' ' || last_name, 
                             first_name || ' ' || last_name) as full_name
                FROM experts_expert
                WHERE is_active = TRUE
            """)
            experts = [
                {
                    'id': row[0], 
                    'first_name': row[1], 
                    'last_name': row[2], 
                    'middle_name': row[3],
                    'full_name': row[4],
                    'normalized_name': self._normalize_name(row[4])
                } 
                for row in self.cursor.fetchall()
            ]
            logger.info(f"Fetched {len(experts)} active experts")
            return experts
        except psycopg2.Error as e:
            logger.error(f"Error fetching experts: {e}")
            return []

    def fetch_resources(self) -> List[Dict[str, Any]]:
        """
        Fetch resources with authors from the database.
        
        Returns:
            List[Dict[str, Any]]: List of resources with authors
        """
        try:
            self.cursor.execute("""
                SELECT 
                    id, 
                    authors,
                    title
                FROM resources_resource
                WHERE authors IS NOT NULL AND authors != '[]'
            """)
            resources = []
            for row in self.cursor.fetchall():
                resource_id, authors_json, title = row
                
                # Parse authors, handling different possible formats
                try:
                    authors = json.loads(authors_json) if isinstance(authors_json, str) else authors_json
                except (json.JSONDecodeError, TypeError):
                    authors = [authors_json] if authors_json else []
                
                # Normalize author names
                normalized_authors = [
                    self._normalize_name(str(author)) 
                    for author in authors 
                    if author
                ]
                
                resources.append({
                    'id': resource_id,
                    'title': title,
                    'authors': authors,
                    'normalized_authors': normalized_authors
                })
            
            logger.info(f"Fetched {len(resources)} resources with authors")
            return resources
        except psycopg2.Error as e:
            logger.error(f"Error fetching resources: {e}")
            return []

    def match_experts_to_resources(self, experts, resources):
        """
        Match experts to resources based on author names.
        
        Args:
            experts (List[Dict]): List of experts
            resources (List[Dict]): List of resources
        
        Returns:
            Dict[int, List[int]]: Mapping of expert IDs to resource IDs
        """
        # Create expert name lookup
        expert_lookup = {expert['normalized_name']: expert['id'] for expert in experts}
        
        # Track matches
        matches = {}
        match_count = 0
        
        # Match process
        for resource in resources:
            for normalized_author in resource['normalized_authors']:
                if normalized_author in expert_lookup:
                    expert_id = expert_lookup[normalized_author]
                    resource_id = resource['id']
                    
                    # Add to matches
                    if expert_id not in matches:
                        matches[expert_id] = []
                    matches[expert_id].append(resource_id)
                    match_count += 1
        
        logger.info(f"Found {match_count} expert-resource matches")
        return matches

    def store_matches(self, matches):
        """
        Store matches in the expert_resource_links table.
        
        Args:
            matches (Dict[int, List[int]]): Mapping of expert IDs to resource IDs
        """
        try:
            # Track successful and failed matches
            successful_matches = 0
            failed_matches = 0
            
            for expert_id, resource_ids in matches.items():
                for resource_id in resource_ids:
                    try:
                        # Upsert to prevent duplicates
                        self.cursor.execute("""
                            INSERT INTO expert_resource_links 
                            (expert_id, resource_id, confidence_score, created_at)
                            VALUES (%s, %s, 1.0, CURRENT_TIMESTAMP)
                            ON CONFLICT (expert_id, resource_id) DO NOTHING
                        """, (expert_id, resource_id))
                        successful_matches += 1
                    except psycopg2.Error as e:
                        logger.error(f"Error storing match for Expert {expert_id} - Resource {resource_id}: {e}")
                        failed_matches += 1
            
            # Commit the transaction
            self.connection.commit()
            
            logger.info(f"Matches stored: {successful_matches} successful, {failed_matches} failed")
        except psycopg2.Error as e:
            logger.error(f"Transaction error storing matches: {e}")
            self.connection.rollback()

    def run_matching_process(self):
        """
        Run the complete expert-resource matching process.
        """
        try:
            # Establish database connection
            self._connect()
            
            # Fetch experts and resources
            experts = self.fetch_experts()
            resources = self.fetch_resources()
            
            # Perform matching
            matches = self.match_experts_to_resources(experts, resources)
            
            # Store matches
            self.store_matches(matches)
            
            logger.info("Expert-resource matching process completed successfully")
        except Exception as e:
            logger.error(f"Matching process failed: {e}")
        finally:
            # Always close the connection
            self._close()

def main():
    """Main execution function."""
    try:
        # Initialize and run the matcher
        matcher = ExpertResourceMatcher(DB_PARAMS)
        matcher.run_matching_process()
    except Exception as e:
        logger.error(f"Critical error in matching process: {e}")

if __name__ == "__main__":
    main()