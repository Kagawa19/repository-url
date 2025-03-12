#!/usr/bin/env python3
import os
import csv
import ast
import json
import psycopg2
import psycopg2.extras
import logging
from dotenv import load_dotenv
from typing import Any, Dict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
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

class ResourceImporter:
    def __init__(self, csv_path='resources.csv'):
        self.csv_path = csv_path
        self.connection = None
        self.cursor = None
        
        # Tracking variables
        self.total_resources = 0
        self.inserted_resources = 0
        self.skipped_resources = 0
        self.duplicate_resources = 0

    def connect_to_database(self):
        """Establish a database connection"""
        try:
            self.connection = psycopg2.connect(**DB_PARAMS)
            # Ensure each transaction is independent
            self.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.connection.cursor()
        except psycopg2.Error as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

    def close_database_connection(self):
        """Close database connection"""
        if self.connection:
            if self.cursor:
                self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed.")

    def safe_parse(self, value: Any) -> Any:
        """
        Safely parse different string representations
        
        Args:
            value (Any): Value to parse
        
        Returns:
            Parsed value
        """
        # If already a list or dict, return as-is
        if isinstance(value, (list, dict)):
            return value
        
        # Handle empty or whitespace values
        if not value or value in ['', '[]', '{}', 'None', 'none']:
            return None
        
        # Handle tuple-like summaries
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            value = value.strip('()')
        
        # Try parsing as a literal (handles dict-like strings)
        try:
            # Replace single quotes with double quotes for JSON compatibility
            cleaned_value = value.replace("'", '"')
            # Try JSON parsing first
            parsed = json.loads(cleaned_value)
            return parsed
        except (json.JSONDecodeError, TypeError):
            try:
                # Fallback to ast literal evaluation
                parsed = ast.literal_eval(value)
                return parsed
            except (ValueError, SyntaxError):
                # If all parsing fails, return original value or None
                return value if value.strip() else None

    def check_resource_exists(self, doi=None, title=None):
        """
        Check if a resource already exists in the database
        
        Args:
            doi (str): DOI of the resource
            title (str): Title of the resource
        
        Returns:
            bool: True if resource exists, False otherwise
        """
        try:
            # Check by DOI first (if available)
            if doi and doi.strip():
                self.cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM resources_resource WHERE doi = %s)", 
                    (doi,)
                )
                if self.cursor.fetchone()[0]:
                    return True
            
            # If no DOI or DOI check failed, check by title
            if title and title.strip():
                self.cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM resources_resource WHERE title = %s)", 
                    (title,)
                )
                return self.cursor.fetchone()[0]
            
            return False
        except psycopg2.Error as e:
            logger.error(f"Error checking resource existence: {e}")
            return False

    def import_resources(self):
        """
        Import resources from CSV to database
        """
        try:
            # Establish database connection
            self.connect_to_database()
            
            # Open CSV file
            with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
                # Use DictReader to access columns by name
                reader = csv.DictReader(csvfile)
                
                # Track total resources
                resources = list(reader)
                self.total_resources = len(resources)
                
                logger.info(f"Total resources in CSV: {self.total_resources}")
                
                # Process each resource
                for resource in resources:
                    try:
                        # Check if resource already exists
                        if self.check_resource_exists(
                            doi=resource.get('doi'), 
                            title=resource.get('title')
                        ):
                            self.duplicate_resources += 1
                            continue
                        
                        # Clean summary
                        summary = resource.get('summary', '')
                        if isinstance(summary, tuple):
                            summary = str(summary)
                        if summary and summary.startswith('(') and summary.endswith(')'):
                            summary = summary.strip('()')
                        
                        # Prepare insert values with robust parsing
                        insert_values = (
                            resource.get('id'),  # id
                            resource.get('doi'),  # doi
                            resource.get('title'),  # title
                            resource.get('abstract'),  # abstract
                            summary,  # summary
                            self.safe_parse(resource.get('domains', '[]')) or [],  # domains
                            self.safe_parse(resource.get('topics', '{}')) or {},  # topics
                            resource.get('description'),  # description
                            resource.get('expert_id'),  # expert_id
                            resource.get('type'),  # type
                            self.safe_parse(resource.get('subtitles', '{}')) or {},  # subtitles
                            self.safe_parse(resource.get('publishers', '{}')) or {},  # publishers
                            resource.get('collection'),  # collection
                            resource.get('date_issue'),  # date_issue
                            resource.get('citation'),  # citation
                            resource.get('language'),  # language
                            self.safe_parse(resource.get('identifiers', '{}')) or {},  # identifiers
                            resource.get('created_at'),  # created_at
                            resource.get('updated_at'),  # updated_at
                            resource.get('source'),  # source
                            self.safe_parse(resource.get('authors', '[]')) or [],  # authors
                            resource.get('publication_year'),  # publication_year
                            resource.get('field'),  # field
                            resource.get('subfield')  # subfield
                        )
                        
                        # Execute insert
                        self.cursor.execute("""
                            INSERT INTO resources_resource (
                                id, doi, title, abstract, summary, domains, topics, 
                                description, expert_id, type, subtitles, publishers, 
                                collection, date_issue, citation, language, identifiers, 
                                created_at, updated_at, source, authors, publication_year, 
                                field, subfield
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                     %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, insert_values)
                        
                        # Increment inserted resources if insertion was successful
                        if self.cursor.rowcount > 0:
                            self.inserted_resources += 1
                        else:
                            self.skipped_resources += 1
                    
                    except Exception as row_error:
                        logger.error(f"Error processing resource: {row_error}")
                        logger.error(f"Problematic resource: {resource}")
                        self.skipped_resources += 1
        
        except Exception as e:
            logger.error(f"Error importing resources: {e}")
        
        finally:
            # Close database connection
            self.close_database_connection()
            
            # Log summary
            logger.info("Import Summary:")
            logger.info(f"Total Resources: {self.total_resources}")
            logger.info(f"Inserted Resources: {self.inserted_resources}")
            logger.info(f"Duplicate Resources: {self.duplicate_resources}")
            logger.info(f"Skipped Resources: {self.skipped_resources}")

def main():
    """Main function to run the resource importer"""
    importer = ResourceImporter()
    importer.import_resources()

if __name__ == "__main__":
    main()