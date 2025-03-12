#!/usr/bin/env python3
import os
import csv
import ast
import json
import psycopg2
import psycopg2.extras
import logging
from dotenv import load_dotenv

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

    def safe_parse(self, value):
        """
        Safely parse different string representations
        
        Args:
            value (str): Value to parse
        
        Returns:
            Parsed value
        """
        # Handle empty, None, or whitespace values
        if not value or not isinstance(value, str) or value.strip() in ['', '[]', '{}', 'None', 'none', '{}']:
            return None
        
        # Handle summary tuple-like strings
        if value.startswith('(') and value.endswith(')'):
            # Extract content from tuple-like string if possible
            try:
                # Try to parse as a literal tuple
                parsed_tuple = ast.literal_eval(value)
                if isinstance(parsed_tuple, tuple) and len(parsed_tuple) > 0:
                    return parsed_tuple[0]
                return None
            except (ValueError, SyntaxError):
                # If parsing fails, clean up the string manually
                inner_content = value[1:-1].strip()
                if inner_content.startswith('"') and inner_content.endswith('"'):
                    return inner_content[1:-1]
                elif inner_content.startswith("'") and inner_content.endswith("'"):
                    return inner_content[1:-1]
                return None
        
        # Explicit handling for empty arrays and objects
        if value == '[]':
            return []
        if value == '{}':
            return {}
            
        # Try JSON parsing first
        try:
            # Normalize quotes for JSON compatibility but only for dict/list-like strings
            if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                # Replace single quotes with double quotes for JSON compatibility
                cleaned_value = value.replace("'", '"')
                parsed = json.loads(cleaned_value)
                return parsed
        except json.JSONDecodeError:
            pass
            
        # Try ast parsing as a fallback
        try:
            parsed = ast.literal_eval(value)
            return parsed
        except (ValueError, SyntaxError):
            # Return the original string if all parsing fails
            return value

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
                        
                        # Prepare fields with default values if missing
                        domains = self.safe_parse(resource.get('domains'))
                        domains = [] if domains is None else domains
                        
                        topics = self.safe_parse(resource.get('topics'))
                        topics = {} if topics is None else topics
                        
                        subtitles = self.safe_parse(resource.get('subtitles'))
                        subtitles = {} if subtitles is None else subtitles
                        
                        publishers = self.safe_parse(resource.get('publishers'))
                        publishers = {} if publishers is None else publishers
                        
                        identifiers = self.safe_parse(resource.get('identifiers'))
                        identifiers = {} if identifiers is None else identifiers
                        
                        authors = self.safe_parse(resource.get('authors'))
                        authors = [] if authors is None else authors
                        
                        # Handle summary with special care
                        summary = resource.get('summary', '')
                        if summary and summary.startswith('(') and summary.endswith(')'):
                            summary = summary.replace("('", '').replace("',)", '')
                        
                        # Prepare insert values
                        insert_values = (
                            resource.get('id'),  # id
                            resource.get('doi'),  # doi
                            resource.get('title'),  # title
                            resource.get('abstract', ''),  # abstract
                            summary,  # summary
                            json.dumps(domains),  # domains
                            json.dumps(topics),  # topics
                            resource.get('description', ''),  # description
                            resource.get('expert_id'),  # expert_id
                            resource.get('type'),  # type
                            json.dumps(subtitles),  # subtitles
                            json.dumps(publishers),  # publishers
                            resource.get('collection', ''),  # collection
                            resource.get('date_issue', ''),  # date_issue
                            resource.get('citation', ''),  # citation
                            resource.get('language', ''),  # language
                            json.dumps(identifiers),  # identifiers
                            resource.get('created_at'),  # created_at
                            resource.get('updated_at'),  # updated_at
                            resource.get('source', ''),  # source
                            json.dumps(authors),  # authors
                            resource.get('publication_year', ''),  # publication_year
                            resource.get('field', ''),  # field
                            resource.get('subfield', '')  # subfield
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