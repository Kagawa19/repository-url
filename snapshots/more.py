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
        # Handle None values
        if value is None:
            return None
            
        # Handle non-string values
        if not isinstance(value, str):
            return value
            
        # Handle empty or whitespace values
        if not value.strip() or value.strip() in ['', 'None', 'none']:
            return None
            
        # Explicit handling for empty arrays and objects
        if value.strip() == '[]':
            return []
        if value.strip() == '{}':
            return {}
        
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
                        
                        # Handle empty fields explicitly
                        domains_raw = resource.get('domains', '')
                        domains = [] if not domains_raw or domains_raw == '[]' else self.safe_parse(domains_raw) or []
                        
                        topics_raw = resource.get('topics', '')
                        topics = {} if not topics_raw or topics_raw == '{}' else self.safe_parse(topics_raw) or {}
                        
                        subtitles_raw = resource.get('subtitles', '')
                        subtitles = {} if not subtitles_raw or subtitles_raw == '{}' else self.safe_parse(subtitles_raw) or {}
                        
                        publishers_raw = resource.get('publishers', '')
                        publishers = {} if not publishers_raw or publishers_raw == '{}' else self.safe_parse(publishers_raw) or {}
                        
                        identifiers_raw = resource.get('identifiers', '')
                        identifiers = {} if not identifiers_raw or identifiers_raw == '{}' else self.safe_parse(identifiers_raw) or {}
                        
                        authors_raw = resource.get('authors', '')
                        authors = [] if not authors_raw or authors_raw == '[]' else self.safe_parse(authors_raw) or []
                        
                        # Handle summary with special care
                        summary = resource.get('summary', '')
                        if summary and summary.startswith('(') and summary.endswith(')'):
                            # Try to extract the content from the tuple-like string
                            try:
                                parsed = ast.literal_eval(summary)
                                if isinstance(parsed, tuple) and len(parsed) > 0:
                                    summary = parsed[0]
                                else:
                                    summary = summary.replace("('", '').replace("',)", '')
                            except (ValueError, SyntaxError):
                                summary = summary.replace("('", '').replace("',)", '')
                        
                        # Debug logging for problematic resource
                        if resource.get('id') == '2134532010':
                            logger.info(f"Processing problematic resource with ID 2134532010")
                            logger.info(f"Domains before: {resource.get('domains')}, after: {domains}")
                            logger.info(f"Topics before: {resource.get('topics')}, after: {topics}")
                            
                        # Ensure JSON serialization works properly
                        try:
                            domains_json = json.dumps(domains)
                            topics_json = json.dumps(topics)
                            subtitles_json = json.dumps(subtitles)
                            publishers_json = json.dumps(publishers)
                            identifiers_json = json.dumps(identifiers)
                            authors_json = json.dumps(authors)
                        except (TypeError, ValueError) as e:
                            logger.error(f"JSON serialization error: {e}")
                            logger.error(f"Problematic fields: domains={domains}, topics={topics}")
                            # Set to default values if serialization fails
                            domains_json = '[]'
                            topics_json = '{}'
                            subtitles_json = '{}'
                            publishers_json = '{}'
                            identifiers_json = '{}'
                            authors_json = '[]'
                            
                        # Prepare insert values
                        insert_values = (
                            resource.get('id'),  # id
                            resource.get('doi'),  # doi
                            resource.get('title'),  # title
                            resource.get('abstract', ''),  # abstract
                            summary,  # summary
                            domains_json,  # domains
                            topics_json,  # topics
                            resource.get('description', ''),  # description
                            resource.get('expert_id'),  # expert_id
                            resource.get('type'),  # type
                            subtitles_json,  # subtitles
                            publishers_json,  # publishers
                            resource.get('collection', ''),  # collection
                            resource.get('date_issue', ''),  # date_issue
                            resource.get('citation', ''),  # citation
                            resource.get('language', ''),  # language
                            identifiers_json,  # identifiers
                            resource.get('created_at'),  # created_at
                            resource.get('updated_at'),  # updated_at
                            resource.get('source', ''),  # source
                            authors_json,  # authors
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