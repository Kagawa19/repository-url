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
            
            # Register the JSONB adapter
            psycopg2.extras.register_json(self.connection)
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

    def process_empty_fields(self, field_name, field_value):
        """Process fields safely, providing appropriate defaults for empty values"""
        if field_value is None or field_value == '':
            # Return default empty values based on field type
            if field_name in ['domains', 'authors']:
                return []
            elif field_name in ['topics', 'subtitles', 'publishers', 'identifiers']:
                return {}
            else:
                return None
        
        # If the field is not empty, return the original value
        return field_value

    def process_string(self, value, default_empty_value=None):
        """Process a string value for database insertion"""
        if not value or value == '' or value == 'None' or value == 'none':
            return default_empty_value
        
        # Handle tuples formatted as strings (e.g., summary field)
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            try:
                # Try parsing as tuple
                parsed = ast.literal_eval(value)
                if isinstance(parsed, tuple) and len(parsed) > 0:
                    return parsed[0]
            except (ValueError, SyntaxError):
                # If parsing fails, try basic string manipulation
                clean_value = value.strip('()')
                if clean_value.startswith("'") and clean_value.endswith("'"):
                    return clean_value.strip("'")
                return clean_value.strip('"')
        
        return value

    def process_json_field(self, value, default_empty_value):
        """Process a JSON field for database insertion"""
        if value is None or value == '' or value == 'None' or value == 'none':
            return default_empty_value
        
        if value == '[]':
            return []
            
        if value == '{}':
            return {}
        
        # Try to parse the value as JSON
        try:
            if isinstance(value, str):
                if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                    # Replace single quotes with double quotes for JSON compatibility
                    cleaned_value = value.replace("'", '"')
                    return json.loads(cleaned_value)
            
            # If direct JSON parsing fails, try ast.literal_eval
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            # If all parsing methods fail, return the default value
            if isinstance(default_empty_value, list):
                return []
            elif isinstance(default_empty_value, dict):
                return {}
            return default_empty_value

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
                        resource_id = resource.get('id')
                        
                        # Add extra debugging for the problematic resource
                        if resource_id == '2134532010':
                            logger.info(f"Processing problematic resource with ID: {resource_id}")
                            logger.info(f"Original domains value: '{resource.get('domains')}'")
                            logger.info(f"Original topics value: '{resource.get('topics')}'")
                        
                        # Check if resource already exists
                        if self.check_resource_exists(
                            doi=resource.get('doi'), 
                            title=resource.get('title')
                        ):
                            self.duplicate_resources += 1
                            continue
                        
                        # Process string fields
                        title = self.process_string(resource.get('title'))
                        abstract = self.process_string(resource.get('abstract'), '')
                        summary = self.process_string(resource.get('summary'), '')
                        description = self.process_string(resource.get('description'), '')
                        expert_id = self.process_string(resource.get('expert_id'))
                        type_value = self.process_string(resource.get('type'))
                        collection = self.process_string(resource.get('collection'), '')
                        date_issue = self.process_string(resource.get('date_issue'), '')
                        citation = self.process_string(resource.get('citation'), '')
                        language = self.process_string(resource.get('language'), '')
                        source = self.process_string(resource.get('source'), '')
                        pub_year = self.process_string(resource.get('publication_year'), '')
                        field = self.process_string(resource.get('field'), '')
                        subfield = self.process_string(resource.get('subfield'), '')
                        
                        # Process JSON fields
                        domains = self.process_json_field(resource.get('domains'), [])
                        topics = self.process_json_field(resource.get('topics'), {})
                        subtitles = self.process_json_field(resource.get('subtitles'), {})
                        publishers = self.process_json_field(resource.get('publishers'), {})
                        identifiers = self.process_json_field(resource.get('identifiers'), {})
                        authors = self.process_json_field(resource.get('authors'), [])
                        
                        # Additional logging for problematic resource
                        if resource_id == '2134532010':
                            logger.info(f"Processed domains value: {domains}")
                            logger.info(f"Processed topics value: {topics}")
                        
                        # Execute insert with psycopg2 JSON adapter
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
                        """, (
                            resource_id, 
                            resource.get('doi'),
                            title,
                            abstract,
                            summary,
                            psycopg2.extras.Json(domains),  # Use Json adapter
                            psycopg2.extras.Json(topics),   # Use Json adapter
                            description,
                            expert_id,
                            type_value,
                            psycopg2.extras.Json(subtitles),   # Use Json adapter
                            psycopg2.extras.Json(publishers),  # Use Json adapter
                            collection,
                            date_issue,
                            citation,
                            language,
                            psycopg2.extras.Json(identifiers), # Use Json adapter
                            resource.get('created_at'),
                            resource.get('updated_at'),
                            source,
                            psycopg2.extras.Json(authors),    # Use Json adapter
                            pub_year,
                            field,
                            subfield
                        ))
                        
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

def main():
    """Main function to run the resource importer"""
    importer = ResourceImporter()
    importer.import_resources()

if __name__ == "__main__":
    main()