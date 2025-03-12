#!/usr/bin/env python3
import os
import csv
import ast
import json
import psycopg2
import psycopg2.extras
import logging
import time
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
        self.id_conflicts_resolved = 0
        
        # ID generation counter
        self.next_id = None

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

    def get_next_available_id(self):
        """Get the next available ID from the database"""
        if self.next_id is not None:
            return self.next_id
            
        try:
            # Query the max ID from the database
            self.cursor.execute("SELECT MAX(id) FROM resources_resource")
            result = self.cursor.fetchone()
            
            # Start with max_id + 1, or 1 if no records
            max_id = result[0] if result and result[0] else 0
            self.next_id = max_id + 1
            logger.info(f"Next available ID starts at: {self.next_id}")
            return self.next_id
        except psycopg2.Error as e:
            logger.error(f"Error getting next available ID: {e}")
            # Default to a high number to avoid conflicts
            self.next_id = int(time.time())
            return self.next_id

    def generate_new_id(self):
        """Generate a new unique ID"""
        new_id = self.get_next_available_id()
        self.next_id += 1
        return new_id

    def check_id_exists(self, id_value):
        """Check if an ID already exists in the database"""
        try:
            if not id_value:
                return False
                
            # Convert to integer if it's a string
            if isinstance(id_value, str):
                try:
                    id_value = int(id_value)
                except ValueError:
                    return False
                    
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM resources_resource WHERE id = %s)",
                (id_value,)
            )
            return self.cursor.fetchone()[0]
        except psycopg2.Error as e:
            logger.error(f"Error checking ID existence: {e}")
            return True  # Assume it exists on error to be safe

    def parse_integer(self, value):
        """Parse a string to integer, returning None for empty/invalid values"""
        if not value or value.strip() == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def parse_summary(self, value):
        """Parse summary field which is sometimes stored as a tuple-like string"""
        if not value:
            return ""
            
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

    def parse_postgres_array(self, value):
        """
        Parse a string into a PostgreSQL array format
        For domains column which is of type text[]
        """
        if not value or value.strip() in ['', '[]', 'None', 'none']:
            return None  # Return None for empty arrays to use DEFAULT
        
        try:
            # First try to parse as JSON array
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Try to handle JSON-style arrays
                    cleaned = value.replace("'", '"')
                    parsed = json.loads(cleaned)
                    
                    if not parsed:  # Empty array
                        return None
                        
                    # Convert to PostgreSQL array format
                    if all(isinstance(item, str) for item in parsed):
                        # For text[] all elements must be quoted
                        pg_array = "{" + ",".join(f'"{item}"' for item in parsed) + "}"
                        return pg_array
                    else:
                        # Mixed-type arrays might be problematic, convert all to strings
                        pg_array = "{" + ",".join(f'"{str(item)}"' for item in parsed) + "}"
                        return pg_array
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    parsed = ast.literal_eval(value)
                    if not parsed:  # Empty array
                        return None
                    
                    # Convert to PostgreSQL array format
                    pg_array = "{" + ",".join(f'"{str(item)}"' for item in parsed) + "}"
                    return pg_array
        except (ValueError, SyntaxError):
            # If all parsing fails, return NULL for the array
            return None
            
        # If we couldn't parse it as an array, treat as an empty array
        return None

    def parse_jsonb(self, value):
        """Parse a string into a JSONB object"""
        if not value or value.strip() in ['', '{}', 'None', 'none']:
            return {}
            
        if value == '[]':
            return []
            
        try:
            # Try parsing as JSON first
            if isinstance(value, str):
                if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                    # Replace single quotes with double quotes for JSON compatibility
                    cleaned_value = value.replace("'", '"')
                    return json.loads(cleaned_value)
            
            # If JSON parsing fails, try ast.literal_eval
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            # If all parsing methods fail, return empty object
            if value.startswith('['):
                return []
            return {}

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
                        original_id = resource.get('id')
                        
                        # Check if resource already exists by DOI or title
                        if self.check_resource_exists(
                            doi=resource.get('doi'), 
                            title=resource.get('title')
                        ):
                            self.duplicate_resources += 1
                            logger.info(f"Skipping duplicate resource: {resource.get('title')}")
                            continue
                        
                        # Check if ID conflicts, and generate a new one if needed
                        resource_id = self.parse_integer(original_id)
                        if resource_id is None or self.check_id_exists(resource_id):
                            resource_id = self.generate_new_id()
                            self.id_conflicts_resolved += 1
                            logger.info(f"Generated new ID {resource_id} for resource (original ID: {original_id})")
                        
                        # Parse integer fields
                        expert_id = self.parse_integer(resource.get('expert_id'))
                        
                        # Parse domains specially as PostgreSQL array
                        domains_value = self.parse_postgres_array(resource.get('domains'))
                        
                        # Parse other JSON fields
                        topics = self.parse_jsonb(resource.get('topics'))
                        subtitles = self.parse_jsonb(resource.get('subtitles'))
                        publishers = self.parse_jsonb(resource.get('publishers'))
                        identifiers = self.parse_jsonb(resource.get('identifiers'))
                        authors = self.parse_jsonb(resource.get('authors'))
                        
                        # Parse summary with special handling
                        summary = self.parse_summary(resource.get('summary'))
                        
                        # Insert into database with a new ID if needed
                        self.cursor.execute("""
                            INSERT INTO resources_resource (
                                id, doi, title, abstract, summary, domains, topics, 
                                description, expert_id, type, subtitles, publishers, 
                                collection, date_issue, citation, language, identifiers, 
                                created_at, updated_at, source, authors, publication_year, 
                                field, subfield
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                     %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            resource_id,  # Use the new ID if generated
                            resource.get('doi'),
                            resource.get('title'),
                            resource.get('abstract'),
                            summary,
                            domains_value,  
                            psycopg2.extras.Json(topics),
                            resource.get('description'),
                            expert_id,  
                            resource.get('type'),
                            psycopg2.extras.Json(subtitles),
                            psycopg2.extras.Json(publishers),
                            resource.get('collection'),
                            resource.get('date_issue'),
                            resource.get('citation'),
                            resource.get('language'),
                            psycopg2.extras.Json(identifiers),
                            resource.get('created_at'),
                            resource.get('updated_at'),
                            resource.get('source'),
                            psycopg2.extras.Json(authors),
                            resource.get('publication_year'),
                            resource.get('field'),
                            resource.get('subfield')
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
            logger.info(f"ID Conflicts Resolved: {self.id_conflicts_resolved}")
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