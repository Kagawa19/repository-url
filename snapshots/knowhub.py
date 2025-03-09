import requests
import json
import psycopg2
import os
import logging
import time
import re
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("knowhub_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection settings for your local database
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": "localhost",
    "port": "5432"
}

class DSpaceClient:
    """Client for interacting with a DSpace repository"""
    
    def __init__(self, base_url="https://knowhub.aphrc.org"):
        self.base_url = base_url
        self.oai_url = f"{base_url}/oai/request"
        
        # Common handle prefix for DSpace
        self.handle_prefix = "123456789"
        
        # Set headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml, text/xml, */*'
        }
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Define OAI-PMH namespaces
        self.namespaces = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
        }
    
    def get_collections(self) -> Dict[str, str]:
        """Get available collections from OAI-PMH ListSets verb"""
        collections = {}
        
        try:
            url = f"{self.oai_url}?verb=ListSets"
            logger.info(f"Requesting collections from {url}")
            
            response = self.session.get(url, timeout=30, verify=False)
            if response.status_code != 200:
                logger.error(f"Failed to get collections: {response.status_code}")
                return collections
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract sets (collections)
            sets = root.findall('.//oai:set', self.namespaces)
            for set_elem in sets:
                spec = set_elem.find('./oai:setSpec', self.namespaces)
                name = set_elem.find('./oai:setName', self.namespaces)
                
                if spec is not None and spec.text and name is not None and name.text:
                    # Only include collection sets that match expected pattern
                    if spec.text.startswith('col_'):
                        collections[spec.text] = name.text
                        logger.info(f"Found collection: {name.text} (Set: {spec.text})")
            
            return collections
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return collections
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return collections
    
    def get_items_by_collection(self, set_spec: str, limit: int = None) -> List[Dict]:
        """Get items from a specific collection using OAI-PMH"""
        items = []
        
        try:
            # Initial request URL
            url = f"{self.oai_url}?verb=ListRecords&metadataPrefix=oai_dc&set={set_spec}"
            resumption_token = None
            
            while True:
                if resumption_token:
                    url = f"{self.oai_url}?verb=ListRecords&resumptionToken={resumption_token}"
                
                logger.info(f"Requesting items from {url}")
                response = self.session.get(url, timeout=60, verify=False)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get items: {response.status_code}")
                    break
                
                # Parse XML response
                try:
                    root = ET.fromstring(response.text)
                    
                    # Extract records
                    records = root.findall('.//oai:record', self.namespaces)
                    logger.info(f"Found {len(records)} records in this batch")
                    
                    for record in records:
                        try:
                            # Extract header information
                            header = record.find('./oai:header', self.namespaces)
                            if header is None:
                                continue
                                
                            # Check if record is deleted
                            status = header.get('status')
                            if status == 'deleted':
                                continue
                            
                            # Get identifier
                            identifier_elem = header.find('./oai:identifier', self.namespaces)
                            if identifier_elem is None or not identifier_elem.text:
                                continue
                                
                            identifier = identifier_elem.text
                            
                            # Extract handle from identifier
                            handle_match = re.search(r'oai:[^:]+:(\d+/\d+)', identifier)
                            handle = handle_match.group(1) if handle_match else None
                            
                            if not handle:
                                continue
                            
                            # Get metadata
                            metadata = record.find('./oai:metadata/oai_dc:dc', self.namespaces)
                            if metadata is None:
                                continue
                            
                            # Create item dictionary
                            item = {
                                'identifier': identifier,
                                'handle': handle,
                                'url': f"{self.base_url}/handle/{handle}"
                            }
                            
                            # Extract Dublin Core fields
                            # Title
                            title_elem = metadata.find('./dc:title', self.namespaces)
                            if title_elem is not None and title_elem.text:
                                item['title'] = title_elem.text.strip()
                            else:
                                continue  # Skip items without title
                            
                            # Authors
                            authors = []
                            for creator in metadata.findall('./dc:creator', self.namespaces):
                                if creator.text:
                                    authors.append(creator.text.strip())
                            for contributor in metadata.findall('./dc:contributor', self.namespaces):
                                if contributor.text:
                                    authors.append(contributor.text.strip())
                            item['authors'] = authors
                            
                            # Description/Abstract
                            descriptions = []
                            for desc in metadata.findall('./dc:description', self.namespaces):
                                if desc.text:
                                    descriptions.append(desc.text.strip())
                            item['description'] = ' '.join(descriptions) if descriptions else None
                            
                            # Date
                            for date in metadata.findall('./dc:date', self.namespaces):
                                if date.text:
                                    date_str = date.text.strip()
                                    item['date'] = date_str
                                    
                                    # Extract year
                                    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                                    if year_match:
                                        item['publication_year'] = int(year_match.group(0))
                                        break
                            
                            # Type
                            for type_elem in metadata.findall('./dc:type', self.namespaces):
                                if type_elem.text:
                                    item['type'] = type_elem.text.strip().lower().replace(' ', '_')
                                    break
                            
                            # Subject/Keywords
                            keywords = []
                            for subject in metadata.findall('./dc:subject', self.namespaces):
                                if subject.text:
                                    keywords.append(subject.text.strip())
                            item['keywords'] = keywords
                            
                            # Publisher
                            publisher = metadata.find('./dc:publisher', self.namespaces)
                            if publisher is not None and publisher.text:
                                item['publisher'] = publisher.text.strip()
                            
                            # Format
                            format_elem = metadata.find('./dc:format', self.namespaces)
                            if format_elem is not None and format_elem.text:
                                item['format'] = format_elem.text.strip()
                            
                            # Language
                            language = metadata.find('./dc:language', self.namespaces)
                            if language is not None and language.text:
                                item['language'] = language.text.strip()
                            
                            # Rights
                            rights = metadata.find('./dc:rights', self.namespaces)
                            if rights is not None and rights.text:
                                item['rights'] = rights.text.strip()
                            
                            # Source
                            source = metadata.find('./dc:source', self.namespaces)
                            if source is not None and source.text:
                                item['source_publication'] = source.text.strip()
                            
                            # Multiple identifiers (DOI, etc.)
                            for id_elem in metadata.findall('./dc:identifier', self.namespaces):
                                if id_elem.text:
                                    id_text = id_elem.text.strip()
                                    
                                    # URL
                                    if id_text.startswith('http'):
                                        item['url'] = id_text
                                    
                                    # DOI
                                    elif 'doi' in id_text.lower() or id_text.startswith('10.'):
                                        doi_match = re.search(r'(10\.\d{4,}/\S+)', id_text)
                                        if doi_match:
                                            item['doi'] = doi_match.group(1)
                            
                            items.append(item)
                            
                            # Check if we've reached the limit
                            if limit and len(items) >= limit:
                                return items
                                
                        except Exception as e:
                            logger.error(f"Error processing record: {e}")
                            continue
                    
                    # Check for resumption token
                    token_elem = root.find('.//oai:resumptionToken', self.namespaces)
                    if token_elem is not None and token_elem.text:
                        resumption_token = token_elem.text
                        logger.info(f"Found resumption token: {resumption_token}")
                        time.sleep(1)  # Be nice to the server
                    else:
                        break
                    
                except ET.ParseError as e:
                    logger.error(f"XML parsing error: {e}")
                    break
                
            logger.info(f"Retrieved {len(items)} total items from collection")
            return items
            
        except Exception as e:
            logger.error(f"Error getting items: {e}")
            return items

def check_resources_table() -> bool:
    """Check if resources_resource table exists"""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if resources_resource table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'resources_resource'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logger.error("resources_resource table does not exist. Please create it first.")
            return False
        
        logger.info("resources_resource table found.")
        return True
        
    except Exception as e:
        logger.error(f"Error checking resources table: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def insert_resource(item: Dict) -> Optional[int]:
    """Insert resource into database"""
    if not item or 'title' not in item:
        return None
    
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if resource already exists by URL or handle
        if item.get('url'):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get('url'),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                return result[0]
        
        if item.get('handle'):
            cur.execute("SELECT id FROM resources_resource WHERE identifiers::text LIKE %s", 
                       (f"%{item.get('handle')}%",))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with handle {item.get('handle')} already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON
        authors_json = None
        if item.get('authors'):
            # Convert authors to JSON format expected by the database
            authors_json = json.dumps([{"name": author} for author in item.get('authors')])
        
        # Determine resource type - use collection type if item type is missing
        resource_type = item.get('type', item.get('collection_type', 'other'))
        
        # Prepare identifiers JSON
        identifiers = {
            "handle": item.get('handle')
        }
        
        if item.get('keywords'):
            identifiers["keywords"] = item.get('keywords')
            
        if item.get('identifier'):
            identifiers["oai_identifier"] = item.get('identifier')
        
        identifiers_json = json.dumps(identifiers)
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, 
             publication_year, source, identifiers, doi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            item.get('title'),
            item.get('description'),
            item.get('url'),
            resource_type,
            authors_json,
            item.get('publication_year'),
            "knowhub",
            identifiers_json,
            item.get('doi')
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            logger.info(f"Inserted resource with ID {result[0]}: {item.get('title')}")
            return result[0]
        else:
            logger.warning(f"Failed to insert resource: {item.get('title')}")
            return None
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error inserting resource: {e}")
        return None
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    """Main function to harvest DSpace repository"""
    start_time = time.time()
    
    # Check resources table
    if not check_resources_table():
        return
    
    # Initialize client
    dspace = DSpaceClient()
    
    # Get collections
    collections = dspace.get_collections()
    
    if not collections:
        logger.warning("No collections found. Using default collection set mappings.")
        collections = {
            'col_123456789_1': 'Publications',
            'col_123456789_2': 'Documents',
            'col_123456789_3': 'Reports',
            'col_123456789_4': 'Multimedia'
        }
    
    total_items = 0
    
    # Process each collection
    for set_spec, collection_name in collections.items():
        logger.info(f"Processing collection: {collection_name} (Set: {set_spec})")
        
        # Get items from collection
        items = dspace.get_items_by_collection(set_spec)
        
        # Add collection type to items
        collection_type = collection_name.lower().replace(' ', '_')
        for item in items:
            if 'type' not in item:
                item['type'] = collection_type
            item['collection_type'] = collection_type
        
        # Insert items into database
        inserted = 0
        for item in items:
            if insert_resource(item):
                inserted += 1
                total_items += 1
        
        logger.info(f"Inserted {inserted} out of {len(items)} items from {collection_name}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Script completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total items stored: {total_items}")

if __name__ == "__main__":
    main()