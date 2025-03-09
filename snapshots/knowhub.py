import requests
import json
import psycopg2
import os
import logging
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

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

class DSpaceAPI:
    """Client for interacting with DSpace REST API"""
    
    def __init__(self, base_url="https://knowhub.aphrc.org"):
        self.base_url = base_url
        # Most DSpace installations have the REST API at /rest
        self.api_url = f"{base_url}/rest"
        # Default REST API version 7 endpoint for newer DSpace installations
        self.api_v7_url = f"{base_url}/server/api"
        
        # Set proper headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def test_api_endpoint(self) -> str:
        """Test which API endpoint version is available"""
        try:
            # Try REST API v7
            response = self.session.get(f"{self.api_v7_url}/core/items", timeout=30)
            if response.status_code == 200:
                logger.info("Using DSpace REST API v7")
                return "v7"
        except:
            pass
            
        try:
            # Try REST API v6
            response = self.session.get(f"{self.api_url}/items", timeout=30)
            if response.status_code == 200:
                logger.info("Using DSpace REST API v6")
                return "v6"
        except:
            pass
            
        # If no API endpoints work, try OAI-PMH
        try:
            response = self.session.get(f"{self.base_url}/oai/request?verb=Identify", timeout=30)
            if response.status_code == 200 and '<Identify>' in response.text:
                logger.info("Using OAI-PMH endpoint")
                return "oai"
        except:
            pass
            
        logger.warning("No API endpoints found, fallback to direct database access might be needed")
        return "none"
    
    def get_communities(self) -> List[Dict]:
        """Get all communities from DSpace"""
        try:
            response = self.session.get(f"{self.api_url}/communities", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get communities: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting communities: {e}")
            return []
    
    def get_collections(self, community_id=None) -> List[Dict]:
        """Get collections, optionally filtered by community"""
        try:
            if community_id:
                url = f"{self.api_url}/communities/{community_id}/collections"
            else:
                url = f"{self.api_url}/collections"
                
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get collections: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []
    
    def get_items_in_collection(self, collection_id: str, limit: int = 100) -> List[Dict]:
        """Get items in a specific collection"""
        try:
            items = []
            offset = 0
            
            while True:
                url = f"{self.api_url}/collections/{collection_id}/items?limit={limit}&offset={offset}"
                logger.info(f"Fetching items from {url}")
                
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    logger.warning(f"Failed to get items: {response.status_code}")
                    break
                
                batch = response.json()
                if not batch:
                    break
                    
                items.extend(batch)
                logger.info(f"Retrieved {len(batch)} items, total: {len(items)}")
                
                if len(batch) < limit:
                    break
                    
                offset += limit
                time.sleep(1)  # Be nice to the server
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting items: {e}")
            return []
    
    def get_item_metadata(self, item_id: str) -> Dict:
        """Get detailed metadata for a specific item"""
        try:
            url = f"{self.api_url}/items/{item_id}/metadata"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get item metadata: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting item metadata: {e}")
            return {}
    
    def get_items_via_oai(self, set_spec: str = None, limit: int = 100) -> List[Dict]:
        """Get items using OAI-PMH protocol"""
        try:
            items = []
            resumption_token = None
            
            while True:
                if resumption_token:
                    url = f"{self.base_url}/oai/request?verb=ListRecords&resumptionToken={resumption_token}"
                else:
                    url = f"{self.base_url}/oai/request?verb=ListRecords&metadataPrefix=oai_dc"
                    if set_spec:
                        url += f"&set={set_spec}"
                
                logger.info(f"Fetching OAI items from {url}")
                response = self.session.get(url, timeout=60)
                
                if response.status_code != 200:
                    logger.warning(f"OAI request failed: {response.status_code}")
                    break
                
                # Parse XML response
                try:
                    import xml.etree.ElementTree as ET
                    from xml.etree.ElementTree import ParseError
                    
                    root = ET.fromstring(response.text)
                    
                    # Define namespaces
                    namespaces = {
                        'oai': 'http://www.openarchives.org/OAI/2.0/',
                        'dc': 'http://purl.org/dc/elements/1.1/',
                        'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
                    }
                    
                    # Extract records
                    records = root.findall('.//oai:record', namespaces)
                    
                    for record in records:
                        item = {}
                        
                        # Get identifier
                        identifier = record.find('./oai:header/oai:identifier', namespaces)
                        if identifier is not None and identifier.text:
                            item['identifier'] = identifier.text
                        
                        # Get metadata
                        metadata = record.find('./oai:metadata/oai_dc:dc', namespaces)
                        if metadata is not None:
                            # Title
                            title_elem = metadata.find('./dc:title', namespaces)
                            if title_elem is not None and title_elem.text:
                                item['title'] = title_elem.text
                            
                            # Creator/Author
                            authors = []
                            for creator in metadata.findall('./dc:creator', namespaces):
                                if creator.text:
                                    authors.append(creator.text)
                            item['authors'] = authors
                            
                            # Description
                            desc_elem = metadata.find('./dc:description', namespaces)
                            if desc_elem is not None and desc_elem.text:
                                item['description'] = desc_elem.text
                            
                            # Date
                            date_elem = metadata.find('./dc:date', namespaces)
                            if date_elem is not None and date_elem.text:
                                item['date'] = date_elem.text
                                
                                # Extract year
                                year_match = re.search(r'\b(19|20)\d{2}\b', date_elem.text)
                                if year_match:
                                    item['year'] = int(year_match.group(0))
                            
                            # Type
                            type_elem = metadata.find('./dc:type', namespaces)
                            if type_elem is not None and type_elem.text:
                                item['type'] = type_elem.text
                            
                            # Identifier (URL, DOI)
                            for id_elem in metadata.findall('./dc:identifier', namespaces):
                                if id_elem.text:
                                    if id_elem.text.startswith('http'):
                                        item['url'] = id_elem.text
                                    elif 'doi' in id_elem.text.lower():
                                        item['doi'] = id_elem.text
                        
                        if 'title' in item:  # Only add items with at least a title
                            items.append(item)
                    
                    # Check for resumption token
                    resumption_token_elem = root.find('.//oai:resumptionToken', namespaces)
                    if resumption_token_elem is not None and resumption_token_elem.text:
                        resumption_token = resumption_token_elem.text
                    else:
                        break
                        
                    # Stop if we've reached our limit
                    if limit and len(items) >= limit:
                        break
                
                except ParseError as e:
                    logger.error(f"Error parsing OAI XML: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error processing OAI response: {e}")
                    break
                
                time.sleep(2)  # Be nice to the server
            
            return items[:limit] if limit else items
            
        except Exception as e:
            logger.error(f"Error getting items via OAI: {e}")
            return []

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
        
        # Check if resource already exists by URL or identifier
        if item.get('url'):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get('url'),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
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
        
        # Determine resource type
        resource_type = item.get('type', 'other')
        
        # Prepare identifiers JSON if needed
        identifiers_json = None
        identifiers = {}
        
        if item.get('identifier'):
            identifiers['oai_identifier'] = item.get('identifier')
        
        if item.get('doi'):
            identifiers['doi'] = item.get('doi')
            
        if identifiers:
            identifiers_json = json.dumps(identifiers)
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, publication_year, source, identifiers, doi)
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
            item.get('year'),
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
    """Main function to run the DSpace API client"""
    start_time = time.time()
    
    # Check resources table
    if not check_resources_table():
        return
    
    # Initialize DSpace API client
    dspace = DSpaceAPI()
    
    # Test which API endpoint works
    api_version = dspace.test_api_endpoint()
    
    total_items = 0
    
    if api_version == "v6" or api_version == "v7":
        # REST API approach
        collections = dspace.get_collections()
        
        for collection in collections:
            collection_id = collection.get('id')
            collection_name = collection.get('name')
            
            if collection_id:
                logger.info(f"Processing collection: {collection_name} (ID: {collection_id})")
                
                items = dspace.get_items_in_collection(collection_id)
                
                for item in items:
                    item_id = item.get('id')
                    
                    if item_id:
                        # Get detailed metadata
                        metadata = dspace.get_item_metadata(item_id)
                        
                        # Process metadata into a standardized format
                        processed_item = {
                            'title': item.get('name'),
                            'url': f"{dspace.base_url}/handle/{item.get('handle')}",
                            'authors': [],
                            'type': collection_name.lower().replace(' ', '_'),
                            'source': 'knowhub'
                        }
                        
                        # Process metadata fields
                        for meta in metadata:
                            key = meta.get('key', '')
                            value = meta.get('value', '')
                            
                            if 'dc.title' in key and value:
                                processed_item['title'] = value
                            elif 'dc.creator' in key and value:
                                processed_item['authors'].append(value)
                            elif 'dc.contributor.author' in key and value:
                                processed_item['authors'].append(value)
                            elif 'dc.description' in key and value:
                                processed_item['description'] = value
                            elif 'dc.date.issued' in key and value:
                                processed_item['date'] = value
                                # Extract year
                                year_match = re.search(r'\b(19|20)\d{2}\b', value)
                                if year_match:
                                    processed_item['year'] = int(year_match.group(0))
                            elif 'dc.type' in key and value:
                                processed_item['type'] = value
                            elif 'dc.identifier.uri' in key and value:
                                processed_item['url'] = value
                            elif 'dc.identifier.doi' in key and value:
                                processed_item['doi'] = value
                        
                        # Insert into database
                        if insert_resource(processed_item):
                            total_items += 1
    
    elif api_version == "oai":
        # OAI-PMH approach
        # Map collection types to OAI set specs
        set_mappings = {
            'publications': 'col_123456789_1',
            'documents': 'col_123456789_2',
            'reports': 'col_123456789_3',
            'multimedia': 'col_123456789_4'
        }
        
        for resource_type, set_spec in set_mappings.items():
            logger.info(f"Processing {resource_type} via OAI-PMH (set: {set_spec})")
            
            items = dspace.get_items_via_oai(set_spec)
            
            for item in items:
                # Ensure resource type is set
                item['type'] = resource_type
                
                # Insert into database
                if insert_resource(item):
                    total_items += 1
    
    else:
        logger.error("No working API endpoints found. Cannot retrieve data.")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Script completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total items stored: {total_items}")

if __name__ == "__main__":
    main()