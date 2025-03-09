#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
from psycopg2 import sql
from typing import Optional, Dict, List, Any
from bs4 import BeautifulSoup

# Database connection settings - Update for your environment
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

class TextSummarizer:
    """Simple placeholder for text summarization functionality."""
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Placeholder for text summarization."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

class KnowhubScraper:
    def __init__(self, summarizer: Optional[TextSummarizer] = None):
        """Initialize KnowhubScraper with authentication capabilities."""
        self.base_url = os.getenv('KNOWHUB_BASE_URL', 'https://knowhub.aphrc.org')
        self.publications_url = f"{self.base_url}/handle/123456789/1"
        
        # Update endpoints to match exact type names
        self.endpoints = {
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        self.summarizer = summarizer or TextSummarizer()
        
        # Configure session with proper headers and timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def fetch_items_by_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """Fetch items from KnowHub based on resource type.
        
        Args:
            resource_type: The type of resource to fetch (documents, reports, multimedia, publications)
        
        Returns:
            A list of dictionaries containing item information
        """
        if resource_type == 'publications':
            url = self.publications_url
        else:
            url = self.endpoints.get(resource_type)
            if not url:
                print(f"Unknown resource type: {resource_type}")
                return []
        
        try:
            print(f"Fetching {resource_type} from {url}")
            
            # Use a timeout to avoid hanging indefinitely
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract items based on the HTML structure visible in the screenshot
            items = []
            
            # First try to find items in the standard layout shown in the screenshot
            item_elements = soup.select('div.artifact-description')
            
            if not item_elements:
                # Try alternative selectors if the first one doesn't work
                item_elements = soup.select('.ds-artifact-item')
            
            if not item_elements:
                # Try an even more generic selector
                item_elements = soup.select('.item-wrapper') or soup.select('.item')
            
            print(f"Found {len(item_elements)} item elements on the page")
            
            # If still no items found, extract any potential items from the page
            if not item_elements:
                print("No standard item elements found. Attempting to extract information from page structure.")
                # Log a sample of the HTML to debug
                print(f"Page title: {soup.title.string if soup.title else 'No title'}")
                print(f"Sample HTML (first 500 chars): {soup.prettify()[:500]}")
                
                # Look for any links that might be publications
                article_links = soup.select('a[href*="handle"]')
                for link in article_links:
                    title = link.text.strip()
                    if title and len(title) > 5:  # Skip very short titles that are likely navigation
                        item_url = link.get('href')
                        if item_url and not item_url.startswith('http'):
                            item_url = f"{self.base_url}{item_url}"
                        
                        items.append({
                            "title": title,
                            "url": item_url,
                            "author": "",
                            "description": "",
                            "publication_date": "",
                            "resource_type": resource_type,
                            "source": "knowhub"
                        })
            
            # Process standard item elements if found
            for item in item_elements:
                # Extract title and URL
                title_element = item.select_one('h4 a, h3 a, .artifact-title a, a.title')
                if not title_element:
                    # Try other possible title selectors
                    title_element = item.select_one('a[href*="handle"]')
                
                if not title_element:
                    # If still no title element, try to extract any meaningful link
                    title_element = item.select_one('a')
                
                if title_element:
                    title = title_element.text.strip()
                    item_url = title_element.get('href')
                    
                    if item_url and not item_url.startswith('http'):
                        item_url = f"{self.base_url}{item_url}"
                    
                    # Extract authors
                    authors_element = item.select_one('.artifact-authors, .item-authors, .author')
                    authors = authors_element.text.strip() if authors_element else ""
                    
                    # Extract abstract/description
                    description_element = item.select_one('.artifact-abstract, .item-description, .description')
                    description = description_element.text.strip() if description_element else ""
                    
                    # Extract date
                    date_element = item.select_one('.artifact-date, .item-date, .date, span.date')
                    publication_date = date_element.text.strip() if date_element else None
                    
                    # Create item dictionary
                    item_data = {
                        "title": title,
                        "url": item_url,
                        "author": authors,
                        "description": self.summarizer.summarize(description),
                        "publication_date": publication_date,
                        "resource_type": resource_type,
                        "source": "knowhub"
                    }
                    
                    items.append(item_data)
            
            print(f"Extracted {len(items)} {resource_type} items")
            return items
            
        except requests.exceptions.Timeout:
            print(f"Connection timed out when fetching {resource_type} from {url}. Consider increasing the timeout.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {resource_type} from {url}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error when processing {resource_type}: {e}")
            return []
    
    def fetch_all_resources(self) -> List[Dict[str, Any]]:
        """Fetch all types of resources from KnowHub."""
        all_resources = []
        
        # Fetch publications
        publications = self.fetch_items_by_type('publications')
        all_resources.extend(publications)
        
        # Fetch other resource types
        for resource_type in self.endpoints.keys():
            resources = self.fetch_items_by_type(resource_type)
            all_resources.extend(resources)
        
        return all_resources

def check_resources_table() -> bool:
    """Check if resources_resource table exists and has necessary columns."""
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
            print("resources_resource table does not exist. Please create it first.")
            return False
        
        # Check if 'source' column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'resources_resource' AND column_name = 'source'
            );
        """)
        source_column_exists = cur.fetchone()[0]
        
        if not source_column_exists:
            print("'source' column does not exist in resources_resource table. Adding it...")
            cur.execute("""
                ALTER TABLE resources_resource 
                ADD COLUMN IF NOT EXISTS source VARCHAR(50);
            """)
            conn.commit()
        
        print("resources_resource table is ready.")
        return True
        
    except Exception as e:
        print(f"Error checking resources table: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def insert_single_resource(item: Dict[str, Any]) -> Optional[int]:
    """Insert a single KnowHub resource into the resources_resource table.
    
    Args:
        item: Resource item to insert
        
    Returns:
        ID of the inserted resource or None if insertion failed
    """
    if not item:
        return None
    
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if item with this URL already exists
        if item.get("url"):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get("url"),))
            result = cur.fetchone()
            if result:
                print(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON if needed
        authors_json = None
        if item.get("author"):
            # Simple conversion of author string to JSON format
            author_names = [name.strip() for name in item.get("author", "").split(",") if name.strip()]
            if author_names:
                authors_json = json.dumps([{"name": name} for name in author_names])
        
        # Extract publication_year from publication_date if available
        publication_year = None
        if item.get("publication_date"):
            try:
                # Try to extract a year from the date string
                date_str = item.get("publication_date", "")
                # Look for 4-digit year pattern
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    publication_year = int(year_match.group(0))
            except:
                pass
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, 
             publication_year, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            item.get("title"),
            item.get("description"),
            item.get("url"),
            item.get("resource_type"),
            authors_json,
            publication_year,
            item.get("source", "knowhub")
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            print(f"Inserted resource with ID {result[0]}: {item.get('title')}")
            return result[0]
        else:
            print(f"Failed to insert resource: {item.get('title')}")
            return None
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting resource '{item.get('title')}': {e}")
        return None
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def insert_into_resources(data: List[Dict[str, Any]]) -> int:
    """Insert multiple KnowHub resources into resources_resource table.
    
    This is a batch function that calls insert_single_resource() for each item.
    
    Args:
        data: List of resource items to insert
        
    Returns:
        Number of items successfully inserted
    """
    if not data:
        return 0
    
    inserted_count = 0
    
    for item in data:
        result_id = insert_single_resource(item)
        if result_id:
            inserted_count += 1
    
    return inserted_count

def main():
    """Main function to scrape KnowHub and insert data into the database."""
    import time
    start_time = time.time()
    
    try:
        # Check if resources_resource table exists
        if not check_resources_table():
            return
        
        # Initialize scraper with text summarizer
        scraper = KnowhubScraper(TextSummarizer())
        
        # Track insertion statistics
        total_resources = 0
        inserted_count = 0
        
        # Process each resource type separately and insert as we go
        for resource_type in ['publications'] + list(scraper.endpoints.keys()):
            print(f"\nProcessing {resource_type}...")
            resources = scraper.fetch_items_by_type(resource_type)
            
            if not resources:
                print(f"No {resource_type} found.")
                continue
                
            print(f"Found {len(resources)} {resource_type}. Inserting into database...")
            
            # Insert resources one by one for fault tolerance
            for item in resources:
                total_resources += 1
                
                # Insert the individual resource
                result_id = insert_single_resource(item)
                if result_id:
                    inserted_count += 1
                
                # Add a small delay to avoid overwhelming the database
                time.sleep(0.1)
                
            print(f"Completed {resource_type}: {inserted_count} inserted so far.")
        
        if total_resources == 0:
            print("No resources found in KnowHub.")
        
        elapsed_time = time.time() - start_time
        print(f"Done! Inserted {inserted_count} resources out of {total_resources} in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()