#!/usr/bin/env python3
import os
import logging
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime, timezone
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowhubCSVScraper:
    def __init__(self):
        # Base URL for KnowHub
        self.base_url = 'https://knowhub.aphrc.org'
        
        # Endpoints for different collections
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        # Track seen handles to avoid duplicates
        self.seen_handles = set()
        
        # Create a requests session
        self.session = requests.Session()
        
        # Set headers to mimic a browser and configure SSL
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
        # Disable SSL verification
        self.session.verify = False
        
        # Create output directory for CSVs
        self.output_dir = 'knowhub_exports'
        os.makedirs(self.output_dir, exist_ok=True)

    def _fetch_page(self, url):
        """Fetch a page with robust error handling"""
        try:
            # Increased timeout and added retry mechanism
            response = self.session.get(
                url, 
                timeout=30,  # Increased timeout
                verify=False,  # Disable SSL verification
                allow_redirects=True  # Follow redirects
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {url}: {e}")
            return None

    def _extract_item_details(self, item_url):
        """Extract detailed information for a single item"""
        try:
            # Fetch the item page
            page_content = self._fetch_page(item_url)
            if not page_content:
                return None
            
            # Parse the page
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract title
            title_elem = soup.find(['h1', 'h2'], class_=lambda c: c and 'item-title' in c)
            if not title_elem:
                title_elem = soup.find(['h1', 'h2'])
            
            if not title_elem:
                return None
            
            title = title_elem.get_text().strip()
            
            # Determine content type based on URL
            content_type = 'other'
            if '123456789/1' in item_url:
                content_type = 'publications'
            elif '123456789/2' in item_url:
                content_type = 'documents'
            elif '123456789/3' in item_url:
                content_type = 'reports'
            elif '123456789/4' in item_url:
                content_type = 'multimedia'
            
            # Extract metadata
            metadata_section = soup.find('div', class_=lambda c: c and any(x in c for x in ['item-page-field-wrapper', 'item-metadata']))
            
            # Default metadata dictionary
            metadata = {
                'abstract': '',
                'authors': [],
                'date': None,
                'keywords': [],
                'description': '',
                'doi': None,
                'citation': None
            }
            
            if metadata_section:
                # Extract abstract
                abstract_elem = metadata_section.find('span', class_=['abstract', 'description'])
                if abstract_elem:
                    metadata['abstract'] = abstract_elem.get_text().strip()
                
                # Extract authors
                author_elems = metadata_section.find_all('span', class_=['author', 'creator'])
                metadata['authors'] = [
                    author.get_text().strip() 
                    for author in author_elems 
                    if author.get_text().strip()
                ]
                
                # Extract date
                date_elem = metadata_section.find('span', class_=['date', 'issued'])
                if date_elem:
                    metadata['date'] = date_elem.get_text().strip()
                
                # Extract keywords
                keyword_elems = metadata_section.find_all('span', class_=['subject', 'keyword'])
                metadata['keywords'] = [
                    kw.get_text().strip() 
                    for kw in keyword_elems 
                    if kw.get_text().strip()
                ]
                
                # Extract DOI if available
                doi_elem = metadata_section.find('span', class_='doi')
                if doi_elem:
                    doi_text = doi_elem.get_text().strip()
                    doi_match = re.search(r'10\.\d{4,}/\S+', doi_text)
                    if doi_match:
                        metadata['doi'] = doi_match.group(0)
                
                # Extract citation if available
                citation_elem = metadata_section.find('span', class_=['citation', 'reference'])
                if citation_elem:
                    metadata['citation'] = citation_elem.get_text().strip()
            
            # Construct resource dictionary
            resource = {
                'handle': item_url.split('/')[-1],
                'title': title,
                'abstract': metadata['abstract'],
                'authors': '; '.join(metadata['authors']) if metadata['authors'] else '',
                'type': content_type,
                'source': 'knowhub',
                'collection': content_type,
                'date_issue': metadata['date'],
                'doi': metadata['doi'],
                'citation': metadata['citation'],
                'keywords': '; '.join(metadata['keywords']) if metadata['keywords'] else '',
                'publication_year': metadata['date'][:4] if metadata['date'] else None
            }
            
            return resource
        
        except Exception as e:
            logger.error(f"Error extracting details from {item_url}: {e}")
            return None

    def _extract_collection_items(self, collection_url):
        """Extract items from a collection page"""
        try:
            # Fetch the collection page
            page_content = self._fetch_page(collection_url)
            if not page_content:
                logger.warning(f"Could not fetch content from {collection_url}")
                return []
            
            # Parse the page
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Find all item links
            item_links = soup.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
            
            # Extract unique item links
            unique_item_links = set()
            for link in item_links:
                href = link.get('href')
                if href and href not in self.seen_handles:
                    full_url = urljoin(self.base_url, href)
                    unique_item_links.add(full_url)
                    self.seen_handles.add(href)
            
            # Process items
            items = []
            for url in unique_item_links:
                item = self._extract_item_details(url)
                if item:
                    items.append(item)
            
            return items
        
        except Exception as e:
            logger.error(f"Error extracting items from {collection_url}: {e}")
            return []

    def generate_csvs(self):
        """Generate CSV files for each collection"""
        all_collections = {}
        
        # Fetch items from each collection
        for collection_name, collection_url in self.endpoints.items():
            logger.info(f"Fetching items from {collection_name} collection")
            
            # Extract items from the collection
            collection_items = self._extract_collection_items(collection_url)
            
            # Create CSV for this collection
            csv_filename = os.path.join(self.output_dir, f"{collection_name}_export.csv")
            
            if collection_items:
                # Write to CSV
                try:
                    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                        # Determine fieldnames dynamically from first item
                        fieldnames = list(collection_items[0].keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        # Write headers
                        writer.writeheader()
                        
                        # Write rows
                        for item in collection_items:
                            writer.writerow(item)
                    
                    logger.info(f"Created {csv_filename} with {len(collection_items)} items")
                except Exception as e:
                    logger.error(f"Error writing CSV for {collection_name}: {e}")
            else:
                logger.warning(f"No items found for {collection_name}")
            
            # Store collection items
            all_collections[collection_name] = collection_items

        return all_collections

def main():
    """
    Main function to generate CSV exports from KnowHub
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        # Create scraper instance
        scraper = KnowhubCSVScraper()
        
        # Generate CSVs for all collections
        all_collections = scraper.generate_csvs()
        
        # Print summary
        elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info("CSV export completed.")
        
        # Print collection-wise breakdown
        for collection_type, resources in all_collections.items():
            logger.info(f"{collection_type.capitalize()}: {len(resources)} resources")
        
        logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"An error occurred during CSV export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()