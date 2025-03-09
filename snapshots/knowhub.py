from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import time
import psycopg2
import os
import json
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("knowhub_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": "localhost",
    "port": "5432"
}

def check_resources_table():
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

def insert_resource(item):
    """Insert resource data into resources_resource table"""
    if not item:
        return None
    
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Check if a resource with this URL already exists
        if item.get("url"):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get("url"),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON if needed
        authors_json = None
        if item.get("authors"):
            # Convert authors to JSON format expected by the database
            authors_json = json.dumps([{"name": author} for author in item.get("authors")])
        
        # Extract publication_year from publication_date if available
        publication_year = None
        if item.get("publication_date"):
            try:
                # Try to extract a year from the date string
                year_match = re.search(r'\b(19|20)\d{2}\b', item.get("publication_date", ""))
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
            item.get("type", "other"),  # Use the type of publication
            authors_json,
            publication_year,
            item.get("source", "knowhub")
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

def fetch_publications(url, resource_type=None, max_retries=3, page_load_timeout=60):
    """
    Fetch publications from the provided URL with robust timeout handling.
    
    Args:
        url (str): The URL of the page to scrape.
        resource_type (str): Type of resource being scraped.
        max_retries (int): Maximum number of retries for timeout errors.
        page_load_timeout (int): Timeout in seconds for page loading.
        
    Returns:
        List[dict]: A list of dictionaries containing publication details.
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--ignore-certificate-errors')
    
    # Initialize the WebDriver
    publications = []
    driver = None
    
    try:
        logger.info(f"Initializing Chrome WebDriver for {url}")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Set page load timeout
        driver.set_page_load_timeout(page_load_timeout)
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Attempt {retry_count + 1}: Loading URL {url}")
                driver.get(url)
                
                # Wait for the page to load
                logger.info("Waiting for page to load...")
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                
                # Log page title for debugging
                logger.info(f"Page loaded. Title: {driver.title}")
                
                # First wait for any major container to appear
                logger.info("Looking for content containers...")
                selectors = [
                    '.artifact-description',
                    '.ds-artifact-item',
                    '.artifact-title',
                    'div.item', 
                    'h4.artifact-title'
                ]
                
                for selector in selectors:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        logger.info(f"Found container with selector: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                # Let the page fully settle
                time.sleep(3)
                
                # Handle scrolling to load all content
                logger.info("Scrolling to load all content...")
                last_height = driver.execute_script("return document.body.scrollHeight")
                scroll_attempts = 0
                max_scroll_attempts = 5  # Limit scrolling attempts
                
                while scroll_attempts < max_scroll_attempts:
                    # Scroll down
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)  # Wait for content to load
                    
                    # Calculate new scroll height and compare with last scroll height
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                    scroll_attempts += 1
                
                logger.info(f"Completed {scroll_attempts} scroll attempts")
                
                # Try different selectors to find items
                item_selectors = [
                    'div.artifact-description',
                    'div.ds-artifact-item',
                    'div.recent-submissions li',
                    'table.ds-table tr'
                ]
                
                items = []
                successful_selector = None
                
                for selector in item_selectors:
                    try:
                        logger.info(f"Trying to find items with selector: {selector}")
                        items = driver.find_elements(By.CSS_SELECTOR, selector)
                        if items:
                            logger.info(f"Found {len(items)} items with selector: {selector}")
                            successful_selector = selector
                            break
                    except Exception as e:
                        logger.warning(f"Error finding items with selector {selector}: {e}")
                
                if not items:
                    logger.warning("No items found with any selector")
                    break
                
                # Process items
                logger.info(f"Processing {len(items)} items...")
                for i, item in enumerate(items):
                    try:
                        publication = {}
                        
                        # Extract title and URL
                        try:
                            title_selectors = ['h4 a', 'h3 a', '.artifact-title a', 'a[href*="handle"]']
                            title_element = None
                            
                            for title_selector in title_selectors:
                                try:
                                    title_element = item.find_element(By.CSS_SELECTOR, title_selector)
                                    if title_element:
                                        break
                                except:
                                    continue
                            
                            if title_element:
                                publication['title'] = title_element.text.strip()
                                publication['url'] = title_element.get_attribute('href')
                                logger.info(f"Found item {i+1}: {publication['title']}")
                            else:
                                logger.warning(f"No title found for item {i+1}, skipping")
                                continue
                            
                        except Exception as e:
                            logger.error(f"Error extracting title for item {i+1}: {e}")
                            continue
                        
                        # Extract authors
                        try:
                            author_selectors = ['.artifact-author', '.author', '.creators', '.ds-artifact-author']
                            authors = []
                            
                            for author_selector in author_selectors:
                                try:
                                    author_element = item.find_element(By.CSS_SELECTOR, author_selector)
                                    author_text = author_element.text.strip()
                                    
                                    # Split by common separators
                                    if ';' in author_text:
                                        authors = [a.strip() for a in author_text.split(';') if a.strip()]
                                    elif ',' in author_text:
                                        authors = [a.strip() for a in author_text.split(',') if a.strip()]
                                    else:
                                        authors = [author_text]
                                    
                                    if authors:
                                        break
                                except:
                                    continue
                            
                            publication['authors'] = authors
                            
                        except Exception as e:
                            logger.error(f"Error extracting authors for item {i+1}: {e}")
                            publication['authors'] = []
                        
                        # Extract publication date
                        try:
                            date_selectors = ['.date', '.artifact-date', '.issued', '.ds-artifact-date']
                            for date_selector in date_selectors:
                                try:
                                    date_element = item.find_element(By.CSS_SELECTOR, date_selector)
                                    publication['publication_date'] = date_element.text.strip()
                                    break
                                except:
                                    continue
                            
                        except Exception as e:
                            logger.error(f"Error extracting date for item {i+1}: {e}")
                            publication['publication_date'] = None
                        
                        # Extract description
                        try:
                            desc_selectors = ['.artifact-abstract', '.abstract', '.description', '.ds-artifact-abstract']
                            for desc_selector in desc_selectors:
                                try:
                                    desc_element = item.find_element(By.CSS_SELECTOR, desc_selector)
                                    publication['description'] = desc_element.text.strip()
                                    break
                                except:
                                    continue
                            
                        except Exception as e:
                            logger.error(f"Error extracting description for item {i+1}: {e}")
                            publication['description'] = None
                        
                        # Extract publication type
                        try:
                            type_selectors = ['.type', '.artifact-type', '.ds-artifact-type']
                            publication_type = resource_type  # Default to the resource_type parameter
                            
                            for type_selector in type_selectors:
                                try:
                                    type_element = item.find_element(By.CSS_SELECTOR, type_selector)
                                    element_type = type_element.text.strip()
                                    if element_type:
                                        publication_type = element_type
                                        break
                                except:
                                    continue
                            
                            publication['type'] = publication_type
                            
                        except Exception as e:
                            logger.error(f"Error extracting type for item {i+1}: {e}")
                            publication['type'] = resource_type
                        
                        # Add source
                        publication['source'] = 'knowhub'
                        
                        # Add to publications list
                        publications.append(publication)
                        
                    except StaleElementReferenceException:
                        logger.warning(f"Stale element encountered for item {i+1}, skipping")
                    except Exception as e:
                        logger.error(f"Error processing item {i+1}: {e}")
                
                logger.info(f"Successfully extracted {len(publications)} publications")
                break  # Success, exit retry loop
                
            except TimeoutException:
                logger.warning(f"Timeout loading {url}, attempt {retry_count + 1}/{max_retries}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to load {url} after {max_retries} attempts")
                else:
                    logger.info("Retrying...")
                    driver.quit()
                    driver = webdriver.Chrome(options=chrome_options)
                    
            except Exception as e:
                logger.error(f"Error loading {url}: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error initializing WebDriver: {e}")
    
    finally:
        if driver:
            driver.quit()
    
    return publications

def scrape_knowhub():
    """Scrape KnowHub resources and store in database."""
    # URL mappings
    endpoints = {
        'publications': 'https://knowhub.aphrc.org/handle/123456789/1',
        'documents': 'https://knowhub.aphrc.org/handle/123456789/2',
        'reports': 'https://knowhub.aphrc.org/handle/123456789/3',
        'multimedia': 'https://knowhub.aphrc.org/handle/123456789/4'
    }
    
    # Check resources table
    if not check_resources_table():
        logger.error("Database table check failed")
        return
    
    total_stored = 0
    
    # Process each endpoint
    for resource_type, url in endpoints.items():
        try:
            logger.info(f"\nProcessing {resource_type} from {url}")
            
            # Fetch publications
            publications = fetch_publications(url, resource_type=resource_type)
            
            if not publications:
                logger.warning(f"No {resource_type} found")
                continue
            
            logger.info(f"Found {len(publications)} {resource_type}, inserting into database...")
            
            # Store publications
            stored_count = 0
            for pub in publications:
                result_id = insert_resource(pub)
                if result_id:
                    stored_count += 1
            
            logger.info(f"Stored {stored_count} out of {len(publications)} {resource_type}")
            total_stored += stored_count
            
        except Exception as e:
            logger.error(f"Error processing {resource_type}: {e}")
    
    logger.info(f"Total stored: {total_stored} resources")

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        logger.info("Starting KnowHub scraper")
        scrape_knowhub()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()