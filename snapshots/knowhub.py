import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import psycopg2
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
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
        cur.close()
        conn.close()
        return False
    
    logger.info("resources_resource table found.")
    cur.close()
    conn.close()
    return True

def insert_resource(item):
    """Insert work data into resources_resource table"""
    if not item:
        return None
    
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    try:
        # Check if a work with this URL already exists
        if item.get("url"):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get("url"),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                cur.close()
                conn.close()
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON
        authors = None
        if item.get("authors"):
            authors = json.dumps(item.get("authors"))
        
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
            item.get("type"),
            authors,
            item.get("publication_year"),
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
        conn.rollback()
        logger.error(f"Error inserting resource: {e}")
        return None
    
    finally:
        cur.close()
        conn.close()

def scrape_knowhub():
    """Scrape KnowHub using Selenium"""
    # URL of the page
    base_url = 'https://knowhub.aphrc.org'
    endpoints = {
        'publications': f"{base_url}/handle/123456789/1",
        'documents': f"{base_url}/handle/123456789/2",
        'reports': f"{base_url}/handle/123456789/3",
        'multimedia': f"{base_url}/handle/123456789/4"
    }
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    
    # Initialize the WebDriver
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)  # 20 second wait
        logger.info("Chrome WebDriver initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Chrome WebDriver: {e}")
        return
    
    total_items = 0
    
    try:
        # Check database table
        if not check_resources_table():
            driver.quit()
            return
        
        # Process each resource type
        for resource_type, url in endpoints.items():
            logger.info(f"Processing {resource_type} from {url}")
            
            try:
                driver.get(url)
                time.sleep(5)  # Allow page to load
                logger.info(f"Page title: {driver.title}")
                
                # Try different selectors
                selectors = [
                    '.artifact-description',
                    '.ds-artifact-item',
                    '.recent-submissions li',
                    'div.item',
                    'table.ds-table tr'
                ]
                
                items_found = False
                
                for selector in selectors:
                    try:
                        logger.info(f"Trying selector: {selector}")
                        items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
                        
                        if items:
                            logger.info(f"Found {len(items)} items with selector {selector}")
                            items_found = True
                            
                            for item in items:
                                try:
                                    # Extract title
                                    title = None
                                    try:
                                        title_elem = item.find_element(By.CSS_SELECTOR, 'h4 a, h3 a, a.title, a[href*="handle"]')
                                        title = title_elem.text.strip()
                                        # Get URL
                                        url = title_elem.get_attribute('href')
                                    except NoSuchElementException:
                                        # Try alternate selectors
                                        try:
                                            title_elem = item.find_element(By.CSS_SELECTOR, 'a')
                                            title = title_elem.text.strip()
                                            url = title_elem.get_attribute('href')
                                        except:
                                            logger.warning("Could not find title or URL, skipping item")
                                            continue
                                    
                                    if not title or not url:
                                        continue
                                    
                                    # Extract authors
                                    authors = []
                                    try:
                                        # Try different author selectors
                                        for author_selector in ['.artifact-author', '.author', '.creators']:
                                            try:
                                                author_elem = item.find_element(By.CSS_SELECTOR, author_selector)
                                                author_text = author_elem.text.strip()
                                                if author_text:
                                                    # Split and clean author names
                                                    author_list = [name.strip() for name in author_text.split(';')]
                                                    authors = [{"name": name} for name in author_list if name]
                                                    break
                                            except:
                                                continue
                                    except:
                                        pass
                                    
                                    # Extract description/abstract
                                    description = ""
                                    try:
                                        for desc_selector in ['.artifact-abstract', '.abstract', '.description']:
                                            try:
                                                desc_elem = item.find_element(By.CSS_SELECTOR, desc_selector)
                                                description = desc_elem.text.strip()
                                                if description:
                                                    break
                                            except:
                                                continue
                                    except:
                                        pass
                                    
                                    # Extract date/year
                                    publication_year = None
                                    try:
                                        for date_selector in ['.date', '.issued', '.artifact-date']:
                                            try:
                                                date_elem = item.find_element(By.CSS_SELECTOR, date_selector)
                                                date_text = date_elem.text.strip()
                                                # Extract year using regex
                                                import re
                                                year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
                                                if year_match:
                                                    publication_year = int(year_match.group(0))
                                                    break
                                            except:
                                                continue
                                    except:
                                        pass
                                    
                                    # Create resource object
                                    resource = {
                                        "title": title,
                                        "description": description,
                                        "url": url,
                                        "authors": authors,
                                        "publication_year": publication_year,
                                        "type": resource_type,
                                        "source": "knowhub"
                                    }
                                    
                                    # Insert into database
                                    result = insert_resource(resource)
                                    if result:
                                        total_items += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error processing item: {e}")
                            
                            # Break out of selector loop once we find items
                            break
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for selector {selector}")
                    except Exception as e:
                        logger.error(f"Error with selector {selector}: {e}")
                
                if not items_found:
                    logger.warning(f"No items found for {resource_type}")
                
            except Exception as e:
                logger.error(f"Error processing {resource_type}: {e}")
    
    except Exception as e:
        logger.error(f"Error in scrape_knowhub: {e}")
    
    finally:
        driver.quit()
        logger.info(f"Scraping completed. Total items stored: {total_items}")

if __name__ == "__main__":
    scrape_knowhub()