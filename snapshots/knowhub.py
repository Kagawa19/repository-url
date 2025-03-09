from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import psycopg2
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": "localhost",
    "port": "5432"
}

def insert_resource(item):
    """Insert resource into database"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    try:
        # Check if resource already exists
        if item.get("url"):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get("url"),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource already exists: {item.get('title')}")
                return result[0]
        
        # Get next ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON
        authors_json = None
        if item.get("authors"):
            authors_json = json.dumps([{"name": author} for author in [item.get("authors")]])
        
        # Insert resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, publication_year, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            next_id,
            item.get("title"),
            item.get("description"),
            item.get("url"),
            item.get("type", "document"),
            authors_json,
            None,  # publication_year
            "knowhub"
        ))
        
        result = cur.fetchone()
        conn.commit()
        logger.info(f"Inserted resource: {item.get('title')}")
        return result[0]
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting resource: {e}")
        return None
    
    finally:
        cur.close()
        conn.close()

def fetch_publications(url, resource_type):
    """
    Simplified approach to fetch publications
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Initialize the WebDriver - without specifying a path
    driver = webdriver.Chrome(options=chrome_options)
    
    # Set page load timeout to 30 seconds
    driver.set_page_load_timeout(30)
    
    publications = []
    
    try:
        logger.info(f"Accessing URL: {url}")
        driver.get(url)
        
        # Wait for page to load
        logger.info("Waiting for page to load")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Scroll to load all content
        logger.info("Scrolling to load content")
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):  # Limit to 3 scrolls to prevent hanging
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Find publication items
        logger.info("Looking for publication items")
        items = driver.find_elements(By.CLASS_NAME, 'artifact-description')
        
        if not items:
            logger.info("Trying alternative selectors")
            items = driver.find_elements(By.CSS_SELECTOR, '.ds-artifact-item, .item')
        
        logger.info(f"Found {len(items)} items")
        
        for item in items:
            try:
                # Extract title
                title = "No title"
                try:
                    title_tag = item.find_element(By.CLASS_NAME, 'artifact-title')
                    title = title_tag.text.strip()
                except:
                    try:
                        title_tag = item.find_element(By.CSS_SELECTOR, 'h4 a, h3 a')
                        title = title_tag.text.strip()
                    except:
                        pass
                
                # Extract URL
                url = None
                try:
                    url_tag = item.find_element(By.CSS_SELECTOR, 'a[href*="handle"]')
                    url = url_tag.get_attribute('href')
                except:
                    pass
                
                # Skip if no title or URL
                if not title or not url:
                    continue
                
                # Extract authors
                authors = "Unknown"
                try:
                    authors_tag = item.find_element(By.CLASS_NAME, 'artifact-author')
                    authors = authors_tag.text.strip()
                except:
                    try:
                        authors_tag = item.find_element(By.CLASS_NAME, 'author')
                        authors = authors_tag.text.strip()
                    except:
                        pass
                
                # Extract description
                description = ""
                try:
                    desc_tag = item.find_element(By.CLASS_NAME, 'description')
                    description = desc_tag.text.strip()
                except:
                    try:
                        desc_tag = item.find_element(By.CLASS_NAME, 'abstract')
                        description = desc_tag.text.strip()
                    except:
                        pass
                
                # Create publication record
                publication = {
                    'title': title,
                    'url': url,
                    'authors': authors,
                    'description': description,
                    'type': resource_type
                }
                
                publications.append(publication)
                logger.info(f"Found publication: {title}")
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
        
    except Exception as e:
        logger.error(f"Error fetching publications: {e}")
    
    finally:
        driver.quit()
    
    return publications

def main():
    # Define endpoints
    endpoints = {
        'publications': 'https://knowhub.aphrc.org/handle/123456789/1',
        'documents': 'https://knowhub.aphrc.org/handle/123456789/2',
        'reports': 'https://knowhub.aphrc.org/handle/123456789/3',
        'multimedia': 'https://knowhub.aphrc.org/handle/123456789/4'
    }
    
    total_stored = 0
    
    for resource_type, url in endpoints.items():
        try:
            logger.info(f"Processing {resource_type} from {url}")
            publications = fetch_publications(url, resource_type)
            
            if publications:
                logger.info(f"Found {len(publications)} {resource_type}")
                
                # Insert into database
                for pub in publications:
                    if insert_resource(pub):
                        total_stored += 1
                
                logger.info(f"Stored {total_stored} {resource_type}")
            else:
                logger.warning(f"No {resource_type} found")
        
        except Exception as e:
            logger.error(f"Error processing {resource_type}: {e}")
    
    logger.info(f"Total stored: {total_stored} resources")

if __name__ == "__main__":
    main()