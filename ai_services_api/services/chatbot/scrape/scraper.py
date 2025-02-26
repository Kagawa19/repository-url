import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urlparse, urljoin
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Initialize global variables
visited_urls = set()
scraped_pages_count = 0
MAX_SCRAPED_PAGES = 1000  # Set max pages to scrape
PDF_FOLDER = 'pdf_files'
JSON_FILE = 'scraped_data.json'

# Create folders if they do not exist
os.makedirs(PDF_FOLDER, exist_ok=True)

# Setup Selenium WebDriver with necessary options
def setup_selenium():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")
    driver = webdriver.Chrome(options=options)
    return driver

# Extract text, links, and metadata from a page using Selenium
def extract_page_data(url, depth, driver, max_depth=4):
    global scraped_pages_count
    if url in visited_urls or depth > max_depth or scraped_pages_count >= MAX_SCRAPED_PAGES:
        return None
    
    visited_urls.add(url)
    scraped_pages_count += 1
    print(f"Scraping URL: {url} (Depth: {depth})")

    try:
        driver.get(url)

        # Scrolling to load dynamic content
        scroll_attempts = 5
        last_height = driver.execute_script("return document.body.scrollHeight")
        while scroll_attempts > 0:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(2)  # Wait for new content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts -= 1

        # Parse page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for tag in soup.find_all(['img', 'link', 'style', 'script']):
            tag.decompose()

        page_text = soup.get_text(separator=" ", strip=True)
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        pdf_links = [link for link in links if link.lower().endswith('.pdf')]

        # Store page data
        page_data = {
            "url": url,
            "depth": depth,
            "content": page_text,
            "links": links,
            "children": [],
            "pdf_links": []
        }

        # Download PDFs
        for pdf_link in pdf_links:
            if download_pdf(pdf_link):
                page_data["pdf_links"].append(pdf_link)

        # Recursively scrape child links
        for link in links:
            if urlparse(link).netloc == urlparse(url).netloc and link not in visited_urls:
                child_data = extract_page_data(link, depth + 1, driver, max_depth)
                if child_data:
                    page_data["children"].append(child_data)

        return page_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Function to download PDF files
def download_pdf(pdf_url):
    pdf_name = os.path.basename(urlparse(pdf_url).path)
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_url} -> {pdf_path}")
            return True
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
    return False

# Function to store all page data in a single JSON file
def save_all_data_to_json(all_data):
    with open(JSON_FILE, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"All data saved to {JSON_FILE}")

# Crawl the website starting from a given URL
def crawl_website(start_url, max_depth=4):
    driver = setup_selenium()
    all_scraped_data = []
    page_data = extract_page_data(start_url, 0, driver, max_depth)
    if page_data:
        all_scraped_data.append(page_data)

    driver.quit()
    
    # Save all the scraped data in one JSON file
    save_all_data_to_json(all_scraped_data)

# Start URL for the crawl
start_url = 'https://aphrc.org'
crawl_website(start_url)
print("Scraping complete. Data saved.")
