import requests
from bs4 import BeautifulSoup
import re
import logging
from typing import List, Dict
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APHRCAnalyzer:
    def __init__(self, base_url="https://aphrc.org/publications/"):
        self.base_url = base_url
        self.session = requests.Session()
        # Add headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def analyze_structure(self):
        """Analyze the overall structure of the publications page"""
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Analyze link patterns
            logger.info("\n=== Analyzing Link Patterns ===")
            self._analyze_links(soup)

            # Analyze content containers
            logger.info("\n=== Analyzing Content Containers ===")
            self._analyze_containers(soup)

            # Analyze publication elements
            logger.info("\n=== Analyzing Publication Elements ===")
            self._analyze_publication_elements(soup)

            # Sample a publication page for detailed analysis
            self._analyze_sample_publication(soup)

        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")

    def _analyze_links(self, soup):
        """Analyze all link patterns on the page"""
        links = soup.find_all('a', href=True)
        
        url_patterns = {}
        for link in links:
            href = link['href']
            # Get the path pattern
            path = href.split('?')[0]
            url_patterns[path] = url_patterns.get(path, 0) + 1

        logger.info("\nURL Patterns Found:")
        for pattern, count in sorted(url_patterns.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"Pattern: {pattern:<50} Count: {count}")

    def _analyze_containers(self, soup):
        """Analyze content container patterns"""
        containers = soup.find_all(['div', 'section', 'article'])
        
        container_patterns = {}
        for container in containers:
            if container.get('class'):
                pattern = f"{container.name} (class: {' '.join(container['class'])})"
                container_patterns[pattern] = container_patterns.get(pattern, 0) + 1

        logger.info("\nContainer Patterns Found:")
        for pattern, count in sorted(container_patterns.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"Container: {pattern:<50} Count: {count}")

    def _analyze_publication_elements(self, soup):
        """Analyze specific publication-related elements"""
        # Look for title patterns
        logger.info("\nTitle Patterns:")
        for title in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            class_info = f" (class: {' '.join(title['class'])})" if title.get('class') else ""
            logger.info(f"Tag: {title.name}{class_info}")
            logger.info(f"Text: {title.text.strip()[:100]}")
            logger.info("-" * 50)

        # Look for date patterns
        logger.info("\nDate Patterns:")
        date_elements = soup.find_all(text=re.compile(r'\d{4}'))
        for date in date_elements[:5]:  # Show first 5 examples
            logger.info(f"Date text: {date.strip()}")
            if hasattr(date.parent, 'name'):
                logger.info(f"Parent tag: {date.parent.name}")
                logger.info(f"Parent classes: {date.parent.get('class', [])}")
            logger.info("-" * 50)

    def _analyze_sample_publication(self, soup):
        """Analyze a sample publication page in detail"""
        # Try to find a publication link
        publication_links = soup.find_all('a', href=re.compile(r'publication|research|policy-brief', re.I))
        
        if publication_links:
            sample_url = publication_links[0]['href']
            if not sample_url.startswith('http'):
                sample_url = urljoin(self.base_url, sample_url)

            try:
                logger.info(f"\nAnalyzing sample publication: {sample_url}")
                response = self.session.get(sample_url)
                response.raise_for_status()
                pub_soup = BeautifulSoup(response.text, 'html.parser')

                logger.info("\nPublication Page Structure:")
                # Analyze main content area
                main_content = pub_soup.find(['article', 'main', 'div'], 
                                           class_=re.compile(r'content|main|publication', re.I))
                if main_content:
                    logger.info(f"Main content container: {main_content.name} "
                              f"(class: {' '.join(main_content.get('class', []))})")

                # Look for metadata
                logger.info("\nMetadata elements:")
                metadata_elements = pub_soup.find_all(['div', 'span', 'p'], 
                                                    class_=re.compile(r'meta|info|details', re.I))
                for elem in metadata_elements:
                    logger.info(f"Tag: {elem.name}")
                    logger.info(f"Classes: {elem.get('class', [])}")
                    logger.info(f"Text: {elem.text.strip()[:100]}")
                    logger.info("-" * 50)

            except Exception as e:
                logger.error(f"Error analyzing sample publication: {e}")

    def analyze_pagination(self):
        """Analyze pagination structure if it exists"""
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            logger.info("\n=== Analyzing Pagination ===")
            pagination = soup.find_all(['div', 'nav', 'ul'], 
                                     class_=re.compile(r'pagination|pager|pages', re.I))
            
            for page_elem in pagination:
                logger.info(f"\nPagination container: {page_elem.name} "
                          f"(class: {' '.join(page_elem.get('class', []))})")
                
                # Analyze page links
                page_links = page_elem.find_all('a', href=True)
                logger.info("Page link patterns:")
                for link in page_links:
                    logger.info(f"URL: {link['href']}")
                    logger.info(f"Text: {link.text.strip()}")
                    logger.info("-" * 30)

        except Exception as e:
            logger.error(f"Error analyzing pagination: {e}")

def main():
    analyzer = APHRCAnalyzer()
    analyzer.analyze_structure()
    analyzer.analyze_pagination()

if __name__ == "__main__":
    main()