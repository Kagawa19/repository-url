import requests
from bs4 import BeautifulSoup

class KnowhubScraper:
    def __init__(self, base_url='https://knowhub.aphrc.org'):
        self.base_url = base_url
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        # Set proper headers to make request more likely to succeed
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
    
    def fetch_publications(self, endpoint):
        url = self.endpoints.get(endpoint)
        if not url:
            print(f"Endpoint '{endpoint}' not found.")
            return []
        
        try:
            print(f"Fetching {endpoint} from {url}")
            response = requests.get(url, headers=self.headers, timeout=30, verify=False)
            
            if response.status_code == 200:
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract publications from the HTML
                publications = []
                
                # Try different selectors that might contain publication listings
                items = soup.select('.artifact-description') or \
                       soup.select('.ds-artifact-item') or \
                       soup.select('.recent-submissions li')
                
                for item in items:
                    try:
                        # Extract title and URL
                        title_elem = item.select_one('h4 a, h3 a, .artifact-title a, a[href*="handle"]')
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        url = title_elem.get('href')
                        if url and not url.startswith('http'):
                            url = f"{self.base_url}{url}"
                        
                        # Extract authors
                        authors = []
                        author_elem = item.select_one('.artifact-author, .authors, .creator')
                        if author_elem:
                            authors_text = author_elem.text.strip()
                            # Split by common separators
                            for separator in [';', ',', ' and ']:
                                if separator in authors_text:
                                    authors = [name.strip() for name in authors_text.split(separator) if name.strip()]
                                    break
                            # If no separators found, use the whole string
                            if not authors:
                                authors = [authors_text]
                        
                        # Extract description/abstract
                        description = ""
                        desc_elem = item.select_one('.artifact-abstract, .abstract, .description')
                        if desc_elem:
                            description = desc_elem.text.strip()
                        
                        publications.append({
                            'title': title,
                            'url': url,
                            'authors': authors,
                            'description': description,
                            'type': endpoint
                        })
                        
                    except Exception as e:
                        print(f"Error extracting publication: {e}")
                
                return publications
            else:
                print(f"Failed to fetch publications. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            return []

# Example usage
scraper = KnowhubScraper()

for endpoint in ['publications', 'documents', 'reports', 'multimedia']:
    publications = scraper.fetch_publications(endpoint)
    print(f"\nFound {len(publications)} {endpoint}:")
    
    for pub in publications[:5]:  # Show first 5 only
        print(f"Title: {pub['title']}")
        print(f"URL: {pub['url']}")
        print(f"Authors: {', '.join(pub['authors'])}")
        print(f"Description: {pub['description'][:100]}..." if pub['description'] else "No description")
        print()