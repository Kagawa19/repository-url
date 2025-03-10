import requests
from bs4 import BeautifulSoup
import re
import time
import json
import os
from urllib.parse import urljoin, urlparse, parse_qs

class KnowhubInspector:
    """Diagnostic tool to inspect KnowHub repository structure."""
    
    def __init__(self, base_url="https://knowhub.aphrc.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        self.results = {
            'site_structure': {},
            'pagination_info': {},
            'collection_info': {},
            'navigation_urls': {}
        }
    
    def fetch_page(self, url):
        """Fetch a page with basic error handling."""
        print(f"Fetching URL: {url}")
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=10,
                verify=False
            )
            if response.status_code == 200:
                return response.text
            else:
                print(f"Failed to fetch {url}: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def analyze_collections(self, endpoint_name, url):
        """Analyze the collections and their item counts."""
        html_content = self.fetch_page(url)
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find collection names and counts
        collections = {}
        
        # Look for typical DSpace collection pattern - links with counts in brackets
        collection_items = soup.find_all(string=re.compile(r'\[\d+\]$'))
        for item in collection_items:
            # Get the parent element that contains the link
            parent = item.parent
            if parent.name == 'a':
                link = parent
            else:
                link = parent.find('a')
                
            if link and link.get('href'):
                name = link.get_text().strip()
                count_match = re.search(r'\[(\d+)\]$', name)
                if count_match:
                    count = int(count_match.group(1))
                    clean_name = name.replace(count_match.group(0), '').strip()
                    collections[clean_name] = {
                        'count': count,
                        'url': urljoin(self.base_url, link.get('href'))
                    }
        
        # Look for another pattern - lists of communities/collections
        community_lists = soup.find_all('ul', class_=lambda c: c and 'list-group' in c)
        for list_elem in community_lists:
            for link in list_elem.find_all('a'):
                name = link.get_text().strip()
                href = link.get('href')
                if href:
                    collections[name] = {
                        'url': urljoin(self.base_url, href),
                        'count': 'unknown'  # Count might not be visible here
                    }
        
        # Look for subcommunities
        subcommunity_header = soup.find(string=re.compile('Sub-communities', re.IGNORECASE))
        if subcommunity_header:
            subcommunity_section = subcommunity_header.find_parent('div') or subcommunity_header.find_parent('section')
            if subcommunity_section:
                for link in subcommunity_section.find_all('a'):
                    name = link.get_text().strip()
                    href = link.get('href')
                    count_match = re.search(r'\[(\d+)\]$', name)
                    if href and count_match:
                        count = int(count_match.group(1))
                        clean_name = name.replace(count_match.group(0), '').strip()
                        collections[f"Subcommunity: {clean_name}"] = {
                            'count': count,
                            'url': urljoin(self.base_url, href)
                        }
        
        # Record recent submissions if present
        recent_submissions = soup.find(string=re.compile('Recent Submissions', re.IGNORECASE))
        if recent_submissions:
            recent_section = recent_submissions.find_parent('div') or recent_submissions.find_parent('section')
            if recent_section:
                recent_links = []
                for link in recent_section.find_all('a'):
                    href = link.get('href')
                    if href and '/handle/' in href:
                        recent_links.append({
                            'title': link.get_text().strip(),
                            'url': urljoin(self.base_url, href)
                        })
                if recent_links:
                    collections['Recent Submissions'] = {
                        'items': recent_links,
                        'count': len(recent_links)
                    }
        
        return collections
    
    def detect_pagination(self, html_content):
        """Detect pagination mechanisms in the HTML."""
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        pagination_info = {
            'has_pagination': False,
            'next_link': None,
            'prev_link': None,
            'pagination_type': 'none',
            'current_page': 1,
            'total_pages': 1,
            'items_per_page': 0,
            'total_items': 0,
            'offset_params': {},
            'page_params': {}
        }
        
        # Check for pagination elements
        pagination_elements = soup.find_all(['ul', 'div', 'nav'], class_=lambda c: c and re.search(r'pagination|pager', c, re.IGNORECASE))
        
        if pagination_elements:
            pagination_info['has_pagination'] = True
            for pagination in pagination_elements:
                # Look for next/previous links
                next_link = pagination.find('a', string=re.compile(r'next|›|»', re.IGNORECASE)) or \
                           pagination.find('a', title=re.compile(r'next', re.IGNORECASE))
                
                prev_link = pagination.find('a', string=re.compile(r'prev|previous|‹|«', re.IGNORECASE)) or \
                           pagination.find('a', title=re.compile(r'prev|previous', re.IGNORECASE))
                
                if next_link and next_link.get('href'):
                    pagination_info['next_link'] = urljoin(self.base_url, next_link.get('href'))
                    
                    # Parse the URL to get pagination parameters
                    parsed_url = urlparse(pagination_info['next_link'])
                    query_params = parse_qs(parsed_url.query)
                    
                    if 'page' in query_params:
                        pagination_info['pagination_type'] = 'page'
                        pagination_info['page_params'] = {'page': query_params['page'][0]}
                    elif 'offset' in query_params:
                        pagination_info['pagination_type'] = 'offset'
                        pagination_info['offset_params'] = {'offset': query_params['offset'][0]}
                        # Try to determine items per page from offset
                        try:
                            offset = int(query_params['offset'][0])
                            if offset > 0:
                                pagination_info['items_per_page'] = offset
                        except ValueError:
                            pass
                
                if prev_link and prev_link.get('href'):
                    pagination_info['prev_link'] = urljoin(self.base_url, prev_link.get('href'))
                
                # Look for page numbers
                page_numbers = []
                for a in pagination.find_all('a'):
                    try:
                        # Check if the text is a number
                        if a.get_text().strip().isdigit():
                            page_numbers.append(int(a.get_text().strip()))
                    except:
                        pass
                
                if page_numbers:
                    pagination_info['total_pages'] = max(page_numbers)
        
        # Check for "showing X-Y of Z results" text
        results_text = soup.find(string=re.compile(r'showing\s+\d+\s*-\s*\d+\s+of\s+\d+', re.IGNORECASE))
        if results_text:
            match = re.search(r'showing\s+(\d+)\s*-\s*(\d+)\s+of\s+(\d+)', results_text, re.IGNORECASE)
            if match:
                start, end, total = int(match.group(1)), int(match.group(2)), int(match.group(3))
                pagination_info['items_per_page'] = end - start + 1
                pagination_info['total_items'] = total
                pagination_info['has_pagination'] = total > (end - start + 1)
        
        # Alternative: find the count of items on the page
        items_count = len(soup.find_all('div', class_=lambda c: c and re.search(r'item|artifact', c, re.IGNORECASE)))
        if items_count > 0:
            pagination_info['items_per_page'] = items_count
        
        return pagination_info
    
    def analyze_endpoint(self, name, url):
        """Thoroughly analyze a specific endpoint."""
        html_content = self.fetch_page(url)
        if not html_content:
            return
        
        # Analyze the structure and pagination
        self.results['site_structure'][name] = {
            'url': url,
            'page_title': BeautifulSoup(html_content, 'html.parser').title.string if BeautifulSoup(html_content, 'html.parser').title else 'Unknown'
        }
        
        # Detect pagination
        pagination_info = self.detect_pagination(html_content)
        self.results['pagination_info'][name] = pagination_info
        
        # Analyze collections and item counts
        collections = self.analyze_collections(name, url)
        self.results['collection_info'][name] = collections
        
        # If there's pagination, fetch the next page to see how it works
        if pagination_info['next_link']:
            print(f"Testing pagination for {name} with next link: {pagination_info['next_link']}")
            next_page_content = self.fetch_page(pagination_info['next_link'])
            if next_page_content:
                # Check if content is different from the first page
                next_page_soup = BeautifulSoup(next_page_content, 'html.parser')
                first_page_soup = BeautifulSoup(html_content, 'html.parser')
                
                # Compare titles as a simple check
                first_titles = [h.get_text() for h in first_page_soup.find_all(['h3', 'h4'])]
                next_titles = [h.get_text() for h in next_page_soup.find_all(['h3', 'h4'])]
                
                pagination_info['pagination_works'] = first_titles != next_titles
                
                # Detect pagination on the next page as well
                next_pagination = self.detect_pagination(next_page_content)
                pagination_info['next_page_pagination'] = next_pagination
        
        # Look for navigation URLs and handles
        self.extract_navigation_urls(name, html_content)
    
    def extract_navigation_urls(self, name, html_content):
        """Extract navigation URLs and handle patterns from the page."""
        if not html_content:
            return
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all links that contain "handle" in the href
        handle_links = {}
        for link in soup.find_all('a', href=re.compile(r'/handle/')):
            href = link.get('href')
            if href:
                full_url = urljoin(self.base_url, href)
                handle_match = re.search(r'/handle/([0-9/]+)', full_url)
                if handle_match:
                    handle = handle_match.group(1)
                    text = link.get_text().strip()
                    if text:  # Only include links with text
                        if handle not in handle_links:
                            handle_links[handle] = []
                        handle_links[handle].append({
                            'url': full_url,
                            'text': text
                        })
        
        # Find browse options
        browse_links = []
        browse_sections = soup.find_all(string=re.compile('Browse', re.IGNORECASE))
        for browse_section in browse_sections:
            parent = browse_section.find_parent(['div', 'section', 'nav'])
            if parent:
                for link in parent.find_all('a'):
                    href = link.get('href')
                    if href:
                        browse_links.append({
                            'url': urljoin(self.base_url, href),
                            'text': link.get_text().strip()
                        })
        
        # Find search options
        search_forms = []
        for form in soup.find_all('form'):
            action = form.get('action')
            if action:
                inputs = []
                for input_tag in form.find_all(['input', 'select']):
                    name = input_tag.get('name')
                    if name:
                        inputs.append({
                            'name': name,
                            'type': input_tag.get('type', 'text')
                        })
                search_forms.append({
                    'action': urljoin(self.base_url, action),
                    'method': form.get('method', 'get'),
                    'inputs': inputs
                })
        
        self.results['navigation_urls'][name] = {
            'handle_links': handle_links,
            'browse_links': browse_links,
            'search_forms': search_forms
        }
    
    def run_full_inspection(self):
        """Run a full inspection of all endpoints."""
        print(f"Starting inspection of KnowHub repository: {self.base_url}")
        for name, url in self.endpoints.items():
            print(f"\nInspecting {name} endpoint: {url}")
            self.analyze_endpoint(name, url)
            time.sleep(1)  # Be nice to the server
        
        # Add the base URL inspection
        print(f"\nInspecting base URL: {self.base_url}")
        self.analyze_endpoint('base', self.base_url)
        
        # Generate a recommendations section based on findings
        self.generate_recommendations()
        
        return self.results
    
    def generate_recommendations(self):
        """Generate recommendations based on the inspection results."""
        self.results['recommendations'] = {
            'strategy': 'unknown',
            'pagination_method': 'unknown',
            'collection_method': 'unknown',
            'suggested_approach': ''
        }
        
        # Check pagination across endpoints
        pagination_found = any(info.get('has_pagination', False) 
                             for info in self.results['pagination_info'].values())
        
        # Check collection structure
        collections_found = any(len(colls) > 0 
                              for colls in self.results['collection_info'].values())
        
        if pagination_found:
            # Determine the pagination method
            pagination_types = [info.get('pagination_type', 'none') 
                             for info in self.results['pagination_info'].values() 
                             if info.get('has_pagination', False)]
            
            if 'page' in pagination_types:
                self.results['recommendations']['pagination_method'] = 'page'
                self.results['recommendations']['strategy'] = 'pagination'
                
                # Get sample URL with page parameter
                for name, info in self.results['pagination_info'].items():
                    if info.get('next_link') and 'page=' in info.get('next_link', ''):
                        self.results['recommendations']['suggested_approach'] = f"Use pagination with ?page=X parameter, example: {info['next_link']}"
                        break
                        
            elif 'offset' in pagination_types:
                self.results['recommendations']['pagination_method'] = 'offset'
                self.results['recommendations']['strategy'] = 'pagination'
                
                # Get sample URL with offset parameter
                for name, info in self.results['pagination_info'].items():
                    if info.get('next_link') and 'offset=' in info.get('next_link', ''):
                        self.results['recommendations']['suggested_approach'] = f"Use pagination with ?offset=X parameter, example: {info['next_link']}"
                        break
        
        if collections_found:
            self.results['recommendations']['collection_method'] = 'hierarchical'
            
            # If we have major collections with significant item counts
            total_items = 0
            for name, collections in self.results['collection_info'].items():
                for coll_name, coll_info in collections.items():
                    if isinstance(coll_info.get('count'), int) and coll_info.get('count', 0) > 10:
                        total_items += coll_info.get('count', 0)
            
            if total_items > 0:
                if self.results['recommendations']['strategy'] == 'unknown':
                    self.results['recommendations']['strategy'] = 'collections'
                    
                self.results['recommendations']['suggested_approach'] += "\nThe repository appears to use a hierarchical collection structure. Consider scraping each collection individually."
        
        # If no clear approach yet, suggest the best option based on our findings
        if self.results['recommendations']['strategy'] == 'unknown':
            # Look for handle patterns
            handle_patterns = {}
            for name, nav_info in self.results['navigation_urls'].items():
                for handle, links in nav_info.get('handle_links', {}).items():
                    if handle not in handle_patterns:
                        handle_patterns[handle] = 0
                    handle_patterns[handle] += len(links)
            
            if handle_patterns:
                # Sort by frequency
                sorted_handles = sorted(handle_patterns.items(), key=lambda x: x[1], reverse=True)
                top_handle = sorted_handles[0][0]
                
                self.results['recommendations']['strategy'] = 'handle_traversal'
                self.results['recommendations']['suggested_approach'] = f"The repository seems to use consistent handle patterns. Consider traversing handles starting with {top_handle}."
            else:
                self.results['recommendations']['strategy'] = 'combined'
                self.results['recommendations']['suggested_approach'] = "No clear strategy detected. Try a combined approach of following both collection links and pagination."
    
    def save_results(self, filename='knowhub_inspection_results.json'):
        """Save the inspection results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

def main():
    inspector = KnowhubInspector()
    inspector.run_full_inspection()
    inspector.save_results()
    
    # Print key findings
    print("\n==== KEY FINDINGS ====")
    for name, pagination in inspector.results['pagination_info'].items():
        has_pagination = pagination.get('has_pagination', False)
        pagination_type = pagination.get('pagination_type', 'none')
        next_link = pagination.get('next_link', None)
        
        print(f"\n{name.upper()}:")
        print(f"  Has pagination: {has_pagination}")
        print(f"  Pagination type: {pagination_type}")
        print(f"  Next link: {next_link}")
        
        # Print collection info
        collections = inspector.results['collection_info'].get(name, {})
        if collections:
            print(f"  Collections found: {len(collections)}")
            for coll_name, coll_info in collections.items():
                if isinstance(coll_info.get('count'), int):
                    print(f"    - {coll_name}: {coll_info.get('count')} items")
    
    # Print recommendations
    print("\n==== RECOMMENDATIONS ====")
    recommendations = inspector.results.get('recommendations', {})
    print(f"Recommended strategy: {recommendations.get('strategy', 'unknown')}")
    print(f"Pagination method: {recommendations.get('pagination_method', 'none')}")
    print(f"Collection method: {recommendations.get('collection_method', 'none')}")
    print("\nSuggested approach:")
    print(recommendations.get('suggested_approach', 'No specific approach recommended.'))

if __name__ == "__main__":
    main()