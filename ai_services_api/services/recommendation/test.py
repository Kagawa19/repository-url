import aiohttp
import csv
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set

class DomainExpertFinder:
    def __init__(self):
        self.base_url = 'https://api.openalex.org'
        self.logger = logging.getLogger(__name__)
        self.email = "briankimu97@gmail.com"
        self.target_domains = {
            "Life Sciences": [],
            "Health Sciences": [],
            "Physical Sciences": [],
            "Social Sciences": []
        }

    async def _fetch_data(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Helper method to fetch data from OpenAlex API"""
        url = f"{self.base_url}/{endpoint}"
        try:
            params = params or {}
            params['mailto'] = self.email
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                self.logger.error(f"Failed to fetch data. Status: {response.status}")
                return None
        except Exception as e:
            self.logger.error(f"Error in _fetch_data: {e}")
            return None

    async def get_authors_batch(self, session: aiohttp.ClientSession, page: int = 1) -> List[Dict]:
        """Get a batch of authors with ORCIDs"""
        params = {
            'filter': 'has_orcid:true',
            'per-page': 50,
            'page': page,
            'sort': 'cited_by_count:desc'  # Get well-cited authors
        }
        data = await self._fetch_data(session, 'authors', params)
        return data.get('results', []) if data else []

    async def get_expert_domains(self, session: aiohttp.ClientSession, orcid: str) -> List[Dict[str, str]]:
        """Get domains for an expert using your working code logic"""
        if not orcid.startswith('https://orcid.org/'):
            formatted_orcid = f"https://orcid.org/{orcid}"
        else:
            formatted_orcid = orcid

        # First get expert data
        params = {"filter": f"orcid:{formatted_orcid}"}
        expert_data = await self._fetch_data(session, 'authors', params)
        
        if not expert_data or 'results' not in expert_data or not expert_data['results']:
            return []

        # Get their works
        openalex_id = expert_data['results'][0]['id']
        works_params = {
            'filter': f"author.id:{openalex_id}",
            'per-page': 50
        }
        works_data = await self._fetch_data(session, 'works', works_params)

        if not works_data or 'results' not in works_data:
            return []

        # Process domains from works
        domains = set()
        for work in works_data['results']:
            for topic in work.get('topics', []):
                domain = topic.get('domain', {}).get('display_name')
                if domain in self.target_domains:
                    domains.add(domain)

        return [{"domain": d} for d in domains]

    async def find_experts_by_domain(self, session: aiohttp.ClientSession, max_per_domain: int = 5):
        """Find experts for each target domain"""
        page = 1
        max_pages = 10  # Limit the number of pages to try
        
        while page <= max_pages and any(len(experts) < max_per_domain for experts in self.target_domains.values()):
            authors = await self.get_authors_batch(session, page)
            if not authors:
                break

            for author in authors:
                orcid = author.get('orcid')
                if not orcid:
                    continue

                # Get domains for this author
                expert_domains = await self.get_expert_domains(session, orcid)
                name = author.get('display_name', 'Unknown')
                
                # Add author to appropriate domain lists
                for domain_data in expert_domains:
                    domain = domain_data['domain']
                    if domain in self.target_domains and len(self.target_domains[domain]) < max_per_domain:
                        clean_orcid = orcid.replace('https://orcid.org/', '')
                        author_data = {
                            'name': name,
                            'orcid': clean_orcid,
                            'domain': domain
                        }
                        if author_data not in self.target_domains[domain]:
                            self.target_domains[domain].append(author_data)
                            print(f"Added {name} to {domain}")

                # Print current status
                print("\nCurrent counts:")
                for domain, experts in self.target_domains.items():
                    print(f"{domain}: {len(experts)}")
                print()

                # Check if we have enough experts in all domains
                if all(len(experts) >= max_per_domain for experts in self.target_domains.values()):
                    return

            page += 1
            await asyncio.sleep(1)  # Rate limiting

    def save_to_csv(self, filename: str = 'test.csv'):
        """Save all found experts to CSV"""
        all_experts = []
        for experts in self.target_domains.values():
            all_experts.extend(experts)

        if not all_experts:
            print("No experts to save")
            return
            
        fieldnames = ['name', 'orcid', 'domain']
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_experts)
            
        print(f"Saved {len(all_experts)} experts to {filename}")

async def main():
    finder = DomainExpertFinder()
    headers = {
        'User-Agent': f'Research Script (mailto:{finder.email})',
        'Accept': 'application/json'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        await finder.find_experts_by_domain(session)
        finder.save_to_csv()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())