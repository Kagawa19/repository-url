"""
Data Manager for APHRC content.
This module should be placed in: ai_services_api/services/chatbot/utils/data_manager.py
"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import redis
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from ai_services_api.services.centralized_repository.database_setup import get_db_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APHRCDataManager:
    """Unified manager for APHRC data handling"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the unified data manager"""
        try:
            load_dotenv()
            self.setup_redis_connections(redis_host, redis_port, redis_db)
            self.setup_embedding_model()
            self.setup_web_scraper()
            logger.info("APHRC Data Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing APHRC Data Manager: {e}")
            raise

    def setup_redis_connections(self, host, port, db):
        """Setup Redis connections"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.redis_url = os.getenv("REDIS_URL", f"redis://{host}:{port}")
                self.redis_text = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=True,
                    db=db
                )
                self.redis_binary = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=False,
                    db=db
                )
                
                # Test connections
                self.redis_text.ping()
                self.redis_binary.ping()
                logger.info("Redis connections established")
                return
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    def setup_embedding_model(self):
        """Initialize the embedding model"""
        self.embedding_model = SentenceTransformer(
            os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )

    def setup_web_scraper(self):
        """Initialize web scraping components"""
        self.base_url = "https://aphrc.org"
        self.visited_urls = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def is_valid_url(self, url: str) -> bool:
        """Validate URLs for scraping"""
        parsed = urlparse(url)
        return (
            parsed.netloc == 'aphrc.org' and
            not any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.mp4', '.zip']) and
            '#' not in url and
            'mailto:' not in url
        )

    def get_links(self, soup: BeautifulSoup, current_url: str) -> set:
        """Extract valid links from the page"""
        links = set()
        for a in soup.find_all('a', href=True):
            url = urljoin(current_url, a['href'])
            if self.is_valid_url(url) and url not in self.visited_urls:
                links.add(url)
        return links

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        return ' '.join(text.split())

    async def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a single page"""
        try:
            time.sleep(1)  # Be respectful to the server
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('main') or soup.find('article') or soup
            
            content = {
                'url': url,
                'title': self.clean_text(soup.title.string) if soup.title else '',
                'meta_description': '',
                'headers': [],
                'paragraphs': [],
                'soup': soup  # Keep soup for link extraction
            }
            
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                content['meta_description'] = meta_desc.get('content', '')
            
            for h in main_content.find_all(['h1', 'h2', 'h3', 'h4']):
                header_text = self.clean_text(h.get_text())
                if header_text:
                    content['headers'].append(header_text)
            
            for p in main_content.find_all('p'):
                para_text = self.clean_text(p.get_text())
                if para_text:
                    content['paragraphs'].append(para_text)
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    async def store_nav_content(self, content: Dict[str, Any]) -> bool:
        """Store scraped navigation content in database and Redis"""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Store in database
            cur.execute("""
                INSERT INTO navigation_content
                (url, title, meta_description, content, headers, paragraphs)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                content['url'],
                content['title'],
                content['meta_description'],
                ' '.join(content['paragraphs']),
                content['headers'],
                content['paragraphs']
            ))
            
            content_id = cur.fetchone()[0]
            conn.commit()
            
            # Create embeddings and store in Redis
            text_content = self._create_nav_text(content)
            embedding = self.embedding_model.encode(text_content)
            
            base_key = f"nav:{content_id}"
            
            # Store in Redis
            pipeline = self.redis_text.pipeline()
            try:
                # Store text content
                pipeline.set(f"text:{base_key}", text_content)
                
                # Store embedding
                self.redis_binary.set(
                    f"emb:{base_key}",
                    embedding.astype(np.float32).tobytes()
                )
                
                # Store metadata
                metadata = {
                    'id': content_id,
                    'url': content['url'],
                    'title': content['title'],
                    'type': 'navigation',
                    'updated_at': datetime.now().isoformat()
                }
                pipeline.hset(f"meta:{base_key}", mapping=metadata)
                
                pipeline.execute()
                return True
                
            except Exception as e:
                pipeline.reset()
                raise e
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing navigation content: {e}")
            return False
            
        finally:
            cur.close()
            conn.close()

    async def fetch_publications(self) -> List[Dict[str, Any]]:
        """Fetch publications from database"""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT 
                    id, doi, title, abstract, summary, authors,
                    description, expert_id, type, collection,
                    date_issue, citation, language
                FROM resources_resource
                WHERE id IS NOT NULL
            """)
            
            publications = [{
                'id': row[0],
                'doi': row[1],
                'title': row[2],
                'abstract': row[3] or '',
                'summary': row[4] or '',
                'authors': row[5] if row[5] else [],
                'description': row[6] or '',
                'expert_id': row[7],
                'type': row[8] or '',
                'collection': row[9] or '',
                'date_issue': row[10] or '',
                'citation': row[11] or '',
                'language': row[12] or ''
            } for row in cur.fetchall()]
            
            return publications
            
        finally:
            cur.close()
            conn.close()

    async def store_publication(self, pub: Dict[str, Any]) -> bool:
        """Store publication content in Redis"""
        try:
            # Create embeddings
            text_content = self._create_pub_text(pub)
            embedding = self.embedding_model.encode(text_content)
            
            base_key = f"pub:{pub['id']}"
            
            # Store in Redis
            pipeline = self.redis_text.pipeline()
            try:
                # Store text content
                pipeline.set(f"text:{base_key}", text_content)
                
                # Store embedding
                self.redis_binary.set(
                    f"emb:{base_key}",
                    embedding.astype(np.float32).tobytes()
                )
                
                # Store metadata
                metadata = {
                    'id': pub['id'],
                    'doi': pub['doi'],
                    'title': pub['title'],
                    'authors': json.dumps(pub['authors']),
                    'type': 'publication',
                    'collection': pub['collection'],
                    'citation': pub['citation']
                }
                pipeline.hset(f"meta:{base_key}", mapping=metadata)
                
                pipeline.execute()
                return True
                
            except Exception as e:
                pipeline.reset()
                raise e
                
        except Exception as e:
            logger.error(f"Error storing publication content: {e}")
            return False

    def _create_nav_text(self, content: Dict[str, Any]) -> str:
        """Create text content for navigation data"""
        text_parts = [
            f"Title: {content['title']}",
            f"Description: {content['meta_description']}",
            "Headers: " + " | ".join(content['headers']),
            "Content: " + " ".join(content['paragraphs'])
        ]
        return '\n'.join(filter(None, text_parts))

    def _create_pub_text(self, pub: Dict[str, Any]) -> str:
        """Create text content for publication data"""
        text_parts = [
            f"Title: {pub['title']}",
            f"Authors: {', '.join(pub['authors'])}",
            f"Abstract: {pub['abstract']}",
            f"Summary: {pub['summary']}",
            f"Description: {pub['description']}",
            f"Collection: {pub['collection']}",
            f"Citation: {pub['citation']}"
        ]
        return '\n'.join(filter(None, text_parts))

    async def query_content(self, query: str, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query content based on type"""
        try:
            # Generate query embedding
            query_vector = self.embedding_model.encode(query)
            
            # Define search patterns based on type
            if content_type == 'navigation':
                # Get all embeddings from Redis that match the pattern
                nav_keys = self.redis_client.keys('emb:*')  # Gets all embeddings
                results = []
                
                for key in nav_keys:
                    stored_data = self.redis_client.get(key)
                    if stored_data:
                        data = json.loads(stored_data)
                        stored_vector = np.array(data['embedding'])
                        
                        # Calculate similarity
                        similarity = 1 - cosine(query_vector, stored_vector)
                        
                        results.append({
                            'url': data['url'],
                            'similarity': float(similarity),
                            'last_modified': data.get('last_modified'),
                            'stored_at': data.get('stored_at')
                        })
                
                # Sort by similarity and return top 5
                return sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
                
            elif content_type == 'publication':
                pattern_prefix = "pub:"
            else:
                # For general queries, search both
                results_nav = await self._search_content("nav:", query_vector)
                results_pub = await self._search_content("pub:", query_vector)
                return sorted(
                    results_nav + results_pub,
                    key=lambda x: x['similarity'],
                    reverse=True
                )[:5]
            
            return await self._search_content(pattern_prefix, query_vector)
            
        except Exception as e:
            logger.error(f"Error querying content: {e}")
            return []

    async def _search_content(
        self, 
        pattern_prefix: str, 
        query_vector: np.ndarray,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Search content in Redis"""
        results = []
        emb_pattern = f"emb:{pattern_prefix}*"
        
        for key in self.redis_binary.scan_iter(emb_pattern):
            key_str = key.decode('utf-8')
            base_key = key_str.replace('emb:', '')
            
            # Get embedding and calculate similarity
            embedding_bytes = self.redis_binary.get(key_str)
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                similarity = np.dot(query_vector, embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(embedding)
                )
                
                if similarity > 0.6:  # Threshold for relevance
                    # Get associated text and metadata
                    text = self.redis_text.get(f"text:{base_key}")
                    metadata = self.redis_text.hgetall(f"meta:{base_key}")
                    
                    results.append({
                        'similarity': similarity,
                        'text': text,
                        'metadata': metadata
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_n]

    def close(self):
        """Close connections"""
        try:
            if hasattr(self, 'redis_text'):
                self.redis_text.close()
            if hasattr(self, 'redis_binary'):
                self.redis_binary.close()
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
