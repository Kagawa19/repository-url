import json
import logging
import asyncio
import os
import sys
import argparse
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Database related imports
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager

# Expert matching imports
from ai_services_api.services.centralized_repository.expert_matching.matcher import Matcher

# OpenAlex and publication imports
from ai_services_api.services.centralized_repository.openalex.openalex_processor import OpenAlexProcessor
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.openalex.expert_processor import ExpertProcessor

# Web content and AI services imports
from ai_services_api.services.centralized_repository.web_content.services.processor import WebContentProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer

# Search and index imports
from ai_services_api.services.recommendation.graph_initializer import GraphDatabaseInitializer
from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager

# Database setup imports
from ai_services_api.services.centralized_repository.database_setup import DatabaseInitializer, ExpertManager

# Scraper imports
from ai_services_api.services.centralized_repository.orcid.orcid_processor import OrcidProcessor
from ai_services_api.services.centralized_repository.knowhub.knowhub_scraper import KnowhubScraper
from ai_services_api.services.centralized_repository.website.website_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.nexus.researchnexus_scraper import ResearchNexusScraper

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed diagnostics
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/init.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SetupConfig:
    """Configuration class for system setup"""
    skip_database: bool = False
    skip_openalex: bool = False
    skip_publications: bool = False  # Set to False to ensure publications are saved
    skip_graph: bool = False
    skip_search: bool = False
    skip_redis: bool = False
    skip_scraping: bool = False
    skip_classification: bool = False
    expertise_csv: str = ''
    max_workers: int = 4
    batch_size: int = 200
    checkpoint_hours: int = 24

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SetupConfig':
        return cls(
            skip_database=args.skip_database,
            skip_openalex=args.skip_openalex,
            skip_publications=args.skip_publications,
            skip_graph=args.skip_graph,
            skip_search=args.skip_search,
            skip_redis=args.skip_redis,
            skip_scraping=args.skip_scraping,
            skip_classification=args.skip_classification,
            expertise_csv=args.expertise_csv,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            checkpoint_hours=args.checkpoint_hours
        )

    def to_dict(self) -> Dict:
        """Convert SetupConfig to a dictionary"""
        return {
            'skip_database': self.skip_database,
            'skip_openalex': self.skip_openalex,
            'skip_publications': self.skip_publications,
            'skip_graph': self.skip_graph,
            'skip_search': self.skip_search,
            'skip_redis': self.skip_redis,
            'skip_scraping': self.skip_scraping,
            'skip_classification': self.skip_classification,
            'expertise_csv': self.expertise_csv,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'checkpoint_hours': self.checkpoint_hours
        }

class SystemInitializer:
    """Initializes the complete research system, including database, web content,
    expert matching, publication processing, and search indexing."""

    def __init__(self, config: Optional[SetupConfig] = None):
        """Initialize system components and set up environment."""
        load_dotenv()
        self.db = None
        self.web_processor = None

        # Convert SetupConfig to dict and merge with environment variables
        config_dict = config.to_dict() if config else {}
        self.config = {
            'skip_database': config_dict.get('skip_database', False),
            'skip_openalex': config_dict.get('skip_openalex', False),
            'skip_publications': config_dict.get('skip_publications', False),
            'skip_graph': config_dict.get('skip_graph', False),
            'skip_search': config_dict.get('skip_search', False),
            'skip_redis': config_dict.get('skip_redis', False),
            'skip_scraping': config_dict.get('skip_scraping', False),
            'skip_classification': config_dict.get('skip_classification', False),
            'expertise_csv': config_dict.get('expertise_csv', ''),
            'max_workers': int(os.getenv('MAX_WORKERS', config_dict.get('max_workers', 4))),
            'batch_size': int(os.getenv('BATCH_SIZE', config_dict.get('batch_size', 200))),
            'checkpoint_hours': int(os.getenv('CHECKPOINT_HOURS', config_dict.get('checkpoint_hours', 24)))
        }
        logger.info("SystemInitializer initialized with config: %s", self.config)

        self.required_env_vars = [
            'DATABASE_URL',
            'NEO4J_URI',
            'NEO4J_USER',
            'NEO4J_PASSWORD',
            'OPENALEX_API_URL',
            'GEMINI_API_KEY',
            'REDIS_URL',
            'ORCID_CLIENT_ID',
            'ORCID_CLIENT_SECRET',
            'KNOWHUB_BASE_URL',
            'EXPERTISE_CSV',
            'WEBSITE_URL',
            'MAX_WORKERS',
            'BATCH_SIZE',
            'CHECKPOINT_HOURS'
        ]

        try:
            self.db = DatabaseManager()
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {str(e)}", exc_info=True)
            raise

    def verify_environment(self) -> None:
        """Verify all required environment variables are set"""
        missing_vars = [var for var in self.required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Using default values where possible...")

        # Validate configuration values
        if self.config['max_workers'] < 1 or self.config['batch_size'] < 1 or self.config['checkpoint_hours'] < 1:
            raise ValueError("Invalid configuration: MAX_WORKERS, BATCH_SIZE, and CHECKPOINT_HOURS must be positive")

    async def initialize_web_content_processor(self):
        """Initialize the web content processor."""
        try:
            self.web_processor = WebContentProcessor(db=self.db, config=self.config)
            logger.info("Web content processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web content processor: {str(e)}", exc_info=True)
            raise

    async def process_web_content(self) -> Dict:
        """Process web content, relying on ContentPipeline's incremental saves."""
        try:
            logger.info("\n" + "="*50)
            logger.info("Processing web content...")
            logger.info("="*50)

            # Log initial memory usage
            process = psutil.Process()
            logger.debug(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            if not self.web_processor:
                await self.initialize_web_content_processor()

            start_time = time.time()
            results = await self.web_processor.process_content()
            processing_time = time.time() - start_time

            # Validate results structure
            if 'processing_details' not in results or 'webpage_results' not in results['processing_details']:
                logger.warning("No webpage results found in processing details")
                return results

            # Log results
            logger.info(f"""Web Content Processing Results:
                Pages Processed: {results.get('processed_pages', 0)}
                Pages Updated: {results.get('updated_pages', 0)}
                Publications Processed: {results.get('processed_publications', 0)}
                Experts Processed: {results.get('processed_experts', 0)}
                PDFs Processed: {results.get('processed_resources', 0)}
                Processing Time: {processing_time:.2f} seconds
                Average Time Per Page: {processing_time/max(results.get('processed_pages', 1), 1):.2f} seconds
            """)
            logger.info(f"WebsiteScraper metrics: {self.web_processor.content_pipeline.scraper.get_metrics()}")
            logger.debug(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            # Log database state
            try:
                pub_count = self.db.execute("SELECT COUNT(*) FROM publications")[0][0]
                exp_count = self.db.execute("SELECT COUNT(*) FROM experts")[0][0]
                web_count = self.db.execute("SELECT COUNT(*) FROM webpages")[0][0]
                logger.info(f"Database state: {pub_count} publications, {exp_count} experts, {web_count} webpages")
            except Exception as e:
                logger.error(f"Error querying database state: {str(e)}")

            return results
        except Exception as e:
            logger.error(f"Error processing web content: {str(e)}", exc_info=True)
            raise
        finally:
            gc.collect()

    async def match_experts_with_resources(self) -> None:
        """Match experts with publications based on author names."""
        try:
            logger.info("Starting expert-publication matching process...")
            print("🔍 Starting expert-publication matching process...")

            experts = self.db.execute("""
                SELECT id, name FROM experts
                WHERE is_active = TRUE
            """)

            publications = self.db.execute("""
                SELECT id, authors FROM publications
            """)

            if not experts or not publications:
                logger.warning("No experts or publications found for matching.")
                print("⚠️ No experts or publications found for matching.")
                return

            logger.info(f"Found {len(experts)} experts and {len(publications)} publications for matching.")
            print(f"📊 Found {len(experts)} experts and {len(publications)} publications for matching.")

            matcher = Matcher()

            matches = matcher.match_experts_to_resources(
                [(expert[0], {'name': expert[1]}) for expert in experts],
                [(pub[0], {'authors': pub[1] or []}) for pub in publications]
            )

            if matches:
                logger.info(f"Found {len(matches)} expert-publication matches.")
                print(f"✅ Found {len(matches)} expert-publication matches.")

                for expert_id, pub_id in matches:
                    try:
                        existing_match = self.db.execute("""
                            SELECT id FROM expert_resource_mappings
                            WHERE expert_id = %s AND resource_id = %s
                        """, (expert_id, pub_id))

                        if not existing_match:
                            self.db.execute("""
                                INSERT INTO expert_resource_mappings
                                (expert_id, resource_id, match_type, created_at)
                                VALUES (%s, %s, %s, NOW())
                            """, (expert_id, pub_id, 'author_match'))
                            logger.info(f"Stored match: Expert {expert_id} - Publication {pub_id}")
                    except Exception as e:
                        logger.error(f"Error storing match for Expert {expert_id} - Publication {pub_id}: {e}")
                        print(f"❌ Error storing match: {e}")
            else:
                logger.warning("No matches found between experts and publications.")
                print("⚠️ No matches found between experts and publications.")

            logger.info("Expert-publication matching process completed.")
            print("🎉 Expert-publication matching process completed.")

        except Exception as e:
            logger.error(f"Error in expert-publication matching process: {e}", exc_info=True)
            print(f"💥 Critical Error in Expert-Publication Matching: {e}")

    async def initialize_database(self) -> None:
        """Initialize database and create tables using DatabaseInitializer"""
        try:
            logger.info("Initializing database...")
            initializer = DatabaseInitializer()
            initializer.create_database()
            initializer.initialize_schema()
            # Ensure required tables exist
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS publications (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    source VARCHAR,
                    type VARCHAR,
                    authors TEXT[],
                    domains TEXT[],
                    publication_year INTEGER,
                    doi VARCHAR UNIQUE,
                    field TEXT,
                    subfield TEXT
                );
                CREATE TABLE IF NOT EXISTS experts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    affiliation TEXT,
                    contact_email VARCHAR,
                    url VARCHAR UNIQUE,
                    is_active BOOLEAN DEFAULT TRUE
                );
                CREATE TABLE IF NOT EXISTS webpages (
                    id SERIAL PRIMARY KEY,
                    url VARCHAR UNIQUE,
                    content TEXT,
                    navigation_text TEXT,
                    content_hash VARCHAR,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS expert_resource_mappings (
                    id SERIAL PRIMARY KEY,
                    expert_id INTEGER REFERENCES experts(id),
                    resource_id INTEGER REFERENCES publications(id),
                    match_type VARCHAR,
                    created_at TIMESTAMP
                );
            """)
            logger.info("Database initialization complete!")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise

    async def load_initial_experts(self) -> None:
        """Load initial experts from CSV if provided"""
        try:
            csv_path = self.config.get('expertise_csv', 'experts.csv')
            if os.path.exists(csv_path):
                logger.info(f"Loading experts from {csv_path}...")
                expert_manager = ExpertManager()
                expert_manager.load_experts_from_csv(csv_path)
                logger.info("Initial experts loaded successfully!")
            else:
                logger.warning(f"No experts CSV found at {csv_path}. Skipping expert loading.")
        except Exception as e:
            logger.error(f"Error loading initial experts: {e}", exc_info=True)
            raise

    async def initialize_graph(self) -> bool:
        """Initialize the graph with experts and their relationships"""
        try:
            graph_initializer = GraphDatabaseInitializer()
            await graph_initializer.initialize_graph()
            logger.info("Graph initialization complete!")
            return True
        except Exception as e:
            logger.error(f"Graph initialization failed: {e}", exc_info=True)
            return False

    async def classify_all_publications(self, summarizer: Optional[TextSummarizer] = None) -> None:
        """Classify all publications in the corpus."""
        try:
            if summarizer is None:
                summarizer = TextSummarizer()
                print("🔧 Created default TextSummarizer")

            if self.config.get('skip_classification', False):
                logger.info("Skipping corpus classification as requested")
                print("[SKIP] Corpus classification disabled in configuration")
                return

            logger.info("Analyzing existing publications for field classification...")
            print("🔍 Starting publication classification process...")

            try:
                existing_publications = self.db.execute("""
                    SELECT id, title, summary, domains, source 
                    FROM publications 
                    WHERE field IS NULL OR subfield IS NULL
                """)
            except Exception as e:
                logger.error(f"Error retrieving publications: {e}", exc_info=True)
                print(f"⚠️ Database error: {e}")
                existing_publications = []

            if not existing_publications:
                logger.info("No publications found requiring classification.")
                print("✅ All publications are already classified")
                return

            logger.info("Performing corpus content analysis...")
            print("📊 Analyzing corpus to discover field structure...")

            field_structure = None
            try:
                field_structure = summarizer.analyze_content_corpus(
                    [{'title': row[1], 'summary': row[2], 'domains': row[3] or []} for row in existing_publications]
                )
            except Exception as e:
                logger.error(f"Error in corpus analysis: {e}", exc_info=True)
                print(f"⚠️ Corpus analysis error: {e}")

            if not field_structure:
                logger.warning("Corpus analysis did not return valid field structure")
                print("⚠️ Could not determine field structure from corpus")
                field_structure = {
                    "Research": ["General Research", "Study", "Analysis"],
                    "Health": ["Public Health", "Healthcare", "Medical Research"],
                    "Policy": ["Government Policy", "Regulations", "Guidelines"],
                    "Education": ["Learning", "Training", "Academic"],
                    "Development": ["Economic Development", "Social Development", "Progress"]
                }
                logger.info("Using default field structure instead")
                print("🔄 Using default field structure")

            logger.info(f"Field structure: {json.dumps(field_structure, indent=2)}")
            print(f"🌐 Field Structure: {json.dumps(field_structure, indent=2)}")

            total_classified = 0
            total_publications = len(existing_publications)

            for idx, row in enumerate(existing_publications, 1):
                try:
                    publication_id, title, abstract, domains, source = row
                    print(f"🏷️ Classifying Publication {idx}/{total_publications}: {title}")

                    if domains is None:
                        domains = []

                    field, subfield = self._classify_publication(title, abstract or "", domains, field_structure)

                    try:
                        self.db.execute("""
                            UPDATE publications 
                            SET field = %s, subfield = %s
                            WHERE id = %s
                        """, (field, subfield, publication_id))
                        logger.info(f"Classified {source} publication - {title}: {field}/{subfield}")
                        print(f"✔️ Classified: {title} → {field}/{subfield}")
                        total_classified += 1
                    except Exception as e:
                        logger.error(f"Error updating publication {title}: {e}", exc_info=True)
                        print(f"❌ Database update error for {title}: {e}")
                        continue

                except Exception as e:
                    title = row[1] if row and len(row) > 1 else "unknown"
                    logger.error(f"Error classifying publication {title}: {e}", exc_info=True)
                    print(f"❌ Classification error for {title}: {e}")
                    continue

            logger.info(f"Classification complete! Classified {total_classified} publications.")
            print(f"🎉 Classification Complete! Classified {total_classified}/{total_publications} publications")

        except Exception as e:
            logger.error(f"Error in publication classification: {e}", exc_info=True)
            print(f"💥 Critical Error in Publication Classification: {e}")

    def _classify_publication(self, title: str, abstract: str, domains: List[str], field_structure: Dict) -> Tuple[str, str]:
        """Classify a single publication based on the generated field structure."""
        logger.info(f"Classification attempt for publication: {title}")
        print(f"🔬 Attempting to classify: {title}")

        if not field_structure:
            logger.warning("No field structure available. Using generic classification.")
            print("⚠️ No field structure found. Using generic classification.")
            return "Unclassified", "General"

        for field, subfields in field_structure.items():
            if any(keyword.lower() in (title + " " + abstract).lower() for keyword in subfields):
                classification_result = (field, subfields[0])
                logger.info(f"Matched classification: {classification_result}")
                print(f"✔️ Matched Classification: {field}/{subfields[0]}")
                return classification_result

        first_field = list(field_structure.keys())[0]
        default_classification = (first_field, field_structure[first_field][0])
        logger.info(f"No direct match. Using default classification: {default_classification}")
        print(f"❓ No direct match. Using default: {default_classification[0]}/{default_classification[1]}")
        return default_classification

    async def process_publications(self, summarizer: Optional[TextSummarizer] = None) -> None:
        """Process publications from all sources without classification."""
        openalex_processor = None
        expert_processor = None
        try:
            openalex_processor = OpenAlexProcessor()
            publication_processor = PublicationProcessor(openalex_processor.db, TextSummarizer())
            expert_processor = ExpertProcessor(openalex_processor.db, os.getenv('OPENALEX_API_URL'))

            if summarizer is None:
                summarizer = TextSummarizer()

            logger.info("Updating experts with OpenAlex data...")
            await openalex_processor.update_experts_with_openalex()
            logger.info("Expert data enrichment complete!")

            if not self.config.get('skip_publications', False):
                logger.info("Processing publications data from all sources...")

                if not self.config.get('skip_openalex', False):
                    try:
                        logger.info("Processing OpenAlex publications...")
                        await openalex_processor.process_publications(publication_processor, source='openalex')
                    except Exception as e:
                        logger.error(f"Error processing OpenAlex publications: {e}", exc_info=True)

                try:
                    logger.info("Processing ORCID publications...")
                    orcid_processor = OrcidProcessor()
                    await orcid_processor.process_publications(publication_processor, source='orcid')
                    orcid_processor.close()
                except Exception as e:
                    logger.error(f"Error processing ORCID publications: {e}", exc_info=True)

                try:
                    logger.info("\n" + "="*50)
                    logger.info("Processing KnowHub content...")
                    logger.info("="*50)

                    knowhub_scraper = KnowhubScraper(summarizer=TextSummarizer())
                    all_content = knowhub_scraper.fetch_all_content()

                    for content_type, items in all_content.items():
                        if items:
                            logger.info(f"\nProcessing {len(items)} items from {content_type}")
                            for item in items:
                                try:
                                    publication_processor.process_single_work(item, source='knowhub')
                                    logger.info(f"Successfully processed {content_type} item: {item.get('title', 'Unknown Title')}")
                                except Exception as e:
                                    logger.error(f"Error processing {content_type} item: {e}", exc_info=True)
                                    continue
                        else:
                            logger.warning(f"No items found for {content_type}")

                    knowhub_scraper.close()
                    logger.info("\nKnowHub content processing complete!")

                except Exception as e:
                    logger.error(f"Error processing KnowHub content: {e}", exc_info=True)
                finally:
                    if 'knowhub_scraper' in locals():
                        knowhub_scraper.close()

                try:
                    logger.info("Processing Research Nexus publications...")
                    research_nexus_scraper = ResearchNexusScraper(summarizer=TextSummarizer())
                    research_nexus_publications = research_nexus_scraper.fetch_content()

                    if research_nexus_publications:
                        for pub in research_nexus_publications:
                            try:
                                publication_processor.process_single_work(pub, source='researchnexus')
                                logger.info(f"Successfully processed research nexus publication: {pub.get('title', 'Unknown Title')}")
                            except Exception as e:
                                logger.error(f"Error processing research nexus publication: {e}", exc_info=True)
                                continue
                    else:
                        logger.warning("No Research Nexus publications found")

                except Exception as e:
                    logger.error(f"Error processing Research Nexus publications: {e}", exc_info=True)
                finally:
                    if 'research_nexus_scraper' in locals():
                        research_nexus_scraper.close()

                try:
                    logger.info("\n" + "="*50)
                    logger.info("Processing Website publications...")
                    logger.info("="*50)

                    website_scraper = WebsiteScraper(summarizer=TextSummarizer(), max_browsers=2)
                    website_publications = website_scraper.fetch_content(max_pages=500)

                    if website_publications:
                        logger.info(f"\nProcessing {len(website_publications)} website publications")
                        for pub in website_publications:
                            try:
                                publication_processor.process_single_work(pub, source='website')
                                logger.info(f"Successfully processed website publication: {pub.get('title', 'Unknown Title')}")
                            except Exception as e:
                                logger.error(f"Error processing website publication: {e}", exc_info=True)
                                continue
                    else:
                        logger.warning("No website publications found")

                    logger.info(f"WebsiteScraper metrics: {website_scraper.get_metrics()}")
                    website_scraper.close()
                    logger.info("\nWebsite publications processing complete!")

                except Exception as e:
                    logger.error(f"Error processing Website publications: {e}", exc_info=True)
                finally:
                    if 'website_scraper' in locals():
                        website_scraper.close()

                logger.info("Publication processing complete! All sources have been processed.")

        except Exception as e:
            logger.error(f"Data processing failed: {e}", exc_info=True)
            raise
        finally:
            if openalex_processor:
                openalex_processor.close()
            if expert_processor:
                expert_processor.close()

    async def create_search_index(self) -> bool:
        """Create the FAISS search index."""
        index_creator = ExpertSearchIndexManager()
        try:
            if not self.config.get('skip_search', False):
                logger.info("Creating FAISS search index...")
                if not index_creator.create_faiss_index():
                    raise Exception("FAISS index creation failed")
            return True
        except Exception as e:
            logger.error(f"FAISS search index creation failed: {e}", exc_info=True)
            return False

    async def create_redis_index(self) -> bool:
        """Create the Redis search index."""
        try:
            if not self.config.get('skip_redis', False):
                logger.info("Creating Redis search index...")
                redis_creator = ExpertRedisIndexManager()
                if not (redis_creator.clear_redis_indexes() and redis_creator.create_redis_index()):
                    raise Exception("Redis index creation failed")
            return True
        except Exception as e:
            logger.error(f"Redis search index creation failed: {e}", exc_info=True)
            return False

    def cleanup(self):
        """Clean up resources, including web processor and database connections."""
        try:
            if self.web_processor:
                self.web_processor.cleanup()
                logger.info("Web content processor cleaned up")
                self.web_processor = None
            if self.db:
                self.db.close()
                logger.info("Database connection closed")
                self.db = None
            logger.info("System initializer cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

    async def initialize(self):
        """Run the full initialization process."""
        try:
            logger.info("Starting system initialization...")

            # Step 1: Verify environment and settings
            self.verify_environment()
            logger.debug('Environment verified successfully.')

            # Step 2: Database initialization (if not skipped)
            if not self.config.get('skip_database', False):
                await self.initialize_database()
                await self.load_initial_experts()

                logger.info('Starting expert fields processing...')
                openalex_processor = OpenAlexProcessor()
                expert_processor = ExpertProcessor(openalex_processor.db, os.getenv('OPENALEX_API_URL'))
                try:
                    expert_processor.process_expert_fields()
                    logger.info('Expert fields processing complete!')
                except Exception as e:
                    logger.error(f'Error processing expert fields: {e}', exc_info=True)
                finally:
                    expert_processor.close()
                    openalex_processor.close()

            # Step 3: Initialize text summarizer for AI-powered tasks
            summarizer = TextSummarizer()
            logger.info('Text summarizer initialized successfully.')

            # Step 4: Process publications (if not skipped)
            if not self.config.get('skip_publications', False):
                await self.process_publications(summarizer)
                await self.classify_all_publications(summarizer)

            # Step 5: Initialize graph database for recommendations
            if not self.config.get('skip_graph', False):
                logger.info('Initializing graph...')
                await self.initialize_graph()
                logger.info('Graph initialized successfully.')

            # Step 6: Create search indexes
            await self.create_search_index()
            await self.create_redis_index()

            # Step 7: Process web content
            if not self.config.get('skip_scraping', False):
                await self.initialize_web_content_processor()
                results = await self.process_web_content()
                logger.info(f"Web content processing completed with {results.get('processed_pages', 0)} pages processed")

            # Step 8: Match experts with publications
            await self.match_experts_with_resources()

            logger.info("System initialization completed successfully!")
            return {"status": "success", "message": "System initialization completed successfully"}
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Initialize and populate the research database system.')
    parser.add_argument('--skip-database', action='store_true', help='Skip database initialization')
    parser.add_argument('--skip-openalex', action='store_true', help='Skip OpenAlex data enrichment')
    parser.add_argument('--skip-publications', action='store_true', help='Skip publication processing')
    parser.add_argument('--skip-graph', action='store_true', help='Skip graph database initialization')
    parser.add_argument('--skip-search', action='store_true', help='Skip search index creation')
    parser.add_argument('--skip-redis', action='store_true', help='Skip Redis index creation')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip web content scraping')
    parser.add_argument('--skip-classification', action='store_true', help='Skip the publication classification')
    parser.add_argument('--expertise-csv', type=str, default='', help='Path to the CSV file containing initial expert data')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--batch-size', type=int, default=200, help='Batch size for web content processing')
    parser.add_argument('--checkpoint-hours', type=int, default=24, help='Hours between processing checkpoints')
    return parser.parse_args()

async def main():
    """Main entry point for system initialization."""
    args = parse_arguments()
    config = SetupConfig.from_args(args)
    initializer = SystemInitializer(config)
    try:
        await initializer.initialize()
        logger.info("Main process completed successfully")
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}", exc_info=True)
        raise
    finally:
        initializer.cleanup()

def run():
    """Entry point function that can be called from command line or other modules."""
    try:
        if os.name == 'nt':  # For Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run()