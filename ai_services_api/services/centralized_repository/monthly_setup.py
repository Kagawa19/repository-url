import os
import logging
import argparse
import asyncio
import sys
from dotenv import load_dotenv

# Core processors and services
from ai_services_api.services.centralized_repository.openalex.openalex_processor import OpenAlexProcessor
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.recommendation.graph_initializer import GraphDatabaseInitializer
from ai_services_api.services.centralized_repository.database_setup import DatabaseInitializer, ExpertManager
from ai_services_api.services.centralized_repository.openalex.expert_processor import ExpertProcessor

# Search and indexing
from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager

# Data source processors
from ai_services_api.services.centralized_repository.orcid.orcid_processor import OrcidProcessor
from ai_services_api.services.centralized_repository.knowhub.knowhub_scraper import KnowhubScraper
from ai_services_api.services.centralized_repository.website.website_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.nexus.researchnexus_scraper import ResearchNexusScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Ensure all required environment variables are set."""
    load_dotenv()
    
    required_vars = [
        'DATABASE_URL',
        'NEO4J_URI',
        'NEO4J_USER',
        'NEO4J_PASSWORD',
        'OPENALEX_API_URL',
        'GEMINI_API_KEY',
        'REDIS_URL',
        'ORCID_CLIENT_ID',
        'ORCID_CLIENT_SECRET',
        'KNOWHUB_BASE_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

async def process_publications(processor, summarizer):
    """Process publications from all sources."""
    try:
        logger.info("Processing publications data...")
        pub_processor = PublicationProcessor(processor.db, summarizer)
        
        # Process OpenAlex publications
        try:
            logger.info("Processing OpenAlex publications...")
            await processor.process_publications(pub_processor, source='openalex')
        except Exception as e:
            logger.error(f"Error processing OpenAlex publications: {e}")

        # Process ORCID publications
        try:
            logger.info("Processing ORCID publications...")
            orcid_processor = OrcidProcessor()
            await orcid_processor.process_publications(pub_processor, source='orcid')
            orcid_processor.close()
        except Exception as e:
            logger.error(f"Error processing ORCID publications: {e}")

        # Process KnowHub content
        try:
            knowhub_scraper = KnowhubScraper(summarizer=summarizer)
            all_content = knowhub_scraper.fetch_all_content(limit=2)
            
            for content_type, items in all_content.items():
                if items:
                    for item in items:
                        try:
                            pub_processor.process_single_work(item, source='knowhub')
                        except Exception as e:
                            logger.error(f"Error processing {content_type} item: {e}")
                            continue
            knowhub_scraper.close()
            
        except Exception as e:
            logger.error(f"Error processing KnowHub content: {e}")
        finally:
            if 'knowhub_scraper' in locals():
                knowhub_scraper.close()

        # Process Website publications
        try:
            website_scraper = WebsiteScraper(summarizer=summarizer)
            website_publications = website_scraper.fetch_content(limit=2)
            
            if website_publications:
                for pub in website_publications:
                    try:
                        pub_processor.process_single_work(pub, source='website')
                    except Exception as e:
                        logger.error(f"Error processing website publication: {e}")
                        continue
            website_scraper.close()
            
        except Exception as e:
            logger.error(f"Error processing Website publications: {e}")
        finally:
            if 'website_scraper' in locals():
                website_scraper.close()

        # Process ResearchNexus publications
        try:
            research_nexus_scraper = ResearchNexusScraper(summarizer=summarizer)
            research_nexus_publications = research_nexus_scraper.fetch_content(limit=2)

            if research_nexus_publications:
                for pub in research_nexus_publications:
                    pub_processor.process_single_work(pub, source='researchnexus')
            
        except Exception as e:
            logger.error(f"Error processing Research Nexus publications: {e}")
        finally:
            if 'research_nexus_scraper' in locals():
                research_nexus_scraper.close()

    except Exception as e:
        logger.error(f"Publication processing failed: {e}")
        raise

async def process_topics(processor, summarizer):
    """Process topics for all publications."""
    try:
        logger.info("Starting topic classification...")
        
        publications = processor.db.get_all_publications()
        
        if publications:
            topics = summarizer.generate_topics(publications)
            logger.info(f"Generated topics: {topics}")
            
            batch_size = 100
            total_processed = 0
            
            for i in range(0, len(publications), batch_size):
                batch = publications[i:i + batch_size]
                for pub in batch:
                    try:
                        assigned_topics = summarizer.assign_topics(pub, topics)
                        processor.db.update_publication_topics(pub['id'], assigned_topics)
                        total_processed += 1
                        
                        if total_processed % 10 == 0:
                            logger.info(f"Processed {total_processed}/{len(publications)} publications")
                            
                    except Exception as e:
                        logger.error(f"Error processing publication {pub.get('id')}: {e}")
                        continue
            
            logger.info(f"Completed topic classification for {total_processed} publications")
        else:
            logger.warning("No publications found for topic classification")
        
    except Exception as e:
        logger.error(f"Topic classification failed: {e}")
        raise

def create_search_index():
    """Create FAISS search index."""
    try:
        index_creator = ExpertSearchIndexManager()
        logger.info("Creating FAISS search index...")
        if index_creator.create_faiss_index():
            logger.info("FAISS index creation complete!")
            return True
        return False
    except Exception as e:
        logger.error(f"FAISS index creation failed: {e}")
        return False

def create_redis_index():
    """Create Redis search index."""
    try:
        redis_creator = ExpertRedisIndexManager()
        logger.info("Creating Redis search index...")
        
        if redis_creator.clear_redis_indexes():
            if redis_creator.create_redis_index():
                logger.info("Redis index creation complete!")
                return True
        return False
    except Exception as e:
        logger.error(f"Redis index creation failed: {e}")
        return False

async def run_monthly_setup(
    expertise_csv=None,
    skip_database=False,
    skip_openalex=False,
    skip_publications=False,
    skip_graph=False,
    skip_search=False,
    skip_redis=False,
    skip_topics=False
):
    """Monthly setup process with full initialization."""
    try:
        logger.info("Starting monthly setup process")
        
        # Setup environment
        setup_environment()
        
        # Initialize database if needed
        if not skip_database:
            logger.info("Initializing database...")
            db_initializer = DatabaseInitializer()
            db_initializer.initialize_schema()
            
            # Load experts if CSV provided
            if expertise_csv and os.path.exists(expertise_csv):
                logger.info(f"Loading experts from {expertise_csv}")
                expert_manager = ExpertManager()
                expert_manager.load_experts_from_csv(expertise_csv)
        
        # Initialize processors
        openalex_processor = OpenAlexProcessor()
        summarizer = TextSummarizer()
        
        try:
            # Process OpenAlex experts data
            if not skip_openalex:
                logger.info("Processing OpenAlex expert data...")
                expert_processor = ExpertProcessor(openalex_processor.db, os.getenv('OPENALEX_API_URL'))
                try:
                    await openalex_processor.update_experts_with_openalex()
                finally:
                    expert_processor.close()
            
            # Process publications
            if not skip_publications:
                await process_publications(openalex_processor, summarizer)
            
            # Process topics
            if not skip_topics:
                await process_topics(openalex_processor, summarizer)
            
            # Initialize graph database
            if not skip_graph:
                logger.info("Initializing graph database...")
                graph_initializer = GraphDatabaseInitializer()
                if not await graph_initializer.initialize_graph():
                    raise RuntimeError("Graph initialization failed")
            
            # Create search indices
            if not skip_search and not create_search_index():
                raise RuntimeError("FAISS index creation failed")
            
            if not skip_redis and not create_redis_index():
                raise RuntimeError("Redis index creation failed")
            
            logger.info("Monthly setup completed successfully!")
            return True
            
        finally:
            openalex_processor.close()
        
    except Exception as e:
        logger.error(f"Monthly setup failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Monthly Research Data Setup')
    parser.add_argument('--expertise-csv', default=None)
    parser.add_argument('--skip-database', action='store_true')
    parser.add_argument('--skip-openalex', action='store_true')
    parser.add_argument('--skip-publications', action='store_true')
    parser.add_argument('--skip-graph', action='store_true')
    parser.add_argument('--skip-search', action='store_true')
    parser.add_argument('--skip-redis', action='store_true')
    parser.add_argument('--skip-topics', action='store_true')
    
    args = parser.parse_args()
    
    asyncio.run(run_monthly_setup(
        expertise_csv=args.expertise_csv,
        skip_database=args.skip_database,
        skip_openalex=args.skip_openalex,
        skip_publications=args.skip_publications,
        skip_graph=args.skip_graph,
        skip_search=args.skip_search,
        skip_redis=args.skip_redis,
        skip_topics=args.skip_topics
    ))

if __name__ == "__main__":
    main()