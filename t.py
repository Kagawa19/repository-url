import os
import logging
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2
from psycopg2 import sql
import logging
import json
import secrets
import string
import pandas as pd
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_db_connection_params():
    """Get database connection parameters from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        parsed_url = urlparse(database_url)
        return {
            'host': parsed_url.hostname,
            'port': parsed_url.port,
            'dbname': parsed_url.path[1:],
            'user': parsed_url.username,
            'password': parsed_url.password
        }
    
    in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
    return {
        'host': '167.86.85.127' if in_docker else 'localhost',
        'port': '5432',
        'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
    }

@contextmanager
def get_db_connection(dbname=None):
    """
    Get database connection with proper error handling and connection cleanup.
    
    Args:
        dbname (str, optional): Override default database name if needed
        
    Yields:
        psycopg2.extensions.connection: Database connection object
    """
    params = get_db_connection_params()
    if dbname:
        params['dbname'] = dbname
    
    conn = None
    try:
        conn = psycopg2.connect(**params)
        logger.info(f"Connected to database: {params['dbname']} at {params['host']}")
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")

@contextmanager
def get_db_cursor(autocommit=False):
    """
    Get database cursor with transaction management.
    
    Args:
        autocommit (bool): Whether to enable autocommit mode
        
    Yields:
        tuple: (cursor, connection) tuple for database operations
    """
    with get_db_connection() as conn:
        conn.autocommit = autocommit
        cur = conn.cursor()
        try:
            yield cur, conn
        except Exception as e:
            if not autocommit:
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            cur.close()

class SchemaManager:
    """Database schema management."""
    
    def __init__(self):
        self.table_definitions = self._load_table_definitions()
        self.index_definitions = self._load_index_definitions()
        self.view_definitions = self._load_view_definitions()
        
    @staticmethod
    def _load_table_definitions() -> Dict[str, Dict[str, str]]:
        """Load SQL definitions for all tables in the system."""
        return {
            'chat_tables': {
                'chat_sessions': """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL UNIQUE,
                        user_id VARCHAR(255) NOT NULL,
                        start_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        end_timestamp TIMESTAMP WITH TIME ZONE,
                        total_messages INTEGER DEFAULT 0,
                        successful BOOLEAN DEFAULT TRUE,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'chat_interactions': """
                    CREATE TABLE IF NOT EXISTS chat_interactions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        response_time FLOAT,
                        intent_type VARCHAR(50),
                        intent_confidence FLOAT,
                        navigation_matches INTEGER DEFAULT 0,
                        publication_matches INTEGER DEFAULT 0,
                        error_occurred BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    )
                """,
                'sentiment_metrics': """
                    CREATE TABLE IF NOT EXISTS sentiment_metrics (
                        id SERIAL PRIMARY KEY,
                        interaction_id INTEGER NOT NULL,
                        sentiment_score FLOAT,
                        emotion_labels TEXT[],
                        satisfaction_score FLOAT,
                        urgency_score FLOAT,
                        clarity_score FLOAT,
                        FOREIGN KEY (interaction_id) REFERENCES chat_interactions(id)
                    )
                """,
                'chat_analytics': """
                    CREATE TABLE IF NOT EXISTS chat_analytics (
                        id SERIAL PRIMARY KEY,
                        interaction_id INTEGER NOT NULL,
                        content_id VARCHAR(255) NOT NULL,
                        content_type VARCHAR(50) NOT NULL,
                        similarity_score FLOAT,
                        rank_position INTEGER,
                        clicked BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (interaction_id) REFERENCES chat_interactions(id)
                    )
                """
            },
            'core_tables': {
                'content_webpage': """
                    CREATE TABLE IF NOT EXISTS content_webpage (
                        url TEXT PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        content_hash TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'content_pdf': """
                     CREATE TABLE IF NOT EXISTS content_pdf (
                        url TEXT,
                        chunk_index INTEGER,
                        chunk_content TEXT,
                        total_chunks INTEGER,
                        content_hash TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (url, chunk_index)
                    )
                """,
                'resources_resource': """
                    CREATE TABLE IF NOT EXISTS resources_resource (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        doi VARCHAR(255) UNIQUE,
                        authors JSON,
                        domains TEXT[],
                        type VARCHAR(50) DEFAULT 'publication',
                        publication_year INTEGER,
                        summary TEXT,
                        source VARCHAR(50) DEFAULT 'openalex',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'content_hashes': """
                      CREATE TABLE IF NOT EXISTS content_hashes (
                        url TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        last_checked TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,

                'content_embeddings': """
                      CREATE TABLE IF NOT EXISTS content_embeddings (
                        url TEXT PRIMARY KEY,
                        embedding_key TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'experts_expert': """
                    CREATE TABLE IF NOT EXISTS experts_expert (
                        id SERIAL PRIMARY KEY,
                        first_name VARCHAR(255) NOT NULL,
                        last_name VARCHAR(255) NOT NULL,
                        designation VARCHAR(255),
                        theme VARCHAR(255),
                        unit VARCHAR(255),
                        contact_details VARCHAR(255),
                        knowledge_expertise JSONB,
                        orcid VARCHAR(255),
                        domains TEXT[],
                        fields TEXT[],
                        subfields TEXT[],
                        password VARCHAR(255), 
                        is_superuser BOOLEAN DEFAULT FALSE,
                        is_staff BOOLEAN DEFAULT FALSE,
                        is_active BOOLEAN DEFAULT TRUE,
                        last_login TIMESTAMP WITH TIME ZONE,
                        date_joined TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        bio TEXT,
                        email VARCHAR(200),
                        middle_name VARCHAR(200)
                    )
                """,
                'expert_resource_links': """
                    CREATE TABLE IF NOT EXISTS expert_resource_links (
                        id SERIAL PRIMARY KEY,
                        expert_id INTEGER NOT NULL,
                        resource_id INTEGER NOT NULL,
                        author_position INTEGER,
                        confidence_score FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (expert_id) REFERENCES experts_expert(id) ON DELETE CASCADE,
                        FOREIGN KEY (resource_id) REFERENCES resources_resource(id) ON DELETE CASCADE,
                        UNIQUE(expert_id, resource_id)
                    )
                """,
                'chatbot_logs': """
                    CREATE TABLE IF NOT EXISTS chatbot_logs (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        response_time FLOAT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
            },
            'analytics_tables': {
                'search_sessions': """
                    -- Create sequence for session_id
                    CREATE SEQUENCE IF NOT EXISTS search_session_id_seq;
                    
                    -- Create search_sessions table
                    CREATE TABLE IF NOT EXISTS search_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER NOT NULL DEFAULT nextval('search_session_id_seq'),
                        user_id VARCHAR(255) NOT NULL,
                        start_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        end_timestamp TIMESTAMP WITH TIME ZONE,
                        is_active BOOLEAN DEFAULT TRUE,
                        query_count INTEGER DEFAULT 0,
                        successful_searches INTEGER DEFAULT 0
                    )
                """,
                'search_analytics': """
                    CREATE TABLE IF NOT EXISTS search_analytics (
                        id SERIAL PRIMARY KEY,
                        search_id INTEGER UNIQUE NOT NULL,
                        query TEXT NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        response_time FLOAT,
                        result_count INTEGER,
                        search_type VARCHAR(50),
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (search_id) REFERENCES search_sessions(id)

                    )
                """,
                'expert_search_matches': """
                    CREATE TABLE IF NOT EXISTS expert_search_matches (
                        id SERIAL PRIMARY KEY,
                        search_id INTEGER NOT NULL,
                        expert_id VARCHAR(255) NOT NULL,
                        rank_position INTEGER,
                        similarity_score FLOAT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (search_id) REFERENCES search_sessions(id)

                    )
                """,
                'user_recommendations': """
                    CREATE TABLE IF NOT EXISTS user_recommendations (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        expert_id INTEGER NOT NULL,
                        recommendation_score FLOAT NOT NULL,
                        match_details JSONB NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    

                )
                """
            },
            'interaction_tables': {
                'expert_interactions': """
                    CREATE TABLE IF NOT EXISTS expert_interactions (
                        id SERIAL PRIMARY KEY,
                        sender_id INTEGER NOT NULL,
                        receiver_id INTEGER NOT NULL,
                        interaction_type VARCHAR(100) NOT NULL,
                        success BOOLEAN DEFAULT TRUE,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (sender_id) REFERENCES experts_expert(id),
                        FOREIGN KEY (receiver_id) REFERENCES experts_expert(id)
                    )
                """,
                'expert_messages': """
                    CREATE TABLE IF NOT EXISTS expert_messages (
                        id SERIAL PRIMARY KEY,
                        sender_id INTEGER NOT NULL,
                        receiver_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        draft BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE,
                        FOREIGN KEY (sender_id) REFERENCES experts_expert(id),
                        FOREIGN KEY (receiver_id) REFERENCES experts_expert(id)
                    )
                """
            },
            'ml_tables': {
                'query_predictions': """
                    CREATE TABLE IF NOT EXISTS query_predictions (
                        id SERIAL PRIMARY KEY,
                        partial_query TEXT NOT NULL,
                        predicted_query TEXT NOT NULL,
                        confidence_score FLOAT,
                        user_id VARCHAR(255) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'expert_matching_logs': """
                    CREATE TABLE IF NOT EXISTS expert_matching_logs (
                        id SERIAL PRIMARY KEY,
                        expert_id VARCHAR(255) NOT NULL,
                        matched_expert_id VARCHAR(255) NOT NULL,
                        similarity_score FLOAT,
                        shared_domains TEXT[],
                        shared_fields INTEGER,
                        shared_skills INTEGER,
                        successful BOOLEAN DEFAULT TRUE,
                        user_id VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
            },
            'analytics_metadata_tables': {
                'domain_expertise_analytics': """
                    CREATE TABLE IF NOT EXISTS domain_expertise_analytics (
                        domain_name VARCHAR(255) PRIMARY KEY,
                        match_count INTEGER DEFAULT 0,
                        total_clicks INTEGER DEFAULT 0,
                        avg_similarity_score FLOAT DEFAULT 0.0,
                        last_matched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                'interactions': """
                    CREATE TABLE IF NOT EXISTS interactions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metrics JSONB,
                        response_time FLOAT,
                        sentiment_score FLOAT,
                        error_occurred BOOLEAN DEFAULT FALSE
                    )
                """
            }
        }

    @staticmethod
    def _load_index_definitions() -> List[str]:
        """Load SQL definitions for all indexes."""
        return [
            # Chat session indexes
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_timestamp ON chat_sessions(start_timestamp)",
            
            # Chat interaction indexes
            "CREATE INDEX IF NOT EXISTS idx_chat_interactions_session ON chat_interactions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_interactions_user ON chat_interactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_interactions_timestamp ON chat_interactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_chat_interactions_intent ON chat_interactions(intent_type)",
            
            # Sentiment indexes
            "CREATE INDEX IF NOT EXISTS idx_sentiment_interaction ON sentiment_metrics(interaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_sentiment_score ON sentiment_metrics(sentiment_score)",
            
            # Chat analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_chat_analytics_interaction ON chat_analytics(interaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_analytics_content ON chat_analytics(content_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_analytics_type ON chat_analytics(content_type)",
            
            # Core table indexes
            "CREATE INDEX IF NOT EXISTS idx_experts_name ON experts_expert (first_name, last_name)",
            "CREATE INDEX IF NOT EXISTS idx_resources_source ON resources_resource(source)",
            "CREATE INDEX IF NOT EXISTS idx_expert_resource_expert ON expert_resource_links(expert_id)",
            "CREATE INDEX IF NOT EXISTS idx_expert_resource_resource ON expert_resource_links(resource_id)",
            
            # Search and analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_search_sessions_user ON search_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_analytics_user ON search_analytics(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_analytics_timestamp ON search_analytics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_expert_search_matches_search_id ON expert_search_matches(search_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_sessions_session_id ON search_sessions(session_id)",
            
            # Interaction indexes
            "CREATE INDEX IF NOT EXISTS idx_expert_interactions_sender ON expert_interactions(sender_id)",
            "CREATE INDEX IF NOT EXISTS idx_expert_interactions_receiver ON expert_interactions(receiver_id)",
            "CREATE INDEX IF NOT EXISTS idx_expert_interactions_type ON expert_interactions(interaction_type)",
            "CREATE INDEX IF NOT EXISTS idx_expert_messages_sender ON expert_messages(sender_id)",
            "CREATE INDEX IF NOT EXISTS idx_expert_messages_receiver ON expert_messages(receiver_id)",
            
            # ML and matching indexes
            "CREATE INDEX IF NOT EXISTS idx_expert_matching_logs_expert ON expert_matching_logs(expert_id)",
            "CREATE INDEX IF NOT EXISTS idx_query_predictions_user ON query_predictions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_query_predictions_partial ON query_predictions(partial_query)",
            
            # General interaction indexes
            "CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_session_user ON interactions(session_id, user_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_metrics ON interactions USING GIN (metrics)",
            
            # Additional search and expert indexes
            "CREATE INDEX IF NOT EXISTS idx_search_analytics_query ON search_analytics(query)",
            "CREATE INDEX IF NOT EXISTS idx_expert_search_matches_expert ON expert_search_matches(expert_id)",
            "CREATE INDEX IF NOT EXISTS idx_expert_matches_search ON expert_search_matches(search_id)",
            "CREATE INDEX IF NOT EXISTS idx_domain_analytics_count ON domain_expertise_analytics(match_count DESC)"
        ]
    @staticmethod
    def _load_view_definitions() -> Dict[str, str]:
        """Load SQL definitions for all views."""
        return {
            'intent_performance_metrics': """
                CREATE OR REPLACE VIEW intent_performance_metrics AS
                SELECT 
                    intent_type,
                    COUNT(*) as total_queries,
                    AVG(intent_confidence) as avg_confidence,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as error_rate
                FROM chat_interactions
                WHERE intent_type IS NOT NULL
                GROUP BY intent_type
            """,
            'sentiment_analysis_metrics': """
                CREATE OR REPLACE VIEW sentiment_analysis_metrics AS
                SELECT 
                    DATE(ci.timestamp) as date,
                    AVG(sm.sentiment_score) as avg_sentiment,
                    AVG(sm.satisfaction_score) as avg_satisfaction,
                    AVG(sm.urgency_score) as avg_urgency,
                    AVG(sm.clarity_score) as avg_clarity,
                    COUNT(*) as total_interactions
                FROM chat_interactions ci
                JOIN sentiment_metrics sm ON ci.id = sm.interaction_id
                GROUP BY DATE(ci.timestamp)
            """,
            'content_matching_metrics': """
                CREATE OR REPLACE VIEW content_matching_metrics AS
                SELECT 
                    content_type,
                    COUNT(*) as total_matches,
                    AVG(similarity_score) as avg_similarity,
                    SUM(CASE WHEN rank_position = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as top_rank_rate,
                    SUM(CASE WHEN clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as click_through_rate
                FROM chat_analytics
                GROUP BY content_type
            """,
            'daily_search_metrics': """
                CREATE OR REPLACE VIEW daily_search_metrics AS
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                    AVG(result_count) as avg_results,
                    COUNT(DISTINCT session_id) as total_sessions
                FROM search_analytics
                GROUP BY DATE(timestamp)
                ORDER BY date
            """,
            'expert_interaction_metrics': """
                CREATE OR REPLACE VIEW expert_interaction_metrics AS
                SELECT 
                    sender_id,
                    receiver_id,
                    interaction_type,
                    COUNT(*) as total_interactions,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                    MIN(created_at) as first_interaction,
                    MAX(created_at) as last_interaction
                FROM expert_interactions
                GROUP BY sender_id, receiver_id, interaction_type
            """,
            'domain_expertise_ranking': """
                CREATE OR REPLACE VIEW domain_expertise_ranking AS
                SELECT 
                    domain_name,
                    match_count,
                    last_matched_at,
                    RANK() OVER (ORDER BY match_count DESC) as domain_rank
                FROM domain_expertise_analytics
            """,
            'expert_matching_performance': """
                CREATE OR REPLACE VIEW expert_matching_performance AS
                SELECT 
                    expert_id,
                    COUNT(*) as total_matches,
                    AVG(similarity_score) as avg_similarity_score,
                    SUM(CASE WHEN successful THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
                FROM expert_matching_logs
                GROUP BY expert_id
            """,
            'expert_publications': """
                CREATE OR REPLACE VIEW expert_publications AS
                SELECT 
                    e.id as expert_id,
                    e.first_name,
                    e.last_name,
                    r.id as resource_id,
                    r.title,
                    r.publication_year,
                    erl.author_position,
                    erl.confidence_score
                FROM experts_expert e
                JOIN expert_resource_links erl ON e.id = erl.expert_id
                JOIN resources_resource r ON r.id = erl.resource_id
                ORDER BY e.id, r.publication_year DESC
            """,
            'expert_search_performance': """
                CREATE OR REPLACE VIEW expert_search_performance AS
                SELECT 
                    esm.expert_id,
                    COUNT(*) as total_matches,
                    AVG(esm.similarity_score) as avg_similarity,
                    AVG(esm.rank_position) as avg_rank,
                    SUM(CASE WHEN esm.clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as click_through_rate,
                    COUNT(DISTINCT sa.user_id) as unique_users
                FROM expert_search_matches esm
                JOIN search_analytics sa ON esm.search_id = sa.search_id
                GROUP BY esm.expert_id
            """,
            'daily_search_metrics': """
                CREATE OR REPLACE VIEW daily_search_metrics AS
                SELECT 
                    DATE(sa.timestamp) as date,
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT sa.user_id) as unique_users,
                    AVG(sa.response_time) as avg_response_time,
                    SUM(CASE WHEN sa.result_count > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                    AVG(sa.result_count) as avg_results,
                    COUNT(DISTINCT ss.session_id) as total_sessions,
                    -- Expert matching metrics
                    COUNT(DISTINCT esm.expert_id) as matched_experts,
                    AVG(esm.similarity_score) as avg_similarity,
                    AVG(esm.rank_position) as avg_rank_position
                FROM search_analytics sa
                LEFT JOIN search_sessions ss ON sa.search_id = ss.id
                LEFT JOIN expert_search_matches esm ON sa.search_id = esm.search_id
                WHERE sa.timestamp IS NOT NULL
                GROUP BY DATE(sa.timestamp)
                ORDER BY date DESC
            """,
            

            'domain_performance_metrics': """
                CREATE OR REPLACE VIEW domain_performance_metrics AS
                SELECT 
                    domain_name,
                    match_count,
                    total_clicks,
                    avg_similarity_score,
                    last_matched_at,
                    RANK() OVER (ORDER BY match_count DESC) as popularity_rank,
                    total_clicks::FLOAT / NULLIF(match_count, 0) as click_rate
                FROM domain_expertise_analytics
            """,
            'session_analytics': """
                CREATE OR REPLACE VIEW session_analytics AS
                SELECT 
                    DATE(start_timestamp) as date,
                    COUNT(*) as total_sessions,
                    AVG(query_count) as avg_queries_per_session,
                    AVG(successful_searches::FLOAT / NULLIF(query_count, 0)) as session_success_rate,
                    AVG(EXTRACT(EPOCH FROM (end_timestamp - start_timestamp))) as avg_session_duration
                FROM search_sessions
                WHERE end_timestamp IS NOT NULL
                GROUP BY DATE(start_timestamp)
                ORDER BY date
            """,
            'query_patterns': """
                CREATE OR REPLACE VIEW query_patterns AS
                SELECT 
                    query,
                    COUNT(*) as usage_count,
                    AVG(result_count) as avg_results,
                    AVG(response_time) as avg_response_time,
                    COUNT(DISTINCT user_id) as unique_users
                FROM search_analytics
                GROUP BY query
                HAVING COUNT(*) > 1
                ORDER BY usage_count DESC
            """
        }
class DatabaseInitializer:
    """Handle database initialization and setup."""
    
    def __init__(self):
        self.schema_manager = SchemaManager()
    
    def create_database(self):
        """Create the database if it doesn't exist."""
        params = get_db_connection_params()
        target_dbname = params['dbname']
        
        try:
            # Try connecting to target database first
            with get_db_connection():
                logger.info(f"Database {target_dbname} already exists")
                return
        except psycopg2.OperationalError:
            # Create database if connection failed
            with get_db_connection('postgres') as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_dbname,))
                    if not cur.fetchone():
                        logger.info(f"Creating database {target_dbname}...")
                        cur.execute(sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(target_dbname)))
                        logger.info(f"Database {target_dbname} created successfully")
    
    def initialize_schema(self):
        """Initialize the complete database schema."""
        with get_db_cursor(autocommit=True) as (cur, _):
            # Create extensions
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            
            # Create tables in a specific order
            table_groups = [
                self.schema_manager.table_definitions['chat_tables'],  # Add this back
                self.schema_manager.table_definitions['core_tables'],
                self.schema_manager.table_definitions['analytics_tables'],
                self.schema_manager.table_definitions['interaction_tables'],
                self.schema_manager.table_definitions['ml_tables'],
                self.schema_manager.table_definitions['analytics_metadata_tables']
            ]
            
            for table_group in table_groups:
                for table_name, table_sql in table_group.items():
                    try:
                        cur.execute(table_sql)
                        logger.info(f"Created table: {table_name}")
                    except Exception as e:
                        logger.error(f"Error creating {table_name}: {e}")
                        raise
            
            # Create indexes
            for index_sql in self.schema_manager.index_definitions:
                try:
                    cur.execute(index_sql)
                    logger.info("Created index successfully")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            # Create views
            for view_name, view_sql in self.schema_manager.view_definitions.items():
                try:
                    cur.execute(view_sql)
                    logger.info(f"Created view: {view_name}")
                except Exception as e:
                    logger.warning(f"View creation warning: {e}")
            
            logger.info("Schema initialization completed successfully")

class ExpertManager:
    """Handle expert-related database operations."""
    def load_experts_from_csv(self, csv_path: str):
        """
        Load expert data from CSV file with comprehensive error handling.
        
        Args:
            csv_path (str): Path to the CSV file containing expert data
        
        Returns:
            int: Number of successfully processed experts
        
        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If there are critical issues with the CSV
        """
        # Validate file existence
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Expert CSV file not found: {csv_path}")
        
        # Read CSV with robust parsing
        try:
            df = pd.read_csv(csv_path, 
                            dtype=str,  # Read all columns as strings
                            keep_default_na=False)  # Prevent NaN conversion
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning(f"CSV file is empty: {csv_path}")
            return 0
        
        # Validate required columns
        required_columns = ['First_name', 'Last_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
        
        # Prepare tracking variables
        total_experts = len(df)
        processed_experts = 0
        failed_experts = 0
        
        with get_db_cursor() as (cur, conn):
            try:
                # Process experts in a single transaction
                for idx, row in df.iterrows():
                    try:
                        # Process individual expert row
                        expert_id = self._process_expert_row(cur, row)
                        if expert_id:
                            processed_experts += 1
                    except Exception as e:
                        failed_experts += 1
                        logger.warning(f"Failed to process expert at row {idx}: {e}")
                        # Continue processing other experts even if one fails
                
                # Commit transaction if any experts were processed
                if processed_experts > 0:
                    conn.commit()
                
                # Log comprehensive summary
                logger.info(
                    f"Expert CSV Processing Summary:\n"
                    f"  Total experts in CSV:   {total_experts}\n"
                    f"  Successfully processed: {processed_experts}\n"
                    f"  Failed to process:      {failed_experts}"
                )
                
                return processed_experts
            
            except Exception as e:
                # Rollback in case of any unexpected errors
                conn.rollback()
                logger.error(f"Transaction failed during expert loading: {e}")
                raise

    def _process_expert_row(self, cur, row):
        """Process a single expert row from CSV."""
        first_name = row.get('First_name', '').strip()
        last_name = row.get('Last_name', '').strip()
        
        if not first_name or not last_name:
            logger.warning(f"Skipping expert row due to missing first or last name")
            return None
        
        designation = row.get('Designation', '').strip()
        theme = row.get('Theme', '').strip()
        unit = row.get('Unit', '').strip()
        contact_details = row.get('Contact Details', '').strip()
        
        expertise_str = row.get('Knowledge and Expertise', '')
        expertise_list = [exp.strip() for exp in expertise_str.split(',') if exp.strip()]
        
        try:
            email = contact_details if contact_details and '@' in contact_details else None
            
            if email:
                cur.execute(
                    "SELECT id FROM experts_expert WHERE email = %s OR contact_details = %s",
                    (email, contact_details)
                )
                existing_expert = cur.fetchone()
                
                if existing_expert:
                    cur.execute("""
                        UPDATE experts_expert SET
                            first_name = %s,
                            last_name = %s,
                            designation = %s,
                            theme = %s,
                            unit = %s,
                            contact_details = %s,
                            knowledge_expertise = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id
                    """, (
                        first_name, last_name, designation, theme, unit,
                        contact_details,
                        json.dumps(expertise_list) if expertise_list else None,
                        existing_expert[0]
                    ))
                else:
                    cur.execute("""
                        INSERT INTO experts_expert (
                            first_name, last_name, designation, theme, unit,
                            contact_details, knowledge_expertise, email,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 
                                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING id
                    """, (
                        first_name, last_name, designation, theme, unit,
                        contact_details,
                        json.dumps(expertise_list) if expertise_list else None,
                        email
                    ))
            else:
                cur.execute("""
                    INSERT INTO experts_expert (
                        first_name, last_name, designation, theme, unit,
                        contact_details, knowledge_expertise,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (
                    first_name, last_name, designation, theme, unit,
                    contact_details,
                    json.dumps(expertise_list) if expertise_list else None
                ))
            
            expert_id = cur.fetchone()[0]
            logger.info(f"Processed expert: {first_name} {last_name} (ID: {expert_id})")
            return expert_id
            
        except Exception as e:
            logger.error(f"Error processing expert {first_name} {last_name}: {e}")
            raise

class ExpertResourceLinker:
    """Handle linking between experts and resources based on author names."""
    
    def __init__(self):
        self.name_cache = {}
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        return ' '.join(name.lower().split())
    
    def _get_name_parts(self, author_name: str) -> tuple:
        """Extract first and last name from author string."""
        parts = author_name.strip().split()
        if len(parts) == 1:
            return ('', parts[0])
        return (' '.join(parts[:-1]), parts[-1])
    
    def link_experts_to_resources(self):
        """Link experts to resources based on author names."""
        with get_db_cursor() as (cur, conn):
            try:
                # Get all experts
                cur.execute("""
                    SELECT id, first_name, last_name, middle_name 
                    FROM experts_expert
                """)
                experts = cur.fetchall()
                
                # Build expert name lookup
                expert_lookup = {}
                for expert_id, first_name, last_name, middle_name in experts:
                    full_name = self._normalize_name(f"{first_name} {middle_name or ''} {last_name}")
                    expert_lookup[full_name] = expert_id
                    # Also store without middle name
                    simple_name = self._normalize_name(f"{first_name} {last_name}")
                    expert_lookup[simple_name] = expert_id
                
                # Get all resources with authors
                cur.execute("SELECT id, authors FROM resources_resource WHERE authors IS NOT NULL")
                resources = cur.fetchall()
                
                # Process each resource
                for resource_id, authors in resources:
                    if not authors:
                        continue
                        
                    author_list = json.loads(authors) if isinstance(authors, str) else authors
                    
                    for position, author in enumerate(author_list, 1):
                        # Try different name combinations
                        author_name = self._normalize_name(author)
                        expert_id = expert_lookup.get(author_name)
                        
                        if expert_id:
                            # Calculate confidence score based on name match quality
                            confidence_score = 1.0 if ' ' in author_name else 0.8
                            
                            # Insert link with upsert
                            cur.execute("""
                                INSERT INTO expert_resource_links 
                                    (expert_id, resource_id, author_position, confidence_score)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (expert_id, resource_id) 
                                DO UPDATE SET 
                                    author_position = EXCLUDED.author_position,
                                    confidence_score = EXCLUDED.confidence_score
                            """, (expert_id, resource_id, position, confidence_score))
                
                conn.commit()
                logger.info("Successfully linked experts to resources")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error linking experts to resources: {e}")
                raise

def main():
    """Main entry point for database initialization."""
    try:
        # Initialize database
        initializer = DatabaseInitializer()
        initializer.create_database()
        initializer.initialize_schema()
        
        # Load initial expert data if CSV exists
        expert_manager = ExpertManager()
        expertise_csv = os.getenv('EXPERTISE_CSV')
        if expertise_csv and os.path.exists(expertise_csv):
            expert_manager.load_experts_from_csv(expertise_csv)
        
        # Initialize expert-resource links
        linker = ExpertResourceLinker()
        linker.link_experts_to_resources()
            
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    main()