import os
import logging
from contextlib import contextmanager
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_db_connection_params():
    """Get database connection parameters from environment variables."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
    }

@contextmanager
def get_db_connection():
    """Database connection context manager."""
    params = get_db_connection_params()
    conn = None
    try:
        conn = psycopg2.connect(**params)
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()

def create_response_quality_metrics_table():
    """Create the response_quality_metrics table in the database."""
    
    logger.info("Starting creation of response_quality_metrics table")
    
    # SQL to create the table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS response_quality_metrics (
        id SERIAL PRIMARY KEY,
        interaction_id INTEGER NOT NULL,
        helpfulness_score FLOAT,
        hallucination_risk FLOAT,
        factual_grounding_score FLOAT,
        unclear_elements TEXT[],
        potentially_fabricated_elements TEXT[],
        FOREIGN KEY (interaction_id) REFERENCES chatbot_logs(id)
    );
    """
    
    # SQL to create an index to improve query performance
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_response_quality_interaction_id 
    ON response_quality_metrics(interaction_id);
    """
    
    try:
        with get_db_connection() as conn:
            # Auto-commit is disabled by default
            conn.autocommit = True
            
            with conn.cursor() as cur:
                # First check if the table already exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'response_quality_metrics'
                    );
                """)
                
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    logger.info("Table response_quality_metrics already exists")
                else:
                    # Create the table
                    logger.info("Creating response_quality_metrics table")
                    cur.execute(create_table_sql)
                    logger.info("Table created successfully")
                
                    # Create the index
                    logger.info("Creating index on interaction_id")
                    cur.execute(create_index_sql)
                    logger.info("Index created successfully")
        
        logger.info("Response quality metrics table setup completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error creating response_quality_metrics table: {e}")
        return False

def main():
    try:
        success = create_response_quality_metrics_table()
        if success:
            logger.info("Table creation completed successfully")
        else:
            logger.error("Table creation failed")
    except Exception as e:
        logger.error(f"Script execution failed: {e}")

if __name__ == "__main__":
    main()