import os
import logging
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
import psycopg2
import psycopg2.extras

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

class DatabaseExporter:
    def __init__(self):
        self.export_dir = 'exports'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def create_export_directory(self):
        """Create the export directory if it doesn't exist."""
        timestamped_dir = os.path.join(self.export_dir, self.timestamp)
        os.makedirs(timestamped_dir, exist_ok=True)
        return timestamped_dir

    def get_all_tables(self, conn):
        """Get all table names from the database."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        """
        with conn.cursor() as cur:
            cur.execute(query)
            return [table[0] for table in cur.fetchall()]

    def get_all_views(self, conn):
        """Get all view names from the database."""
        query = """
        SELECT table_name 
        FROM information_schema.views 
        WHERE table_schema = 'public'
        """
        with conn.cursor() as cur:
            cur.execute(query)
            return [view[0] for view in cur.fetchall()]

    def export_table(self, conn, table_name: str, export_dir: str):
        """Export a single table to CSV."""
        try:
            logger.info(f"Exporting table: {table_name}")
            
            # Use RealDictCursor to get column names
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f'SELECT * FROM {table_name}')
                rows = cur.fetchall()
                
                if not rows:
                    logger.warning(f"No data found in table: {table_name}")
                    # Create empty DataFrame with columns if table exists but is empty
                    df = pd.DataFrame(columns=[desc[0] for desc in cur.description])
                else:
                    df = pd.DataFrame(rows)

                # Handle special data types
                for column in df.columns:
                    # Convert JSON/JSONB columns to strings
                    if df[column].dtype == 'object':
                        df[column] = df[column].apply(lambda x: str(x) if x is not None else None)

                filename = os.path.join(export_dir, f"{table_name}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Successfully exported {len(df)} rows to {filename}")
                
                return len(df)

        except Exception as e:
            logger.error(f"Error exporting table {table_name}: {e}")
            raise

    def export_all(self):
        """Export all tables and views to CSV files."""
        export_dir = self.create_export_directory()
        logger.info(f"Exporting to directory: {export_dir}")
        
        total_tables = 0
        total_rows = 0
        
        try:
            with get_db_connection() as conn:
                # Export tables
                tables = self.get_all_tables(conn)
                logger.info(f"Found {len(tables)} tables to export")
                
                for table in tables:
                    rows = self.export_table(conn, table, export_dir)
                    total_tables += 1
                    total_rows += rows
                
                # Export views
                views = self.get_all_views(conn)
                logger.info(f"Found {len(views)} views to export")
                
                for view in views:
                    rows = self.export_table(conn, view, export_dir)
                    total_tables += 1
                    total_rows += rows

            logger.info(f"""
            Export Summary:
            - Total tables/views exported: {total_tables}
            - Total rows exported: {total_rows}
            - Export directory: {export_dir}
            """)
            
            return export_dir

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

def main():
    try:
        exporter = DatabaseExporter()
        export_dir = exporter.export_all()
        logger.info(f"Database export completed successfully. Files saved in: {export_dir}")
    except Exception as e:
        logger.error(f"Database export failed: {e}")
        raise

if __name__ == "__main__":
    main()