import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging
import os
from datetime import datetime, timedelta
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database connection utilities - based on your schema
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
def get_db_connection():
    """Get database connection with proper error handling and connection cleanup."""
    params = get_db_connection_params()
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

def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table_name,))
        return cursor.fetchone()[0]
    finally:
        cursor.close()

def get_table_columns(conn, table_name: str) -> List[str]:
    """Get list of columns for a table."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = %s
        """, (table_name,))
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()

def get_search_metrics(conn, start_date, end_date):
    """Get comprehensive search metrics using our views, with dynamic table/column checks"""
    cursor = conn.cursor()
    
    # Initialize empty dataframes for results
    daily_metrics = pd.DataFrame()
    domain_metrics = pd.DataFrame()
    
    try:
        # Check which tables exist
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        search_sessions_exists = check_table_exists(conn, 'search_sessions')
        expert_search_matches_exists = check_table_exists(conn, 'expert_search_matches')
        domain_expertise_analytics_exists = check_table_exists(conn, 'domain_expertise_analytics')
        experts_expert_exists = check_table_exists(conn, 'experts_expert')
        
        if not search_analytics_exists:
            logger.warning("search_analytics table doesn't exist")
            return {'daily_metrics': daily_metrics, 'domain_metrics': domain_metrics}
        
        # Get columns for search_analytics
        search_analytics_columns = get_table_columns(conn, 'search_analytics')
        required_columns = ['search_id', 'user_id', 'timestamp', 'response_time', 'result_count']
        missing_columns = [col for col in required_columns if col not in search_analytics_columns]
        
        if missing_columns:
            logger.warning(f"search_analytics missing columns: {missing_columns}")
            return {'daily_metrics': daily_metrics, 'domain_metrics': domain_metrics}
        
        # Build a dynamic query based on available tables and columns
        daily_metrics_cte = """
            WITH DailyMetrics AS (
                SELECT 
                    DATE(sa.timestamp) as date,
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT sa.user_id) as unique_users,
                    AVG(sa.response_time) as avg_response_time,
                    SUM(CASE WHEN sa.result_count > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) as success_rate,
                    AVG(sa.result_count) as avg_results
        """
        
        if search_sessions_exists:
            daily_metrics_cte += """,
                    COUNT(DISTINCT ss.session_id) as total_sessions
            """
        else:
            daily_metrics_cte += """,
                    0 as total_sessions
            """
        
        daily_metrics_cte += """
                FROM search_analytics sa
        """
        
        if search_sessions_exists:
            daily_metrics_cte += """
                LEFT JOIN search_sessions ss ON sa.search_id = ss.id
            """
        
        daily_metrics_cte += """
                WHERE sa.timestamp BETWEEN %s AND %s
                GROUP BY DATE(sa.timestamp)
            )
        """
        
        # Session metrics CTE - only if table exists
        session_metrics_cte = ""
        if search_sessions_exists:
            session_columns = get_table_columns(conn, 'search_sessions')
            if all(col in session_columns for col in ['start_timestamp', 'query_count', 'successful_searches']):
                session_metrics_cte = """
                    , SessionMetrics AS (
                        SELECT 
                            DATE(start_timestamp) as date,
                            COUNT(*) as total_sessions,
                            AVG(query_count) as avg_queries_per_session,
                            AVG(CASE 
                                WHEN successful_searches > 0 AND query_count > 0 
                                THEN successful_searches::float / NULLIF(query_count, 0)
                                ELSE 0 
                            END) as session_success_rate
                """
                
                if 'end_timestamp' in session_columns:
                    session_metrics_cte += """,
                            AVG(EXTRACT(epoch FROM (end_timestamp - start_timestamp))) as avg_session_duration
                    """
                else:
                    session_metrics_cte += """,
                            0 as avg_session_duration
                    """
                
                session_metrics_cte += """
                        FROM search_sessions
                        WHERE start_timestamp BETWEEN %s AND %s
                """
                
                if 'end_timestamp' in session_columns:
                    session_metrics_cte += """
                            AND end_timestamp IS NOT NULL
                    """
                
                session_metrics_cte += """
                        GROUP BY DATE(start_timestamp)
                    )
                """
        
        # Expert metrics CTE - only if tables exist
        expert_metrics_cte = ""
        if expert_search_matches_exists:
            expert_columns = get_table_columns(conn, 'expert_search_matches')
            if all(col in expert_columns for col in ['search_id', 'expert_id', 'similarity_score', 'rank_position']):
                expert_metrics_cte = """
                    , ExpertMetrics AS (
                        SELECT 
                            DATE(sa.timestamp) as date,
                            COUNT(DISTINCT esm.expert_id) as matched_experts,
                            AVG(esm.similarity_score) as avg_similarity,
                            AVG(esm.rank_position) as avg_rank
                        FROM search_analytics sa
                        JOIN expert_search_matches esm ON sa.search_id = esm.search_id
                        WHERE sa.timestamp BETWEEN %s AND %s
                        GROUP BY DATE(sa.timestamp)
                    )
                """
        
        # Main query SELECT and JOIN
        select_query = """
            SELECT d.*
        """
        
        if session_metrics_cte:
            select_query += """,
                s.avg_queries_per_session,
                s.session_success_rate,
                s.avg_session_duration
            """
        else:
            select_query += """,
                0 as avg_queries_per_session,
                0 as session_success_rate,
                0 as avg_session_duration
            """
        
        if expert_metrics_cte:
            select_query += """,
                e.matched_experts,
                e.avg_similarity,
                e.avg_rank
            """
        else:
            select_query += """,
                0 as matched_experts,
                0 as avg_similarity,
                0 as avg_rank
            """
        
        select_query += """
            FROM DailyMetrics d
        """
        
        if session_metrics_cte:
            select_query += """
            LEFT JOIN SessionMetrics s ON d.date = s.date
            """
        
        if expert_metrics_cte:
            select_query += """
            LEFT JOIN ExpertMetrics e ON d.date = e.date
            """
        
        select_query += """
            ORDER BY d.date
        """
        
        # Combine CTEs and main query
        full_query = daily_metrics_cte + session_metrics_cte + expert_metrics_cte + select_query
        
        # Create parameter list based on query needs
        params = [start_date, end_date]  # For DailyMetrics
        if session_metrics_cte:
            params.extend([start_date, end_date])  # For SessionMetrics
        if expert_metrics_cte:
            params.extend([start_date, end_date])  # For ExpertMetrics
        
        # Execute query for daily metrics
        try:
            cursor.execute(full_query, tuple(params))
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            daily_metrics = pd.DataFrame(data, columns=columns)
        except Exception as e:
            logger.error(f"Error executing daily metrics query: {e}")
            daily_metrics = pd.DataFrame()
        
        # Domain performance metrics query - only if all required tables exist
        if domain_expertise_analytics_exists and experts_expert_exists and expert_search_matches_exists:
            try:
                # Check for required columns
                domain_columns = get_table_columns(conn, 'domain_expertise_analytics')
                experts_columns = get_table_columns(conn, 'experts_expert')
                
                if 'domain_name' in domain_columns and 'domains' in experts_columns:
                    cursor.execute("""
                        SELECT 
                            d.domain_name,
                            COUNT(*) as match_count,
                            AVG(esm.similarity_score) as avg_similarity,
                            AVG(esm.rank_position) as avg_rank
                        FROM domain_expertise_analytics d
                        JOIN experts_expert e ON d.domain_name = ANY(e.domains)
                        JOIN expert_search_matches esm ON e.id::text = esm.expert_id
                        JOIN search_analytics sa ON esm.search_id = sa.search_id
                        WHERE sa.timestamp BETWEEN %s AND %s
                        GROUP BY d.domain_name
                        ORDER BY match_count DESC
                        LIMIT 10
                    """, (start_date, end_date))
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    domain_metrics = pd.DataFrame(data, columns=columns)
            except Exception as e:
                logger.error(f"Error executing domain metrics query: {e}")
                domain_metrics = pd.DataFrame()
        
        return {
            'daily_metrics': daily_metrics,
            'domain_metrics': domain_metrics
        }
    except Exception as e:
        logger.error(f"Error in get_search_metrics: {e}")
        return {
            'daily_metrics': pd.DataFrame(),
            'domain_metrics': pd.DataFrame()
        }
    finally:
        cursor.close()

def ensure_required_tables(conn):
    """Ensure all required tables exist, creating them if needed."""
    cursor = conn.cursor()
    try:
        # Check for search_analytics table
        if not check_table_exists(conn, 'search_analytics'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_analytics (
                    id SERIAL PRIMARY KEY,
                    search_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    response_time FLOAT,
                    result_count INTEGER,
                    search_type VARCHAR(50),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Created search_analytics table")
        
        # Check for search_sessions table
        if not check_table_exists(conn, 'search_sessions'):
            cursor.execute("""
                CREATE SEQUENCE IF NOT EXISTS search_session_id_seq;
                
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
            """)
            conn.commit()
            logger.info("Created search_sessions table")
        
        # Check for expert_search_matches table
        if not check_table_exists(conn, 'expert_search_matches'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_search_matches (
                    id SERIAL PRIMARY KEY,
                    search_id INTEGER NOT NULL,
                    expert_id VARCHAR(255) NOT NULL,
                    rank_position INTEGER,
                    similarity_score FLOAT,
                    clicked BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Created expert_search_matches table")
        
        # Check for domain_expertise_analytics table
        if not check_table_exists(conn, 'domain_expertise_analytics'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_expertise_analytics (
                    domain_name VARCHAR(255) PRIMARY KEY,
                    match_count INTEGER DEFAULT 0,
                    total_clicks INTEGER DEFAULT 0,
                    avg_similarity_score FLOAT DEFAULT 0.0,
                    last_matched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Created domain_expertise_analytics table")
        
        # Create indexes if needed
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_analytics_timestamp ON search_analytics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_search_sessions_start ON search_sessions(start_timestamp);
            CREATE INDEX IF NOT EXISTS idx_expert_matches_search ON expert_search_matches(search_id);
        """)
        conn.commit()
        logger.info("Created necessary indexes")
            
    except Exception as e:
        logger.error(f"Error ensuring required tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def display_search_analytics(metrics: Dict[str, pd.DataFrame], filters: Dict = None):
    """Display search analytics using updated metrics"""
    st.subheader("Search Analytics Dashboard")

    daily_data = metrics['daily_metrics']
    if daily_data.empty:
        st.warning("No search data available for the selected period")
        return

    # Ensure all required columns exist with default values if missing
    required_columns = [
        'total_searches', 'unique_users', 'success_rate', 'avg_response_time',
        'avg_queries_per_session', 'session_success_rate', 'avg_session_duration',
        'matched_experts', 'avg_similarity', 'avg_rank'
    ]
    
    for col in required_columns:
        if col not in daily_data.columns:
            daily_data[col] = 0

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Searches", 
            f"{daily_data['total_searches'].sum():,}"
        )
    with col2:
        st.metric(
            "Unique Users", 
            f"{daily_data['unique_users'].sum():,}"
        )
    with col3:
        success_rate = daily_data['success_rate'].mean()
        if pd.isna(success_rate):
            success_rate = 0
        st.metric(
            "Avg Success Rate", 
            f"{success_rate:.1%}"
        )
    with col4:
        avg_response = daily_data['avg_response_time'].mean()
        if pd.isna(avg_response):
            avg_response = 0
        st.metric(
            "Avg Response", 
            f"{avg_response:.2f}s"
        )

    # Create dashboard layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Search Volume Trends",
            "Response & Success Metrics",
            "Session Analytics",
            "Expert Matching Performance"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    # Convert date to datetime if it's not already
    if 'date' in daily_data.columns and not pd.api.types.is_datetime64_any_dtype(daily_data['date']):
        daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Ensure date column exists
    if 'date' not in daily_data.columns:
        st.error("Date column missing from metrics data")
        return

    # 1. Search Volume Trends
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_searches'],
            name="Total Searches",
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['unique_users'],
            name="Unique Users",
            line=dict(color='green')
        ),
        row=1, col=1,
        secondary_y=True
    )

    # 2. Response Time and Success Rate
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_response_time'],
            name="Response Time",
            line=dict(color='orange')
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['success_rate'],
            name="Success Rate",
            line=dict(color='purple')
        ),
        row=1, col=2,
        secondary_y=True
    )

    # 3. Session Analytics (only if data is available)
    has_session_data = not (daily_data['avg_queries_per_session'] == 0).all()
    if has_session_data:
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['avg_queries_per_session'],
                name="Queries/Session",
                line=dict(color='red')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['session_success_rate'],
                name="Session Success",
                line=dict(color='cyan')
            ),
            row=2, col=1,
            secondary_y=True
        )
    else:
        # Add message to plot if no session data
        fig.add_annotation(
            x=0.25, y=0.25,
            xref="paper", yref="paper",
            text="No session data available",
            showarrow=False,
            font=dict(size=16),
            row=2, col=1
        )

    # 4. Expert Matching Performance (only if data is available)
    has_expert_data = not (daily_data['matched_experts'] == 0).all()
    if has_expert_data:
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['avg_similarity'],
                name="Match Similarity",
                line=dict(color='darkblue')
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['avg_rank'],
                name="Avg Rank",
                line=dict(color='darkred')
            ),
            row=2, col=2,
            secondary_y=True
        )
    else:
        # Add message to plot if no expert data
        fig.add_annotation(
            x=0.75, y=0.25,
            xref="paper", yref="paper",
            text="No expert matching data available",
            showarrow=False,
            font=dict(size=16),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes titles
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Users", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Response Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate", secondary_y=True, row=1, col=2)
    
    if has_session_data:
        fig.update_yaxes(title_text="Queries", row=2, col=1)
        fig.update_yaxes(title_text="Success Rate", secondary_y=True, row=2, col=1)
    
    if has_expert_data:
        fig.update_yaxes(title_text="Similarity", row=2, col=2)
        fig.update_yaxes(title_text="Rank", secondary_y=True, row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Domain Performance Table
    domain_data = metrics['domain_metrics']
    if not domain_data.empty:
        st.subheader("Domain Performance")
        try:
            # Format DataFrame for display
            domain_data_display = domain_data.copy()
            for col in domain_data_display.columns:
                if col == 'domain_name':
                    continue
                if domain_data_display[col].dtype == 'float':
                    domain_data_display[col] = domain_data_display[col].round(3)
            
            # Display with styling
            st.dataframe(domain_data_display, use_container_width=True)
            
            # Create bar chart for domain match counts
            st.subheader("Domain Match Distribution")
            domain_fig = px.bar(
                domain_data,
                x='domain_name',
                y='match_count',
                color='avg_similarity',
                labels={'match_count': 'Number of Matches', 'domain_name': 'Domain'},
                title='Domain Match Counts',
                color_continuous_scale='Viridis'
            )
            domain_fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(domain_fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error displaying domain metrics: {e}")
            st.error("Error displaying domain performance metrics")
    else:
        st.info("No domain performance data available for the selected period")

    # Add date/time of last update
    st.markdown(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

def main():
    st.set_page_config(page_title="Search Analytics Dashboard", page_icon="🔍", layout="wide")
    
    # Dashboard filters in sidebar
    st.sidebar.title("Dashboard Filters")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date.date(), end_date.date()),
        max_value=end_date.date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Format for database query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get metrics with selected filters
        with get_db_connection() as conn:
            # Ensure all required tables exist
            ensure_required_tables(conn)
            
            # Get metrics
            metrics = get_search_metrics(conn, start_date_str, end_date_str)
            
            # Display dashboard
            display_search_analytics(metrics, {
                'start_date': start_date,
                'end_date': end_date
            })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()