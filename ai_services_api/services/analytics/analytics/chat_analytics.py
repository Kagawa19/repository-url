import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import asyncio
from datetime import datetime, timedelta
import os
import logging
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2

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

def get_table_columns(conn, table_name: str) -> list:
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

async def get_daily_metrics(conn, start_date, end_date):
    """Get daily chat metrics aggregated from chatbot_logs and chat_sessions."""
    cursor = conn.cursor()
    try:
        # First check if both tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chatbot_logs'
            ) AS chatbot_logs_exists,
            EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_sessions'
            ) AS chat_sessions_exists
        """)
        
        table_check = cursor.fetchone()
        chatbot_logs_exists, chat_sessions_exists = table_check[0], table_check[1]
        
        if not chatbot_logs_exists and not chat_sessions_exists:
            logger.warning("Neither chatbot_logs nor chat_sessions tables exist")
            return pd.DataFrame()
        
        # Get columns for chat_sessions if it exists
        chat_sessions_columns = []
        if chat_sessions_exists:
            chat_sessions_columns = get_table_columns(conn, 'chat_sessions')
        
        # Get columns for chatbot_logs if it exists
        chatbot_logs_columns = []
        if chatbot_logs_exists:
            chatbot_logs_columns = get_table_columns(conn, 'chatbot_logs')
        
        # Adjust query based on available tables and columns
        if chatbot_logs_exists and chat_sessions_exists:
            # Check if total_messages column exists in chat_sessions
            has_total_messages = 'total_messages' in chat_sessions_columns
            has_successful = 'successful' in chat_sessions_columns
            
            # Dynamic session metrics part based on available columns
            session_metrics_parts = [
                "DATE(start_timestamp) as date",
                "COUNT(*) as total_sessions"
            ]
            
            if has_total_messages:
                session_metrics_parts.append("AVG(total_messages) as avg_messages_per_session")
            else:
                session_metrics_parts.append("0 as avg_messages_per_session")
            
            if has_successful:
                session_metrics_parts.append("SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate")
            else:
                session_metrics_parts.append("0 as session_success_rate")
            
            session_metrics_query = f"""
                daily_session_metrics AS (
                    SELECT 
                        {', '.join(session_metrics_parts)}
                    FROM chat_sessions
                    WHERE start_timestamp BETWEEN %s AND %s
                    GROUP BY DATE(start_timestamp)
                )
            """
            
            # Check if response_time exists in chatbot_logs
            has_response_time = 'response_time' in chatbot_logs_columns
            
            # Dynamic chat metrics part based on available columns
            chat_metrics_parts = [
                "DATE(timestamp) as date",
                "COUNT(*) as total_chats",
                "COUNT(DISTINCT user_id) as unique_users"
            ]
            
            if has_response_time:
                chat_metrics_parts.append("AVG(response_time) as avg_response_time")
            else:
                chat_metrics_parts.append("0 as avg_response_time")
            
            chat_metrics_query = f"""
                daily_chat_metrics AS (
                    SELECT 
                        {', '.join(chat_metrics_parts)}
                    FROM chatbot_logs 
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
                )
            """
            
            # The full query
            query = f"""
                WITH {chat_metrics_query},
                {session_metrics_query}
                SELECT 
                    COALESCE(dcm.date, dse.date) as date,
                    COALESCE(total_chats, 0) as total_chats,
                    COALESCE(unique_users, 0) as unique_users,
                    COALESCE(avg_response_time, 0) as avg_response_time,
                    COALESCE(total_sessions, 0) as total_sessions,
                    COALESCE(avg_messages_per_session, 0) as avg_messages_per_session,
                    COALESCE(session_success_rate, 0) as session_success_rate
                FROM daily_chat_metrics dcm
                FULL OUTER JOIN daily_session_metrics dse ON dcm.date = dse.date
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date, start_date, end_date))
        
        elif chatbot_logs_exists:
            # Only chatbot_logs exists - check for response_time
            has_response_time = 'response_time' in chatbot_logs_columns
            
            # Dynamic chat metrics based on available columns
            select_parts = [
                "DATE(timestamp) as date",
                "COUNT(*) as total_chats",
                "COUNT(DISTINCT user_id) as unique_users"
            ]
            
            if has_response_time:
                select_parts.append("AVG(response_time) as avg_response_time")
            else:
                select_parts.append("0 as avg_response_time")
            
            # Add session metrics placeholders
            select_parts.extend([
                "0 as total_sessions",
                "0 as avg_messages_per_session",
                "0 as session_success_rate"
            ])
            
            query = f"""
                SELECT 
                    {', '.join(select_parts)}
                FROM chatbot_logs 
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        
        elif chat_sessions_exists:
            # Only chat_sessions exists
            has_total_messages = 'total_messages' in chat_sessions_columns
            has_successful = 'successful' in chat_sessions_columns
            
            # Dynamic session metrics based on available columns
            select_parts = [
                "DATE(start_timestamp) as date",
                "0 as total_chats",
                "COUNT(DISTINCT user_id) as unique_users",
                "0 as avg_response_time",
                "COUNT(*) as total_sessions"
            ]
            
            if has_total_messages:
                select_parts.append("AVG(total_messages) as avg_messages_per_session")
            else:
                select_parts.append("0 as avg_messages_per_session")
            
            if has_successful:
                select_parts.append("SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate")
            else:
                select_parts.append("0 as session_success_rate")
            
            query = f"""
                SELECT 
                    {', '.join(select_parts)}
                FROM chat_sessions
                WHERE start_timestamp BETWEEN %s AND %s
                GROUP BY DATE(start_timestamp)
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        logger.error(f"Error getting daily metrics: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

async def get_quality_metrics_trends(conn, start_date, end_date):
    """Get quality metrics from response_quality_metrics and chatbot_logs."""
    cursor = conn.cursor()
    try:
        # First check if tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'response_quality_metrics'
            ) AS quality_metrics_exists,
            EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chatbot_logs'
            ) AS chatbot_logs_exists
        """)
        
        table_check = cursor.fetchone()
        quality_metrics_exists, chatbot_logs_exists = table_check[0], table_check[1]
        
        if not quality_metrics_exists or not chatbot_logs_exists:
            logger.warning("Required tables for quality metrics don't exist")
            return pd.DataFrame()
        
        # Check columns in response_quality_metrics
        response_quality_columns = get_table_columns(conn, 'response_quality_metrics')
        
        # Check for interaction_id column to ensure we can join properly
        if 'interaction_id' not in response_quality_columns:
            logger.warning("response_quality_metrics table missing interaction_id column")
            return pd.DataFrame()
        
        # Build a dynamic query based on available columns
        select_parts = ["DATE(cl.timestamp) as date"]
        column_mappings = {
            'helpfulness_score': 'avg_helpfulness',
            'hallucination_risk': 'avg_hallucination_risk',
            'factual_grounding_score': 'avg_factual_grounding'
        }
        
        for col, alias in column_mappings.items():
            if col in response_quality_columns:
                select_parts.append(f"AVG(rqm.{col}) as {alias}")
            else:
                select_parts.append(f"0 as {alias}")
        
        select_parts.append("COUNT(*) as total_quality_evaluations")
        
        if 'helpfulness_score' in response_quality_columns:
            select_parts.append("COUNT(CASE WHEN rqm.helpfulness_score >= 0.7 THEN 1 END)::float / NULLIF(COUNT(*), 0) as high_quality_ratio")
        else:
            select_parts.append("0 as high_quality_ratio")
        
        query = f"""
            SELECT {', '.join(select_parts)}
            FROM chatbot_logs cl
            JOIN response_quality_metrics rqm ON cl.id = rqm.interaction_id
            WHERE cl.timestamp BETWEEN %s AND %s
            GROUP BY DATE(cl.timestamp)
            ORDER BY date
        """
        
        cursor.execute(query, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        logger.error(f"Error getting quality metrics: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

async def get_search_metrics(conn, start_date, end_date):
    """Get search metrics from search_analytics and search_sessions."""
    cursor = conn.cursor()
    try:
        # Check if tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'search_analytics'
            ) AS search_analytics_exists,
            EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'search_sessions'
            ) AS search_sessions_exists,
            EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'daily_search_metrics'
            ) AS daily_search_metrics_exists
        """)
        
        table_check = cursor.fetchone()
        search_analytics_exists, search_sessions_exists, daily_search_metrics_exists = table_check
        
        if not search_analytics_exists and not search_sessions_exists and not daily_search_metrics_exists:
            logger.warning("No search tables exist")
            return pd.DataFrame()
        
        # Try the daily_search_metrics view first if it exists
        if daily_search_metrics_exists:
            cursor.execute("""
                SELECT * FROM daily_search_metrics
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, (start_date, end_date))
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        
        # If no view, check raw tables and get columns
        search_analytics_columns = []
        if search_analytics_exists:
            search_analytics_columns = get_table_columns(conn, 'search_analytics')
        
        search_sessions_columns = []
        if search_sessions_exists:
            search_sessions_columns = get_table_columns(conn, 'search_sessions')
        
        # If both tables exist, join them
        if search_analytics_exists and search_sessions_exists:
            # Check available columns for dynamic query construction
            has_response_time = 'response_time' in search_analytics_columns
            has_result_count = 'result_count' in search_analytics_columns
            has_search_type = 'search_type' in search_analytics_columns
            has_successful_searches = 'successful_searches' in search_sessions_columns
            has_query_count = 'query_count' in search_sessions_columns
            
            # Analytics metrics
            analytics_parts = [
                "DATE(sa.timestamp) as date",
                "COUNT(*) as total_searches",
                "COUNT(DISTINCT sa.user_id) as unique_users"
            ]
            
            if has_response_time:
                analytics_parts.append("AVG(sa.response_time) as avg_response_time")
            else:
                analytics_parts.append("0 as avg_response_time")
            
            if has_result_count:
                analytics_parts.append("AVG(sa.result_count) as avg_results")
                analytics_parts.append("SUM(CASE WHEN sa.result_count > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) * 100 as success_rate")
            else:
                analytics_parts.append("0 as avg_results")
                analytics_parts.append("0 as success_rate")
            
            # Session metrics if we join with sessions
            join_condition = ""
            session_metrics = []
            
            if 'search_id' in search_analytics_columns and 'id' in search_sessions_columns:
                join_condition = "LEFT JOIN search_sessions ss ON sa.search_id = ss.id"
                
                session_metrics.append("COUNT(DISTINCT ss.id) as total_sessions")
                
                if has_successful_searches and has_query_count:
                    session_metrics.append("SUM(ss.successful_searches)::float / NULLIF(SUM(ss.query_count), 0) * 100 as session_success_rate")
                else:
                    session_metrics.append("0 as session_success_rate")
            else:
                # If no proper join condition, add default values
                session_metrics.extend([
                    "0 as total_sessions",
                    "0 as session_success_rate"
                ])
                join_condition = ""
            
            # Add search type distribution if available
            search_type_metrics = []
            if has_search_type:
                search_type_metrics.append("""
                    SUM(CASE WHEN sa.search_type = 'expert' THEN 1 ELSE 0 END) as expert_searches,
                    SUM(CASE WHEN sa.search_type = 'publication' THEN 1 ELSE 0 END) as publication_searches,
                    SUM(CASE WHEN sa.search_type = 'general' THEN 1 ELSE 0 END) as general_searches
                """)
            else:
                search_type_metrics.extend([
                    "0 as expert_searches",
                    "0 as publication_searches",
                    "0 as general_searches"
                ])
            
            # Combine all metrics
            all_metrics = analytics_parts + session_metrics + search_type_metrics
            
            query = f"""
                SELECT 
                    {', '.join(all_metrics)}
                FROM search_analytics sa
                {join_condition}
                WHERE sa.timestamp BETWEEN %s AND %s
                GROUP BY DATE(sa.timestamp)
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        elif search_analytics_exists:
            # Only search_analytics exists
            has_response_time = 'response_time' in search_analytics_columns
            has_result_count = 'result_count' in search_analytics_columns
            has_search_type = 'search_type' in search_analytics_columns
            
            select_parts = [
                "DATE(timestamp) as date",
                "COUNT(*) as total_searches",
                "COUNT(DISTINCT user_id) as unique_users"
            ]
            
            if has_response_time:
                select_parts.append("AVG(response_time) as avg_response_time")
            else:
                select_parts.append("0 as avg_response_time")
            
            if has_result_count:
                select_parts.append("AVG(result_count) as avg_results")
                select_parts.append("SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) * 100 as success_rate")
            else:
                select_parts.append("0 as avg_results")
                select_parts.append("0 as success_rate")
            
            # Add placeholder session metrics
            select_parts.extend([
                "0 as total_sessions",
                "0 as session_success_rate"
            ])
            
            # Add search type distribution if available
            if has_search_type:
                select_parts.extend([
                    "SUM(CASE WHEN search_type = 'expert' THEN 1 ELSE 0 END) as expert_searches",
                    "SUM(CASE WHEN search_type = 'publication' THEN 1 ELSE 0 END) as publication_searches",
                    "SUM(CASE WHEN search_type = 'general' THEN 1 ELSE 0 END) as general_searches"
                ])
            else:
                select_parts.extend([
                    "0 as expert_searches",
                    "0 as publication_searches",
                    "0 as general_searches"
                ])
            
            query = f"""
                SELECT 
                    {', '.join(select_parts)}
                FROM search_analytics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        elif search_sessions_exists:
            # Only search_sessions exists
            has_successful_searches = 'successful_searches' in search_sessions_columns
            has_query_count = 'query_count' in search_sessions_columns
            
            select_parts = [
                "DATE(start_timestamp) as date",
                "SUM(query_count) as total_searches",
                "COUNT(DISTINCT user_id) as unique_users",
                "0 as avg_response_time",
                "0 as avg_results",
                "COUNT(*) as total_sessions"
            ]
            
            if has_successful_searches and has_query_count:
                select_parts.append("SUM(successful_searches)::float / NULLIF(SUM(query_count), 0) * 100 as session_success_rate")
                select_parts.append("SUM(successful_searches)::float / NULLIF(SUM(query_count), 0) * 100 as success_rate")
            else:
                select_parts.append("0 as session_success_rate")
                select_parts.append("0 as success_rate")
            
            # Add placeholder search type metrics
            select_parts.extend([
                "0 as expert_searches",
                "0 as publication_searches",
                "0 as general_searches"
            ])
            
            query = f"""
                SELECT 
                    {', '.join(select_parts)}
                FROM search_sessions
                WHERE start_timestamp BETWEEN %s AND %s
                GROUP BY DATE(start_timestamp)
                ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        logger.error(f"Error getting search metrics: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

def get_chat_and_search_metrics(conn, start_date, end_date):
    """
    Synchronous wrapper for getting chat and search metrics.
    
    Args:
        conn: Database connection
        start_date: Start date for metrics query
        end_date: End date for metrics query
        
    Returns:
        DataFrame containing chat and search metrics
    """
    async def get_metrics_async():
        # Gather metrics concurrently
        daily_metrics, quality_metrics, search_metrics = await asyncio.gather(
            get_daily_metrics(conn, start_date, end_date),
            get_quality_metrics_trends(conn, start_date, end_date),
            get_search_metrics(conn, start_date, end_date)
        )
        
        # Combine all metrics into one DataFrame if possible
        dfs = []
        if not daily_metrics.empty:
            dfs.append(daily_metrics)
        if not quality_metrics.empty:
            dfs.append(quality_metrics)
        if not search_metrics.empty:
            dfs.append(search_metrics)
        
        if not dfs:
            return pd.DataFrame()
        
        # Start with the first dataframe
        result_df = dfs[0]
        
        # Merge with other dataframes if available
        for df in dfs[1:]:
            result_df = pd.merge(
                result_df,
                df,
                on='date',
                how='outer'
            )
        
        return result_df.fillna(0)

    # Run async code in sync context
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metrics = loop.run_until_complete(get_metrics_async())
        loop.close()
        return metrics
    except Exception as e:
        st.error(f"Error getting metrics: {e}")
        logger.error(f"Error in get_chat_and_search_metrics: {e}")
        return pd.DataFrame()

def display_chat_analytics(metrics_df, filters):
    """Display comprehensive chat analytics dashboard with improved visualizations."""
    st.title("Chat & Search Analytics Dashboard")
    
    try:
        if metrics_df.empty:
            st.warning("No data available for the selected date range.")
            return
        
        # Ensure date column is datetime
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        
        # Check if required columns exist and create them with default values if missing
        required_columns = ['total_chats', 'unique_users', 'avg_response_time', 'total_sessions']
        for col in required_columns:
            if col not in metrics_df.columns:
                metrics_df[col] = 0
                st.warning(f"Data for '{col}' is not available.")
        
        
        # ---- KEY PERFORMANCE INDICATORS ----
        st.markdown("### Key Performance Indicators")
        
        # Use a clean card layout for KPIs
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        
        # Total Chats KPI
        with row1_col1:
            total_chats = int(metrics_df['total_chats'].sum())
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Total Interactions</h4>
                    <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_chats:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Chat Interactions</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Response Time KPI
        with row1_col2:
            avg_response = metrics_df['avg_response_time'].mean() if 'avg_response_time' in metrics_df.columns else 0
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Response Time</h4>
                    <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{avg_response:.2f}s</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Average Response Time</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Quality Score KPI
        with row1_col3:
            avg_helpfulness = metrics_df['avg_helpfulness'].mean() if 'avg_helpfulness' in metrics_df.columns else 0
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Quality Score</h4>
                    <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{avg_helpfulness:.2f}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Average Helpfulness</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        # Sessions KPI
        with row2_col1:
            total_sessions = int(metrics_df['total_sessions'].sum()) if 'total_sessions' in metrics_df.columns else 0
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Total Sessions</h4>
                    <h2 style="margin:0;padding:10px 0;color:#9467bd;font-size:28px;">{total_sessions:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Chat Sessions</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Users KPI
        with row2_col2:
            unique_users = int(metrics_df['unique_users'].sum()) if 'unique_users' in metrics_df.columns else 0
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Unique Users</h4>
                    <h2 style="margin:0;padding:10px 0;color:#d62728;font-size:28px;">{unique_users:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Total Unique Users</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Success Rate KPI
        with row2_col3:
            # Check which success rate column is available
            if 'session_success_rate' in metrics_df.columns:
                success_rate = metrics_df['session_success_rate'].mean() * 100
                success_type = "Session Success Rate"
            elif 'success_rate' in metrics_df.columns:
                success_rate = metrics_df['success_rate'].mean()
                success_type = "Search Success Rate"
            elif 'high_quality_ratio' in metrics_df.columns:
                success_rate = metrics_df['high_quality_ratio'].mean() * 100
                success_type = "High Quality Response Rate"
            else:
                success_rate = 0
                success_type = "Success Rate"
                
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">{success_type}</h4>
                    <h2 style="margin:0;padding:10px 0;color:#8c564b;font-size:28px;">{success_rate:.1f}%</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Successful Interactions</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # ---- TABBED ANALYSIS SECTIONS ----
        tab1, tab2, tab3 = st.tabs(["Chat Activity", "Quality Metrics", "Search Performance"])
        
        with tab1:
            st.markdown("### Chat Activity Trends")
            
            # Chat Volume Trends - improved visualization
            has_chats = not metrics_df['total_chats'].isna().all() and metrics_df['total_chats'].sum() > 0
            has_users = not metrics_df['unique_users'].isna().all() and metrics_df['unique_users'].sum() > 0
            has_sessions = 'total_sessions' in metrics_df.columns and metrics_df['total_sessions'].sum() > 0
            
            if has_chats or has_users or has_sessions:
                volume_fig = go.Figure()
                
                # Add traces based on available data
                if has_chats:
                    volume_fig.add_trace(go.Scatter(
                        x=metrics_df['date'], 
                        y=metrics_df['total_chats'], 
                        mode='lines+markers',
                        name='Total Chats',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                
                if has_users:
                    volume_fig.add_trace(go.Scatter(
                        x=metrics_df['date'], 
                        y=metrics_df['unique_users'], 
                        mode='lines+markers', 
                        name='Unique Users',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8)
                    ))
                
                if has_sessions:
                    volume_fig.add_trace(go.Scatter(
                        x=metrics_df['date'], 
                        y=metrics_df['total_sessions'], 
                        mode='lines+markers', 
                        name='Total Sessions',
                        line=dict(color='#2ca02c', width=3),
                        marker=dict(size=8)
                    ))
                
                # Update layout for better aesthetics
                volume_fig.update_layout(
                    height=500,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d",
                        tickmode="auto",
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Count",
                        gridcolor='rgba(0,0,0,0.1)'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
            else:
                st.info("No chat activity data available for the selected period")
            
            # Session Metrics (if session data is available)
            if has_sessions:
                st.markdown("### Session Performance")
                
                # Check if we have message metrics
                has_messages = 'avg_messages_per_session' in metrics_df.columns and not metrics_df['avg_messages_per_session'].isna().all()
                has_success_rate = 'session_success_rate' in metrics_df.columns and not metrics_df['session_success_rate'].isna().all()
                
                session_fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Session Trends", "Session Metrics"),
                    specs=[[{"type": "scatter"}, {"type": "bar"}]]
                )
                
                # Add time-series trace for sessions
                session_fig.add_trace(
                    go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df['total_sessions'],
                        mode='lines+markers',
                        name='Total Sessions',
                        line=dict(color='#9467bd', width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                # Add bar charts for session metrics
                if has_messages and has_success_rate:
                    # Calculate average metrics for the period
                    avg_msgs = metrics_df['avg_messages_per_session'].mean()
                    avg_success = metrics_df['session_success_rate'].mean() * 100
                    
                    session_fig.add_trace(
                        go.Bar(
                            x=['Avg Messages per Session', 'Session Success Rate (%)'],
                            y=[avg_msgs, avg_success],
                            marker_color=['#17becf', '#bcbd22']
                        ),
                        row=1, col=2
                    )
                    
                    # Add percentage sign to success rate bar
                    session_fig.update_traces(
                        text=[f"{avg_msgs:.1f}", f"{avg_success:.1f}%"],
                        textposition='auto',
                        row=1, col=2
                    )
                elif has_messages:
                    avg_msgs = metrics_df['avg_messages_per_session'].mean()
                    session_fig.add_trace(
                        go.Bar(
                            x=['Avg Messages per Session'],
                            y=[avg_msgs],
                            marker_color=['#17becf']
                        ),
                        row=1, col=2
                    )
                    session_fig.update_traces(
                        text=[f"{avg_msgs:.1f}"],
                        textposition='auto',
                        row=1, col=2
                    )
                elif has_success_rate:
                    avg_success = metrics_df['session_success_rate'].mean() * 100
                    session_fig.add_trace(
                        go.Bar(
                            x=['Session Success Rate (%)'],
                            y=[avg_success],
                            marker_color=['#bcbd22']
                        ),
                        row=1, col=2
                    )
                    session_fig.update_traces(
                        text=[f"{avg_success:.1f}%"],
                        textposition='auto',
                        row=1, col=2
                    )
                
                # Update layout
                session_fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                # Update axis titles
                session_fig.update_xaxes(
                    title="Date", 
                    tickangle=-45,
                    row=1, col=1
                )
                session_fig.update_yaxes(
                    title="Session Count",
                    gridcolor='rgba(0,0,0,0.1)',
                    row=1, col=1
                )
                session_fig.update_yaxes(
                    title="Value",
                    gridcolor='rgba(0,0,0,0.1)',
                    row=1, col=2
                )
                
                st.plotly_chart(session_fig, use_container_width=True)
            
            # Response Time Analysis (if response time data is available)
            has_response_time = 'avg_response_time' in metrics_df.columns and not metrics_df['avg_response_time'].isna().all()
            
            if has_response_time:
                st.markdown("### Response Time Analysis")
                
                # Create a more attractive response time visualization
                response_fig = go.Figure()
                
                # Add the main line chart
                response_fig.add_trace(go.Scatter(
                    x=metrics_df['date'], 
                    y=metrics_df['avg_response_time'], 
                    mode='lines+markers', 
                    name='Average Response Time',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ))
                
                # Add a filled area for better visualization
                response_fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_response_time'],
                    fill='tozeroy',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                # Calculate and add a trendline
                if len(metrics_df) > 1:
                    try:
                        x_numeric = pd.to_numeric(metrics_df.reset_index().index)
                        y_values = metrics_df['avg_response_time'].values
                        
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
                        
                        trend_y = intercept + slope * x_numeric
                        
                        response_fig.add_trace(go.Scatter(
                            x=metrics_df['date'],
                            y=trend_y,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#d62728', width=2, dash='dash')
                        ))
                    except:
                        # Skip trendline if there's an error
                        pass
                
                # Update layout for better aesthetics
                response_fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d",
                        tickmode="auto",
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Response Time (seconds)",
                        gridcolor='rgba(0,0,0,0.1)'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(response_fig, use_container_width=True)
                
                # Add response time distribution
                avg_time = metrics_df['avg_response_time'].mean()
                max_time = metrics_df['avg_response_time'].max()
                min_time = metrics_df['avg_response_time'].min()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Response Time", f"{avg_time:.2f}s")
                col2.metric("Minimum Response Time", f"{min_time:.2f}s")
                col3.metric("Maximum Response Time", f"{max_time:.2f}s")
        
        with tab2:
            # Quality Metrics Analysis (if quality metrics are available)
            quality_columns = ['avg_helpfulness', 'avg_hallucination_risk', 'avg_factual_grounding']
            available_quality = [col for col in quality_columns if col in metrics_df.columns and not metrics_df[col].isna().all()]
            
            if available_quality:
                st.markdown("### Response Quality Analysis")
                
                # Create a more attractive quality metrics visualization
                quality_fig = go.Figure()
                
                # Color mapping for quality metrics
                color_map = {
                    'avg_helpfulness': '#2ca02c',        # Green for helpfulness
                    'avg_hallucination_risk': '#d62728', # Red for hallucination risk
                    'avg_factual_grounding': '#1f77b4'   # Blue for factual grounding
                }
                
                # Name mapping for quality metrics
                name_map = {
                    'avg_helpfulness': 'Helpfulness',
                    'avg_hallucination_risk': 'Hallucination Risk',
                    'avg_factual_grounding': 'Factual Grounding'
                }
                
                # Add available quality metrics
                for col in available_quality:
                    quality_fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df[col],
                        mode='lines+markers',
                        name=name_map.get(col, col),
                        line=dict(color=color_map.get(col, None), width=3),
                        marker=dict(size=8)
                    ))
                
                # Update layout for better aesthetics
                quality_fig.update_layout(
                    height=450,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d",
                        tickmode="auto",
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Score",
                        gridcolor='rgba(0,0,0,0.1)',
                        range=[0, 1]
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(quality_fig, use_container_width=True)
                
                # Add High Quality Responses ratio chart if available
                if 'high_quality_ratio' in metrics_df.columns and not metrics_df['high_quality_ratio'].isna().all():
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create a better high quality ratio chart
                        quality_ratio_fig = go.Figure()
                        
                        # Add the line chart
                        quality_ratio_fig.add_trace(go.Scatter(
                            x=metrics_df['date'], 
                            y=metrics_df['high_quality_ratio'] * 100, 
                            mode='lines+markers', 
                            name='High Quality %',
                            line=dict(color='#2ca02c', width=3),
                            marker=dict(size=8)
                        ))
                        
                        # Add a filled area
                        quality_ratio_fig.add_trace(go.Scatter(
                            x=metrics_df['date'],
                            y=metrics_df['high_quality_ratio'] * 100,
                            fill='tozeroy',
                            fillcolor='rgba(44,160,44,0.2)',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        # Add reference line at 70%
                        quality_ratio_fig.add_shape(
                            type="line",
                            x0=metrics_df['date'].min(),
                            x1=metrics_df['date'].max(),
                            y0=70,
                            y1=70,
                            line=dict(
                                color="rgba(255, 0, 0, 0.5)",
                                width=2,
                                dash="dash",
                            )
                        )
                        
                        # Add annotation for reference line
                        quality_ratio_fig.add_annotation(
                            x=metrics_df['date'].max(),
                            y=70,
                            text="70% Target",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="rgba(255, 0, 0, 0.7)")
                        )
                        
                        # Update layout
                        quality_ratio_fig.update_layout(
                            title="Percentage of High Quality Responses (Helpfulness ≥ 0.7)",
                            height=350,
                            hovermode="x unified",
                            showlegend=False,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis=dict(
                                title="Date",
                                tickangle=-45,
                                tickformat="%b %d"
                            ),
                            yaxis=dict(
                                title="Percentage (%)",
                                range=[0, 100],
                                gridcolor='rgba(0,0,0,0.1)'
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(quality_ratio_fig, use_container_width=True)
                    
                    with col2:
                        # Add a gauge chart for overall quality ratio
                        avg_high_quality = metrics_df['high_quality_ratio'].mean() * 100
                        
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg_high_quality,
                            title={'text': "Average High Quality Rate"},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "#2ca02c" if avg_high_quality >= 70 else "#ff7f0e"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 70], 'color': "rgba(255, 127, 14, 0.3)"},
                                    {'range': [70, 100], 'color': "rgba(44, 160, 44, 0.3)"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        
                        gauge_fig.update_layout(
                            height=350,
                            margin=dict(l=30, r=30, t=50, b=20),
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Add quality score distribution
                st.markdown("### Quality Score Distribution")
                
                if 'avg_helpfulness' in metrics_df.columns and not metrics_df['avg_helpfulness'].isna().all():
                    # Define bins for quality score distribution
                    metrics_df['quality_bin'] = pd.cut(
                        metrics_df['avg_helpfulness'],
                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
                    )
                    
                    quality_dist = metrics_df.groupby('quality_bin').size().reset_index(name='count')
                    
                    # Distribution charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart
                        dist_fig = px.bar(
                            quality_dist,
                            x='quality_bin',
                            y='count',
                            color='quality_bin',
                            color_discrete_sequence=px.colors.sequential.Greens,
                            labels={'quality_bin': 'Quality Score Range', 'count': 'Number of Days'},
                            title="Distribution of Daily Quality Scores"
                        )
                        
                        dist_fig.update_layout(
                            showlegend=False,
                            xaxis_title="Quality Score Range",
                            yaxis_title="Number of Days",
                            height=350,
                            margin=dict(l=20, r=20, t=50, b=20),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(dist_fig, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        pie_fig = px.pie(
                            quality_dist,
                            values='count',
                            names='quality_bin',
                            color='quality_bin',
                            color_discrete_sequence=px.colors.sequential.Greens,
                            hole=0.4,
                            title="Proportion of Quality Score Ranges"
                        )
                        
                        pie_fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        pie_fig.update_traces(textinfo='percent+label')
                        
                        st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.info("No quality metrics data available for the selected period")
        
        with tab3:
            # Search Performance (if search data is available)
            has_searches = 'total_searches' in metrics_df.columns and metrics_df['total_searches'].sum() > 0
            
            if has_searches:
                st.markdown("### Search Performance Analysis")
                
                # Search Volume Trends
                search_fig = go.Figure()
                
                # Add total searches line
                search_fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['total_searches'],
                    mode='lines+markers',
                    name='Total Searches',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Add unique users if available
                if 'unique_users' in metrics_df.columns and metrics_df['unique_users'].sum() > 0:
                    search_fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df['unique_users'],
                        mode='lines+markers',
                        name='Unique Users',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8)
                    ))
                
                # Update layout
                search_fig.update_layout(
                    title="Daily Search Volume",
                    height=400,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d",
                        tickmode="auto",
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Count",
                        gridcolor='rgba(0,0,0,0.1)'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(search_fig, use_container_width=True)
                
                # Search Success Rate
                if 'success_rate' in metrics_df.columns and not metrics_df['success_rate'].isna().all():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create success rate chart
                        success_fig = go.Figure()
                        
                        # Add the line chart
                        success_fig.add_trace(go.Scatter(
                            x=metrics_df['date'],
                            y=metrics_df['success_rate'],
                            mode='lines+markers',
                            name='Success Rate (%)',
                            line=dict(color='#2ca02c', width=3),
                            marker=dict(size=8)
                        ))
                        
                        # Add a filled area
                        success_fig.add_trace(go.Scatter(
                            x=metrics_df['date'],
                            y=metrics_df['success_rate'],
                            fill='tozeroy',
                            fillcolor='rgba(44,160,44,0.2)',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        # Update layout
                        success_fig.update_layout(
                            title="Search Success Rate",
                            height=350,
                            hovermode="x unified",
                            showlegend=False,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis=dict(
                                title="Date",
                                tickangle=-45,
                                tickformat="%b %d"
                            ),
                            yaxis=dict(
                                title="Success Rate (%)",
                                gridcolor='rgba(0,0,0,0.1)'
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(success_fig, use_container_width=True)
                    
                    with col2:
                        # Add Response Time chart if available
                        if 'avg_response_time' in metrics_df.columns and not metrics_df['avg_response_time'].isna().all():
                            response_fig = go.Figure()
                            
                            # Add the line chart
                            response_fig.add_trace(go.Scatter(
                                x=metrics_df['date'],
                                y=metrics_df['avg_response_time'],
                                mode='lines+markers',
                                name='Avg Response Time',
                                line=dict(color='#d62728', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # Update layout
                            response_fig.update_layout(
                                title="Search Response Time",
                                height=350,
                                hovermode="x unified",
                                showlegend=False,
                                margin=dict(l=20, r=20, t=50, b=20),
                                xaxis=dict(
                                    title="Date",
                                    tickangle=-45,
                                    tickformat="%b %d"
                                ),
                                yaxis=dict(
                                    title="Response Time (s)",
                                    gridcolor='rgba(0,0,0,0.1)'
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(response_fig, use_container_width=True)
                
                # Search Type Distribution if available
                search_type_cols = ['expert_searches', 'publication_searches', 'general_searches']
                if any(col in metrics_df.columns for col in search_type_cols) and sum(metrics_df[col].sum() for col in search_type_cols if col in metrics_df.columns) > 0:
                    st.markdown("### Search Type Distribution")
                    
                    # Prepare data for search type distribution
                    search_types = {}
                    for col in search_type_cols:
                        if col in metrics_df.columns:
                            # Extract type name from column name
                            type_name = col.replace('_searches', '').capitalize()
                            search_types[type_name] = metrics_df[col].sum()
                    
                    # Create a DataFrame for the pie chart
                    search_type_df = pd.DataFrame({
                        'Search Type': list(search_types.keys()),
                        'Count': list(search_types.values())
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create pie chart for search type distribution
                        pie_fig = px.pie(
                            search_type_df,
                            values='Count',
                            names='Search Type',
                            color='Search Type',
                            title="Search Types by Volume"
                        )
                        
                        bar_fig.update_layout(
                            height=350,
                            showlegend=False,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis_title="Search Type",
                            yaxis_title="Number of Searches",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.info("No search data available for the selected period")
        
        # Show detailed metrics in an expandable section
        with st.expander("Detailed Metrics"):
            if not metrics_df.empty:
                # Create a copy to avoid SettingWithCopyWarning
                display_df = metrics_df.copy()
                
                # Format date column nicely
                if 'date' in display_df.columns:
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                
                # Round numeric columns
                numeric_cols = display_df.select_dtypes(include=['float']).columns
                display_df[numeric_cols] = display_df[numeric_cols].round(2)
                
                # Add tabs for different metric groups
                tab1, tab2, tab3 = st.tabs(["Chat Metrics", "Quality Metrics", "Search Metrics"])
                
                with tab1:
                    # Identify chat-related columns
                    chat_cols = ['date', 'total_chats', 'unique_users', 'avg_response_time', 
                                'total_sessions', 'avg_messages_per_session', 'session_success_rate']
                    
                    # Filter columns that exist in the dataframe
                    chat_df = display_df[[col for col in chat_cols if col in display_df.columns]]
                    
                    if not chat_df.empty:
                        # Style the dataframe
                        style_columns = [col for col in ['total_chats', 'avg_response_time', 'session_success_rate'] 
                                        if col in chat_df.columns]
                        
                        if style_columns:
                            st.dataframe(chat_df.style.background_gradient(
                                subset=style_columns,
                                cmap='RdYlGn'
                            ), use_container_width=True)
                        else:
                            st.dataframe(chat_df, use_container_width=True)
                        
                        # Add download button for CSV
                        csv = chat_df.to_csv(index=False)
                        st.download_button(
                            label="Download Chat Metrics as CSV",
                            data=csv,
                            file_name="chat_metrics.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No chat metrics available")
                
                with tab2:
                    # Identify quality-related columns
                    quality_cols = ['date', 'avg_helpfulness', 'avg_hallucination_risk', 
                                    'avg_factual_grounding', 'high_quality_ratio', 'total_quality_evaluations']
                    
                    # Filter columns that exist in the dataframe
                    quality_df = display_df[[col for col in quality_cols if col in display_df.columns]]
                    
                    if not quality_df.empty:
                        # Style the dataframe
                        style_columns = [col for col in ['avg_helpfulness', 'high_quality_ratio'] 
                                        if col in quality_df.columns]
                        
                        if style_columns:
                            st.dataframe(quality_df.style.background_gradient(
                                subset=style_columns,
                                cmap='RdYlGn'
                            ), use_container_width=True)
                        else:
                            st.dataframe(quality_df, use_container_width=True)
                        
                        # Add download button for CSV
                        csv = quality_df.to_csv(index=False)
                        st.download_button(
                            label="Download Quality Metrics as CSV",
                            data=csv,
                            file_name="quality_metrics.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No quality metrics available")
                
                with tab3:
                    # Identify search-related columns
                    search_cols = ['date', 'total_searches', 'unique_users', 'avg_response_time',
                                  'avg_results', 'success_rate', 'expert_searches', 
                                  'publication_searches', 'general_searches']
                    
                    # Filter columns that exist in the dataframe
                    search_df = display_df[[col for col in search_cols if col in display_df.columns]]
                    
                    if not search_df.empty and search_df['total_searches'].sum() > 0:
                        # Style the dataframe
                        style_columns = [col for col in ['total_searches', 'success_rate', 'avg_results'] 
                                        if col in search_df.columns]
                        
                        if style_columns:
                            st.dataframe(search_df.style.background_gradient(
                                subset=style_columns,
                                cmap='RdYlGn'
                            ), use_container_width=True)
                        else:
                            st.dataframe(search_df, use_container_width=True)
                        
                        # Add download button for CSV
                        csv = search_df.to_csv(index=False)
                        st.download_button(
                            label="Download Search Metrics as CSV",
                            data=csv,
                            file_name="search_metrics.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No search metrics available")
            else:
                st.info("No metrics data available")
        
        # Add date/time of last update with better styling
        st.markdown(
            f"""
            <div style="text-align:right;color:#888;font-size:0.8em;padding-top:20px;">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error displaying chat analytics: {e}")
        logger.error(f"Error in display_chat_analytics: {e}")

def main():
    st.set_page_config(page_title="Chat & Search Analytics Dashboard", page_icon="📊", layout="wide")
    
    # Dashboard title and introduction
    st.title("Chat & Search Analytics Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analytics on chat interactions, search performance, and response quality metrics.
    Use the date range selector to view metrics for specific time periods.
    """)
    
    # Dashboard filters in sidebar
    st.sidebar.title("Dashboard Filters")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date.date(), end_date.date()),
        max_value=end_date.date()
    )
    
    # Add additional filter for data granularity
    granularity = st.sidebar.selectbox(
        "Data Granularity",
        ["Daily", "Weekly", "Monthly"],
        index=0
    )
    
    # Information about data sources
    with st.sidebar.expander("Data Sources"):
        st.info("""
        This dashboard uses data from multiple sources:
        - Chat logs and sessions
        - Response quality metrics
        - Search analytics and sessions
        """)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Format for database query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Show loading indicator
        with st.spinner('Loading data...'):
            # Get metrics with selected filters
            with get_db_connection() as conn:
                metrics_df = get_chat_and_search_metrics(conn, start_date_str, end_date_str)
            
            # If weekly or monthly granularity is selected, resample the data
            if granularity == "Weekly" and not metrics_df.empty and 'date' in metrics_df.columns:
                metrics_df = metrics_df.set_index('date')
                # Resample to weekly frequency, taking mean for rates and sum for counts
                numeric_cols = metrics_df.select_dtypes(include=['number']).columns
                count_cols = [col for col in numeric_cols if 'total' in col or 'count' in col or 'searches' in col or 'users' in col]
                rate_cols = [col for col in numeric_cols if col not in count_cols]
                
                # Create a new DataFrame with resampled data
                weekly_df = pd.DataFrame()
                if count_cols:
                    weekly_df = metrics_df[count_cols].resample('W').sum()
                if rate_cols:
                    weekly_rates = metrics_df[rate_cols].resample('W').mean()
                    weekly_df = pd.concat([weekly_df, weekly_rates], axis=1)
                
                # Reset index to make date a column again
                weekly_df = weekly_df.reset_index()
                metrics_df = weekly_df
            
            elif granularity == "Monthly" and not metrics_df.empty and 'date' in metrics_df.columns:
                metrics_df = metrics_df.set_index('date')
                # Resample to monthly frequency
                numeric_cols = metrics_df.select_dtypes(include=['number']).columns
                count_cols = [col for col in numeric_cols if 'total' in col or 'count' in col or 'searches' in col or 'users' in col]
                rate_cols = [col for col in numeric_cols if col not in count_cols]
                
                # Create a new DataFrame with resampled data
                monthly_df = pd.DataFrame()
                if count_cols:
                    monthly_df = metrics_df[count_cols].resample('MS').sum()
                if rate_cols:
                    monthly_rates = metrics_df[rate_cols].resample('MS').mean()
                    monthly_df = pd.concat([monthly_df, monthly_rates], axis=1)
                
                # Reset index to make date a column again
                monthly_df = monthly_df.reset_index()
                metrics_df = monthly_df
        
        # Display dashboard
        display_chat_analytics(metrics_df, {
            'start_date': start_date,
            'end_date': end_date,
            'granularity': granularity
        })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()
                          