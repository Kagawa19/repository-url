import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
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
        
        # Adjust query based on available tables
        if chatbot_logs_exists and chat_sessions_exists:
            cursor.execute("""
                WITH daily_chat_metrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_chats,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(response_time) as avg_response_time
                    FROM chatbot_logs 
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
                ),
                daily_session_metrics AS (
                    SELECT 
                        DATE(start_timestamp) as date,
                        COUNT(*) as total_sessions,
                        AVG(total_messages) as avg_messages_per_session,
                        SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate
                    FROM chat_sessions
                    WHERE start_timestamp BETWEEN %s AND %s
                    GROUP BY DATE(start_timestamp)
                )
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
            """, (start_date, end_date, start_date, end_date))
        
        elif chatbot_logs_exists:
            # Only chatbot_logs exists
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_chats,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(response_time) as avg_response_time,
                    0 as total_sessions,
                    0 as avg_messages_per_session,
                    0 as session_success_rate
                FROM chatbot_logs 
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (start_date, end_date))
        
        elif chat_sessions_exists:
            # Only chat_sessions exists
            cursor.execute("""
                SELECT 
                    DATE(start_timestamp) as date,
                    0 as total_chats,
                    COUNT(DISTINCT user_id) as unique_users,
                    0 as avg_response_time,
                    COUNT(*) as total_sessions,
                    AVG(total_messages) as avg_messages_per_session,
                    SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate
                FROM chat_sessions
                WHERE start_timestamp BETWEEN %s AND %s
                GROUP BY DATE(start_timestamp)
                ORDER BY date
            """, (start_date, end_date))
        
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
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'response_quality_metrics'
        """)
        available_columns = [row[0] for row in cursor.fetchall()]
        
        # Build a dynamic query based on available columns
        select_parts = ["DATE(cl.timestamp) as date"]
        column_mappings = {
            'helpfulness_score': 'avg_helpfulness',
            'hallucination_risk': 'avg_hallucination_risk',
            'factual_grounding_score': 'avg_factual_grounding'
        }
        
        for col, alias in column_mappings.items():
            if col in available_columns:
                select_parts.append(f"AVG(rqm.{col}) as {alias}")
            else:
                select_parts.append(f"0 as {alias}")
        
        select_parts.append("COUNT(*) as total_interactions")
        
        if 'helpfulness_score' in available_columns:
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

def check_and_create_quality_metrics_table(conn):
    """Check if response_quality_metrics table exists, create if not."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'response_quality_metrics'
            )
        """)
        
        if not cursor.fetchone()[0]:
            logger.info("Creating response_quality_metrics table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_quality_metrics (
                    id SERIAL PRIMARY KEY,
                    interaction_id INTEGER NOT NULL,
                    helpfulness_score FLOAT,
                    hallucination_risk FLOAT,
                    factual_grounding_score FLOAT,
                    evaluation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_response_quality_interaction_id 
                ON response_quality_metrics(interaction_id)
            """)
            
            conn.commit()
            logger.info("Created response_quality_metrics table")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking/creating quality metrics table: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def get_chat_metrics(conn, start_date, end_date):
    """
    Synchronous wrapper for getting chat metrics.
    
    Args:
        conn: Database connection
        start_date: Start date for metrics query
        end_date: End date for metrics query
        
    Returns:
        DataFrame containing chat metrics
    """
    # Check if quality metrics table exists, create if needed
    check_and_create_quality_metrics_table(conn)
    
    async def get_metrics_async():
        # Gather metrics concurrently
        daily_metrics, quality_metrics = await asyncio.gather(
            get_daily_metrics(conn, start_date, end_date),
            get_quality_metrics_trends(conn, start_date, end_date)
        )
        
        # If one dataframe is empty, return the other
        if daily_metrics.empty:
            return quality_metrics
        if quality_metrics.empty:
            return daily_metrics
        
        # Only merge if both dataframes have data
        if not daily_metrics.empty and not quality_metrics.empty:
            # Merge daily and quality metrics on date
            metrics = pd.merge(
                daily_metrics,
                quality_metrics,
                on='date',
                how='outer'
            )
            return metrics.fillna(0)
        
        return pd.DataFrame()

    # Run async code in sync context
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metrics = loop.run_until_complete(get_metrics_async())
        loop.close()
        return metrics
    except Exception as e:
        st.error(f"Error getting chat metrics: {e}")
        logger.error(f"Error in get_chat_metrics: {e}")
        return pd.DataFrame()

def display_chat_analytics(metrics_df, filters):
    """Display comprehensive chat analytics dashboard"""
    st.title("Chat Analytics Dashboard")
    
    try:
        if not isinstance(metrics_df, pd.DataFrame):
            st.error("Invalid metrics data received.")
            return
            
        if metrics_df.empty:
            st.warning("No data available for the selected date range.")
            return
        
        # Ensure date column is datetime
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_chats = metrics_df['total_chats'].sum()
            st.metric("Total Chats", f"{total_chats:,}")
        
        with col2:
            avg_response = metrics_df['avg_response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
        
        with col3:
            # Check if helpfulness metric exists
            if 'avg_helpfulness' in metrics_df.columns:
                avg_helpfulness = metrics_df['avg_helpfulness'].mean() if not pd.isna(metrics_df['avg_helpfulness'].mean()) else 0
                st.metric("Avg Helpfulness", f"{avg_helpfulness:.2f}")
            else:
                st.metric("Avg Helpfulness", "N/A")
        
        with col4:
            total_sessions = metrics_df['total_sessions'].sum()
            st.metric("Total Sessions", f"{total_sessions:,}")

        # Chat Volume Trends
        st.subheader("Chat Volume Trends")
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Scatter(
            x=metrics_df['date'], 
            y=metrics_df['total_chats'], 
            mode='lines', 
            name='Total Chats'
        ))
        volume_fig.add_trace(go.Scatter(
            x=metrics_df['date'], 
            y=metrics_df['unique_users'], 
            mode='lines', 
            name='Unique Users'
        ))
        volume_fig.update_layout(title="Daily Chat Volume")
        st.plotly_chart(volume_fig)

        # Session Metrics (if session data is available)
        if 'total_sessions' in metrics_df.columns and metrics_df['total_sessions'].sum() > 0:
            st.subheader("Session Performance")
            session_fig = go.Figure()
            
            # Total Sessions
            session_fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['total_sessions'],
                mode='lines',
                name='Total Sessions',
                yaxis='y1'
            ))
            
            # Average Messages per Session (if available)
            if 'avg_messages_per_session' in metrics_df.columns:
                session_fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_messages_per_session'],
                    mode='lines',
                    name='Avg Messages per Session',
                    yaxis='y2'
                ))
            
            # Session Success Rate (if available)
            if 'session_success_rate' in metrics_df.columns:
                session_fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['session_success_rate'] * 100,  # Convert to percentage
                    mode='lines',
                    name='Session Success Rate (%)',
                    yaxis='y3'
                ))
            
            # Update layout for multiple y-axes
            session_fig.update_layout(
                title="Session Metrics",
                yaxis=dict(title='Total Sessions'),
                yaxis2=dict(title='Avg Messages', overlaying='y', side='right'),
                yaxis3=dict(title='Success Rate (%)', overlaying='y', side='right', anchor='free', position=1)
            )
            st.plotly_chart(session_fig)

        # Response Time Analysis (if response time data is available)
        if 'avg_response_time' in metrics_df.columns and not metrics_df['avg_response_time'].isna().all():
            st.subheader("Response Time Analysis")
            response_fig = go.Figure(
                data=go.Scatter(
                    x=metrics_df['date'], 
                    y=metrics_df['avg_response_time'], 
                    mode='lines', 
                    name='Average Response Time'
                )
            )
            response_fig.update_layout(title="Average Response Time")
            st.plotly_chart(response_fig)

        # Quality Metrics Analysis (if quality metrics are available)
        quality_columns = ['avg_helpfulness', 'avg_hallucination_risk', 'avg_factual_grounding']
        if any(col in metrics_df.columns for col in quality_columns):
            st.subheader("Response Quality Analysis")
            quality_fig = go.Figure()
            
            # Add available quality metrics
            y_axis_count = 1
            for col, title in zip(quality_columns, ['Avg Helpfulness', 'Avg Hallucination Risk', 'Avg Factual Grounding']):
                if col in metrics_df.columns and not metrics_df[col].isna().all():
                    quality_fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df[col],
                        mode='lines',
                        name=title,
                        yaxis=f'y{y_axis_count}'
                    ))
                    y_axis_count += 1
            
            # Only show quality chart if we have at least one metric
            if y_axis_count > 1:
                # Prepare layout for multiple y-axes
                layout_dict = {
                    'title': "Response Quality Trends",
                    'yaxis': dict(title='Helpfulness')
                }
                
                if y_axis_count > 2:
                    layout_dict['yaxis2'] = dict(title='Hallucination Risk', overlaying='y', side='right')
                
                if y_axis_count > 3:
                    layout_dict['yaxis3'] = dict(
                        title='Factual Grounding', 
                        overlaying='y', 
                        side='right', 
                        anchor='free', 
                        position=1
                    )
                
                quality_fig.update_layout(**layout_dict)
                st.plotly_chart(quality_fig)
            
            # Add High Quality Responses ratio chart if available
            if 'high_quality_ratio' in metrics_df.columns and not metrics_df['high_quality_ratio'].isna().all():
                quality_ratio_fig = go.Figure(
                    data=go.Scatter(
                        x=metrics_df['date'], 
                        y=metrics_df['high_quality_ratio'] * 100, 
                        mode='lines', 
                        fill='tozeroy',
                        name='High Quality Responses (%)'
                    )
                )
                quality_ratio_fig.update_layout(
                    title="Percentage of High Quality Responses (Helpfulness ≥ 0.7)",
                    yaxis=dict(title='Percentage (%)', range=[0, 100])
                )
                st.plotly_chart(quality_ratio_fig)

        # Add date/time of last update
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
    except Exception as e:
        st.error(f"Error displaying chat analytics: {e}")
        logger.error(f"Error in display_chat_analytics: {e}")

def main():
    st.set_page_config(page_title="Chat Analytics Dashboard", page_icon="📊", layout="wide")
    
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
            metrics_df = get_chat_metrics(conn, start_date_str, end_date_str)
        
        # Display dashboard
        display_chat_analytics(metrics_df, {
            'start_date': start_date,
            'end_date': end_date
        })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()