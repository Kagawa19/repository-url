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
from typing import Dict, List, Optional, Any

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

def ensure_required_tables(conn):
    """Ensure required tables exist, creating them if necessary."""
    cursor = conn.cursor()
    try:
        # Check and create interactions table if needed
        if not check_table_exists(conn, 'interactions'):
            cursor.execute("""
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
            """)
            conn.commit()
            logger.info("Created interactions table")
        
        # Check and create response_quality_metrics table if needed
        if not check_table_exists(conn, 'response_quality_metrics'):
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
            conn.commit()
            logger.info("Created response_quality_metrics table")
        
        # Check and create expert_matching_logs table if needed
        if not check_table_exists(conn, 'expert_matching_logs'):
            cursor.execute("""
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
            """)
            conn.commit()
            logger.info("Created expert_matching_logs table")
        
        # Check and create expert_messages table if needed
        if not check_table_exists(conn, 'expert_messages'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_messages (
                    id SERIAL PRIMARY KEY,
                    sender_id INTEGER NOT NULL,
                    receiver_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    draft BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
            """)
            conn.commit()
            logger.info("Created expert_messages table")
        
    except Exception as e:
        logger.error(f"Error ensuring required tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def get_overview_metrics(conn, start_date, end_date):
    """Get overview metrics with dynamic table detection and query building"""
    cursor = conn.cursor()
    try:
        # Check which tables exist
        interactions_exists = check_table_exists(conn, 'interactions')
        chatbot_logs_exists = check_table_exists(conn, 'chatbot_logs')
        response_quality_exists = check_table_exists(conn, 'response_quality_metrics')
        expert_matching_exists = check_table_exists(conn, 'expert_matching_logs')
        expert_messages_exists = check_table_exists(conn, 'expert_messages')
        
        # Get columns for interactions table if it exists
        interactions_columns = []
        if interactions_exists:
            interactions_columns = get_table_columns(conn, 'interactions')
        
        # Build query components based on available tables
        cte_parts = []
        aliases = []
        join_tables = []
        select_columns = ["COALESCE({}) as date".format(
            ", ".join([f"{alias}.date" for alias in ['InteractionMetrics', 'QualityMetrics', 'ExpertMetrics', 'MessageMetrics'] 
                      if alias in ['InteractionMetrics'] or 
                      (alias == 'QualityMetrics' and chatbot_logs_exists and response_quality_exists) or
                      (alias == 'ExpertMetrics' and expert_matching_exists) or
                      (alias == 'MessageMetrics' and expert_messages_exists)])
        )]
        
        # InteractionMetrics CTE
        if interactions_exists:
            metrics_json_fields = 'metrics' in interactions_columns
            response_time_field = 'response_time' in interactions_columns
            error_field = 'error_occurred' in interactions_columns
            
            # Build the CTE based on available columns
            interaction_cte = """
                InteractionMetrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT user_id) as unique_users,
            """
            
            # Add response_time calculation based on available columns
            if response_time_field:
                interaction_cte += "AVG(response_time) as avg_response_time,"
            elif metrics_json_fields:
                interaction_cte += "AVG(CAST(metrics->>'response_time' AS FLOAT)) as avg_response_time,"
            else:
                interaction_cte += "0.0 as avg_response_time,"
            
            # Add error_rate calculation based on available columns
            if error_field:
                interaction_cte += """
                        COUNT(CASE WHEN error_occurred THEN 1 END)::FLOAT / 
                            NULLIF(COUNT(*), 0) * 100 as error_rate,
                """
            elif metrics_json_fields:
                interaction_cte += """
                        COUNT(CASE WHEN CAST(metrics->>'error_occurred' AS BOOLEAN) THEN 1 END)::FLOAT / 
                            NULLIF(COUNT(*), 0) * 100 as error_rate,
                """
            else:
                interaction_cte += "0.0 as error_rate,"
            
            # Add placeholder for join
            interaction_cte += """
                        0.0 as placeholder_score
                    FROM interactions
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
                )
            """
            
            cte_parts.append(interaction_cte)
            aliases.append('InteractionMetrics')
            join_tables.append('InteractionMetrics')
            
            # Add interaction columns to select
            select_columns.extend([
                "InteractionMetrics.total_interactions",
                "InteractionMetrics.unique_users",
                "InteractionMetrics.avg_response_time",
                "InteractionMetrics.error_rate"
            ])
        else:
            # If interactions table doesn't exist, add default values
            select_columns.extend([
                "0 as total_interactions",
                "0 as unique_users",
                "0.0 as avg_response_time",
                "0.0 as error_rate"
            ])
        
        # QualityMetrics CTE
        if chatbot_logs_exists and response_quality_exists:
            # Check response_quality_metrics columns
            quality_columns = get_table_columns(conn, 'response_quality_metrics')
            has_helpfulness = 'helpfulness_score' in quality_columns
            has_hallucination = 'hallucination_risk' in quality_columns
            has_factual = 'factual_grounding_score' in quality_columns
            
            quality_cte = """
                QualityMetrics AS (
                    SELECT 
                        DATE(cl.timestamp) as date,
            """
            
            # Add quality metrics based on available columns
            if has_helpfulness:
                quality_cte += "AVG(rqm.helpfulness_score) as avg_helpfulness,"
            else:
                quality_cte += "0.0 as avg_helpfulness,"
                
            if has_hallucination:
                quality_cte += "AVG(rqm.hallucination_risk) as avg_hallucination_risk,"
            else:
                quality_cte += "0.0 as avg_hallucination_risk,"
                
            if has_factual:
                quality_cte += "AVG(rqm.factual_grounding_score) as avg_factual_grounding,"
            else:
                quality_cte += "0.0 as avg_factual_grounding,"
            
            quality_cte += """
                        COUNT(*) as quality_evaluations
                    FROM chatbot_logs cl
                    JOIN response_quality_metrics rqm ON cl.id = rqm.interaction_id
                    WHERE cl.timestamp BETWEEN %s AND %s
                    GROUP BY DATE(cl.timestamp)
                )
            """
            
            cte_parts.append(quality_cte)
            aliases.append('QualityMetrics')
            join_tables.append('QualityMetrics')
            
            # Add quality metrics to select
            select_columns.extend([
                "COALESCE(QualityMetrics.avg_helpfulness, 0.0) as avg_quality_score",
                "QualityMetrics.avg_helpfulness",
                "QualityMetrics.avg_hallucination_risk",
                "QualityMetrics.avg_factual_grounding"
            ])
        else:
            # If quality tables don't exist, add default values
            select_columns.extend([
                "0.0 as avg_quality_score",
                "0.0 as avg_helpfulness",
                "0.0 as avg_hallucination_risk",
                "0.0 as avg_factual_grounding"
            ])
        
        # ExpertMetrics CTE
        if expert_matching_exists:
            expert_cte = """
                ExpertMetrics AS (
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as expert_matches,
                        AVG(similarity_score) as avg_similarity,
                        COUNT(CASE WHEN successful THEN 1 END)::FLOAT / 
                            NULLIF(COUNT(*), 0) * 100 as success_rate
                    FROM expert_matching_logs
                    WHERE created_at BETWEEN %s AND %s
                    GROUP BY DATE(created_at)
                )
            """
            
            cte_parts.append(expert_cte)
            aliases.append('ExpertMetrics')
            join_tables.append('ExpertMetrics')
            
            # Add expert metrics to select
            select_columns.extend([
                "COALESCE(ExpertMetrics.expert_matches, 0) as expert_matches",
                "COALESCE(ExpertMetrics.avg_similarity, 0.0) as avg_similarity",
                "COALESCE(ExpertMetrics.success_rate, 0.0) as success_rate"
            ])
        else:
            # If expert matching table doesn't exist, add default values
            select_columns.extend([
                "0 as expert_matches",
                "0.0 as avg_similarity",
                "0.0 as success_rate"
            ])
        
        # MessageMetrics CTE
        if expert_messages_exists:
            message_cte = """
                MessageMetrics AS (
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as total_messages,
                        COUNT(CASE WHEN draft THEN 1 END) as draft_messages
                    FROM expert_messages
                    WHERE created_at BETWEEN %s AND %s
                    GROUP BY DATE(created_at)
                )
            """
            
            cte_parts.append(message_cte)
            aliases.append('MessageMetrics')
            join_tables.append('MessageMetrics')
            
            # Add message metrics to select
            select_columns.extend([
                "COALESCE(MessageMetrics.total_messages, 0) as total_messages",
                "COALESCE(MessageMetrics.draft_messages, 0) as draft_messages"
            ])
        else:
            # If messages table doesn't exist, add default values
            select_columns.extend([
                "0 as total_messages",
                "0 as draft_messages"
            ])
        
        # Build the full query
        if not cte_parts:
            # If no tables exist, return an empty dataframe with default columns
            logger.warning("No relevant tables found for overview metrics")
            columns = ['date', 'total_interactions', 'unique_users', 'avg_response_time', 
                      'error_rate', 'avg_quality_score', 'avg_helpfulness', 
                      'avg_hallucination_risk', 'avg_factual_grounding',
                      'expert_matches', 'avg_similarity', 'success_rate',
                      'total_messages', 'draft_messages']
            return pd.DataFrame(columns=columns)
        
        # Build the WITH clause
        with_clause = "WITH " + ",\n".join(cte_parts)
        
        # Build the SELECT clause
        select_clause = "SELECT\n    " + ",\n    ".join(select_columns)
        
        # Build the FROM and JOIN clauses
        if join_tables:
            primary_table = join_tables[0]
            from_clause = f"FROM {primary_table}"
            
            # Add JOINs for other tables
            join_clauses = []
            for table in join_tables[1:]:
                join_clauses.append(f"FULL OUTER JOIN {table} ON {primary_table}.date = {table}.date")
            
            join_clause = "\n".join(join_clauses)
        else:
            # Should never get here, but just in case
            from_clause = ""
            join_clause = ""
        
        # Put it all together
        query = f"""
            {with_clause}
            {select_clause}
            {from_clause}
            {join_clause}
            ORDER BY date;
        """
        
        # Calculate number of date pairs needed for params
        param_count = len([cte for cte in cte_parts if '%s' in cte])
        params = (start_date, end_date) * param_count
        
        # Execute the query
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        # Create DataFrame and handle missing dates
        df = pd.DataFrame(data, columns=columns)
        
        # Convert 'date' to datetime if not already
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Fill missing values with sensible defaults
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting overview metrics: {e}")
        # Return empty DataFrame with expected columns
        columns = ['date', 'total_interactions', 'unique_users', 'avg_response_time', 
                  'error_rate', 'avg_quality_score', 'avg_helpfulness', 
                  'avg_hallucination_risk', 'avg_factual_grounding',
                  'expert_matches', 'avg_similarity', 'success_rate',
                  'total_messages', 'draft_messages']
        return pd.DataFrame(columns=columns)
    finally:
        cursor.close()

def display_overview_analytics(metrics_df, filters):
    st.subheader("Overview Analytics")
    
    if metrics_df.empty or 'date' not in metrics_df.columns:
        st.warning("No data available for the selected period")
        return
    
    # Fill NaN values with zeros for calculations
    metrics_df_filled = metrics_df.fillna(0)
    
    # Calculate key metrics
    total_interactions = metrics_df_filled['total_interactions'].sum()
    total_messages = metrics_df_filled['total_messages'].sum()
    success_rate = metrics_df_filled['success_rate'].mean()
    avg_quality = metrics_df_filled['avg_quality_score'].mean()

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Interactions", f"{total_interactions:,}")
    with col2:
        st.metric("Total Messages", f"{total_messages:,}")
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Avg Quality", f"{avg_quality:.2f}")

    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Daily Activity",
            "User Engagement",
            "Performance Metrics",
            "Response Quality"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Determine which metrics to show based on data availability
    has_interactions = not (metrics_df_filled['total_interactions'] == 0).all()
    has_messages = not (metrics_df_filled['total_messages'] == 0).all()
    has_users = not (metrics_df_filled['unique_users'] == 0).all()
    has_performance = not (metrics_df_filled['avg_response_time'] == 0).all() or \
                      not (metrics_df_filled['error_rate'] == 0).all()
    has_quality = not (metrics_df_filled['avg_helpfulness'] == 0).all()
    has_expert_matching = not (metrics_df_filled['expert_matches'] == 0).all()

    # Daily Activity chart
    if has_interactions or has_messages:
        if has_interactions:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['total_interactions'],
                    name='Interactions',
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        if has_messages:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['total_messages'],
                    name='Messages',
                    mode='lines+markers'
                ),
                row=1, col=1
            )
    else:
        # Add annotation if no data
        fig.add_annotation(
            x=0.25, y=0.75,
            xref="paper", yref="paper",
            text="No activity data available",
            showarrow=False,
            font=dict(size=14),
            row=1, col=1
        )

    # User Engagement chart
    if has_users:
        fig.add_trace(
            go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['unique_users'],
                name='Unique Users',
                mode='lines+markers'
            ),
            row=1, col=2
        )
    else:
        # Add annotation if no data
        fig.add_annotation(
            x=0.75, y=0.75,
            xref="paper", yref="paper",
            text="No user engagement data available",
            showarrow=False,
            font=dict(size=14),
            row=1, col=2
        )

    # Performance Metrics chart
    if has_performance:
        if not (metrics_df_filled['avg_response_time'] == 0).all():
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_response_time'],
                    name='Response Time',
                    mode='lines'
                ),
                row=2, col=1
            )
        
        if not (metrics_df_filled['error_rate'] == 0).all():
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['error_rate'],
                    name='Error Rate',
                    mode='lines'
                ),
                row=2, col=1
            )
    else:
        # Add annotation if no data
        fig.add_annotation(
            x=0.25, y=0.25,
            xref="paper", yref="paper",
            text="No performance data available",
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )

    # Response Quality or Expert Matching chart
    if has_quality:
        # Show response quality metrics
        fig.add_trace(
            go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_helpfulness'],
                name='Helpfulness',
                mode='lines'
            ),
            row=2, col=2
        )
        
        if not (metrics_df_filled['avg_hallucination_risk'] == 0).all():
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_hallucination_risk'],
                    name='Hallucination Risk',
                    mode='lines'
                ),
                row=2, col=2
            )
    elif has_expert_matching:
        # Fallback to expert matching if quality metrics aren't available
        fig.add_trace(
            go.Bar(
                x=metrics_df['date'],
                y=metrics_df['expert_matches'],
                name='Expert Matches'
            ),
            row=2, col=2
        )
        
        if not (metrics_df_filled['success_rate'] == 0).all():
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['success_rate'],
                    name='Success Rate',
                    mode='lines'
                ),
                row=2, col=2
            )
    else:
        # Add annotation if no data
        fig.add_annotation(
            x=0.75, y=0.25,
            xref="paper", yref="paper",
            text="No quality or expert matching data available",
            showarrow=False,
            font=dict(size=14),
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
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Users", row=1, col=2)
    fig.update_yaxes(title_text="Time (s) / Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed metrics in expandable section
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
            
            # Display with styling
            try:
                style_columns = [col for col in ['total_interactions', 'success_rate', 
                                              'avg_response_time', 'avg_quality_score'] 
                              if col in display_df.columns]
                
                if style_columns:
                    st.dataframe(display_df.style.background_gradient(
                        subset=style_columns,
                        cmap='RdYlGn'
                    ), use_container_width=True)
                else:
                    st.dataframe(display_df, use_container_width=True)
            except Exception as e:
                logger.error(f"Error styling dataframe: {e}")
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No detailed metrics available")

    # Add date/time of last update
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

def main():
    st.set_page_config(page_title="Overview Analytics Dashboard", page_icon="📊", layout="wide")
    
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
            metrics_df = get_overview_metrics(conn, start_date_str, end_date_str)
            
            # Display dashboard
            display_overview_analytics(metrics_df, {
                'start_date': start_date,
                'end_date': end_date
            })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()