import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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
        # Create chat_analytics table if needed
        if not check_table_exists(conn, 'chat_analytics'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_analytics (
                    id SERIAL PRIMARY KEY,
                    interaction_id INTEGER NOT NULL,
                    content_id VARCHAR(255) NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    similarity_score FLOAT,
                    rank_position INTEGER,
                    clicked BOOLEAN DEFAULT FALSE
                )
            """)
            conn.commit()
            logger.info("Created chat_analytics table")
            
        # Create expert_search_matches table if needed
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
            
        # Create experts_expert table if needed (simplified version)
        if not check_table_exists(conn, 'experts_expert'):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experts_expert (
                    id integer NOT NULL PRIMARY KEY,
                    first_name character varying(255) NOT NULL,
                    last_name character varying(255) NOT NULL,
                    unit character varying(255),
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            conn.commit()
            logger.info("Created experts_expert table")
            
    except Exception as e:
        logger.error(f"Error ensuring required tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def get_expert_metrics(conn, start_date, end_date, expert_count):
    """Get expert metrics with table and column detection"""
    cursor = conn.cursor()
    try:
        # Check which tables exist
        chat_analytics_exists = check_table_exists(conn, 'chat_analytics')
        chat_interactions_exists = check_table_exists(conn, 'chat_interactions')
        expert_search_matches_exists = check_table_exists(conn, 'expert_search_matches')
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        experts_expert_exists = check_table_exists(conn, 'experts_expert')
        
        # If experts table doesn't exist, no point continuing
        if not experts_expert_exists:
            logger.warning("experts_expert table doesn't exist")
            return pd.DataFrame(columns=[
                'expert_name', 'unit', 'chat_matches', 'chat_similarity',
                'search_matches', 'search_avg_rank', 'search_similarity'
            ])
        
        # Check columns in experts_expert
        experts_columns = get_table_columns(conn, 'experts_expert')
        has_first_name = 'first_name' in experts_columns
        has_last_name = 'last_name' in experts_columns
        has_unit = 'unit' in experts_columns
        has_is_active = 'is_active' in experts_columns
        
        if not (has_first_name and has_last_name):
            logger.warning("experts_expert table missing required name columns")
            return pd.DataFrame(columns=[
                'expert_name', 'unit', 'chat_matches', 'chat_similarity',
                'search_matches', 'search_avg_rank', 'search_similarity'
            ])
        
        # Build CTEs based on available tables
        cte_parts = []
        
        # ChatExperts CTE - if tables exist
        if chat_analytics_exists and chat_interactions_exists:
            # Check if chat_analytics has similarity_score
            chat_analytics_columns = get_table_columns(conn, 'chat_analytics')
            chat_interactions_columns = get_table_columns(conn, 'chat_interactions')
            
            if 'similarity_score' in chat_analytics_columns and 'timestamp' in chat_interactions_columns:
                cte_parts.append("""
                    ChatExperts AS (
                        SELECT 
                            a.content_id as id,
                            COUNT(*) as chat_matches,
                            AVG(a.similarity_score) as chat_similarity
                        FROM chat_analytics a
                        JOIN chat_interactions i ON a.interaction_id = i.id
                        WHERE i.timestamp BETWEEN %s AND %s
                        GROUP BY a.content_id
                    )
                """)
            else:
                cte_parts.append("""
                    ChatExperts AS (
                        SELECT 
                            '0' as id,
                            0 as chat_matches,
                            0.0 as chat_similarity
                        WHERE FALSE
                    )
                """)
        else:
            # Empty ChatExperts CTE
            cte_parts.append("""
                ChatExperts AS (
                    SELECT 
                        '0' as id,
                        0 as chat_matches,
                        0.0 as chat_similarity
                    WHERE FALSE
                )
            """)
        
        # SearchExperts CTE - if tables exist
        if expert_search_matches_exists and search_analytics_exists:
            # Check if expert_search_matches has required columns
            esm_columns = get_table_columns(conn, 'expert_search_matches')
            sa_columns = get_table_columns(conn, 'search_analytics')
            
            if all(col in esm_columns for col in ['expert_id', 'rank_position', 'similarity_score']) and 'timestamp' in sa_columns:
                cte_parts.append("""
                    SearchExperts AS (
                        SELECT 
                            expert_id,
                            COUNT(*) as search_matches,
                            AVG(rank_position) as avg_rank,
                            AVG(similarity_score) as search_similarity
                        FROM expert_search_matches
                        JOIN search_analytics sl ON expert_search_matches.search_id = sl.id
                        WHERE sl.timestamp BETWEEN %s AND %s
                        GROUP BY expert_id
                    )
                """)
            else:
                cte_parts.append("""
                    SearchExperts AS (
                        SELECT 
                            '0' as expert_id,
                            0 as search_matches,
                            0.0 as avg_rank,
                            0.0 as search_similarity
                        WHERE FALSE
                    )
                """)
        else:
            # Empty SearchExperts CTE
            cte_parts.append("""
                SearchExperts AS (
                    SELECT 
                        '0' as expert_id,
                        0 as search_matches,
                        0.0 as avg_rank,
                        0.0 as search_similarity
                    WHERE FALSE
                )
            """)
        
        # Build the expert name selection based on available columns
        expert_name_select = "e.id::text as expert_name"
        if has_first_name and has_last_name:
            expert_name_select = "e.first_name || ' ' || e.last_name as expert_name"
        
        # Build unit selection based on availability
        unit_select = "'Unknown' as unit"
        if has_unit:
            unit_select = "e.unit"
        
        # Build the active filter based on availability
        active_filter = ""
        if has_is_active:
            active_filter = "WHERE e.is_active = true"
        
        # Build the main query
        with_clause = "WITH " + ",\n".join(cte_parts)
        
        main_query = f"""
            {with_clause}
            SELECT 
                {expert_name_select},
                {unit_select},
                COALESCE(ce.chat_matches, 0) as chat_matches,
                COALESCE(ce.chat_similarity, 0) as chat_similarity,
                COALESCE(se.search_matches, 0) as search_matches,
                COALESCE(se.avg_rank, 0) as search_avg_rank,
                COALESCE(se.search_similarity, 0) as search_similarity
            FROM experts_expert e
            LEFT JOIN ChatExperts ce ON e.id::text = ce.id
            LEFT JOIN SearchExperts se ON e.id::text = se.expert_id
            {active_filter}
            ORDER BY (COALESCE(ce.chat_matches, 0) + COALESCE(se.search_matches, 0)) DESC
            LIMIT %s
        """
        
        # Calculate number of date pairs needed for params
        num_date_pairs = 0
        for cte in cte_parts:
            num_date_pairs += cte.count("BETWEEN %s AND %s")
        
        # Build params list
        params = []
        for _ in range(num_date_pairs):
            params.extend([start_date, end_date])
        params.append(expert_count)
        
        # Execute the query
        cursor.execute(main_query, tuple(params))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        # Create DataFrame and convert to appropriate types
        df = pd.DataFrame(data, columns=columns)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['chat_matches', 'chat_similarity', 'search_matches', 
                       'search_avg_rank', 'search_similarity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting expert metrics: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'expert_name', 'unit', 'chat_matches', 'chat_similarity',
            'search_matches', 'search_avg_rank', 'search_similarity'
        ])
    finally:
        cursor.close()

def display_expert_analytics(expert_metrics, filters):
    """
    Display enhanced expert analytics visualizations with resilience to missing data.
    
    Parameters:
    - expert_metrics (pandas.DataFrame): A DataFrame containing the expert metrics data.
    - filters (dict): Filters applied to the analytics
    """
    st.subheader("Expert Analytics")
    
    # Check if we have data
    if expert_metrics.empty:
        st.warning("No expert data available for the selected period")
        return
    
    # Ensure all required columns exist
    required_columns = ['expert_name', 'unit', 'chat_matches', 'chat_similarity',
                       'search_matches', 'search_avg_rank', 'search_similarity']
    
    for col in required_columns:
        if col not in expert_metrics.columns:
            expert_metrics[col] = 0

    # Key metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experts", f"{len(expert_metrics):,}")
    with col2:
        st.metric("Total Chat Matches", f"{expert_metrics['chat_matches'].sum():,}")
    with col3:
        st.metric("Total Search Matches", f"{expert_metrics['search_matches'].sum():,}")
    with col4:
        avg_similarity = expert_metrics['chat_similarity'].mean()
        st.metric("Avg Chat Similarity", f"{avg_similarity:.2f}")
    
    # Check if we have any matches data to display
    has_chat_data = expert_metrics['chat_matches'].sum() > 0
    has_search_data = expert_metrics['search_matches'].sum() > 0
    
    if not has_chat_data and not has_search_data:
        st.info("No chat or search match data available for visualization")
        
        # Just show the expert table
        st.subheader("Expert Performance Details")
        display_columns = [
            'expert_name', 'unit', 'chat_matches', 'search_matches',
            'chat_similarity', 'search_similarity', 'search_avg_rank'
        ]
        
        expert_display = expert_metrics[display_columns].sort_values('chat_matches', ascending=False)
        st.dataframe(expert_display, use_container_width=True)
        
        return

    # Create multi-panel visualization with data availability checks
    fig = make_subplots(
        rows=2, 
        cols=2, 
        subplot_titles=(
            "Expert Performance Matrix", 
            "Top Experts by Matches", 
            "Expert Similarity Distribution", 
            "Search vs Chat Engagement"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Expert Performance Matrix (Top Left)
    # Check if we have enough data
    if len(expert_metrics) > 2 and (has_chat_data or has_search_data):
        # Limit to top 10 experts for better readability
        top_experts = expert_metrics.nlargest(10, 'chat_matches') if has_chat_data else expert_metrics.nlargest(10, 'search_matches')
        
        performance_matrix = go.Heatmap(
            z=[
                top_experts.chat_similarity if has_chat_data else [0] * len(top_experts),
                top_experts.search_similarity if has_search_data else [0] * len(top_experts),
                top_experts.search_avg_rank if has_search_data else [0] * len(top_experts)
            ],
            x=top_experts.expert_name,
            y=['Chat Similarity', 'Search Similarity', 'Search Rank'],
            colorscale='Viridis',
            name='Performance Matrix'
        )
        fig.add_trace(performance_matrix, row=1, col=1)
    else:
        fig.add_annotation(
            x=0.25, y=0.75,
            xref="paper", yref="paper",
            text="Insufficient data for performance matrix",
            showarrow=False,
            font=dict(size=14),
            row=1, col=1
        )

    # 2. Top Experts by Matches (Top Right)
    if has_chat_data or has_search_data:
        # Decide which metric to use
        if has_chat_data:
            match_col = 'chat_matches'
            title = 'Chat Matches'
        else:
            match_col = 'search_matches'
            title = 'Search Matches'
            
        top_experts = expert_metrics.nlargest(10, match_col)
        top_experts_bar = go.Bar(
            x=top_experts['expert_name'], 
            y=top_experts[match_col], 
            name=title,
            text=top_experts[match_col].round(2),
            textposition='auto',
        )
        fig.add_trace(top_experts_bar, row=1, col=2)
    else:
        fig.add_annotation(
            x=0.75, y=0.75,
            xref="paper", yref="paper",
            text="No match data available",
            showarrow=False,
            font=dict(size=14),
            row=1, col=2
        )

    # 3. Expert Similarity Distribution (Bottom Left)
    if has_chat_data:
        similarity_hist = go.Histogram(
            x=expert_metrics['chat_similarity'], 
            name='Chat Similarity Distribution',
            nbinsx=20
        )
        fig.add_trace(similarity_hist, row=2, col=1)
    else:
        fig.add_annotation(
            x=0.25, y=0.25,
            xref="paper", yref="paper",
            text="No chat similarity data available",
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )

    # 4. Search vs Chat Engagement (Bottom Right)
    if has_chat_data and has_search_data:
        scatter = go.Scatter(
            x=expert_metrics['chat_matches'], 
            y=expert_metrics['search_matches'], 
            mode='markers',
            name='Expert Engagement',
            text=expert_metrics['expert_name'],
            hoverinfo='text+x+y',
            marker=dict(
                size=10,
                color=expert_metrics['chat_similarity'],
                colorscale='Viridis',
                showscale=True
            )
        )
        fig.add_trace(scatter, row=2, col=2)
    else:
        fig.add_annotation(
            x=0.75, y=0.25,
            xref="paper", yref="paper",
            text="Insufficient data for engagement comparison",
            showarrow=False,
            font=dict(size=14),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Update axis labels
    fig.update_xaxes(title_text="Experts", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="Experts", row=1, col=2, tickangle=-45)
    fig.update_xaxes(title_text="Chat Similarity", row=2, col=1)
    fig.update_xaxes(title_text="Chat Matches", row=2, col=2)
    
    fig.update_yaxes(title_text="Metrics", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Search Matches", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Expert Details Table
    st.subheader("Expert Performance Details")
    display_columns = [
        'expert_name', 'unit', 'chat_matches', 'search_matches',
        'chat_similarity', 'search_similarity', 'search_avg_rank'
    ]
    
    sort_column = 'chat_matches' if has_chat_data else 'search_matches' if has_search_data else 'expert_name'
    expert_display = expert_metrics[display_columns].sort_values(sort_column, ascending=False)
    
    try:
        # Try to apply styling but fall back to plain dataframe if it fails
        style_columns = [col for col in ['chat_matches', 'search_matches', 'chat_similarity', 'search_similarity'] 
                        if col in expert_display.columns]
        
        if style_columns and expert_display[style_columns].sum().sum() > 0:
            st.dataframe(
                expert_display.style.background_gradient(
                    subset=style_columns, 
                    cmap='YlGnBu'
                ),
                use_container_width=True
            )
        else:
            st.dataframe(expert_display, use_container_width=True)
    except Exception as e:
        logger.error(f"Error styling dataframe: {e}")
        st.dataframe(expert_display, use_container_width=True)

    # Insights and Analysis
    if has_chat_data or has_search_data:
        with st.expander("Expert Analytics Insights"):
            insights = []
            
            if has_chat_data:
                top_chat_expert = expert_metrics.loc[expert_metrics['chat_matches'].idxmax(), 'expert_name']
                insights.append(f"Top Chat Expert: {top_chat_expert}")
                
                avg_similarity = expert_metrics['chat_similarity'].mean()
                if avg_similarity > 0:
                    similarity_std = expert_metrics['chat_similarity'].std()
                    insights.append(f"Average Chat Similarity: {avg_similarity:.2f} ± {similarity_std:.2f}")
            
            if has_chat_data and has_search_data:
                engagement_correlation = expert_metrics['chat_matches'].corr(expert_metrics['search_matches'])
                insights.append(f"Chat vs Search Matches Correlation: {engagement_correlation:.2f}")
            
            if not insights:
                st.write("Not enough data to generate insights.")
            else:
                for insight in insights:
                    st.write(f"• {insight}")
    
    if len(expert_metrics) < 5:
        st.warning("Limited expert data. More comprehensive insights will become available with more experts.")

def main():
    st.set_page_config(page_title="Expert Analytics Dashboard", page_icon="👥", layout="wide")
    
    # Dashboard filters in sidebar
    st.sidebar.title("Dashboard Filters")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date.date(), end_date.date()),
        max_value=end_date.date()
    )
    
    expert_count = st.sidebar.slider(
        "Number of Experts to Display",
        min_value=5,
        max_value=50,
        value=20,
        step=5
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
            expert_metrics = get_expert_metrics(conn, start_date_str, end_date_str, expert_count)
            
            # Display dashboard
            display_expert_analytics(expert_metrics, {
                'start_date': start_date,
                'end_date': end_date,
                'expert_count': expert_count
            })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()