import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import logging
import os
import numpy as np
from datetime import datetime, timedelta
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2
from typing import Dict, List, Optional, Any, Union

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
    
    # In Docker Compose, always use service name
    return {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),  # Always use service name in Docker
        'port': os.getenv('POSTGRES_PORT', '5432'),
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

def get_expert_metrics(conn, start_date: str, end_date: str, expert_count: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive expert metrics with dynamic table and column detection.
    
    Args:
        conn: Database connection
        start_date: Start date for filtering metrics
        end_date: End date for filtering metrics
        expert_count: Number of top experts to include
        
    Returns:
        Dictionary containing various expert metrics DataFrames
    """
    # Initialize results dictionary
    metrics = {
        'experts': pd.DataFrame(),
        'expert_search_matches': pd.DataFrame(),
        'expert_messages': pd.DataFrame(),
        'expert_domains': pd.DataFrame(),
        'expert_resources': pd.DataFrame()
    }
    
    cursor = conn.cursor()
    try:
        # Check which tables exist
        experts_expert_exists = check_table_exists(conn, 'experts_expert')
        expert_search_matches_exists = check_table_exists(conn, 'expert_search_matches')
        expert_search_performance_exists = check_table_exists(conn, 'expert_search_performance')
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        expert_messages_exists = check_table_exists(conn, 'expert_messages')
        resources_resource_exists = check_table_exists(conn, 'resources_resource')
        expert_resource_links_exists = check_table_exists(conn, 'expert_resource_links')
        
        # First, check if we have experts_expert table
        if not experts_expert_exists:
            logger.warning("experts_expert table doesn't exist")
            return metrics
        
        # Get expert columns
        expert_columns = get_table_columns(conn, 'experts_expert')
        
        # Check for required columns
        has_first_name = 'first_name' in expert_columns
        has_last_name = 'last_name' in expert_columns
        has_id = 'id' in expert_columns
        
        if not (has_id and has_first_name and has_last_name):
            logger.warning("experts_expert table missing required columns")
            return metrics
        
        # Get all experts
        expert_select = [
            "id",
            "first_name",
            "last_name"
        ]
        
        # Add optional columns if they exist
        if 'designation' in expert_columns:
            expert_select.append("designation")
        else:
            expert_select.append("NULL as designation")
            
        if 'unit' in expert_columns:
            expert_select.append("unit")
        else:
            expert_select.append("NULL as unit")
            
        if 'theme' in expert_columns:
            expert_select.append("theme")
        else:
            expert_select.append("NULL as theme")
            
        if 'domains' in expert_columns:
            expert_select.append("domains")
        else:
            expert_select.append("NULL as domains")
            
        if 'email' in expert_columns:
            expert_select.append("email")
        else:
            expert_select.append("NULL as email")
            
        # Add active status
        if 'is_active' in expert_columns:
            expert_select.append("is_active")
            active_where = "WHERE is_active = TRUE"
        else:
            expert_select.append("TRUE as is_active")
            active_where = ""
        
        # Get all active experts
        query = f"""
            SELECT {', '.join(expert_select)}
            FROM experts_expert
            {active_where}
        """
        
        try:
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            if data:
                metrics['experts'] = pd.DataFrame(data, columns=columns)
                logger.info(f"Retrieved {len(data)} experts")
        except Exception as e:
            logger.error(f"Error retrieving experts: {e}")
        
        # Get expert search match data if available
        if expert_search_matches_exists and search_analytics_exists:
            # Get columns
            esm_columns = get_table_columns(conn, 'expert_search_matches')
            sa_columns = get_table_columns(conn, 'search_analytics')
            
            # Check for required columns
            has_esm_columns = all(col in esm_columns for col in ['search_id', 'expert_id', 'similarity_score'])
            has_sa_timestamp = 'timestamp' in sa_columns
            
            if has_esm_columns and has_sa_timestamp:
                # Build select list based on available columns
                select_cols = [
                    "esm.expert_id",
                    "COUNT(*) as match_count",
                    "AVG(esm.similarity_score) as avg_similarity"
                ]
                
                if 'rank_position' in esm_columns:
                    select_cols.append("AVG(esm.rank_position) as avg_rank")
                
                if 'clicked' in esm_columns:
                    select_cols.append("SUM(CASE WHEN esm.clicked THEN 1 ELSE 0 END) as clicks")
                    select_cols.append("SUM(CASE WHEN esm.clicked THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) * 100 as click_rate")
                
                # Query expert search matches
                query = f"""
                    SELECT {', '.join(select_cols)}
                    FROM expert_search_matches esm
                    JOIN search_analytics sa ON esm.search_id = sa.id
                    WHERE sa.timestamp BETWEEN %s AND %s
                    GROUP BY esm.expert_id
                    ORDER BY match_count DESC
                    LIMIT %s
                """
                
                try:
                    cursor.execute(query, (start_date, end_date, expert_count))
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    if data:
                        metrics['expert_search_matches'] = pd.DataFrame(data, columns=columns)
                        logger.info(f"Retrieved {len(data)} expert search match records")
                except Exception as e:
                    logger.error(f"Error retrieving expert search matches: {e}")
        
        # Get expert messages if available
        if expert_messages_exists:
            # Get columns
            em_columns = get_table_columns(conn, 'expert_messages')
            
            # Check for required columns
            has_em_columns = all(col in em_columns for col in ['sender_id', 'receiver_id', 'created_at'])
            
            if has_em_columns:
                # Build query based on available columns
                select_cols = [
                    "sender_id",
                    "COUNT(*) as sent_count",
                    "COUNT(DISTINCT receiver_id) as unique_receivers"
                ]
                
                if 'draft' in em_columns:
                    select_cols.append("SUM(CASE WHEN draft THEN 1 ELSE 0 END) as draft_count")
                
                # Query expert messages
                query = f"""
                    SELECT {', '.join(select_cols)}
                    FROM expert_messages
                    WHERE created_at BETWEEN %s AND %s
                    GROUP BY sender_id
                    ORDER BY sent_count DESC
                    LIMIT %s
                """
                
                try:
                    cursor.execute(query, (start_date, end_date, expert_count))
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    if data:
                        metrics['expert_messages'] = pd.DataFrame(data, columns=columns)
                        logger.info(f"Retrieved {len(data)} expert message stats")
                except Exception as e:
                    logger.error(f"Error retrieving expert messages: {e}")
        
        # Get expert domains if available
        if 'domains' in expert_columns:
            try:
                query = """
                    SELECT 
                        d.domain,
                        COUNT(*) as expert_count
                    FROM experts_expert e, 
                        UNNEST(e.domains) d(domain)
                    GROUP BY d.domain
                    ORDER BY expert_count DESC
                    LIMIT 20
                """
                
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['expert_domains'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} expert domain stats")
            except Exception as e:
                logger.error(f"Error retrieving expert domains: {e}")
        
        # Get expert resources if available
        if resources_resource_exists and expert_resource_links_exists:
            try:
                query = """
                    SELECT 
                        erl.expert_id,
                        COUNT(*) as resource_count,
                        COUNT(DISTINCT r.type) as resource_types
                    FROM expert_resource_links erl
                    JOIN resources_resource r ON erl.resource_id = r.id
                    GROUP BY erl.expert_id
                    ORDER BY resource_count DESC
                    LIMIT %s
                """
                
                cursor.execute(query, (expert_count,))
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['expert_resources'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} expert resource stats")
            except Exception as e:
                logger.error(f"Error retrieving expert resources: {e}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in get_expert_metrics: {e}")
        return metrics
    finally:
        cursor.close()

def display_expert_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict] = None):
    """
    Display comprehensive expert analytics with enhanced visualizations.
    
    Args:
        metrics: Dictionary of metrics dataframes
        filters: Optional display filters
    """
    st.title("Expert Analytics Dashboard")
    st.markdown("This dashboard provides insights into expert profiles, search matching, and messaging activity.")
    
    # Check if we have basic expert data
    if metrics['experts'].empty:
        st.warning("No expert data available in the system")
        return
    
    # Combine expert data with search matches and messages
    experts_df = metrics['experts']
    search_matches_df = metrics['expert_search_matches']
    messages_df = metrics['expert_messages']
    
    # Create combined dataframe for analysis
    combined_df = experts_df.copy()
    combined_df['expert_name'] = combined_df['first_name'] + ' ' + combined_df['last_name']
    
    # Add search match metrics if available
    if not search_matches_df.empty:
        # Convert expert_id to proper type for joining
        search_matches_df['expert_id'] = search_matches_df['expert_id'].astype(str)
        combined_df['id'] = combined_df['id'].astype(str)
        
        # Merge with experts
        combined_df = pd.merge(
            combined_df,
            search_matches_df,
            left_on='id',
            right_on='expert_id',
            how='left'
        )
        
        # Fill NaN values with 0
        for col in ['match_count', 'avg_similarity', 'avg_rank', 'clicks', 'click_rate']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(0)
    else:
        # Add placeholder columns
        combined_df['match_count'] = 0
        combined_df['avg_similarity'] = 0
        combined_df['avg_rank'] = 0
        combined_df['clicks'] = 0
        combined_df['click_rate'] = 0
    
    # Add message metrics if available
    if not messages_df.empty:
        # Convert IDs to same type for joining
        messages_df['sender_id'] = messages_df['sender_id'].astype(str)
        combined_df['id'] = combined_df['id'].astype(str)
        
        # Merge with experts
        combined_df = pd.merge(
            combined_df,
            messages_df,
            left_on='id',
            right_on='sender_id',
            how='left'
        )
        
        # Fill NaN values with 0
        for col in ['sent_count', 'unique_receivers', 'draft_count']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(0)
    else:
        # Add placeholder columns
        combined_df['sent_count'] = 0
        combined_df['unique_receivers'] = 0
        combined_df['draft_count'] = 0
    
    # Add resource metrics if available
    if not metrics['expert_resources'].empty:
        # Convert IDs to same type for joining
        resource_df = metrics['expert_resources']
        resource_df['expert_id'] = resource_df['expert_id'].astype(str)
        combined_df['id'] = combined_df['id'].astype(str)
        
        # Merge with experts
        combined_df = pd.merge(
            combined_df,
            resource_df,
            left_on='id',
            right_on='expert_id',
            how='left'
        )
        
        # Fill NaN values with 0
        for col in ['resource_count', 'resource_types']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(0)
    else:
        # Add placeholder columns
        combined_df['resource_count'] = 0
        combined_df['resource_types'] = 0
    
    # Create KPI metrics cards at the top
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Experts KPI
    with col1:
        total_experts = len(combined_df)
        active_experts = combined_df['is_active'].sum() if 'is_active' in combined_df.columns else total_experts
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Total Experts</h4>
                <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{active_experts:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Active Experts</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Search Matches KPI
    with col2:
        total_matches = int(combined_df['match_count'].sum()) if 'match_count' in combined_df.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Search Matches</h4>
                <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{total_matches:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Total Expert Matches</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Messages KPI
    with col3:
        total_messages = int(combined_df['sent_count'].sum()) if 'sent_count' in combined_df.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Messages Sent</h4>
                <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{total_messages:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Expert Messages</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Resources KPI
    with col4:
        total_resources = int(combined_df['resource_count'].sum()) if 'resource_count' in combined_df.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Resources</h4>
                <h2 style="margin:0;padding:10px 0;color:#d62728;font-size:28px;">{total_resources:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Linked Resources</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Expert Performance", "Search Matching", "Messaging & Resources"])
    
    with tab1:
        st.header("Expert Performance Overview")
        
        # Create overall performance visualization
        has_search_data = combined_df['match_count'].sum() > 0
        has_message_data = combined_df['sent_count'].sum() > 0
        has_resource_data = combined_df['resource_count'].sum() > 0
        
        # Create performance matrix for top experts
        if has_search_data or has_message_data or has_resource_data:
            # Determine which metric to use for ranking
            if has_search_data:
                rank_col = 'match_count'
            elif has_message_data:
                rank_col = 'sent_count'
            elif has_resource_data:
                rank_col = 'resource_count'
            else:
                rank_col = 'id'
            
            # Get top experts
            top_n = min(10, len(combined_df))
            top_experts = combined_df.nlargest(top_n, rank_col)
            
            # Create performance matrix
            fig = go.Figure()
            
            # Create heatmap data
            metrics_cols = []
            metrics_labels = []
            
            if has_search_data:
                metrics_cols.extend(['match_count', 'avg_similarity'])
                metrics_labels.extend(['Search Matches', 'Match Similarity'])
                
                if 'clicks' in top_experts.columns:
                    metrics_cols.append('click_rate')
                    metrics_labels.append('Click Rate (%)')
            
            if has_message_data:
                metrics_cols.extend(['sent_count', 'unique_receivers'])
                metrics_labels.extend(['Messages Sent', 'Unique Recipients'])
            
            if has_resource_data:
                metrics_cols.extend(['resource_count', 'resource_types'])
                metrics_labels.extend(['Resources', 'Resource Types'])
            
            # Create heatmap with normalized values for better comparison
            z_data = []
            for col in metrics_cols:
                if top_experts[col].max() > 0:
                    # Normalize values to 0-1 range
                    normalized = top_experts[col] / top_experts[col].max()
                    z_data.append(normalized.tolist())
                else:
                    z_data.append([0] * len(top_experts))
            
            # Create heatmap
            if z_data:
                heatmap = go.Heatmap(
                    z=z_data,
                    x=top_experts['expert_name'],
                    y=metrics_labels,
                    colorscale='Viridis',
                    hoverongaps=False
                )
                
                fig.add_trace(heatmap)
                
                # Update layout
                fig.update_layout(
                    title="Expert Performance Matrix",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=50),
                    xaxis=dict(
                        title="Expert",
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title="Metric",
                        tickmode='array',
                        tickvals=list(range(len(metrics_labels))),
                        ticktext=metrics_labels
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Expert Domains visualization
        domain_df = metrics['expert_domains']
        if not domain_df.empty:
            st.subheader("Expert Domains Distribution")
            
            # Create bar chart for domains
            fig = px.bar(
                domain_df.head(10),
                x='domain',
                y='expert_count',
                title='Top 10 Expert Domains',
                color='expert_count',
                color_continuous_scale='blues',
                labels={'domain': 'Domain', 'expert_count': 'Number of Experts'}
            )
            
            fig.update_layout(
                xaxis_title="Domain",
                yaxis_title="Number of Experts",
                title_font=dict(size=18),
                margin=dict(l=20, r=20, t=50, b=50),
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Expert table with essential information
        st.subheader("Expert Details")
        
        # Create a filtered dataframe for display
        display_cols = ['expert_name', 'designation', 'unit', 'theme']
        
        if has_search_data:
            display_cols.extend(['match_count', 'avg_similarity'])
            
            if 'clicks' in combined_df.columns:
                display_cols.append('clicks')
                display_cols.append('click_rate')
        
        if has_message_data:
            display_cols.extend(['sent_count', 'unique_receivers'])
        
        if has_resource_data:
            display_cols.extend(['resource_count', 'resource_types'])
        
        # Filter columns that actually exist
        display_cols = [col for col in display_cols if col in combined_df.columns]
        
        # Create display dataframe
        display_df = combined_df[display_cols].copy()
        
        # Sort by appropriate column
        if has_search_data:
            display_df = display_df.sort_values('match_count', ascending=False)
        elif has_message_data:
            display_df = display_df.sort_values('sent_count', ascending=False)
        elif has_resource_data:
            display_df = display_df.sort_values('resource_count', ascending=False)
        
        # Apply styling to numeric columns
        numeric_cols = display_df.select_dtypes(include=['number']).columns.tolist()
        
        try:
            if numeric_cols:
                st.dataframe(
                    display_df.style.background_gradient(
                        subset=numeric_cols,
                        cmap='Blues'
                    ),
                    use_container_width=True
                )
            else:
                st.dataframe(display_df, use_container_width=True)
            
            # Add download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Expert Data CSV",
                data=csv,
                file_name="expert_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            logger.error(f"Error styling dataframe: {e}")
            st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.header("Expert Search Matching")
        
        if has_search_data:
            # Create columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Top experts by match count
                top_search_experts = combined_df.nlargest(10, 'match_count')
                
                fig = px.bar(
                    top_search_experts,
                    x='expert_name',
                    y='match_count',
                    title='Top 10 Experts by Search Matches',
                    color='avg_similarity',
                    color_continuous_scale='viridis',
                    labels={'expert_name': 'Expert', 'match_count': 'Match Count', 'avg_similarity': 'Avg Similarity'}
                )
                
                fig.update_layout(
                    xaxis_title="Expert",
                    yaxis_title="Match Count",
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=50),
                    xaxis=dict(tickangle=-45),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Match similarity distribution
                if 'avg_similarity' in combined_df.columns:
                    fig = px.histogram(
                        combined_df[combined_df['match_count'] > 0],
                        x='avg_similarity',
                        title='Match Similarity Distribution',
                        nbins=20,
                        color_discrete_sequence=['#1f77b4'],
                        labels={'avg_similarity': 'Average Similarity'}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Average Similarity",
                        yaxis_title="Number of Experts",
                        title_font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=50),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Create visualization for match rank and click rate if available
            if 'avg_rank' in combined_df.columns and 'click_rate' in combined_df.columns:
                st.subheader("Expert Ranking vs Click Rate")
                
                fig = px.scatter(
                    combined_df[combined_df['match_count'] > 0].head(20),
                    x='avg_rank',
                    y='click_rate',
                    size='match_count',
                    color='avg_similarity',
                    hover_name='expert_name',
                    title='Ranking Performance by Expert',
                    color_continuous_scale='viridis',
                    labels={
                        'avg_rank': 'Average Rank Position',
                        'click_rate': 'Click-through Rate (%)',
                        'match_count': 'Match Count',
                        'avg_similarity': 'Similarity'
                    }
                )
                
                fig.update_layout(
                    height=500,
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=50),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expert search matching data available for the selected period")
    
    with tab3:
        st.header("Expert Messaging & Resources")
        
        # Create columns for messaging and resources
        col1, col2 = st.columns(2)
        
        with col1:
            if has_message_data:
                # Top experts by messages sent
                top_message_experts = combined_df.nlargest(10, 'sent_count')
                
                fig = px.bar(
                    top_message_experts,
                    x='expert_name',
                    y='sent_count',
                    title='Top 10 Experts by Messages Sent',
                    color='unique_receivers',
                    color_continuous_scale='Blues',
                    labels={
                        'expert_name': 'Expert', 
                        'sent_count': 'Messages Sent', 
                        'unique_receivers': 'Unique Recipients'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Expert",
                    yaxis_title="Messages Sent",
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=50),
                    xaxis=dict(tickangle=-45),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create message recipient network visualization if we have enough data
                if top_message_experts['unique_receivers'].sum() > 0:
                    st.subheader("Messaging Network Size")
                    
                    fig = px.scatter(
                        combined_df[combined_df['sent_count'] > 0],
                        x='sent_count',
                        y='unique_receivers',
                        size='sent_count',
                        color='unique_receivers',
                        hover_name='expert_name',
                        title='Message Network Analysis',
                        labels={
                            'sent_count': 'Total Messages Sent',
                            'unique_receivers': 'Unique Recipients'
                        }
                    )
                    
                    # Add line for equal values (1:1 ratio)
                    max_val = max(
                        combined_df['sent_count'].max(),
                        combined_df['unique_receivers'].max()
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='1:1 Ratio',
                            showlegend=True
                        )
                    )
                    
                    fig.update_layout(
                        height=400,
                        title_font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=50),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expert messaging data available for the selected period")
        
        with col2:
            if has_resource_data:
                # Top experts by resources
                top_resource_experts = combined_df.nlargest(10, 'resource_count')
                
                fig = px.bar(
                    top_resource_experts,
                    x='expert_name',
                    y='resource_count',
                    title='Top 10 Experts by Resources',
                    color='resource_types',
                    color_continuous_scale='Greens',
                    labels={
                        'expert_name': 'Expert', 
                        'resource_count': 'Resource Count', 
                        'resource_types': 'Resource Types'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Expert",
                    yaxis_title="Resource Count",
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=50),
                    xaxis=dict(tickangle=-45),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Resource type diversity
                if 'resource_types' in combined_df.columns and combined_df['resource_types'].sum() > 0:
                    # Create resource diversity pie chart
                    diversity_data = combined_df[combined_df['resource_count'] > 0].groupby('resource_types').size().reset_index(name='count')
                    diversity_data.columns = ['Number of Resource Types', 'Experts']
                    
                    fig = px.pie(
                        diversity_data,
                        values='Experts',
                        names='Number of Resource Types',
                        title='Expert Resource Type Diversity',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    
                    fig.update_layout(
                        title_font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend_title="Resource Types",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    fig.update_traces(textinfo='percent+label')
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expert resource data available for the selected period")
    
        # Initialize corr_columns at a higher scope to fix variable scope issue
        corr_columns = []
        
        # Combined metrics correlation
        if (has_search_data and has_message_data) or (has_search_data and has_resource_data) or (has_message_data and has_resource_data):
            st.subheader("Expert Activity Correlation Analysis")
            
            # Fill the already initialized corr_columns list
            if has_search_data:
                corr_columns.extend(['match_count', 'avg_similarity'])
            if has_message_data:
                corr_columns.extend(['sent_count', 'unique_receivers'])
            if has_resource_data:
                corr_columns.extend(['resource_count', 'resource_types'])
            
            # Filter to only include experts with activity
            active_experts = combined_df[
                (combined_df['match_count'] > 0) | 
                (combined_df['sent_count'] > 0) | 
                (combined_df['resource_count'] > 0)
            ]
            
            if len(active_experts) >= 3 and len(corr_columns) >= 2:
                # Calculate correlation matrix
                corr_matrix = active_experts[corr_columns].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="Metric", y="Metric", color="Correlation"),
                    title="Correlation Between Expert Activities"
                )
                
                fig.update_layout(
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=50),
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Expert Performance Insights
    with st.expander("Expert Performance Insights"):
        st.subheader("Key Observations")
        
        insights = []
        
        # Add search matching insights
        if has_search_data:
            top_match_expert = combined_df.loc[combined_df['match_count'].idxmax(), 'expert_name']
            top_match_count = int(combined_df['match_count'].max())
            avg_match_count = combined_df['match_count'].mean()
            active_experts = (combined_df['match_count'] > 0).sum()
            
            insights.append(f"• **Top matching expert**: {top_match_expert} with {top_match_count} matches")
            insights.append(f"• **Average matches per expert**: {avg_match_count:.1f}")
            insights.append(f"• **Active experts in search**: {active_experts} ({active_experts/len(combined_df)*100:.1f}% of all experts)")
            
            if 'avg_similarity' in combined_df.columns:
                avg_similarity = combined_df[combined_df['match_count'] > 0]['avg_similarity'].mean()
                insights.append(f"• **Average match similarity**: {avg_similarity:.2f}")
        
        # Add messaging insights
        if has_message_data:
            active_messagers = (combined_df['sent_count'] > 0).sum()
            avg_sent = combined_df[combined_df['sent_count'] > 0]['sent_count'].mean()
            avg_recipients = combined_df[combined_df['sent_count'] > 0]['unique_receivers'].mean()
            
            insights.append(f"• **Active messaging experts**: {active_messagers} ({active_messagers/len(combined_df)*100:.1f}% of all experts)")
            insights.append(f"• **Average messages sent**: {avg_sent:.1f} per active expert")
            insights.append(f"• **Average unique recipients**: {avg_recipients:.1f} per messaging expert")
        
        # Add resource insights
        if has_resource_data:
            experts_with_resources = (combined_df['resource_count'] > 0).sum()
            avg_resources = combined_df[combined_df['resource_count'] > 0]['resource_count'].mean()
            
            insights.append(f"• **Experts with resources**: {experts_with_resources} ({experts_with_resources/len(combined_df)*100:.1f}% of all experts)")
            insights.append(f"• **Average resources per expert**: {avg_resources:.1f}")
        
        # Add correlation insights if corr_columns exists and has elements
        if len(corr_columns) >= 2:
            # Find highest correlation
            corr_matrix = combined_df[corr_columns].corr()
            # Mask the diagonal since self-correlations will always be 1.0
            np.fill_diagonal(corr_matrix.values, np.nan)
            max_corr = corr_matrix.max().max()
            max_corr_idx = corr_matrix.unstack().idxmax()
            
            if not pd.isna(max_corr):
                insights.append(f"• **Strongest correlation**: {max_corr:.2f} between {max_corr_idx[0]} and {max_corr_idx[1]}")
        
        # Display insights or message if none
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("Not enough data to generate insights. Try selecting a different date range or wait for more expert activity.")

def main():
    st.set_page_config(page_title="Expert Analytics Dashboard", page_icon="👥", layout="wide")
    
    # Dashboard title and description
    st.title("Expert Analytics Dashboard")
    st.markdown("This dashboard provides insights into expert profiles, search matching, and messaging activity.")
    
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
        
        # Display loading spinner
        with st.spinner("Loading expert metrics..."):
            # Get metrics with selected filters
            with get_db_connection() as conn:
                # Get metrics
                metrics = get_expert_metrics(conn, start_date_str, end_date_str, expert_count)
                
                # Add domain filter if expert domains data is available
                domain_filter = ["All"]
                if 'expert_domains' in metrics and not metrics['expert_domains'].empty:
                    domain_options = ["All"] + sorted(metrics['expert_domains']['domain'].tolist())
                    domain_filter = st.sidebar.multiselect(
                        "Filter by Domain",
                        domain_options,
                        default=["All"]
                    )
                
                # Display dashboard with filters
                display_expert_analytics(metrics, {
                    'start_date': start_date,
                    'end_date': end_date,
                    'expert_count': expert_count,
                    'domain_filter': domain_filter
                })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    import numpy as np
    main()