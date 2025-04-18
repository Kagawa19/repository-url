import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
import logging
import json
import os
import numpy as np
import streamlit as st
from contextlib import contextmanager
import psycopg2
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger('analytics_dashboard')

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
def get_db_connection(dbname=None):
    """Get database connection with proper error handling and connection cleanup."""
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

def get_content_metrics(
    conn, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> Dict:
    """
    Retrieve content metrics from the database.
    
    Args:
        conn: Database connection
        start_date: Optional start date for filtering metrics
        end_date: Optional end date for filtering metrics
    
    Returns:
        dict: Dictionary of metrics from different database tables
    """
    # Initialize metrics dictionary
    metrics = {
        'resource_metrics': pd.DataFrame(),
        'expert_metrics': pd.DataFrame(),
        'content_metrics': pd.DataFrame(),
        'message_metrics': pd.DataFrame()
    }
    
    try:
        cursor = conn.cursor()
        
        # Check which tables exist for content analytics
        resources_exist = check_table_exists(conn, 'resources_resource')
        experts_exist = check_table_exists(conn, 'experts_expert')
        messages_exist = check_table_exists(conn, 'expert_messages')
        domains_metadata_exists = check_table_exists(conn, 'domains_metadata')
        
        # Prepare date filtering conditions
        date_condition = ""
        date_params = []
        if start_date:
            date_condition += " AND created_at >= %s"
            date_params.append(start_date)
        if end_date:
            date_condition += " AND created_at <= %s"
            date_params.append(end_date)
        
        # Get resources_resource columns
        if resources_exist:
            # Get resources_resource columns
            resource_columns = get_table_columns(conn, 'resources_resource')
            
            # Build dynamic query based on available columns
            select_columns = ["COUNT(*) as total_resources"]
            group_by_columns = []
            
            # Add type column if available
            if 'type' in resource_columns:
                select_columns.append("type")
                group_by_columns.append("type")
            else:
                select_columns.append("'unknown' as type")
            
            # Add source column if available
            if 'source' in resource_columns:
                # Use COALESCE to handle NULL values
                select_columns.append("COALESCE(source, 'unknown') as source")
                group_by_columns.append("COALESCE(source, 'unknown')")
            else:
                select_columns.append("'unknown' as source")
            
            # Create full query
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM resources_resource
                WHERE 1=1 {date_condition}
                {f"GROUP BY {', '.join(group_by_columns)}" if group_by_columns else ""}
            """
            
            # Execute query with date parameters
            cursor.execute(query, tuple(date_params))
                
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            if data:
                # Create DataFrame and add missing columns if needed
                resource_metrics = pd.DataFrame(data, columns=columns)
                
                # Ensure required columns exist
                if 'resource_type' not in resource_metrics.columns and 'type' in resource_metrics.columns:
                    resource_metrics = resource_metrics.rename(columns={'type': 'resource_type'})
                elif 'resource_type' not in resource_metrics.columns:
                    resource_metrics['resource_type'] = 'unknown'
                
                # Make sure source doesn't have any NULLs
                if 'source' in resource_metrics.columns:
                    resource_metrics['source'] = resource_metrics['source'].fillna('unknown')
                    resource_metrics['source'] = resource_metrics['source'].replace('', 'unknown')
                    
                # Add publication_year distribution if available
                if 'publication_year' in resource_columns:
                    year_query = """
                        SELECT 
                            publication_year, 
                            COUNT(*) as count
                        FROM resources_resource
                        WHERE publication_year IS NOT NULL
                        {date_condition}
                        GROUP BY publication_year
                        ORDER BY publication_year
                    """.format(date_condition=date_condition)
                    
                    cursor.execute(year_query, tuple(date_params))
                    
                    year_data = cursor.fetchall()
                    if year_data:
                        year_df = pd.DataFrame(year_data, columns=['publication_year', 'count'])
                        # Add to metrics dictionary
                        metrics['publication_years'] = year_df
                
                # Store resource metrics
                metrics['resource_metrics'] = resource_metrics
                
                # Also get direct source distribution
                source_query = f"""
                    SELECT 
                        COALESCE(source, 'unknown') as source,
                        COUNT(*) as count
                    FROM resources_resource
                    WHERE 1=1 {date_condition}
                    GROUP BY COALESCE(source, 'unknown')
                    ORDER BY count DESC
                """
                
                cursor.execute(source_query, tuple(date_params))
                
                source_data = cursor.fetchall()
                if source_data:
                    source_df = pd.DataFrame(source_data, columns=['source', 'count'])
                    metrics['source_distribution'] = source_df
                
                # Get domain distribution from resources
                if 'domains' in resource_columns:
                    try:
                        domain_query = f"""
                            SELECT 
                                d.domain,
                                COUNT(*) as resource_count
                            FROM resources_resource e, 
                                UNNEST(e.domains) d(domain)
                            WHERE 1=1 {date_condition}
                            GROUP BY d.domain
                            ORDER BY resource_count DESC
                            LIMIT 15
                        """
                        
                        cursor.execute(domain_query, tuple(date_params))
                        
                        domain_data = cursor.fetchall()
                        if domain_data:
                            domain_df = pd.DataFrame(domain_data, columns=['domain', 'resource_count'])
                            # Add to metrics dictionary
                            metrics['resource_domain_distribution'] = domain_df
                    except Exception as e:
                        logger.error(f"Error getting resource domain distribution: {e}")
                
                # Get topics distribution from resources
                if 'topics' in resource_columns:
                    try:
                        # Count publications by topic
                        topics_query = f"""
                            SELECT 
                                jsonb_array_elements_text(topics) AS topic,
                                COUNT(*) AS count
                            FROM resources_resource
                            WHERE topics IS NOT NULL AND topics != '[]'::jsonb {date_condition}
                            GROUP BY topic
                            ORDER BY count DESC
                            LIMIT 15
                        """
                        
                        cursor.execute(topics_query, tuple(date_params))
                        
                        topics_data = cursor.fetchall()
                        if topics_data:
                            topics_df = pd.DataFrame(topics_data, columns=['topic', 'count'])
                            # Add to metrics dictionary
                            metrics['resource_topics_distribution'] = topics_df
                    except Exception as e:
                        logger.error(f"Error getting resource topics distribution: {e}")
        
        # Get domains_metadata table statistics
        if domains_metadata_exists:
            try:
                # Get all domains and their publication counts
                domain_query = """
                    SELECT 
                        domain_name,
                        publication_count,
                        last_updated,
                        jsonb_array_length(topics) as topic_count
                    FROM domains_metadata
                    ORDER BY publication_count DESC
                    LIMIT 20
                """
                
                cursor.execute(domain_query)
                
                domain_data = cursor.fetchall()
                if domain_data:
                    domain_df = pd.DataFrame(domain_data, columns=['domain_name', 'publication_count', 'last_updated', 'topic_count'])
                    # Add to metrics dictionary
                    metrics['domain_metadata'] = domain_df
                
                # Get topic distribution across all domains
                topic_query = """
                    SELECT 
                        t.topic,
                        COUNT(*) as domain_count,
                        SUM(d.publication_count) as publication_count
                    FROM domains_metadata d,
                        jsonb_array_elements_text(d.topics) t(topic)
                    GROUP BY t.topic
                    ORDER BY publication_count DESC
                    LIMIT 30
                """
                
                cursor.execute(topic_query)
                
                topic_data = cursor.fetchall()
                if topic_data:
                    topic_df = pd.DataFrame(topic_data, columns=['topic', 'domain_count', 'publication_count'])
                    # Add to metrics dictionary
                    metrics['topic_metadata'] = topic_df
            except Exception as e:
                logger.error(f"Error getting domains_metadata statistics: {e}")
        
        # Get expert metrics if experts_expert table exists
        if experts_exist:
            # Get expert columns
            expert_columns = get_table_columns(conn, 'experts_expert')
            
            # Count experts by various fields if available
            domain_query = """
                SELECT 
                    COUNT(*) as total_experts
            """
            
            # Add domains if available
            if 'domains' in expert_columns:
                domain_query += """, 
                    COUNT(CASE WHEN domains IS NOT NULL AND domains != '{}' THEN 1 END) as experts_with_domains,
                    COUNT(CASE WHEN normalized_domains IS NOT NULL AND normalized_domains != '{}' THEN 1 END) as experts_with_normalized_domains
                """
            
            # Add fields if available
            if 'fields' in expert_columns:
                domain_query += """,
                    COUNT(CASE WHEN fields IS NOT NULL AND fields != '{}' THEN 1 END) as experts_with_fields
                """
                
            domain_query += f" FROM experts_expert WHERE 1=1 {date_condition}"
            
            cursor.execute(domain_query, tuple(date_params))
            expert_counts = cursor.fetchone()
            
            if expert_counts:
                # Create a counts dictionary
                counts_dict = {'total_experts': expert_counts[0]}
                col_idx = 1
                
                if 'domains' in expert_columns:
                    counts_dict['experts_with_domains'] = expert_counts[col_idx]
                    col_idx += 1
                    counts_dict['experts_with_normalized_domains'] = expert_counts[col_idx]
                    col_idx += 1
                
                if 'fields' in expert_columns:
                    counts_dict['experts_with_fields'] = expert_counts[col_idx]
                
                # Convert to DataFrame with a single row
                expert_metrics = pd.DataFrame([counts_dict])
                
                # Store expert metrics
                metrics['expert_metrics'] = expert_metrics
                
                # Get domain distribution if available
                if 'domains' in expert_columns:
                    try:
                        domain_query = f"""
                            SELECT 
                                d.domain,
                                COUNT(*) as expert_count
                            FROM experts_expert e, 
                                UNNEST(e.domains) d(domain)
                            WHERE 1=1 {date_condition}
                            GROUP BY d.domain
                            ORDER BY expert_count DESC
                            LIMIT 10
                        """
                        
                        cursor.execute(domain_query, tuple(date_params))
                        
                        domain_data = cursor.fetchall()
                        if domain_data:
                            domain_df = pd.DataFrame(domain_data, columns=['domain', 'expert_count'])
                            # Add to metrics dictionary
                            metrics['domain_distribution'] = domain_df
                    except Exception as e:
                        logger.error(f"Error getting domain distribution: {e}")
        
        # Get message metrics if expert_messages table exists
        if messages_exist:
            # Get message columns
            message_columns = get_table_columns(conn, 'expert_messages')
            
            # Build dynamic query based on available columns
            select_columns = ["COUNT(*) as total_messages"]
            
            # Add draft column if available
            if 'draft' in message_columns:
                select_columns.append("COUNT(CASE WHEN draft THEN 1 END) as draft_count")
            
            # Create full query
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM expert_messages
                WHERE 1=1 {date_condition}
            """
            
            # Execute query
            cursor.execute(query, tuple(date_params))
                
            message_data = cursor.fetchone()
            
            if message_data:
                # Create a dict with message metrics
                message_dict = {'total_messages': message_data[0]}
                
                if 'draft' in message_columns:
                    message_dict['draft_count'] = message_data[1]
                else:
                    message_dict['draft_count'] = 0
                
                # Convert to DataFrame with a single row
                message_metrics = pd.DataFrame([message_dict])
                
                # Store message metrics
                metrics['message_metrics'] = message_metrics
                
                # Get message distribution by time if available
                if 'created_at' in message_columns:
                    cursor.execute(f"""
                        SELECT 
                            EXTRACT(HOUR FROM created_at) as hour,
                            COUNT(*) as message_count
                        FROM expert_messages
                        WHERE 1=1 {date_condition}
                        GROUP BY hour
                        ORDER BY hour
                    """, tuple(date_params))
                    
                    hour_data = cursor.fetchall()
                    if hour_data:
                        hour_df = pd.DataFrame(hour_data, columns=['hour', 'message_count'])
                        # Add to metrics dictionary
                        metrics['message_hours'] = hour_df
                
                # Get sender-receiver pairs if available
                if 'sender_id' in message_columns and 'receiver_id' in message_columns:
                    cursor.execute(f"""
                        SELECT 
                            sender_id,
                            COUNT(*) as sent_count,
                            COUNT(DISTINCT receiver_id) as unique_receivers
                        FROM expert_messages
                        WHERE 1=1 {date_condition}
                        GROUP BY sender_id
                        ORDER BY sent_count DESC
                        LIMIT 10
                    """, tuple(date_params))
                    
                    sender_data = cursor.fetchall()
                    if sender_data:
                        sender_df = pd.DataFrame(sender_data, columns=['sender_id', 'sent_count', 'unique_receivers'])
                        # Add to metrics dictionary
                        metrics['sender_stats'] = sender_df
        
        # Close cursor
        cursor.close()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error retrieving content metrics: {e}")
        # Return empty DataFrames in case of error
        return metrics
def display_content_analytics(
    metrics: Dict[str, pd.DataFrame], 
    filters: Optional[Dict] = None
) -> None:
    """
    Display a comprehensive content analytics dashboard with multiple visualizations.
    
    Args:
        metrics (Dict[str, pd.DataFrame]): Dictionary of metrics from database analysis
        filters (Optional[Dict]): Optional filters to apply to the dashboard
    """
    # Dashboard Title
    st.title("Content & Resource Analytics")
    st.markdown("This dashboard provides insights into publications, experts, and messaging activity.")
    
    # Check if we have data to display
    has_resource_data = 'resource_metrics' in metrics and not metrics['resource_metrics'].empty
    has_expert_data = 'expert_metrics' in metrics and not metrics['expert_metrics'].empty
    has_message_data = 'message_metrics' in metrics and not metrics['message_metrics'].empty
    has_domain_metadata = 'domain_metadata' in metrics and not metrics['domain_metadata'].empty
    has_topic_metadata = 'topic_metadata' in metrics and not metrics['topic_metadata'].empty
    
    if not (has_resource_data or has_expert_data or has_message_data or has_domain_metadata):
        st.warning("No content data available for the selected period.")
        return
    
    # Create KPI metrics cards at the top
    col1, col2, col3, col4 = st.columns(4)
    
    # Resources KPI
    with col1:
        total_resources = 0
        if has_resource_data and 'total_resources' in metrics['resource_metrics'].columns:
            total_resources = metrics['resource_metrics']['total_resources'].sum()
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Total Resources</h4>
                <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_resources:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Publications & Documents</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Experts KPI
    with col2:
        total_experts = 0
        if has_expert_data and 'total_experts' in metrics['expert_metrics'].columns:
            total_experts = metrics['expert_metrics']['total_experts'].iloc[0]
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Total Experts</h4>
                <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{total_experts:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Registered Experts</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Messages KPI
    with col3:
        total_messages = 0
        if has_message_data and 'total_messages' in metrics['message_metrics'].columns:
            total_messages = metrics['message_metrics']['total_messages'].iloc[0]
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Total Messages</h4>
                <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{total_messages:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Expert Messages</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Domains KPI
    with col4:
        total_domains = 0
        if has_domain_metadata:
            total_domains = len(metrics['domain_metadata'])
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Knowledge Domains</h4>
                <h2 style="margin:0;padding:10px 0;color:#9467bd;font-size:28px;">{total_domains:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Classification Domains</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Resources", "Experts", "Messages", "Domains & Topics"])
    
    with tab1:
        st.header("Resource Analytics")
        
        if has_resource_data:
            # Resource Distribution by Type
            if 'resource_type' in metrics['resource_metrics'].columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create aggregated type counts
                    type_counts = metrics['resource_metrics'].groupby('resource_type')['total_resources'].sum().reset_index()
                    
                    resource_type_fig = px.pie(
                        type_counts, 
                        values='total_resources', 
                        names='resource_type',
                        title='Resource Distribution by Type',
                        color_discrete_sequence=px.colors.sequential.Blues,
                        hole=0.4
                    )
                    
                    resource_type_fig.update_traces(textinfo='percent+label')
                    resource_type_fig.update_layout(
                        title_font=dict(size=18),
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(resource_type_fig, use_container_width=True)
                
                with col2:
                    # Create aggregated source counts
                    if 'source' in metrics['resource_metrics'].columns:
                        # Use the dedicated source distribution if available
                        if 'source_distribution' in metrics and not metrics['source_distribution'].empty:
                            source_counts = metrics['source_distribution']
                        else:
                            # Clean empty or null values
                            metrics['resource_metrics']['source'] = metrics['resource_metrics']['source'].fillna('unknown')
                            metrics['resource_metrics']['source'] = metrics['resource_metrics']['source'].replace('', 'unknown')
                            
                            # Create the aggregated counts
                            source_counts = metrics['resource_metrics'].groupby('source')['total_resources'].sum().reset_index()
                        
                        # Create the visualization
                        resource_source_fig = px.bar(
                            source_counts, 
                            x='source', 
                            y='count' if 'count' in source_counts.columns else 'total_resources',
                            title='Resources by Source',
                            color='count' if 'count' in source_counts.columns else 'total_resources',
                            color_continuous_scale='Blues'
                        )
                        
                        resource_source_fig.update_layout(
                            xaxis_title="Source",
                            yaxis_title="Resource Count",
                            title_font=dict(size=18),
                            margin=dict(t=50, b=0, l=0, r=0)
                        )
                        
                        st.plotly_chart(resource_source_fig, use_container_width=True)
            
            # Publication years distribution if available
            if 'publication_years' in metrics and not metrics['publication_years'].empty:
                st.subheader("Publication Year Distribution")
                
                year_fig = px.bar(
                    metrics['publication_years'],
                    x='publication_year',
                    y='count',
                    title='Resources by Publication Year',
                    color='count',
                    color_continuous_scale='Blues'
                )
                
                year_fig.update_layout(
                    xaxis_title="Publication Year",
                    yaxis_title="Number of Resources",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(year_fig, use_container_width=True)
            
            # Resource domains distribution if available
            if 'resource_domain_distribution' in metrics and not metrics['resource_domain_distribution'].empty:
                st.subheader("Resource Domains Distribution")
                
                domain_fig = px.bar(
                    metrics['resource_domain_distribution'],
                    x='domain',
                    y='resource_count',
                    title='Resources by Domain',
                    color='resource_count',
                    color_continuous_scale='Blues'
                )
                
                domain_fig.update_layout(
                    xaxis_title="Domain",
                    yaxis_title="Number of Resources",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(domain_fig, use_container_width=True)
            
            # Resource topics distribution if available
            if 'resource_topics_distribution' in metrics and not metrics['resource_topics_distribution'].empty:
                st.subheader("Resource Topics Distribution")
                
                topics_fig = px.bar(
                    metrics['resource_topics_distribution'],
                    x='topic',
                    y='count',
                    title='Resources by Topic',
                    color='count',
                    color_continuous_scale='Blues'
                )
                
                topics_fig.update_layout(
                    xaxis_title="Topic",
                    yaxis_title="Number of Resources",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(topics_fig, use_container_width=True)
        else:
            st.info("No resource data available for the selected period.")
    
    with tab2:
        st.header("Expert Analytics")
        
        if has_expert_data:
            # Expert Domain Distribution
            if 'domain_distribution' in metrics and not metrics['domain_distribution'].empty:
                st.subheader("Top 10 Expert Domains")
                
                domain_fig = px.bar(
                    metrics['domain_distribution'],
                    x='domain',
                    y='expert_count',
                    title='Expert Count by Domain',
                    color='expert_count',
                    color_continuous_scale='Oranges'
                )
                
                domain_fig.update_layout(
                    xaxis_title="Domain",
                    yaxis_title="Number of Experts",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(domain_fig, use_container_width=True)
            
            # Expert Metadata Completeness
            if 'experts_with_domains' in metrics['expert_metrics'].columns:
                completeness_data = {
                    'Metadata': ['With Domains', 'With Normalized Domains', 'With Fields'],
                    'Count': [
                        metrics['expert_metrics']['experts_with_domains'].iloc[0],
                        metrics['expert_metrics']['experts_with_normalized_domains'].iloc[0] if 'experts_with_normalized_domains' in metrics['expert_metrics'].columns else 0,
                        metrics['expert_metrics']['experts_with_fields'].iloc[0] if 'experts_with_fields' in metrics['expert_metrics'].columns else 0
                    ],
                    'Total': [
                        metrics['expert_metrics']['total_experts'].iloc[0],
                        metrics['expert_metrics']['total_experts'].iloc[0],
                        metrics['expert_metrics']['total_experts'].iloc[0]
                    ]
                }
                
                completeness_df = pd.DataFrame(completeness_data)
                completeness_df['Percentage'] = (completeness_df['Count'] / completeness_df['Total'] * 100).round(1)
                
                st.subheader("Expert Metadata Completeness")
                
                # Create completeness chart
                completeness_fig = go.Figure()
                
                for i, row in completeness_df.iterrows():
                    completeness_fig.add_trace(go.Bar(
                        x=[row['Metadata']],
                        y=[row['Percentage']],
                        name=row['Metadata'],
                        text=[f"{row['Count']} / {row['Total']} ({row['Percentage']}%)"],
                        textposition='auto'
                    ))
                
                completeness_fig.update_layout(
                    xaxis_title="Metadata Type",
                    yaxis_title="Percentage Complete",
                    yaxis=dict(range=[0, 100]),
                    title="Expert Profile Completeness",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(completeness_fig, use_container_width=True)
        else:
            st.info("No expert data available for the selected period.")
    
    with tab3:
        st.header("Message Analytics")
        
        if has_message_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create drafts vs published pie chart
                drafts = metrics['message_metrics']['draft_count'].iloc[0]
                published = metrics['message_metrics']['total_messages'].iloc[0] - drafts
                
                status_df = pd.DataFrame({
                    'Status': ['Published', 'Draft'],
                    'Count': [published, drafts]
                })
                
                status_fig = px.pie(
                    status_df,
                    values='Count',
                    names='Status',
                    title='Message Status Distribution',
                    color_discrete_sequence=['#2ca02c', '#d62728'],
                    hole=0.4
                )
                
                status_fig.update_traces(textinfo='percent+label')
                status_fig.update_layout(
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(status_fig, use_container_width=True)
            
            with col2:
                # If we have message hours data, show hourly distribution
                if 'message_hours' in metrics and not metrics['message_hours'].empty:
                    hours_fig = px.bar(
                        metrics['message_hours'],
                        x='hour',
                        y='message_count',
                        title='Messages by Hour of Day',
                        color='message_count',
                        color_continuous_scale='Greens'
                    )
                    
                    hours_fig.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Message Count",
                        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                        title_font=dict(size=18),
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(hours_fig, use_container_width=True)
            
            # If we have sender stats, show top senders
            if 'sender_stats' in metrics and not metrics['sender_stats'].empty:
                st.subheader("Top Message Senders")
                
                # Create two visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    sent_fig = px.bar(
                        metrics['sender_stats'],
                        x='sender_id',
                        y='sent_count',
                        title='Top Senders by Message Count',
                        color='sent_count',
                        color_continuous_scale='Greens'
                    )
                    
                    sent_fig.update_layout(
                        xaxis_title="Sender ID",
                        yaxis_title="Messages Sent",
                        title_font=dict(size=18),
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(sent_fig, use_container_width=True)
                
                with col2:
                    diversity_fig = px.scatter(
                        metrics['sender_stats'],
                        x='sent_count',
                        y='unique_receivers',
                        title='Sender Diversity (Unique Recipients)',
                        size='sent_count',
                        color='unique_receivers',
                        color_continuous_scale='Greens',
                        hover_name='sender_id'
                    )
                    
                    diversity_fig.update_layout(
                        xaxis_title="Total Messages Sent",
                        yaxis_title="Unique Recipients",
                        title_font=dict(size=18),
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(diversity_fig, use_container_width=True)
        else:
            st.info("No message data available for the selected period.")
    
    # New tab for Domains & Topics
    with tab4:
        st.header("Domains & Topics Analytics")
        
        if has_domain_metadata:
            st.subheader("Domain Analytics")
            
            # Domain visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Top domains by publication count
                domain_pub_fig = px.bar(
                    metrics['domain_metadata'],
                    x='domain_name',
                    y='publication_count',
                    title='Top Domains by Publication Count',
                    color='publication_count',
                    color_continuous_scale='Purples'
                )
                
                domain_pub_fig.update_layout(
                    xaxis_title="Domain",
                    yaxis_title="Publication Count",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(domain_pub_fig, use_container_width=True)
            
            with col2:
                # Domains by topic count
                domain_topic_fig = px.bar(
                    metrics['domain_metadata'],
                    x='domain_name',
                    y='topic_count',
                    title='Domains by Number of Topics',
                    color='topic_count',
                    color_continuous_scale='Purples'
                )
                
                domain_topic_fig.update_layout(
                    xaxis_title="Domain",
                    yaxis_title="Number of Topics",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(domain_topic_fig, use_container_width=True)
            
            # Recent domains update time
            metrics['domain_metadata']['days_since_update'] = (pd.Timestamp.now() - pd.to_datetime(metrics['domain_metadata']['last_updated'])).dt.days
            
            update_fig = px.scatter(
                metrics['domain_metadata'],
                x='publication_count',
                y='days_since_update',
                size='topic_count',
                color='days_since_update',
                hover_name='domain_name',
                title='Domain Freshness (Days Since Last Update)',
                color_continuous_scale='Purples_r'  # Reversed scale - darker is fresher
            )
            
            update_fig.update_layout(
                xaxis_title="Publication Count",
                yaxis_title="Days Since Last Update",
                title_font=dict(size=18),
                margin=dict(t=50, b=0, l=0, r=0)
            )
            
            st.plotly_chart(update_fig, use_container_width=True)
            
            # Topic analysis section
            if has_topic_metadata:
                st.subheader("Topic Analytics")
                
                # Top topics by publication count
                topic_pub_fig = px.bar(
                    metrics['topic_metadata'].head(15),
                    x='topic',
                    y='publication_count',
                    title='Top 15 Topics by Publication Count',
                    color='publication_count',
                    color_continuous_scale='Purples'
                )
                
                topic_pub_fig.update_layout(
                    xaxis_title="Topic",
                    yaxis_title="Publication Count",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(topic_pub_fig, use_container_width=True)
                
                # Topics by domain span
                topic_domain_fig = px.scatter(
                    metrics['topic_metadata'].head(30),
                    x='publication_count',
                    y='domain_count',
                    size='publication_count',
                    color='domain_count',
                    hover_name='topic',
                    title='Topic Spread Across Domains',
                    color_continuous_scale='Purples'
                )
                
                topic_domain_fig.update_layout(
                    xaxis_title="Publication Count",
                    yaxis_title="Number of Domains",
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(topic_domain_fig, use_container_width=True)
                
                # Topic tree map
                topic_tree_fig = px.treemap(
                    metrics['topic_metadata'].head(20),
                    path=['topic'],
                    values='publication_count',
                    color='domain_count',
                    title='Topic Hierarchy by Publication Count',
                    color_continuous_scale='Purples'
                )
                
                topic_tree_fig.update_layout(
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(topic_tree_fig, use_container_width=True)
        else:
            st.info("No domain and topic metadata available. Please ensure the domains_metadata table exists.")
    
    # Add detailed metrics tables in an expander
    with st.expander("View Detailed Metrics"):
        metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs(["Resource Metrics", "Expert Metrics", "Message Metrics", "Domain Metrics"])
        
        with metrics_tab1:
            if has_resource_data:
                st.dataframe(metrics['resource_metrics'], use_container_width=True)
                
                # Show source distribution if available
                if 'source_distribution' in metrics and not metrics['source_distribution'].empty:
                    st.subheader("Source Distribution")
                    st.dataframe(metrics['source_distribution'], use_container_width=True)
                
                # Show resource domains distribution if available
                if 'resource_domain_distribution' in metrics and not metrics['resource_domain_distribution'].empty:
                    st.subheader("Resource Domains Distribution")
                    st.dataframe(metrics['resource_domain_distribution'], use_container_width=True)
                
                # Show resource topics distribution if available
                if 'resource_topics_distribution' in metrics and not metrics['resource_topics_distribution'].empty:
                    st.subheader("Resource Topics Distribution")
                    st.dataframe(metrics['resource_topics_distribution'], use_container_width=True)
                
                # Add download button with a unique key
                csv = metrics['resource_metrics'].to_csv(index=False)
                st.download_button(
                    label="Download Resource Metrics CSV",
                    data=csv,
                    file_name="resource_metrics.csv",
                    mime="text/csv",
                    key="resource_metrics_download"  # Added unique key
                )
            else:
                st.info("No resource metrics available.")
        
        with metrics_tab2:
            if has_expert_data:
                st.dataframe(metrics['expert_metrics'], use_container_width=True)
                
                if 'domain_distribution' in metrics and not metrics['domain_distribution'].empty:
                    st.subheader("Domain Distribution")
                    st.dataframe(metrics['domain_distribution'], use_container_width=True)
                
                # Add download button with a unique key
                csv = metrics['expert_metrics'].to_csv(index=False)
                st.download_button(
                    label="Download Expert Metrics CSV",
                    data=csv,
                    file_name="expert_metrics.csv",
                    mime="text/csv",
                    key="expert_metrics_download"  # Added unique key
                )
            else:
                st.info("No expert metrics available.")
        
        with metrics_tab3:
            if has_message_data:
                st.dataframe(metrics['message_metrics'], use_container_width=True)
                
                if 'sender_stats' in metrics and not metrics['sender_stats'].empty:
                    st.subheader("Sender Statistics")
                    st.dataframe(metrics['sender_stats'], use_container_width=True)
                
                # Add download button with a unique key
                csv = metrics['message_metrics'].to_csv(index=False)
                st.download_button(
                    label="Download Message Metrics CSV",
                    data=csv,
                    file_name="message_metrics.csv",
                    mime="text/csv",
                    key="message_metrics_download"  # Added unique key
                )
            else:
                st.info("No message metrics available.")
        
        with metrics_tab4:
            if has_domain_metadata:
                st.subheader("Domain Metadata")
                st.dataframe(metrics['domain_metadata'], use_container_width=True)
                
                if has_topic_metadata:
                    st.subheader("Topic Metadata")
                    st.dataframe(metrics['topic_metadata'], use_container_width=True)
                
                # Add download buttons with unique keys
                domain_csv = metrics['domain_metadata'].to_csv(index=False)
                st.download_button(
                    label="Download Domain Metadata CSV",
                    data=domain_csv,
                    file_name="domain_metadata.csv",
                    mime="text/csv",
                    key="domain_metadata_download"
                )
                
                if has_topic_metadata:
                    topic_csv = metrics['topic_metadata'].to_csv(index=False)
                    st.download_button(
                        label="Download Topic Metadata CSV",
                        data=topic_csv,
                        file_name="topic_metadata.csv",
                        mime="text/csv",
                        key="topic_metadata_download"
                    )
            else:
                st.info("No domain metadata available.")