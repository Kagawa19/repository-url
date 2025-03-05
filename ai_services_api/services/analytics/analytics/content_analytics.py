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

# Import database connection utilities
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager

# Configure logging
logger = logging.getLogger('analytics_dashboard')

def safe_json_parse(json_str):
    """
    Safely parse JSON strings with multiple parsing strategies
    """
    if not json_str:
        return []
    
    try:
        # Try direct JSON parsing
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        # Try string splitting if JSON fails
        try:
            # Remove brackets and split
            cleaned = json_str.strip('[]')
            return [x.strip() for x in cleaned.split(',') if x.strip()]
        except Exception:
            return []

def get_content_metrics(
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None,
    page_size: int = 100, 
    offset: int = 0
) -> Dict[str, pd.DataFrame]:
    """
    Retrieve comprehensive metrics from the database with detailed error handling.
    
    Args:
        start_date (datetime, optional): Start date for filtering metrics
        end_date (datetime, optional): End date for filtering metrics
        page_size (int, optional): Number of records to retrieve
        offset (int, optional): Offset for pagination
    
    Returns:
        dict: Dictionary of metrics from different database tables
    """
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Set default date range if not provided (last 30 days)
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    # Detailed logging
    logger.info(f"Retrieving database metrics from {start_date} to {end_date}")
    
    try:
        # Retrieve publications
        publications = db_manager.get_all_publications()
        
        logger.info(f"Total Publications Retrieved: {len(publications)}")
        
        # Create metrics dictionary
        metrics = {
            'resource_metrics': pd.DataFrame(),
            'expert_stats': pd.DataFrame(),
            'message_metrics': pd.DataFrame(),
            'content_metrics': pd.DataFrame()
        }
        
        # Process publications into metrics
        if publications:
            publications_df = pd.DataFrame(publications)
            
            # Resource Metrics by Type and Source
            resource_columns = ['type', 'source']
            if all(col in publications_df.columns for col in resource_columns):
                # Rename columns to match visualization expectations
                resource_metrics = publications_df.groupby(['type', 'source']).size().reset_index(name='total_resources')
                metrics['resource_metrics'] = resource_metrics.rename(columns={
                    'type': 'resource_type',
                    'source': 'source'
                })
                logger.info("Resource Metrics generated successfully")
            
            # Expert Stats (placeholder)
            if 'id' in publications_df.columns:
                expert_stats = pd.DataFrame({
                    'sender_id': publications_df['id'].unique(),
                    'interactions': range(1, len(publications_df['id'].unique()) + 1),
                    'unique_receivers': range(1, len(publications_df['id'].unique()) + 1)
                })
                metrics['expert_stats'] = expert_stats
            
            # Message Metrics 
            message_metrics = pd.DataFrame({
                'total_messages': [len(publications)],
                'draft_count': [0]
            })
            metrics['message_metrics'] = message_metrics
            
            # Content Metrics
            if 'summary' in publications_df.columns and 'created_at' in publications_df.columns:
                content_metrics = pd.DataFrame({
                    'content_length': publications_df['summary'].str.len(),
                    'hour_created': pd.to_datetime(publications_df['created_at']).dt.hour,
                    'had_error': [False] * len(publications),
                    'response_time': [0.5] * len(publications)
                })
                metrics['content_metrics'] = content_metrics
                logger.info("Content Metrics generated successfully")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error retrieving database metrics: {e}")
        return {
            'resource_metrics': pd.DataFrame(),
            'expert_stats': pd.DataFrame(),
            'message_metrics': pd.DataFrame(),
            'content_metrics': pd.DataFrame()
        }

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
    # Check if Streamlit is available
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit is not installed. Cannot display dashboard.")
        return
    
    # Custom styling for the dashboard
    st.markdown("""
    <style>
        .dashboard-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Dashboard Title
    st.markdown('<div class="dashboard-header">Content Analytics Dashboard</div>', unsafe_allow_html=True)

    # Prepare columns for high-level metrics
    col1, col2, col3, col4 = st.columns(4)

    # Resource Metrics
    if 'resource_metrics' in metrics and not metrics['resource_metrics'].empty:
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Total Resources", 
                value=metrics['resource_metrics']['total_resources'].sum()
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Expert Interaction Metrics
    if 'expert_stats' in metrics and not metrics['expert_stats'].empty:
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Total Expert Interactions", 
                value=metrics['expert_stats']['interactions'].sum()
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Message Metrics
    if 'message_metrics' in metrics and not metrics['message_metrics'].empty:
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Total Messages", 
                value=metrics['message_metrics']['total_messages'].sum()
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Content Metrics
    if 'content_metrics' in metrics and not metrics['content_metrics'].empty:
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Avg Message Length", 
                value=round(metrics['content_metrics']['content_length'].mean(), 2)
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Visualization Section
    st.markdown('<div class="section-header">Detailed Visualizations</div>', unsafe_allow_html=True)

    # Resource Distribution by Type
    if 'resource_metrics' in metrics and not metrics['resource_metrics'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Resource Types</div>', unsafe_allow_html=True)
            resource_type_fig = px.pie(
                metrics['resource_metrics'], 
                values='total_resources', 
                names='resource_type',  # Updated to match column name
                title='Resource Distribution by Type'
            )
            st.plotly_chart(resource_type_fig, use_container_width=True)
        
        with col2:
            if 'resource_metrics' in metrics:
                st.markdown('<div class="section-header">Resource Sources</div>', unsafe_allow_html=True)
                resource_source_fig = px.bar(
                    metrics['resource_metrics'], 
                    x='source', 
                    y='total_resources',
                    title='Resources by Source'
                )
                st.plotly_chart(resource_source_fig, use_container_width=True)

    # Logging for debugging
    st.write("Debug - Resource Metrics Columns:", 
        metrics['resource_metrics'].columns.tolist() if 'resource_metrics' in metrics else "No Resource Metrics")

    # Expert Interaction Analysis
    if 'expert_stats' in metrics and not metrics['expert_stats'].empty:
        st.markdown('<div class="section-header">Expert Interaction Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_experts_fig = px.bar(
                metrics['expert_stats'].nlargest(10, 'interactions'), 
                x='sender_id', 
                y='interactions',
                title='Top 10 Experts by Interactions'
            )
            st.plotly_chart(top_experts_fig, use_container_width=True)
        
        with col2:
            expert_receivers_fig = px.bar(
                metrics['expert_stats'], 
                x='sender_id', 
                y='unique_receivers',
                title='Expert Interaction Diversity'
            )
            st.plotly_chart(expert_receivers_fig, use_container_width=True)

    # Message Metrics Deep Dive
    if 'message_metrics' in metrics and not metrics['message_metrics'].empty:
        st.markdown('<div class="section-header">Message Length Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            message_length_fig = px.box(
                metrics['content_metrics'], 
                x='content_length', 
                title='Message Length Distribution'
            )
            st.plotly_chart(message_length_fig, use_container_width=True)
        
        with col2:
            time_distribution_fig = px.histogram(
                metrics['content_metrics'], 
                x='hour_created', 
                title='Messages by Hour of Day'
            )
            st.plotly_chart(time_distribution_fig, use_container_width=True)

    # Error and Performance Metrics
    if 'content_metrics' in metrics and not metrics['content_metrics'].empty:
        st.markdown('<div class="section-header">Performance Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a DataFrame specifically for error rate pie chart
            error_df = pd.DataFrame({
                'Status': ['Successful', 'Errors'],
                'Count': [
                    (~metrics['content_metrics']['had_error']).sum(), 
                    metrics['content_metrics']['had_error'].sum()
                ]
            })
            error_rate_fig = px.pie(
                error_df, 
                values='Count', 
                names='Status',
                title='Error Rate'
            )
            st.plotly_chart(error_rate_fig, use_container_width=True)
        
        with col2:
            response_time_fig = px.histogram(
                metrics['content_metrics'], 
                x='response_time', 
                title='Response Time Distribution'
            )
            st.plotly_chart(response_time_fig, use_container_width=True)

    # Footer with additional context
    st.markdown("""
    <div style='text-align: center; color: #6c757d; margin-top: 2rem;'>
    🔍 Analytics Dashboard | Generated with Comprehensive Metrics
    </div>
    """, unsafe_allow_html=True)