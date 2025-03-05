import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import logging
import os
from datetime import datetime, timedelta
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2
from typing import Dict, Any, Optional, List, Union
import numpy as np
import json
import requests

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

def get_resource_metrics(conn, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Retrieve resource metrics from the database.
    
    Args:
        conn: Database connection
        start_date: Start date for filtering metrics
        end_date: End date for filtering metrics
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing various resource metrics DataFrames
    """
    metrics = {
        'airflow_dags': pd.DataFrame(),
        'api_usage': pd.DataFrame(),
        'db_metrics': pd.DataFrame(),
        'resource_counts': pd.DataFrame(),
        'summary': pd.DataFrame()
    }
    
    cursor = conn.cursor()
    try:
        # 1. Get Airflow DAG information if available
        dag_table_exists = check_table_exists(conn, 'dag')
        
        if dag_table_exists:
            # Get DAG columns to ensure we're working with what's available
            dag_columns = get_table_columns(conn, 'dag')
            
            # Check for required columns
            required_cols = ['dag_id', 'schedule_interval', 'is_active', 'is_paused']
            if all(col in dag_columns for col in required_cols):
                # Query DAG information
                query = """
                    SELECT 
                        dag_id, 
                        is_active, 
                        is_paused, 
                        schedule_interval, 
                        owners,
                        description
                    FROM dag
                    ORDER BY dag_id
                """
                
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    # Create DataFrame
                    metrics['airflow_dags'] = pd.DataFrame(data, columns=columns)
                    
                    # Create DAG status summary
                    status_data = {
                        'Status': ['Active', 'Paused', 'Inactive'],
                        'Count': [
                            len(metrics['airflow_dags'][(metrics['airflow_dags']['is_active'] == True) & (metrics['airflow_dags']['is_paused'] == False)]),
                            len(metrics['airflow_dags'][(metrics['airflow_dags']['is_active'] == True) & (metrics['airflow_dags']['is_paused'] == True)]),
                            len(metrics['airflow_dags'][metrics['airflow_dags']['is_active'] == False])
                        ]
                    }
                    
                    metrics['dag_status'] = pd.DataFrame(status_data)
                    
                    # Create schedule interval summary
                    schedule_data = metrics['airflow_dags']['schedule_interval'].value_counts().reset_index()
                    schedule_data.columns = ['Schedule', 'Count']
                    metrics['dag_schedules'] = schedule_data
                    
                    logger.info(f"Retrieved {len(data)} Airflow DAGs")
        
        # 2. Try to get API usage metrics (using tables that might exist in your system)
        if check_table_exists(conn, 'api_requests'):
            api_columns = get_table_columns(conn, 'api_requests')
            
            if 'timestamp' in api_columns and 'api_name' in api_columns:
                # Query API request counts grouped by API name
                query = """
                    SELECT 
                        api_name, 
                        COUNT(*) as request_count
                    FROM api_requests
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY api_name
                    ORDER BY request_count DESC
                """
                
                cursor.execute(query, (start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['api_usage'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved API usage data with {len(data)} APIs")
        
        # 3. If not, create a mock API usage dataset (demonstrating Gemini usage)
        if metrics['api_usage'].empty:
            # Create mock data for Gemini API usage
            mock_api_data = {
                'api_name': ['Gemini', 'OpenAI', 'Embeddings', 'Search', 'User Auth'],
                'request_count': [450, 120, 380, 850, 980]
            }
            metrics['api_usage'] = pd.DataFrame(mock_api_data)
            logger.info("Created mock API usage data")
        
        # 4. Get database resource metrics
        resources_resource_exists = check_table_exists(conn, 'resources_resource')
        experts_expert_exists = check_table_exists(conn, 'experts_expert')
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        chat_sessions_exists = check_table_exists(conn, 'chat_sessions')
        
        # Create resource counts summary
        resource_counts = []
        
        if resources_resource_exists:
            cursor.execute("SELECT COUNT(*) FROM resources_resource")
            resource_count = cursor.fetchone()[0]
            resource_counts.append(('Publications', resource_count))
        
        if experts_expert_exists:
            cursor.execute("SELECT COUNT(*) FROM experts_expert")
            expert_count = cursor.fetchone()[0]
            resource_counts.append(('Experts', expert_count))
        
        if search_analytics_exists:
            cursor.execute("SELECT COUNT(*) FROM search_analytics WHERE timestamp BETWEEN %s AND %s", (start_date, end_date))
            search_count = cursor.fetchone()[0]
            resource_counts.append(('Searches', search_count))
        
        if chat_sessions_exists:
            cursor.execute("SELECT COUNT(*) FROM chat_sessions WHERE start_timestamp BETWEEN %s AND %s", (start_date, end_date))
            chat_count = cursor.fetchone()[0]
            resource_counts.append(('Chat Sessions', chat_count))
        
        # Create resource counts DataFrame
        if resource_counts:
            metrics['resource_counts'] = pd.DataFrame(resource_counts, columns=['Resource', 'Count'])
            logger.info(f"Retrieved resource counts for {len(resource_counts)} resource types")
        
        # 5. Create an overall summary for the KPI cards
        summary_data = {
            'Metric': ['Total DAGs', 'API Requests', 'Total Resources', 'Database Tables'],
            'Value': [
                len(metrics['airflow_dags']) if not metrics['airflow_dags'].empty else 0,
                metrics['api_usage']['request_count'].sum() if not metrics['api_usage'].empty else 0,
                metrics['resource_counts']['Count'].sum() if not metrics['resource_counts'].empty else 0,
                len([table for table in ['resources_resource', 'experts_expert', 'search_analytics', 'chat_sessions'] 
                     if check_table_exists(conn, table)])
            ]
        }
        
        metrics['summary'] = pd.DataFrame(summary_data)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error retrieving resource metrics: {e}")
        return metrics
    finally:
        cursor.close()

def display_resource_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict[str, Any]] = None):
    """
    Display comprehensive resource analytics with enhanced visualizations.
    
    Args:
        metrics: Dictionary of metrics dataframes
        filters: Optional display filters
    """
    st.title("Resource & Infrastructure Analytics")
    st.markdown("This dashboard provides insights into system resources, API usage, and infrastructure performance.")
    
    # Check if we have any data to display
    if all(df.empty for df in metrics.values()):
        st.warning("No resource metrics available for the selected period")
        return
    
    # Get metrics
    summary = metrics.get('summary', pd.DataFrame())
    airflow_dags = metrics.get('airflow_dags', pd.DataFrame())
    api_usage = metrics.get('api_usage', pd.DataFrame())
    resource_counts = metrics.get('resource_counts', pd.DataFrame())
    dag_status = metrics.get('dag_status', pd.DataFrame())
    
    # Create KPI metrics cards at the top
    if not summary.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        # Extract values from summary
        kpi_values = {}
        for _, row in summary.iterrows():
            kpi_values[row['Metric']] = row['Value']
        
        # Total DAGs KPI
        with col1:
            total_dags = kpi_values.get('Total DAGs', 0)
            
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Airflow DAGs</h4>
                    <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_dags:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Total DAGs</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # API Requests KPI
        with col2:
            api_requests = kpi_values.get('API Requests', 0)
            
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">API Usage</h4>
                    <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{api_requests:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Total API Requests</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Total Resources KPI
        with col3:
            total_resources = kpi_values.get('Total Resources', 0)
            
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Resources</h4>
                    <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{total_resources:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Total System Resources</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Database Tables KPI
        with col4:
            db_tables = kpi_values.get('Database Tables', 0)
            
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                    <h4 style="margin:0;padding:0;color:#31333F;">Database</h4>
                    <h2 style="margin:0;padding:10px 0;color:#d62728;font-size:28px;">{db_tables:,}</h2>
                    <p style="margin:0;color:#666;font-size:14px;">Active Tables</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Create tabs for different resource categories
    tab1, tab2, tab3 = st.tabs(["Airflow DAGs", "API Usage", "System Resources"])
    
    with tab1:
        st.header("Airflow DAG Management")
        
        if not airflow_dags.empty:
            # Create DAG status visualization
            col1, col2 = st.columns(2)
            
            with col1:
                if not dag_status.empty:
                    # Create pie chart for DAG status
                    fig = px.pie(
                        dag_status,
                        values='Count',
                        names='Status',
                        title='DAG Status Distribution',
                        color='Status',
                        color_discrete_map={
                            'Active': '#2ca02c',
                            'Paused': '#ff7f0e',
                            'Inactive': '#d62728'
                        },
                        hole=0.4
                    )
                    
                    fig.update_layout(
                        title_font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend_title="Status",
                        height=350
                    )
                    
                    fig.update_traces(textinfo='percent+label')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create bar chart for DAG schedules
                if 'dag_schedules' in metrics and not metrics['dag_schedules'].empty:
                    fig = px.bar(
                        metrics['dag_schedules'],
                        x='Schedule',
                        y='Count',
                        title='DAG Schedule Types',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(
                        title_font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=20),
                        xaxis_title="Schedule Type",
                        yaxis_title="Number of DAGs",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # DAG Details Table
            st.subheader("DAG Configuration Details")
            
            # Clean up the DataFrame for display
            display_cols = ['dag_id', 'description', 'schedule_interval', 'owners', 'is_active', 'is_paused']
            display_cols = [col for col in display_cols if col in airflow_dags.columns]
            
            # Create display DataFrame
            dag_display = airflow_dags[display_cols].copy()
            
            # Create status column
            dag_display['status'] = 'Unknown'
            if 'is_active' in dag_display.columns and 'is_paused' in dag_display.columns:
                dag_display['status'] = dag_display.apply(
                    lambda x: 'Active' if x['is_active'] and not x['is_paused'] 
                    else 'Paused' if x['is_active'] and x['is_paused'] 
                    else 'Inactive', 
                    axis=1
                )
            
            # Display color-coded table based on status
            if 'status' in dag_display.columns:
                # Define color coding function
                def color_status(val):
                    if val == 'Active':
                        return 'background-color: #d4f7d4'  # Light green
                    elif val == 'Paused':
                        return 'background-color: #ffefd5'  # Light orange
                    elif val == 'Inactive':
                        return 'background-color: #ffdddd'  # Light red
                    return ''
                
                # Apply styling
                st.dataframe(
                    dag_display.style.applymap(
                        color_status, 
                        subset=['status']
                    ),
                    use_container_width=True
                )
            else:
                st.dataframe(dag_display, use_container_width=True)
            
            # Add download button
            csv = dag_display.to_csv(index=False)
            st.download_button(
                label="Download DAG Configuration CSV",
                data=csv,
                file_name="airflow_dags.csv",
                mime="text/csv"
            )
        else:
            st.info("No Airflow DAG data available")
    
    with tab2:
        st.header("API Usage Analysis")
        
        if not api_usage.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create bar chart for API usage
                fig = px.bar(
                    api_usage,
                    y='api_name',
                    x='request_count',
                    title='API Request Distribution',
                    color='request_count',
                    color_continuous_scale='Oranges',
                    orientation='h'
                )
                
                fig.update_layout(
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Request Count",
                    yaxis_title="API Name",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create pie chart for API usage percentage
                fig = px.pie(
                    api_usage,
                    values='request_count',
                    names='api_name',
                    title='API Usage Distribution',
                    hole=0.4
                )
                
                fig.update_layout(
                    title_font=dict(size=16),
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend_title="API",
                    height=400
                )
                
                fig.update_traces(textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # API Usage Metrics
            st.subheader("API Usage Details")
            
            total_requests = api_usage['request_count'].sum()
            top_api = api_usage.loc[api_usage['request_count'].idxmax(), 'api_name']
            top_api_percentage = (api_usage['request_count'].max() / total_requests * 100)
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            col1.metric("Total API Requests", f"{total_requests:,}")
            col2.metric("Top API", top_api)
            col3.metric("Top API Usage", f"{top_api_percentage:.1f}%")
            
            # Cost Estimation (mockup)
            st.subheader("API Cost Estimation")
            
            # Create cost table with some common API pricing structures
            cost_data = {
                'API': ['Gemini', 'OpenAI', 'Embeddings', 'Search', 'User Auth'],
                'Requests': api_usage['request_count'].tolist(),
                'Cost Per 1K': ['$0.0125', '$0.0200', '$0.0001', '$0.0000', '$0.0000'],
                'Estimated Cost': [
                    f"${api_usage.iloc[0]['request_count'] * 0.0125 / 1000:.2f}",
                    f"${api_usage.iloc[1]['request_count'] * 0.0200 / 1000:.2f}" if len(api_usage) > 1 else "$0.00",
                    f"${api_usage.iloc[2]['request_count'] * 0.0001 / 1000:.2f}" if len(api_usage) > 2 else "$0.00",
                    "$0.00",
                    "$0.00"
                ]
            }
            
            cost_df = pd.DataFrame(cost_data)
            
            # Add total row
            total_cost = sum([float(cost.replace('$', '')) for cost in cost_data['Estimated Cost']])
            total_row = pd.DataFrame({
                'API': ['Total'],
                'Requests': [total_requests],
                'Cost Per 1K': [''],
                'Estimated Cost': [f"${total_cost:.2f}"]
            })
            
            cost_df = pd.concat([cost_df, total_row])
            
            # Display the cost table
            st.dataframe(cost_df, use_container_width=True)
            
            # Add download button
            csv = cost_df.to_csv(index=False)
            st.download_button(
                label="Download API Cost Estimation CSV",
                data=csv,
                file_name="api_costs.csv",
                mime="text/csv"
            )
        else:
            st.info("No API usage data available")
    
    with tab3:
        st.header("System Resources")
        
        if not resource_counts.empty:
            # Create visualization for resource counts
            fig = px.bar(
                resource_counts,
                x='Resource',
                y='Count',
                title='Resource Distribution',
                color='Count',
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(
                title_font=dict(size=16),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="Resource Type",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Resource Utilization (mockup)
            st.subheader("Resource Utilization")
            
            # Create mockup for resource utilization
            utilization_data = {
                'Resource': ['CPU', 'Memory', 'Disk', 'Network'],
                'Current': [35, 62, 48, 28],
                'Average': [42, 58, 50, 32],
                'Peak': [78, 85, 72, 64]
            }
            
            util_df = pd.DataFrame(utilization_data)
            
            # Create utilization chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=util_df['Resource'],
                y=util_df['Current'],
                name='Current'
            ))
            
            fig.add_trace(go.Bar(
                x=util_df['Resource'],
                y=util_df['Average'],
                name='Average'
            ))
            
            fig.add_trace(go.Bar(
                x=util_df['Resource'],
                y=util_df['Peak'],
                name='Peak'
            ))
            
            fig.update_layout(
                title="Resource Utilization (%)",
                title_font=dict(size=16),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="Resource",
                yaxis_title="Utilization (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Resource allocation recommendations
            st.subheader("Resource Allocation Recommendations")
            
            recommendations = [
                "**Memory Optimization**: Memory utilization is near 60%. Consider optimizing memory-intensive processes or increasing allocation if utilization consistently rises above 70%.",
                "**DAG Scheduling**: Several DAGs are paused. Review scheduling to balance resource utilization throughout the month.",
                "**API Usage**: Gemini API accounts for most of the API costs. Monitor usage patterns to optimize costs.",
                "**Disk Space**: Current utilization is moderate (48%). Implement regular cleanup jobs for temporary files and logs."
            ]
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.info("No system resource data available")
    
    # Resource insights
    with st.expander("Resource Management Insights"):
        insights = [
            "**Airflow DAG Management**: You have multiple DAGs scheduled on a monthly basis. Consider staggering these to distribute resource utilization more evenly throughout the month.",
            "**API Usage Optimization**: Gemini API is your most used API. Consider implementing caching for frequent similar requests to reduce costs.",
            "**Resource Monitoring**: Set up alerts for resource utilization exceeding 80% to proactively manage system resources.",
            "**Cost Management**: Estimated API costs are primarily driven by Gemini usage. Monitor this closely and consider implementing usage quotas if necessary."
        ]
        
        st.subheader("Key Insights")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Add cost saving opportunities
        st.subheader("Cost Saving Opportunities")
        
        savings = [
            "Implement API request batching to reduce the total number of API calls",
            "Schedule resource-intensive DAGs during off-peak hours",
            "Use tiered API access strategies based on user requirements",
            "Cache common API responses to reduce duplicate requests"
        ]
        
        for save in savings:
            st.markdown(f"• {save}")

def main():
    st.set_page_config(page_title="Resource Analytics Dashboard", page_icon="📊", layout="wide")
    
    # Dashboard title and description
    st.title("Resource Analytics Dashboard")
    st.markdown("This dashboard provides insights into system resources, API usage, and infrastructure performance.")
    
    # Dashboard filters in sidebar
    st.sidebar.title("Dashboard Filters")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date.date(), end_date.date()),
        max_value=end_date.date()
    )
    
    # Add resource type filter
    resource_type = st.sidebar.multiselect(
        "Resource Type",
        ["Airflow DAGs", "API Keys", "Database", "System Resources"],
        default=["Airflow DAGs", "API Keys"]
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Format for database query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Display loading spinner
        with st.spinner("Loading resource metrics..."):
            # Get metrics with selected filters
            with get_db_connection() as conn:
                # Get metrics
                metrics = get_resource_metrics(conn, start_date_str, end_date_str)
                
                # Display dashboard
                display_resource_analytics(metrics, {
                    'start_date': start_date,
                    'end_date': end_date,
                    'resource_type': resource_type
                })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()