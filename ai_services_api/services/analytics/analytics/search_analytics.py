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

def get_search_metrics(conn, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive search metrics using available tables with dynamic table/column checks.
    
    Args:
        conn: Database connection
        start_date: Start date for filtering metrics
        end_date: End date for filtering metrics
    
    Returns:
        dict: Dictionary containing different search metrics dataframes
    """
    cursor = conn.cursor()
    
    # Initialize empty dataframes for results
    metrics = {
        'daily_metrics': pd.DataFrame(),
        'domain_metrics': pd.DataFrame(),
        'query_metrics': pd.DataFrame(),
        'expert_match_metrics': pd.DataFrame()
    }
    
    try:
        # Check which tables exist and match our required schema
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        search_sessions_exists = check_table_exists(conn, 'search_sessions')
        expert_search_matches_exists = check_table_exists(conn, 'expert_search_matches')
        expert_search_performance_exists = check_table_exists(conn, 'expert_search_performance')
        daily_search_metrics_exists = check_table_exists(conn, 'daily_search_metrics')
        
        # If daily_search_metrics view exists, use it directly
        if daily_search_metrics_exists:
            try:
                cursor.execute("""
                    SELECT * FROM daily_search_metrics
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date
                """, (start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['daily_metrics'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} rows from daily_search_metrics view")
            except Exception as e:
                logger.error(f"Error querying daily_search_metrics view: {e}")
        
        # If we don't have the view or it failed, build metrics from raw tables
        if metrics['daily_metrics'].empty and search_analytics_exists:
            # Get columns for search_analytics
            search_analytics_columns = get_table_columns(conn, 'search_analytics')
            
            # Build the daily metrics query based on available columns
            daily_select = ["DATE(timestamp) as date", "COUNT(*) as total_searches", "COUNT(DISTINCT user_id) as unique_users"]
            
            # Add response_time if available
            if 'response_time' in search_analytics_columns:
                daily_select.append("AVG(response_time) as avg_response_time")
            else:
                daily_select.append("0 as avg_response_time")
            
            # Add result_count if available
            if 'result_count' in search_analytics_columns:
                daily_select.append("AVG(result_count) as avg_results")
                daily_select.append("SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) * 100 as success_rate")
            else:
                daily_select.append("0 as avg_results")
                daily_select.append("0 as success_rate")
            
            # Add search_type distribution if available
            if 'search_type' in search_analytics_columns:
                daily_select.append("COUNT(CASE WHEN search_type = 'expert' THEN 1 END) as expert_searches")
                daily_select.append("COUNT(CASE WHEN search_type = 'publication' THEN 1 END) as publication_searches")
                daily_select.append("COUNT(CASE WHEN search_type NOT IN ('expert', 'publication') THEN 1 END) as other_searches")
            
            # Execute the query
            query = f"""
                SELECT {', '.join(daily_select)}
                FROM search_analytics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            
            try:
                cursor.execute(query, (start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['daily_metrics'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} rows of daily search metrics")
            except Exception as e:
                logger.error(f"Error querying search_analytics for daily metrics: {e}")
        
        # Enhance with session metrics if available
        if search_sessions_exists and not metrics['daily_metrics'].empty:
            session_columns = get_table_columns(conn, 'search_sessions')
            
            # Check for required session columns
            has_session_columns = all(col in session_columns for col in ['start_timestamp', 'query_count'])
            has_success_data = 'successful_searches' in session_columns
            has_duration_data = 'end_timestamp' in session_columns
            
            if has_session_columns:
                # Build session metrics query
                session_select = [
                    "DATE(start_timestamp) as date",
                    "COUNT(*) as total_sessions",
                    "AVG(query_count) as avg_queries_per_session"
                ]
                
                if has_success_data:
                    session_select.append("""
                        AVG(CASE 
                            WHEN successful_searches > 0 AND query_count > 0 
                            THEN successful_searches::float / NULLIF(query_count, 0)
                            ELSE 0 
                        END) * 100 as session_success_rate
                    """)
                
                if has_duration_data:
                    session_select.append("""
                        AVG(EXTRACT(epoch FROM (end_timestamp - start_timestamp))) as avg_session_duration
                    """)
                
                # Add where clause for date filtering
                query = f"""
                    SELECT {', '.join(session_select)}
                    FROM search_sessions
                    WHERE start_timestamp BETWEEN %s AND %s
                """
                
                if has_duration_data:
                    query += " AND end_timestamp IS NOT NULL"
                
                query += " GROUP BY DATE(start_timestamp)"
                
                try:
                    cursor.execute(query, (start_date, end_date))
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    if data:
                        session_metrics = pd.DataFrame(data, columns=columns)
                        
                        # Merge with daily metrics
                        if not session_metrics.empty and 'date' in session_metrics.columns:
                            # Convert date columns to same type
                            metrics['daily_metrics']['date'] = pd.to_datetime(metrics['daily_metrics']['date'])
                            session_metrics['date'] = pd.to_datetime(session_metrics['date'])
                            
                            # Merge on date
                            metrics['daily_metrics'] = pd.merge(
                                metrics['daily_metrics'],
                                session_metrics,
                                on='date',
                                how='left'
                            ).fillna(0)
                            
                            logger.info(f"Enhanced daily metrics with {len(data)} rows of session data")
                except Exception as e:
                    logger.error(f"Error querying search_sessions for session metrics: {e}")
        
        # Add expert search matching metrics if available
        if expert_search_matches_exists and 'search_id' in get_table_columns(conn, 'search_analytics'):
            expert_match_columns = get_table_columns(conn, 'expert_search_matches')
            
            # Check for required columns
            has_match_data = all(col in expert_match_columns for col in ['search_id', 'expert_id'])
            has_similarity = 'similarity_score' in expert_match_columns
            has_rank = 'rank_position' in expert_match_columns
            has_clicked = 'clicked' in expert_match_columns
            
            if has_match_data:
                # Build expert match query
                match_select = [
                    "DATE(sa.timestamp) as date",
                    "COUNT(DISTINCT esm.expert_id) as matched_experts"
                ]
                
                if has_similarity:
                    match_select.append("AVG(esm.similarity_score) as avg_similarity")
                
                if has_rank:
                    match_select.append("AVG(esm.rank_position) as avg_rank")
                
                if has_clicked:
                    match_select.append("""
                        SUM(CASE WHEN esm.clicked THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) * 100 as click_rate
                    """)
                
                # Build query
                query = f"""
                    SELECT {', '.join(match_select)}
                    FROM search_analytics sa
                    JOIN expert_search_matches esm ON sa.search_id = esm.search_id
                    WHERE sa.timestamp BETWEEN %s AND %s
                    GROUP BY DATE(sa.timestamp)
                    ORDER BY date
                """
                
                try:
                    cursor.execute(query, (start_date, end_date))
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    if data:
                        match_metrics = pd.DataFrame(data, columns=columns)
                        
                        # Merge with daily metrics
                        if not match_metrics.empty and 'date' in match_metrics.columns:
                            # Convert date columns to same type
                            metrics['daily_metrics']['date'] = pd.to_datetime(metrics['daily_metrics']['date'])
                            match_metrics['date'] = pd.to_datetime(match_metrics['date'])
                            
                            # Merge on date
                            metrics['daily_metrics'] = pd.merge(
                                metrics['daily_metrics'],
                                match_metrics,
                                on='date',
                                how='left'
                            ).fillna(0)
                            
                            logger.info(f"Enhanced daily metrics with {len(data)} rows of expert match data")
                except Exception as e:
                    logger.error(f"Error querying expert_search_matches: {e}")
        
        # Get domain performance metrics if expert_search_performance exists
        if expert_search_performance_exists:
            try:
                cursor.execute("""
                    SELECT *
                    FROM expert_search_performance
                    ORDER BY total_matches DESC
                    LIMIT 10
                """)
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['expert_match_metrics'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} rows of expert performance metrics")
            except Exception as e:
                logger.error(f"Error querying expert_search_performance: {e}")
        
        # Get popular queries if search_analytics has query column
        if search_analytics_exists and 'query' in get_table_columns(conn, 'search_analytics'):
            try:
                cursor.execute("""
                    SELECT 
                        query,
                        COUNT(*) as query_count,
                        AVG(response_time) as avg_response_time,
                        AVG(result_count) as avg_results,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM search_analytics
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY query
                    HAVING COUNT(*) > 1
                    ORDER BY query_count DESC
                    LIMIT 20
                """, (start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if data:
                    metrics['query_metrics'] = pd.DataFrame(data, columns=columns)
                    logger.info(f"Retrieved {len(data)} popular queries")
            except Exception as e:
                logger.error(f"Error querying search_analytics for query metrics: {e}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error in get_search_metrics: {e}")
        return metrics
    finally:
        cursor.close()

def display_search_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict] = None):
    """
    Display comprehensive search analytics with enhanced visualizations.
    
    Args:
        metrics: Dictionary of metrics dataframes
        filters: Optional display filters
    """
    st.title("Search Analytics Dashboard")
    st.markdown("This dashboard provides insights into search performance, user behavior, and expert matching.")
    
    # Check if we have data to display
    daily_metrics = metrics['daily_metrics']
    if daily_metrics.empty:
        st.warning("No search data available for the selected period")
        return
    
    # Ensure date column is datetime
    if 'date' in daily_metrics.columns and not pd.api.types.is_datetime64_any_dtype(daily_metrics['date']):
        daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
    
    # Create KPI cards at the top
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Searches KPI
    with col1:
        total_searches = int(daily_metrics['total_searches'].sum()) if 'total_searches' in daily_metrics.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Total Searches</h4>
                <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_searches:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Search Queries</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Unique Users KPI
    with col2:
        unique_users = int(daily_metrics['unique_users'].sum()) if 'unique_users' in daily_metrics.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Unique Users</h4>
                <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{unique_users:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Distinct Users</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Success Rate KPI
    with col3:
        success_rate = daily_metrics['success_rate'].mean() if 'success_rate' in daily_metrics.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Success Rate</h4>
                <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{success_rate:.1f}%</h2>
                <p style="margin:0;color:#666;font-size:14px;">Search Success Rate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Response Time KPI
    with col4:
        avg_response = daily_metrics['avg_response_time'].mean() if 'avg_response_time' in daily_metrics.columns else 0
        
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Avg Response</h4>
                <h2 style="margin:0;padding:10px 0;color:#d62728;font-size:28px;">{avg_response:.2f}s</h2>
                <p style="margin:0;color:#666;font-size:14px;">Response Time</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Search Activity", "Performance Metrics", "Expert Matching"])
    
    with tab1:
        st.header("Search Activity Trends")
        
        # Create search volume chart
        fig_volume = go.Figure()
        
        # Add total searches line
        if 'total_searches' in daily_metrics.columns:
            fig_volume.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['total_searches'],
                    mode='lines+markers',
                    name='Total Searches',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                )
            )
        
        # Add unique users line
        if 'unique_users' in daily_metrics.columns:
            fig_volume.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['unique_users'],
                    mode='lines+markers',
                    name='Unique Users',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                )
            )
        
        # Update layout
        fig_volume.update_layout(
            title="Daily Search Volume",
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Count"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Display search volume chart
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Check if we have search type distribution
        has_search_types = all(col in daily_metrics.columns for col in ['expert_searches', 'publication_searches', 'other_searches'])
        
        if has_search_types:
            st.subheader("Search Type Distribution")
            
            # Aggregate search types
            total_expert = daily_metrics['expert_searches'].sum()
            total_publication = daily_metrics['publication_searches'].sum()
            total_other = daily_metrics['other_searches'].sum()
            
            search_types = pd.DataFrame({
                'Search Type': ['Expert', 'Publication', 'Other'],
                'Count': [total_expert, total_publication, total_other]
            })
            
            # Create columns for pie and bar charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Create pie chart
                pie_fig = px.pie(
                    search_types,
                    values='Count',
                    names='Search Type',
                    title='Search Type Distribution',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
                    hole=0.4
                )
                
                pie_fig.update_traces(textinfo='percent+label')
                pie_fig.update_layout(
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                
                st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                # Create bar chart
                bar_fig = px.bar(
                    search_types,
                    x='Search Type',
                    y='Count',
                    title='Search Type Counts',
                    color='Search Type',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
                
                bar_fig.update_layout(
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0),
                    xaxis_title="",
                    yaxis_title="Number of Searches"
                )
                
                st.plotly_chart(bar_fig, use_container_width=True)
        
        # Display popular queries if available
        query_metrics = metrics.get('query_metrics', pd.DataFrame())
        if not query_metrics.empty:
            st.subheader("Popular Search Queries")
            
            # Create horizontal bar chart for popular queries
            query_fig = px.bar(
                query_metrics.head(10),
                y='query',
                x='query_count',
                title='Top 10 Search Queries',
                color='avg_results',
                color_continuous_scale='RdYlGn',
                labels={'query_count': 'Number of Searches', 'query': 'Search Query', 'avg_results': 'Avg Results'},
                orientation='h'
            )
            
            query_fig.update_layout(
                title_font=dict(size=18),
                margin=dict(t=50, b=0, l=0, r=0),
                height=500,
                yaxis=dict(title="", autorange="reversed")
            )
            
            st.plotly_chart(query_fig, use_container_width=True)
            
            # Show detailed query data in expander
            with st.expander("View Detailed Query Metrics"):
                st.dataframe(query_metrics, use_container_width=True)
                
                # Add download button
                csv = query_metrics.to_csv(index=False)
                st.download_button(
                    label="Download Query Metrics CSV",
                    data=csv,
                    file_name="query_metrics.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.header("Performance Metrics")
        
        # Create columns for performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Response Time Chart
            if 'avg_response_time' in daily_metrics.columns:
                response_fig = go.Figure()
                
                # Add response time line
                response_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=daily_metrics['avg_response_time'],
                        mode='lines+markers',
                        name='Avg Response Time',
                        line=dict(color='#d62728', width=3),
                        marker=dict(size=8)
                    )
                )
                
                # Add filled area
                response_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=daily_metrics['avg_response_time'],
                        fill='tozeroy',
                        fillcolor='rgba(214,39,40,0.2)',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                # Update layout
                response_fig.update_layout(
                    title="Response Time Trend",
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Response Time (seconds)"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(response_fig, use_container_width=True)
        
        with col2:
            # Success Rate Chart
            if 'success_rate' in daily_metrics.columns:
                success_fig = go.Figure()
                
                # Add success rate line
                success_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=daily_metrics['success_rate'],
                        mode='lines+markers',
                        name='Success Rate',
                        line=dict(color='#2ca02c', width=3),
                        marker=dict(size=8)
                    )
                )
                
                # Add filled area
                success_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=daily_metrics['success_rate'],
                        fill='tozeroy',
                        fillcolor='rgba(44,160,44,0.2)',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                # Add reference line at 50%
                success_fig.add_shape(
                    type="line",
                    x0=daily_metrics['date'].min(),
                    x1=daily_metrics['date'].max(),
                    y0=50,
                    y1=50,
                    line=dict(
                        color="rgba(255, 0, 0, 0.5)",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Update layout
                success_fig.update_layout(
                    title="Search Success Rate Trend",
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Success Rate (%)"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(success_fig, use_container_width=True)
        
        # Session Performance Metrics
        has_session_data = all(col in daily_metrics.columns for col in ['avg_queries_per_session', 'session_success_rate'])
        has_duration_data = 'avg_session_duration' in daily_metrics.columns
        
        if has_session_data:
            st.subheader("Session Performance")
            
            # Create session metrics chart
            session_fig = go.Figure()
            
            # Add queries per session line
            session_fig.add_trace(
                go.Bar(
                    x=daily_metrics['date'],
                    y=daily_metrics['avg_queries_per_session'],
                    name='Queries per Session',
                    marker_color='#9467bd'
                )
            )
            
            # Add session success rate line on secondary y-axis
            session_fig.add_trace(
                go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['session_success_rate'],
                    mode='lines+markers',
                    name='Session Success Rate (%)',
                    line=dict(color='#8c564b', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                )
            )
            
            # Add session duration if available
            if has_duration_data:
                # Convert seconds to minutes for better readability
                duration_minutes = daily_metrics['avg_session_duration'] / 60
                
                session_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=duration_minutes,
                        mode='lines+markers',
                        name='Avg Session Duration (min)',
                        line=dict(color='#e377c2', width=3),
                        marker=dict(size=8),
                        yaxis='y3'
                    )
                )
            
            # Update layout with multiple y-axes
            session_fig.update_layout(
                title="Session Performance Metrics",
                height=450,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Queries per Session"),
                yaxis2=dict(
                    title="Success Rate (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Add third y-axis for duration if needed
            if has_duration_data:
                session_fig.update_layout(
                    yaxis3=dict(
                        title="Duration (min)",
                        overlaying="y",
                        side="right",
                        anchor="free",
                        position=0.95
                    )
                )
            
            st.plotly_chart(session_fig, use_container_width=True)
    
    with tab3:
        st.header("Expert Matching Analysis")
        
        # Check if we have expert matching data
        has_expert_data = all(col in daily_metrics.columns for col in ['matched_experts', 'avg_similarity'])
        has_rank_data = 'avg_rank' in daily_metrics.columns
        has_click_data = 'click_rate' in daily_metrics.columns
        
        if has_expert_data:
            # Create columns for expert matching charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Expert Matches Chart
                match_fig = go.Figure()
                
                # Add matched experts bars
                match_fig.add_trace(
                    go.Bar(
                        x=daily_metrics['date'],
                        y=daily_metrics['matched_experts'],
                        name='Matched Experts',
                        marker_color='#1f77b4'
                    )
                )
                
                # Add similarity score line on secondary y-axis
                match_fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['date'],
                        y=daily_metrics['avg_similarity'],
                        mode='lines+markers',
                        name='Avg Similarity Score',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8),
                        yaxis='y2'
                    )
                )
                
                # Update layout
                match_fig.update_layout(
                    title="Expert Matching Volume & Quality",
                    height=350,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Matched Experts"),
                    yaxis2=dict(
                        title="Similarity Score",
                        overlaying="y",
                        side="right",
                        range=[0, 1]
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(match_fig, use_container_width=True)
            
            with col2:
                # Create second expert chart if we have rank or click data
                if has_rank_data or has_click_data:
                    rank_fig = go.Figure()
                    
                    # Add rank data if available
                    if has_rank_data:
                        rank_fig.add_trace(
                            go.Scatter(
                                x=daily_metrics['date'],
                                y=daily_metrics['avg_rank'],
                                mode='lines+markers',
                                name='Avg Rank Position',
                                line=dict(color='#2ca02c', width=3),
                                marker=dict(size=8)
                            )
                        )
                    
                    # Add click rate if available
                    if has_click_data:
                        rank_fig.add_trace(
                            go.Scatter(
                                x=daily_metrics['date'],
                                y=daily_metrics['click_rate'],
                                mode='lines+markers',
                                name='Click-through Rate (%)',
                                line=dict(color='#d62728', width=3),
                                marker=dict(size=8),
                                yaxis='y2'
                            )
                        )
                    
                    # Update layout
                    rank_fig.update_layout(
                        title="Expert Match Performance",
                        height=350,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        xaxis=dict(title="Date"),
                        yaxis=dict(title="Rank Position"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Add second y-axis if we have click data
                    if has_click_data:
                        rank_fig.update_layout(
                            yaxis2=dict(
                                title="Click Rate (%)",
                                overlaying="y",
                                side="right",
                                range=[0, 100]
                            )
                        )
                    
                    st.plotly_chart(rank_fig, use_container_width=True)
        
        # Display expert performance metrics if available
        expert_metrics = metrics.get('expert_match_metrics', pd.DataFrame())
        if not expert_metrics.empty:
            st.subheader("Expert Performance Overview")
            
            # Check what columns we have
            expert_cols = expert_metrics.columns.tolist()
            
            # Create expert performance visualization
            if 'expert_id' in expert_cols and 'total_matches' in expert_cols:
                # Create horizontal bar chart for top experts
                expert_fig = px.bar(
                    expert_metrics.head(10),
                    y='expert_id',
                    x='total_matches',
                    title='Top 10 Matched Experts',
                    color='avg_similarity' if 'avg_similarity' in expert_cols else None,
                    color_continuous_scale='RdYlGn',
                    labels={'total_matches': 'Number of Matches', 'expert_id': 'Expert ID'},
                    orientation='h'
                )
                
                expert_fig.update_layout(
                    title_font=dict(size=18),
                    margin=dict(t=50, b=0, l=0, r=0),
                    height=500,
                    yaxis=dict(title="", autorange="reversed")
                )
                
                st.plotly_chart(expert_fig, use_container_width=True)
                
                # Show detailed expert data in expander
                with st.expander("View Detailed Expert Performance Metrics"):
                    st.dataframe(expert_metrics, use_container_width=True)
                    
                    # Add download button
                    csv = expert_metrics.to_csv(index=False)
                    st.download_button(
                        label="Download Expert Performance Metrics CSV",
                        data=csv,
                        file_name="expert_performance_metrics.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No expert matching data available for the selected period")
    
    # Add detailed metrics tables in an expander
    with st.expander("View All Search Metrics"):
        st.dataframe(daily_metrics, use_container_width=True)
        
        # Add download button
        csv = daily_metrics.to_csv(index=False)
        st.download_button(
            label="Download Search Metrics CSV",
            data=csv,
            file_name="search_metrics.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(page_title="Search Analytics Dashboard", page_icon="🔍", layout="wide")
    
    # Dashboard title and description
    st.title("Search Analytics Dashboard")
    st.markdown("This dashboard provides insights into search performance, user engagement, and expert matching metrics.")
    
    # Dashboard filters in sidebar
    st.sidebar.title("Dashboard Filters")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date.date(), end_date.date()),
        max_value=end_date.date()
    )
    
    # Add granularity filter
    granularity = st.sidebar.selectbox(
        "Data Granularity",
        ["Daily", "Weekly", "Monthly"],
        index=0
    )
    
    # Add search type filter if needed
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["All", "Expert", "Publication", "General"],
        index=0
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Format for database query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Display loading spinner
        with st.spinner("Loading search metrics..."):
            # Get metrics with selected filters
            with get_db_connection() as conn:
                # Get metrics
                metrics = get_search_metrics(conn, start_date_str, end_date_str)
                
                # Apply granularity transformation if needed
                daily_metrics = metrics['daily_metrics']
                if not daily_metrics.empty and 'date' in daily_metrics.columns:
                    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
                    
                    if granularity == "Weekly":
                        # Group by week
                        daily_metrics = daily_metrics.set_index('date')
                        
                        # Identify count vs. average columns
                        count_cols = [col for col in daily_metrics.columns if col.startswith('total_') or col.endswith('_count') or col == 'unique_users' or col == 'matched_experts']
                        avg_cols = [col for col in daily_metrics.columns if col not in count_cols]
                        
                        # Create weekly aggregations
                        weekly_counts = daily_metrics[count_cols].resample('W').sum() if count_cols else pd.DataFrame()
                        weekly_avgs = daily_metrics[avg_cols].resample('W').mean() if avg_cols else pd.DataFrame()
                        
                        # Combine the aggregations
                        if not weekly_counts.empty and not weekly_avgs.empty:
                            daily_metrics = pd.concat([weekly_counts, weekly_avgs], axis=1)
                        elif not weekly_counts.empty:
                            daily_metrics = weekly_counts
                        elif not weekly_avgs.empty:
                            daily_metrics = weekly_avgs
                            
                        daily_metrics = daily_metrics.reset_index()
                    
                    elif granularity == "Monthly":
                        # Group by month
                        daily_metrics = daily_metrics.set_index('date')
                        
                        # Identify count vs. average columns
                        count_cols = [col for col in daily_metrics.columns if col.startswith('total_') or col.endswith('_count') or col == 'unique_users' or col == 'matched_experts']
                        avg_cols = [col for col in daily_metrics.columns if col not in count_cols]
                        
                        # Create monthly aggregations
                        monthly_counts = daily_metrics[count_cols].resample('M').sum() if count_cols else pd.DataFrame()
                        monthly_avgs = daily_metrics[avg_cols].resample('M').mean() if avg_cols else pd.DataFrame()
                        
                        # Combine the aggregations
                        if not monthly_counts.empty and not monthly_avgs.empty:
                            daily_metrics = pd.concat([monthly_counts, monthly_avgs], axis=1)
                        elif not monthly_counts.empty:
                            daily_metrics = monthly_counts
                        elif not monthly_avgs.empty:
                            daily_metrics = monthly_avgs
                            
                        daily_metrics = daily_metrics.reset_index()
                
                # Update metrics with transformed data
                metrics['daily_metrics'] = daily_metrics
                
                # Display dashboard
                display_search_analytics(metrics, {
                    'start_date': start_date,
                    'end_date': end_date,
                    'granularity': granularity,
                    'search_type': search_type
                })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()
