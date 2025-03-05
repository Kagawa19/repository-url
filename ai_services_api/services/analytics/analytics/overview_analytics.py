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

def get_overview_metrics(conn, start_date, end_date):
    """Get overview metrics with dynamic table detection and query building"""
    cursor = conn.cursor()
    try:
        # Check which tables exist
        chat_sessions_exists = check_table_exists(conn, 'chat_sessions')
        chatbot_logs_exists = check_table_exists(conn, 'chatbot_logs')
        response_quality_exists = check_table_exists(conn, 'response_quality_metrics')
        expert_search_matches_exists = check_table_exists(conn, 'expert_search_matches')
        expert_messages_exists = check_table_exists(conn, 'expert_messages')
        search_sessions_exists = check_table_exists(conn, 'search_sessions')
        search_analytics_exists = check_table_exists(conn, 'search_analytics')
        
        # Get columns for tables if they exist
        chatbot_logs_columns = []
        if chatbot_logs_exists:
            chatbot_logs_columns = get_table_columns(conn, 'chatbot_logs')
        
        # Build query components based on available tables
        cte_parts = []
        aliases = []
        join_tables = []
        select_columns = ["COALESCE({}) as date".format(
            ", ".join([f"{alias}.date" for alias in ['ChatMetrics', 'QualityMetrics', 'ExpertMetrics', 'MessageMetrics', 'SearchMetrics'] 
                      if (alias == 'ChatMetrics' and chatbot_logs_exists) or
                      (alias == 'QualityMetrics' and chatbot_logs_exists and response_quality_exists) or
                      (alias == 'ExpertMetrics' and expert_search_matches_exists) or
                      (alias == 'MessageMetrics' and expert_messages_exists) or
                      (alias == 'SearchMetrics' and search_analytics_exists)])
        )]
        
        # ChatMetrics CTE (using chatbot_logs instead of interactions)
        if chatbot_logs_exists:
            response_time_field = 'response_time' in chatbot_logs_columns
            
            # Build the CTE based on available columns
            chat_cte = """
                ChatMetrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT user_id) as unique_users,
            """
            
            # Add response_time calculation based on available columns
            if response_time_field:
                chat_cte += "AVG(response_time) as avg_response_time,"
            else:
                chat_cte += "0.0 as avg_response_time,"
            
            # Add error_rate placeholder - we don't have error_occurred in chatbot_logs
            # but we can add it as a placeholder
            chat_cte += """
                    0.0 as error_rate
                FROM chatbot_logs
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
            )
            """
            
            cte_parts.append(chat_cte)
            aliases.append('ChatMetrics')
            join_tables.append('ChatMetrics')
            
            # Add chat metrics columns to select
            select_columns.extend([
                "ChatMetrics.total_interactions",
                "ChatMetrics.unique_users",
                "ChatMetrics.avg_response_time",
                "ChatMetrics.error_rate"
            ])
        else:
            # If chatbot_logs table doesn't exist, add default values
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
        
        # ExpertMetrics CTE (using expert_search_matches instead of expert_matching_logs)
        if expert_search_matches_exists:
            expert_cte = """
                ExpertMetrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as expert_matches,
                        AVG(similarity_score) as avg_similarity,
                        COUNT(CASE WHEN clicked THEN 1 END)::FLOAT / 
                            NULLIF(COUNT(*), 0) * 100 as success_rate
                    FROM expert_search_matches
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
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
            # If expert matches table doesn't exist, add default values
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

        # SearchMetrics CTE
        if search_analytics_exists:
            search_cte = """
                SearchMetrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_searches,
                        COUNT(DISTINCT user_id) as unique_searchers,
                        AVG(response_time) as avg_search_time,
                        SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END)::FLOAT / 
                            NULLIF(COUNT(*), 0) * 100 as search_success_rate
                    FROM search_analytics
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
                )
            """
            
            cte_parts.append(search_cte)
            aliases.append('SearchMetrics')
            join_tables.append('SearchMetrics')
            
            # Add search metrics to select
            select_columns.extend([
                "COALESCE(SearchMetrics.total_searches, 0) as total_searches",
                "COALESCE(SearchMetrics.unique_searchers, 0) as unique_searchers",
                "COALESCE(SearchMetrics.avg_search_time, 0.0) as avg_search_time",
                "COALESCE(SearchMetrics.search_success_rate, 0.0) as search_success_rate"
            ])
        else:
            # If search analytics table doesn't exist, add default values
            select_columns.extend([
                "0 as total_searches",
                "0 as unique_searchers",
                "0.0 as avg_search_time",
                "0.0 as search_success_rate"
            ])
        
        # Build the full query
        if not cte_parts:
            # If no tables exist, return an empty dataframe with default columns
            logger.warning("No relevant tables found for overview metrics")
            columns = ['date', 'total_interactions', 'unique_users', 'avg_response_time', 
                      'error_rate', 'avg_quality_score', 'avg_helpfulness', 
                      'avg_hallucination_risk', 'avg_factual_grounding',
                      'expert_matches', 'avg_similarity', 'success_rate',
                      'total_messages', 'draft_messages',
                      'total_searches', 'unique_searchers', 'avg_search_time', 'search_success_rate']
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
                  'total_messages', 'draft_messages',
                  'total_searches', 'unique_searchers', 'avg_search_time', 'search_success_rate']
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
    
    # Calculate key metrics (now including search metrics)
    total_interactions = metrics_df_filled['total_interactions'].sum()
    total_messages = metrics_df_filled['total_messages'].sum()
    total_searches = metrics_df_filled.get('total_searches', pd.Series([0])).sum()
    success_rate = metrics_df_filled['success_rate'].mean()
    avg_quality = metrics_df_filled['avg_quality_score'].mean()
    search_success_rate = metrics_df_filled.get('search_success_rate', pd.Series([0])).mean()

    # Display key metrics in a more attractive layout
    st.markdown("### Key Performance Indicators")
    
    # Use a clean card layout for KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Interactions</h4>
                <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_interactions:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Total User Interactions</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Expert Matches</h4>
                <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{success_rate:.1f}%</h2>
                <p style="margin:0;color:#666;font-size:14px;">Match Success Rate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Response Quality</h4>
                <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{avg_quality:.2f}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Average Quality Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Messages</h4>
                <h2 style="margin:0;padding:10px 0;color:#9467bd;font-size:28px;">{total_messages:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Total Expert Messages</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Searches</h4>
                <h2 style="margin:0;padding:10px 0;color:#d62728;font-size:28px;">{total_searches:,}</h2>
                <p style="margin:0;color:#666;font-size:14px;">Total Search Queries</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                <h4 style="margin:0;padding:0;color:#31333F;">Search Success</h4>
                <h2 style="margin:0;padding:10px 0;color:#8c564b;font-size:28px;">{search_success_rate:.1f}%</h2>
                <p style="margin:0;color:#666;font-size:14px;">Search Success Rate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Create improved visualization layout
    st.markdown("### Trend Analysis")
    
    # Activity Trends Tab
    tab1, tab2, tab3 = st.tabs(["User Activity", "Quality Metrics", "Expert & Search Performance"])
    
    with tab1:
        # User Activity Trends (Interactions, Messages, Users)
        user_activity_df = metrics_df.copy()
        
        # Determine which metrics to show based on data availability
        has_interactions = not (metrics_df_filled['total_interactions'] == 0).all()
        has_messages = not (metrics_df_filled['total_messages'] == 0).all()
        has_users = not (metrics_df_filled['unique_users'] == 0).all()
        has_searchers = 'unique_searchers' in metrics_df_filled.columns and not (metrics_df_filled['unique_searchers'] == 0).all()
        
        # Only keep columns with data
        activity_cols = []
        if has_interactions:
            activity_cols.append('total_interactions')
        if has_messages:
            activity_cols.append('total_messages')
        if has_users:
            activity_cols.append('unique_users')
        if has_searchers:
            activity_cols.append('unique_searchers')
        
        if not activity_cols:
            st.info("No activity data available for the selected period")
        else:
            # Create a clean line chart for activity trends
            fig = go.Figure()
            
            colors = {
                'total_interactions': '#1f77b4',
                'total_messages': '#9467bd',
                'unique_users': '#2ca02c',
                'unique_searchers': '#d62728'
            }
            
            names = {
                'total_interactions': 'Total Interactions',
                'total_messages': 'Total Messages',
                'unique_users': 'Unique Users',
                'unique_searchers': 'Unique Searchers'
            }
            
            for col in activity_cols:
                fig.add_trace(go.Scatter(
                    x=user_activity_df['date'],
                    y=user_activity_df[col],
                    mode='lines+markers',
                    name=names.get(col, col),
                    line=dict(color=colors.get(col, None), width=3),
                    marker=dict(size=8)
                ))
            
            # Format the layout for better readability
            fig.update_layout(
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a bar chart comparing weekly or monthly totals
            if len(user_activity_df) > 1:
                st.subheader("Activity Distribution")
                
                # Create a weekly or monthly aggregation based on data length
                activity_dist_df = user_activity_df.copy()
                activity_dist_df = activity_dist_df[['date'] + activity_cols]
                
                if filters.get('granularity') == 'Daily' and len(activity_dist_df) > 7:
                    # Create weekly distribution
                    activity_dist_df['week'] = activity_dist_df['date'].dt.isocalendar().week
                    activity_dist_df['year'] = activity_dist_df['date'].dt.isocalendar().year
                    activity_dist_df['week_label'] = activity_dist_df['date'].dt.strftime('Week %U')
                    
                    # Group by week
                    weekly_data = activity_dist_df.groupby('week_label')[activity_cols].sum().reset_index()
                    
                    # Create a grouped bar chart
                    fig = px.bar(
                        weekly_data.melt(id_vars='week_label', value_vars=activity_cols, var_name='Metric', value_name='Count'),
                        x='week_label',
                        y='Count',
                        color='Metric',
                        color_discrete_map={
                            'total_interactions': '#1f77b4',
                            'total_messages': '#9467bd',
                            'unique_users': '#2ca02c',
                            'unique_searchers': '#d62728'
                        },
                        barmode='group',
                        labels={'week_label': 'Week', 'Count': 'Total Count'},
                        category_orders={"week_label": sorted(weekly_data['week_label'].unique())}
                    )
                    
                    fig.update_layout(
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Quality Metrics Tab (Response Quality, Helpfulness, etc.)
        has_quality = not (metrics_df_filled['avg_helpfulness'] == 0).all()
        has_hallucination = 'avg_hallucination_risk' in metrics_df_filled.columns and not (metrics_df_filled['avg_hallucination_risk'] == 0).all()
        has_factual = 'avg_factual_grounding' in metrics_df_filled.columns and not (metrics_df_filled['avg_factual_grounding'] == 0).all()
        
        if not has_quality and not has_hallucination and not has_factual:
            st.info("No quality metrics available for the selected period")
        else:
            # Create a clean line chart for quality metrics
            fig = go.Figure()
            
            if has_quality:
                fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_helpfulness'],
                    mode='lines',
                    name='Helpfulness Score',
                    line=dict(color='#2ca02c', width=3)
                ))
            
            if has_hallucination:
                fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_hallucination_risk'],
                    mode='lines',
                    name='Hallucination Risk',
                    line=dict(color='#d62728', width=3)
                ))
            
            if has_factual:
                fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_factual_grounding'],
                    mode='lines',
                    name='Factual Grounding',
                    line=dict(color='#1f77b4', width=3)
                ))
            
            # Format the layout for better readability
            fig.update_layout(
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
                    title="Score",
                    gridcolor='rgba(0,0,0,0.1)',
                    tickformat='.2f'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a quality score distribution chart
            if has_quality:
                st.subheader("Quality Score Distribution")
                
                # Create bins for quality scores
                metrics_df['quality_bin'] = pd.cut(
                    metrics_df['avg_helpfulness'],
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
                )
                
                quality_dist = metrics_df.groupby('quality_bin').size().reset_index(name='count')
                
                # Create a pie chart for quality distribution
                fig = px.pie(
                    quality_dist,
                    values='count',
                    names='quality_bin',
                    color='quality_bin',
                    color_discrete_sequence=px.colors.sequential.Greens,
                    hole=0.4,
                    labels={'quality_bin': 'Quality Score Range'}
                )
                
                fig.update_layout(
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig.update_traces(textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Expert & Search Performance Tab
        col1, col2 = st.columns(2)
        
        with col1:
            # Expert matching visualization
            has_expert_matching = not (metrics_df_filled['expert_matches'] == 0).all()
            has_success_rate = not (metrics_df_filled['success_rate'] == 0).all()
            
            if not has_expert_matching:
                st.info("No expert matching data available")
            else:
                # Create a combined chart for expert matches with success rate as a line
                fig = go.Figure()
                
                # Add expert matches as bars
                fig.add_trace(go.Bar(
                    x=metrics_df['date'],
                    y=metrics_df['expert_matches'],
                    name='Expert Matches',
                    marker_color='#3366cc'
                ))
                
                # Add success rate as a line on secondary y-axis if available
                if has_success_rate:
                    fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df['success_rate'],
                        name='Success Rate (%)',
                        mode='lines+markers',
                        marker=dict(color='#109618', size=8),
                        line=dict(color='#109618', width=3),
                        yaxis='y2'
                    ))
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title_text='Expert Matching Performance',
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
                    yaxis=dict(
                        title="Expert Matches",
                        gridcolor='rgba(0,0,0,0.1)',
                        side='left'
                    ),
                    yaxis2=dict(
                        title="Success Rate (%)",
                        overlaying='y',
                        side='right',
                        range=[0, 100],
                        tickformat='.1f'
                    ),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d"
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Search performance visualization
            has_searches = 'total_searches' in metrics_df_filled.columns and not (metrics_df_filled['total_searches'] == 0).all()
            has_search_success = 'search_success_rate' in metrics_df_filled.columns and not (metrics_df_filled['search_success_rate'] == 0).all()
            
            if not has_searches:
                st.info("No search performance data available")
            else:
                # Create a combined chart for searches with success rate as a line
                fig = go.Figure()
                
                # Add search count as bars
                fig.add_trace(go.Bar(
                    x=metrics_df['date'],
                    y=metrics_df['total_searches'],
                    name='Total Searches',
                    marker_color='#dc3912'
                ))
                
                # Add search success rate as a line on secondary y-axis if available
                if has_search_success:
                    fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df['search_success_rate'],
                        name='Success Rate (%)',
                        mode='lines+markers',
                        marker=dict(color='#ff9900', size=8),
                        line=dict(color='#ff9900', width=3),
                        yaxis='y2'
                    ))
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title_text='Search Performance',
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
                    yaxis=dict(
                        title="Search Count",
                        gridcolor='rgba(0,0,0,0.1)',
                        side='left'
                    ),
                    yaxis2=dict(
                        title="Success Rate (%)",
                        overlaying='y',
                        side='right',
                        range=[0, 100],
                        tickformat='.1f'
                    ),
                    xaxis=dict(
                        title="Date",
                        tickangle=-45,
                        tickformat="%b %d"
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics row
        has_response_time = not (metrics_df_filled['avg_response_time'] == 0).all()
        has_search_time = 'avg_search_time' in metrics_df_filled.columns and not (metrics_df_filled['avg_search_time'] == 0).all()
        
        if has_response_time or has_search_time:
            st.subheader("Response Time Performance")
            
            # Create a performance line chart
            fig = go.Figure()
            
            if has_response_time:
                fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_response_time'],
                    mode='lines+markers',
                    name='Chat Response Time',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ))
            
            if has_search_time:
                fig.add_trace(go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['avg_search_time'],
                    mode='lines+markers',
                    name='Search Response Time',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=8)
                ))
            
            # Format the layout for better readability
            fig.update_layout(
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
                    gridcolor='rgba(0,0,0,0.1)',
                    tickformat='.2f'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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
            tab1, tab2, tab3 = st.tabs(["Interaction Metrics", "Expert & Quality Metrics", "Search Metrics"])
            
            with tab1:
                interaction_cols = ['date', 'total_interactions', 'unique_users', 
                                  'avg_response_time', 'total_messages', 'draft_messages']
                interaction_df = display_df[[col for col in interaction_cols if col in display_df.columns]]
                
                if not interaction_df.empty:
                    style_columns = [col for col in ['total_interactions', 'avg_response_time'] 
                                    if col in interaction_df.columns]
                    
                    if style_columns:
                        st.dataframe(interaction_df.style.background_gradient(
                            subset=style_columns,
                            cmap='RdYlGn'
                        ), use_container_width=True)
                    else:
                        st.dataframe(interaction_df, use_container_width=True)
                    
                    # Add download button for CSV
                    csv = interaction_df.to_csv(index=False)
                    st.download_button(
                        label="Download Interaction Data as CSV",
                        data=csv,
                        file_name="interaction_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No interaction metrics available")
            
            with tab2:
                expert_cols = ['date', 'expert_matches', 'avg_similarity', 'success_rate',
                             'avg_quality_score', 'avg_helpfulness', 'avg_hallucination_risk', 
                             'avg_factual_grounding']
                expert_df = display_df[[col for col in expert_cols if col in display_df.columns]]
                
                if not expert_df.empty:
                    style_columns = [col for col in ['success_rate', 'avg_quality_score', 'avg_helpfulness'] 
                                    if col in expert_df.columns]
                    
                    if style_columns:
                        st.dataframe(expert_df.style.background_gradient(
                            subset=style_columns,
                            cmap='RdYlGn'
                        ), use_container_width=True)
                    else:
                        st.dataframe(expert_df, use_container_width=True)
                    
                    # Add download button for CSV
                    csv = expert_df.to_csv(index=False)
                    st.download_button(
                        label="Download Expert & Quality Data as CSV",
                        data=csv,
                        file_name="expert_quality_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No expert or quality metrics available")
            
            with tab3:
                search_cols = ['date', 'total_searches', 'unique_searchers', 
                              'avg_search_time', 'search_success_rate']
                search_df = display_df[[col for col in search_cols if col in display_df.columns]]
                
                if not search_df.empty and not (search_df['total_searches'] == 0).all():
                    style_columns = [col for col in ['total_searches', 'search_success_rate'] 
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
                        label="Download Search Data as CSV",
                        data=csv,
                        file_name="search_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No search metrics available")
        else:
            st.info("No detailed metrics available")

    # Add date/time of last update with better styling
    st.markdown(
        f"""
        <div style="text-align:right;color:#888;font-size:0.8em;padding-top:20px;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Overview Analytics Dashboard", page_icon="📊", layout="wide")
    
    # Dashboard title and description
    st.title("Overview Analytics Dashboard")
    st.markdown("""
    This dashboard provides an overview of system performance, user engagement, and content quality metrics.
    Use the date filter to explore metrics for specific time periods.
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
        - Chat interactions & sessions
        - Expert matching activity
        - Response quality metrics
        - Search analytics
        """)
    
    if len(date_range) == l:
        start_date, end_date = date_range
        
        # Format for database query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get metrics with selected filters
        with get_db_connection() as conn:
            # Get metrics
            metrics_df = get_overview_metrics(conn, start_date_str, end_date_str)
            
            # If weekly or monthly granularity is selected, resample the data
            if granularity == "Weekly" and not metrics_df.empty and 'date' in metrics_df.columns:
                metrics_df = metrics_df.set_index('date')
                # Resample to weekly frequency, taking mean for rates and sum for counts
                numeric_cols = metrics_df.select_dtypes(include=['number']).columns
                count_cols = [col for col in numeric_cols if 'total' in col or 'count' in col or 'matches' in col or 'messages' in col or 'users' in col or 'searches' in col]
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
                count_cols = [col for col in numeric_cols if 'total' in col or 'count' in col or 'matches' in col or 'messages' in col or 'users' in col or 'searches' in col]
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
            display_overview_analytics(metrics_df, {
                'start_date': start_date,
                'end_date': end_date,
                'granularity': granularity
            })
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()