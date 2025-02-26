import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('usage_analytics')

def get_usage_metrics(conn, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Get comprehensive usage metrics from all relevant tables"""
    cursor = conn.cursor()
    try:
        # Get metrics from multiple sources
        cursor.execute("""
            WITH GeneralUsage AS (
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(CAST(metrics->>'response_time' AS FLOAT)) as avg_response_time,
                    COUNT(CASE WHEN CAST(metrics->>'error_occurred' AS BOOLEAN) THEN 1 END) as error_count,
                    COUNT(CASE WHEN CAST(metrics->>'error_occurred' AS BOOLEAN) THEN 1 END)::float / 
                        NULLIF(COUNT(*), 0) * 100 as error_rate,
                    -- Changed from sentiment_score to a fixed query that doesn't reference i.timestamp
                    COALESCE(
                        AVG(sentiment_score),
                        0.0
                    ) as avg_quality_score
                FROM interactions i
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            ),
            QualityMetrics AS (
                SELECT 
                    DATE(cl.timestamp) as date,
                    AVG(rqm.helpfulness_score) as avg_helpfulness,
                    AVG(rqm.hallucination_risk) as avg_hallucination_risk,
                    AVG(rqm.factual_grounding_score) as avg_factual_grounding,
                    COUNT(*) as quality_evaluations,
                    COUNT(CASE WHEN rqm.helpfulness_score >= 0.7 THEN 1 END)::float / 
                        NULLIF(COUNT(*), 0) * 100 as high_quality_ratio
                FROM chatbot_logs cl
                JOIN response_quality_metrics rqm ON cl.id = rqm.interaction_id
                WHERE cl.timestamp BETWEEN %s AND %s
                GROUP BY DATE(cl.timestamp)
                ORDER BY date
            ),
            -- Use a join to add quality metrics to general usage instead of a correlated subquery
            CombinedMetrics AS (
                SELECT 
                    g.date,
                    g.total_interactions,
                    g.unique_users,
                    g.avg_response_time,
                    g.error_count,
                    g.error_rate,
                    COALESCE(q.avg_helpfulness, g.avg_quality_score) as avg_quality_score,
                    q.avg_helpfulness,
                    q.avg_hallucination_risk,
                    q.avg_factual_grounding,
                    q.quality_evaluations,
                    q.high_quality_ratio
                FROM GeneralUsage g
                LEFT JOIN QualityMetrics q ON g.date = q.date
            ),
            ChatMetrics AS (
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as chat_count,
                    COUNT(DISTINCT user_id) as chat_users,
                    AVG(response_time) as chat_response_time,
                    SUM(navigation_matches) as total_nav_matches,
                    SUM(publication_matches) as total_pub_matches
                FROM chat_interactions
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
            ),
            SearchMetrics AS (
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as search_count,
                    COUNT(DISTINCT user_id) as search_users,
                    AVG(response_time) as search_response_time,
                    SUM(result_count) as total_results
                FROM search_analytics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
            ),
            HourlyPattern AS (
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as hourly_count
                FROM interactions
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            ),
            ContentMatching AS (
                SELECT 
                    content_type,
                    COUNT(*) as match_count,
                    AVG(similarity_score) as avg_similarity
                FROM chat_analytics
                WHERE interaction_id IN (
                    SELECT id FROM chat_interactions 
                    WHERE timestamp BETWEEN %s AND %s
                )
                GROUP BY content_type
            )
            SELECT json_build_object(
                'general_usage', COALESCE((SELECT json_agg(row_to_json(CombinedMetrics)) FROM CombinedMetrics), '[]'::json),
                'chat_metrics', COALESCE((SELECT json_agg(row_to_json(ChatMetrics)) FROM ChatMetrics), '[]'::json),
                'search_metrics', COALESCE((SELECT json_agg(row_to_json(SearchMetrics)) FROM SearchMetrics), '[]'::json),
                'hourly_pattern', COALESCE((SELECT json_agg(row_to_json(HourlyPattern)) FROM HourlyPattern), '[]'::json),
                'content_matching', COALESCE((SELECT json_agg(row_to_json(ContentMatching)) FROM ContentMatching), '[]'::json),
                'quality_metrics', COALESCE((SELECT json_agg(row_to_json(QualityMetrics)) FROM QualityMetrics), '[]'::json)
            ) as metrics;
        """, (
            start_date, end_date,  # For GeneralUsage
            start_date, end_date,  # For QualityMetrics
            start_date, end_date,  # For ChatMetrics
            start_date, end_date,  # For SearchMetrics
            start_date, end_date,  # For HourlyPattern
            start_date, end_date   # For ContentMatching
        ))
        
        result = cursor.fetchone()[0]
        
        metrics = {}
        for key in ['general_usage', 'chat_metrics', 'search_metrics', 'hourly_pattern', 'content_matching', 'quality_metrics']:
            try:
                df = pd.DataFrame(result.get(key, []))
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                metrics[key] = df
            except Exception as e:
                logger.error(f"Error processing {key}: {str(e)}")
                metrics[key] = pd.DataFrame()
        
        return metrics
        
    finally:
        cursor.close()

def display_usage_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict[str, Any]] = None):
    """Display comprehensive usage analytics dashboard"""
    st.title("Usage Analytics Dashboard")
    
    try:
        general_data = metrics.get('general_usage', pd.DataFrame())
        chat_data = metrics.get('chat_metrics', pd.DataFrame())
        search_data = metrics.get('search_metrics', pd.DataFrame())
        hourly_data = metrics.get('hourly_pattern', pd.DataFrame())
        content_data = metrics.get('content_matching', pd.DataFrame())
        quality_data = metrics.get('quality_metrics', pd.DataFrame())
        
        if general_data.empty:
            st.warning("No usage data available for the selected period")
            return
            
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_interactions = general_data['total_interactions'].sum()
            st.metric("Total Interactions", f"{total_interactions:,}")
            
        with col2:
            total_users = general_data['unique_users'].sum()
            st.metric("Total Users", f"{total_users:,}")
            
        with col3:
            avg_response = general_data['avg_response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
            
        with col4:
            # Changed from avg_sentiment to avg_quality_score
            avg_quality = general_data['avg_quality_score'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.2f}")

        # Usage Trends
        st.subheader("Usage Analytics")
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Daily Activity",
                "Chat vs Search Usage",
                "Response Time Trends",
                "Error Rate Trend",
                "Hourly Usage Pattern",
                "Response Quality Metrics"  # Changed from Content Type Distribution
            ),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Daily Activity
        fig.add_trace(
            go.Scatter(
                x=general_data['date'],
                y=general_data['total_interactions'],
                name='Total Interactions',
                mode='lines+markers'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=general_data['date'],
                y=general_data['unique_users'],
                name='Unique Users',
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        # Chat vs Search Usage
        if not chat_data.empty and not search_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=chat_data['date'],
                    y=chat_data['chat_count'],
                    name='Chat Interactions',
                    mode='lines'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=search_data['date'],
                    y=search_data['search_count'],
                    name='Search Queries',
                    mode='lines'
                ),
                row=1, col=2
            )
        
        # Response Time Trends
        fig.add_trace(
            go.Scatter(
                x=general_data['date'],
                y=general_data['avg_response_time'],
                name='Response Time',
                mode='lines'
            ),
            row=2, col=1
        )
        
        # Error Rate Trend
        fig.add_trace(
            go.Scatter(
                x=general_data['date'],
                y=general_data['error_rate'],
                name='Error Rate',
                mode='lines'
            ),
            row=2, col=2
        )
        
        # Hourly Pattern
        if not hourly_data.empty:
            fig.add_trace(
                go.Bar(
                    x=hourly_data['hour'],
                    y=hourly_data['hourly_count'],
                    name='Hourly Usage'
                ),
                row=3, col=1
            )
        
        # Response Quality Metrics (replacing Content Distribution)
        if not quality_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=quality_data['date'],
                    y=quality_data['avg_helpfulness'],
                    name='Helpfulness',
                    mode='lines'
                ),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=quality_data['date'],
                    y=quality_data['avg_hallucination_risk'],
                    name='Hallucination Risk',
                    mode='lines'
                ),
                row=3, col=2
            )
            
        fig.update_layout(
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics Tables
        with st.expander("View Detailed Metrics"):
            tabs = st.tabs(["General", "Chat", "Search", "Content", "Quality"])  # Added Quality tab
            
            with tabs[0]:
                if not general_data.empty:
                    st.dataframe(
                        general_data.style.format({
                            'avg_response_time': '{:.2f}s',
                            'error_rate': '{:.1f}%',
                            'avg_quality_score': '{:.2f}'  # Changed from avg_sentiment
                        })
                    )
            
            with tabs[1]:
                if not chat_data.empty:
                    st.dataframe(
                        chat_data.style.format({
                            'chat_response_time': '{:.2f}s'
                        })
                    )
            
            with tabs[2]:
                if not search_data.empty:
                    st.dataframe(
                        search_data.style.format({
                            'search_response_time': '{:.2f}s'
                        })
                    )
            
            with tabs[3]:
                if not content_data.empty:
                    st.dataframe(
                        content_data.style.format({
                            'avg_similarity': '{:.2f}'
                        })
                    )
            
            # New Quality Metrics Tab
            with tabs[4]:
                if not quality_data.empty:
                    st.dataframe(
                        quality_data.style.format({
                            'avg_helpfulness': '{:.2f}',
                            'avg_hallucination_risk': '{:.2f}',
                            'avg_factual_grounding': '{:.2f}',
                            'high_quality_ratio': '{:.1f}%'
                        })
                    )
                    
                    # Add a quality distribution chart
                    st.subheader("Response Quality Distribution")
                    quality_fig = go.Figure()
                    quality_fig.add_trace(go.Bar(
                        x=quality_data['date'],
                        y=quality_data['high_quality_ratio'],
                        name='High Quality Responses (%)'
                    ))
                    st.plotly_chart(quality_fig)
                    
    except Exception as e:
        logger.error(f"Error displaying usage analytics: {str(e)}")
        st.error("An error occurred while displaying the analytics.")
        
    # Add timestamp
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")