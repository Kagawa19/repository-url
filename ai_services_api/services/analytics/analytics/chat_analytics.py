import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import asyncio
from datetime import datetime, timedelta

async def get_daily_metrics(conn, start_date, end_date):
    """Get daily chat metrics aggregated from chatbot_logs and chat_sessions."""
    cursor = conn.cursor()
    try:
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
                    SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / COUNT(*) as session_success_rate
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
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    finally:
        cursor.close()

async def get_quality_metrics_trends(conn, start_date, end_date):
    """Get quality metrics from response_quality_metrics and chatbot_logs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                DATE(cl.timestamp) as date,
                AVG(rqm.helpfulness_score) as avg_helpfulness,
                AVG(rqm.hallucination_risk) as avg_hallucination_risk,
                AVG(rqm.factual_grounding_score) as avg_factual_grounding,
                COUNT(*) as total_interactions,
                COUNT(CASE WHEN rqm.helpfulness_score >= 0.7 THEN 1 END)::float / COUNT(*) as high_quality_ratio
            FROM chatbot_logs cl
            JOIN response_quality_metrics rqm ON cl.id = rqm.interaction_id
            WHERE cl.timestamp BETWEEN %s AND %s
            GROUP BY DATE(cl.timestamp)
            ORDER BY date
        """, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    finally:
        cursor.close()

def get_chat_metrics(conn, start_date, end_date):
    """
    Synchronous wrapper for getting chat metrics.
    """
    async def get_metrics_async():
        # Gather metrics concurrently
        daily_metrics, quality_metrics = await asyncio.gather(
            get_daily_metrics(conn, start_date, end_date),
            get_quality_metrics_trends(conn, start_date, end_date)
        )
        
        # Merge daily and quality metrics on date
        metrics = pd.merge(
            daily_metrics,
            quality_metrics,
            on='date',
            how='left'
        )
        
        return metrics

    # Run async code in sync context
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metrics = loop.run_until_complete(get_metrics_async())
        loop.close()
        return metrics
    except Exception as e:
        st.error(f"Error getting chat metrics: {e}")
        raise

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
            # Changed from sentiment to helpfulness
            avg_helpfulness = metrics_df['avg_helpfulness'].mean() if 'avg_helpfulness' in metrics_df.columns else 0
            st.metric("Avg Helpfulness", f"{avg_helpfulness:.2f}")
        
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

        # Session Metrics
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
        
        # Average Messages per Session
        session_fig.add_trace(go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['avg_messages_per_session'],
            mode='lines',
            name='Avg Messages per Session',
            yaxis='y2'
        ))
        
        # Session Success Rate
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

        # Response Time Analysis
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

        # Quality Metrics Analysis (replacing Sentiment Analysis)
        if 'avg_helpfulness' in metrics_df.columns:
            st.subheader("Response Quality Analysis")
            quality_fig = go.Figure()
            
            # Helpfulness Score
            quality_fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_helpfulness'],
                mode='lines',
                name='Avg Helpfulness',
                yaxis='y1'
            ))
            
            # Hallucination Risk
            quality_fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_hallucination_risk'],
                mode='lines',
                name='Avg Hallucination Risk',
                yaxis='y2'
            ))
            
            # Factual Grounding Score
            quality_fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_factual_grounding'],
                mode='lines',
                name='Avg Factual Grounding',
                yaxis='y3'
            ))
            
            quality_fig.update_layout(
                title="Response Quality Trends",
                yaxis=dict(title='Helpfulness'),
                yaxis2=dict(title='Hallucination Risk', overlaying='y', side='right'),
                yaxis3=dict(title='Factual Grounding', overlaying='y', side='right', anchor='free', position=1)
            )
            st.plotly_chart(quality_fig)
            
            # Add High Quality Responses ratio chart if available
            if 'high_quality_ratio' in metrics_df.columns:
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
        raise