import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging
from typing import Dict, Optional, Any

def get_overview_metrics(conn, start_date, end_date):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            WITH InteractionMetrics AS (
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(CAST(metrics->>'response_time' AS FLOAT)) as avg_response_time,
                    COUNT(CASE WHEN CAST(metrics->>'error_occurred' AS BOOLEAN) THEN 1 END)::FLOAT / 
                        NULLIF(COUNT(*), 0) * 100 as error_rate,
                    0.0 as placeholder_score  -- Placeholder for join
                FROM interactions
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
            ),
            QualityMetrics AS (
                SELECT 
                    DATE(cl.timestamp) as date,
                    AVG(rqm.helpfulness_score) as avg_helpfulness,
                    AVG(rqm.hallucination_risk) as avg_hallucination_risk,
                    AVG(rqm.factual_grounding_score) as avg_factual_grounding,
                    COUNT(*) as quality_evaluations
                FROM chatbot_logs cl
                JOIN response_quality_metrics rqm ON cl.id = rqm.interaction_id
                WHERE cl.timestamp BETWEEN %s AND %s
                GROUP BY DATE(cl.timestamp)
            ),
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
            ),
            MessageMetrics AS (
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN draft THEN 1 END) as draft_messages
                FROM expert_messages
                WHERE created_at BETWEEN %s AND %s
                GROUP BY DATE(created_at)
            )
            SELECT 
                COALESCE(im.date, qm.date, em.date, mm.date) as date,
                im.total_interactions,
                im.unique_users,
                im.avg_response_time,
                im.error_rate,
                COALESCE(qm.avg_helpfulness, 0.0) as avg_quality_score,  -- Use helpfulness instead of sentiment
                qm.avg_helpfulness,
                qm.avg_hallucination_risk,
                qm.avg_factual_grounding,
                em.expert_matches,
                em.avg_similarity,
                em.success_rate,
                mm.total_messages,
                mm.draft_messages
            FROM InteractionMetrics im
            FULL OUTER JOIN QualityMetrics qm ON im.date = qm.date
            FULL OUTER JOIN ExpertMetrics em ON im.date = em.date
            FULL OUTER JOIN MessageMetrics mm ON im.date = mm.date
            ORDER BY date;
        """, (start_date, end_date) * 4)  # Added one more date range pair for QualityMetrics
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    finally:
        cursor.close()

def display_overview_analytics(metrics_df, filters):
    st.subheader("Overview Analytics")

    total_interactions = metrics_df['total_interactions'].sum()
    total_messages = metrics_df['total_messages'].sum()
    success_rate = metrics_df['success_rate'].mean()
    avg_quality = metrics_df['avg_quality_score'].mean()  # Now using quality score instead of sentiment

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Interactions", f"{total_interactions:,}")
    with col2:
        st.metric("Total Messages", f"{total_messages:,}")
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Avg Quality", f"{avg_quality:.2f}")  # Changed label from "Avg Sentiment" to "Avg Quality"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Daily Activity",
            "User Engagement",
            "Performance Metrics",
            "Response Quality"  # Changed from "Expert Matching"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Daily Activity
    fig.add_trace(
        go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['total_interactions'],
            name='Interactions',
            mode='lines+markers'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['total_messages'],
            name='Messages',
            mode='lines+markers'
        ),
        row=1, col=1
    )

    # User Engagement
    fig.add_trace(
        go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['unique_users'],
            name='Unique Users',
            mode='lines+markers'
        ),
        row=1, col=2
    )

    # Performance
    fig.add_trace(
        go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['avg_response_time'],
            name='Response Time',
            mode='lines'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df['date'],
            y=metrics_df['error_rate'],
            name='Error Rate',
            mode='lines'
        ),
        row=2, col=1
    )

    # Response Quality - replaced Expert Matching
    if 'avg_helpfulness' in metrics_df.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_helpfulness'],
                name='Helpfulness',
                mode='lines'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['avg_hallucination_risk'],
                name='Hallucination Risk',
                yaxis='y2',
                mode='lines'
            ),
            row=2, col=2
        )
    else:
        # Fallback to expert matching if quality metrics aren't available
        fig.add_trace(
            go.Bar(
                x=metrics_df['date'],
                y=metrics_df['expert_matches'],
                name='Expert Matches'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['success_rate'],
                name='Success Rate',
                yaxis='y2',
                mode='lines'
            ),
            row=2, col=2
        )

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

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Users", row=1, col=2)
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)  # Changed from "Matches"

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Metrics"):
        st.dataframe(metrics_df.style.background_gradient(
            subset=['total_interactions', 'success_rate', 'avg_response_time', 'avg_quality_score'],  # Added avg_quality_score
            cmap='RdYlGn'
        ))