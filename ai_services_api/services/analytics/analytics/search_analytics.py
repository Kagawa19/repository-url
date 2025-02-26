import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging
from typing import Dict

def get_search_metrics(conn, start_date, end_date):
    """Get comprehensive search metrics using our views"""
    cursor = conn.cursor()
    try:
        # Get all metrics in a single query with proper date filtering
        cursor.execute("""
            WITH DailyMetrics AS (
                SELECT 
                    DATE(sa.timestamp) as date,
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT sa.user_id) as unique_users,
                    AVG(sa.response_time) as avg_response_time,
                    SUM(CASE WHEN sa.result_count > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                    AVG(sa.result_count) as avg_results,
                    COUNT(DISTINCT ss.session_id) as total_sessions
                FROM search_analytics sa
                LEFT JOIN search_sessions ss ON sa.search_id = ss.id
                WHERE sa.timestamp BETWEEN %s AND %s
                GROUP BY DATE(sa.timestamp)
            ),
            SessionMetrics AS (
                SELECT 
                    DATE(start_timestamp) as date,
                    COUNT(*) as total_sessions,
                    AVG(query_count) as avg_queries_per_session,
                    AVG(CASE 
                        WHEN successful_searches > 0 AND query_count > 0 
                        THEN successful_searches::float / query_count 
                        ELSE 0 
                    END) as session_success_rate,
                    AVG(EXTRACT(epoch FROM (end_timestamp - start_timestamp))) as avg_session_duration
                FROM search_sessions
                WHERE start_timestamp BETWEEN %s AND %s
                    AND end_timestamp IS NOT NULL
                GROUP BY DATE(start_timestamp)
            ),
            ExpertMetrics AS (
                SELECT 
                    DATE(sa.timestamp) as date,
                    COUNT(DISTINCT esm.expert_id) as matched_experts,
                    AVG(esm.similarity_score) as avg_similarity,
                    AVG(esm.rank_position) as avg_rank
                FROM search_analytics sa
                JOIN expert_search_matches esm ON sa.search_id = esm.search_id
                WHERE sa.timestamp BETWEEN %s AND %s
                GROUP BY DATE(sa.timestamp)
            )
            SELECT 
                d.*,
                s.avg_queries_per_session,
                s.session_success_rate,
                s.avg_session_duration,
                e.matched_experts,
                e.avg_similarity,
                e.avg_rank
            FROM DailyMetrics d
            LEFT JOIN SessionMetrics s ON d.date = s.date
            LEFT JOIN ExpertMetrics e ON d.date = e.date
            ORDER BY d.date
        """, (start_date, end_date, start_date, end_date, start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        daily_metrics = pd.DataFrame(data, columns=columns)
        
        # Get domain performance metrics with fixed query
        cursor.execute("""
            SELECT 
                d.domain_name,
                COUNT(*) as match_count,
                AVG(esm.similarity_score) as avg_similarity,
                AVG(esm.rank_position) as avg_rank
            FROM domain_expertise_analytics d
            JOIN experts_expert e ON d.domain_name = ANY(e.domains)
            JOIN expert_search_matches esm ON e.id::text = esm.expert_id
            JOIN search_analytics sa ON esm.search_id = sa.search_id
            WHERE sa.timestamp BETWEEN %s AND %s
            GROUP BY d.domain_name
            ORDER BY match_count DESC
            LIMIT 10
        """, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        domain_metrics = pd.DataFrame(data, columns=columns)
        
        return {
            'daily_metrics': daily_metrics,
            'domain_metrics': domain_metrics
        }
    finally:
        cursor.close()

def display_search_analytics(metrics: Dict[str, pd.DataFrame], filters: Dict = None):
    """Display search analytics using updated metrics"""
    st.subheader("Search Analytics Dashboard")

    daily_data = metrics['daily_metrics']
    if daily_data.empty:
        st.warning("No search data available for the selected period")
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Searches", 
            f"{daily_data['total_searches'].sum():,}"
        )
    with col2:
        st.metric(
            "Unique Users", 
            f"{daily_data['unique_users'].sum():,}"
        )
    with col3:
        st.metric(
            "Avg Success Rate", 
            f"{daily_data['success_rate'].mean():.1%}"
        )
    with col4:
        st.metric(
            "Avg Response", 
            f"{daily_data['avg_response_time'].mean():.2f}s"
        )

    # Create dashboard layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Search Volume Trends",
            "Response & Success Metrics",
            "Session Analytics",
            "Expert Matching Performance"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    # 1. Search Volume Trends
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_searches'],
            name="Total Searches",
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['unique_users'],
            name="Unique Users",
            line=dict(color='green')
        ),
        row=1, col=1,
        secondary_y=True
    )

    # 2. Response Time and Success Rate
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_response_time'],
            name="Response Time",
            line=dict(color='orange')
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['success_rate'],
            name="Success Rate",
            line=dict(color='purple')
        ),
        row=1, col=2,
        secondary_y=True
    )

    # 3. Session Analytics
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_queries_per_session'],
            name="Queries/Session",
            line=dict(color='red')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['session_success_rate'],
            name="Session Success",
            line=dict(color='cyan')
        ),
        row=2, col=1,
        secondary_y=True
    )

    # 4. Expert Matching Performance
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_similarity'],
            name="Match Similarity",
            line=dict(color='darkblue')
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_rank'],
            name="Avg Rank",
            line=dict(color='darkred')
        ),
        row=2, col=2,
        secondary_y=True
    )

    # Update layout
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

    # Update axes titles
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Users", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Response Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate", secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text="Queries", row=2, col=1)
    fig.update_yaxes(title_text="Success Rate", secondary_y=True, row=2, col=1)
    fig.update_yaxes(title_text="Similarity", row=2, col=2)
    fig.update_yaxes(title_text="Rank", secondary_y=True, row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Domain Performance Table
    st.subheader("Domain Performance")
    domain_data = metrics['domain_metrics']
    if not domain_data.empty:
        styled_domain_data = domain_data.style.format({
            'avg_similarity': '{:.2%}',
            'avg_rank': '{:.1f}',
            'match_count': '{:,.0f}'
        }).background_gradient(
            subset=['match_count', 'avg_similarity'],
            cmap='Blues'
        )
        st.dataframe(styled_domain_data)

    # Add date/time of last update
    st.markdown(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")