import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.logger import setup_logger
from utils.db_utils import DatabaseConnector

logger = setup_logger(__name__)

class UnifiedAnalyticsDashboard:
    def __init__(self):
        self.db = DatabaseConnector()
        self.conn = self.db.get_connection()
        # Initialize theme state
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        if st.session_state.theme == 'light':
            st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'light'
        # Apply theme configuration
        self.apply_theme()

    def apply_theme(self):
        """Apply comprehensive theme configuration"""
        if st.session_state.theme == 'dark':
            # Dark theme configuration
            st.markdown("""
                <style>
                    /* Main app background and text */
                    .stApp {
                        background-color: #0E1117;
                        color: #FAFAFA;
                    }
                    
                    /* Sidebar */
                    .css-1d391kg, [data-testid="stSidebar"] {
                        background-color: #262730;
                    }
                    
                    /* Metric cards */
                    [data-testid="stMetricValue"] {
                        background-color: #262730;
                        color: #FFFFFF !important;
                    }
                    
                    /* Inputs and controls */
                    .stSelectbox, .stSlider, .stDateInput {
                        background-color: #262730;
                        color: #FFFFFF;
                    }
                    
                    /* DataFrames */
                    .stDataFrame {
                        background-color: #262730;
                    }
                    .dataframe {
                        background-color: #262730;
                        color: #FFFFFF;
                    }
                    .dataframe th {
                        background-color: #404040;
                        color: #FFFFFF;
                    }
                    .dataframe td {
                        background-color: #262730;
                        color: #FFFFFF;
                    }
                    
                    /* Headers and text */
                    h1, h2, h3, h4, h5, h6, .css-10trblm {
                        color: #FFFFFF !important;
                    }
                    .css-145kmo2 {
                        color: #FFFFFF;
                    }
                    
                    /* Buttons */
                    .stButton > button {
                        background-color: #262730;
                        color: #FFFFFF;
                        border: 1px solid #4F4F4F;
                    }
                    .stButton > button:hover {
                        background-color: #404040;
                        color: #FFFFFF;
                        border: 1px solid #4F4F4F;
                    }
                    
                    /* Warning messages */
                    .stAlert {
                        background-color: #262730;
                        color: #FFFFFF;
                    }
                    
                    /* Tooltips */
                    .tooltip {
                        background-color: #262730 !important;
                        color: #FFFFFF !important;
                    }
                    
                    /* Plotly figure backgrounds */
                    .js-plotly-plot .plotly {
                        background-color: #262730 !important;
                    }
                    
                    /* Custom styling for selection boxes */
                    .SelectBox {
                        background-color: #262730 !important;
                        color: #FFFFFF !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Update plot templates for dark theme
            self.plot_template = {
                'layout': {
                    'paper_bgcolor': '#262730',
                    'plot_bgcolor': '#262730',
                    'font': {'color': '#FFFFFF'},
                    'xaxis': {
                        'gridcolor': '#4F4F4F',
                        'linecolor': '#4F4F4F',
                        'zerolinecolor': '#4F4F4F',
                        'tickfont': {'color': '#FFFFFF'}
                    },
                    'yaxis': {
                        'gridcolor': '#4F4F4F',
                        'linecolor': '#4F4F4F',
                        'zerolinecolor': '#4F4F4F',
                        'tickfont': {'color': '#FFFFFF'}
                    },
                    'legend': {'font': {'color': '#FFFFFF'}}
                }
            }
        else:
            # Light theme configuration (default Streamlit)
            st.markdown("""
                <style>
                    .stApp {
                        background-color: #FFFFFF;
                        color: #000000;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Update plot template for light theme
            self.plot_template = {
                'layout': {
                    'paper_bgcolor': '#FFFFFF',
                    'plot_bgcolor': '#FFFFFF',
                    'font': {'color': '#000000'}
                }
            }

    def update_plot_theme(self, fig):
        """Update plot theme based on current theme setting"""
        if st.session_state.theme == 'dark':
            fig.update_layout(
                paper_bgcolor='#262730',
                plot_bgcolor='#262730',
                font={'color': '#FFFFFF'},
                xaxis=dict(
                    gridcolor='#4F4F4F',
                    linecolor='#4F4F4F',
                    zerolinecolor='#4F4F4F'
                ),
                yaxis=dict(
                    gridcolor='#4F4F4F',
                    linecolor='#4F4F4F',
                    zerolinecolor='#4F4F4F'
                )
            )
        return fig

    def create_plot(self, plot_func, *args, **kwargs):
        """Wrapper for creating plots with proper theming"""
        fig = plot_func(*args, **kwargs)
        return self.update_plot_theme(fig)

    def create_sidebar_filters(self):
        st.sidebar.title("Settings")
        
        # Theme toggle at the top of sidebar
        theme_label = "🌙 Dark Mode" if st.session_state.theme == 'light' else "☀️ Light Mode"
        st.sidebar.button(theme_label, on_click=self.toggle_theme)
        
        st.sidebar.title("Filters")
        
        # Date range selector
        self.start_date = st.sidebar.date_input(
            "Start Date",
            datetime.now() - timedelta(days=30)
        )
        self.end_date = st.sidebar.date_input(
            "End Date",
            datetime.now()
        )

        # Changed to dropdown
        self.analytics_type = st.sidebar.selectbox(
            "Analytics Type",
            ["chat", "search", "expert matching", "sentiment"],
            index=0
        )

        # Additional recommendation network filters
        self.min_similarity = st.sidebar.slider(
            "Minimum Similarity Score",
            0.0, 1.0, 0.5
        )

        # Expert count filter
        self.expert_count = st.sidebar.slider(
            "Number of Experts to Show",
            5, 50, 20
        )
    def get_expert_metrics(self) -> pd.DataFrame:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                WITH ChatExperts AS (
                    SELECT 
                        a.expert_id,
                        COUNT(*) as chat_matches,
                        AVG(a.similarity_score) as chat_similarity,
                        SUM(CASE WHEN a.clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as chat_click_rate
                    FROM chat_analytics a
                    JOIN chat_interactions i ON a.interaction_id = i.id
                    WHERE i.timestamp BETWEEN %s AND %s
                    GROUP BY a.expert_id
                ),
                SearchExperts AS (
                    SELECT 
                        expert_id,
                        COUNT(*) as search_matches,
                        AVG(rank_position) as avg_rank,
                        SUM(CASE WHEN expert_searches.clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as search_click_rate
                    FROM expert_searches
                    JOIN search_logs sl ON expert_searches.search_id = sl.id
                    WHERE sl.timestamp BETWEEN %s AND %s
                    GROUP BY expert_id
                )
                SELECT 
                    e.first_name || ' ' || e.last_name as expert_name,
                    e.unit,
                    COALESCE(ce.chat_matches, 0) as chat_matches,
                    COALESCE(ce.chat_similarity, 0) as chat_similarity,
                    COALESCE(ce.chat_click_rate, 0) as chat_click_rate,
                    COALESCE(se.search_matches, 0) as search_matches,
                    COALESCE(se.avg_rank, 0) as search_avg_rank,
                    COALESCE(se.search_click_rate, 0) as search_click_rate
                FROM experts_expert e
                LEFT JOIN ChatExperts ce ON e.id::text = ce.expert_id
                LEFT JOIN SearchExperts se ON e.id::text = se.expert_id
                WHERE e.is_active = true
                ORDER BY (COALESCE(ce.chat_matches, 0) + COALESCE(se.search_matches, 0)) DESC
            """, (self.start_date, self.end_date, self.start_date, self.end_date))
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def main(self):
        """Initialize and run the dashboard with proper theme handling."""
        st.set_page_config(
            page_title="APHRC Analytics Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar filters
        self.create_sidebar_filters()
        
        # Apply theme settings
        self.apply_theme()
        
        # Set title with theme-aware styling
        if st.session_state.theme == 'dark':
            title_color = "#FFFFFF"
        else:
            title_color = "#000000"
            
        st.markdown(
            f'<h1 style="color: {title_color};">APHRC Analytics Dashboard</h1>',
            unsafe_allow_html=True
        )
        
        # Configure plot templates based on theme
        if st.session_state.theme == 'dark':
            plotly_template = 'plotly_dark'
            plot_config = {
                'layout': {
                    'paper_bgcolor': '#262730',
                    'plot_bgcolor': '#262730',
                    'font': {'color': '#FFFFFF'},
                    'xaxis': {'gridcolor': '#4F4F4F'},
                    'yaxis': {'gridcolor': '#4F4F4F'}
                }
            }
        else:
            plotly_template = 'plotly_white'
            plot_config = {
                'layout': {
                    'paper_bgcolor': '#FFFFFF',
                    'plot_bgcolor': '#FFFFFF',
                    'font': {'color': '#000000'}
                }
            }
        
        # Set global plotting defaults
        px.defaults.template = plotly_template
        for fig_type in [px.line, px.bar, px.scatter, px.area]:
            fig_type.update_layout = lambda fig, **kwargs: dict(
                fig.update_layout(**{**plot_config['layout'], **kwargs})
            )
        
        # Display metrics and analytics based on selected type
        self.display_overall_metrics()
        
        # Display specific analytics based on selection
        analytics_type = self.analytics_type.lower()
        if analytics_type == "chat":
            self.display_chat_analytics()
        elif analytics_type == "search":
            self.display_search_analytics()
        elif analytics_type == "sentiment":
            self.display_sentiment_analytics()
        elif analytics_type == "expert matching":
            self.display_expert_analytics()
            self.display_recommendation_network()
        
        # Add footer with theme-aware styling
        st.markdown(
            f"""
            <div style="
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                padding: 10px;
                background-color: {'#262730' if st.session_state.theme == 'dark' else '#FFFFFF'};
                color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
            ">
                APHRC Analytics Dashboard • {datetime.now().year}
            </div>
            """,
            unsafe_allow_html=True
        )

    def get_chat_metrics(self) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                WITH ChatMetrics AS (
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(response_time) as avg_response_time,
                        SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as error_rate
                    FROM chat_interactions
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY DATE(timestamp)
                ),
                ExpertMatchMetrics AS (
                    SELECT 
                        DATE(i.timestamp) as date,
                        COUNT(*) as total_matches,
                        AVG(a.similarity_score) as avg_similarity,
                        SUM(CASE WHEN a.clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as click_rate
                    FROM chat_analytics a
                    JOIN chat_interactions i ON a.interaction_id = i.id
                    WHERE i.timestamp BETWEEN %s AND %s
                    GROUP BY DATE(i.timestamp)
                )
                SELECT 
                    cm.*,
                    em.total_matches,
                    em.avg_similarity,
                    em.click_rate
                FROM ChatMetrics cm
                LEFT JOIN ExpertMatchMetrics em ON cm.date = em.date
                ORDER BY cm.date
            """, (self.start_date, self.end_date, self.start_date, self.end_date))
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def get_search_metrics(self) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(EXTRACT(EPOCH FROM response_time)) as avg_response_time,
                    SUM(CASE WHEN clicked THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as click_through_rate
                FROM search_logs
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (self.start_date, self.end_date))
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def get_sentiment_metrics(self) -> pd.DataFrame:
        """Get sentiment metrics with proper handling of emotion arrays."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                WITH DailyMetrics AS (
                    SELECT 
                        DATE(sm.timestamp) as date,
                        AVG(sm.sentiment_score) as avg_sentiment,
                        AVG(sm.satisfaction_score) as satisfaction_score,
                        AVG(sm.urgency_score) as urgency_score,
                        AVG(sm.clarity_score) as clarity_score,
                        COUNT(*) as total_interactions
                    FROM sentiment_metrics sm
                    WHERE sm.timestamp BETWEEN %s AND %s
                    GROUP BY DATE(sm.timestamp)
                ),
                DailyEmotions AS (
                    SELECT 
                        DATE(sm.timestamp) as date,
                        emotion
                    FROM sentiment_metrics sm,
                        LATERAL unnest(sm.emotion_labels) as emotion
                    WHERE sm.timestamp BETWEEN %s AND %s
                ),
                CommonEmotion AS (
                    SELECT 
                        date,
                        emotion as common_emotion,
                        emotion_count,
                        ROW_NUMBER() OVER (PARTITION BY date ORDER BY emotion_count DESC) as rn
                    FROM (
                        SELECT 
                            date,
                            emotion,
                            COUNT(*) as emotion_count
                        FROM DailyEmotions
                        GROUP BY date, emotion
                    ) counted
                )
                SELECT 
                    dm.date,
                    dm.avg_sentiment,
                    dm.satisfaction_score,
                    dm.urgency_score,
                    dm.clarity_score,
                    COALESCE(ce.common_emotion, 'neutral') as common_emotion,
                    dm.total_interactions
                FROM DailyMetrics dm
                LEFT JOIN CommonEmotion ce ON dm.date = ce.date AND ce.rn = 1
                ORDER BY dm.date
            """, (self.start_date, self.end_date, self.start_date, self.end_date))
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def display_overall_metrics(self):
        col1, col2, col3, col4 = st.columns(4)
        
        # Get overall metrics
        cursor = self.conn.cursor()
        try:
            # Combined chat and search metrics
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM chat_interactions 
                    WHERE timestamp BETWEEN %s AND %s) as total_chat_interactions,
                    (SELECT COUNT(*) FROM search_logs 
                    WHERE timestamp BETWEEN %s AND %s) as total_searches,
                    (SELECT COUNT(DISTINCT user_id) FROM (
                        SELECT user_id FROM chat_interactions 
                        WHERE timestamp BETWEEN %s AND %s
                        UNION
                        SELECT user_id FROM search_logs 
                        WHERE timestamp BETWEEN %s AND %s
                    ) u) as unique_users,
                    (SELECT COUNT(*) FROM chat_analytics 
                    WHERE clicked = true) +
                    (SELECT COUNT(*) FROM expert_searches 
                    WHERE clicked = true) as total_expert_clicks
            """, (self.start_date, self.end_date) * 4)
            
            metrics = cursor.fetchone()
            
            with col1:
                st.metric("Total Interactions", f"{metrics[0] + metrics[1]:,}")
            with col2:
                st.metric("Chat Interactions", f"{metrics[0]:,}")
            with col3:
                st.metric("Unique Users", f"{metrics[2]:,}")
            with col4:
                st.metric("Expert Clicks", f"{metrics[3]:,}")
                
        finally:
            cursor.close()

    def display_chat_analytics(self):
        st.subheader("Chat Analytics")
        
        # Get chat metrics
        chat_metrics = self.get_chat_metrics()
        
        # Daily chat volume
        st.plotly_chart(
            px.line(
                chat_metrics,
                x="date",
                y=["total_interactions", "unique_sessions"],
                title="Daily Chat Volume",
                labels={"value": "Count", "variable": "Metric"}
            )
        )
        
        # Response time and error rate
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.line(
                    chat_metrics,
                    x="date",
                    y="avg_response_time",
                    title="Average Response Time"
                )
            )
        with col2:
            st.plotly_chart(
                px.line(
                    chat_metrics,
                    x="date",
                    y="error_rate",
                    title="Error Rate"
                )
            )

    def display_sentiment_analytics(self):
        st.subheader("Sentiment Analytics")
        
        # Get sentiment metrics
        sentiment_data = self.get_sentiment_metrics()
        
        # 1. Overall Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Average Sentiment",
                f"{sentiment_data['avg_sentiment'].mean():.2f}",
                f"{(sentiment_data['avg_sentiment'].diff().mean() * 100):.1f}%"
            )
        with col2:
            st.metric(
                "Average Satisfaction",
                f"{sentiment_data['satisfaction_score'].mean():.2f}",
                f"{(sentiment_data['satisfaction_score'].diff().mean() * 100):.1f}%"
            )
        with col3:
            st.metric(
                "Average Clarity",
                f"{sentiment_data['clarity_score'].mean():.2f}",
                f"{(sentiment_data['clarity_score'].diff().mean() * 100):.1f}%"
            )
        with col4:
            st.metric(
                "Total Interactions",
                f"{sentiment_data['total_interactions'].sum():,}",
                f"{(sentiment_data['total_interactions'].diff().mean()):.0f}/day"
            )

        # 2. Main Sentiment Trend with Range Selector
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sentiment_data['date'],
            y=sentiment_data['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='rgb(49, 130, 189)'),
            fill='tonexty'
        ))
        
        # Apply theme
        fig = self.update_plot_theme(fig)
        fig.update_layout(
            title='Sentiment Trend Over Time',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
            yaxis=dict(title='Average Sentiment Score')
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Sentiment Components Comparison
        col1, col2 = st.columns(2)
        with col1:
            # Radar Chart for Average Scores
            categories = ['Sentiment', 'Satisfaction', 'Urgency', 'Clarity']
            values = [
                sentiment_data['avg_sentiment'].mean(),
                sentiment_data['satisfaction_score'].mean(),
                sentiment_data['urgency_score'].mean(),
                sentiment_data['clarity_score'].mean()
            ]
            
            radar_fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself'
            ))
            radar_fig = self.update_plot_theme(radar_fig)
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title='Average Sentiment Components'
            )
            st.plotly_chart(radar_fig)

        with col2:
            # Create emotion pivot table
            dates = sentiment_data['date']
            emotions = sentiment_data['common_emotion']
            
            # Create pivot table with date index and emotions as columns
            emotion_pivot = pd.crosstab(
                dates,
                emotions,
                normalize='index'
            ).fillna(0)
            
            # Create the area plot
            emotion_fig = px.area(
                emotion_pivot,
                title='Emotion Distribution Trend',
                labels={'value': 'Proportion', 'variable': 'Emotion'}
            )
            emotion_fig = self.update_plot_theme(emotion_fig)
            st.plotly_chart(emotion_fig)

        # 4. Correlation Heatmap
        correlation_data = sentiment_data[[
            'avg_sentiment',
            'satisfaction_score',
            'urgency_score',
            'clarity_score',
            'total_interactions'
        ]].corr()

        heatmap_fig = px.imshow(
            correlation_data,
            title='Correlation Between Metrics',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        heatmap_fig = self.update_plot_theme(heatmap_fig)
        st.plotly_chart(heatmap_fig)

        # 5. Hourly Analysis
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(satisfaction_score) as avg_satisfaction,
                    COUNT(*) as interaction_count
                FROM sentiment_metrics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            """, (self.start_date, self.end_date))
            
            hourly_data = pd.DataFrame(cursor.fetchall(), 
                                    columns=['hour', 'avg_sentiment', 'avg_satisfaction', 'interaction_count'])

            hourly_fig = go.Figure()
            hourly_fig.add_trace(go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['avg_sentiment'],
                name='Sentiment',
                mode='lines+markers'
            ))
            hourly_fig.add_trace(go.Bar(
                x=hourly_data['hour'],
                y=hourly_data['interaction_count'],
                name='Interactions',
                yaxis='y2',
                opacity=0.3
            ))
            
            hourly_fig = self.update_plot_theme(hourly_fig)
            hourly_fig.update_layout(
                title='Hourly Sentiment Analysis',
                xaxis=dict(title='Hour of Day'),
                yaxis=dict(title='Average Sentiment'),
                yaxis2=dict(title='Number of Interactions', overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(hourly_fig)

        finally:
            cursor.close()

        # 6. Latest Sentiment Trends Table
        st.subheader("Recent Sentiment Trends")
        latest_data = sentiment_data.tail(10).sort_values('date', ascending=False)
        
        # Format the data
        display_cols = ['date', 'avg_sentiment', 'satisfaction_score', 'urgency_score', 
                    'clarity_score', 'common_emotion', 'total_interactions']
        display_data = latest_data[display_cols].copy()
        
        # Round numeric columns
        numeric_cols = ['avg_sentiment', 'satisfaction_score', 'urgency_score', 'clarity_score']
        display_data[numeric_cols] = display_data[numeric_cols].round(3)
        
        # Create styled dataframe
        st.dataframe(
            display_data,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "avg_sentiment": st.column_config.NumberColumn(
                    "Average Sentiment",
                    help="Average sentiment score",
                    format="%.3f"
                ),
                "satisfaction_score": st.column_config.NumberColumn(
                    "Satisfaction",
                    help="Satisfaction score",
                    format="%.3f"
                ),
                "urgency_score": st.column_config.NumberColumn(
                    "Urgency",
                    help="Urgency score",
                    format="%.3f"
                ),
                "clarity_score": st.column_config.NumberColumn(
                    "Clarity",
                    help="Clarity score",
                    format="%.3f"
                ),
                "common_emotion": "Common Emotion",
                "total_interactions": st.column_config.NumberColumn(
                    "Total Interactions",
                    help="Number of interactions"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    def display_search_analytics(self):
        st.subheader("Search Analytics")
        
        # Get search metrics
        search_metrics = self.get_search_metrics()
        
        # Daily search volume
        st.plotly_chart(
            px.line(
                search_metrics,
                x="date",
                y=["total_searches", "unique_users"],
                title="Daily Search Volume",
                labels={"value": "Count", "variable": "Metric"}
            )
        )
        
        # Click-through rate
        st.plotly_chart(
            px.line(
                search_metrics,
                x="date",
                y="click_through_rate",
                title="Click-through Rate"
            )
        )

    def display_expert_analytics(self):
        st.subheader("Expert Analytics")
        
        # Get expert metrics
        expert_metrics = self.get_expert_metrics()
        
        # Expert performance heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[
                expert_metrics.chat_similarity,
                expert_metrics.chat_click_rate,
                expert_metrics.search_click_rate
            ],
            x=expert_metrics.expert_name,
            y=['Similarity Score', 'Chat CTR', 'Search CTR'],
            colorscale='Viridis'
        ))
        fig.update_layout(title='Expert Performance Matrix')
        st.plotly_chart(fig)
        
        # Expert metrics table
        st.dataframe(
            expert_metrics[[
                'expert_name', 'unit', 'chat_matches', 'search_matches',
                'chat_click_rate', 'search_click_rate'
            ]].sort_values('chat_matches', ascending=False)
        )

    def display_user_behavior(self):
        st.subheader("User Behavior Analytics")
        
        # Get user behavior metrics
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                WITH UserMetrics AS (
                    SELECT 
                        user_id,
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT session_id) as total_sessions,
                        AVG(EXTRACT(EPOCH FROM (
                            SELECT MAX(timestamp) - MIN(timestamp) 
                            FROM chat_interactions ci2 
                            WHERE ci2.session_id = ci1.session_id
                        ))) as avg_session_duration
                    FROM chat_interactions ci1
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY user_id
                )
                SELECT 
                    CASE 
                        WHEN total_interactions <= 5 THEN 'Low'
                        WHEN total_interactions <= 15 THEN 'Medium'
                        ELSE 'High'
                    END as engagement_level,
                    COUNT(*) as user_count,
                    AVG(total_sessions) as avg_sessions,
                    AVG(avg_session_duration) as avg_duration
                FROM UserMetrics
                GROUP BY 
                    CASE 
                        WHEN total_interactions <= 5 THEN 'Low'
                        WHEN total_interactions <= 15 THEN 'Medium'
                        ELSE 'High'
                    END
            """, (self.start_date, self.end_date))
            
            columns = [desc[0] for desc in cursor.description]
            behavior_data = pd.DataFrame(cursor.fetchall(), columns=columns)
            
            # User engagement distribution
            st.plotly_chart(
                px.bar(
                    behavior_data,
                    x='engagement_level',
                    y='user_count',
                    title='User Engagement Distribution'
                )
            )
            
            # Session metrics
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    px.bar(
                        behavior_data,
                        x='engagement_level',
                        y='avg_sessions',
                        title='Average Sessions per User'
                    )
                )
            with col2:
                st.plotly_chart(
                    px.bar(
                        behavior_data,
                        x='engagement_level',
                        y='avg_duration',
                        title='Average Session Duration (seconds)'
                    )
                )
            
        finally:
            cursor.close()

    def display_domain_analytics(self):
        st.subheader("Domain Analytics")
        
        cursor = self.conn.cursor()
        try:
            # Domain expertise analytics
            cursor.execute("""
                SELECT 
                    domain_name,
                    expert_count,
                    match_count,
                    match_count::float / NULLIF(expert_count, 0) as engagement_rate
                FROM domain_expertise_analytics
                ORDER BY match_count DESC
                LIMIT %s
            """, (self.expert_count,))
            
            columns = [desc[0] for desc in cursor.description]
            domain_data = pd.DataFrame(cursor.fetchall(), columns=columns)
            
            # Domain popularity bar chart
            st.plotly_chart(
                px.bar(
                    domain_data,
                    x='domain_name',
                    y='match_count',
                    title='Domain Popularity',
                    hover_data=['expert_count', 'engagement_rate']
                )
            )
            
            # Domain engagement heatmap
            fig = go.Figure(data=go.Heatmap(
                z=domain_data[['expert_count', 'match_count', 'engagement_rate']].values,
                x=['Expert Count', 'Match Count', 'Engagement Rate'],
                y=domain_data['domain_name'],
                colorscale='Viridis'
            ))
            fig.update_layout(title='Domain Engagement Matrix')
            st.plotly_chart(fig)
            
            # Detailed domain metrics table
            st.dataframe(
                domain_data.sort_values('match_count', ascending=False),
                use_container_width=True
            )
            
        finally:
            cursor.close()

    def display_recommendation_network(self):
        st.subheader("Expert Recommendation Network")
        
        cursor = self.conn.cursor()
        try:
            # Get expert matching data
            cursor.execute("""
                WITH MatchingData AS (
                    SELECT 
                        expert_id,
                        matched_expert_id,
                        similarity_score,
                        shared_domains,
                        ROW_NUMBER() OVER (ORDER BY similarity_score DESC) as rank
                    FROM expert_matching_logs
                    WHERE similarity_score >= %s
                )
                SELECT 
                    expert_id,
                    matched_expert_id,
                    similarity_score,
                    shared_domains
                FROM MatchingData
                WHERE rank <= %s
            """, (self.min_similarity, self.expert_count))
            
            matching_data = pd.DataFrame(
                cursor.fetchall(), 
                columns=['source', 'target', 'similarity', 'shared_domains']
            )
            
            # If no matching data, show a message
            if matching_data.empty:
                st.warning("No expert matches found with the current filters.")
                return
            
            # Create networkx graph
            G = nx.from_pandas_edgelist(
                matching_data, 
                'source', 
                'target', 
                ['similarity', 'shared_domains']
            )
            
            # Create network visualization
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            edge_colors = []
            
            # Prepare edges
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                # Color edges based on similarity
                edge_colors.append(edge[2]['similarity'])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, 
                y=edge_y,
                line=dict(
                    width=1, 
                    color=edge_colors, 
                    colorscale='Viridis',
                    showscale=True
                ),
                hoverinfo='text',
                mode='lines'
            ))
            
            # Add nodes
            node_degrees = [len(list(G.neighbors(node))) for node in G.nodes()]
            fig.add_trace(go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                marker=dict(
                    size=[10 + 5 * degree for degree in node_degrees],
                    color=node_degrees,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Connections')
                ),
                text=[f"Expert {node}" for node in G.nodes()],
                textposition='top center'
            ))
            
            fig.update_layout(
                title="Expert Recommendation Network",
                showlegend=False,
                hovermode='closest'
            )
            
            st.plotly_chart(fig)
            
            # Expert matching metrics
            st.subheader("Expert Matching Metrics")
            cursor.execute("""
                SELECT 
                    AVG(similarity_score) as avg_similarity,
                    COUNT(*) as total_matches,
                    COUNT(DISTINCT expert_id) as unique_experts,
                    COUNT(DISTINCT matched_expert_id) as matched_experts
                FROM expert_matching_logs
                WHERE timestamp BETWEEN %s AND %s
                AND similarity_score >= %s
            """, (self.start_date, self.end_date, self.min_similarity))
            
            metrics = cursor.fetchone()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Similarity", f"{metrics[0]:.2f}")
            with col2:
                st.metric("Total Matches", f"{metrics[1]:,}")
            with col3:
                st.metric("Unique Experts", f"{metrics[2]:,}")
            with col4:
                st.metric("Matched Experts", f"{metrics[3]:,}")
            
            # Detailed matching data
            st.subheader("Top Expert Matches")
            cursor.execute("""
                SELECT 
                    expert_id,
                    matched_expert_id,
                    similarity_score,
                    shared_domains,
                    timestamp
                FROM expert_matching_logs
                WHERE timestamp BETWEEN %s AND %s
                AND similarity_score >= %s
                ORDER BY similarity_score DESC
                LIMIT 50
            """, (self.start_date, self.end_date, self.min_similarity))
            
            columns = [desc[0] for desc in cursor.description]
            matches_data = pd.DataFrame(cursor.fetchall(), columns=columns)
            
            st.dataframe(matches_data, use_container_width=True)
            
        finally:
            cursor.close()

if __name__ == "__main__":
    dashboard = UnifiedAnalyticsDashboard()
    dashboard.main()
