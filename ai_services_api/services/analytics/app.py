from analytics.chat_analytics import get_chat_and_search_metrics, display_chat_analytics
from analytics.search_analytics import get_search_metrics, display_search_analytics
from analytics.expert_analytics import get_expert_metrics, display_expert_analytics
from analytics.overview_analytics import get_overview_metrics, display_overview_analytics
from analytics.resource_analytics import get_resource_metrics, display_resource_analytics
from analytics.content_analytics import get_content_metrics, display_content_analytics
from components.sidebar import create_sidebar_filters
from utils.db_utils import DatabaseConnector as db
import os
from urllib.parse import urlparse
import psycopg2
from contextlib import contextmanager
from utils.logger import setup_logger
from utils.theme import toggle_theme, apply_theme, update_plot_theme
from datetime import datetime, date, timedelta
import pandas as pd
import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio

class UnifiedAnalyticsDashboard:
    """
    Main dashboard class that integrates all analytics components and manages the application state.
    Features a dynamic sidebar with interactive navigation and contextual filters.
    """
    
    def __init__(self):
        """Initialize the dashboard with database connection and basic configuration."""
        try:
            self.logger = setup_logger(name="analytics_dashboard")
        except Exception as e:
            print(f"Warning: Logger initialization failed: {str(e)}")
            self.logger = logging.getLogger("analytics_dashboard")
            self.logger.setLevel(logging.INFO)
            
        # Initialize database connection
        try:
            self.conn = db.get_connection()
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            st.error("Failed to connect to the database. Please check your connection settings.")
            self.conn = None
            
        # Initialize session state
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

    def main(self):
        """Main application loop with enhanced sidebar integration."""
        try:
            st.set_page_config(
                page_title="APHRC Analytics Dashboard",
                page_icon="📊",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            apply_theme()
            
            # Get filters and selected analytics type from enhanced sidebar
            start_date, end_date, analytics_type, filters = create_sidebar_filters()
            
            # Add auto-refresh option
            refresh_interval = st.sidebar.selectbox(
                "Auto Refresh",
                [None, "1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=0
            )
            
            # Handle auto-refresh
            if refresh_interval:
                minutes = {
                    "1 minute": 1, 
                    "5 minutes": 5, 
                    "15 minutes": 15, 
                    "30 minutes": 30, 
                    "1 hour": 60
                }.get(refresh_interval, 0)
                
                if (datetime.now() - st.session_state.last_refresh).total_seconds() >= minutes * 60:
                    st.session_state.last_refresh = datetime.now()
                    st.experimental_rerun()
            
            # Display header with selected analytics type
            self.display_header(analytics_type)
            
            # Display dashboard quick stats at the top
            if analytics_type == "Overview":
                self.display_quick_stats_cards(start_date, end_date)
            
            # Display analytics content
            try:
                self.display_analytics(analytics_type, start_date, end_date, filters)
            except Exception as e:
                self.logger.error(f"Error displaying analytics: {str(e)}")
                st.error(f"An error occurred while displaying analytics: {str(e)}")
            
            self.display_footer()
            
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")

    def display_quick_stats_cards(self, start_date, end_date):
        """Display key metrics in attractive cards at the top of the overview dashboard."""
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59' if isinstance(end_date, datetime) else end_date
        
        # Create a 4-column layout for metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # Get metrics from relevant tables - using simple queries with error handling
            cursor = self.conn.cursor()
            
            # Total interactions (chats + searches)
            total_interactions = 0
            total_users = 0
            success_rate = 0
            total_experts = 0
            
            # Check if chatbot_logs exists and get metrics
            try:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'chatbot_logs'
                    )
                """)
                if cursor.fetchone()[0]:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as chats,
                            COUNT(DISTINCT user_id) as users
                        FROM chatbot_logs
                        WHERE timestamp BETWEEN %s AND %s
                    """, (start_date_str, end_date_str))
                    
                    chat_result = cursor.fetchone()
                    if chat_result:
                        total_interactions += chat_result[0]
                        total_users += chat_result[1]
            except Exception as e:
                self.logger.error(f"Error getting chatbot metrics: {str(e)}")
            
            # Check if search_analytics exists and get metrics
            try:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'search_analytics'
                    )
                """)
                if cursor.fetchone()[0]:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as searches,
                            COUNT(DISTINCT user_id) as users,
                            AVG(CASE WHEN result_count > 0 THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM search_analytics
                        WHERE timestamp BETWEEN %s AND %s
                    """, (start_date_str, end_date_str))
                    
                    search_result = cursor.fetchone()
                    if search_result:
                        total_interactions += search_result[0]
                        total_users += search_result[1]
                        success_rate = search_result[2] * 100 if search_result[2] is not None else 0
            except Exception as e:
                self.logger.error(f"Error getting search metrics: {str(e)}")
            
            # Check if experts_expert exists and get count
            try:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'experts_expert'
                    )
                """)
                if cursor.fetchone()[0]:
                    cursor.execute("SELECT COUNT(*) FROM experts_expert")
                    expert_result = cursor.fetchone()
                    if expert_result:
                        total_experts = expert_result[0]
            except Exception as e:
                self.logger.error(f"Error getting expert count: {str(e)}")
            
            # Close cursor
            cursor.close()
            
            # Display metrics in attractive cards
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                        <h4 style="margin:0;padding:0;color:#31333F;">Total Interactions</h4>
                        <h2 style="margin:0;padding:10px 0;color:#1f77b4;font-size:28px;">{total_interactions:,}</h2>
                        <p style="margin:0;color:#666;font-size:14px;">Chats & Searches</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                        <h4 style="margin:0;padding:0;color:#31333F;">Unique Users</h4>
                        <h2 style="margin:0;padding:10px 0;color:#ff7f0e;font-size:28px;">{total_users:,}</h2>
                        <p style="margin:0;color:#666;font-size:14px;">Platform Users</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                        <h4 style="margin:0;padding:0;color:#31333F;">Success Rate</h4>
                        <h2 style="margin:0;padding:10px 0;color:#2ca02c;font-size:28px;">{success_rate:.1f}%</h2>
                        <p style="margin:0;color:#666;font-size:14px;">Search Success</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;margin:5px;">
                        <h4 style="margin:0;padding:0;color:#31333F;">Total Experts</h4>
                        <h2 style="margin:0;padding:10px 0;color:#9467bd;font-size:28px;">{total_experts:,}</h2>
                        <p style="margin:0;color:#666;font-size:14px;">Registered Experts</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Add separator
            st.markdown("<hr style='margin-top:15px;margin-bottom:15px;'>", unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Error displaying quick stats: {str(e)}")
            # Don't show error to user for this non-critical component

    def display_analytics(self, analytics_type, start_date, end_date, filters):
        """Display analytics based on selected type and filters."""
        # Format dates for query if they're datetime objects
        start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59' if isinstance(end_date, datetime) else end_date
        
        # Show loading spinner while fetching data
        with st.spinner(f"Loading {analytics_type} analytics..."):
            # Display specific analytics based on selection
            analytics_map = {
                "Overview": (get_overview_metrics, display_overview_analytics),
                "Chat": (get_chat_and_search_metrics, display_chat_analytics),
                "Search": (get_search_metrics, display_search_analytics),
                "Expert": (get_expert_metrics, display_expert_analytics),
                "Content": (get_content_metrics, display_content_analytics),
                "Resources": (get_resource_metrics, display_resource_analytics),
            }
            
            if analytics_type in analytics_map:
                get_metrics, display_analytics = analytics_map[analytics_type]
                
                try:
                    # Get metrics with appropriate filters
                    if analytics_type == "Expert":
                        metrics = get_metrics(
                            self.conn, 
                            start_date_str, 
                            end_date_str, 
                            filters.get('expert_count', 20)
                        )
                    else:
                        metrics = get_metrics(self.conn, start_date_str, end_date_str)
                    
                    # Check if we have data to display
                    if isinstance(metrics, pd.DataFrame) and metrics.empty:
                        st.warning(f"No {analytics_type.lower()} data available for the selected period.")
                        return
                    
                    # Display analytics based on the type
                    display_analytics(metrics, filters)
                
                except Exception as e:
                    self.logger.error(f"Error in display_analytics for {analytics_type}: {str(e)}")
                    st.error(f"An error occurred while processing {analytics_type} analytics: {str(e)}")
            else:
                st.error(f"Unknown analytics type: {analytics_type}")

    def display_header(self, analytics_type):
        """Display the dashboard header with current analytics type."""
        # Create a header with logo and title
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Logo placeholder - replace with actual logo if available
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <div style="background-color: #3366cc; color: white; width: 80px; height: 80px; 
                         border-radius: 50%; display: flex; justify-content: center; align-items: center; 
                         font-size: 24px; font-weight: bold;">
                        APHRC
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            # Title and description
            st.markdown(f"# APHRC Analytics Dashboard")
            st.markdown(f"#### {analytics_type} Analytics")
        
        # Add a horizontal line
        st.markdown("<hr>", unsafe_allow_html=True)

    def display_footer(self):
        """Display the dashboard footer with helpful information."""
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### About This Dashboard")
            st.markdown("""
            This analytics dashboard provides insights into the APHRC platform's 
            performance, user engagement, and content metrics.
            """)
        
        with col2:
            st.markdown("### Data Sources")
            st.markdown("""
            - Chat interactions and sessions
            - Search analytics and sessions
            - Expert profiles and matching data
            - Resource and content metrics
            """)
        
        with col3:
            st.markdown("### Last Updated")
            st.markdown(f"**Data refreshed:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add refresh button
            if st.button("Refresh Data Now"):
                st.session_state.last_refresh = datetime.now()
                st.experimental_rerun()
        
        # Add copyright footer
        st.markdown(
            f"""
            <div style="
                width: 100%;
                text-align: center;
                padding: 10px;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            ">
                APHRC Analytics Dashboard • {datetime.now().year} • Built with Streamlit
            </div>
            """,
            unsafe_allow_html=True
        )

# Execute the app
if __name__ == "__main__":
    dashboard = UnifiedAnalyticsDashboard()
    dashboard.main()