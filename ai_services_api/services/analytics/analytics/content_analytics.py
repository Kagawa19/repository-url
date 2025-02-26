import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import sys
import traceback
import time
import logging
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, date
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('analytics_dashboard')

def safe_json_loads(value: str) -> dict:
   try:
       return json.loads(value) if value else {}
   except (json.JSONDecodeError, TypeError) as e:
       logger.warning(f"Failed to parse JSON: {str(e)}")
       return {}

def safe_aggregate(df: pd.DataFrame, column: str, operation: str = 'sum') -> float:
   try:
       if df.empty or column not in df.columns:
           return 0
       if operation == 'sum':
           return df[column].sum()
       elif operation == 'count':
           return df[column].count()
       return 0
   except Exception as e:
       logger.warning(f"Failed to aggregate column {column}: {str(e)}")
       return 0

def get_content_metrics(conn, start_date, end_date, page_size=10, offset=0):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            WITH ResourceMetrics AS (
                SELECT 
                    r.type as resource_type,
                    r.source,
                    COUNT(DISTINCT r.id) as total_resources
                FROM resources_resource r
                GROUP BY r.type, r.source
            ),
            ExpertStats AS (
                SELECT 
                    ei.sender_id,
                    COUNT(*) as interactions,
                    COUNT(DISTINCT receiver_id) as unique_receivers 
                FROM expert_interactions ei
                WHERE ei.created_at BETWEEN %s AND %s
                GROUP BY ei.sender_id
            ),
            MessageMetrics AS (
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN draft THEN 1 END) as draft_count,
                    AVG(LENGTH(content)) as avg_length
                FROM expert_messages em
                WHERE em.created_at BETWEEN %s AND %s
            ),
            ContentMetrics AS (
                SELECT 
                    em.id,
                    em.content,
                    LENGTH(em.content) as content_length,
                    em.created_at,
                    em.sender_id,
                    em.receiver_id,
                    em.draft,
                    i.metrics->>'response_time' as response_time,
                    CAST(i.metrics->>'error_occurred' AS BOOLEAN) as had_error
                FROM expert_messages em
                LEFT JOIN interactions i ON i.id = em.id 
                WHERE em.created_at BETWEEN %s AND %s
                ORDER BY em.created_at DESC
                LIMIT %s OFFSET %s
            )
            SELECT json_build_object(
                'resource_metrics', COALESCE((SELECT json_agg(row_to_json(ResourceMetrics)) FROM ResourceMetrics), '[]'::json),
                'expert_stats', COALESCE((SELECT json_agg(row_to_json(ExpertStats)) FROM ExpertStats), '[]'::json), 
                'message_metrics', COALESCE((SELECT json_agg(row_to_json(MessageMetrics)) FROM MessageMetrics), '[]'::json),
                'content_metrics', COALESCE((SELECT json_agg(row_to_json(ContentMetrics)) FROM ContentMetrics), '[]'::json)
            ) as metrics;
        """, (start_date, end_date, start_date, end_date, start_date, end_date, page_size, offset))

        result = cursor.fetchone()[0]
        metrics = {}
        for key in ['resource_metrics', 'expert_stats', 'message_metrics', 'content_metrics']:
            try:
                df = pd.DataFrame(result.get(key, []))
                metrics[key] = df
            except Exception as e:
                logger.error(f"Error processing {key}: {str(e)}")
                metrics[key] = pd.DataFrame()
                
        return metrics

    finally:
        cursor.close()

def create_visualization(data, viz_type, x, y, title, **kwargs):
   try:
       if data.empty:
           return None
           
       if viz_type == 'bar':
           fig = px.bar(data, x=x, y=y, title=title, **kwargs)
       elif viz_type == 'pie':
           fig = px.pie(data, values=y, names=x, title=title, **kwargs)
       elif viz_type == 'scatter':
           fig = px.scatter(data, x=x, y=y, title=title, **kwargs)
       elif viz_type == 'line':
           fig = px.line(data, x=x, y=y, title=title, **kwargs)
       else:
           return None
           
       return fig
   except Exception as e:
       logger.error(f"Visualization error: {str(e)}")
       return None

def display_content_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict] = None):
   try:
       if not isinstance(metrics, dict) or all(df.empty for df in metrics.values()):
           st.warning("No data available for the selected filters")
           return

       st.subheader("Content Analytics")

       # Overview metrics
       cols = st.columns(4)
       with cols[0]:
           total_messages = safe_aggregate(metrics['message_metrics'], 'total_messages')
           st.metric("Total Messages", f"{total_messages:,}")
       with cols[1]:
           total_experts = safe_aggregate(metrics['expert_stats'], 'unique_receivers')
           st.metric("Total Experts", f"{total_experts:,}")
       with cols[2]:
           avg_length = safe_aggregate(metrics['message_metrics'], 'avg_length')
           st.metric("Avg Message Length", f"{avg_length:.0f}")
       with cols[3]:
           draft_pct = (safe_aggregate(metrics['message_metrics'], 'draft_count') / total_messages) * 100 if total_messages else 0
           st.metric("Draft %", f"{draft_pct:.1f}%")

       # Resource distribution
       if not metrics['resource_metrics'].empty:
           cols = st.columns(2)
           with cols[0]:
               fig = create_visualization(
                   metrics['resource_metrics'], 
                   'pie',
                   'source',
                   'total_resources',
                   'Content Sources'
               )
               if fig:
                   st.plotly_chart(fig, use_container_width=True)
                   
           with cols[1]:
               fig = create_visualization(
                   metrics['resource_metrics'],
                   'pie', 
                   'resource_type',
                   'total_resources',
                   'Content Types'
               )
               if fig:
                   st.plotly_chart(fig, use_container_width=True)

       # Performance metrics
       if not metrics['content_metrics'].empty:
           st.subheader("Message Performance")
           
           fig = make_subplots(rows=2, cols=2)
           content_data = metrics['content_metrics']
           
           fig.add_trace(
               go.Histogram(x=content_data['response_time'], name="Response Times"),
               row=1, col=1
           )
           
           fig.add_trace(
               go.Histogram(x=content_data['content_length'], name="Message Length"),
               row=1, col=2
           )
           
           draft_counts = content_data['draft'].value_counts()
           fig.add_trace(
               go.Pie(values=draft_counts.values, labels=draft_counts.index, name="Draft Status"),
               row=2, col=1
           )
           
           error_rate = (content_data['had_error'].sum() / len(content_data)) * 100
           fig.add_trace(
               go.Indicator(
                   mode="gauge+number",
                   value=error_rate,
                   title={'text': "Error Rate (%)"},
                   gauge={'axis': {'range': [0, 100]}},
               ),
               row=2, col=2
           )
           
           fig.update_layout(height=800, title_text="Message Metrics")
           st.plotly_chart(fig, use_container_width=True)

       # Details tables
       with st.expander("Detailed Metrics"):
           tabs = st.tabs(["Resources", "Messages", "Performance"])
           
           with tabs[0]:
               if not metrics['resource_metrics'].empty:
                   st.dataframe(metrics['resource_metrics'])
                   
           with tabs[1]:
               if not metrics['message_metrics'].empty:
                   st.dataframe(metrics['message_metrics'])
                   
           with tabs[2]:
               if not metrics['content_metrics'].empty:
                   display_cols = ['content_length', 'response_time', 'draft', 'had_error']
                   st.dataframe(metrics['content_metrics'][display_cols])

   except Exception as e:
       logger.error(f"Display error: {str(e)}")
       st.error("Error displaying analytics")