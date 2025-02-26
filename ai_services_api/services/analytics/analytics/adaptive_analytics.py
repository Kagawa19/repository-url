import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, Optional, Any
from neo4j import GraphDatabase

# Comprehensive Neo4j Queries
DAILY_METRICS_QUERY = """
MATCH (e1:Expert)-[r:RECOMMENDED]->(e2:Expert)
WHERE r.timestamp >= datetime($start_date) 
  AND r.timestamp <= datetime($end_date)
WITH date(r.timestamp) as date, r, e1, e2
OPTIONAL MATCH (e1)-[:HAS_CONCEPT]->(c)<-[:HAS_CONCEPT]-(e2)
WITH date, r, e1, e2, COUNT(DISTINCT c) as concept_overlap
RETURN 
    date,
    COUNT(r) as total_recommendations,
    COUNT(DISTINCT e1.id) as unique_requesters,
    COUNT(DISTINCT e2.id) as unique_experts_recommended,
    AVG(COALESCE(r.similarity_score, 0)) as avg_similarity_score,
    AVG(concept_overlap) as avg_shared_concepts,
    CASE 
        WHEN COUNT(r) > 0 
        THEN toFloat(COUNT(CASE WHEN COALESCE(r.similarity_score, 0) >= 0.7 THEN 1 END)) / COUNT(r) 
        ELSE 0 
    END as success_rate
ORDER BY date
"""

DOMAIN_METRICS_QUERY = """
MATCH (e1:Expert)-[r:RECOMMENDED]->(e2:Expert)
WHERE r.timestamp >= datetime($start_date) 
  AND r.timestamp <= datetime($end_date)
MATCH (e2)-[:HAS_CONCEPT]->(c:Concept)
WITH c.name as domain, 
     COUNT(r) as recommendations,
     AVG(r.similarity_score) as avg_similarity,
     toFloat(COUNT(CASE WHEN r.similarity_score >= 0.7 THEN 1 END)) / COUNT(r) as success_rate
RETURN 
    domain,
    recommendations,
    avg_similarity,
    success_rate
ORDER BY recommendations DESC
LIMIT 10
"""

def get_neo4j_driver():
    """Create a connection to Neo4j database."""
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        return driver
    except Exception as e:
        logging.error(f"Error connecting to Neo4j: {e}")
        raise

def get_adaptive_metrics(conn, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Retrieve adaptive metrics with comprehensive error handling"""
    driver = get_neo4j_driver()
    if not driver:
        st.error("Failed to obtain Neo4j driver")
        return {}
    
    try:
        with driver.session() as session:
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
            
            metrics_queries = [
                ('daily_metrics', DAILY_METRICS_QUERY),
                ('domain_metrics', DOMAIN_METRICS_QUERY),
            ]
            
            results = {}
            for name, query in metrics_queries:
                try:
                    query_result = session.run(query, params)
                    data = [dict(record) for record in query_result]
                    
                    if data:
                        df = pd.DataFrame(data)
                        results[name] = df
                        st.write(f"Retrieved {len(df)} {name} records")
                    else:
                        st.warning(f"No data retrieved for {name}")
                
                except Exception as query_error:
                    st.error(f"Error in {name} query: {query_error}")
            
            return results
    
    except Exception as e:
        st.error(f"Metrics retrieval error: {e}")
        return {}
    finally:
        if driver:
            driver.close()

def display_adaptive_analytics(metrics: Dict[str, pd.DataFrame], filters: Optional[Dict[str, Any]] = None):
    """Streamlit dashboard for adaptive analytics"""
    st.title("Expert Recommendation Analytics")
    
    if not metrics:
        st.error("No recommendation data available. Check database connection.")
        return
    
    daily_data = metrics.get('daily_metrics')
    domain_data = metrics.get('domain_metrics')
    
    if daily_data is None or daily_data.empty:
        st.warning("No daily metrics data found.")
        return
    
    # Overview Metrics
    st.header("Recommendation Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_recs = daily_data['total_recommendations'].sum()
        st.metric("Total Recommendations", f"{total_recs:,}")
    
    with col2:
        unique_experts = daily_data['unique_experts_recommended'].sum()
        st.metric("Unique Experts", f"{unique_experts:,}")
    
    with col3:
        avg_success = daily_data['success_rate'].mean()
        st.metric("Avg Success Rate", f"{avg_success:.2%}")
    
    # Daily Recommendations Chart
    st.header("Daily Recommendation Trends")
    fig_daily = px.line(
        daily_data, 
        x='date', 
        y='total_recommendations', 
        title='Daily Recommendation Volume'
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Domain Performance
    if domain_data is not None and not domain_data.empty:
        st.header("Domain Performance")
        fig_domain = px.bar(
            domain_data, 
            x='domain', 
            y='recommendations', 
            color='success_rate',
            title='Recommendations by Domain'
        )
        st.plotly_chart(fig_domain, use_container_width=True)
        
        st.dataframe(domain_data.style.format({
            'recommendations': '{:,.0f}',
            'avg_similarity': '{:.2f}',
            'success_rate': '{:.2%}'
        }))

def main():
    st.set_page_config(
        page_title="Expert Recommendation Analytics", 
        layout="wide"
    )
    
    # Date range selection
    st.sidebar.header("Analytics Period")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=datetime.now() - timedelta(days=90)
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        value=datetime.now()
    )
    
    # Retrieve metrics
    conn = None  # Placeholder for connection
    metrics = get_adaptive_metrics(
        conn, 
        datetime.combine(start_date, datetime.min.time()), 
        datetime.combine(end_date, datetime.max.time())
    )
    
    # Display analytics
    display_adaptive_analytics(metrics)

if __name__ == "__main__":
    main()