import os
import psycopg2
import pandas as pd
from ai_services_api.services.analytics.utils.db_utils import DatabaseConnector
from fastapi import APIRouter, Query
from typing import Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel

# Configure logging
import logging
logger = logging.getLogger("router")

# Create router for analytics endpoints
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Helper function to convert DataFrame to dict
def dataframe_to_dict(df):
    """Convert DataFrame to dictionary suitable for JSON"""
    if df is None:
        return []
    
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return []
            
        if 'date' in df.columns:
            # Convert date columns to strings for JSON serialization
            df = df.copy()
            df['date'] = df['date'].astype(str)
        return df.to_dict(orient="records")
    elif isinstance(df, dict):
        # Handle dictionary of dataframes
        result = {}
        for key, value in df.items():
            if isinstance(value, pd.DataFrame):
                result[key] = dataframe_to_dict(value)
            else:
                result[key] = value
        return result
    return []

# Define response models for analytics
class AnalyticsResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

# Apply granularity transformation
def apply_granularity(df, granularity):
    """Apply weekly or monthly grouping to daily metrics"""
    if df is None or df.empty:
        return df
    
    if 'date' not in df.columns:
        return df
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Identify count vs. average columns
    count_cols = [col for col in df.columns if col.startswith('total_') or 
                  col.endswith('_count') or col == 'unique_users' or 
                  'matches' in col or 'searches' in col or
                  'messages' in col or 'experts' in col or
                  'interactions' in col or 'sessions' in col]
    avg_cols = [col for col in df.columns if col not in count_cols]
    
    # Create aggregations based on granularity
    if granularity == "weekly":
        freq = 'W'
    else: # monthly
        freq = 'M'
        
    # Create aggregations
    agg_counts = df[count_cols].resample(freq).sum() if count_cols else pd.DataFrame()
    agg_avgs = df[avg_cols].resample(freq).mean() if avg_cols else pd.DataFrame()
    
    # Combine the aggregations
    if not agg_counts.empty and not agg_avgs.empty:
        result_df = pd.concat([agg_counts, agg_avgs], axis=1)
    elif not agg_counts.empty:
        result_df = agg_counts
    elif not agg_avgs.empty:
        result_df = agg_avgs
    else:
        return df
        
    return result_df.reset_index()

# Calculate summary statistics
def calculate_summary(df):
    """Calculate summary statistics for a metrics DataFrame"""
    if df is None or df.empty:
        return {}
        
    summary = {}
    
    # Add total values for count columns
    for col in df.columns:
        if (col.startswith('total_') or col.endswith('_count') or 
            'searches' in col or 'interactions' in col or 
            'messages' in col or 'experts' in col or
            'sessions' in col) and col != 'date':
            try:
                summary[f"total_{col}"] = float(df[col].sum())
            except:
                pass
    
    # Add averages for rate/time columns
    for col in df.columns:
        if ('rate' in col or 'time' in col or 'avg_' in col or 
            'score' in col or 'quality' in col or 
            'similarity' in col) and col != 'date':
            try:
                summary[f"avg_{col}"] = float(df[col].mean())
            except:
                pass
    
    return summary

# Database connection utility
def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'aphrc'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'p0stgres')
    )

@router.get("/chat", response_model=AnalyticsResponse)
def get_chat_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    granularity: str = Query("daily", description="Data granularity: daily, weekly, monthly")
):
    try:
        # Import the metrics function
        from ai_services_api.services.analytics.analytics.chat_analytics_api import get_chat_and_search_metrics
        
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with get_db_connection() as conn:
            # Get metrics data
            metrics_df = get_chat_and_search_metrics(conn, start_date_str, end_date_str)
            
            # Apply granularity transformation if needed
            if granularity in ["weekly", "monthly"] and metrics_df is not None and not metrics_df.empty:
                metrics_df = apply_granularity(metrics_df, granularity)
            
            # Convert to dictionary/JSON format
            result = {
                "metrics": dataframe_to_dict(metrics_df),
                "summary": calculate_summary(metrics_df),
                "period": {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "granularity": granularity
                }
            }
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting chat analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Search analytics endpoint
@router.get("/search", response_model=AnalyticsResponse)
async def get_search_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    granularity: str = Query("daily", description="Data granularity: daily, weekly, monthly"),
    search_type: Optional[str] = Query(None, description="Filter by search type: expert, publication, general")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            from ai_services_api.services.analytics.analytics.search_analytics import get_search_metrics
            
            # Get metrics data
            metrics = get_search_metrics(conn, start_date_str, end_date_str)
            
            # Process each DataFrame in the metrics dictionary
            result = {}
            for key, df in metrics.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if key == 'daily_metrics' and granularity in ["weekly", "monthly"]:
                        df = apply_granularity(df, granularity)
                        
                    # Apply search type filter if provided and column exists
                    if search_type and key == 'daily_metrics' and f"{search_type}_searches" in df.columns:
                        # Filter or highlight specific search type data
                        pass
                        
                    result[key] = dataframe_to_dict(df)
                else:
                    result[key] = []
            
            # Calculate summary for daily metrics
            if 'daily_metrics' in metrics and not metrics['daily_metrics'].empty:
                result["summary"] = calculate_summary(metrics['daily_metrics'])
            
            # Add period info
            result["period"] = {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "granularity": granularity
            }
            
            # If filter applied, add to response
            if search_type:
                result["filter"] = {"search_type": search_type}
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting search analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Expert analytics endpoint
@router.get("/expert", response_model=AnalyticsResponse)
async def get_expert_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    expert_count: int = Query(20, description="Number of experts to include"),
    domain: Optional[str] = Query(None, description="Filter by domain")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            from ai_services_api.services.analytics.analytics.expert_analytics import get_expert_metrics
            
            # Get metrics data with expert count parameter
            metrics = get_expert_metrics(conn, start_date_str, end_date_str, expert_count)
            
            # Process each DataFrame in the metrics dictionary
            result = {}
            for key, df in metrics.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Apply domain filter if provided and applicable
                    if domain and key == 'experts' and 'domains' in df.columns:
                        # This depends on how domains are stored (array, string, etc.)
                        # Example implementation if domains is a string column with comma-separated values
                        try:
                            df = df[df['domains'].str.contains(domain, case=False, na=False)]
                        except:
                            # If filtering fails, log but continue without filtering
                            logger.warning(f"Could not filter by domain: {domain}")
                    
                    result[key] = dataframe_to_dict(df)
                else:
                    result[key] = []
            
            # Calculate summary statistics
            summary = {}
            if 'experts' in metrics and not metrics['experts'].empty:
                summary["total_experts"] = len(metrics['experts'])
                
            if 'expert_search_matches' in metrics and not metrics['expert_search_matches'].empty:
                summary["total_matches"] = int(metrics['expert_search_matches']['match_count'].sum())
                summary["avg_similarity"] = float(metrics['expert_search_matches']['avg_similarity'].mean())
                
            if 'expert_messages' in metrics and not metrics['expert_messages'].empty:
                summary["total_messages"] = int(metrics['expert_messages']['sent_count'].sum())
                
            result["summary"] = summary
            
            # Add period info
            result["period"] = {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "expert_count": expert_count
            }
            
            # If filter applied, add to response
            if domain:
                result["filter"] = {"domain": domain}
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting expert analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Content analytics endpoint
@router.get("/content", response_model=AnalyticsResponse)
async def get_content_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            from ai_services_api.services.analytics.analytics.content_analytics import get_content_metrics
            
            # Get metrics data
            metrics = get_content_metrics(conn, start_date_str, end_date_str)
            
            # Process the metrics dictionary
            result = {}
            for key, df in metrics.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    result[key] = dataframe_to_dict(df)
                else:
                    result[key] = []
            
            # Calculate summary statistics
            summary = {}
            
            # Add resource counts if available
            if 'resource_metrics' in metrics and not metrics['resource_metrics'].empty:
                if 'total_resources' in metrics['resource_metrics'].columns:
                    summary["total_resources"] = int(metrics['resource_metrics']['total_resources'].sum())
            
            # Add expert counts if available
            if 'expert_metrics' in metrics and not metrics['expert_metrics'].empty:
                if 'total_experts' in metrics['expert_metrics'].columns:
                    summary["total_experts"] = int(metrics['expert_metrics']['total_experts'].iloc[0])
            
            # Add message counts if available
            if 'message_metrics' in metrics and not metrics['message_metrics'].empty:
                if 'total_messages' in metrics['message_metrics'].columns:
                    summary["total_messages"] = int(metrics['message_metrics']['total_messages'].iloc[0])
            
            result["summary"] = summary
            
            # Add period info
            result["period"] = {
                "start_date": start_date_str,
                "end_date": end_date_str
            }
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting content analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Overview analytics endpoint
@router.get("/overview", response_model=AnalyticsResponse)
async def get_overview_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    granularity: str = Query("daily", description="Data granularity: daily, weekly, monthly")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            from ai_services_api.services.analytics.analytics.overview_analytics import get_overview_metrics
            
            # Get metrics data
            metrics_df = get_overview_metrics(conn, start_date_str, end_date_str)
            
            # Apply granularity transformation if needed
            if granularity in ["weekly", "monthly"] and not metrics_df.empty:
                metrics_df = apply_granularity(metrics_df, granularity)
            
            # Calculate quick stats for the overview
            quick_stats = {}
            if not metrics_df.empty:
                if 'total_interactions' in metrics_df.columns:
                    quick_stats['total_interactions'] = int(metrics_df['total_interactions'].sum())
                if 'unique_users' in metrics_df.columns:
                    quick_stats['unique_users'] = int(metrics_df['unique_users'].sum())
                if 'success_rate' in metrics_df.columns:
                    quick_stats['avg_success_rate'] = float(metrics_df['success_rate'].mean())
                if 'avg_quality_score' in metrics_df.columns:
                    quick_stats['avg_quality_score'] = float(metrics_df['avg_quality_score'].mean())
                if 'total_searches' in metrics_df.columns:
                    quick_stats['total_searches'] = int(metrics_df['total_searches'].sum())
                if 'avg_response_time' in metrics_df.columns:
                    quick_stats['avg_response_time'] = float(metrics_df['avg_response_time'].mean())
            
            # Convert to dictionary/JSON format
            result = {
                "metrics": dataframe_to_dict(metrics_df),
                "summary": calculate_summary(metrics_df),
                "quick_stats": quick_stats,
                "period": {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "granularity": granularity
                }
            }
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting overview analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Resource analytics endpoint
@router.get("/resources", response_model=AnalyticsResponse)
async def get_resources_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            from ai_services_api.services.analytics.analytics.resource_analytics import get_resource_metrics
            
            # Get metrics data
            metrics = get_resource_metrics(conn, start_date_str, end_date_str)
            
            # Process each DataFrame in the metrics dictionary
            result = {}
            for key, df in metrics.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Apply resource type filter if provided and applicable
                    if resource_type and key == 'resource_metrics' and 'resource_type' in df.columns:
                        df = df[df['resource_type'] == resource_type]
                    
                    result[key] = dataframe_to_dict(df)
                else:
                    result[key] = []
            
            # Calculate summary statistics
            summary = {}
            
            # Add key metrics to summary from each dataframe
            if 'summary' in metrics and not metrics['summary'].empty:
                try:
                    # If summary metrics are already calculated, use them
                    for _, row in metrics['summary'].iterrows():
                        summary[row['Metric']] = row['Value']
                except:
                    pass
            
            # Add resource counts if available
            if 'resource_counts' in metrics and not metrics['resource_counts'].empty:
                try:
                    for _, row in metrics['resource_counts'].iterrows():
                        summary[f"total_{row['Resource'].lower()}"] = row['Count']
                except:
                    pass
            
            # Add API usage if available
            if 'api_usage' in metrics and not metrics['api_usage'].empty:
                try:
                    summary['total_api_requests'] = int(metrics['api_usage']['request_count'].sum())
                except:
                    pass
            
            result["summary"] = summary
            
            # Add period info
            result["period"] = {
                "start_date": start_date_str,
                "end_date": end_date_str
            }
            
            # If filter applied, add to response
            if resource_type:
                result["filter"] = {"resource_type": resource_type}
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting resource analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))

# Endpoint for all analytics data combined (for dashboards)
@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard_analytics(
    start_date: date = Query(..., description="Start date for analytics"),
    end_date: date = Query(..., description="End date for analytics"),
    granularity: str = Query("daily", description="Data granularity: daily, weekly, monthly")
):
    try:
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d') + ' 23:59:59'
        
        # Get DB connection
        with DatabaseConnector.get_connection() as conn:
            # Import necessary functions
            from ai_services_api.services.analytics.analytics.overview_analytics import get_overview_metrics
            from ai_services_api.services.analytics.analytics.chat_analytics import get_chat_and_search_metrics
            from ai_services_api.services.analytics.analytics.search_analytics import get_search_metrics
            
            # Get metrics from different modules
            overview_df = get_overview_metrics(conn, start_date_str, end_date_str)
            chat_df = get_chat_and_search_metrics(conn, start_date_str, end_date_str)
            search_metrics = get_search_metrics(conn, start_date_str, end_date_str)
            
            # Apply granularity transformations if needed
            if granularity in ["weekly", "monthly"]:
                if not overview_df.empty:
                    overview_df = apply_granularity(overview_df, granularity)
                if not chat_df.empty:
                    chat_df = apply_granularity(chat_df, granularity)
                if 'daily_metrics' in search_metrics and not search_metrics['daily_metrics'].empty:
                    search_metrics['daily_metrics'] = apply_granularity(search_metrics['daily_metrics'], granularity)
            
            # Combine results into a comprehensive dashboard dataset
            result = {
                "overview": {
                    "metrics": dataframe_to_dict(overview_df),
                    "summary": calculate_summary(overview_df)
                },
                "chat": {
                    "metrics": dataframe_to_dict(chat_df),
                    "summary": calculate_summary(chat_df)
                },
                "search": dataframe_to_dict(search_metrics),
                "period": {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "granularity": granularity
                }
            }
            
            # Calculate combined quick stats
            quick_stats = {}
            
            # From overview
            if not overview_df.empty:
                if 'total_interactions' in overview_df.columns:
                    quick_stats['total_interactions'] = int(overview_df['total_interactions'].sum())
                if 'unique_users' in overview_df.columns:
                    quick_stats['unique_users'] = int(overview_df['unique_users'].sum())
            
            # From chat
            if not chat_df.empty:
                if 'total_sessions' in chat_df.columns:
                    quick_stats['total_sessions'] = int(chat_df['total_sessions'].sum())
            
            # From search
            if 'daily_metrics' in search_metrics and not search_metrics['daily_metrics'].empty:
                search_df = search_metrics['daily_metrics']
                if 'total_searches' in search_df.columns:
                    quick_stats['total_searches'] = int(search_df['total_searches'].sum())
            
            result["quick_stats"] = quick_stats
            
            return AnalyticsResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {str(e)}")
        return AnalyticsResponse(success=False, data={}, message=str(e))