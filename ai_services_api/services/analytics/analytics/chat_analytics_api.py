import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_chat_and_search_metrics(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve chat and search metrics from the database.
    
    Args:
        conn: Database connection
        start_date: Start date for metrics query
        end_date: End date for metrics query
    
    Returns:
        DataFrame containing chat and search metrics
    """
    cursor = conn.cursor()
    try:
        # First check if both tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chatbot_logs'
            ) AS chatbot_logs_exists,
            EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_sessions'
            ) AS chat_sessions_exists
        """)
        
        table_check = cursor.fetchone()
        chatbot_logs_exists, chat_sessions_exists = table_check[0], table_check[1]
        
        if not chatbot_logs_exists and not chat_sessions_exists:
            logger.warning("Neither chatbot_logs nor chat_sessions tables exist")
            return pd.DataFrame()
        
        # Construct query based on existing tables
        if chatbot_logs_exists and chat_sessions_exists:
            # Combined query for both tables
            query = """
            WITH daily_chat_metrics AS (
                SELECT 
                    DATE(cl.timestamp) as date,
                    COUNT(*) as total_chats,
                    COUNT(DISTINCT cl.user_id) as unique_users,
                    AVG(cl.response_time) as avg_response_time
                FROM chatbot_logs cl
                WHERE cl.timestamp BETWEEN %s AND %s
                GROUP BY DATE(cl.timestamp)
            ),
            daily_session_metrics AS (
                SELECT 
                    DATE(cs.start_timestamp) as date,
                    COUNT(*) as total_sessions,
                    AVG(cs.total_messages) as avg_messages_per_session,
                    SUM(CASE WHEN cs.successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate
                FROM chat_sessions cs
                WHERE cs.start_timestamp BETWEEN %s AND %s
                GROUP BY DATE(cs.start_timestamp)
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
            """
            
            cursor.execute(query, (start_date, end_date, start_date, end_date))
        
        elif chatbot_logs_exists:
            # Only chatbot_logs exists
            query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_chats,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(response_time) as avg_response_time,
                0 as total_sessions,
                0 as avg_messages_per_session,
                0 as session_success_rate
            FROM chatbot_logs
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY DATE(timestamp)
            ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        
        elif chat_sessions_exists:
            # Only chat_sessions exists
            query = """
            SELECT 
                DATE(start_timestamp) as date,
                0 as total_chats,
                COUNT(DISTINCT user_id) as unique_users,
                0 as avg_response_time,
                COUNT(*) as total_sessions,
                AVG(total_messages) as avg_messages_per_session,
                SUM(CASE WHEN successful THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as session_success_rate
            FROM chat_sessions
            WHERE start_timestamp BETWEEN %s AND %s
            GROUP BY DATE(start_timestamp)
            ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date))
        
        # Fetch columns and data
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        # Convert to DataFrame
        return pd.DataFrame(data, columns=columns)
    
    except Exception as e:
        logger.error(f"Error retrieving chat metrics: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

def process_chat_metrics(conn, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Process and prepare chat metrics for API response
    
    Args:
        conn: Database connection
        start_date: Start date for metrics query
        end_date: End date for metrics query
    
    Returns:
        Dictionary with processed metrics
    """
    try:
        # Retrieve metrics DataFrame
        metrics_df = get_chat_and_search_metrics(conn, start_date, end_date)
        
        # Calculate summary statistics
        summary = {}
        
        # Total chats
        if 'total_chats' in metrics_df.columns:
            summary['total_chats'] = int(metrics_df['total_chats'].sum())
        
        # Unique users
        if 'unique_users' in metrics_df.columns:
            summary['unique_users'] = int(metrics_df['unique_users'].sum())
        
        # Average response time
        if 'avg_response_time' in metrics_df.columns:
            summary['avg_response_time'] = float(metrics_df['avg_response_time'].mean())
        
        # Total sessions
        if 'total_sessions' in metrics_df.columns:
            summary['total_sessions'] = int(metrics_df['total_sessions'].sum())
        
        # Session success rate
        if 'session_success_rate' in metrics_df.columns:
            summary['avg_session_success_rate'] = float(metrics_df['session_success_rate'].mean())
        
        return {
            "metrics": metrics_df.to_dict(orient="records"),
            "summary": summary
        }
    
    except Exception as e:
        logger.error(f"Error processing chat metrics: {e}")
        return {"metrics": [], "summary": {}}