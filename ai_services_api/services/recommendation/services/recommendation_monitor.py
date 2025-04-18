import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recommendation_monitoring.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RecommendationMonitor:
    """
    Monitors the adaptiveness and performance of the recommendation system
    by tracking key metrics over time.
    """
    
    def __init__(self):
        """Initialize the RecommendationMonitor with database connection info"""
        self.logger = logging.getLogger(__name__)
        
        # Define connection parameters
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            parsed_url = urlparse(database_url)
            self.conn_params = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        else:
            self.conn_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }
        
        self.logger.info("RecommendationMonitor initialized successfully")
        
        # Create the recommendation_metrics table if it doesn't exist
        self._ensure_metrics_table_exists()
        
    def _ensure_metrics_table_exists(self):
        """Create the necessary tables for monitoring if they don't exist."""
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                # Create recommendation_metrics table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metric_type VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    user_id VARCHAR(255) NULL,
                    expert_id VARCHAR(255) NULL,
                    details JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_timestamp ON recommendation_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_type ON recommendation_metrics(metric_type);
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_user ON recommendation_metrics(user_id);
                """)
                
                # Create recommendation_feedback table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_feedback (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255) NOT NULL,
                    expert_id VARCHAR(255) NOT NULL,
                    recommendation_method VARCHAR(100) NOT NULL,
                    interaction_type VARCHAR(100) NOT NULL,
                    feedback_score FLOAT NULL,
                    details JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_timestamp ON recommendation_feedback(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_user ON recommendation_feedback(user_id);
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_expert ON recommendation_feedback(expert_id);
                """)
                
                # Create recommendation_snapshot table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_snapshot (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255) NOT NULL,
                    recommendations JSONB NOT NULL,
                    weights JSONB NULL,
                    features_used JSONB NULL,
                    data_sources_used JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_snapshot_timestamp ON recommendation_snapshot(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_snapshot_user ON recommendation_snapshot(user_id);
                """)
                
                conn.commit()
                self.logger.info("Recommendation monitoring tables created or already exist")
        except Exception as e:
            self.logger.error(f"Error creating monitoring tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_metric(self, metric_type: str, metric_name: str, metric_value: float, 
                     user_id: Optional[str] = None, expert_id: Optional[str] = None, 
                     details: Optional[Dict] = None):
        """
        Record a metric for monitoring recommendation system performance.
        
        Args:
            metric_type: Category of metric (e.g., 'adaptation', 'performance', 'diversity')
            metric_name: Specific metric name (e.g., 'click_through_rate', 'domain_coverage')
            metric_value: Numerical value of the metric
            user_id: Optional user ID if metric is user-specific
            expert_id: Optional expert ID if metric is expert-specific
            details: Optional dictionary with additional details
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_metrics
                    (timestamp, metric_type, metric_name, metric_value, user_id, expert_id, details)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    metric_type, 
                    metric_name, 
                    metric_value, 
                    user_id, 
                    expert_id, 
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded metric: {metric_type}.{metric_name} = {metric_value}")
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_feedback(self, user_id: str, expert_id: str, recommendation_method: str,
                       interaction_type: str, feedback_score: Optional[float] = None,
                       details: Optional[Dict] = None):
        """
        Record user feedback on recommendations for adaptation analysis.
        
        Args:
            user_id: The user who received the recommendation
            expert_id: The expert who was recommended
            recommendation_method: Method used to generate recommendation (e.g., 'messaging', 'semantic')
            interaction_type: Type of interaction (e.g., 'click', 'message', 'ignore')
            feedback_score: Optional numerical score (-1 to 1)
            details: Optional dictionary with additional details
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_feedback
                    (timestamp, user_id, expert_id, recommendation_method, interaction_type, feedback_score, details)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    expert_id,
                    recommendation_method,
                    interaction_type,
                    feedback_score,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded feedback from user {user_id} for expert {expert_id}")
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_recommendation_snapshot(self, user_id: str, recommendations: List[Dict],
                                     weights: Optional[Dict] = None,
                                     features_used: Optional[List[str]] = None,
                                     data_sources_used: Optional[List[str]] = None):
        """
        Record a snapshot of recommendations for later analysis of system adaptation.
        
        Args:
            user_id: The user who received the recommendations
            recommendations: List of recommendation objects
            weights: Optional dictionary of weights used in the recommendation algorithm
            features_used: Optional list of features used in making recommendations
            data_sources_used: Optional list of data sources used
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_snapshot
                    (timestamp, user_id, recommendations, weights, features_used, data_sources_used)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    json.dumps(recommendations),
                    json.dumps(weights) if weights else None,
                    json.dumps(features_used) if features_used else None,
                    json.dumps(data_sources_used) if data_sources_used else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded recommendation snapshot for user {user_id} with {len(recommendations)} recommendations")
        except Exception as e:
            self.logger.error(f"Error recording recommendation snapshot: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def measure_domain_field_adoption(self, start_date: Optional[datetime] = None, 
                                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Measure how domain and field data from messages is being used in recommendations.
        
        Args:
            start_date: Optional start date for analysis window
            end_date: Optional end date for analysis window
            
        Returns:
            Dictionary with metrics about domain/field adoption
        """
        conn = None
        try:
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
                
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count messaging interactions
                cur.execute("""
                SELECT 
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(ARRAY_LENGTH(domains, 1)) as avg_domains_per_message,
                    AVG(ARRAY_LENGTH(fields, 1)) as avg_fields_per_message,
                    COUNT(*) FILTER (WHERE ARRAY_LENGTH(domains, 1) > 0) as messages_with_domains,
                    COUNT(*) FILTER (WHERE ARRAY_LENGTH(fields, 1) > 0) as messages_with_fields
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                """, (start_date, end_date))
                
                message_stats = cur.fetchone()
                
                # Compare with recommendation snapshots
                cur.execute("""
                SELECT 
                    COUNT(*) as total_snapshots,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(*) FILTER (
                        WHERE data_sources_used @> '"messaging"' OR 
                              data_sources_used @> '"message_domains"' OR
                              data_sources_used @> '"message_fields"'
                    ) as snapshots_using_messaging
                FROM recommendation_snapshot
                WHERE timestamp BETWEEN %s AND %s
                """, (start_date, end_date))
                
                recommendation_stats = cur.fetchone()
                
                # Get top domains and fields mentioned
                cur.execute("""
                SELECT 
                    UNNEST(domains) as domain,
                    COUNT(*) as mention_count
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                GROUP BY domain
                ORDER BY mention_count DESC
                LIMIT 10
                """, (start_date, end_date))
                
                top_domains = [dict(row) for row in cur.fetchall()]
                
                cur.execute("""
                SELECT 
                    UNNEST(fields) as field,
                    COUNT(*) as mention_count
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                GROUP BY field
                ORDER BY mention_count DESC
                LIMIT 10
                """, (start_date, end_date))
                
                top_fields = [dict(row) for row in cur.fetchall()]
                
                # Calculate messaging influence metrics
                messaging_adoption = 0
                if recommendation_stats['total_snapshots'] > 0:
                    messaging_adoption = recommendation_stats['snapshots_using_messaging'] / recommendation_stats['total_snapshots']
                
                # Compile metrics
                metrics = {
                    "message_stats": message_stats,
                    "recommendation_stats": recommendation_stats,
                    "top_domains": top_domains,
                    "top_fields": top_fields,
                    "messaging_adoption": messaging_adoption,
                    "analysis_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": (end_date - start_date).days
                    }
                }
                
                # Record these metrics
                self.record_metric(
                    metric_type="adaptation",
                    metric_name="messaging_domain_field_adoption",
                    metric_value=messaging_adoption,
                    details={
                        "message_count": message_stats['total_interactions'],
                        "recommendation_count": recommendation_stats['total_snapshots'],
                        "period_days": (end_date - start_date).days
                    }
                )
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error measuring domain/field adoption: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def analyze_recommendation_evolution(self, user_id: str, 
                                        lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze how recommendations for a specific user have evolved over time.
        
        Args:
            user_id: User ID to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with metrics about recommendation evolution
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get snapshots for this user
                start_date = datetime.now() - timedelta(days=lookback_days)
                
                cur.execute("""
                SELECT 
                    id,
                    timestamp,
                    recommendations,
                    weights,
                    features_used,
                    data_sources_used
                FROM recommendation_snapshot
                WHERE user_id = %s AND timestamp > %s
                ORDER BY timestamp
                """, (user_id, start_date))
                
                snapshots = [dict(row) for row in cur.fetchall()]
                
                if not snapshots:
                    return {
                        "user_id": user_id,
                        "analysis": "No recommendation snapshots found for this user in the specified time period."
                    }
                
                # Extract expert IDs from each snapshot
                evolution_data = []
                for snapshot in snapshots:
                    expert_ids = []
                    try:
                        recommendations = json.loads(snapshot['recommendations'])
                        expert_ids = [rec.get('id') for rec in recommendations if rec.get('id')]
                        
                        # Extract data sources and features
                        data_sources = []
                        if snapshot['data_sources_used']:
                            data_sources = json.loads(snapshot['data_sources_used'])
                        
                        features = []
                        if snapshot['features_used']:
                            features = json.loads(snapshot['features_used'])
                        
                        evolution_data.append({
                            "timestamp": snapshot['timestamp'].isoformat(),
                            "expert_ids": expert_ids,
                            "expert_count": len(expert_ids),
                            "data_sources": data_sources,
                            "features": features,
                            "uses_messaging": any(
                                source in ['messaging', 'message_domains', 'message_fields'] 
                                for source in data_sources
                            )
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing snapshot {snapshot['id']}: {e}")
                
                # Calculate change metrics
                if len(evolution_data) >= 2:
                    first_experts = set(evolution_data[0]['expert_ids'])
                    last_experts = set(evolution_data[-1]['expert_ids'])
                    
                    # Calculate Jaccard similarity
                    intersection = len(first_experts.intersection(last_experts))
                    union = len(first_experts.union(last_experts))
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    # Calculate turnover rate
                    turnover_rate = 1.0 - jaccard_similarity
                    
                    # Check for increasing use of messaging data
                    messaging_adoption_trend = []
                    for data_point in evolution_data:
                        messaging_adoption_trend.append(1 if data_point['uses_messaging'] else 0)
                    
                    # Calculate if messaging usage is increasing
                    messaging_trend = 0
                    if len(messaging_adoption_trend) >= 2:
                        early_usage = sum(messaging_adoption_trend[:len(messaging_adoption_trend)//2])
                        late_usage = sum(messaging_adoption_trend[len(messaging_adoption_trend)//2:])
                        messaging_trend = late_usage - early_usage
                    
                    evolution_metrics = {
                        "jaccard_similarity": jaccard_similarity,
                        "turnover_rate": turnover_rate,
                        "snapshot_count": len(evolution_data),
                        "messaging_adoption_trend": messaging_trend,
                        "first_timestamp": evolution_data[0]['timestamp'],
                        "last_timestamp": evolution_data[-1]['timestamp']
                    }
                    
                    # Record the turnover metric
                    self.record_metric(
                        metric_type="adaptation",
                        metric_name="recommendation_turnover_rate",
                        metric_value=turnover_rate,
                        user_id=user_id,
                        details={
                            "jaccard_similarity": jaccard_similarity,
                            "snapshot_count": len(evolution_data),
                            "period_days": lookback_days
                        }
                    )
                else:
                    evolution_metrics = {
                        "message": "Not enough snapshots for evolution analysis"
                    }
                
                return {
                    "user_id": user_id,
                    "evolution_data": evolution_data,
                    "evolution_metrics": evolution_metrics,
                    "analysis_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": datetime.now().isoformat(),
                        "days": lookback_days
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing recommendation evolution: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def generate_adaptation_report(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  output_format: str = 'json') -> Any:
        """
        Generate a comprehensive report on recommendation system adaptation.
        
        Args:
            start_date: Optional start date for analysis window
            end_date: Optional end date for analysis window
            output_format: Format of the report ('json', 'csv', 'html')
            
        Returns:
            Report in the specified format
        """
        conn = None
        try:
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
                
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Gather metrics from recommendation_metrics table
                cur.execute("""
                SELECT 
                    metric_type,
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value,
                    COUNT(*) as measurement_count,
                    MIN(timestamp) as first_measurement,
                    MAX(timestamp) as last_measurement
                FROM recommendation_metrics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY metric_type, metric_name
                ORDER BY metric_type, metric_name
                """, (start_date, end_date))
                
                metrics_summary = [dict(row) for row in cur.fetchall()]
                
                # Gather domain/field usage stats
                domain_field_stats = self.measure_domain_field_adoption(start_date, end_date)
                
                # Gather user feedback stats
                cur.execute("""
                SELECT 
                    recommendation_method,
                    interaction_type,
                    COUNT(*) as interaction_count,
                    AVG(feedback_score) as avg_feedback_score
                FROM recommendation_feedback
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY recommendation_method, interaction_type
                ORDER BY recommendation_method, interaction_type
                """, (start_date, end_date))
                
                feedback_stats = [dict(row) for row in cur.fetchall()]
                
                # Gather system adaptation metrics over time
                cur.execute("""
                SELECT 
                    DATE_TRUNC('day', timestamp) as day,
                    COUNT(*) as snapshot_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(*) FILTER (
                        WHERE data_sources_used @> '"message_domains"' OR
                              data_sources_used @> '"message_fields"'
                    ) as snapshots_using_messaging
                FROM recommendation_snapshot
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE_TRUNC('day', timestamp)
                ORDER BY DATE_TRUNC('day', timestamp)
                """, (start_date, end_date))
                
                daily_adaptation = [dict(row) for row in cur.fetchall()]
                
                # Compile the report
                report = {
                    "report_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": (end_date - start_date).days
                    },
                    "metrics_summary": metrics_summary,
                    "domain_field_stats": domain_field_stats,
                    "feedback_stats": feedback_stats,
                    "daily_adaptation": daily_adaptation,
                    "generated_at": datetime.now().isoformat()
                }
                
                # Format the report as requested
                if output_format == 'json':
                    return report
                elif output_format == 'csv':
                    return self._report_to_csv(report)
                elif output_format == 'html':
                    return self._report_to_html(report)
                else:
                    return report
                
        except Exception as e:
            self.logger.error(f"Error generating adaptation report: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def _report_to_csv(self, report: Dict) -> str:
        """Convert report data to CSV format"""
        try:
            # Create a temporary file to store CSV data
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
                filename = temp.name
                
                # Write metrics summary to CSV
                writer = csv.writer(temp)
                writer.writerow(['Metric Type', 'Metric Name', 'Average Value', 'Min Value', 'Max Value', 'Count'])
                
                for metric in report['metrics_summary']:
                    writer.writerow([
                        metric['metric_type'],
                        metric['metric_name'],
                        metric['avg_value'],
                        metric['min_value'],
                        metric['max_value'],
                        metric['measurement_count']
                    ])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Domain/Field Adoption Metrics'])
                writer.writerow([])
                
                # Write domain/field stats
                if 'message_stats' in report['domain_field_stats']:
                    stats = report['domain_field_stats']['message_stats']
                    writer.writerow(['Total Interactions', 'Unique Users', 'Avg Domains/Message', 'Avg Fields/Message'])
                    writer.writerow([
                        stats['total_interactions'],
                        stats['unique_users'],
                        stats['avg_domains_per_message'],
                        stats['avg_fields_per_message']
                    ])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Top Domains'])
                writer.writerow(['Domain', 'Mention Count'])
                
                # Write top domains
                if 'top_domains' in report['domain_field_stats']:
                    for domain in report['domain_field_stats']['top_domains']:
                        writer.writerow([domain['domain'], domain['mention_count']])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Daily Adaptation'])
                writer.writerow(['Day', 'Snapshot Count', 'Unique Users', 'Using Messaging'])
                
                # Write daily adaptation
                for day in report['daily_adaptation']:
                    writer.writerow([
                        day['day'].strftime('%Y-%m-%d') if isinstance(day['day'], datetime) else day['day'],
                        day['snapshot_count'],
                        day['unique_users'],
                        day['snapshots_using_messaging']
                    ])
            
            # Read the temporary file
            with open(filename, 'r') as f:
                csv_content = f.read()
            
            # Clean up the temporary file
            os.unlink(filename)
            
            return csv_content
            
        except Exception as e:
            self.logger.error(f"Error converting report to CSV: {e}")
            return f"Error: {str(e)}"
    
    def _report_to_html(self, report: Dict) -> str:
        """Convert report data to HTML format"""
        try:
            html = f"""
            <html>
            <head>
                <title>Recommendation System Adaptation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-good {{ color: green; }}
                    .metric-bad {{ color: red; }}
                    .metric-neutral {{ color: orange; }}
                </style>
            </head>
            <body>
                <h1>Recommendation System Adaptation Report</h1>
                <p>Period: {report['report_period']['start_date']} to {report['report_period']['end_date']} ({report['report_period']['days']} days)</p>
                
                <h2>Metrics Summary</h2>
                <table>
                    <tr>
                        <th>Metric Type</th>
                        <th>Metric Name</th>
                        <th>Average Value</th>
                        <th>Min Value</th>
                        <th>Max Value</th>
                        <th>Measurements</th>
                    </tr>
            """
            
            # Add metrics rows
            for metric in report['metrics_summary']:
                html += f"""
                    <tr>
                        <td>{metric['metric_type']}</td>
                        <td>{metric['metric_name']}</td>
                        <td>{metric['avg_value']:.4f}</td>
                        <td>{metric['min_value']:.4f}</td>
                        <td>{metric['max_value']:.4f}</td>
                        <td>{metric['measurement_count']}</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <h2>Domain/Field Adoption</h2>
            """
            
            # Add domain/field stats
            if 'message_stats' in report['domain_field_stats']:
                stats = report['domain_field_stats']['message_stats']
                html += f"""
                <table>
                    <tr>
                        <th>Total Interactions</th>
                        <th>Unique Users</th>
                        <th>Avg Domains/Message</th>
                        <th>Avg Fields/Message</th>
                    </tr>
                    <tr>
                        <td>{stats['total_interactions']}</td>
                        <td>{stats['unique_users']}</td>
                        <td>{stats['avg_domains_per_message']:.2f}</td>
                        <td>{stats['avg_fields_per_message']:.2f}</td>
                    </tr>
                </table>
                """
            
            # Add messaging adoption metric
            if 'messaging_adoption' in report['domain_field_stats']:
                adoption = report['domain_field_stats']['messaging_adoption']
                adoption_class = 'metric-good' if adoption > 0.7 else ('metric-neutral' if adoption > 0.3 else 'metric-bad')
                html += f"""
                <p>Messaging Adoption Rate: <span class="{adoption_class}">{adoption:.2%}</span></p>
                """
            
            # Add top domains
            if 'top_domains' in report['domain_field_stats']:
                html += """
                <h3>Top Domains</h3>
                <table>
                    <tr>
                        <th>Domain</th>
                        <th>Mention Count</th>
                    </tr>
                """
                
                for domain in report['domain_field_stats']['top_domains']:
                    html += f"""
                    <tr>
                        <td>{domain['domain']}</td>
                        <td>{domain['mention_count']}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Add top fields
            if 'top_fields' in report['domain_field_stats']:
                html += """
                <h3>Top Fields</h3>
                <table>
                    <tr>
                        <th>Field</th>
                        <th>Mention Count</th>
                    </tr>
                """
                
                for field in report['domain_field_stats']['top_fields']:
                    html += f"""
                    <tr>
                        <td>{field['field']}</td>
                        <td>{field['mention_count']}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Add feedback stats
            if report['feedback_stats']:
                html += """
                <h2>User Feedback</h2>
                <table>
                    <tr>
                        <th>Recommendation Method</th>
                        <th>Interaction Type</th>
                        <th>Count</th>
                        <th>Avg Feedback Score</th>
                    </tr>
                """
                
                for feedback in report['feedback_stats']:
                    score_class = 'metric-neutral'
                    if feedback['avg_feedback_score'] is not None:
                        score_class = 'metric-good' if feedback['avg_feedback_score'] > 0.7 else ('metric-neutral' if feedback['avg_feedback_score'] > 0.3 else 'metric-bad')
                    
                    html += f"""
                    <tr>
                        <td>{feedback['recommendation_method']}</td>
                        <td>{feedback['interaction_type']}</td>
                        <td>{feedback['interaction_count']}</td>
                        <td class="{score_class}">{feedback['avg_feedback_score']:.2f if feedback['avg_feedback_score'] is not None else 'N/A'}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Close the HTML document
            html += f"""
                <p><em>Report generated at: {report['generated_at']}</em></p>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error converting report to HTML: {e}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def monitor_recommendation_changes(self, user_id: str, 
                                       new_recommendations: List[Dict], 
                                       weights: Optional[Dict] = None,
                                       features_used: Optional[List[str]] = None,
                                       data_sources_used: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Monitor and record changes in recommendations for a user.
        
        Args:
            user_id: The user who received the recommendations
            new_recommendations: List of new recommendation objects
            weights: Optional dictionary of weights used in the recommendation algorithm
            features_used: Optional list of features used in making recommendations
            data_sources_used: Optional list of data sources used
            
        Returns:
            Dictionary with change metrics
        """
        # First, record the snapshot
        self.record_recommendation_snapshot(
            user_id=user_id,
            recommendations=new_recommendations,
            weights=weights,
            features_used=features_used,
            data_sources_used=data_sources_used
        )
        
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the most recent previous snapshot for this user
                cur.execute("""
                SELECT 
                    id,
                    timestamp,
                    recommendations
                FROM recommendation_snapshot
                WHERE user_id = %s AND id < (
                    SELECT MAX(id) FROM recommendation_snapshot WHERE user_id = %s
                )
                ORDER BY id DESC
                LIMIT 1
                """, (user_id, user_id))
                
                prev_snapshot = cur.fetchone()
                
                if not prev_snapshot:
                    return {
                        "user_id": user_id,
                        "message": "No previous snapshots available for comparison",
                        "is_first_recommendation": True
                    }
                
                # Extract expert IDs from snapshots
                try:
                    prev_recommendations = json.loads(prev_snapshot['recommendations'])
                    prev_expert_ids = [rec.get('id') for rec in prev_recommendations if rec.get('id')]
                    new_expert_ids = [rec.get('id') for rec in new_recommendations if rec.get('id')]
                    
                    # Calculate metrics
                    prev_set = set(prev_expert_ids)
                    new_set = set(new_expert_ids)
                    
                    # Calculate Jaccard similarity
                    intersection = len(prev_set.intersection(new_set))
                    union = len(prev_set.union(new_set))
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    # Calculate turnover rate
                    turnover_rate = 1.0 - jaccard_similarity
                    
                    # Get added and removed experts
                    added_experts = list(new_set - prev_set)
                    removed_experts = list(prev_set - new_set)
                    
                    # Get whether messaging factored into the changes
                    uses_messaging = any(
                        source in ['messaging', 'message_domains', 'message_fields'] 
                        for source in (data_sources_used or [])
                    )
                    
                    # Compile change metrics
                    change_metrics = {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "jaccard_similarity": jaccard_similarity,
                        "turnover_rate": turnover_rate,
                        "added_experts": added_experts,
                        "removed_experts": removed_experts,
                        "prev_snapshot_id": prev_snapshot['id'],
                        "prev_snapshot_timestamp": prev_snapshot['timestamp'].isoformat(),
                        "time_since_last_snapshot": (datetime.now() - prev_snapshot['timestamp']).total_seconds() / 3600,  # in hours
                        "uses_messaging_data": uses_messaging
                    }
                    
                    # Record the turnover metric
                    self.record_metric(
                        metric_type="adaptation",
                        metric_name="recommendation_change_rate",
                        metric_value=turnover_rate,
                        user_id=user_id,
                        details={
                            "jaccard_similarity": jaccard_similarity,
                            "added_experts": len(added_experts),
                            "removed_experts": len(removed_experts),
                            "uses_messaging": uses_messaging
                        }
                    )
                    
                    return change_metrics
                    
                except Exception as e:
                    self.logger.error(f"Error comparing recommendation snapshots: {e}")
                    return {"error": str(e)}
                
        except Exception as e:
            self.logger.error(f"Error monitoring recommendation changes: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()