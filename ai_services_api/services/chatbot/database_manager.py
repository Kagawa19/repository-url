from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def start_test_session(self) -> str:
        """Start a new session for test user using main database."""
        session = self.Session()
        try:
            # Create a new session in your existing sessions table
            new_session = {
                'session_id': f"test_{datetime.utcnow().timestamp()}",
                'user_id': 'test_user',
                'start_time': datetime.utcnow(),
                'metadata': {
                    'is_test': True,
                    'test_timestamp': datetime.utcnow().isoformat()
                }
            }
            
            session_id = self.redis_text.hset(
                f"session:{new_session['session_id']}", 
                mapping=new_session
            )
            
            logger.info(f"Created test session: {new_session['session_id']}")
            return new_session['session_id']
            
        except Exception as e:
            logger.error(f"Error creating test session: {e}")
            raise
        finally:
            session.close()

    def record_test_interaction(self, session_id: str, query: str, response: str, metrics: dict):
        """Record test interaction in main interaction table."""
        session = self.Session()
        try:
            # Store interaction in your existing chat_interactions collection
            interaction_data = {
                'session_id': session_id,
                'user_id': 'test_user',
                'query': query,
                'response': response,
                'timestamp': datetime.utcnow(),
                'response_time': metrics.get('response_time', 0),
                'intent_type': metrics.get('intent_type'),
                'intent_confidence': metrics.get('intent_confidence'),
                'expert_matches': metrics.get('expert_matches', 0),
                'is_test': True
            }
            
            # Store in Redis
            interaction_id = f"interaction:{datetime.utcnow().timestamp()}"
            self.redis_text.hset(
                interaction_id,
                mapping=interaction_data
            )
            
            # Store expert matches if any
            if metrics.get('matched_experts'):
                for idx, expert in enumerate(metrics['matched_experts']):
                    expert_match = {
                        'interaction_id': interaction_id,
                        'expert_id': expert['id'],
                        'similarity_score': expert.get('similarity', 0),
                        'rank': idx + 1,
                        'is_test': True
                    }
                    self.redis_text.hset(
                        f"expert_match:{interaction_id}:{expert['id']}", 
                        mapping=expert_match
                    )
            
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error recording test interaction: {e}")
            raise
        finally:
            session.close()

    def get_test_metrics(self, hours: int = 24):
        """Get metrics for test interactions."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Query your existing analytics collection with test filter
            test_interactions = self.redis_text.keys("interaction:*")
            
            metrics = {
                'total_test_interactions': 0,
                'avg_response_time': 0,
                'intent_distribution': {},
                'expert_match_rates': {},
                'error_rate': 0
            }
            
            total_response_time = 0
            error_count = 0
            intent_counts = {}
            
            for key in test_interactions:
                interaction = self.redis_text.hgetall(key)
                if interaction.get('is_test') == 'True' and \
                   datetime.fromisoformat(interaction['timestamp']) >= start_time:
                    
                    metrics['total_test_interactions'] += 1
                    total_response_time += float(interaction['response_time'])
                    
                    if interaction.get('error_occurred') == 'True':
                        error_count += 1
                        
                    intent_type = interaction.get('intent_type', 'unknown')
                    intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
            
            # Calculate averages and rates
            if metrics['total_test_interactions'] > 0:
                metrics['avg_response_time'] = total_response_time / metrics['total_test_interactions']
                metrics['error_rate'] = error_count / metrics['total_test_interactions']
                metrics['intent_distribution'] = intent_counts
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting test metrics: {e}")
            raise
