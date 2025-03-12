import os
import redis
import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def delete_expert_search_indexes():
    """Delete all expert search indexes from filesystem and Redis."""
    try:
        # 1. Delete FAISS index and mapping files
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        models_dir = current_dir / 'models'
        
        # Find and delete the index files
        index_path = models_dir / 'expert_faiss_index.idx'
        mapping_path = models_dir / 'expert_mapping.pkl'
        
        files_deleted = 0
        
        if index_path.exists():
            index_path.unlink()
            logger.info(f"Deleted FAISS index: {index_path}")
            files_deleted += 1
        else:
            logger.info(f"FAISS index not found at: {index_path}")
            
        if mapping_path.exists():
            mapping_path.unlink()
            logger.info(f"Deleted mapping file: {mapping_path}")
            files_deleted += 1
        else:
            logger.info(f"Mapping file not found at: {mapping_path}")
            
        # 2. Delete all expert data from Redis
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_EMBEDDINGS_DB', 1)),
            decode_responses=True
        )
        
        # Find all expert keys
        expert_keys = redis_client.keys("expert:*")
        if expert_keys:
            # Delete all expert keys
            deleted_count = redis_client.delete(*expert_keys)
            logger.info(f"Deleted {deleted_count} expert records from Redis")
        else:
            logger.info("No expert records found in Redis")
            
        logger.info(f"Index cleanup completed. Deleted {files_deleted} files and {len(expert_keys)} Redis records.")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting expert search indexes: {e}")
        return False

if __name__ == "__main__":
    delete_expert_search_indexes()