# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_TEXT_DB = int(os.getenv('REDIS_TEXT_DB', 0))
REDIS_EMBEDDINGS_DB = int(os.getenv('REDIS_EMBEDDINGS_DB', 1))

# Website Configuration
WEBSITE_URL = os.getenv('WEBSITE_URL', 'https://aphrc.org')
MAX_PAGES = int(os.getenv('MAX_PAGES', 1000))
CHECK_INTERVAL_HOURS = int(os.getenv('CHECK_INTERVAL_HOURS', 24))

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'aphrc')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))

# Path Configuration
PDF_FOLDER = os.getenv('PDF_FOLDER', 'data/pdf_files')
LOG_FOLDER = os.getenv('LOG_FOLDER', 'logs')