# config/settings.py

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / 'cache'

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Redis Configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),  # Explicitly use db 0
    'password': os.getenv('REDIS_PASSWORD', None),
    'ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true'
}

# Web Scraping Configuration
WEBSITE_URL = os.getenv('WEBSITE_URL', 'https://aphrc.org')
MAX_PAGES = int(os.getenv('MAX_PAGES', '1000'))
MAX_DEPTH = int(os.getenv('MAX_DEPTH', '3'))
SELENIUM_TIMEOUT = int(os.getenv('SELENIUM_TIMEOUT', '30'))
SCROLL_PAUSE_TIME = float(os.getenv('SCROLL_PAUSE_TIME', '1.0'))
CHROME_DRIVER_PATH = os.getenv('CHROME_DRIVER_PATH', None)
SCRAPE_STATE_FILE = os.getenv('SCRAPE_STATE_FILE', 'scrape_state.json')

# PDF Processing Configuration
PDF_FOLDER = os.getenv('PDF_FOLDER', str(DATA_DIR / 'pdf_files'))
PDF_CHUNK_SIZE = int(os.getenv('PDF_CHUNK_SIZE', '1000'))
CLEANUP_PDFS = os.getenv('CLEANUP_PDFS', 'false').lower() == 'true'
MAX_PDF_SIZE = int(os.getenv('MAX_PDF_SIZE', '50')) * 1024 * 1024  # Convert MB to bytes

# Model Configuration
MODEL_CONFIG = {
    'name': os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
    'max_tokens': int(os.getenv('MAX_TOKENS', '512')),
    'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
    'cache_dir': os.getenv('MODEL_CACHE_DIR', str(CACHE_DIR / 'models')),
    'device': os.getenv('MODEL_DEVICE', None)  # None for auto-detection
}

# API Configuration
API_CONFIG = {
    'version': '1.0.0',
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', '8000')),
    'workers': int(os.getenv('API_WORKERS', '4')),
    'timeout': int(os.getenv('API_TIMEOUT', '60')),
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(DATA_DIR / 'app.log'),
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'propagate': True
        }
    }
}

# Content Processing Configuration
CONTENT_CONFIG = {
    'max_workers': int(os.getenv('MAX_WORKERS', '4')),
    'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
    'min_chunk_size': int(os.getenv('MIN_CHUNK_SIZE', '100')),
    'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '2000')),
    'allowed_domains': os.getenv('ALLOWED_DOMAINS', WEBSITE_URL).split(','),
    'exclude_patterns': os.getenv('EXCLUDE_PATTERNS', '').split(','),
    'timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
}

# Security Configuration
SECURITY_CONFIG = {
    'ssl_verify': os.getenv('SSL_VERIFY', 'true').lower() == 'true',
    'api_key_header': os.getenv('API_KEY_HEADER', 'X-API-Key'),
    'api_key': os.getenv('API_KEY', None),
    'rate_limit': int(os.getenv('RATE_LIMIT', '100')),
    'rate_limit_period': int(os.getenv('RATE_LIMIT_PERIOD', '60')),
}

def get_all_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary"""
    return {
        'database': DATABASE_CONFIG,
        'redis': REDIS_CONFIG,
        'model': MODEL_CONFIG,
        'api': API_CONFIG,
        'content': CONTENT_CONFIG,
        'security': SECURITY_CONFIG,
        'scraping': {
            'website_url': WEBSITE_URL,
            'max_pages': MAX_PAGES,
            'max_depth': MAX_DEPTH,
            'selenium_timeout': SELENIUM_TIMEOUT,
            'scroll_pause_time': SCROLL_PAUSE_TIME,
        },
        'pdf': {
            'folder': PDF_FOLDER,
            'chunk_size': PDF_CHUNK_SIZE,
            'cleanup': CLEANUP_PDFS,
            'max_size': MAX_PDF_SIZE,
        },
    }

def validate_settings() -> bool:
    """Validate critical settings"""
    try:
        # Check database configuration
        assert all(DATABASE_CONFIG.values()), "Missing database configuration"
        
        # Check Redis configuration
        assert REDIS_CONFIG['host'], "Missing Redis host"
        
        # Check content configuration
        assert WEBSITE_URL, "Missing website URL"
        assert MAX_PAGES > 0, "Invalid max pages value"
        
        # Check model configuration
        assert MODEL_CONFIG['name'], "Missing model name"
        assert MODEL_CONFIG['max_tokens'] > 0, "Invalid max tokens value"
        
        return True
    except AssertionError as e:
        logging.error(f"Settings validation failed: {str(e)}")
        return False

# Validate settings on import
if not validate_settings():
    raise ValueError("Invalid configuration settings")