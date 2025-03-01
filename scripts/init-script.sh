#!/bin/bash
set -e

# Environment setup
INIT_MARKER="/app/.initialization_complete/init.done"
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"
mkdir -p "/app/.initialization_complete"

# Create and set SSL disable script
mkdir -p /app/patches
cat > "/app/patches/disable_ssl.py" << 'EOF'
# disable_ssl.py
import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a custom SSL context that doesn't verify certificates
class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=None)
        # Don't verify SSL certificates
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Patch requests to use our custom adapter
session = requests.Session()
session.mount('https://', CustomHTTPAdapter())

# Patch the original requests.get function
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

# Also patch requests.post, requests.put, etc.
original_post = requests.post
def patched_post(*args, **kwargs):
    kwargs['verify'] = False
    return original_post(*args, **kwargs)
requests.post = patched_post

# Set this environment variable
import os
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

print("SSL verification disabled for all HTTP requests")
EOF

# Set environment variables to disable SSL verification
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

# Create cache directories with proper permissions
mkdir -p /tmp/transformers_cache /app/cache
chmod -R 777 /tmp/transformers_cache /app/cache
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/transformers_cache

# Logging setup
exec 1> >(tee -a "${LOG_DIR}/init.log") 2>&1
echo "[$(date)] Starting initialization..."
echo "[$(date)] SSL verification disabled"

# Function to wait for service availability
wait_for_service() {
    local service=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=0
    
    echo "[$(date)] Waiting for $service..."
    until nc -z "$host" "$port" >/dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            echo "[$(date)] ERROR: $service not available after $max_attempts attempts"
            return 1
        fi
        echo "[$(date)] Waiting for $service... (attempt $attempt/$max_attempts)"
        sleep 5
    done
    echo "[$(date)] $service is available"
}

# Wait for all required services
wait_for_services() {
    wait_for_service "PostgreSQL" "postgres" 5432 60 || return 1
    wait_for_service "Redis" "redis" 6379 30 || return 1
    wait_for_service "Neo4j" "neo4j" 7687 30 || return 1
    return 0
}

# Run the Python setup script with environment-based configuration
run_setup() {
    local setup_args=""
    
    # Build setup arguments from environment variables
    for flag in DATABASE EXPERTS OPENALEX PUBLICATIONS GRAPH SEARCH REDIS SCRAPING CLASSIFICATION; do
        skip_var="SKIP_${flag}"  # Construct the dynamic variable name
        if [ "${!skip_var:-false}" = "true" ]; then  # Use indirect expansion
            setup_args="$setup_args --skip-${flag,,}"  # Append to setup_args
        fi
    done
    
    echo "[$(date)] Running setup with args: $setup_args"
    
    # Run the SSL patch first, then run the setup
    if python -c "
import sys
sys.path.insert(0, '/app/patches')
from disable_ssl import *
from setup import run
sys.exit(0 if run() else 1)
" $setup_args; then
        return 0
    else
        echo "[$(date)] Setup failed"
        return 1
    fi
}

# Main initialization function
initialize() {
    # If initialization is complete and system is healthy, skip initialization
    if [ -f "$INIT_MARKER" ] && python -m ai_services_api.health_check; then
        echo "[$(date)] System already initialized and healthy"
        return 0
    fi
    
    # Remove marker if it exists but system isn't healthy
    [ -f "$INIT_MARKER" ] && rm "$INIT_MARKER"
    
    # Wait for required services
    if ! wait_for_services; then
        echo "[$(date)] Required services not available"
        return 1
    fi
    
    # Run setup
    if ! run_setup; then
        echo "[$(date)] Setup failed"
        return 1
    fi
    
    # Create initialization marker
    date > "$INIT_MARKER"
    echo "[$(date)] Initialization complete"
    return 0
}

# Main execution
if initialize; then
    echo "[$(date)] Starting application..."
    exec uvicorn ai_services_api.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "[$(date)] Initialization failed!"
    exit 1
fi