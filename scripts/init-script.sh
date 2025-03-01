#!/bin/bash
set -e

# Environment setup
INIT_MARKER="/app/.initialization_complete/init.done"
LOG_DIR="/app/logs"
CACHE_DIR="/app/cache"
TMP_CACHE_DIR="/tmp/transformers_cache"
TMP_HF_DIR="/tmp/hf_home"

mkdir -p "$LOG_DIR"
mkdir -p "/app/.initialization_complete"

# Create and set proper permissions for cache directories
mkdir -p "$CACHE_DIR" "$TMP_CACHE_DIR" "$TMP_HF_DIR"
chmod -R 777 "$CACHE_DIR" "$TMP_CACHE_DIR" "$TMP_HF_DIR"

# Override environment variables to use temp directories with proper permissions
export TRANSFORMERS_CACHE="$TMP_CACHE_DIR"
export HF_HOME="$TMP_HF_DIR"

# Logging setup
exec 1> >(tee -a "${LOG_DIR}/init.log") 2>&1
echo "[$(date)] Starting initialization..."
echo "[$(date)] Using temporary cache directories with proper permissions"

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
    
    # Use Python to create a wrapper that explicitly sets the model cache path
    if python -c "
import os, sys
os.environ['TRANSFORMERS_CACHE'] = '$TMP_CACHE_DIR'
os.environ['HF_HOME'] = '$TMP_HF_DIR'
sys.path.insert(0, '/app')
from ai_services_api.services.setup import init
sys.exit(0 if init.run() else 1)
"; then
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