#!/bin/bash

echo "=== Complete Model Cache Reset Tool ==="
echo "This will remove all sentence transformer models and force fresh downloads"
echo ""

# Get container names from running containers
API_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'api-1$|api$')
WORKER_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'airflow-worker-1$|airflow-worker$')

# Check if containers are running
if [ -z "$API_CONTAINER" ]; then
    echo "❌ API container not found or not running!"
    API_RUNNING=false
else
    echo "✅ Found API container: $API_CONTAINER"
    API_RUNNING=true
fi

if [ -z "$WORKER_CONTAINER" ]; then
    echo "❌ Airflow worker container not found or not running!"
    WORKER_RUNNING=false
else
    echo "✅ Found Airflow worker container: $WORKER_CONTAINER"
    WORKER_RUNNING=true
fi

if [ "$API_RUNNING" = false ] && [ "$WORKER_RUNNING" = false ]; then
    echo "No running containers found. Please start your containers first."
    exit 1
fi

echo ""
echo "This will completely remove all model files, disable offline mode,"
echo "and force a fresh download of the models."
echo ""
echo "WARNING: This will modify your containers' environment variables temporarily."
read -p "Do you want to continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Step 1: Clearing all model caches..."

# Clear API container
if [ "$API_RUNNING" = true ]; then
    echo "Removing models from API container..."
    docker exec "$API_CONTAINER" bash -c "rm -rf /app/models/* /app/.model_cache/* /tmp/.cache/torch /home/appuser/.cache/torch"
    RESULT_API=$?
    if [ $RESULT_API -eq 0 ]; then
        echo "✅ Successfully cleared API models"
    else
        echo "❌ Failed to clear API models"
    fi
fi

# Clear Worker container
if [ "$WORKER_RUNNING" = true ]; then
    echo "Removing models from Airflow worker container..."
    docker exec "$WORKER_CONTAINER" bash -c "rm -rf /app/models/* /app/.model_cache/* /tmp/.cache/torch /home/appuser/.cache/torch"
    RESULT_WORKER=$?
    if [ $RESULT_WORKER -eq 0 ]; then
        echo "✅ Successfully cleared worker models"
    else
        echo "❌ Failed to clear worker models"
    fi
fi

echo ""
echo "Step 2: Temporarily disabling offline mode and forcing fresh download..."

# Create download script
cat > force_download.py << 'EOL'
#!/usr/bin/env python3
import os
import shutil
import sys

def force_download_model():
    """Force a fresh download of SentenceTransformer model"""
    print("Starting fresh model download...")
    
    # Clear environment variables that might affect download
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    
    # Set download location
    cache_dir = "/app/models"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    
    # Force clean directories
    model_paths = [
        os.path.join(cache_dir, "sentence-transformers"),
        os.path.join(cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2"),
        os.path.join(cache_dir, "all-MiniLM-L6-v2"),
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path, ignore_errors=True)
    
    # Download fresh model
    try:
        print("Downloading model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
        
        # Test the model
        test_emb = model.encode("Test sentence to verify model works")
        print(f"Model downloaded and tested successfully! Embedding shape: {test_emb.shape}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = force_download_model()
    sys.exit(0 if success else 1)
EOL

# Copy and run the download script on the API container
if [ "$API_RUNNING" = true ]; then
    echo "Forcing fresh download in API container..."
    docker cp force_download.py "$API_CONTAINER":/app/force_download.py
    docker exec "$API_CONTAINER" python /app/force_download.py
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded fresh model in API container"
    else
        echo "❌ Failed to download fresh model in API container"
    fi
fi

# No need to download in worker too, since it should use the same volume

echo ""
echo "Step 3: Restarting containers..."

# Restart API container
if [ "$API_RUNNING" = true ]; then
    echo "Restarting API container..."
    docker restart "$API_CONTAINER"
    if [ $? -eq 0 ]; then
        echo "✅ API container restarted"
    else
        echo "❌ Failed to restart API container"
    fi
fi

# Restart Worker container
if [ "$WORKER_RUNNING" = true ]; then
    echo "Restarting Airflow worker container..."
    docker restart "$WORKER_CONTAINER"
    if [ $? -eq 0 ]; then
        echo "✅ Airflow worker container restarted"
    else
        echo "❌ Failed to restart Airflow worker container"
    fi
fi

# Cleanup
rm -f force_download.py

echo ""
echo "Model cache reset and fresh download completed!"
echo ""