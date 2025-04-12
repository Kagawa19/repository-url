#!/usr/bin/env python3
import os
import shutil
import sys

def clear_model_cache():
    """Clear all SentenceTransformer model caches in the container"""
    
    # Define all possible cache locations based on your .env and code
    cache_paths = [
        '/app/.model_cache',
        '/app/models',
        '.model_cache',
        os.environ.get('MODEL_CACHE_DIR', '/app/.model_cache'),
        os.environ.get('TRANSFORMERS_CACHE', '/app/.model_cache'),
        os.environ.get('HF_HOME', '/app/.model_cache'),
        os.environ.get('MODEL_PATH', '/app/.model_cache'),
    ]
    
    # Get unique paths
    unique_paths = set()
    for path in cache_paths:
        if path:
            unique_paths.add(os.path.abspath(path))
    
    # Clear each path
    removed = False
    for path in unique_paths:
        if os.path.exists(path):
            print(f"Examining path: {path}")
            
            # Check if it's a directory
            if os.path.isdir(path):
                # Option 1: Clear the entire directory
                if path.endswith('.model_cache') or path.endswith('models'):
                    try:
                        print(f"Removing entire directory: {path}")
                        shutil.rmtree(path)
                        os.makedirs(path, exist_ok=True)  # Recreate empty directory
                        removed = True
                        print(f"Successfully removed and recreated: {path}")
                    except Exception as e:
                        print(f"Error removing directory {path}: {e}")
                
                # Option 2: Clear specific model subdirectories
                else:
                    model_dirs = [
                        os.path.join(path, d) for d in os.listdir(path)
                        if d.startswith('sentence-transformers') or 
                           d.startswith('models--sentence-transformers') or
                           d == 'all-MiniLM-L6-v2'
                    ]
                    
                    for model_dir in model_dirs:
                        try:
                            print(f"Removing model directory: {model_dir}")
                            shutil.rmtree(model_dir)
                            removed = True
                        except Exception as e:
                            print(f"Error removing {model_dir}: {e}")
    
    print("\nChecking environment variables:")
    for var in ['MODEL_CACHE_DIR', 'TRANSFORMERS_CACHE', 'HF_HOME', 'MODEL_PATH']:
        print(f"{var}={os.environ.get(var, 'not set')}")
    
    # If TRANSFORMERS_OFFLINE=1, warn that new model won't be downloaded
    if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        print("\nWARNING: TRANSFORMERS_OFFLINE=1 is set!")
        print("After clearing the cache, the model won't be able to download again.")
        print("You should either:")
        print("1. Rebuild the Docker image")
        print("2. Temporarily set TRANSFORMERS_OFFLINE=0 to allow download")
        print("3. Mount a volume with the model to /app/.model_cache\n")
    
    if removed:
        print("Model cache cleared successfully")
    else:
        print("No model caches found to remove")

if __name__ == "__main__":
    answer = input("This will clear all SentenceTransformer model caches. Continue? (y/n): ")
    if answer.lower() in ['y', 'yes']:
        clear_model_cache()
    else:
        print("Operation cancelled")