#!/usr/bin/env python3
import os
import shutil
import sys

def clear_model_cache():
    """Clear all SentenceTransformer model caches in the container"""
    
    # Define all possible cache locations
    cache_paths = [
        '/app/models/sentence-transformers',
        '/app/models/models--sentence-transformers--all-MiniLM-L6-v2',
        '/app/models/safetensors',
        os.path.join(os.environ.get('HF_HOME', '/app/models'), 'models--sentence-transformers--all-MiniLM-L6-v2'),
        os.path.join(os.environ.get('TRANSFORMERS_CACHE', '/app/models'), 'models--sentence-transformers--all-MiniLM-L6-v2'),
    ]
    
    removed = False
    for path in cache_paths:
        if os.path.exists(path):
            print(f"Removing cache at: {path}")
            try:
                shutil.rmtree(path)
                removed = True
            except Exception as e:
                print(f"Error removing {path}: {e}")
    
    # If TRANSFORMERS_OFFLINE=1, warn that new model won't be downloaded
    if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        print("\nWARNING: TRANSFORMERS_OFFLINE=1 is set!")
        print("After clearing the cache, the model won't be able to download again.")
        print("You should either:")
        print("1. Rebuild the Docker image")
        print("2. Temporarily set TRANSFORMERS_OFFLINE=0 to allow download")
        print("3. Mount a volume with the model to /app/models\n")
    
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