#!/usr/bin/env python3
import os
import shutil
import sys
import subprocess

def clear_model_cache():
    """Clear ALL SentenceTransformer model caches in the container"""
    
    # List all directories to check
    cache_dirs = [
        '/app/models',
        '/app/.model_cache',
        '/home/appuser/models',
        '/tmp/.cache'
    ]
    
    paths_removed = []
    
    # First try finding all model-related paths
    print("Searching for all model-related directories:")
    for root_dir in cache_dirs:
        if not os.path.exists(root_dir):
            print(f"  {root_dir} (doesn't exist)")
            continue
            
        print(f"  {root_dir}/")
        
        # Check direct paths - thorough approach
        potential_model_paths = [
            os.path.join(root_dir, 'all-MiniLM-L6-v2'),
            os.path.join(root_dir, 'sentence-transformers', 'all-MiniLM-L6-v2'),
            os.path.join(root_dir, 'models--sentence-transformers--all-MiniLM-L6-v2'),
            os.path.join(root_dir, 'hub', 'models--sentence-transformers--all-MiniLM-L6-v2'),
        ]
        
        for path in potential_model_paths:
            if os.path.exists(path):
                print(f"    ✓ Found: {path}")
                try:
                    print(f"    Removing: {path}")
                    shutil.rmtree(path)
                    paths_removed.append(path)
                except Exception as e:
                    print(f"    Error removing {path}: {e}")
                    
        # Search recursively for any remaining sentence-transformer related directories
        for subdir, dirs, files in os.walk(root_dir):
            for d in dirs:
                if 'sentence-transformer' in d.lower() or 'miniml' in d.lower():
                    full_path = os.path.join(subdir, d)
                    print(f"    ✓ Found: {full_path}")
                    try:
                        print(f"    Removing: {full_path}")
                        shutil.rmtree(full_path)
                        paths_removed.append(full_path)
                    except Exception as e:
                        print(f"    Error removing {full_path}: {e}")
    
    # Report what was done
    if paths_removed:
        print("\nSuccessfully removed these model directories:")
        for path in paths_removed:
            print(f"  - {path}")
    else:
        print("\nNo model directories found to remove")
    
    # Check huggingface cache
    try:
        from huggingface_hub import scan_cache_dir, delete_from_cache
        print("\nChecking HuggingFace cache:")
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if "sentence-transformers" in repo.repo_id or "all-MiniLM-L6-v2" in repo.repo_id:
                print(f"  Deleting: {repo.repo_id}")
                delete_from_cache(repo_id=repo.repo_id)
    except ImportError:
        print("\nHuggingFace Hub not installed, skipping that cache check")
    except Exception as e:
        print(f"\nError checking HuggingFace cache: {e}")
    
    # Check environment variables
    print("\nRelevant environment variables:")
    env_vars = ['TRANSFORMERS_CACHE', 'HF_HOME', 'EMBEDDING_MODEL', 'MODEL_CACHE_DIR', 'MODEL_PATH', 'TRANSFORMERS_OFFLINE']
    for var in env_vars:
        print(f"  {var}={os.environ.get(var, 'not set')}")
    
    if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        print("\n⚠️ WARNING: TRANSFORMERS_OFFLINE=1 is set!")
        print("After clearing the cache, the model won't be able to download again automatically.")
        print("Options:")
        print("  1. Rebuild the Docker image")
        print("  2. Temporarily set TRANSFORMERS_OFFLINE=0 to allow download")
    
    # Try to release memory if the model might be loaded
    try:
        import gc
        gc.collect()
        print("\nGarbage collection completed")
    except:
        pass
    
    return bool(paths_removed)

if __name__ == "__main__":
    print("=" * 60)
    print("SENTENCE TRANSFORMER MODEL CACHE CLEANER")
    print("=" * 60)
    print("This will search for and remove ALL instances of SentenceTransformer models")
    print("in various cache directories throughout the container.")
    print()
    answer = input("Continue? (y/n): ")
    if answer.lower() in ['y', 'yes']:
        removed = clear_model_cache()
        
        if removed:
            print("\n✅ Model caches cleared successfully")
            print("\nTo prevent the model from being loaded again, you might need to:")
            print("1. Restart the container")
            print("2. Rebuild the Docker image")
        else:
            print("\n❌ No model caches were found or could be removed.")
    else:
        print("Operation cancelled")