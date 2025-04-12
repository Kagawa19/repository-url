#!/usr/bin/env python3
import os
import shutil
import sys
import time

def redownload_model():
    """Redownload SentenceTransformer model with fresh data"""
    
    # Get the correct cache directory from environment or default
    cache_dir = os.environ.get('MODEL_CACHE_DIR', '/app/.model_cache')
    
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save original offline setting and temporarily disable it
    orig_offline = os.environ.get('TRANSFORMERS_OFFLINE')
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    # Make sure HF_HUB_DISABLE_SSL_VERIFICATION is set for easier downloads
    os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
    
    # Set all cache locations to be consistent
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['MODEL_PATH'] = cache_dir
    
    print(f"\nDownloading fresh SentenceTransformer model to {cache_dir}...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Create a fresh model instance
        model_name = 'all-MiniLM-L6-v2'
        temp_dir = os.path.join(cache_dir, f'fresh_download_{int(time.time())}')
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Downloading model to temporary location: {temp_dir}")
        model = SentenceTransformer(model_name, cache_folder=temp_dir)
        
        # Test the model to ensure it's working
        test_embedding = model.encode("This is a test sentence")
        print(f"Model downloaded and working! Embedding shape: {test_embedding.shape}")
        
        # Use the model's location to find where it was actually saved
        actual_location = model._modules['0'].auto_model.config._name_or_path
        print(f"Model was actually saved to: {actual_location}")
        
        print(f"Download completed successfully")
        
    except Exception as e:
        print(f"Error downloading model using SentenceTransformer: {e}")
        print("\nTrying alternate download method...")
        
        try:
            # Alternative: manual file download
            target_dir = os.path.join(cache_dir, 'sentence-transformers/all-MiniLM-L6-v2')
            os.makedirs(target_dir, exist_ok=True)
            os.chdir(target_dir)
            
            # Download model files directly
            import subprocess
            files_to_download = [
                'config.json',
                'model.safetensors',
                'tokenizer.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'modules.json',
                'vocab.txt'
            ]
            
            for file in files_to_download:
                url = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/{file}"
                print(f"Downloading {url}")
                subprocess.run(['wget', '--no-check-certificate', '-O', file, url], check=True)
            
            print(f"Manual download completed successfully to {target_dir}")
            
        except Exception as manual_err:
            print(f"Manual download failed: {manual_err}")
    
    # Restore original offline setting
    if orig_offline is not None:
        os.environ['TRANSFORMERS_OFFLINE'] = orig_offline
    
    print("\nModel redownload process completed")

if __name__ == "__main__":
    answer = input("This will redownload the SentenceTransformer model with fresh data. Continue? (y/n): ")
    if answer.lower() in ['y', 'yes']:
        redownload_model()
    else:
        print("Operation cancelled")