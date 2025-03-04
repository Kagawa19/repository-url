#!/usr/bin/env python3
import os
import json
import gzip
import requests
import psycopg2
import pandas as pd
from io import BytesIO, StringIO
import time
import concurrent.futures
import boto3

# Database connection settings - Updated for your environment
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": "5432"
}

# OpenAlex S3 bucket
OPENALEX_BUCKET = "openalex"
# Base URL for HTTP access to OpenAlex S3 data
S3_BASE_URL = "https://openalex.s3.amazonaws.com"

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

def setup_database():
    """Set up the PostgreSQL database for resources_resource table if needed"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Check if resources_resource table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'resources_resource'
        );
    """)
    table_exists = cur.fetchone()[0]
    
    if not table_exists:
        print("resources_resource table does not exist. Please create it first.")
    else:
        print("resources_resource table found.")
    
    conn.commit()
    cur.close()
    conn.close()

def find_aphrc_institution():
    """Find the APHRC institution in OpenAlex data"""
    print("Looking for APHRC institution in OpenAlex data...")
    
    # Use only the known working APHRC ID
    aphrc_id = "I4210152772"
    print(f"Using verified APHRC ID: {aphrc_id}")
    
    # Verify the ID
    url = f"https://api.openalex.org/institutions/{aphrc_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "African Population and Health Research Center" in data.get("display_name", ""):
                print(f"✅ Found APHRC with ID: {aphrc_id}")
                return aphrc_id, data.get("display_name", "")
    except Exception as e:
        print(f"Error checking institution {aphrc_id}: {e}")
    
    # If verification fails, return the ID anyway
    return aphrc_id, "African Population and Health Research Center"

def reconstruct_abstract(abstract_inverted_index):
    """Reconstruct abstract text from OpenAlex's inverted index format"""
    if not abstract_inverted_index:
        return ""
    
    # Create a list of words
    words = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            while len(words) <= pos:
                words.append("")
            words[pos] = word
    
    # Join words to form the abstract
    return " ".join(words)

def fallback_to_api(aphrc_id):
    """Get APHRC works directly from the OpenAlex API"""
    print("Fetching APHRC works from OpenAlex API...")
    
    all_works = []
    page = 1
    per_page = 200
    has_more = True
    
    while has_more:
        print(f"Fetching page {page}...")
        url = f"https://api.openalex.org/works?filter=institutions.id:{aphrc_id}&page={page}&per-page={per_page}"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                break
            
            data = response.json()
            results = data.get("results", [])
            all_works.extend(results)
            
            # Check if there are more pages
            meta = data.get("meta", {})
            total_count = meta.get("count", 0)
            current_page = meta.get("page", 0)
            
            if len(results) < per_page or current_page * per_page >= total_count:
                has_more = False
            else:
                page += 1
                time.sleep(0.5)  # Be nice to the API
        
        except Exception as e:
            print(f"Error fetching from API: {e}")
            break
    
    print(f"Found {len(all_works)} APHRC works from API")
    return all_works

def process_works_for_resources(works):
    """Process OpenAlex works data for insertion into resources_resource table"""
    data = []
    
    print(f"Processing {len(works)} publications for resources_resource table...")
    
    for i, work in enumerate(works):
        try:
            # Extract ID for resources_resource table (integer)
            work_id_str = work.get("id", "").replace("https://openalex.org/W", "")
            # Take the last 8 digits to ensure it fits in an integer
            work_id = int(work_id_str[-8:]) if work_id_str else i + 1000000
            
            # Extract title
            title = work.get("title", "Untitled APHRC Publication")
            
            # Extract abstract
            abstract_index = work.get("abstract_inverted_index", {})
            abstract = reconstruct_abstract(abstract_index) if abstract_index else None
            
            # Extract DOI
            doi = work.get("doi", None)
            
            # Process authors into JSON format
            authors_list = []
            for authorship in work.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name", "")
                if author_name:
                    authors_list.append({"name": author_name})
            authors_json = json.dumps(authors_list) if authors_list else None
            
            # Extract publication type
            resource_type = work.get("type", "journal-article")
            
            # Extract journal info for publishers
            host_venue = work.get("host_venue", {})
            journal_name = host_venue.get("display_name", "")
            publishers = json.dumps([{"name": journal_name}]) if journal_name else None
            
            # Extract publication year
            publication_year = str(work.get("publication_year", "")) if work.get("publication_year") is not None else None
            
            # Set source as "openalex"
            source = "openalex"
            
            # Extract identifiers
            identifiers = {}
            if work.get("id"):
                identifiers["openalex"] = work.get("id")
            if doi:
                identifiers["doi"] = doi
            identifiers_json = json.dumps(identifiers) if identifiers else None
            
            # Create row data tuple with NULL for fields we don't have
            row = (
                work_id,              # id 
                doi,                  # doi
                title,                # title
                abstract,             # abstract
                None,                 # summary
                None,                 # domains
                None,                 # topics
                None,                 # description
                None,                 # expert_id
                resource_type,        # type
                None,                 # subtitles
                publishers,           # publishers
                None,                 # collection
                None,                 # date_issue
                None,                 # citation
                None,                 # language
                identifiers_json,     # identifiers
                source,               # source
                authors_json,         # authors
                publication_year      # publication_year
            )
            
            data.append(row)
            
            # Show progress for large datasets
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(works)} publications...")
                
        except Exception as e:
            print(f"Error processing publication {i}: {e}")
            continue
    
    return data

def insert_to_resources_table(data):
    """Insert data into resources_resource table"""
    if not data:
        print("No data to insert.")
        return
    
    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Column names for resources_resource table
    columns = [
        "id", "doi", "title", "abstract", "summary", "domains", 
        "topics", "description", "expert_id", "type", "subtitles", 
        "publishers", "collection", "date_issue", "citation", 
        "language", "identifiers", "source", "authors", "publication_year"
    ]
    
    # Create placeholders for INSERT
    placeholders = ", ".join(["%s"] * len(columns))
    columns_str = ", ".join(columns)
    
    total_inserted = 0
    print(f"Inserting {len(data)} records into resources_resource table...")
    
    # Insert data in batches of 50
    batch_size = 50
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        try:
            for row in batch:
                try:
                    cur.execute(f"""
                        INSERT INTO resources_resource ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (id) DO NOTHING
                    """, row)
                    total_inserted += 1
                except Exception as e:
                    print(f"Error inserting row: {e}")
                    continue
            
            # Commit after each batch
            conn.commit()
            print(f"Inserted {i + len(batch)}/{len(data)} records")
            
        except Exception as e:
            conn.rollback()
            print(f"Error during batch insert: {e}")
    
    print(f"Successfully inserted {total_inserted} records into resources_resource table.")
    
    # Close database connection
    cur.close()
    conn.close()

def main():
    start_time = time.time()
    
    try:
        # Set up database
        setup_database()
        
        # Find APHRC institution
        aphrc_id, aphrc_name = find_aphrc_institution()
        print(f"Working with: {aphrc_name} (ID: {aphrc_id})")
        
        # Get publications from API
        works = fallback_to_api(aphrc_id)
        
        if not works or len(works) < 5:
            print("Not enough APHRC publications found.")
            return
        
        # Process works for resources_resource table
        data = process_works_for_resources(works)
        
        # Insert into resources_resource table
        insert_to_resources_table(data)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Done! Processed {len(works)} publications in {elapsed_time:.2f} seconds.")
        print("Data has been inserted into the resources_resource table.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()