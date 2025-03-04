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
import json

# Database connection settings - update these to your production database
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),  # Your DB name
    "user": os.getenv("POSTGRES_USER", "postgres"),  # Your DB user
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),  # Your DB password
    "host": os.getenv("POSTGRES_HOST", "postgres"),  # Your DB host
    "port": "5432"
}

# OpenAlex S3 bucket
S3_BASE_URL = "https://openalex.s3.amazonaws.com"

def find_aphrc_institution():
    """Find the APHRC institution in OpenAlex"""
    print("Looking for APHRC institution in OpenAlex data...")
    
    # APHRC ID that worked previously
    aphrc_id = "I4210152772"
    print(f"Using established APHRC ID: {aphrc_id}")
    return aphrc_id, "African Population and Health Research Center"

def fallback_to_api(aphrc_id):
    """Try to get APHRC works directly from the API"""
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

def extract_work_data_for_resources(works):
    """Extract relevant data from OpenAlex works for resources_resource table"""
    data = []
    
    print(f"Preparing {len(works)} APHRC works for insertion into resources_resource...")
    for i, work in enumerate(works):
        # Generate id - use OpenAlex id number portion or index as fallback
        work_id = int(work.get("id", "").replace("https://openalex.org/W", "")[-8:]) 
        
        # Extract DOI
        doi = work.get("doi", "")
        
        # Extract title
        title = work.get("title", "Untitled APHRC Publication")
        
        # Extract abstract
        abstract_index = work.get("abstract_inverted_index", {})
        abstract = reconstruct_abstract(abstract_index) if abstract_index else ""
        
        # Set source as "openalex"
        source = "openalex"
        
        # Extract publication year
        publication_year = str(work.get("publication_year", ""))
        
        # Process authors into JSON format
        authors_list = []
        for authorship in work.get("authorships", []):
            author_name = authorship.get("author", {}).get("display_name", "")
            if author_name:
                authors_list.append({"name": author_name})
        authors_json = json.dumps(authors_list)
        
        # Extract journal info for publishers
        host_venue = work.get("host_venue", {})
        journal_name = host_venue.get("display_name", "")
        publishers = json.dumps([{"name": journal_name}]) if journal_name else None
        
        # Extract type
        resource_type = work.get("type", "journal-article")
        
        # Extract identifiers
        identifiers = {
            "openalex": work.get("id", ""),
            "doi": doi
        }
        identifiers_json = json.dumps(identifiers)
        
        # Create row data tuple with NULL for fields we don't have
        row = (
            work_id,                  # id
            doi,                      # doi
            title,                    # title
            abstract,                 # abstract
            None,                     # summary (NULL)
            None,                     # domains (NULL)
            None,                     # topics (NULL)
            None,                     # description (NULL)
            None,                     # expert_id (NULL)
            resource_type,            # type
            None,                     # subtitles (NULL)
            publishers,               # publishers
            None,                     # collection (NULL)
            None,                     # date_issue (NULL)
            None,                     # citation (NULL)
            None,                     # language (NULL)
            identifiers_json,         # identifiers
            source,                   # source
            authors_json,             # authors
            publication_year          # publication_year
        )
        data.append(row)
        
        # Show progress for large datasets
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(works)} works...")
    
    return data

def insert_to_resources_table(data):
    """Insert data into resources_resource table"""
    if not data:
        print("No data to insert.")
        return
    
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Column names matching the data tuple order
    columns = [
        "id", "doi", "title", "abstract", "summary", "domains", 
        "topics", "description", "expert_id", "type", "subtitles", 
        "publishers", "collection", "date_issue", "citation", 
        "language", "identifiers", "source", "authors", "publication_year"
    ]
    
    # Create placeholders for INSERT statement
    placeholders = ", ".join(["%s"] * len(columns))
    columns_str = ", ".join(columns)
    
    # Insert data row by row with ON CONFLICT DO NOTHING
    total_inserted = 0
    print(f"Inserting {len(data)} records into resources_resource table...")
    
    try:
        for i, row in enumerate(data):
            try:
                cur.execute(f"""
                    INSERT INTO resources_resource ({columns_str})
                    VALUES ({placeholders})
                    ON CONFLICT (id) DO NOTHING
                """, row)
                
                if i % 100 == 0:
                    conn.commit()
                    print(f"Committed {i}/{len(data)} records")
                
                total_inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"Error inserting row {i}: {e}")
        
        # Final commit
        conn.commit()
        print(f"Successfully inserted {total_inserted} records into resources_resource table.")
    except Exception as e:
        conn.rollback()
        print(f"Error during insertion: {e}")
    
    cur.close()
    conn.close()

def main():
    start_time = time.time()
    
    try:
        # Find APHRC institution
        aphrc_id, aphrc_name = find_aphrc_institution()
        print(f"Working with: {aphrc_name} (ID: {aphrc_id})")
        
        # Get APHRC works from API
        works = fallback_to_api(aphrc_id)
        
        # Prepare and insert data if works were found
        if works:
            data = extract_work_data_for_resources(works)
            insert_to_resources_table(data)
            
            elapsed_time = time.time() - start_time
            print(f"Done! Processed {len(works)} works in {elapsed_time:.2f} seconds.")
        else:
            print("No APHRC works found.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()