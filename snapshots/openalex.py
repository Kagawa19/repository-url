import os
import requests
import psycopg2
import pandas as pd
from io import StringIO
import json
import time

# Database connection settings - update these to your production database
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),  # Your DB name
    "user": os.getenv("POSTGRES_USER", "postgres"),  # Your DB user
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),  # Your DB password
    "host": os.getenv("POSTGRES_HOST", "postgres"),  # Your DB host
    "port": "5432"
}

# APHRC searching - using the correct APHRC ID from OpenAlex
APHRC_ID = "I4210152772"  # Updated APHRC Institution ID
BASE_URL = f"https://api.openalex.org/works?filter=institutions.id:{APHRC_ID}"

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

def fetch_all_works():
    """
    Fetches all APHRC works from OpenAlex API using the correct institution ID
    """
    all_works = []
    page = 1
    per_page = 200  # Maximum allowed by OpenAlex
    has_more = True
    
    print("Starting to fetch APHRC works from OpenAlex...")
    print(f"Using institution ID: {APHRC_ID}")
    
    # Try different search methods if the first doesn't work
    search_methods = [
        # Method 1: By institution ID (primary method now that we have the correct ID)
        f"https://api.openalex.org/works?filter=institutions.id:{APHRC_ID}",
        # Method 2: By institution name exact match (more precise)
        "https://api.openalex.org/works?filter=institutions.display_name:African%20Population%20and%20Health%20Research%20Center",
        # Method 3: By author affiliation text search (specific)
        "https://api.openalex.org/works?filter=authorships.institutions.display_name:APHRC"
    ]
    
    # Try each search method until we find works
    for method_index, base_url in enumerate(search_methods):
        print(f"Trying search method {method_index + 1}...")
        
        url = f"{base_url}&page=1&per-page=5"  # Test with just 5 results
        try:
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error with method {method_index + 1}: {response.status_code}")
                continue
                
            data = response.json()
            total_count = data.get("meta", {}).get("count", 0)
            
            if total_count > 0 and total_count < 10000:  # Make sure count is reasonable
                print(f"Method {method_index + 1} found {total_count} works. Using this method.")
                # Reset for full fetch
                page = 1
                has_more = True
                
                # Now fetch all pages with this method
                while has_more:
                    print(f"Fetching page {page}...")
                    full_url = f"{base_url}&page={page}&per-page={per_page}"
                    response = requests.get(full_url)
                    
                    if response.status_code != 200:
                        print(f"Error fetching data: {response.status_code}")
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    all_works.extend(results)
                    
                    # Check if there are more pages
                    meta = data.get("meta", {})
                    current_page = meta.get("page", 0)
                    per_page_count = meta.get("per_page", 0)
                    
                    if len(results) < per_page or current_page * per_page_count >= total_count:
                        has_more = False
                    else:
                        page += 1
                        time.sleep(0.5)  # Be nice to the API
                
                # If we found works, stop trying methods
                break
            else:
                if total_count > 10000:
                    print(f"Method {method_index + 1} found {total_count} works, which is too many (likely not specific to APHRC). Trying next method.")
                else:
                    print(f"Method {method_index + 1} found no works. Trying next method.")
        except Exception as e:
            print(f"Error with method {method_index + 1}: {e}")
            continue
    
    print(f"Fetched {len(all_works)} works in total.")
    return all_works

def reconstruct_abstract(abstract_inverted_index):
    """
    Reconstruct abstract text from OpenAlex's inverted index format
    """
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

def process_works_for_resources(works):
    """
    Process works data for resources_resource table
    """
    data = []
    
    print(f"\nProcessing {len(works)} publications for resources_resource table...")
    
    for i, work in enumerate(works):
        try:
            # Extract ID for resources_resource table - need an integer
            # Use the numerical part of the OpenAlex ID as the primary key
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
                print(f"Processed {i + 1}/{len(works)} works...")
                
        except Exception as e:
            print(f"Error processing work {i}: {e}")
            continue
    
    return data

def insert_to_resources_table(data):
    """
    Insert data into resources_resource table
    """
    if not data:
        print("No data to insert.")
        return
    
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

def main():
    # Start timer
    start_time = time.time()
    
    try:
        # Fetch all works
        works = fetch_all_works()
        
        if not works:
            print("No works were found.")
            return
        
        # Process works for resources_resource table
        print("Processing works for resources_resource table...")
        data = process_works_for_resources(works)
        
        # Insert into resources_resource table
        print("Inserting into resources_resource table...")
        insert_to_resources_table(data)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Done! Processed {len(works)} works in {elapsed_time:.2f} seconds.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close connections
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()