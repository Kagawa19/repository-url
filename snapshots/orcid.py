#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
import time
from datetime import datetime

# Database connection settings - Update for your environment
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": "5432"
}

# ORCID API settings
ORCID_API_BASE = "https://pub.orcid.org/v3.0/"
SEARCH_API = "https://pub.orcid.org/v3.0/search/"

# APHRC affiliation keywords
APHRC_KEYWORDS = [
    "African Population and Health Research Center",
    "APHRC",
    "African Population and Health",
    "APHRC Nairobi"
]

def check_resources_table():
    """Check if resources_resource table exists"""
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
        cur.close()
        conn.close()
        return False
    
    print("resources_resource table found.")
    cur.close()
    conn.close()
    return True

def search_aphrc_researchers():
    """Search for APHRC-affiliated researchers using ORCID API"""
    all_orcids = []
    
    print("Searching for APHRC researchers in ORCID...")
    
    for keyword in APHRC_KEYWORDS:
        print(f"Searching for keyword: '{keyword}'")
        encoded_keyword = requests.utils.quote(keyword)
        
        # Search for the keyword in affiliation names
        search_url = f"{SEARCH_API}?q=affiliation-org-name:({encoded_keyword})"
        
        try:
            headers = {
                "Accept": "application/json"
            }
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("result", [])
                print(f"Found {len(results)} results for '{keyword}'")
                
                for result in results:
                    orcid_id = result.get("orcid-identifier", {}).get("path")
                    if orcid_id and orcid_id not in [r["orcid_id"] for r in all_orcids]:
                        all_orcids.append({"orcid_id": orcid_id, "keyword": keyword})
            else:
                print(f"Error searching for keyword '{keyword}': {response.status_code}")
                
            # Be nice to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"Error searching for keyword '{keyword}': {e}")
    
    print(f"Found {len(all_orcids)} unique ORCID IDs for APHRC researchers")
    return all_orcids

def fetch_researcher_works(orcid_id):
    """Fetch researcher works from ORCID API"""
    print(f"Fetching works for ORCID ID: {orcid_id}")
    
    # Get works data
    works_url = f"{ORCID_API_BASE}{orcid_id}/works"
    
    try:
        headers = {
            "Accept": "application/json"
        }
        response = requests.get(works_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching works data for {orcid_id}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching works data for {orcid_id}: {e}")
        return None

def fetch_researcher_data(orcid_id):
    """Fetch researcher data from ORCID API"""
    print(f"Fetching profile data for ORCID ID: {orcid_id}")
    
    # Get person data
    person_url = f"{ORCID_API_BASE}{orcid_id}/person"
    
    try:
        headers = {
            "Accept": "application/json"
        }
        response = requests.get(person_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching person data for {orcid_id}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching person data for {orcid_id}: {e}")
        return None

def extract_works_info(works_data, orcid_id, researcher_name):
    """Extract works information from ORCID data"""
    if not works_data:
        return []
    
    works_list = []
    
    try:
        # Get works group
        groups = works_data.get("group", [])
        
        for group in groups:
            work_summaries = group.get("work-summary", [])
            
            if work_summaries:
                # Take the first work summary (preferred)
                work = work_summaries[0]
                
                # Extract work details
                title = work.get("title", {}).get("title", {}).get("value", "")
                if not title:
                    continue  # Skip works without titles
                
                type_value = work.get("type", "")
                
                # Extract publication date
                publication_date = None
                publication_year = None
                pub_date = work.get("publication-date")
                if pub_date:
                    year = pub_date.get("year", {}).get("value", "")
                    month = pub_date.get("month", {}).get("value", "01")
                    day = pub_date.get("day", {}).get("value", "01")
                    
                    if year:
                        publication_year = year
                        try:
                            date_str = f"{year}-{month or '01'}-{day or '01'}"
                            publication_date = date_str
                        except:
                            publication_date = None
                
                # Extract journal title
                journal_title = work.get("journal-title", {}).get("value", "")
                
                # Extract external IDs
                external_ids = {}
                doi = None
                for ext_id in work.get("external-ids", {}).get("external-id", []):
                    id_type = ext_id.get("external-id-type", "")
                    id_value = ext_id.get("external-id-value", "")
                    if id_type and id_value:
                        external_ids[id_type] = id_value
                        if id_type == "doi":
                            doi = id_value
                
                # Extract URL
                url = None
                for ext_id in work.get("external-ids", {}).get("external-id", []):
                    if ext_id.get("external-id-type") == "url":
                        url = ext_id.get("external-id-value", "")
                        break
                
                # Create a resource object for insertion
                work_resource = {
                    "title": title,
                    "type": type_value,
                    "doi": doi,
                    "publication_date": publication_date,
                    "publication_year": publication_year,
                    "journal_title": journal_title,
                    "identifiers": external_ids,
                    "url": url,
                    "authors": [{"name": researcher_name, "orcid": orcid_id}],
                    "source": "orcid"
                }
                
                works_list.append(work_resource)
    except Exception as e:
        print(f"Error extracting works info: {e}")
    
    return works_list

def extract_person_info(person_data, orcid_id):
    """Extract person information from ORCID data"""
    if not person_data:
        return None
    
    try:
        # Get name information
        name_data = person_data.get("name", {})
        given_names = name_data.get("given-names", {}).get("value", "")
        family_name = name_data.get("family-name", {}).get("value", "")
        full_name = f"{given_names} {family_name}".strip()
        
        return {
            "full_name": full_name,
            "orcid_id": orcid_id
        }
    except Exception as e:
        print(f"Error extracting person info: {e}")
        return None

def insert_into_resources(work, next_id=None):
    """Insert work data into resources_resource table"""
    if not work:
        return
    
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    try:
        # Check if a work with this DOI already exists
        if work.get("doi"):
            cur.execute("SELECT id FROM resources_resource WHERE doi = %s", (work.get("doi"),))
            result = cur.fetchone()
            if result:
                print(f"Resource with DOI {work.get('doi')} already exists with ID {result[0]}")
                cur.close()
                conn.close()
                return result[0]
        
        # If no ID provided, get next available ID
        if next_id is None:
            cur.execute("SELECT MAX(id) FROM resources_resource")
            max_id = cur.fetchone()[0]
            next_id = (max_id or 0) + 1
        
        # Prepare publishers JSON (from journal title)
        publishers = None
        if work.get("journal_title"):
            publishers = json.dumps([{"name": work.get("journal_title")}])
        
        # Prepare identifiers JSON
        identifiers = None
        if work.get("identifiers"):
            identifiers = json.dumps(work.get("identifiers"))
        
        # Prepare authors JSON
        authors = None
        if work.get("authors"):
            authors = json.dumps(work.get("authors"))
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, doi, title, abstract, type, publishers, 
             identifiers, source, authors, publication_year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            work.get("doi"),
            work.get("title"),
            None,  # abstract not provided by ORCID
            work.get("type"),
            publishers,
            identifiers,
            work.get("source", "orcid"),
            authors,
            work.get("publication_year")
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            print(f"Inserted resource with ID {result[0]}: {work.get('title')}")
            return result[0]
        else:
            print(f"Failed to insert resource: {work.get('title')}")
            return None
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting resource: {e}")
        return None
    
    finally:
        cur.close()
        conn.close()

def main():
    start_time = time.time()
    
    try:
        # Check if resources_resource table exists
        if not check_resources_table():
            return
        
        # Search for APHRC researchers
        researchers = search_aphrc_researchers()
        
        if not researchers:
            print("No APHRC researchers found in ORCID database.")
            return
        
        total_resources = 0
        
        # Process each researcher
        for researcher in researchers:
            orcid_id = researcher["orcid_id"]
            
            # Fetch researcher profile data
            person_data = fetch_researcher_data(orcid_id)
            if not person_data:
                continue
            
            # Extract person info
            researcher_info = extract_person_info(person_data, orcid_id)
            if not researcher_info:
                continue
            
            researcher_name = researcher_info["full_name"]
            
            # Fetch researcher works
            works_data = fetch_researcher_works(orcid_id)
            if not works_data:
                continue
            
            # Extract works info
            works_info = extract_works_info(works_data, orcid_id, researcher_name)
            
            # Insert works into resources_resource table
            for work in works_info:
                insert_into_resources(work)
                total_resources += 1
            
            print(f"Processed {len(works_info)} works for {researcher_name} (ORCID: {orcid_id})")
            
            # Be nice to the API
            time.sleep(1)
        
        elapsed_time = time.time() - start_time
        print(f"Done! Processed {len(researchers)} researchers and added {total_resources} resources in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()