#!/bin/bash

# Make script executable
chmod +x run_orcid_extract.sh

echo "APHRC Researchers ORCID Data Extraction"
echo "--------------------------------------"
echo

# Copy the script to the container
echo "Copying extraction script to container..."
docker cp extract_aphrc_orcids.py repository-url-api-1:/app/

# Install required packages
echo "Installing required packages..."
docker exec -it repository-url-api-1 pip install requests pandas psycopg2-binary

# Run the extraction script
echo "Running ORCID extraction script..."
docker exec -it repository-url-api-1 python /app/snapshots/orcid.py

echo
echo "Process complete!"
echo "APHRC researchers and their works have been extracted from ORCID and stored in the database."
echo "You can query the data using:"
echo "  - docker exec -it repository-url-api-1 psql -U postgres -d aphrc -c \"SELECT * FROM aphrc_researchers LIMIT 10;\""
echo "  - docker exec -it repository-url-api-1 psql -U postgres -d aphrc -c \"SELECT * FROM aphrc_researcher_works LIMIT 10;\""