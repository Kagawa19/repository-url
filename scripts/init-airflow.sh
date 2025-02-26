#!/bin/bash

set -e

# Wait for database
until nc -z postgres 5432; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Initialize connections
airflow connections add 'postgres_default' \
    --conn-type 'postgres' \
    --conn-host 'postgres' \
    --conn-login "${POSTGRES_USER}" \
    --conn-password "${POSTGRES_PASSWORD}" \
    --conn-port 5432 \
    --conn-schema "${POSTGRES_DB}"

airflow connections add 'redis_default' \
    --conn-type 'redis' \
    --conn-host 'redis' \
    --conn-port 6379 \
    --conn-extra '{"db": 1}'

airflow connections add 'neo4j_default' \
    --conn-type 'neo4j' \
    --conn-host 'bolt://neo4j:7687' \
    --conn-login "${NEO4J_USER}" \
    --conn-password "${NEO4J_PASSWORD}"

# Initialize variables
airflow variables set WEBSITE_URL "${WEBSITE_URL:-https://aphrc.org}"
airflow variables set POSTGRES_HOST "${POSTGRES_HOST}"
airflow variables set POSTGRES_DB "${POSTGRES_DB}"
airflow variables set POSTGRES_USER "${POSTGRES_USER}"
airflow variables set POSTGRES_PASSWORD "${POSTGRES_PASSWORD}"
airflow variables set NEO4J_URI "${NEO4J_URI}"
airflow variables set NEO4J_USER "${NEO4J_USER}"
airflow variables set NEO4J_PASSWORD "${NEO4J_PASSWORD}"
airflow variables set REDIS_HOST "redis"
airflow variables set REDIS_PORT "6379"

echo "Airflow initialization complete!"