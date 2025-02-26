
curl -X GET "http://localhost:8000/recommendation/recommendation/recommend" -H "X-User-ID: 1"

# For FERNET_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# For SECRET_KEY
python -c "import secrets; print(secrets.token_hex(16))"

docker-compose -f docker-compose.debug.yml up postgres redis neo4j -d
docker-compose -f docker-compose.debug.yml up api