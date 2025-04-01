import requests
import json

# Test chat endpoint with the correct path
response = requests.post(
    "http://localhost:8000/chatbot/conversation/chat",  # The correct endpoint path
    headers={"X-User-ID": "1", "Content-Type": "application/json"},
    json={"message": "i need a summary of a covid 19 publication?"}
)
print(json.dumps(response.json(), indent=2))