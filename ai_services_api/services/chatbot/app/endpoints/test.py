import requests
import json

# Test chat endpoint with the correct path
response = requests.post(
    "http://localhost:8000/chatbot/conversation/chat",  # The correct endpoint path
    headers={"X-User-ID": "1", "Content-Type": "application/json"},
    json={"message": "give me a summary of any of the chilrens health publications"}
)
print(json.dumps(response.json(), indent=2))