import requests
import json

# Test chat endpoint with the correct path
response = requests.post(
    "http://localhost:8000/chatbot/conversation/chat",  # The correct endpoint path
    headers={"X-User-ID": "1", "Content-Type": "application/json"},
    json={"message": "give me details on this publication Neighborhood, social isolation and mental health outcome among older people in Ghana"}
)
print(json.dumps(response.json(), indent=2))