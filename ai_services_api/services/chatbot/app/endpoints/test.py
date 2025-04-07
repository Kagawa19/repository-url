import requests
import json

# Test chat endpoint with the correct path
response = requests.post(
    "http://localhost:8000/chatbot/conversation/chat",  # The correct endpoint path
    headers={"X-User-ID": "1", "Content-Type": "application/json"},
    json={"message": "give me details on this publication The Estimated Incidence of Induced Abortion in Ethiopia, 2014: Changes in the Provision of Services Since 2008"}
)
print(json.dumps(response.json(), indent=2))