import requests
import json

# Test chat endpoint with the correct path
response = requests.post(
    "http://localhost:8000/chatbot/conversation/chat",  # The correct endpoint path
    headers={"X-User-ID": "1", "Content-Type": "application/json"},
    json={"message": "give me a publication on Global, regional, and national incidence, prevalence, and years lived with disability for 310 diseases and injuries"}
)
print(json.dumps(response.json(), indent=2))