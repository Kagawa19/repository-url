import requests
import certifi

api_url = "https://knowhub.aphrc.org/rest/items/1602"

# Use certifi's CA bundle
response = requests.get(api_url, verify=certifi.where())

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
