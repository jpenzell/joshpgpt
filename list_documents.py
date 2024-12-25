import requests
import json

ACCESS_TOKEN = "b9dce454-c073-43de-95b4-15cade219d1d"
REFRESH_TOKEN = "fd9c1bc5-7520-4369-a957-3dc09d1a5d33"

# Base API URL
BASE_URL = "https://api.shoeboxed.com/v2"

# Test endpoints
endpoints = {
    "documents": "/documents",
    "user": "/user"  # Let's try getting user info first as a test
}

def make_request(endpoint):
    url = BASE_URL + endpoint
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json"
    }
    
    print(f"\nTrying endpoint: {url}")
    print("Headers:", json.dumps(headers, indent=2))
    
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print("Response Headers:", json.dumps(dict(response.headers), indent=2))
    print("Response Body:", response.text)
    
    return response

# Try both endpoints
for name, endpoint in endpoints.items():
    print(f"\nTesting {name} endpoint...")
    response = make_request(endpoint)
