import requests
import base64

CLIENT_ID = '651d7ed8983746f6b60a20123efb4a76'
CLIENT_SECRET = 'Y8SRJP225pofXhFh2rm94.Um8FHTSHgJ9YaVWdxbLNMsEaSSHWeHa'
REDIRECT_URI = 'http://localhost:8000/callback'
AUTHORIZATION_CODE = 'cc910aaf-872b-48bd-a718-b58e9acf1776'

token_url = 'https://id.shoeboxed.com/oauth/token'

# Request only 'internal' scope during token exchange
data = {
    'code': AUTHORIZATION_CODE,
    'grant_type': 'authorization_code',
    'redirect_uri': REDIRECT_URI,
    'scope': 'all'  # Using 'all' scope consistently
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://id.shoeboxed.com',
    'Referer': 'https://id.shoeboxed.com/',
    'Content-Type': 'application/x-www-form-urlencoded'
}

auth = (CLIENT_ID, CLIENT_SECRET)

response = requests.post(token_url, data=data, headers=headers, auth=auth)
print('Response Status Code:', response.status_code)
print('Response Text:', response.text)

if response.status_code == 200:
    tokens = response.json()
    print('Access Token:', tokens.get('access_token'))
    print('Refresh Token:', tokens.get('refresh_token'))
else:
    print('Error: Failed to get tokens')
