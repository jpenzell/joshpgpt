import webbrowser

# Replace these with your actual Shoeboxed client credentials
CLIENT_ID = '651d7ed8983746f6b60a20123efb4a76'
REDIRECT_URI = 'http://localhost:8000/callback'
STATE = 'xK9m2Pq5vR8nL3tY'  # Random 16-character string for security

# Construct the authorization URL with 'all' scope
auth_url = (
    f'https://id.shoeboxed.com/oauth/authorize?'
    f'client_id={CLIENT_ID}&'
    f'response_type=code&'
    f'redirect_uri={REDIRECT_URI}&'
    f'scope=all&'  # Just 'all' as shown in docs
    f'state={STATE}'
)

print('Authorization URL:', auth_url)

# Automatically open the URL in the default web browser
webbrowser.open(auth_url)
