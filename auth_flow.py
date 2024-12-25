import http.server
import socketserver
import webbrowser
import urllib.parse
import requests
import secrets
import json
from datetime import datetime, timedelta
import time

# Configuration
CLIENT_ID = '651d7ed8983746f6b60a20123efb4a76'
CLIENT_SECRET = 'Y8SRJP225pofXhFh2rm94.Um8FHTSHgJ9YaVWdxbLNMsEaSSHWeHa'
REDIRECT_URI = 'http://localhost:8000/callback'
AUTH_URL = 'https://id.shoeboxed.com/oauth/authorize'
TOKEN_URL = 'https://id.shoeboxed.com/oauth/token'

# Generate a secure state token
state_token = secrets.token_urlsafe(16)
auth_code = None

class OAuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/callback'):
            # Parse the query parameters
            query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            
            # Check if there's an error
            if 'error' in query_components:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Error: {query_components['error'][0]}".encode())
                return

            # Verify state token
            received_state = query_components.get('state', [None])[0]
            if received_state != state_token:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State token mismatch - possible CSRF attack")
                return

            # Get the authorization code
            global auth_code
            auth_code = query_components.get('code', [None])[0]
            
            # Send success response
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this window.")
            
            # Stop the server
            self.server.server_close()

def get_tokens(authorization_code):
    """Exchange authorization code for tokens"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = {
                'code': authorization_code,
                'grant_type': 'authorization_code',
                'redirect_uri': REDIRECT_URI,
                'scope': 'all'
            }
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Origin': 'https://id.shoeboxed.com',
                'Referer': 'https://id.shoeboxed.com/',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            print(f"\nAttempting to get tokens (attempt {attempt + 1}/{max_retries})")
            print(f"Request data: {json.dumps(data, indent=2)}")
            
            response = requests.post(
                TOKEN_URL,
                data=data,
                auth=(CLIENT_ID, CLIENT_SECRET),
                headers=headers,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
            
            if response.status_code == 200:
                tokens = response.json()
                # Add expiration time and scope
                tokens['expires_at'] = (datetime.now() + timedelta(seconds=tokens['expires_in'])).isoformat()
                tokens['scope'] = 'all'
                
                # Save tokens to file
                with open('tokens.json', 'w') as f:
                    json.dump(tokens, f, indent=2)
                
                print("Successfully obtained tokens")
                return tokens
            elif response.status_code == 401:
                print("Authentication failed - invalid client credentials")
                raise Exception("Invalid client credentials")
            else:
                print(f"Failed to get tokens: {response.text}")
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                raise Exception(f"Failed to get tokens after {max_retries} attempts")
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {str(e)}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise Exception(f"Network error after {max_retries} attempts: {str(e)}")
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise

def refresh_access_token(refresh_token):
    """Get new access token using refresh token"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'scope': 'all'
            }
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Origin': 'https://id.shoeboxed.com',
                'Referer': 'https://id.shoeboxed.com/',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            print(f"\nAttempting to refresh token (attempt {attempt + 1}/{max_retries})")
            response = requests.post(
                TOKEN_URL,
                data=data,
                headers=headers,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                tokens = response.json()
                # Add expiration time and scope
                tokens['expires_at'] = (datetime.now() + timedelta(seconds=tokens['expires_in'])).isoformat()
                tokens['scope'] = 'all'
                
                # Save updated tokens
                with open('tokens.json', 'w') as f:
                    json.dump(tokens, f, indent=2)
                
                print("Successfully refreshed token")
                return tokens
            elif response.status_code == 401:
                print("Refresh token is invalid or expired")
                raise Exception("Invalid refresh token")
            else:
                print(f"Failed to refresh token: {response.text}")
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                raise Exception(f"Failed to refresh token after {max_retries} attempts")
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {str(e)}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise Exception(f"Network error after {max_retries} attempts: {str(e)}")
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise

def is_token_expired(tokens):
    """Check if the current token is expired"""
    if not tokens or 'expires_at' not in tokens:
        return True
    expires_at = datetime.fromisoformat(tokens['expires_at'])
    # Add 5-minute buffer
    return datetime.now() + timedelta(minutes=5) >= expires_at

def ensure_valid_token():
    """Ensure we have a valid token, refreshing if necessary"""
    try:
        with open('tokens.json', 'r') as f:
            tokens = json.load(f)
            
        if is_token_expired(tokens):
            print("Token is expired or about to expire, refreshing...")
            return refresh_access_token(tokens['refresh_token'])
        return tokens
    except FileNotFoundError:
        print("No tokens file found")
        return None
    except Exception as e:
        print(f"Error ensuring valid token: {str(e)}")
        return None

def main():
    # Construct authorization URL
    auth_params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': 'all',
        'state': state_token
    }
    full_auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"
    
    # Start local server to handle the callback
    with socketserver.TCPServer(("", 8000), OAuthHandler) as httpd:
        print("Starting server at http://localhost:8000")
        print("Opening browser for authorization...")
        webbrowser.open(full_auth_url)
        
        # Wait for the callback
        httpd.handle_request()
    
    if auth_code:
        print("Getting tokens...")
        tokens = get_tokens(auth_code)
        print("Authentication successful!")
        print(f"Access token: {tokens['access_token']}")
        print(f"Refresh token: {tokens['refresh_token']}")
        print(f"Token expires at: {tokens['expires_at']}")
    else:
        print("Failed to get authorization code")

if __name__ == "__main__":
    main() 