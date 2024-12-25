import os
import streamlit as st
from openai import OpenAI
import json
import time
import webbrowser
import secrets
import urllib.parse
import http.server
import socketserver
import threading
import base64
from pdf2image import convert_from_bytes
from io import BytesIO
import traceback
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from fetch_documents import fetch_documents
from process_for_pinecone import extract_text_from_pdf, create_embedding, ProcessingState

# Load environment variables
load_dotenv()

# Constants from environment variables
CLIENT_ID = os.getenv('SHOEBOXED_CLIENT_ID')
CLIENT_SECRET = os.getenv('SHOEBOXED_CLIENT_SECRET')
AUTH_URL = os.getenv('SHOEBOXED_AUTH_URL')
TOKEN_URL = os.getenv('SHOEBOXED_TOKEN_URL')
REDIRECT_URI = os.getenv('SHOEBOXED_REDIRECT_URI')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_INDEX_HOST = os.getenv('PINECONE_INDEX_HOST')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def load_tokens():
    """Load tokens from file"""
    try:
        with open('.auth_success', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception("No tokens found. Please authenticate first.")

def refresh_if_needed(tokens):
    """Check if token needs refresh and refresh if necessary"""
    try:
        expires_at = datetime.fromisoformat(tokens.get('expires_at', datetime.now().isoformat()))
        if datetime.now() >= expires_at:
            # Implement token refresh logic here
            # For now, we'll just re-authenticate
            print("Token expired. Please re-authenticate.")
            return tokens
        return tokens
    except Exception as e:
        print(f"Error checking token expiration: {str(e)}")
        return tokens

def init_session_state():
    """Initialize session state variables"""
    # Initialize authentication state if not present
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check for existing auth token
    if os.path.exists('.auth_success'):
        try:
            with open('.auth_success', 'r') as f:
                auth_data = json.load(f)
                # Check if token exists and is not expired
                if auth_data.get('access_token'):
                    # Verify token with Shoeboxed
                    headers = {
                        'Authorization': f'Bearer {auth_data["access_token"]}',
                        'Accept': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Origin': 'https://api.shoeboxed.com',
                        'Referer': 'https://api.shoeboxed.com/',
                        'Accept-Language': 'en-US,en;q=0.9'
                    }
                    response = requests.get('https://api.shoeboxed.com/v2/user', headers=headers)
                    print(f"Verification response: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    print(f"Response body: {response.text}")
                    
                    if response.status_code == 200:
                        st.session_state.authenticated = True
                        print("Successfully verified existing auth token")
                    else:
                        print(f"Token verification failed: {response.status_code}")
                        print(f"Response: {response.text}")
                        os.remove('.auth_success')
                        st.session_state.authenticated = False
        except Exception as e:
            print(f"Error checking auth token: {str(e)}")
            if os.path.exists('.auth_success'):
                os.remove('.auth_success')
            st.session_state.authenticated = False
    
    # Initialize other session state variables
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

def init_pinecone():
    """Initialize Pinecone client"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(
            name=PINECONE_INDEX_NAME,
            host=PINECONE_INDEX_HOST
        )
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

def extract_text_with_gpt4o(image_base64):
    """Extract text from image using GPT-4 Vision"""
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_VISION_MODEL'),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract all text from this receipt or document image. Include all numbers, dates, and details. Format it clearly."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in GPT-4 Vision text extraction: {str(e)}")
        return None

class OAuthHandler(http.server.BaseHTTPRequestHandler):
    """Handle OAuth callback"""
    def do_GET(self):
        """Handle GET request"""
        try:
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            
            # Get authorization code
            code = params.get('code', [''])[0]
            state = params.get('state', [''])[0]
            
            if code and state:
                # Exchange code for tokens
                success = exchange_code_for_tokens(code)
                
                if success:
                    # Force session state update
                    st.session_state.authenticated = True
                    print("Authentication successful, updating session state")
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = """
                    <html>
                        <body>
                            <h1>Authentication Successful!</h1>
                            <p>You can now close this window and return to the application.</p>
                            <script>
                                window.onload = function() {
                                    // Force reload of the main window
                                    if (window.opener) {
                                        window.opener.location.href = window.opener.location.href;
                                    }
                                    // Close this window after a short delay
                                    setTimeout(function() {
                                        window.close();
                                    }, 1000);
                                };
                            </script>
                        </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                    
                    # Signal the server to stop
                    self.server.should_stop = True
                else:
                    print("Token exchange failed")
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Error exchanging code for tokens")
            else:
                print("Missing code or state parameter")
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Missing code or state parameter")
                
        except Exception as e:
            print(f"Error in callback handler: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Internal server error")

def start_callback_server(state_token):
    """Start the callback server for OAuth"""
    port = 8000
    max_retries = 5
    server = None
    
    class StoppableServer(socketserver.TCPServer):
        def __init__(self, *args, **kwargs):
            self.should_stop = False
            super().__init__(*args, **kwargs)
        
        def serve_forever(self):
            while not self.should_stop:
                self.handle_request()
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)  # Wait a bit between retries
            server = StoppableServer(("", port), OAuthHandler)
            print(f"Server started successfully on port {port}")
            return server, port
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"Port {port} is busy, waiting for it to be released...")
                if attempt < max_retries - 1:
                    continue
            raise
        except Exception as e:
            print(f"Error starting server: {str(e)}")
            raise
    
    if not server:
        raise Exception(f"Failed to start server after {max_retries} attempts")

def handle_auth():
    """Handle the authentication flow"""
    try:
        # Generate state token
        state = secrets.token_urlsafe(32)
        
        # Start callback server
        server, port = start_callback_server(state)
        
        # Build authorization URL with correct scope
        params = {
            'client_id': CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': REDIRECT_URI,
            'scope': 'all',  # Just use 'all' as per documentation
            'state': state,
            'access_type': 'offline'
        }
        auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
        
        print("Opening browser for authorization...")
        print(f"Authorization URL: {auth_url}")
        
        # Open browser for authorization
        webbrowser.open(auth_url)
        
        # Start server and wait for callback
        server.serve_forever()
        
        # Server will stop when authentication is complete
        server.server_close()
        
        # Check if authentication was successful
        if os.path.exists('.auth_success'):
            with open('.auth_success', 'r') as f:
                tokens = json.load(f)
                if tokens.get('access_token'):
                    # Verify token with proper headers
                    headers = {
                        'Authorization': f'Bearer {tokens["access_token"]}',
                        'Accept': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Origin': 'https://api.shoeboxed.com',
                        'Referer': 'https://api.shoeboxed.com/',
                        'Accept-Language': 'en-US,en;q=0.9'
                    }
                    response = requests.get('https://api.shoeboxed.com/v2/user', headers=headers)
                    print(f"Verification response: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    print(f"Response body: {response.text}")
                    
                    if response.status_code == 200:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error(f"Token verification failed: {response.status_code}")
                else:
                    st.error("Authentication failed: No access token received")
        else:
            st.error("Authentication failed: No tokens file created")
        
    except Exception as e:
        error_message = f"Authentication failed: {str(e)}"
        print(error_message)
        print(f"Traceback: {traceback.format_exc()}")
        st.error(error_message)
        st.session_state.authenticated = False

def process_documents():
    """Process documents from Shoeboxed in batches"""
    try:
        # Load tokens
        tokens = load_tokens()
        
        # Refresh token if needed
        tokens = refresh_if_needed(tokens)
        
        # Fetch document IDs first
        print("🚀 Starting document ID retrieval...")
        try:
            account_id = get_organization_id(tokens['access_token'])
            print(f"✅ Retrieved Account ID: {account_id}")
        except Exception as account_error:
            print(f"❌ Failed to retrieve account ID: {str(account_error)}")
            st.error(f"Could not retrieve account ID: {str(account_error)}")
            return
        
        list_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents"
        headers = {
            'Authorization': f'Bearer {tokens["access_token"]}',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Batch processing parameters
        batch_size = 10
        offset = 0
        all_document_ids = []
        
        # Collect all document IDs
        while True:
            params = {
                'limit': 50,  # Fetch 50 document IDs at a time
                'offset': offset,
                'include': 'metadata'
            }
            
            print(f"📡 Fetching documents with params: {params}")
            
            try:
                response = requests.get(list_url, headers=headers, params=params)
                
                print(f"📥 Response Status: {response.status_code}")
                print(f"📥 Response Headers: {response.headers}")
                
                if response.status_code != 200:
                    print(f"❌ Failed to fetch document list. Status: {response.status_code}")
                    print(f"❌ Response Body: {response.text}")
                    st.error(f"Failed to fetch documents. Status: {response.status_code}")
                    break
                
                data = response.json()
                doc_list = data.get('documents', [])
                
                print(f"📊 Documents in this batch: {len(doc_list)}")
                
                # Collect document IDs
                batch_ids = [doc['id'] for doc in doc_list]
                all_document_ids.extend(batch_ids)
                
                # Check if there are more documents
                if len(doc_list) < params['limit']:
                    break
                
                offset += params['limit']
                
            except requests.RequestException as req_error:
                print(f"❌ Network error during document retrieval: {str(req_error)}")
                st.error(f"Network error: {str(req_error)}")
                break
            except Exception as general_error:
                print(f"❌ Unexpected error during document retrieval: {str(general_error)}")
                st.error(f"Unexpected error: {str(general_error)}")
                break
        
        print(f"📈 Total documents found: {len(all_document_ids)}")
        
        # Initialize Pinecone
        print("🔢 Initializing Pinecone vector database...")
        index = init_pinecone()
        
        # Initialize processing state
        state = ProcessingState()
        state.load_progress()
        
        # Create a timestamp-based directory for documents
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        documents_dir = f'documents_{timestamp}'
        os.makedirs(documents_dir, exist_ok=True)
        
        # Process documents in batches
        processed_count = 0
        failed_count = 0
        
        for i in range(0, len(all_document_ids), batch_size):
            # Get batch of document IDs
            batch_ids = all_document_ids[i:i+batch_size]
            batch_documents = []
            
            # Fetch details for each document in the batch
            print(f"\n🔍 Fetching details for batch {i//batch_size + 1}")
            for doc_id in batch_ids:
                try:
                    doc_url = f"{list_url}/{doc_id}"
                    doc_response = requests.get(doc_url, headers=headers)
                    
                    if doc_response.status_code == 200:
                        doc_details = doc_response.json()
                        batch_documents.append(doc_details)
                        print(f"✅ Fetched details for document {doc_id}")
                    else:
                        print(f"❌ Failed to fetch details for document {doc_id}")
                    
                    time.sleep(0.5)  # Be nice to the API
                except Exception as e:
                    print(f"❌ Error fetching document {doc_id}: {str(e)}")
            
            # Process this batch of documents
            print(f"🔢 Processing batch of {len(batch_documents)} documents")
            for doc in batch_documents:
                try:
                    # Process single document
                    success = process_single_document(doc, documents_dir, index, state)
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    # Optional: Add a small delay to prevent overwhelming the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error processing individual document: {str(e)}")
                    failed_count += 1
            
            # Update Streamlit status
            st.session_state.processing_status = f"Processed {processed_count} out of {len(all_document_ids)} documents"
        
        # Final summary
        print("\n🏁 Document Processing Complete!")
        print(f"📈 Total Documents: {len(all_document_ids)}")
        print(f"✅ Successfully Processed: {processed_count}")
        print(f"❌ Failed to Process: {failed_count}")
        
        # Update session state
        st.session_state.processing_complete = True
        st.session_state.processing_status = f"Processed {processed_count} out of {len(all_document_ids)} documents"
        
    except Exception as e:
        print(f"❌ Critical error in document processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"Document processing failed: {str(e)}")

def chat_interface():
    """Chat interface for interacting with processed documents"""
    try:
        # Initialize Pinecone
        index = init_pinecone()
        if not index:
            st.error("Failed to initialize Pinecone")
            return
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents"):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Create embedding for the query
            query_embedding = create_embedding(prompt)
            if not query_embedding:
                st.error("Failed to create query embedding")
                return
            
            # Search Pinecone with improved parameters
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=os.getenv('PINECONE_INDEX_NAME')
            )
            
            # Format context from search results
            context = []
            for match in results['matches']:
                metadata = match['metadata']
                score = match['score']
                context.append(f"""
                Document (Relevance: {score:.2f})
                Category: {metadata.get('category', 'Unknown')}
                Date: {metadata.get('created', 'Unknown')}
                Amount: ${metadata.get('total', 'N/A')}
                Content: {metadata.get('text', 'No text available')}
                ---
                """)
            
            # Generate response using GPT-4
            system_message = """You are an AI assistant analyzing financial documents. 
            Use the provided document excerpts to answer questions accurately and concisely.
            Include specific dates, amounts, and categories when relevant.
            If information is unclear or missing, say so."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Based on these documents:\n\n{''.join(context)}\n\nQuestion: {prompt}"}
            ]
            
            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model=os.getenv('OPENAI_CHAT_MODEL'),
                    messages=messages,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                
    except Exception as e:
        st.error(f"Error in chat interface: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

def exchange_code_for_tokens(code):
    """Exchange authorization code for access and refresh tokens"""
    try:
        # Use form-encoded data instead of JSON
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'scope': 'all'
        }
        
        print("Attempting token exchange...")
        print(f"Token URL: {TOKEN_URL}")
        print(f"Redirect URI: {REDIRECT_URI}")
        
        # Set proper headers to look like a browser
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://id.shoeboxed.com',
            'Referer': 'https://id.shoeboxed.com/'
        }
        
        # Make the token exchange request
        response = requests.post(TOKEN_URL, data=data, headers=headers)
        
        print(f"Token exchange response status: {response.status_code}")
        print(f"Token exchange response headers: {dict(response.headers)}")
        print(f"Token exchange response body: {response.text}")
        
        if response.status_code == 200:
            token_data = response.json()
            # Add expiration time
            token_data['expires_at'] = (datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))).isoformat()
            
            # Save tokens
            with open('.auth_success', 'w') as f:
                json.dump(token_data, f)
            return True
        else:
            print(f"Error exchanging code for tokens (HTTP {response.status_code}): {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception during token exchange: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def generate_auth_url():
    """Generate the authorization URL"""
    state_token = secrets.token_urlsafe(32)
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': 'user.read organization.read document.read document.write document.delete document.categorize document.export document.process document.share document.stats document.verify document.view document.search document.suggest document.process.ocr document.process.ocr.suggest document.process.ocr.verify document.process.ocr.view document.process.ocr.search document.process.ocr.suggest',
        'state': state_token,
        'access_type': 'offline'
    }
    return AUTH_URL + '?' + urllib.parse.urlencode(params), state_token

def main():
    """Main application function"""
    st.title("Shoeboxed Document Processor")
    
    # Initialize session state
    init_session_state()
    
    # Debug information
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Authentication Status: {st.session_state.authenticated}")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Main authentication flow
    if not st.session_state.authenticated:
        st.info("Please authenticate with Shoeboxed to begin.")
        if st.button("🔑 Authenticate with Shoeboxed"):
            handle_auth()
    else:
        st.sidebar.success("✅ Authenticated")
        
        # Add processing controls
        if st.sidebar.button("📄 Process Documents"):
            with st.spinner("Processing documents..."):
                process_documents()
            
        # Add chat interface
        if st.sidebar.button("💬 Chat with Documents"):
            chat_interface()
        
        # Add logout button
        if st.sidebar.button("🚪 Logout"):
            if os.path.exists('.auth_success'):
                os.remove('.auth_success')
            st.session_state.authenticated = False
            st.session_state.chat_messages = []  # Clear chat history
            st.session_state.processing_complete = False  # Reset processing state
            st.rerun()
        
        # Main content area
        st.write("Ready to process documents or chat!")
        
        # Show processing status if available
        if hasattr(st.session_state, 'processing_status'):
            st.write(st.session_state.processing_status)

# Add cleanup on app exit
def on_shutdown():
    """Clean up resources when the app shuts down"""
    if os.path.exists('.auth_success'):
        os.remove('.auth_success')
    if os.path.exists('processing_progress.json'):
        os.remove('processing_progress.json')

# Register cleanup
import atexit
atexit.register(on_shutdown)

# Add this helper function to get organization ID
def get_organization_id(access_token):
    """Get the organization ID for the authenticated user"""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get('https://api.shoeboxed.com/v2/user', headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            # Get the first account ID from the user's accounts
            if 'accounts' in user_data and len(user_data['accounts']) > 0:
                account_id = user_data['accounts'][0].get('id')
                if account_id:
                    return account_id
        
        raise Exception(f"No account ID found in user data. User response: {response.text}")
        
    except Exception as e:
        print(f"Error getting account ID: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
