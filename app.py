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
import logging
import boto3

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

def extract_text_with_gpt4o(pdf_bytes):
    """Extract text from PDF using GPT-4o Vision"""
    try:
        # Ensure we have bytes
        if not isinstance(pdf_bytes, bytes):
            logging.error("Input must be bytes")
            return None
            
        # Create a BytesIO object from the bytes
        pdf_stream = BytesIO(pdf_bytes)
        
        # Convert PDF to images
        try:
            images = convert_from_bytes(pdf_stream.read())
        except Exception as e:
            logging.error(f"Error converting PDF to images: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
        
        # Process first page (or all pages if needed)
        image_texts = []
        for i, image in enumerate(images):
            try:
                # Convert image to bytes
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                
                # Base64 encode the image
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Call GPT-4o
                response = client.chat.completions.create(
                    model=os.getenv('OPENAI_VISION_MODEL', "gpt-4-vision-preview"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this document comprehensively. Include numbers, dates, and key details. Provide a structured, clear extraction."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.2,
                    top_p=0.1,  # More focused response
                    frequency_penalty=0.1,  # Reduce repetition
                    presence_penalty=0.1    # Encourage novel information
                )
                
                page_text = response.choices[0].message.content
                if page_text:
                    image_texts.append(f"Page {i+1}:\n{page_text}")
                
            except Exception as e:
                logging.error(f"Error processing page {i}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Combine texts from all pages
        return "\n\n".join(image_texts) if image_texts else None
    
    except Exception as e:
        logging.error(f"Error in GPT-4o document text extraction: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
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

def retrieve_all_document_ids(tokens):
    """
    Retrieve all document IDs with robust, resumable mechanism
    
    Supports:
    - Resumable pagination
    - Detailed filtering
    - Comprehensive logging
    """
    # Determine cache file path
    cache_file = 'document_id_cache.json'
    
    # Try to load existing progress
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            last_offset = cache_data.get('last_offset', 0)
            cached_document_ids = cache_data.get('document_ids', [])
    except FileNotFoundError:
        last_offset = 0
        cached_document_ids = []
    
    # Get account ID
    account_id = get_organization_id(tokens['access_token'])
    list_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents"
    
    headers = {
        'Authorization': f'Bearer {tokens["access_token"]}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Batch processing parameters
    batch_size = 100  # Maximum allowed by API
    total_documents_retrieved = len(cached_document_ids)
    
    print(f"🔍 Resuming document ID retrieval from offset {last_offset}")
    print(f"📦 Already cached: {total_documents_retrieved} document IDs")
    
    try:
        while True:
            # Enhanced query parameters
            params = {
                'offset': last_offset,
                'limit': batch_size,
                'order_by_desc': 'uploaded',  # Sort by most recently uploaded first
                'trashed': 'false'  # Only exclude trashed documents
            }
            
            print(f"📡 Fetching document batch starting at offset {last_offset}")
            print(f"📋 Query Parameters: {params}")
            
            response = requests.get(list_url, headers=headers, params=params)
            
            # Detailed logging of API response
            print(f"🌐 Response Status: {response.status_code}")
            print(f"🔑 Response Headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch document list. Status: {response.status_code}")
                print(f"Response Body: {response.text}")
                break
            
            # Parse response
            data = response.json()
            
            # Log total document count
            total_count = data.get('totalCount', 0)
            total_filtered_count = data.get('totalCountFiltered', 0)
            print(f"📊 Total Documents: {total_count}")
            print(f"🔍 Filtered Documents: {total_filtered_count}")
            
            # Extract document list
            doc_list = data.get('documents', [])
            
            if not doc_list:
                print("🏁 No more documents to retrieve")
                break  # No more documents
            
            # Extract and log document details
            new_doc_ids = []
            for doc in doc_list:
                doc_id = doc.get('id')
                doc_type = doc.get('type', 'Unknown')
                doc_uploaded = doc.get('uploaded', 'Unknown')
                doc_vendor = doc.get('vendor', 'N/A')
                
                print(f"📄 Document: {doc_id}")
                print(f"   Type: {doc_type}")
                print(f"   Uploaded: {doc_uploaded}")
                print(f"   Vendor: {doc_vendor}")
                
                new_doc_ids.append(doc_id)
            
            # Add new document IDs
            cached_document_ids.extend(new_doc_ids)
            
            # Update progress
            total_documents_retrieved = len(cached_document_ids)
            last_offset += batch_size
            
            # Save progress periodically
            if total_documents_retrieved % (batch_size * 10) == 0:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'last_offset': last_offset,
                        'document_ids': cached_document_ids,
                        'timestamp': datetime.now().isoformat(),
                        'total_count': total_count,
                        'filtered_count': total_filtered_count
                    }, f, indent=2)
                
                print(f"💾 Saved progress: {total_documents_retrieved} document IDs")
            
            # Optional: Add a small delay to be nice to the API
            time.sleep(0.5)
            
            # Break if we've retrieved all documents
            if last_offset >= total_count:
                print("🏁 Retrieved all documents")
                break
    
    except Exception as e:
        print(f"❌ Error retrieving document IDs: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Final save of progress
        with open(cache_file, 'w') as f:
            json.dump({
                'last_offset': last_offset,
                'document_ids': cached_document_ids,
                'timestamp': datetime.now().isoformat(),
                'total_count': total_count,
                'filtered_count': total_filtered_count
            }, f, indent=2)
    
    print(f"🏁 Document ID Retrieval Complete")
    print(f"📊 Total Documents Found: {total_documents_retrieved}")
    
    return cached_document_ids

def get_shoeboxed_headers(access_token=None):
    """Generate headers for Shoeboxed API requests"""
    if not access_token:
        tokens = load_tokens()
        access_token = tokens.get('access_token')
    
    return {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Origin': 'https://api.shoeboxed.com',
        'Referer': 'https://api.shoeboxed.com/'
    }

def download_document(doc_details, tokens):
    """
    Download document from Shoeboxed or S3
    
    Args:
        doc_details (dict): Document details from Shoeboxed
        tokens (dict): Authentication tokens
    
    Returns:
        bytes or None: Document content or None if download fails
    """
    try:
        doc_id = doc_details.get('id', 'Unknown')
        
        # Get attachment URL from document metadata
        attachment = doc_details.get('attachment', {})
        if not attachment or not attachment.get('url'):
            logging.error(f"No attachment URL found for document {doc_id}")
            logging.error(f"Document data: {json.dumps(doc_details, indent=2)}")
            return None
        
        url = attachment['url']
        logging.info(f"\n{'='*80}\nDownload attempt for document {doc_id}")
        logging.info(f"URL: {url}")
        
        # For S3 pre-signed URLs (which is what we get from Shoeboxed),
        # make a clean request without any additional headers
        try:
            logging.info(f"Making clean request to URL")
            response = requests.get(
                url,
                timeout=30,
                stream=True,
                allow_redirects=True  # Allow redirects for pre-signed URLs
            )
            
            logging.info(f"Response status: {response.status_code}")
            logging.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                content = response.content
                if not content:
                    logging.error("Downloaded empty content")
                    return None
                
                content_length = len(content)
                logging.info(f"Successfully downloaded {content_length} bytes")
                return content
            
            # Log error information
            logging.error(f"Download failed. Status: {response.status_code}")
            logging.error(f"Response Headers: {dict(response.headers)}")
            logging.error(f"Response Content: {response.text[:500]}")
            return None
            
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    except Exception as e:
        logging.error(f"Document download error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def process_single_document(doc_details, tokens, state):
    """
    Enhanced document processing with robust error handling and detailed logging
    """
    try:
        doc_id = doc_details.get('id')
        print(f"\n📄 Processing document {doc_id}:")
        print(f"  Type: {doc_details.get('type')}")
        print(f"  Vendor: {doc_details.get('vendor', 'Unknown')}")
        print(f"  Upload Date: {doc_details.get('uploaded')}")
        
        # Skip if document is already processed
        if state.is_document_processed(doc_id):
            print(f"⏭️  Document {doc_id} already processed. Skipping.")
            return False
        
        # Download document
        print(f"📥 Downloading document...")
        document_content = download_document(doc_details, tokens)
        if not document_content:
            print(f"❌ Failed to download document {doc_id}")
            return False
        print(f"✅ Downloaded document: {len(document_content)} bytes")
        
        # Save document locally
        os.makedirs('documents', exist_ok=True)
        local_file_path = os.path.join('documents', f"{doc_id}.pdf")
        with open(local_file_path, 'wb') as f:
            f.write(document_content)
        print(f"💾 Saved document locally: {local_file_path}")
        
        # Extract text using GPT-4V
        print(f"🔍 Extracting text using {os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview')}...")
        extracted_text = extract_text_with_gpt4o(document_content)
        if not extracted_text:
            print(f"❌ Failed to extract text from document {doc_id}")
            return False
        print(f"✅ Successfully extracted text ({len(extracted_text)} characters)")
        
        # Create embedding
        print(f"🧮 Creating embedding...")
        text_embedding = create_embedding(extracted_text)
        if not text_embedding:
            print(f"❌ Failed to create embedding for document {doc_id}")
            return False
        print(f"✅ Created embedding vector")
        
        # Upload to S3
        print(f"☁️  Uploading to S3...")
        s3_url = upload_to_s3(local_file_path, doc_id)
        if not s3_url:
            print(f"❌ Failed to upload document {doc_id} to S3")
            return False
        print(f"✅ Uploaded to S3: {s3_url}")
        
        # Prepare metadata
        metadata = {
            'document_id': doc_id,
            'type': doc_details.get('type', 'unknown'),
            'vendor': doc_details.get('vendor') or 'unknown',
            'total': float(doc_details.get('total', 0) or 0),
            'uploaded_date': doc_details.get('uploaded', '') or '',
            'categories': doc_details.get('categories', []) or [],
            'text': (extracted_text[:4000] if extracted_text else '') or '',
            's3_url': s3_url
        }
        
        # Remove any remaining null values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Log metadata before upsert
        print(f"📝 Prepared metadata:")
        print(json.dumps(metadata, indent=2))
        
        # Upsert to Pinecone
        print(f"📤 Upserting to Pinecone...")
        upsert_to_pinecone(doc_id, text_embedding, metadata)
        print(f"✅ Successfully upserted to Pinecone")
        
        # Mark document as processed
        state.mark_processed(doc_id)
        print(f"✅ Document {doc_id} fully processed")
        
        return True
    
    except Exception as e:
        print(f"❌ Error processing document {doc_id}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def process_documents():
    """Process documents from Shoeboxed in batches"""
    try:
        # Load tokens
        tokens = load_tokens()
        
        # Refresh token if needed
        tokens = refresh_if_needed(tokens)
        
        # Get organization ID
        account_id = get_organization_id(tokens['access_token'])
        
        # Retrieve ALL document IDs (resumable)
        all_document_ids = retrieve_all_document_ids(tokens)
        total_documents = len(all_document_ids)
        
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
        
        # Initialize progress metrics
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        batch_size = 10
        total_batches = (total_documents + batch_size - 1) // batch_size
        
        # Create progress bar placeholder in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        total_metric = metrics_cols[0].empty()
        processed_metric = metrics_cols[1].empty()
        failed_metric = metrics_cols[2].empty()
        skipped_metric = metrics_cols[3].empty()
        
        # Update initial metrics
        total_metric.metric("Total Documents", total_documents)
        processed_metric.metric("Processed", processed_count)
        failed_metric.metric("Failed", failed_count)
        skipped_metric.metric("Skipped", skipped_count)
        
        for i in range(0, total_documents, batch_size):
            # Get batch of document IDs
            batch_ids = all_document_ids[i:i+batch_size]
            batch_documents = []
            current_batch = i//batch_size + 1
            
            # Update progress
            progress = i / total_documents
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {current_batch} of {total_batches} ({progress:.1%} complete)")
            
            # Fetch details for each document in the batch
            print(f"\n🔍 Fetching details for batch {current_batch}/{total_batches}")
            for doc_id in batch_ids:
                try:
                    # Construct the correct endpoint URL
                    doc_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}"
                    
                    # Prepare headers with proper authorization
                    headers = get_shoeboxed_headers(tokens['access_token'])
                    
                    # Make the API request
                    doc_response = requests.get(doc_url, headers=headers)
                    
                    # Log detailed response information
                    print(f"📄 Document {doc_id} Retrieval:")
                    print(f"  Status Code: {doc_response.status_code}")
                    
                    # Check response status
                    if doc_response.status_code == 200:
                        doc_details = doc_response.json()
                        
                        # Additional logging for document details
                        print(f"  Document Type: {doc_details.get('type', 'Unknown')}")
                        print(f"  Processing State: {doc_details.get('processingState', 'Unknown')}")
                        print(f"  Uploaded: {doc_details.get('uploaded', 'Unknown')}")
                        
                        # Only add documents that are fully processed
                        if doc_details.get('processingState') == 'PROCESSED':
                            batch_documents.append(doc_details)
                            print(f"✅ Successfully fetched details for document {doc_id}")
                        else:
                            print(f"⚠️ Skipping document {doc_id} - Processing State: {doc_details.get('processingState')}")
                            skipped_count += 1
                    else:
                        print(f"❌ Failed to fetch details for document {doc_id}")
                        failed_count += 1
                    
                    # Be nice to the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Critical error fetching document {doc_id}: {str(e)}")
                    failed_count += 1
            
            # Process this batch of documents
            print(f"🔢 Processing batch of {len(batch_documents)} documents")
            for doc in batch_documents:
                try:
                    # Process single document
                    success = process_single_document(doc, tokens, state)
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    # Update metrics
                    processed_metric.metric("Processed", processed_count)
                    failed_metric.metric("Failed", failed_count)
                    skipped_metric.metric("Skipped", skipped_count)
                    
                    # Optional: Add a small delay to prevent overwhelming the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error processing individual document: {str(e)}")
                    failed_count += 1
            
            # Update progress message
            status_message = (
                f"Processed: {processed_count} | "
                f"Failed: {failed_count} | "
                f"Skipped: {skipped_count} | "
                f"Progress: {progress:.1%}"
            )
            st.session_state.processing_status = status_message
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Final summary
        print("\n🏁 Document Processing Complete!")
        print(f"📈 Total Documents: {total_documents}")
        print(f"✅ Successfully Processed: {processed_count}")
        print(f"❌ Failed to Process: {failed_count}")
        print(f"⚠️ Skipped: {skipped_count}")
        
        # Update session state
        st.session_state.processing_complete = True
        st.session_state.processing_status = (
            f"Complete! Processed: {processed_count} | "
            f"Failed: {failed_count} | "
            f"Skipped: {skipped_count}"
        )
        
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
        st.sidebar.success("Authenticated")
        
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

def upload_to_s3(local_file_path, document_id):
    try:
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('S3_BUCKET_NAME', 'shoeboxed-documents')
        s3_key = f"processed_documents/{document_id}.pdf"
        
        # Ensure the file exists before uploading
        if not os.path.exists(local_file_path):
            logging.error(f"Local file not found: {local_file_path}")
            return None
        
        s3_client.upload_file(
            local_file_path, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': 'application/pdf'}
        )
        
        # Generate a presigned URL that expires in 7 days
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=7*24*3600  # 7 days
        )
        
        logging.info(f"Successfully uploaded document {document_id} to S3: {s3_key}")
        return presigned_url
    except Exception as e:
        logging.error(f"S3 Upload Error for document {document_id}: {str(e)}")
        return None

def upsert_to_pinecone(document_id, text_embedding, metadata):
    try:
        pinecone_index = init_pinecone()
        
        # Retry mechanism with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pinecone_index.upsert(
                    vectors=[(document_id, text_embedding, metadata)]
                )
                logging.info(f"Successfully upserted document {document_id} to Pinecone")
                break
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.warning(f"Pinecone upsert attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise retry_error
    except Exception as e:
        logging.error(f"Pinecone upsert error for document {document_id}: {str(e)}")
        # Optionally, you could implement additional error handling or logging here

def fetch_document_details(account_id, doc_id, access_token):
    """
    Fetch detailed information for a specific document using Shoeboxed V2 API.
    
    Args:
        account_id (str): Shoeboxed account ID
        doc_id (str): Document ID to fetch
        access_token (str): OAuth access token
    
    Returns:
        dict or None: Detailed document information
    """
    # Construct the precise API endpoint for document details
    endpoint = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Origin': 'https://api.shoeboxed.com',
        'Referer': 'https://api.shoeboxed.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        # First get document details
        response = requests.get(endpoint, headers=headers, timeout=10)
        
        if response.status_code == 200:
            doc_data = response.json()
            doc_type = doc_data.get('type', '').lower()
            
            print(f"\n🔍 Raw document data for {doc_id}:")
            print("=" * 80)
            print(json.dumps(doc_data, indent=2))
            print("=" * 80)
            
            # Now get the download URL based on document type
            download_endpoint = None
            if doc_type == 'receipt':
                download_endpoint = f"https://api.shoeboxed.com/v2/accounts/{account_id}/receipts/{doc_id}/download"
            elif doc_type == 'business-card':
                download_endpoint = f"https://api.shoeboxed.com/v2/accounts/{account_id}/business-cards/{doc_id}/download"
            else:
                download_endpoint = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}/download"
            
            print(f"Checking download URL: {download_endpoint}")
            
            # Make a HEAD request to verify the download URL
            download_headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/pdf,image/*',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            try:
                download_response = requests.head(
                    download_endpoint,
                    headers=download_headers,
                    allow_redirects=True,
                    timeout=10
                )
                
                print(f"Download response status: {download_response.status_code}")
                print(f"Download response headers: {dict(download_response.headers)}")
                
                if download_response.status_code in [200, 302]:
                    # For S3 URLs, get the Location header from the redirect
                    if download_response.status_code == 302:
                        actual_download_url = download_response.headers.get('Location')
                        print(f"🔄 Redirected to: {actual_download_url}")
                        doc_data['attachment'] = {
                            'url': actual_download_url,
                            'name': f"{doc_type}_{doc_id}.pdf"
                        }
                    else:
                        doc_data['attachment'] = {
                            'url': download_endpoint,
                            'name': f"{doc_type}_{doc_id}.pdf"
                        }
                    print(f"✅ Successfully added download URL to document {doc_id}")
                else:
                    print(f"❌ Failed to verify download URL for document {doc_id}")
                    print(f"Response status: {download_response.status_code}")
                    print(f"Response headers: {download_response.headers}")
            except Exception as e:
                print(f"❌ Error checking download URL: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Add account ID to document data
            doc_data['accountId'] = account_id
            
            # Validate processing state
            processing_state = doc_data.get('processingState')
            if processing_state not in ['PROCESSED', 'NEEDS_USER_PROCESSING']:
                print(f"⚠️ Document {doc_id} in non-standard processing state: {processing_state}")
            
            # Enrich document metadata
            doc_data['metadata'] = {
                'uploaded': doc_data.get('uploaded'),
                'modified': doc_data.get('modified'),
                'vendor': doc_data.get('vendor'),
                'total': doc_data.get('total'),
                'type': doc_type,
                'processing_state': processing_state,
                'accountId': account_id
            }
            
            return doc_data
        
        # Log unsuccessful responses
        print(f"❌ Failed to fetch document {doc_id}. Status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        return None
    
    except Exception as e:
        print(f"❌ Error fetching document {doc_id}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()
