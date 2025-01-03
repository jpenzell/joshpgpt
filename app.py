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
import shutil

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
            tokens = json.load(f)
            if not tokens.get('access_token'):
                raise Exception("Invalid token data")
            return tokens
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error loading tokens: {str(e)}")
        return None

def refresh_if_needed(tokens):
    """Check if token needs refresh and refresh if necessary"""
    if not tokens:
        return None
        
    try:
        expires_at = datetime.fromisoformat(tokens.get('expires_at', datetime.now().isoformat()))
        if datetime.now() >= expires_at:
            # Implement token refresh
            refresh_token = tokens.get('refresh_token')
            if not refresh_token:
                return None
                
            response = requests.post(
                os.getenv('SHOEBOXED_TOKEN_URL'),
                data={
                    'grant_type': 'refresh_token',
                    'refresh_token': refresh_token,
                    'client_id': os.getenv('SHOEBOXED_CLIENT_ID'),
                    'client_secret': os.getenv('SHOEBOXED_CLIENT_SECRET')
                }
            )
            
            if response.status_code == 200:
                new_tokens = response.json()
                # Update expiration
                new_tokens['expires_at'] = (
                    datetime.now() + 
                    timedelta(seconds=new_tokens.get('expires_in', 3600))
                ).isoformat()
                
                # Save new tokens
                with open('.auth_success', 'w') as f:
                    json.dump(new_tokens, f)
                    
                return new_tokens
            else:
                print(f"Token refresh failed: {response.status_code}")
                return None
                
        return tokens
    except Exception as e:
        print(f"Error refreshing token: {str(e)}")
        return None

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
                    model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o'),
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
                    top_p=0.1,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
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

def retrieve_all_document_ids(tokens, checkpoint):
    """
    Retrieve document IDs with robust resumption and date-based filtering
    
    Args:
        tokens (dict): Authentication tokens
        checkpoint (ProcessingCheckpoint): Checkpoint system for tracking progress
    """
    # Create a Streamlit placeholder for the running count
    running_count = st.empty()
    total_to_process = 0
    
    print("\n📋 Document Processing Status:")
    print("--------------------------------")
    print(f"✅ Fully Processed (in Pinecone): {len(checkpoint.processed_docs)}")
    print(f"❌ Failed Documents: {len(checkpoint.failed_docs)}")
    print(f"⏭️ Skipped Documents: {len(checkpoint.skipped_docs)}")
    
    # Get account ID and prepare API endpoint
    account_id = get_organization_id(tokens['access_token'])
    list_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents"
    headers = get_shoeboxed_headers(tokens['access_token'])
    
    # Batch processing parameters
    batch_size = 100  # Maximum allowed by API
    all_document_ids = []
    current_offset = 0
    
    try:
        # First get total count
        params = {
            'offset': 0,
            'limit': 1,
            'order_by_desc': 'uploaded',
            'trashed': 'false'
        }
        
        response = requests.get(list_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Failed to fetch document count. Status: {response.status_code}")
            return []
            
        total_count = response.json().get('totalCount', 0)
        print(f"\n📊 Total documents in Shoeboxed: {total_count}")
        
        while True:
            # Prepare query parameters
            params = {
                'offset': current_offset,
                'limit': batch_size,
                'order_by_desc': 'uploaded',
                'trashed': 'false'
            }
            
            print(f"\n📡 Fetching document batch starting at offset {current_offset}")
            
            response = requests.get(list_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch document list. Status: {response.status_code}")
                break
            
            data = response.json()
            doc_list = data.get('documents', [])
            
            if not doc_list:
                break
            
            # Process and filter documents
            batch_stats = {
                'total': 0,
                'to_process': 0
            }
            
            for doc in doc_list:
                doc_id = doc.get('id')
                if not doc_id:
                    continue
                
                batch_stats['total'] += 1
                
                # Only skip if successfully processed
                if doc_id in checkpoint.processed_docs:
                    continue
                
                batch_stats['to_process'] += 1
                all_document_ids.append(doc_id)
            
            # Update total count and display
            total_to_process = len(all_document_ids)
            running_count.metric("Documents To Process", total_to_process)
            
            # Log batch statistics
            print(f"\n📊 Batch Statistics:")
            print(f"   Total Documents: {batch_stats['total']}")
            print(f"   To Process: {batch_stats['to_process']}")
            print(f"   Running Total To Process: {total_to_process}")
            
            # Update offset and check if we've retrieved all documents
            current_offset += batch_size
            if current_offset >= total_count:
                break
            
            time.sleep(0.5)  # Be nice to the API
    
    except Exception as e:
        print(f"❌ Error retrieving document IDs: {str(e)}")
        print(traceback.format_exc())
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total Documents Found: {total_count}")
    print(f"   Already in Pinecone: {len(checkpoint.processed_docs)}")
    print(f"   Failed Previously: {len(checkpoint.failed_docs)}")
    print(f"   Skipped: {len(checkpoint.skipped_docs)}")
    print(f"   To Be Processed: {total_to_process}")
    
    return all_document_ids

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
    Process a single document regardless of its Shoeboxed processing state
    """
    try:
        doc_id = doc_details.get('id')
        print(f"\n{'='*80}", flush=True)
        print(f"📄 Processing document {doc_id}:", flush=True)
        print(f"  Type: {doc_details.get('type')}", flush=True)
        print(f"  Vendor: {doc_details.get('vendor', 'Unknown')}", flush=True)
        print(f"  Upload Date: {doc_details.get('uploaded')}", flush=True)
        print(f"  Total Amount: ${doc_details.get('total', 'N/A')}", flush=True)
        
        # Fix categories handling
        categories = doc_details.get('categories')
        if isinstance(categories, list):
            categories_str = ', '.join(categories) or 'None'
        else:
            categories_str = str(categories) if categories else 'None'
        print(f"  Categories: {categories_str}", flush=True)
        
        print(f"  Processing State: {doc_details.get('processingState', 'Unknown')}", flush=True)
        if doc_id in state.failed_docs:
            print(f"  Previous Failure: {state.failure_reasons.get(doc_id, 'Unknown reason')}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Skip if document is already processed in our system
        if not state.should_process_doc(doc_id):
            print(f"⏭️  Document {doc_id} already processed in our system. Skipping.", flush=True)
            return False
        
        # Download document
        print(f"\n📥 Downloading document...", flush=True)
        document_content = download_document(doc_details, tokens)
        if not document_content:
            state.mark_failed(doc_id, "Failed to download document")
            print(f"❌ Failed to download document {doc_id}", flush=True)
            print(f"   Document details: {json.dumps(doc_details, indent=2)}", flush=True)
            return False
        print(f"✅ Downloaded document: {len(document_content):,} bytes", flush=True)
        
        # Save document locally
        os.makedirs('documents', exist_ok=True)
        local_file_path = os.path.join('documents', f"{doc_id}.pdf")
        with open(local_file_path, 'wb') as f:
            f.write(document_content)
        print(f"💾 Saved document locally: {local_file_path}")
        print(f"   File size: {os.path.getsize(local_file_path):,} bytes")
        
        # Extract text using GPT-4V
        print(f"\n🔍 Extracting text using {os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview')}...")
        start_time = time.time()
        extracted_text = extract_text_with_gpt4o(document_content)
        extraction_time = time.time() - start_time
        if not extracted_text:
            state.mark_failed(doc_id, "Failed to extract text")
            print(f"❌ Failed to extract text from document {doc_id}")
            return False
        print(f"✅ Successfully extracted text:")
        print(f"   Characters: {len(extracted_text):,}")
        print(f"   Words: {len(extracted_text.split()):,}")
        print(f"   Time taken: {extraction_time:.2f} seconds")
        
        # Create embedding
        print(f"\n🧮 Creating embedding...")
        start_time = time.time()
        text_embedding = create_embedding(extracted_text)
        embedding_time = time.time() - start_time
        if not text_embedding:
            state.mark_failed(doc_id, "Failed to create embedding")
            print(f"❌ Failed to create embedding for document {doc_id}")
            return False
        print(f"✅ Created embedding vector")
        print(f"   Vector dimensions: {len(text_embedding)}")
        print(f"   Time taken: {embedding_time:.2f} seconds")
        
        # Upload to S3
        print(f"\n☁️  Uploading to S3...")
        start_time = time.time()
        s3_url = upload_to_s3(local_file_path, doc_id)
        upload_time = time.time() - start_time
        if not s3_url:
            state.mark_failed(doc_id, "Failed to upload to S3")
            print(f"❌ Failed to upload document {doc_id} to S3")
            return False
        print(f"✅ Uploaded to S3:")
        print(f"   URL: {s3_url}")
        print(f"   Time taken: {upload_time:.2f} seconds")
        
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
        print(f"\n📝 Document Metadata:")
        print(json.dumps(metadata, indent=2))
        
        # Upsert to Pinecone
        print(f"\n📤 Upserting to Pinecone...")
        start_time = time.time()
        try:
            upsert_to_pinecone(doc_id, text_embedding, metadata)
            upsert_time = time.time() - start_time
            print(f"✅ Successfully upserted to Pinecone")
            print(f"   Time taken: {upsert_time:.2f} seconds")
        except Exception as e:
            state.mark_failed(doc_id, f"Failed to upsert to Pinecone: {str(e)}")
            print(f"❌ Failed to upsert to Pinecone: {str(e)}")
            return False
        
        # Mark document as processed
        state.mark_processed(doc_id)
        print(f"\n✅ Document {doc_id} fully processed")
        print(f"Total processing time: {extraction_time + embedding_time + upload_time + upsert_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        return True
    
    except Exception as e:
        error_msg = f"Error type: {type(e).__name__}, Message: {str(e)}"
        state.mark_failed(doc_id, error_msg)
        print(f"\n❌ Error processing document {doc_id}:")
        print(f"   {error_msg}")
        print(f"   Traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        return False

class ProcessingCheckpoint:
    def __init__(self):
        self.checkpoint_file = 'processing_checkpoint.json'
        self.current_batch = 0
        self.processed_docs = set()
        self.failed_docs = set()
        self.skipped_docs = set()
        self.failure_reasons = {}  # Track why each document failed
        self.last_processed_time = None
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load the last checkpoint if it exists"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.current_batch = data.get('current_batch', 0)
                    self.processed_docs = set(data.get('processed_docs', []))
                    self.failed_docs = set(data.get('failed_docs', []))
                    self.skipped_docs = set(data.get('skipped_docs', []))
                    self.failure_reasons = data.get('failure_reasons', {})
                    self.last_processed_time = data.get('last_processed_time')
                print(f"📥 Loaded checkpoint: Batch {self.current_batch}")
                print(f"✅ Processed: {len(self.processed_docs)} documents")
                print(f"❌ Failed: {len(self.failed_docs)} documents")
                print(f"⏭️ Skipped: {len(self.skipped_docs)} documents")
                if self.failed_docs:
                    print("\n❌ Failed Documents and Reasons:")
                    for doc_id in self.failed_docs:
                        reason = self.failure_reasons.get(doc_id, "Unknown reason")
                        print(f"   {doc_id}: {reason}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {str(e)}")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            data = {
                'current_batch': self.current_batch,
                'processed_docs': list(self.processed_docs),
                'failed_docs': list(self.failed_docs),
                'skipped_docs': list(self.skipped_docs),
                'failure_reasons': self.failure_reasons,
                'last_processed_time': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved checkpoint at batch {self.current_batch}")
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {str(e)}")
    
    def mark_processed(self, doc_id):
        """Mark a document as processed"""
        self.processed_docs.add(doc_id)
        if doc_id in self.failed_docs:
            self.failed_docs.remove(doc_id)
            self.failure_reasons.pop(doc_id, None)
        self.save_checkpoint()
    
    def mark_failed(self, doc_id, reason="Unknown error"):
        """Mark a document as failed with a reason"""
        self.failed_docs.add(doc_id)
        self.failure_reasons[doc_id] = reason
        self.save_checkpoint()
    
    def mark_skipped(self, doc_id):
        """Mark a document as skipped"""
        self.skipped_docs.add(doc_id)
        self.save_checkpoint()
    
    def update_batch(self, batch_num):
        """Update current batch number"""
        self.current_batch = batch_num
        self.save_checkpoint()
    
    def should_process_doc(self, doc_id):
        """Check if document should be processed"""
        # Only skip if successfully processed or explicitly skipped
        return doc_id not in self.processed_docs and \
               doc_id not in self.skipped_docs

def process_documents():
    """Process documents from Shoeboxed in batches"""
    try:
        print("\n🔍 DEBUG: Starting document processing...", flush=True)
        
        # Initialize checkpoint system first
        checkpoint = ProcessingCheckpoint()
        print("\n📋 Loading checkpoint...", flush=True)
        
        # Record start time
        start_time = time.time()
        
        # Load tokens and ensure we have access token
        print("\n🔑 DEBUG: Loading tokens...", flush=True)
        tokens = load_tokens()
        if not tokens or 'access_token' not in tokens:
            st.error("No valid Shoeboxed access token found. Please authenticate first.")
            return
            
        # Store access token in environment for other functions to use
        os.environ['SHOEBOXED_ACCESS_TOKEN'] = tokens['access_token']
        
        # Refresh token if needed
        tokens = refresh_if_needed(tokens)
        if not tokens:
            st.error("Failed to refresh token. Please authenticate again.")
            return
            
        # Update access token in environment after refresh
        os.environ['SHOEBOXED_ACCESS_TOKEN'] = tokens['access_token']
        
        # Get organization ID
        print("\n🏢 DEBUG: Getting organization ID...", flush=True)
        account_id = get_organization_id(tokens['access_token'])
        print(f"   Organization ID: {account_id}", flush=True)
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        total_metric = metrics_cols[0].empty()
        processed_metric = metrics_cols[1].empty()
        failed_metric = metrics_cols[2].empty()
        skipped_metric = metrics_cols[3].empty()
        
        # Initialize counters
        processed_count = len(checkpoint.processed_docs)
        failed_count = len(checkpoint.failed_docs)
        skipped_count = len(checkpoint.skipped_docs)
        
        # Update initial metrics
        total_metric.metric("Total Documents", "Loading...")
        processed_metric.metric("Processed", processed_count)
        failed_metric.metric("Failed", failed_count)
        skipped_metric.metric("Skipped", skipped_count)
        
        # Retrieve document IDs
        print("\n🔍 DEBUG: Retrieving document IDs...", flush=True)
        all_document_ids = retrieve_all_document_ids(tokens, checkpoint)
        total_documents = len(all_document_ids)
        
        # Update total metric now that we have the count
        total_metric.metric("Total Documents", total_documents)
        
        if total_documents == 0:
            print("✅ No new documents to process!")
            st.success("All documents have been processed!")
            return
        
        print(f"\n📊 Processing Summary:")
        print(f"{'='*80}")
        print(f"Total Documents to Process: {total_documents:,}")
        print(f"Already Processed: {processed_count:,}")
        print(f"Previously Failed: {failed_count:,}")
        print(f"Previously Skipped: {skipped_count:,}")
        print(f"{'='*80}\n")
        
        # Process documents in batches
        batch_size = 10
        total_batches = (total_documents + batch_size - 1) // batch_size
        
        for batch_index in range(total_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, total_documents)
            current_batch = all_document_ids[start_idx:end_idx]
            
            print(f"\n🔄 Processing batch {batch_index + 1} of {total_batches}")
            print(f"   Documents {start_idx + 1} to {end_idx} of {total_documents}")
            
            # Update progress
            progress = (batch_index + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {batch_index + 1} of {total_batches}")
            
            # Process each document in the batch
            for doc_id in current_batch:
                try:
                    print(f"\n📄 Processing document {doc_id}")
                    
                    # Skip if already processed
                    if not checkpoint.should_process_doc(doc_id):
                        print(f"⏭️ Skipping already processed document {doc_id}")
                        skipped_count += 1
                        skipped_metric.metric("Skipped", skipped_count)
                        continue
                    
                    # Get document details
                    doc_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}"
                    response = requests.get(doc_url, headers=get_shoeboxed_headers(tokens['access_token']))
                    
                    if response.status_code != 200:
                        print(f"❌ Failed to get document details. Status: {response.status_code}")
                        checkpoint.mark_failed(doc_id, f"HTTP {response.status_code}")
                        failed_count += 1
                        failed_metric.metric("Failed", failed_count)
                        continue
                    
                    doc_data = response.json()
                    
                    # Process the document
                    if process_single_document(doc_data, tokens['access_token'], checkpoint):
                        processed_count += 1
                        processed_metric.metric("Processed", processed_count)
                        print(f"✅ Successfully processed document {doc_id}")
                    else:
                        failed_count += 1
                        failed_metric.metric("Failed", failed_count)
                        print(f"❌ Failed to process document {doc_id}")
                    
                    # Update metrics
                    total_metric.metric("Total Documents", total_documents)
                    processed_metric.metric("Processed", processed_count)
                    failed_metric.metric("Failed", failed_count)
                    skipped_metric.metric("Skipped", skipped_count)
                    
                except Exception as e:
                    print(f"❌ Error processing document {doc_id}: {str(e)}")
                    checkpoint.mark_failed(doc_id, str(e))
                    failed_count += 1
                    failed_metric.metric("Failed", failed_count)
                    continue
            
            # Save checkpoint after each batch
            checkpoint.update_batch(batch_index)
            checkpoint.save_checkpoint()
            
            # Refresh token if needed after each batch
            tokens = refresh_if_needed(tokens)
            if not tokens:
                st.error("Failed to refresh token. Please authenticate again.")
                return
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Display final metrics
        total_metric.metric("Total Documents", total_documents)
        processed_metric.metric("Processed", processed_count)
        failed_metric.metric("Failed", failed_count)
        skipped_metric.metric("Skipped", skipped_count)
        
        print("\n✅ Document processing complete!")
        print(f"📊 Final Statistics:")
        print(f"   Total Documents: {total_documents}")
        print(f"   Processed: {processed_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Time taken: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error in document processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"An error occurred during processing: {str(e)}")
        raise

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

def reset_processing_state():
    """Reset all processing state and clear databases"""
    try:
        print("\n🧹 Starting cleanup process...")
        
        # Clear Pinecone index
        print("🔄 Clearing Pinecone index...")
        index = init_pinecone()
        if index:
            try:
                index.delete(delete_all=True)
                print("✅ Pinecone index cleared")
            except Exception as e:
                if '404' in str(e):
                    print("ℹ️ Pinecone index is already empty")
                else:
                    print(f"⚠️ Error clearing Pinecone index: {str(e)}")
        
        # Delete local files
        files_to_delete = [
            'processing_checkpoint.json',
            'document_id_cache.json',
            'retrieval_metadata.json'  # Add new metadata file
        ]
        
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Deleted {file}")
            else:
                print(f"ℹ️ File not found: {file}")
        
        # Clear document directories
        found_dirs = False
        for dir_name in os.listdir('.'):
            if dir_name.startswith('documents_'):
                found_dirs = True
                try:
                    shutil.rmtree(dir_name)
                    print(f"✅ Deleted directory {dir_name}")
                except Exception as e:
                    print(f"⚠️ Error deleting directory {dir_name}: {str(e)}")
        
        if not found_dirs:
            print("ℹ️ No document directories found to clean")
        
        # Clear S3 bucket
        try:
            s3_client = boto3.client('s3')
            bucket_name = os.environ.get('S3_BUCKET_NAME', 'shoeboxed-documents')
            
            print(f"🔄 Clearing S3 bucket {bucket_name}/processed_documents/...")
            
            # List and delete all objects in the processed_documents prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            objects_to_delete = []
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix='processed_documents/'):
                if 'Contents' in page:
                    objects_to_delete.extend(
                        {'Key': obj['Key']} for obj in page['Contents']
                    )
            
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': objects_to_delete}
                )
                print(f"✅ Deleted {len(objects_to_delete)} objects from S3")
            else:
                print("ℹ️ No objects found in S3 to delete")
                
        except Exception as e:
            print(f"⚠️ Error clearing S3 bucket: {str(e)}")
        
        print("\n✅ Reset complete! The system is ready for fresh document processing.")
        return True
        
    except Exception as e:
        print(f"❌ Error during reset: {str(e)}")
        return False

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
        
        # Add reset control at the top
        if st.sidebar.button("🔄 Reset Processing"):
            with st.spinner("Resetting all processing state..."):
                if reset_processing_state():
                    st.success("Reset complete! Ready for fresh document processing.")
                else:
                    st.error("Error during reset. Please check the logs.")
        
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

def create_document_metadata(doc_data, extracted_text):
    """Create metadata for document storage"""
    return {
        'document_id': doc_data.get('id'),
        'type': doc_data.get('type', 'unknown'),
        'vendor': doc_data.get('vendor') or 'unknown',
        'total': float(doc_data.get('total', 0) or 0),
        'uploaded_date': doc_data.get('uploaded', '') or '',
        'categories': doc_data.get('categories', []) or [],
        'text': (extracted_text[:4000] if extracted_text else '') or '',
        'processing_state': doc_data.get('processingState', 'unknown'),
        'modified_date': doc_data.get('modified', '') or ''
    }

def store_in_pinecone(index, doc_id, extracted_text, metadata):
    """Store document in Pinecone"""
    try:
        # Create embedding for the text
        text_embedding = create_embedding(extracted_text)
        if not text_embedding:
            print(f"❌ Failed to create embedding for document {doc_id}")
            return False
            
        # Upsert to Pinecone
        upsert_to_pinecone(doc_id, text_embedding, metadata)
        return True
    except Exception as e:
        print(f"❌ Error storing document {doc_id} in Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    main()
