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
from fetch_documents import fetch_all_documents  # Updated import
from process_for_pinecone import create_embedding  # Added import for create_embedding
import logging
import boto3
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import queue
import concurrent.futures
import re
from processing_metrics import ProcessingMetrics

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

# Processing constants
MAX_THREADS = 4  # Maximum number of concurrent document processing threads
RATE_LIMIT_DELAY = 1  # Delay between documents in seconds to respect rate limits

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
        # Add buffer time to refresh before expiration
        REFRESH_BUFFER_MINUTES = 5
        expires_at = datetime.fromisoformat(tokens.get('expires_at', datetime.now().isoformat()))
        
        # Check if token will expire soon (within buffer time)
        if datetime.now() + timedelta(minutes=REFRESH_BUFFER_MINUTES) >= expires_at:
            print("\n🔄 Token needs refresh...")
            
            # Implement token refresh with retries
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    refresh_token = tokens.get('refresh_token')
                    if not refresh_token:
                        print("❌ No refresh token available")
                        return None
                    
                    print(f"Refresh attempt {attempt + 1}/{max_retries}")
                    response = requests.post(
                        os.getenv('SHOEBOXED_TOKEN_URL'),
                        data={
                            'grant_type': 'refresh_token',
                            'refresh_token': refresh_token,
                            'client_id': os.getenv('SHOEBOXED_CLIENT_ID'),
                            'client_secret': os.getenv('SHOEBOXED_CLIENT_SECRET'),
                            'scope': 'all'
                        },
                        headers={
                            'Content-Type': 'application/x-www-form-urlencoded',
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            'Accept': 'application/json'
                        }
                    )
                    
                    if response.status_code == 200:
                        new_tokens = response.json()
                        # Preserve the refresh token if not included in response
                        if 'refresh_token' not in new_tokens and refresh_token:
                            new_tokens['refresh_token'] = refresh_token
                            
                        # Update expiration with buffer
                        new_tokens['expires_at'] = (
                            datetime.now() + 
                            timedelta(seconds=new_tokens.get('expires_in', 3600))
                        ).isoformat()
                        
                        # Save new tokens
                        with open('.auth_success', 'w') as f:
                            json.dump(new_tokens, f)
                        
                        print("✅ Token refresh successful")
                        return new_tokens
                    else:
                        print(f"⚠️ Token refresh failed (Attempt {attempt + 1}): Status {response.status_code}")
                        print(f"Response: {response.text}")
                        
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print("❌ All refresh attempts failed")
                            return None
                            
                except Exception as e:
                    print(f"⚠️ Error during refresh attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print("❌ All refresh attempts failed")
                        return None
                        
        return tokens
        
    except Exception as e:
        print(f"❌ Error in refresh check: {str(e)}")
        return None

def init_session_state():
    """Initialize session state variables"""
    # Initialize all session state variables first
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    
    # Only verify token if not already authenticated and not in a paused state
    if not st.session_state.authenticated and not st.session_state.paused:
        if os.path.exists('.auth_success'):
            try:
                with open('.auth_success', 'r') as f:
                    auth_data = json.load(f)
                    if auth_data.get('access_token'):
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

def init_pinecone():
    """Initialize Pinecone client"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(
            name=PINECONE_INDEX_NAME,
            host=PINECONE_INDEX_HOST
        )
        # Test connection
        index.describe_index_stats()
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

def validate_and_repair_pdf(pdf_bytes):
    """Validate and attempt to repair PDF if needed"""
    try:
        import pikepdf
        import tempfile
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Try to open and save the PDF (this will attempt basic repairs)
            with pikepdf.open(temp_pdf_path, allow_overwriting_input=True) as pdf:
                # Save to a new temporary file
                repaired_path = temp_pdf_path + '_repaired.pdf'
                pdf.save(repaired_path)
                
                # Read the repaired PDF
                with open(repaired_path, 'rb') as f:
                    repaired_bytes = f.read()
                
                # Clean up temporary files
                os.unlink(repaired_path)
                os.unlink(temp_pdf_path)
                
                return repaired_bytes
        except Exception as e:
            logging.error(f"PDF repair failed: {str(e)}")
            os.unlink(temp_pdf_path)
            return None
            
    except Exception as e:
        logging.error(f"PDF validation error: {str(e)}")
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
        
        # Try multiple PDF to image conversion methods
        images = None
        conversion_methods = [
            ('pdf2image', lambda: convert_from_bytes(pdf_stream.read())),
            ('PyMuPDF', lambda: convert_with_pymupdf(pdf_stream)),
            ('Poppler', lambda: convert_with_poppler(pdf_stream))
        ]
        
        for method_name, converter in conversion_methods:
            try:
                logging.info(f"Attempting PDF conversion using {method_name}...")
                pdf_stream.seek(0)  # Reset stream position
                images = converter()
                if images:
                    logging.info(f"Successfully converted PDF using {method_name}")
                    break
            except Exception as e:
                logging.error(f"{method_name} conversion failed: {str(e)}")
                continue
        
        if not images:
            logging.error("All PDF conversion methods failed")
            return None
        
        # Process pages concurrently with improved error handling
        image_texts = []
        with ThreadPoolExecutor() as executor:
            future_to_page = {
                executor.submit(process_page_with_retry, i, image): i 
                for i, image in enumerate(images)
            }
            
            for future in as_completed(future_to_page):
                try:
                    page_num = future_to_page[future]
                    page_text = future.result()
                    if page_text:
                        image_texts.append(f"Page {page_num + 1}:\n{page_text}")
                except Exception as e:
                    logging.error(f"Error processing page: {str(e)}")
        
        # Combine texts from all pages
        return "\n\n".join(image_texts) if image_texts else None
    
    except Exception as e:
        logging.error(f"Error in GPT-4o document text extraction: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def convert_with_pymupdf(pdf_stream):
    """Convert PDF to images using PyMuPDF"""
    import fitz
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.tobytes())
        images.append(img)
    doc.close()
    return images if images else None

def convert_with_poppler(pdf_stream):
    """Convert PDF to images using Poppler directly"""
    import subprocess
    import tempfile
    
    # Save PDF to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_stream.read())
        pdf_path = temp_pdf.name
    
    # Create temporary directory for output images
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Use pdftoppm (from Poppler) directly
            subprocess.run([
                'pdftoppm',
                '-png',
                pdf_path,
                f"{temp_dir}/page"
            ], check=True)
            
            # Load the generated images
            images = []
            for img_file in sorted(os.listdir(temp_dir)):
                if img_file.endswith('.png'):
                    img_path = os.path.join(temp_dir, img_file)
                    images.append(Image.open(img_path))
            
            return images
        finally:
            os.unlink(pdf_path)
    
    return None

def process_page_with_retry(i, image, max_retries=3):
    """Process a single page with retry logic"""
    for attempt in range(max_retries):
        try:
            return process_page(i, image)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to process page {i} after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def process_page(i, image):
    """Process a single page image and extract text"""
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
        return f"Page {i+1}:\n{page_text}" if page_text else None
    
    except Exception as e:
        logging.error(f"Error processing page {i}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def upsert_to_pinecone_batch(documents):
    """Batch upsert documents to Pinecone"""
    try:
        pinecone_index = init_pinecone()
        
        # Prepare batch data
        vectors = [(doc['id'], doc['embedding'], doc['metadata']) for doc in documents]
        
        # Retry mechanism with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pinecone_index.upsert(vectors=vectors)
                logging.info(f"Successfully upserted batch to Pinecone")
                break
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.warning(f"Pinecone upsert attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise retry_error
    except Exception as e:
        logging.error(f"Pinecone batch upsert error: {str(e)}")

def retry_request(func, *args, max_retries=3, retry_delay=5, **kwargs):
    """Retry a function call with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error during attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logging.error("All retry attempts failed")
                raise

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
        # First get total count with minimal filtering
        params = {
            'offset': 0,
            'limit': 1,
            'include': 'attachment,stats,categories,type',
            'trashed': 'false'
        }
        
        print("\n🔍 Checking total document count...")
        print(f"Request URL: {list_url}")
        print(f"Request params: {params}")
        print(f"Request headers: {headers}")
        
        response = requests.get(list_url, headers=headers, params=params)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text}")
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch document count. Status: {response.status_code}")
            return []
            
        total_count = response.json().get('totalCount', 0)
        print(f"\n📊 Total documents in Shoeboxed: {total_count}")
        
        while True:
            # Prepare query parameters with minimal filtering
            params = {
                'offset': current_offset,
                'limit': batch_size,
                'include': 'attachment,stats,categories,type',
                'trashed': 'false'
            }
            
            print(f"\n📡 Fetching document batch starting at offset {current_offset}")
            print(f"Request params: {params}")
            
            response = requests.get(list_url, headers=headers, params=params)
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch document list. Status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                print(f"Request URL: {response.url}")
                
                # Refresh token and retry once
                tokens = refresh_if_needed(tokens)
                if tokens:
                    headers = get_shoeboxed_headers(tokens['access_token'])
                    response = requests.get(list_url, headers=headers, params=params)
                    if response.status_code == 200:
                        print("✅ Request succeeded after token refresh")
                    else:
                        print(f"❌ Request still failed after token refresh: {response.status_code}")
                        break
                else:
                    print("❌ Token refresh failed")
                    break
            
            data = response.json()
            doc_list = data.get('documents', [])
            
            if not doc_list:
                print("No more documents found in this batch")
                break
            
            print(f"\n📄 Found {len(doc_list)} documents in current batch")
            
            # Process and filter documents
            batch_stats = {
                'total': 0,
                'to_process': 0,
                'already_processed': 0,
                'types': {}
            }
            
            for doc in doc_list:
                doc_id = doc.get('id')
                doc_type = doc.get('type', 'unknown')
                
                if not doc_id:
                    print(f"⚠️ Document without ID found: {doc}")
                    continue
                
                batch_stats['total'] += 1
                batch_stats['types'][doc_type] = batch_stats['types'].get(doc_type, 0) + 1
                
                if doc_id in checkpoint.processed_docs:
                    batch_stats['already_processed'] += 1
                    continue
                
                batch_stats['to_process'] += 1
                all_document_ids.append(doc_id)
            
            # Update total count and display
            total_to_process = len(all_document_ids)
            running_count.metric("Documents To Process", total_to_process)
            
            # Log batch statistics
            print(f"\n📊 Batch Statistics:")
            print(f"   Total Documents: {batch_stats['total']}")
            print(f"   Already Processed: {batch_stats['already_processed']}")
            print(f"   To Process: {batch_stats['to_process']}")
            print(f"   Document Types Found:")
            for doc_type, count in batch_stats['types'].items():
                print(f"      - {doc_type}: {count}")
            print(f"   Running Total To Process: {total_to_process}")
            
            # Update offset and check if we've retrieved all documents
            current_offset += len(doc_list)  # Use actual number of documents received
            if current_offset >= total_count:
                print(f"Reached total count ({total_count}), stopping retrieval")
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

def sanitize_pdf(pdf_bytes):
    """Attempt to sanitize and repair a PDF file"""
    try:
        import pikepdf
        import tempfile
        import os
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Try to open and save the PDF (this will attempt repairs)
            with pikepdf.open(temp_pdf_path, allow_overwriting_input=True) as pdf:
                # Save to a new temporary file
                repaired_path = temp_pdf_path + '_repaired.pdf'
                pdf.save(repaired_path)
                
                # Read the repaired PDF
                with open(repaired_path, 'rb') as f:
                    repaired_bytes = f.read()
                
                # Clean up temporary files
                os.unlink(repaired_path)
                os.unlink(temp_pdf_path)
                
                return repaired_bytes
        except Exception as e:
            logging.error(f"PDF repair failed: {str(e)}")
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
            return None
            
    except Exception as e:
        logging.error(f"PDF sanitization error: {str(e)}")
        return None

def process_single_document(doc_details, tokens, state):
    """Process a single document regardless of its Shoeboxed processing state"""
    try:
        doc_id = doc_details.get('id')
        retry_count = state.get_retry_count(doc_id)
        
        # Implement exponential backoff if this is a retry
        if retry_count > 0:
            backoff_time = min(300, 5 * (2 ** (retry_count - 1)))  # Max 5 minutes
            print(f"⏳ Retry backoff: waiting {backoff_time} seconds before attempt {retry_count + 1}", flush=True)
            time.sleep(backoff_time)
        
        print(f"\n{'='*80}", flush=True)
        print(f"📄 Processing document {doc_id}:", flush=True)
        print(f"  Type: {doc_details.get('type')}", flush=True)
        print(f"  Vendor: {doc_details.get('vendor', 'Unknown')}", flush=True)
        print(f"  Upload Date: {doc_details.get('uploaded')}", flush=True)
        print(f"  Total Amount: ${doc_details.get('total', 'N/A')}", flush=True)
        
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
            return False
        print(f"✅ Downloaded document: {len(document_content):,} bytes", flush=True)
        
        # Try to sanitize/repair PDF if needed
        print(f"🔧 Attempting to sanitize/repair PDF...", flush=True)
        sanitized_content = sanitize_pdf(document_content)
        if sanitized_content:
            print(f"✅ Successfully sanitized PDF", flush=True)
            document_content = sanitized_content
        else:
            print(f"⚠️ Could not sanitize PDF, proceeding with original", flush=True)
        
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

def split_large_document(document_content, chunk_size=500_000):
    """Split a large document into smaller chunks"""
    chunks = []
    total_size = len(document_content)
    for i in range(0, total_size, chunk_size):
        chunk = document_content[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

class ProcessingCheckpoint:
    def __init__(self):
        self.checkpoint_file = 'processing_checkpoint.json'
        self.current_batch = 0
        self.processed_docs = set()
        self.failed_docs = set()
        self.skipped_docs = set()
        self.failure_reasons = {}  # Track why each document failed
        self.retry_counts = {}     # Track retry attempts per document
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
                    self.retry_counts = data.get('retry_counts', {})
                    self.last_processed_time = data.get('last_processed_time')
                print(f"📥 Loaded checkpoint: Batch {self.current_batch}")
                print(f"✅ Processed: {len(self.processed_docs)} documents")
                print(f"❌ Failed: {len(self.failed_docs)} documents")
                print(f"⏭️ Skipped: {len(self.skipped_docs)} documents")
                if self.failed_docs:
                    print("\n❌ Failed Documents and Reasons:")
                    for doc_id in self.failed_docs:
                        reason = self.failure_reasons.get(doc_id, "Unknown reason")
                        retries = self.retry_counts.get(doc_id, 0)
                        print(f"   {doc_id}: {reason} (Retries: {retries})")
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
                'retry_counts': self.retry_counts,
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
        self.retry_counts.pop(doc_id, None)  # Clear retry count on success
        self.save_checkpoint()
    
    def mark_failed(self, doc_id, reason="Unknown error"):
        """Mark a document as failed with a reason"""
        self.failed_docs.add(doc_id)
        self.failure_reasons[doc_id] = reason
        # Increment retry count
        self.retry_counts[doc_id] = self.retry_counts.get(doc_id, 0) + 1
        self.save_checkpoint()
    
    def get_retry_count(self, doc_id):
        """Get the number of retry attempts for a document"""
        return self.retry_counts.get(doc_id, 0)
    
    def should_retry(self, doc_id, max_retries=3):
        """Check if a document should be retried based on error category and retry count"""
        if doc_id not in self.failed_docs:
            return True
            
        error_message = self.failure_reasons.get(doc_id, "Unknown error")
        error_category = categorize_error(error_message)
        retry_count = self.get_retry_count(doc_id)
        
        # Log retry decision information
        print(f"🔄 Retry Decision for {doc_id}:")
        print(f"   Error Category: {error_category}")
        print(f"   Current Retry Count: {retry_count}")
        
        should_retry = should_retry_error(error_category, retry_count)
        if not should_retry:
            print(f"❌ Maximum retries reached for {error_category} (max allowed: {max_retries})")
        
        return should_retry
    
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

class ProcessingStats:
    def __init__(self):
        self.start_time = time.time()
        self.processing_times = []  # List of processing times for each document
        self.total_documents = 0
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.current_batch = 0
        
    def add_processing_time(self, duration):
        """Add a new processing time and update averages"""
        self.processing_times.append(duration)
        # Keep only the last 20 times for rolling average
        if len(self.processing_times) > 20:
            self.processing_times.pop(0)
    
    def get_average_time(self):
        """Get rolling average of processing times"""
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_estimated_completion_time(self, remaining_docs):
        """Calculate estimated completion time based on rolling average"""
        if not self.processing_times:
            return "Calculating..."
            
        avg_time = self.get_average_time()
        total_remaining_time = avg_time * remaining_docs
        
        # Convert to hours and minutes
        hours = int(total_remaining_time // 3600)
        minutes = int((total_remaining_time % 3600) // 60)
        
        if hours > 0:
            return f"~{hours}h {minutes}m"
        else:
            return f"~{minutes}m"
    
    def get_processing_rate(self):
        """Calculate current processing rate (docs per minute)"""
        if not self.processing_times:
            return 0
        avg_time = self.get_average_time()
        if avg_time == 0:
            return 0
        return 60 / avg_time  # Convert to docs per minute
    
    def update_counts(self, processed=0, failed=0, skipped=0):
        """Update document counts"""
        self.processed_count += processed
        self.failed_count += failed
        self.skipped_count += skipped
    
    def get_progress(self):
        """Calculate overall progress percentage"""
        if self.total_documents == 0:
            return 0
        total_processed = self.processed_count + self.failed_count + self.skipped_count
        return (total_processed / self.total_documents) * 100

def process_documents():
    """Process documents from Shoeboxed in batches"""
    try:
        print("\n🔍 DEBUG: Starting document processing...", flush=True)
        
        # Load tokens and ensure we have access token
        print("\n🔑 DEBUG: Loading tokens...", flush=True)
        tokens = load_tokens()
        if not tokens or 'access_token' not in tokens:
            st.error("No valid Shoeboxed access token found. Please authenticate first.")
            st.session_state.processing_active = False
            return
            
        # Store access token in environment for other functions to use
        os.environ['SHOEBOXED_ACCESS_TOKEN'] = tokens['access_token']
        
        # Add token refresh before starting main processing
        tokens = refresh_if_needed(tokens)
        if not tokens:
            st.error("Failed to refresh token. Please authenticate again.")
            st.session_state.processing_active = False
            return
            
        # Update access token in environment after refresh
        os.environ['SHOEBOXED_ACCESS_TOKEN'] = tokens['access_token']
        
        # Get organization ID
        account_id = get_organization_id(tokens['access_token'])
        if not account_id:
            st.error("Failed to get organization ID")
            st.session_state.processing_active = False
            return

        # Create main title and description
        st.title("Document Processing")
        st.write("Processing documents from Shoeboxed and updating the knowledge base.")
        
        # Initialize checkpoint for tracking processing state
        checkpoint = ProcessingCheckpoint()
        
        # Set up UI elements for progress tracking
        progress_container = st.container()
        with progress_container:
            # Create metrics at the top
            col1, col2, col3 = st.columns(3)
            total_metric = col1.empty()
            processed_metric = col2.empty()
            failed_metric = col3.empty()
            
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processing details (collapsible)
            with st.expander("Processing Details", expanded=False):
                processing_details = st.empty()
        
        # Initialize checkpoint if not exists
        if not hasattr(st.session_state, 'processor'):
            st.session_state.processor = DocumentProcessor()
        
        # Get initial counts from checkpoint
        processed_count = len(checkpoint.processed_docs)
        failed_count = len(checkpoint.failed_docs)
        skipped_count = len(checkpoint.skipped_docs)
        
        # Retrieve document IDs
        print("\n🔍 DEBUG: Retrieving document IDs...", flush=True)
        all_document_ids = retrieve_all_document_ids(tokens, checkpoint)
        total_documents = len(all_document_ids)
        
        # Calculate grand total (including already processed)
        grand_total = total_documents + processed_count + failed_count + skipped_count
        remaining_to_process = total_documents
        
        # Update metrics with correct counts
        total_metric.metric(
            "Total Documents",
            f"{grand_total:,}",
            f"{remaining_to_process:,} to process"
        )
        processed_metric.metric(
            "Processed",
            f"{processed_count:,}",
            None
        )
        failed_metric.metric(
            "Failed",
            f"{failed_count:,}",
            None
        )
        
        if total_documents == 0:
            print("✅ No new documents to process!")
            st.success("All documents have been processed!")
            st.session_state.processing_active = False
            return
        
        print(f"\n📊 Processing Summary:")
        print(f"{'='*80}")
        print(f"Total Documents: {grand_total:,}")
        print(f"Already Processed: {processed_count:,}")
        print(f"Failed Previously: {failed_count:,}")
        print(f"Skipped: {skipped_count:,}")
        print(f"To Be Processed: {remaining_to_process:,}")
        print(f"{'='*80}\n")
        
        # Process documents in batches
        batch_size = 10
        total_batches = (total_documents + batch_size - 1) // batch_size
        
        for batch_index in range(total_batches):
            if not st.session_state.processing_active:
                print("Processing stopped.")
                break
                
            if st.session_state.paused:
                status_text.text("⏸️ Processing paused... Click Resume to continue")
                time.sleep(1)  # Increased sleep time to reduce CPU usage
                continue
                
            # Refresh token at the start of each batch
            tokens = refresh_if_needed(tokens)
            if not tokens:
                st.error("Failed to refresh token. Please authenticate again.")
                st.session_state.processing_active = False
                return
            
            # Update access token in environment
            os.environ['SHOEBOXED_ACCESS_TOKEN'] = tokens['access_token']
            
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, total_documents)
            current_batch = all_document_ids[start_idx:end_idx]
            
            # Get document details for the batch
            batch_documents = []
            for doc_id in current_batch:
                doc_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}"
                response = requests.get(doc_url, headers=get_shoeboxed_headers(tokens['access_token']))
                
                if response.status_code == 200:
                    doc_data = response.json()
                    batch_documents.append(doc_data)
                else:
                    print(f"❌ Failed to get document details for {doc_id}. Status: {response.status_code}")
                    continue
            
            # Process the batch using DocumentProcessor
            st.session_state.processor.process_batch(
                batch_documents,
                progress_bar,
                status_text,
                [total_metric, processed_metric, failed_metric],
                processing_details,
                None
            )
            
            # Save checkpoint after each batch
            checkpoint.update_batch(batch_index)
            checkpoint.save_checkpoint()
            
            # Update metrics after batch processing
            current_processed = len(checkpoint.processed_docs)
            current_failed = len(checkpoint.failed_docs)
            remaining = total_documents - (current_processed - processed_count)
            
            total_metric.metric(
                "Total Documents",
                f"{grand_total:,}",
                f"{remaining:,} to process"
            )
            processed_metric.metric(
                "Processed",
                f"{current_processed:,}",
                f"+{current_processed - processed_count}"
            )
            failed_metric.metric(
                "Failed",
                f"{current_failed:,}",
                f"+{current_failed - failed_count}" if current_failed > failed_count else None
            )
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        print("\n✅ Document processing complete!")
        
        # Clean up processor after completion
        st.session_state.processing_active = False
        st.session_state.paused = False
        if hasattr(st.session_state, 'processor'):
            delattr(st.session_state, 'processor')
        
    except Exception as e:
        print(f"❌ Error in document processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"An error occurred during processing: {str(e)}")
        
        # Clean up processor on error
        st.session_state.processing_active = False
        st.session_state.paused = False
        if hasattr(st.session_state, 'processor'):
            delattr(st.session_state, 'processor')

def chat_interface():
    """Enhanced chat interface for second brain interaction"""
    try:
        # Initialize chat messages if not exists
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Initialize Pinecone
        print("\n🔍 Initializing Pinecone connection...")
        index = init_pinecone()
        if not index:
            st.error("Failed to initialize Pinecone. Please check your configuration.")
            return
        print("✅ Pinecone connection established")
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents"):
            print(f"\n📝 Received query: {prompt}")
            
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Create embedding for the query
            print("🔄 Creating query embedding...")
            query_embedding = create_embedding(prompt)
            if not query_embedding:
                st.error("Failed to create query embedding. Please try again.")
                return
            print("✅ Query embedding created")
            
            # Search Pinecone
            print("🔍 Searching Pinecone...")
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                matches = results.matches
                print(f"✅ Found {len(matches)} matching chunks")
                
                # Debug: Print raw results
                print("\n🔍 Raw Pinecone results:")
                for match in matches:
                    print(f"Score: {match.score}")
                    print(f"Metadata: {match.metadata}")
                    print("---")
                
            except Exception as e:
                print(f"❌ Pinecone query error: {str(e)}")
                st.error(f"Error searching documents: {str(e)}")
                return
            
            if not matches:
                print("⚠️ No matching documents found")
                st.warning("No relevant documents found for your query. Please try a different question.")
                return
            
            # Build context from matches
            context = []
            for match in matches:
                metadata = match.metadata
                score = match.score
                
                # Format document context
                doc_context = f"""
                Document (Relevance: {score:.2f}):
                Type: {metadata.get('type', 'Unknown')}
                Created: {metadata.get('created', 'Unknown')}
                Content: {metadata.get('text', 'No text available')}
                """
                context.append(doc_context)
            
            # Combine all context
            full_context = "\n\n".join(context)
            
            # Generate AI response
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an intelligent AI assistant acting as a second brain for the user. 
                    You have access to their document repository and should help them find and understand their information.
                    
                    Current context from documents:
                    {full_context}"""
                }
            ]
            
            # Add chat history for context
            for message in st.session_state.chat_messages[-3:]:  # Last 3 messages
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            
            try:
                response = client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4'),
                    messages=messages,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                print(f"✅ Generated AI response: {ai_response[:100]}...")
                
                # Update chat history
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                
            except Exception as e:
                print(f"⚠️ Error generating response: {str(e)}")
                st.error("I apologize, but I encountered an error while generating a response. Please try again.")
    
    except Exception as e:
        print(f"❌ Error in chat interface: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"Error in chat interface: {str(e)}")

def format_precise_context(context, depth):
    """Format context for precise mode"""
    formatted_chunks = []
    
    # Sort chunks by score
    sorted_chunks = sorted(context['chunks'], key=lambda x: x['score'], reverse=True)
    
    for chunk in sorted_chunks:
        chunk_text = chunk['text']
        metadata = chunk['metadata']
        
        # Format based on depth
        if depth == "concise":
            formatted_chunks.append(f"Document content: {chunk_text}")
        else:
            formatted_chunks.append(f"""
            Document Type: {metadata.get('type', 'Unknown')}
            Creation Date: {metadata.get('temporal_data', {}).get('creation_date', 'Unknown')}
            Content: {chunk_text}
            Topics: {', '.join(metadata.get('content_analysis', {}).get('topics', []))}
            """)
    
    return "\n\n".join(formatted_chunks)

def format_exploratory_context(context, depth, intent):
    """Format context for exploratory mode"""
    sections = []
    
    # Add direct content
    content_section = format_precise_context(context, depth)
    sections.append("Direct Content:\n" + content_section)
    
    # Add relationship insights
    if context['relationship_chains']:
        relationship_section = "Related Document Chains:\n"
        for chain in context['relationship_chains']:
            relationship_section += f"""
            Chain starting from {chain['start_doc']}:
            - Connected to: {', '.join(chain['chain'])}
            - Through: {', '.join(chain['relationship_types'])}
            - Overall connection strength: {chain['total_strength']:.2f}
            """
        sections.append(relationship_section)
    
    # Add temporal insights
    if context['temporal_context']:
        temporal_section = "Temporal Context:\n"
        sorted_temporal = sorted(context['temporal_context'], key=lambda x: x['date'] if x['date'] else "")
        for temp_context in sorted_temporal:
            temporal_section += f"""
            Date: {temp_context['date']}
            Document: {temp_context['document_id']}
            Time References: {', '.join(temp_context['temporal_references'])}
            """
        sections.append(temporal_section)
    
    # Add knowledge graph insights
    if context['knowledge_graph']['nodes']:
        graph_section = "Knowledge Graph Insights:\n"
        for node in context['knowledge_graph']['nodes']:
            graph_section += f"""
            Topic: {node['node_type']} 
            Key Concepts: {', '.join(node['concepts'])}
            Importance: {node['importance']}/10
            """
        sections.append(graph_section)
    
    return "\n\n".join(sections)

def format_smart_context(context, depth):
    """Format context for smart mode (balanced approach)"""
    sections = []
    
    # Add most relevant direct content
    top_chunks = sorted(context['chunks'], key=lambda x: x['score'], reverse=True)[:3]
    content_section = "Most Relevant Content:\n"
    for chunk in top_chunks:
        content_section += f"""
        {chunk['text']}
        (Relevance: {chunk['score']:.2f})
        """
    sections.append(content_section)
    
    # Add key relationships if available
    if context['knowledge_graph']['edges']:
        relationships_section = "Key Relationships:\n"
        strong_connections = [
            edge for edge in context['knowledge_graph']['edges']
            if edge['strength'] > 0.8
        ][:3]
        for connection in strong_connections:
            relationships_section += f"""
            {connection['source']} → {connection['target']}
            Type: {connection['type']}
            """
        sections.append(relationships_section)
    
    # Add temporal context if relevant
    if context['temporal_context']:
        temporal_section = "Temporal Context:\n"
        recent_contexts = sorted(
            context['temporal_context'],
            key=lambda x: x['date'] if x['date'] else "",
            reverse=True
        )[:2]
        for temp_context in recent_contexts:
            temporal_section += f"""
            Date: {temp_context['date']}
            References: {', '.join(temp_context['temporal_references'][:3])}
            """
        sections.append(temporal_section)
    
    return "\n\n".join(sections)

def add_precise_citations(response, context):
    """Add precise citations to response"""
    citations = []
    for chunk in context['chunks']:
        metadata = chunk['metadata']
        citations.append(f"""
        Source: {metadata.get('type', 'Document')} {metadata.get('document_id')}
        Date: {metadata.get('temporal_data', {}).get('creation_date', 'unknown')}
        Relevance: {chunk['score']:.2f}
        """)
    return f"{response}\n\nSources:\n{''.join(citations)}"

def add_exploratory_citations(response, context):
    """Add exploratory citations with relationship context"""
    citations = []
    
    # Document citations
    citations.append("Direct Sources:")
    for chunk in context['chunks']:
        metadata = chunk['metadata']
        citations.append(f"""
        {metadata.get('type', 'Document')} {metadata.get('document_id')}
        - Topics: {', '.join(metadata.get('content_analysis', {}).get('topics', []))}
        - Key Concepts: {', '.join(metadata.get('content_analysis', {}).get('key_concepts', []))}
        """)
    
    # Relationship citations
    if context['relationship_chains']:
        citations.append("\nRelated Document Chains:")
        for chain in context['relationship_chains']:
            citations.append(f"""
            Chain from {chain['start_doc']}
            - Connected documents: {' → '.join(chain['chain'])}
            - Connection types: {', '.join(chain['relationship_types'])}
            """)
    
    return f"{response}\n\nReference Context:\n{''.join(citations)}"

def add_smart_citations(response, context):
    """Add balanced citations with key information"""
    citations = []
    
    # Add most relevant sources
    top_chunks = sorted(context['chunks'], key=lambda x: x['score'], reverse=True)[:3]
    citations.append("Key Sources:")
    for chunk in top_chunks:
        metadata = chunk['metadata']
        citations.append(f"""
        {metadata.get('type', 'Document')} {metadata.get('document_id')}
        - Date: {metadata.get('temporal_data', {}).get('creation_date', 'unknown')}
        - Topics: {', '.join(metadata.get('content_analysis', {}).get('topics', []))[:2]}
        """)
    
    # Add key relationships if highly relevant
    strong_connections = [
        edge for edge in context['knowledge_graph']['edges']
        if edge['strength'] > 0.8
    ][:2]
    if strong_connections:
        citations.append("\nRelated Documents:")
        for connection in strong_connections:
            citations.append(f"""
            {connection['source']} → {connection['target']}
            - Relationship: {connection['type']}
            """)
    
    return f"{response}\n\nSources and Connections:\n{''.join(citations)}"

def get_search_parameters(mode, depth, intent):
    """Get mode-specific search parameters"""
    params = {
        'include_metadata': True,
        'top_k': {
            'concise': 3,
            'standard': 5,
            'detailed': 8
        }[depth]
    }
    
    # Add mode-specific adjustments
    if mode == "precise":
        params['top_k'] = min(params['top_k'], 5)  # Limit results for precision
        params['score_threshold'] = 0.8  # Higher similarity threshold
    elif mode == "exploratory":
        params['top_k'] = params['top_k'] * 2  # Double results for exploration
        params['score_threshold'] = 0.6  # Lower threshold to include more connections
    
    # Add intent-based filters
    filter_dict = build_search_filter(intent)
    if filter_dict:
        params['filter'] = filter_dict
    
    return params

def build_enhanced_context(query_intent, matched_chunks, chat_history):
    """Build enhanced context using query intent, matched chunks, and chat history."""
    context_parts = []
    
    # Add temporal context if available
    if query_intent.get("temporal_context", {}).get("period"):
        context_parts.append(f"Time period of interest: {query_intent['temporal_context']['period']}")
    
    # Group chunks by document type
    chunks_by_type = {}
    for chunk in matched_chunks:
        doc_type = chunk.get("metadata", {}).get("document_type", "unknown")
        if doc_type not in chunks_by_type:
            chunks_by_type[doc_type] = []
        chunks_by_type[doc_type].append(chunk)
    
    # Add document type summaries
    for doc_type, chunks in chunks_by_type.items():
        if chunks:
            context_parts.append(f"\nRelevant {doc_type} documents:")
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                date = metadata.get("date", "unknown date")
                source = metadata.get("source", "unknown source")
                context_parts.append(f"- {chunk['text']} (from {source}, {date})")
    
    # Add relationship context if available
    if query_intent.get("relationship_types"):
        context_parts.append("\nRelated information:")
        for rel_type in query_intent["relationship_types"]:
            # Here we would traverse the knowledge graph based on relationship type
            # For now, we'll just note the relationship type
            context_parts.append(f"- Looking for {rel_type} relationships")
    
    # Add recent chat context if relevant
    if chat_history:
        recent_context = chat_history[-3:]  # Last 3 messages
        context_parts.append("\nRecent conversation context:")
        for msg in recent_context:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"- {role}: {content}")
    
    return "\n".join(context_parts)

def create_enhanced_system_prompt(depth, mode, intent):
    """Create mode-specific system prompt"""
    base_prompt = """You are an intelligent assistant analyzing a personal document archive. 
    This archive contains various types of documents including handwritten notes, essays, medical records, 
    receipts, and other personal papers."""
    
    mode_specific = {
        "precise": "Focus on providing accurate, factual information directly from the documents.",
        "exploratory": "Explore connections between documents and suggest related topics or insights.",
        "smart": "Balance accuracy with insightful connections, providing relevant context and relationships."
    }[mode]
    
    depth_specific = {
        "concise": "Provide brief, focused answers using only the most relevant information.",
        "standard": "Balance detail and brevity, highlighting key information and relationships.",
        "detailed": "Provide comprehensive answers, including supporting details, relationships, and citations."
    }[depth]
    
    knowledge_graph_prompt = """
    Use the knowledge graph to:
    1) Identify key concept clusters
    2) Follow relationship chains
    3) Consider temporal evolution of topics
    4) Highlight important cross-document insights
    """
    
    return f"{base_prompt}\n\n{mode_specific}\n\n{depth_specific}\n\n{knowledge_graph_prompt}"

def enhance_response_with_citations(response, context, mode):
    """Add enhanced citations and evidence to response"""
    if mode == "precise":
        return add_precise_citations(response, context)
    elif mode == "exploratory":
        return add_exploratory_citations(response, context)
    else:
        return add_smart_citations(response, context)

def show_relationship_visualizations(context):
    """Show visualizations of document relationships"""
    if st.checkbox("Show Document Relationships"):
        st.write("### Document Relationship Graph")
        # Here you would add code to visualize the relationships
        # using a library like networkx or pyvis
        pass

def build_search_filter(query_intent):
    """Build Pinecone filter based on query intent"""
    filter_dict = {}
    
    # Add temporal filters if specified
    if 'temporal_context' in query_intent:
        # Implementation for temporal filtering
        pass
    
    # Add document type filters if specified
    if 'required_document_types' in query_intent and query_intent['required_document_types'] != ['all']:
        filter_dict['type'] = {'$in': query_intent['required_document_types']}
    
    # Only return filter_dict if it has filters
    return filter_dict if filter_dict else None

def format_document_context(metadata, score, depth, query_intent):
    """Format document context based on depth and relevance"""
    context_parts = []
    
    # Basic document info
    context_parts.append(f"""
    Document (Relevance: {score:.2f})
    Type: {metadata.get('type', 'Unknown')}
    Date: {metadata.get('temporal_data', {}).get('creation_date', 'Unknown')}
    """)
    
    # Add document structure info
    structure = metadata.get('document_structure', {})
    context_parts.append("""
    Document Structure:
    - {'Multi-page' if structure.get('multi_page') else 'Single-page'} document
    - {'Contains' if structure.get('has_handwriting') else 'No'} handwriting
    - {'Contains' if structure.get('has_drawings') else 'No'} drawings/diagrams
    - {'Contains' if structure.get('has_tables') else 'No'} tables
    - {'Contains' if structure.get('has_lists') else 'No'} lists
    """)
    
    # Add content analysis based on depth
    if depth in ["standard", "detailed"]:
        content_analysis = metadata.get('content_analysis', {})
        context_parts.append(f"""
        Key Concepts: {', '.join(content_analysis.get('key_concepts', []))}
        Topics: {', '.join(content_analysis.get('topics', []))}
        """)
        
        if depth == "detailed":
            context_parts.append(f"""
            Named Entities: {json.dumps(content_analysis.get('named_entities', {}), indent=2)}
            Action Items: {', '.join(content_analysis.get('action_items', []))}
            Sentiment: {content_analysis.get('sentiment', 'neutral')}
            Importance: {content_analysis.get('importance_score', 5)}/10
            """)
    
    # Add document content
    context_parts.append(f"""
    Content: {metadata.get('text', 'No text available')}
    """)
    
    # Add relationships if available
    if depth == "detailed":
        relationships = metadata.get('relationships', {})
        if any(relationships.values()):
            context_parts.append("""
            Related Documents:
            - References: {', '.join(relationships.get('references', []))}
            - Referenced by: {', '.join(relationships.get('referenced_by', []))}
            - Follows up on: {', '.join(relationships.get('follows_up', []))}
            """)
    
    return "\n".join(context_parts)

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
        
        # Reset session state variables first
        for key in list(st.session_state.keys()):
            if key not in ['authenticated']:  # Keep authentication state
                del st.session_state[key]
        
        # Initialize clean state
        st.session_state.processing_active = False
        st.session_state.paused = False
        st.session_state.current_interface = 'processor'
        
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
        
        # Delete all local state and cache files (except auth)
        files_to_delete = [
            'processing_checkpoint.json',
            'document_id_cache.json',
            'retrieval_metadata.json',
            'processing_progress.json',
            'embedding_cache.json',
            'document_list_cache.json'
        ]  # Removed .auth_success from this list
        
        print("\n🗑️ Cleaning up local files...")
        for file in files_to_delete:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"✅ Deleted {file}")
                except Exception as e:
                    print(f"⚠️ Error deleting {file}: {str(e)}")
            else:
                print(f"ℹ️ File not found: {file}")
        
        # Clear all document directories
        print("\n📁 Cleaning up directories...")
        found_dirs = False
        for dir_name in os.listdir('.'):
            if dir_name.startswith('documents_') or dir_name == 'documents' or dir_name == 'cache':
                found_dirs = True
                try:
                    shutil.rmtree(dir_name)
                    print(f"✅ Deleted directory {dir_name}")
                except Exception as e:
                    print(f"⚠️ Error deleting directory {dir_name}: {str(e)}")
        
        if not found_dirs:
            print("ℹ️ No document directories found to clean")
        
        # Clear S3 bucket
        print("\n☁️ Cleaning up S3...")
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            bucket_name = os.getenv('S3_BUCKET_NAME')
            
            if bucket_name:
                print(f"🔄 Clearing S3 bucket {bucket_name}...")
                
                # List and delete all objects in batches
                paginator = s3_client.get_paginator('list_objects_v2')
                total_objects = 0
                
                for page in paginator.paginate(Bucket=bucket_name):
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        if objects:
                            # Delete in batches of 1000 (S3 limit)
                            for i in range(0, len(objects), 1000):
                                batch = objects[i:i + 1000]
                                s3_client.delete_objects(
                                    Bucket=bucket_name,
                                    Delete={'Objects': batch}
                                )
                                total_objects += len(batch)
                
                if total_objects > 0:
                    print(f"✅ Deleted {total_objects} objects from S3")
                else:
                    print("ℹ️ No objects found in S3 to delete")
            else:
                print("ℹ️ S3 bucket name not configured")
                
        except Exception as e:
            print(f"⚠️ Error clearing S3 bucket: {str(e)}")
        
        print("\n✅ Reset complete! The system is ready for fresh document processing.")
        print("ℹ️ Shoeboxed authentication has been preserved.")
        return True
        
    except Exception as e:
        print(f"❌ Error during reset: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Main application function"""
    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    if 'current_interface' not in st.session_state:
        st.session_state.current_interface = 'processor'
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Main authentication flow
    if not st.session_state.authenticated:
        st.title("Shoeboxed Document Processor")
        st.info("Please authenticate with Shoeboxed to begin.")
        if st.button("🔑 Authenticate with Shoeboxed"):
            handle_auth()
    else:
        st.sidebar.success("Authenticated")
        
        # Load checkpoint data immediately after login
        checkpoint = ProcessingCheckpoint()
        
        # Add processing controls with pause/resume
        col1, col2, col3 = st.sidebar.columns(3)
        
        # Process button
        if col1.button("📄 Process"):
            st.session_state.current_interface = 'processor'
            # Clean up any existing processor
            if hasattr(st.session_state, 'processor'):
                delattr(st.session_state, 'processor')
            st.session_state.processing_active = True
            st.session_state.paused = False
            st.session_state.processor = DocumentProcessor()
            st.session_state.processor.checkpoint = checkpoint
            process_documents()
            
        # Pause/Resume button
        pause_button_label = "⏸️ Pause" if not st.session_state.paused else "▶️ Resume"
        if col2.button(pause_button_label):
            st.session_state.paused = not st.session_state.paused
            if st.session_state.paused:
                st.sidebar.warning("Processing paused")
            else:
                st.sidebar.success("Processing resumed")
                # When resuming, ensure processor is active
                if not hasattr(st.session_state, 'processor'):
                    st.session_state.processor = DocumentProcessor()
                    st.session_state.processor.checkpoint = checkpoint
                    process_documents()
            st.rerun()
        
        # Reset button - enabled when processing is paused or not active
        if col3.button("🔄 Reset"):
            try:
                # Stop any running processor
                if hasattr(st.session_state, 'processor'):
                    st.session_state.processor.stop_event.set()
                    delattr(st.session_state, 'processor')
                
                # Call the reset function
                if reset_processing_state():
                    st.success("Reset complete! The system is ready for fresh document processing.")
                else:
                    st.error("Error occurred during reset. Please try again.")
                
                # Force a rerun to refresh the UI
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting processor: {str(e)}")
                print(f"Error in reset: {str(e)}")
                print(traceback.format_exc())
        
        # Chat interface button - enabled only when not processing
        chat_button = st.sidebar.button(
            "💬 Chat with Documents",
            disabled=st.session_state.processing_active,
            key="chat_button"
        )
        
        if chat_button:
            st.session_state.current_interface = 'chat'
            st.rerun()
        
        # Logout button - enabled only when not processing
        logout_button = st.sidebar.button(
            "🚪 Logout",
            disabled=st.session_state.processing_active,
            key="logout_button"
        )
        
        if logout_button and not st.session_state.processing_active:
            if os.path.exists('.auth_success'):
                os.remove('.auth_success')
            # Clear all session state
            st.session_state.clear()
            st.rerun()
        
        # Display appropriate interface based on current_interface
        if st.session_state.current_interface == 'chat':
            st.title("Chat with Your Documents")
            chat_interface()
        else:
            st.title("Shoeboxed Document Processor")
            # Display processing statistics in the main area
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Previously Processed", len(checkpoint.processed_docs))
            col2.metric("❌ Failed Documents", len(checkpoint.failed_docs))
            col3.metric("⏭️ Skipped Documents", len(checkpoint.skipped_docs))
            
            # Show failed document details in an expander if there are any
            if checkpoint.failed_docs:
                with st.expander("Failed Documents Details"):
                    for doc_id in checkpoint.failed_docs:
                        reason = checkpoint.failure_reasons.get(doc_id, "Unknown reason")
                        retries = checkpoint.retry_counts.get(doc_id, 0)
                        st.text(f"Document {doc_id}:")
                        st.text(f"  Reason: {reason}")
                        st.text(f"  Retry attempts: {retries}")
            
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
    """Create comprehensive metadata for document storage"""
    # Get document type and enhance it
    doc_type = doc_data.get('type', 'unknown').lower()
    
    # Extract named entities and concepts using GPT
    try:
        entity_response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Analyze the following text and extract: 1) Named entities (people, places, organizations), 2) Key concepts/themes, 3) Action items/tasks, 4) Important dates, and 5) Related concepts. Format as JSON."
            }, {
                "role": "user",
                "content": extracted_text[:4000]  # First 4000 chars for entity analysis
            }],
            temperature=0.2
        )
        entities_and_concepts = json.loads(entity_response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        entities_and_concepts = {}

    # Analyze document sentiment and importance
    try:
        analysis_response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Analyze the text and provide: 1) Overall sentiment (positive/negative/neutral), 2) Document importance (1-10), 3) Document category/topics. Format as JSON."
            }, {
                "role": "user",
                "content": extracted_text[:4000]
            }],
            temperature=0.2
        )
        document_analysis = json.loads(analysis_response.choices[0].message.content)
    except Exception as e:
        print(f"Error analyzing document: {str(e)}")
        document_analysis = {}

    # Base metadata with enhanced structure
    metadata = {
        # Basic document information
        'document_id': doc_data.get('id'),
        'type': doc_type,
        'uploaded_date': doc_data.get('uploaded', '') or '',
        'modified_date': doc_data.get('modified', '') or '',
        'categories': doc_data.get('categories', []) or [],
        'text': (extracted_text[:4000] if extracted_text else '') or '',
        'processing_state': doc_data.get('processingState', 'unknown'),
        
        # Document characteristics
        'document_structure': {
            'multi_page': '\nPage ' in extracted_text,
            'has_tables': 'table' in extracted_text.lower(),
            'has_lists': any(marker in extracted_text for marker in ['•', '-', '1.', '2.', '3.']),
            'has_handwriting': any(term in extracted_text.lower() for term in ['handwritten', 'handwriting', 'written by hand']),
            'has_drawings': any(term in extracted_text.lower() for term in ['drawing', 'diagram', 'sketch', 'illustration']),
            'has_signatures': any(term in extracted_text.lower() for term in ['signed', 'signature']),
            'has_forms': any(term in extracted_text.lower() for term in ['form', 'checkbox', 'field', 'fill out']),
        },
        
        # Enhanced content understanding
        'content_analysis': {
            'named_entities': entities_and_concepts.get('named_entities', {}),
            'key_concepts': entities_and_concepts.get('key_concepts', []),
            'action_items': entities_and_concepts.get('action_items', []),
            'related_concepts': entities_and_concepts.get('related_concepts', []),
            'sentiment': document_analysis.get('sentiment', 'neutral'),
            'importance_score': document_analysis.get('importance', 5),
            'topics': document_analysis.get('topics', []),
        },
        
        # Temporal information
        'temporal_data': {
            'creation_date': doc_data.get('created', ''),
            'dates_mentioned': extract_dates(extracted_text),
            'time_references': extract_time_references(extracted_text),
        },
        
        # Source information
        'source_metadata': {
            'origin_system': 'shoeboxed',
            'original_filename': doc_data.get('attachment', {}).get('filename', ''),
            'file_type': doc_data.get('attachment', {}).get('type', 'unknown'),
            'confidence_score': calculate_confidence_score(extracted_text),
        },
        
        # Relationship tracking
        'relationships': {
            'related_documents': [],  # To be populated by relationship analysis
            'references': [],         # Documents this one references
            'referenced_by': [],      # Documents that reference this one
            'follows_up': [],         # Documents this one follows up on
            'followed_by': [],        # Documents that follow up on this one
        }
    }
    
    # Add type-specific metadata
    if doc_type == 'receipt':
        metadata.update({
            'financial_data': {
                'vendor': doc_data.get('vendor') or 'unknown',
                'total': float(doc_data.get('total', 0) or 0),
                'currency': doc_data.get('currency', 'USD'),
                'payment_method': extract_payment_method(extracted_text),
                'items': extract_line_items(extracted_text),
            }
        })
    elif doc_type == 'document':
        metadata.update({
            'document_specific': {
                'author': extract_author(extracted_text),
                'recipients': extract_recipients(extracted_text),
                'subject': extract_subject(extracted_text),
            }
        })
    
    return metadata

def extract_dates(text):
    """Extract dates from text using comprehensive patterns"""
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY, DD/MM/YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # YYYY-MM-DD
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',  # Full month
        r'\b\d{1,2}(?:st|nd|rd|th) (?:of )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'  # 1st of January 2024
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(set(dates))

def extract_time_references(text):
    """Extract temporal references from text"""
    time_patterns = [
        r'\b(?:today|tomorrow|yesterday)\b',
        r'\b(?:next|last) (?:week|month|year)\b',
        r'\b(?:in|after|before) \d+ (?:days?|weeks?|months?|years?)\b',
        r'\b(?:morning|afternoon|evening|night)\b',
        r'\b\d{1,2}:\d{2} ?(?:AM|PM|am|pm)?\b'
    ]
    references = []
    for pattern in time_patterns:
        references.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(set(references))

def calculate_confidence_score(text):
    """Calculate confidence score based on text quality"""
    score = 1.0
    if len(text.strip()) < 100:
        score *= 0.8
    if text.count('\n') < 2:
        score *= 0.9
    if any(marker in text.lower() for marker in ['unclear', 'illegible', 'unreadable']):
        score *= 0.7
    return round(score, 2)

def extract_payment_method(text):
    """Extract payment method from text"""
    payment_methods = {
        'credit': ['credit card', 'visa', 'mastercard', 'amex', 'discover'],
        'debit': ['debit card', 'debit'],
        'cash': ['cash', 'paid in cash'],
        'check': ['check', 'cheque', 'check #'],
        'digital': ['paypal', 'venmo', 'zelle', 'apple pay', 'google pay']
    }
    text_lower = text.lower()
    for method, keywords in payment_methods.items():
        if any(keyword in text_lower for keyword in keywords):
            return method
    return 'unknown'

def extract_line_items(text):
    """Extract line items from receipt text"""
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Extract line items from this receipt text. Format as a list of JSON objects with 'item', 'quantity', and 'price' fields."
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except:
        return []

def extract_author(text):
    """Extract author information from document"""
    # Implementation would use GPT to identify author
    return None

def extract_recipients(text):
    """Extract recipient information from document"""
    # Implementation would use GPT to identify recipients
    return []

def extract_subject(text):
    """Extract subject/title from document"""
    # Implementation would use GPT to identify subject
    return None

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

def categorize_error(error_message):
    """Categorize errors to help with retry strategies"""
    error_message = error_message.lower()
    if any(term in error_message for term in ['timeout', 'connection refused', 'connection error']):
        return 'NETWORK_ERROR'
    elif any(term in error_message for term in ['token', 'unauthorized', 'authentication']):
        return 'AUTH_ERROR'
    elif any(term in error_message for term in ['rate limit', 'too many requests']):
        return 'RATE_LIMIT'
    elif any(term in error_message for term in ['not found', '404']):
        return 'NOT_FOUND'
    elif any(term in error_message for term in ['server error', '500', '502', '503', '504']):
        return 'SERVER_ERROR'
    elif 'file size' in error_message or 'too large' in error_message:
        return 'SIZE_ERROR'
    else:
        return 'UNKNOWN_ERROR'

def should_retry_error(error_category, retry_count):
    """Determine if an error should be retried based on its category"""
    max_retries = {
        'NETWORK_ERROR': 5,
        'AUTH_ERROR': 3,
        'RATE_LIMIT': 5,
        'NOT_FOUND': 1,
        'SERVER_ERROR': 4,
        'SIZE_ERROR': 2,
        'UNKNOWN_ERROR': 3
    }
    return retry_count < max_retries.get(error_category, 3)

class ProcessingResult:
    def __init__(self, doc_id, success, processing_time, error=None):
        self.doc_id = doc_id
        self.success = success
        self.processing_time = processing_time
        self.error = error

class DocumentProcessor:
    def __init__(self):
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.checkpoint = ProcessingCheckpoint()
        self.metrics = ProcessingMetrics()
        
    def process_document_worker(self):
        """Worker thread for processing documents"""
        while not self.stop_event.is_set():
            try:
                # Check for pause state
                if st.session_state.get('paused', False):
                    self.metrics.pause()
                    time.sleep(1)
                    continue
                else:
                    self.metrics.resume()
                
                try:
                    doc = self.processing_queue.get_nowait()
                    if not isinstance(doc, dict) or 'id' not in doc:
                        print(f"⚠️ Invalid document format: {doc}")
                        continue
                except queue.Empty:
                    break
                
                start_time = time.time()
                try:
                    # Load fresh tokens for each document
                    current_tokens = load_tokens()
                    if not current_tokens:
                        raise Exception("Failed to load tokens")
                    
                    success = process_single_document(doc, current_tokens, self.checkpoint)
                    processing_time = time.time() - start_time
                    
                    with self.lock:
                        if success:
                            self.metrics.processed_count += 1
                        else:
                            self.metrics.failed_count += 1
                        self.metrics.add_processing_time(processing_time)
                    
                    self.results_queue.put(ProcessingResult(
                        doc_id=doc['id'],
                        success=success,
                        processing_time=processing_time
                    ))
                
                except Exception as e:
                    error_msg = f"Error processing document {doc.get('id', 'Unknown')}: {str(e)}"
                    print(f"❌ {error_msg}")
                    
                    with self.lock:
                        self.metrics.failed_count += 1
                        self.metrics.add_processing_time(time.time() - start_time)
                    
                    self.results_queue.put(ProcessingResult(
                        doc_id=doc.get('id', 'Unknown'),
                        success=False,
                        processing_time=time.time() - start_time,
                        error=str(e)
                    ))
                
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                print(f"Error in worker thread: {str(e)}")
                print(traceback.format_exc())
                continue

    def process_batch(self, documents, progress_bar, status_text, metrics_cols, processing_details, recent_files):
        """Process a batch of documents with real-time progress updates"""
        try:
            batch_start_time = time.time()
            
            # Initialize processing queue if empty
            if self.processing_queue.empty():
                for doc in documents:
                    self.processing_queue.put(doc)
                self.metrics.total_documents = len(self.checkpoint.processed_docs) + len(documents)
            
            # Create and start worker threads
            num_threads = min(MAX_THREADS, len(documents))
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for _ in range(num_threads):
                    futures.append(executor.submit(self.process_document_worker))
                
                # Monitor progress while threads are running
                while any(not f.done() for f in futures):
                    try:
                        # Process results and update UI
                        self._process_results(progress_bar, status_text, metrics_cols, processing_details)
                        
                        # Get current progress stats
                        stats = self.metrics.get_progress_stats()
                        completion_estimate = stats["estimated_completion"]
                        
                        # Update status message
                        if st.session_state.get('paused', False):
                            status_text.text("⏸️ Processing paused... Click Resume to continue")
                        else:
                            rate = stats["processing_rate"]["docs_per_minute"]
                            status_text.text(
                                f"🔄 Progress: {stats['progress_percentage']:.1f}% | "
                                f"Rate: {rate:.1f} docs/min | "
                                f"Est. remaining: {completion_estimate['time_remaining']} "
                                f"(Accuracy: {completion_estimate['accuracy']})"
                            )
                        
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error in progress monitoring: {str(e)}")
                        continue
                
                # Process any remaining results
                self._process_results(progress_bar, status_text, metrics_cols, processing_details)
                
                # Wait for all workers to complete
                concurrent.futures.wait(futures)
                
                # Record batch processing time
                self.metrics.add_batch_time(time.time() - batch_start_time)
                
        except Exception as e:
            print(f"Error in process_batch: {str(e)}")
            print(traceback.format_exc())
    
    def _process_results(self, progress_bar, status_text, metrics_cols, processing_details):
        """Process results from the results queue and update UI"""
        try:
            while True:
                try:
                    result = self.results_queue.get_nowait()
                except queue.Empty:
                    break
                
                # Get current progress stats
                stats = self.metrics.get_progress_stats()
                
                # Update progress bar
                progress_bar.progress(stats["progress_percentage"] / 100)
                
                # Update metrics
                metrics_cols[0].metric(
                    "Total Documents",
                    f"{stats['total_documents']:,}",
                    f"{stats['remaining']:,} remaining"
                )
                metrics_cols[1].metric(
                    "Processed",
                    f"{stats['processed']:,}",
                    f"+{stats['processed'] - self.checkpoint.processed_count}" if stats['processed'] > self.checkpoint.processed_count else None
                )
                metrics_cols[2].metric(
                    "Failed",
                    f"{stats['failed']:,}",
                    f"+{stats['failed'] - len(self.checkpoint.failed_docs)}" if stats['failed'] > len(self.checkpoint.failed_docs) else None
                )
                
                # Log processing details
                if result.success:
                    processing_details.info(
                        f"✅ Processed document {result.doc_id} ({result.processing_time:.1f}s) | "
                        f"Rate: {stats['processing_rate']['docs_per_minute']:.1f} docs/min"
                    )
                else:
                    processing_details.error(
                        f"❌ Failed document {result.doc_id}: {result.error} "
                        f"({result.processing_time:.1f}s)"
                    )
                
        except Exception as e:
            print(f"Error processing results: {str(e)}")
            print(traceback.format_exc())

def chunk_document(text, metadata, max_chunk_size=1000):
    """
    Split document into semantic chunks while preserving context
    
    Args:
        text (str): Full document text
        metadata (dict): Document metadata
        max_chunk_size (int): Maximum chunk size in characters
    
    Returns:
        list: List of chunks with their metadata
    """
    try:
        # Use GPT to identify semantic boundaries
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Split this text into semantic chunks. Preserve paragraph and section boundaries. Each chunk should be self-contained and meaningful."
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.2
        )
        
        chunks = json.loads(response.choices[0].message.content)
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create chunk-specific metadata
            chunk_metadata = {
                **metadata,  # Base metadata
                'chunk_info': {
                    'chunk_id': f"{metadata['document_id']}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'is_first_chunk': i == 0,
                    'is_last_chunk': i == len(chunks) - 1,
                    'semantic_context': analyze_chunk_context(chunk)
                }
            }
            processed_chunks.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return processed_chunks
    except Exception as e:
        print(f"❌ Error chunking document: {str(e)}")
        return [{'text': text, 'metadata': metadata}]

def analyze_chunk_context(chunk_text):
    """Analyze the semantic context of a chunk"""
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Analyze this text chunk and provide: 1) Main topic, 2) Key entities, 3) Section type (e.g., introduction, conclusion, details), 4) Information density (1-10)"
            }, {
                "role": "user",
                "content": chunk_text
            }],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Error analyzing chunk context: {str(e)}")
        return {}

def analyze_document_relationships(doc_id, metadata, index):
    """
    Analyze relationships between documents using semantic similarity and temporal context
    
    Args:
        doc_id (str): Current document ID
        metadata (dict): Document metadata
        index: Pinecone index
    
    Returns:
        dict: Enhanced relationship metadata
    """
    try:
        # Get potential related documents based on temporal proximity
        temporal_data = metadata['temporal_data']
        creation_date = temporal_data.get('creation_date')
        
        # Query for documents with similar topics or entities
        query_embedding = create_embedding(
            f"{' '.join(metadata['content_analysis']['key_concepts'])} {' '.join(metadata['content_analysis']['topics'])}"
        )
        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={
                'document_id': {'$ne': doc_id}  # Exclude current document
            }
        )
        
        relationships = {
            'semantic_connections': [],
            'temporal_connections': [],
            'topical_threads': [],
            'reference_chain': []
        }
        
        # Analyze each potential relationship
        for match in results.matches:
            related_metadata = match.metadata
            
            # Calculate relationship strength and type
            relationship = analyze_document_connection(
                metadata,
                related_metadata,
                match.score
            )
            
            if relationship['strength'] > 0.7:  # Strong relationship threshold
                relationships['semantic_connections'].append({
                    'document_id': related_metadata['document_id'],
                    'relationship_type': relationship['type'],
                    'strength': relationship['strength'],
                    'shared_concepts': relationship['shared_concepts']
                })
        
        # Update metadata with relationship information
        metadata['relationships'] = relationships
        
        # Create knowledge graph connections
        metadata['knowledge_graph'] = create_knowledge_graph_nodes(
            doc_id,
            metadata,
            relationships
        )
        
        return metadata
    except Exception as e:
        print(f"❌ Error analyzing document relationships: {str(e)}")
        return metadata

def analyze_document_connection(doc1_metadata, doc2_metadata, similarity_score):
    """Analyze the connection between two documents"""
    try:
        # Prepare context for analysis
        context = {
            'doc1': {
                'concepts': doc1_metadata['content_analysis']['key_concepts'],
                'topics': doc1_metadata['content_analysis']['topics'],
                'temporal': doc1_metadata['temporal_data']
            },
            'doc2': {
                'concepts': doc2_metadata['content_analysis']['key_concepts'],
                'topics': doc2_metadata['content_analysis']['topics'],
                'temporal': doc2_metadata['temporal_data']
            },
            'similarity_score': similarity_score
        }
        
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{
                "role": "system",
                "content": "Analyze the relationship between these documents and provide: 1) Relationship type (e.g., follow-up, reference, related topic), 2) Relationship strength (0-1), 3) Shared concepts"
            }, {
                "role": "user",
                "content": json.dumps(context)
            }],
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Error analyzing document connection: {str(e)}")
        return {'type': 'unknown', 'strength': 0, 'shared_concepts': []}

def create_knowledge_graph_nodes(doc_id, metadata, relationships):
    """Create knowledge graph representation of document connections"""
    return {
        'node_id': doc_id,
        'node_type': metadata['type'],
        'concepts': metadata['content_analysis']['key_concepts'],
        'connections': relationships['semantic_connections'],
        'temporal_position': metadata['temporal_data']['creation_date'],
        'importance': metadata['content_analysis']['importance_score']
    }

# Update store_in_pinecone function to use chunking
def store_in_pinecone(index, doc_id, extracted_text, metadata):
    """Store document chunks in Pinecone with enhanced metadata"""
    try:
        # Create chunks
        chunks = chunk_document(extracted_text, metadata)
        success = True
        
        # Analyze document relationships
        enhanced_metadata = analyze_document_relationships(doc_id, metadata, index)
        
        # Store each chunk
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_metadata = chunk['metadata']
            
            # Add relationship data to chunk metadata
            chunk_metadata['document_relationships'] = enhanced_metadata['relationships']
            chunk_metadata['knowledge_graph'] = enhanced_metadata['knowledge_graph']
            
            # Create embedding for the chunk
            chunk_embedding = create_embedding(chunk_text)
            if not chunk_embedding:
                print(f"❌ Failed to create embedding for chunk {chunk_metadata['chunk_info']['chunk_id']}")
                success = False
                continue
            
            # Upsert chunk to Pinecone
            upsert_to_pinecone(
                chunk_metadata['chunk_info']['chunk_id'],
                chunk_embedding,
                chunk_metadata
            )
        
        return success
    except Exception as e:
        print(f"❌ Error storing document {doc_id} in Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    main()
