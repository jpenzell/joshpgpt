import json
import os
import pinecone
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
import openai
import time
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import signal
import sys
import io
import base64
import traceback
import requests
from pdf2image import convert_from_bytes
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """Retry a function with exponential backoff"""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if retry < max_retries - 1:
                    print(f"Attempt {retry + 1} failed: {str(e)}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
        
        print(f"All {max_retries} attempts failed. Last error: {str(last_exception)}")
        raise last_exception
    
    return wrapper

@retry_with_backoff
def create_embedding(text):
    """Create embedding for text using OpenAI with retry logic"""
    try:
        response = client.embeddings.create(
            model=os.getenv('OPENAI_EMBEDDING_MODEL'),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        raise

@retry_with_backoff
def extract_text_with_vision(image_base64):
    """Extract text using GPT-4o Vision with retry logic"""
    try:
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
                                "url": f"data:image/jpeg;base64,{image_base64}",
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
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in GPT-4o Vision API call: {str(e)}")
        raise

def download_document(doc, access_token):
    """Download document from Shoeboxed with improved error handling"""
    try:
        # Get attachment URL from document metadata
        attachment = doc.get('attachment', {})
        if not attachment or not attachment.get('url'):
            print(f"No attachment URL found for document {doc.get('id', 'Unknown')}")
            return None
            
        url = attachment['url']
        print(f"📥 Downloading from URL: {url}")
        
        # Make request with proper headers
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/pdf,image/*',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Origin': 'https://api.shoeboxed.com',
            'Referer': 'https://api.shoeboxed.com/'
        }
        
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            stream=True,
            allow_redirects=True
        )
        
        print(f"📥 Download response status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.content
            if not content:
                print("❌ Downloaded empty content")
                return None
                
            print(f"✅ Successfully downloaded {len(content):,} bytes")
            return content
            
        print(f"❌ Download failed. Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        return None
        
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def extract_text_from_pdf(doc, access_token):
    """Extract text from PDF using GPT-4o Vision with improved error handling"""
    try:
        # Get document attachment
        attachment = doc.get('attachment', {})
        if not attachment:
            print(f"No attachment found for document {doc['id']}")
            return None
        
        # Download PDF with retry
        @retry_with_backoff
        def download_pdf():
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/pdf,image/*',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Origin': 'https://api.shoeboxed.com',
                'Referer': 'https://api.shoeboxed.com/'
            }
            response = requests.get(attachment['url'], headers=headers)
            response.raise_for_status()
            return response.content
            
        try:
            pdf_content = download_pdf()
            if not pdf_content:
                print(f"❌ No content downloaded for document {doc['id']}")
                return None
                
            # Convert PDF to images
            pdf_bytes = BytesIO(pdf_content)
            images = convert_from_bytes(pdf_bytes.read())
            
            if not images:
                print(f"❌ No images extracted from PDF for document {doc['id']}")
                return None
                
            # Process each page/image
            extracted_text = []
            for i, image in enumerate(images):
                try:
                    # Convert image to base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG", quality=95)
                    image_bytes = buffered.getvalue()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Extract text using GPT-4 Vision with retry
                    page_text = extract_text_with_vision(image_base64)
                    if page_text:
                        extracted_text.append(f"Page {i + 1}:\n{page_text}")
                    else:
                        print(f"⚠️ No text extracted from page {i + 1}")
                        
                except Exception as e:
                    print(f"❌ Error processing page {i + 1}: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
            
            if not extracted_text:
                print(f"❌ No text extracted from any page for document {doc['id']}")
                return None
                
            return "\n\n".join(extracted_text)
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        print(f"❌ Error in extract_text_from_pdf: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

class ProcessingState:
    """Class to manage processing state"""
    def __init__(self):
        self.progress_file = 'processing_progress.json'
        self.processed_docs = set()
    
    def load_progress(self):
        """Load progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_docs = set(data.get('processed_docs', []))
        except Exception as e:
            print(f"Error loading progress: {str(e)}")
    
    def save_progress(self):
        """Save progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'processed_docs': list(self.processed_docs)
                }, f)
        except Exception as e:
            print(f"Error saving progress: {str(e)}")
    
    def mark_processed(self, doc_id):
        """Mark document as processed"""
        self.processed_docs.add(doc_id)
    
    def is_processed(self, doc_id):
        """Check if document is already processed"""
        return doc_id in self.processed_docs

def init_pinecone():
    """Initialize Pinecone with proper configuration"""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pc.Index(
        name=os.getenv('PINECONE_INDEX_NAME'),
        host=os.getenv('PINECONE_INDEX_HOST')
    )

def upload_to_s3(file_content, doc_id, content_type):
    """Upload file to S3"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Generate S3 key
        file_extension = 'pdf' if content_type == 'application/pdf' else 'jpg'
        s3_key = f"documents/{doc_id}.{file_extension}"
        
        # Upload file
        s3_client.put_object(
            Bucket=os.getenv('S3_BUCKET_NAME'),
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        
        return f"s3://{os.getenv('S3_BUCKET_NAME')}/{s3_key}"
    
    except ClientError as e:
        print(f"Error uploading to S3: {str(e)}")
        return None

def process_single_document(doc, access_token, checkpoint):
    """Process a single document and upload to Pinecone"""
    doc_id = doc['id']
    
    print(f"\n{'='*80}")
    print(f"🔍 Processing document: {doc_id}")
    print(f"Document Details:")
    print(f"  - Type: {doc.get('type', 'Unknown')}")
    print(f"  - Created: {doc.get('created', 'N/A')}")
    print(f"  - Modified: {doc.get('modified', 'N/A')}")
    print(f"  - Category: {doc.get('category', 'Uncategorized')}")
    print(f"  - Total: ${doc.get('total', 0.0)}")
    print(f"  - Processing State: {doc.get('processingState', 'Unknown')}")
    print(f"{'='*80}\n")
    
    try:
        # Download document
        print(f"📥 Downloading document...")
        pdf_data = download_document(doc, access_token)
        if not pdf_data:
            print("❌ Failed to download document")
            checkpoint.mark_failed(doc_id, "Download failed")
            return False
        print(f"✅ Successfully downloaded {len(pdf_data):,} bytes")
        
        # Extract text
        print("\n🔤 Extracting text with GPT-4o Vision...")
        text = extract_text_from_pdf(doc, access_token)
        if not text:
            print("❌ Failed to extract text")
            checkpoint.mark_failed(doc_id, "Text extraction failed")
            return False
        print(f"✅ Successfully extracted {len(text):,} characters")
        print(f"📝 Sample text: {text[:200]}...")
        
        # Create embedding
        print("\n🧮 Creating vector embedding...")
        embedding = create_embedding(text)
        if not embedding:
            print("❌ Failed to create embedding")
            checkpoint.mark_failed(doc_id, "Embedding creation failed")
            return False
        print(f"✅ Created embedding vector of length {len(embedding)}")
        
        # Prepare metadata
        print("\n📋 Preparing metadata...")
        metadata = {
            'id': str(doc_id),
            'type': doc.get('type', 'unknown'),
            'created': str(doc.get('created', '')),
            'modified': str(doc.get('modified', '')),
            'category': str(doc.get('category', 'Uncategorized')),
            'total': float(doc.get('total', 0.0)) if doc.get('total') is not None else 0.0,
            'text': text[:3000],  # Store first 3000 chars for context
            'processed_date': datetime.now().isoformat()
        }
        print("✅ Metadata prepared")
        
        # Initialize Pinecone
        print("\n🔄 Connecting to Pinecone...")
        index = init_pinecone()
        if not index:
            print("❌ Failed to initialize Pinecone")
            checkpoint.mark_failed(doc_id, "Pinecone initialization failed")
            return False
        
        # Upsert to Pinecone
        print("📤 Upserting to Pinecone...")
        try:
            index.upsert(vectors=[(str(doc_id), embedding, metadata)])
            print("✅ Successfully upserted to Pinecone")
        except Exception as e:
            print(f"❌ Failed to upsert to Pinecone: {str(e)}")
            checkpoint.mark_failed(doc_id, f"Pinecone upsert failed: {str(e)}")
            return False
        
        # Mark as processed
        checkpoint.mark_processed(doc_id)
        print(f"\n✅ Document {doc_id} successfully processed!")
        print(f"{'='*80}")
        return True
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        print(f"\n❌ {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        checkpoint.mark_failed(doc_id, error_msg)
        return False

def main():
    global should_continue
    
    # Get API keys from environment
    if not os.getenv('PINECONE_API_KEY') or not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Initialize processing state
    state = ProcessingState()
    
    # Try to load from document cache first
    cache_file = 'document_list_cache.json'
    if os.path.exists(cache_file):
        print("\nLoading documents from cache...")
        with open(cache_file, 'r') as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} documents from cache")
    else:
        # Find the most recent documents directory
        base_dirs = [d for d in os.listdir('.') if d.startswith('documents_')]
        if not base_dirs:
            raise ValueError("No documents directory found. Please run fetch_documents.py first")
        
        most_recent_dir = max(base_dirs)
        print(f"Processing directory: {most_recent_dir}")
        
        # Load documents metadata
        metadata_file = os.path.join(most_recent_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            documents = json.load(f)
    
    # Process documents one at a time
    total = len(documents)
    processed_count = len(state.processed_docs)
    remaining = total - processed_count
    
    print(f"\nDocument Processing Status:")
    print(f"Total documents: {total}")
    print(f"Already processed: {processed_count}")
    print(f"Remaining to process: {remaining}")
    
    if remaining == 0:
        print("\nAll documents have been processed!")
        return
    
    if state.status == 'paused':
        print("\nResuming from previous pause point...")
    else:
        state.set_status('processing')
    
    # Create a list of documents to process (excluding already processed ones)
    documents_to_process = [doc for doc in documents if doc['id'] not in state.processed_docs]
    print(f"\nFiltered down to {len(documents_to_process)} documents to process")
    
    with tqdm(total=total, initial=processed_count, desc="Processing documents") as pbar:
        for i, doc in enumerate(documents_to_process):
            if not should_continue:
                print("\nPausing processing...")
                state.set_status('paused')
                print(f"You can resume later from document {processed_count + i} of {total}")
                break
            
            print(f"\nProcessing document {i+1} of {len(documents_to_process)} (ID: {doc['id']})")
            if process_single_document(doc, os.getenv('SHOEBOXED_ACCESS_TOKEN'), state):
                pbar.update(1)
            
            time.sleep(1)  # Rate limiting between documents
    
    if should_continue:
        state.set_status('completed')
        print("\nProcessing complete!")
    
    final_processed = len(state.processed_docs)
    print(f"\nProcessing Summary:")
    print(f"Total documents: {total}")
    print(f"Successfully processed: {final_processed}")
    print(f"Remaining: {total - final_processed}")
    print(f"Current status: {state.status}")
    
if __name__ == "__main__":
    main() 