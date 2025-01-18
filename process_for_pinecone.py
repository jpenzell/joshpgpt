import json
import os
import pinecone
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
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
import concurrent.futures
from typing import List, Dict
import numpy as np
import multiprocessing
import psutil
import spacy
from transformers import pipeline
import textwrap

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants from environment
EMBEDDING_DIMENSIONS = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
MAX_MEMORY_PERCENT = int(os.getenv('MAX_MEMORY_PERCENT', '85'))

# Dynamic resource allocation
CPU_COUNT = multiprocessing.cpu_count()
MEMORY_GB = psutil.virtual_memory().total / (1024 ** 3)

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, 'embedding_cache.json')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize embedding cache
embedding_cache = {}
if os.path.exists(EMBEDDING_CACHE_FILE):
    try:
        with open(EMBEDDING_CACHE_FILE, 'r') as f:
            embedding_cache = json.load(f)
    except Exception as e:
        print(f"Error loading embedding cache: {str(e)}")

# Lazy loading of models
_nlp = None
_zero_shot_classifier = None
_summarizer = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(os.getenv('SPACY_MODEL', 'en_core_web_lg'))
        except Exception as e:
            print(f"Error loading spaCy model: {str(e)}")
            raise
    return _nlp

def get_zero_shot_classifier():
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        try:
            _zero_shot_classifier = pipeline("zero-shot-classification", 
                                          model=os.getenv('ZERO_SHOT_MODEL', 'facebook/bart-large-mnli'))
        except Exception as e:
            print(f"Error loading zero-shot classifier: {str(e)}")
            raise
    return _zero_shot_classifier

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            _summarizer = pipeline("summarization", 
                                 model=os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn'))
        except Exception as e:
            print(f"Error loading summarizer: {str(e)}")
            raise
    return _summarizer

# Initialize should_continue flag
should_continue = True

def save_embedding_cache():
    """Save embedding cache to disk"""
    try:
        with open(EMBEDDING_CACHE_FILE, 'w') as f:
            json.dump(embedding_cache, f)
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

def check_memory_usage():
    """Check if memory usage is within limits"""
    memory_percent = psutil.virtual_memory().percent
    return memory_percent < MAX_MEMORY_PERCENT

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_dir = os.path.join(os.path.dirname(__file__), 'documents')
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file}: {str(e)}")

def process_document_batch(documents: List[Dict]) -> List[Dict]:
    """Process a batch of documents in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_document, documents))
    return [r for r in results if r is not None]

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a batch of texts using the OpenAI API with caching"""
    try:
        # Check memory usage before processing
        if not check_memory_usage():
            print("⚠️ High memory usage detected, waiting for cleanup...")
            cleanup_temp_files()
            time.sleep(5)  # Wait for memory to be freed
        
        # Initialize results list
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in embedding_cache:
                results.append(embedding_cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If there are uncached texts, get their embeddings
        if uncached_texts:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=uncached_texts,
                dimensions=EMBEDDING_DIMENSIONS  # Use optimized dimensions
            )
            
            # Process new embeddings
            new_embeddings = [embedding.embedding for embedding in response.data]
            
            # Update cache and results
            for i, embedding in zip(uncached_indices, new_embeddings):
                text_hash = hash(texts[i])
                embedding_cache[text_hash] = embedding
                results.insert(i, embedding)
            
            # Save updated cache
            save_embedding_cache()
        
        return results
    except Exception as e:
        print(f"Error creating embeddings batch: {str(e)}")
        return []

def process_documents_with_batching(documents: List[Dict]) -> None:
    """Process documents in batches with parallel processing and batch embeddings"""
    total_docs = len(documents)
    successful_docs = 0
    failed_docs = 0
    skipped_docs = 0
    
    try:
        for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Processing document batches"):
            # Check memory usage before each batch
            if not check_memory_usage():
                print("\n⚠️ High memory usage detected, cleaning up...")
                cleanup_temp_files()
                time.sleep(5)
            
            batch = documents[i:i + BATCH_SIZE]
            batch_size = len(batch)
            
            print(f"\n📦 Processing batch {i//BATCH_SIZE + 1} of {(total_docs + BATCH_SIZE - 1)//BATCH_SIZE}")
            print(f"   Documents {i+1} to {min(i+batch_size, total_docs)} of {total_docs}")
            
            try:
                # Process documents in parallel
                processed_docs = process_document_batch(batch)
                
                # Collect texts for batch embedding
                texts_to_embed = []
                doc_metadata = []
                
                for doc in processed_docs:
                    if doc and 'text' in doc:
                        texts_to_embed.append(doc['text'])
                        doc_metadata.append({
                            'id': doc['id'],
                            'metadata': {
                                'title': doc.get('title', ''),
                                'date': doc.get('date', ''),
                                'category': doc.get('category', ''),
                                'source': doc.get('source', ''),
                                'processed_date': datetime.now().isoformat()
                            }
                        })
                
                # Create embeddings in optimized batches
                for j in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
                    batch_texts = texts_to_embed[j:j + EMBEDDING_BATCH_SIZE]
                    batch_metadata = doc_metadata[j:j + EMBEDDING_BATCH_SIZE]
                    
                    embeddings = create_embeddings_batch(batch_texts)
                    
                    # Prepare vectors for Pinecone
                    vectors = []
                    for embedding, meta in zip(embeddings, batch_metadata):
                        vectors.append({
                            'id': meta['id'],
                            'values': embedding,
                            'metadata': meta['metadata']
                        })
                    
                    # Upsert to Pinecone with retry
                    if vectors:
                        success = False
                        for attempt in range(3):
                            try:
                                index = Pinecone(api_key=os.getenv('PINECONE_API_KEY')).Index(os.getenv('PINECONE_INDEX_NAME'))
                                index.upsert(vectors=vectors)
                                success = True
                                break
                            except Exception as e:
                                print(f"Error upserting to Pinecone (Attempt {attempt + 1}): {str(e)}")
                                if attempt < 2:
                                    time.sleep(2 ** attempt)
                        
                        if success:
                            successful_docs += len(vectors)
                        else:
                            failed_docs += len(vectors)
                
                # Cleanup after each batch
                cleanup_temp_files()
                
            except Exception as batch_error:
                print(f"\n❌ Error processing batch: {str(batch_error)}")
                failed_docs += batch_size
                continue
            
            # Print progress statistics
            print(f"\n📊 Progress Statistics:")
            print(f"   Successful: {successful_docs}")
            print(f"   Failed: {failed_docs}")
            print(f"   Skipped: {skipped_docs}")
            print(f"   Remaining: {total_docs - (successful_docs + failed_docs + skipped_docs)}")
            
    except Exception as e:
        print(f"\n❌ Fatal error in batch processing: {str(e)}")
        print(traceback.format_exc())
    
    finally:
        # Final cleanup
        cleanup_temp_files()
        
        # Print final statistics
        print(f"\n🏁 Processing Complete!")
        print(f"   Total Documents: {total_docs}")
        print(f"   Successfully Processed: {successful_docs}")
        print(f"   Failed: {failed_docs}")
        print(f"   Skipped: {skipped_docs}")
        print(f"   Success Rate: {(successful_docs/total_docs)*100:.2f}%")

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

def create_embedding(text):
    """Create embedding with retry logic"""
    max_retries = int(os.getenv('MAX_RETRIES', '3'))
    
    for attempt in range(max_retries):
        try:
            print(f"Creating embedding (attempt {attempt + 1}/{max_retries})")
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text,
                dimensions=EMBEDDING_DIMENSIONS
            )
            
            embedding = response.data[0].embedding
            
            # Verify embedding dimension
            if len(embedding) != EMBEDDING_DIMENSIONS:
                print(f"Warning: Unexpected embedding dimension {len(embedding)}, expected {EMBEDDING_DIMENSIONS}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

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
        self.failed_docs = {}
        self.skipped_docs = set()
        self.status = 'initialized'
    
    def load_progress(self):
        """Load progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_docs = set(data.get('processed_docs', []))
                    self.failed_docs = data.get('failed_docs', {})
                    self.skipped_docs = set(data.get('skipped_docs', []))
                    self.status = data.get('status', 'initialized')
        except Exception as e:
            print(f"Error loading progress: {str(e)}")
    
    def save_progress(self):
        """Save progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'processed_docs': list(self.processed_docs),
                    'failed_docs': self.failed_docs,
                    'skipped_docs': list(self.skipped_docs),
                    'status': self.status
                }, f)
        except Exception as e:
            print(f"Error saving progress: {str(e)}")
    
    def mark_processed(self, doc_id):
        """Mark document as processed"""
        self.processed_docs.add(doc_id)
        if doc_id in self.failed_docs:
            del self.failed_docs[doc_id]
        self.save_progress()
    
    def mark_failed(self, doc_id, reason):
        """Mark document as failed with reason"""
        self.failed_docs[doc_id] = {
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.save_progress()
    
    def mark_skipped(self, doc_id):
        """Mark document as skipped"""
        self.skipped_docs.add(doc_id)
        self.save_progress()
    
    def is_processed(self, doc_id):
        """Check if document is already processed"""
        return doc_id in self.processed_docs
    
    def set_status(self, status):
        """Set processing status"""
        self.status = status
        self.save_progress()

def init_pinecone():
    """Initialize Pinecone with proper configuration"""
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(
            name=os.getenv('PINECONE_INDEX_NAME'),
            host=os.getenv('PINECONE_INDEX_HOST')
        )
        # Test connection
        index.describe_index_stats()
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

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

def extract_enhanced_metadata(text, doc):
    """Extract enhanced metadata from document text"""
    try:
        # Process with spaCy
        nlp = get_nlp()
        doc_nlp = nlp(text)
        
        # Extract entities
        entities = {
            'people': [ent.text for ent in doc_nlp.ents if ent.label_ == 'PERSON'],
            'organizations': [ent.text for ent in doc_nlp.ents if ent.label_ == 'ORG'],
            'locations': [ent.text for ent in doc_nlp.ents if ent.label_ == 'GPE'],
            'dates': [ent.text for ent in doc_nlp.ents if ent.label_ == 'DATE']
        }
        
        # Document classification
        doc_types = ["essay", "note", "receipt", "tax_document", "correspondence", "medical_record"]
        zero_shot_classifier = get_zero_shot_classifier()
        classification = zero_shot_classifier(text, doc_types)
        
        # Generate summary if text is long enough
        summary = ""
        if len(text.split()) > 50:
            summarizer = get_summarizer()
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        
        # Extract topics using basic keyword extraction
        keywords = [token.text for token in doc_nlp if token.is_alpha and not token.is_stop]
        topics = list(set([word.lower() for word in keywords]))[:10]
        
        return {
            'entities': entities,
            'doc_type': classification['labels'][0],
            'doc_type_confidence': classification['scores'][0],
            'summary': summary,
            'topics': topics,
            'source': doc.get('source', 'unknown'),
            'processed_date': datetime.now().isoformat(),
            'original_date': doc.get('created', 'unknown'),
            'language': doc_nlp.lang_,
            'word_count': len(text.split())
        }
    except Exception as e:
        print(f"Error extracting enhanced metadata: {str(e)}")
        return {
            'entities': {},
            'doc_type': 'unknown',
            'doc_type_confidence': 0.0,
            'summary': '',
            'topics': [],
            'source': doc.get('source', 'unknown'),
            'processed_date': datetime.now().isoformat(),
            'original_date': doc.get('created', 'unknown'),
            'language': 'en',
            'word_count': len(text.split())
        }

def process_single_document(doc, access_token, checkpoint):
    """Process a single document and upload to Pinecone"""
    doc_id = doc['id']
    temp_files = []  # Track files to cleanup
    
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
        # 1. Download document
        print(f"📥 Downloading document...")
        pdf_data = download_document(doc, access_token)
        if not pdf_data:
            print("❌ Failed to download document")
            checkpoint.mark_failed(doc_id, "Download failed")
            return False
            
        # Save temporarily
        os.makedirs('documents', exist_ok=True)
        temp_pdf_path = os.path.join('documents', f"{doc_id}.pdf")
        temp_files.append(temp_pdf_path)  # Track for cleanup
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_data)
        print(f"✅ Successfully downloaded {len(pdf_data):,} bytes")
        print(f"💾 Saved temporarily to: {temp_pdf_path}")
        
        # 2. Extract text
        print("\n🔤 Extracting text with GPT-4o Vision...")
        text = extract_text_from_pdf(doc, access_token)
        if not text:
            print("❌ Failed to extract text")
            checkpoint.mark_failed(doc_id, "Text extraction failed")
            return False
        print(f"✅ Successfully extracted {len(text):,} characters")
        print(f"📝 Sample text: {text[:200]}...")
        
        # 3. Create embedding
        print("\n🧮 Creating vector embedding...")
        embedding = create_embedding(text)
        if not embedding:
            print("❌ Failed to create embedding")
            checkpoint.mark_failed(doc_id, "Embedding creation failed")
            return False
        print(f"✅ Created embedding vector of length {len(embedding)}")
        
        # 4. Prepare metadata and upload to Pinecone
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
        
        # Upload to Pinecone
        print("📤 Upserting to Pinecone...")
        try:
            index.upsert(vectors=[(str(doc_id), embedding, metadata)])
            print("✅ Successfully upserted to Pinecone")
        except Exception as e:
            print(f"❌ Failed to upsert to Pinecone: {str(e)}")
            checkpoint.mark_failed(doc_id, f"Pinecone upsert failed: {str(e)}")
            return False
            
        # 5. Upload to S3
        print("\n☁️ Uploading to S3...")
        s3_url = upload_to_s3(pdf_data, doc_id, 'application/pdf')
        if not s3_url:
            print("❌ Failed to upload to S3")
            checkpoint.mark_failed(doc_id, "S3 upload failed")
            return False
        print(f"✅ Successfully uploaded to S3: {s3_url}")
        
        # Update metadata with S3 URL
        try:
            metadata['s3_url'] = s3_url
            index.upsert(vectors=[(str(doc_id), embedding, metadata)])
            print("✅ Updated Pinecone with S3 URL")
        except Exception as e:
            print(f"⚠️ Warning: Failed to update Pinecone with S3 URL: {str(e)}")
            # Don't fail the process for this
        
        # 6. Cleanup temporary files
        print("\n🧹 Cleaning up temporary files...")
        cleanup_success = True
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"✅ Deleted: {temp_file}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to delete {temp_file}: {str(e)}")
                cleanup_success = False
        
        if cleanup_success:
            print("✅ All temporary files cleaned up")
        else:
            print("⚠️ Some files could not be deleted")
        
        # 7. Mark as processed and save state
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
        
    finally:
        # Ensure cleanup happens even if there's an error
        print("\n🧹 Final cleanup...")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"✅ Deleted: {temp_file}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete {temp_file}: {str(e)}")

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