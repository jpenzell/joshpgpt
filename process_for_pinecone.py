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
import re
import math
from embedding_utils import EmbeddingManager

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

# Initialize embedding manager
embedding_manager = EmbeddingManager()

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

def create_embedding(text, chunk_size=1000, overlap=100):
    """Create embeddings with enhanced quality control and versioning"""
    try:
        # Clean and normalize text
        text = text.replace('\n', ' ').strip()
        
        # Process chunks if text is too long
        if len(text) > chunk_size:
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end < len(text):
                    next_period = text.find('.', end - 50, end + 50)
                    if next_period != -1:
                        end = next_period + 1
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap
        else:
            chunks = [text]
        
        # Create embeddings for each chunk with quality control
        embeddings = []
        chunk_weights = []
        
        for chunk in chunks:
            content_hash = embedding_manager.compute_content_hash(chunk)
            model_name = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            
            # Check if we need to recompute the embedding
            if embedding_manager.should_recompute_embedding(
                content_hash, 
                model_name, 
                "1.0"  # Current version
            ):
                for attempt in range(MAX_RETRIES):
                    try:
                        response = client.embeddings.create(
                            model=model_name,
                            input=chunk,
                            dimensions=EMBEDDING_DIMENSIONS
                        )
                        
                        chunk_embedding = response.data[0].embedding
                        
                        # Validate embedding quality
                        is_valid, quality_metrics = embedding_manager.validate_embedding_quality(chunk_embedding)
                        
                        if not is_valid:
                            print(f"Warning: Low quality embedding detected: {quality_metrics}")
                            if attempt < MAX_RETRIES - 1:
                                continue
                        
                        # Create and store metadata
                        metadata = embedding_manager.create_embedding_metadata(
                            content=chunk,
                            model_name=model_name,
                            dimensions=EMBEDDING_DIMENSIONS,
                            content_type="text",
                            processing_params={"chunk_size": chunk_size, "overlap": overlap},
                            quality_metrics=quality_metrics
                        )
                        
                        embeddings.append(chunk_embedding)
                        chunk_weights.append(len(chunk))
                        break
                        
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            raise e
                        time.sleep(2 ** attempt)
        
        if not embeddings:
            return None
        
        # Combine multiple chunks if necessary
        if len(embeddings) > 1:
            return embedding_manager.combine_embeddings(embeddings, weights=chunk_weights)
        
        return embeddings[0]
    
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
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
        self.processed_documents = set()
        self.failed_documents = set()
        self.skipped_documents = set()
        self.current_batch = 0
        self.failure_reasons = {}
        self.retry_counts = {}
        self.last_processed_time = None
        self.load_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.processed_documents = set(data.get('processed_docs', []))
                    self.failed_documents = set(data.get('failed_docs', []))
                    self.skipped_documents = set(data.get('skipped_docs', []))
                    self.current_batch = data.get('current_batch', 0)
                    self.failure_reasons = data.get('failure_reasons', {})
                    self.retry_counts = data.get('retry_counts', {})
                    self.last_processed_time = data.get('last_processed_time')
        except Exception as e:
            print(f"Error loading state: {str(e)}")

    def save_state(self):
        try:
            data = {
                'processed_docs': list(self.processed_documents),
                'failed_docs': list(self.failed_documents),
                'skipped_docs': list(self.skipped_documents),
                'current_batch': self.current_batch,
                'failure_reasons': self.failure_reasons,
                'retry_counts': self.retry_counts,
                'last_processed_time': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving state: {str(e)}")

    def mark_processed(self, doc_id):
        self.processed_documents.add(doc_id)
        self.save_state()

    def is_processed(self, doc_id):
        return doc_id in self.processed_documents

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

def extract_enhanced_metadata(text, doc_id, doc_type, vendor, total, upload_date, categories):
    """Extract enhanced metadata from document text"""
    try:
        # Initialize base metadata (maintaining backward compatibility)
        metadata = {
            "document_id": doc_id,
            "type": doc_type,
            "vendor": vendor,
            "total": total,
            "uploaded_date": upload_date,
            "categories": categories,
            "text": text,
            "version": "2.0"  # Version tracking for future compatibility
        }

        # Get NLP model for entity extraction
        nlp = get_nlp()
        doc = nlp(text)

        # Enhanced entity extraction
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "money": [],
            "misc": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]:
                entities[ent.label_.lower() + "s"].append({
                    "text": ent.text,
                    "context": text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
                })

        # Document Intelligence
        metadata["cognitive"] = {
            "document_purpose": determine_document_purpose(text),
            "priority_level": calculate_priority(text, entities),
            "confidentiality": determine_confidentiality(text),
            "action_items": extract_action_items(text),
            "deadlines": extract_dates(text, filter_type="deadline")
        }

        # Content Analysis
        metadata["content"] = {
            "key_topics": extract_key_topics(text),
            "projects": extract_project_references(text),
            "decisions": extract_decisions(text),
            "next_steps": extract_next_steps(text),
            "stakeholders": identify_stakeholders(entities)
        }

        # Business Context
        metadata["business"] = {
            "business_unit": extract_business_unit(text),
            "cost_center": extract_cost_center(text),
            "budget_refs": extract_budget_references(text),
            "approval_chain": extract_approval_chain(text),
            "compliance_tags": identify_compliance_items(text)
        }

        # Temporal Context
        metadata["temporal"] = {
            "effective_date": extract_dates(text, filter_type="effective"),
            "expiration_date": extract_dates(text, filter_type="expiration"),
            "review_date": extract_dates(text, filter_type="review"),
            "related_events": extract_events(text),
            "timeline": extract_timeline_markers(text)
        }

        # Relationship Mapping
        metadata["relationships"] = {
            "preceding_docs": extract_document_references(text, "previous"),
            "related_docs": extract_document_references(text, "related"),
            "supersedes": extract_document_references(text, "supersedes"),
            "dependencies": extract_dependencies(text),
            "child_docs": extract_document_references(text, "child")
        }

        # Knowledge Graph Elements
        metadata["knowledge_graph"] = {
            "nodes": extract_knowledge_nodes(text),
            "edges": extract_relationships(text),
            "context_vectors": create_context_vectors(text)
        }

        # AI-Optimized Retrieval Tags
        metadata["retrieval_optimization"] = {
            "semantic_markers": generate_semantic_markers(text),
            "temporal_markers": generate_temporal_markers(upload_date, text),
            "entity_markers": generate_entity_markers(entities),
            "importance_score": calculate_importance(text, entities)
        }

        return metadata

    except Exception as e:
        print(f"Error extracting enhanced metadata: {str(e)}")
        # Return basic metadata if enhancement fails (maintaining backward compatibility)
        return {
            "document_id": doc_id,
            "type": doc_type,
            "vendor": vendor,
            "total": total,
            "uploaded_date": upload_date,
            "categories": categories,
            "text": text,
            "version": "1.0"
        }

def determine_document_purpose(text):
    """Determine the primary purpose of the document"""
    try:
        classifier = get_zero_shot_classifier()
        candidate_labels = [
            "meeting_notes", "contract", "correspondence", "report",
            "proposal", "invoice", "policy", "presentation",
            "agreement", "memo", "specification", "review"
        ]
        result = classifier(text, candidate_labels)
        return {
            "primary": result["labels"][0],
            "confidence": result["scores"][0],
            "secondary": result["labels"][1] if len(result["labels"]) > 1 else None
        }
    except Exception as e:
        print(f"Error determining document purpose: {str(e)}")
        return {"primary": "unknown", "confidence": 0.0, "secondary": None}

def calculate_priority(text, entities):
    """Calculate document priority based on content analysis"""
    try:
        priority_indicators = {
            "high": ["urgent", "immediate", "critical", "asap", "deadline"],
            "medium": ["important", "significant", "needed", "required"],
            "low": ["fyi", "information", "update", "routine"]
        }
        
        # Count priority indicators
        scores = {level: sum(1 for word in words if word.lower() in text.lower())
                 for level, words in priority_indicators.items()}
        
        # Consider entity presence
        if len(entities["people"]) > 3 or len(entities["organizations"]) > 2:
            scores["high"] += 1
            
        # Return highest scoring priority
        max_priority = max(scores.items(), key=lambda x: x[1])
        return {
            "level": max_priority[0],
            "score": max_priority[1],
            "indicators": [word for word in priority_indicators[max_priority[0]] 
                         if word.lower() in text.lower()]
        }
    except Exception as e:
        print(f"Error calculating priority: {str(e)}")
        return {"level": "medium", "score": 0, "indicators": []}

def process_single_document(doc, access_token, checkpoint):
    """Process a single document and upload to Pinecone"""
    try:
        doc_id = doc['id']
        print(f"\n📄 Processing document {doc_id}")
        
        # Download PDF
        print("📥 Downloading PDF...")
        pdf_data = download_pdf(doc, access_token)
        if not pdf_data:
            checkpoint.mark_failed(doc_id, "Download failed")
            return False
        
        # Save PDF locally
        local_file_path = f"documents_{doc_id}/document.pdf"
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(pdf_data)
        print("✅ PDF saved locally")
        
        # Extract text
        print("\n📝 Extracting text...")
        text = extract_text_from_pdf(doc, access_token)
        if not text:
            checkpoint.mark_failed(doc_id, "Text extraction failed")
            return False
        print(f"✅ Successfully extracted {len(text):,} characters")
        
        # Extract enhanced metadata
        print("\n🔍 Extracting metadata...")
        metadata = extract_enhanced_metadata(text, doc_id, doc['type'], doc['vendor'], doc['total'], doc['created'], doc['category'])
        print("✅ Metadata extracted")
        
        # Create embedding
        print("\n🧮 Creating embedding...")
        embedding = create_embedding(text)
        if not embedding:
            checkpoint.mark_failed(doc_id, "Embedding creation failed")
            return False
        print(f"✅ Created embedding vector of length {len(embedding)}")
        
        # Initialize Pinecone
        print("\n🔄 Connecting to Pinecone...")
        index = init_pinecone()
        if not index:
            checkpoint.mark_failed(doc_id, "Pinecone initialization failed")
            return False
        
        # Analyze relationships with other documents
        print("\n🔗 Analyzing document relationships...")
        relationships = analyze_document_relationships(doc_id, text, metadata, index)
        if relationships:
            metadata['relationships'] = relationships
            print(f"✅ Found relationships with {len(relationships.get('similar_documents', []))} documents")
        
        # Upload to Pinecone
        print("\n📤 Upserting to Pinecone...")
        try:
            index.upsert(vectors=[(str(doc_id), embedding, metadata)])
            print("✅ Successfully upserted to Pinecone")
        except Exception as e:
            print(f"❌ Failed to upsert to Pinecone: {str(e)}")
            checkpoint.mark_failed(doc_id, f"Pinecone upsert failed: {str(e)}")
            return False
        
        # Upload to S3
        print("\n☁️ Uploading to S3...")
        s3_url = upload_to_s3(pdf_data, doc_id, 'application/pdf')
        if not s3_url:
            checkpoint.mark_failed(doc_id, "S3 upload failed")
            return False
        print(f"✅ Successfully uploaded to S3: {s3_url}")
        
        # Update metadata with S3 URL and mark as processed
        try:
            metadata['s3_url'] = s3_url
            metadata['processing_status'] = 'completed'
            metadata['last_updated'] = datetime.now().isoformat()
            index.upsert(vectors=[(str(doc_id), embedding, metadata)])
            print("✅ Updated Pinecone with final metadata")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to update Pinecone with S3 URL: {str(e)}")
            return True  # Still consider it successful as the document is processed
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        checkpoint.mark_failed(doc_id, f"Processing failed: {str(e)}")
        return False

def extract_references(text):
    """Extract references to other documents from text"""
    reference_patterns = {
        "invoice": r"(?i)invoice\s*#?\s*([A-Z0-9-]+)",
        "order": r"(?i)order\s*#?\s*([A-Z0-9-]+)",
        "contract": r"(?i)contract\s*#?\s*([A-Z0-9-]+)",
        "case": r"(?i)case\s*#?\s*([A-Z0-9-]+)"
    }
    references = {}
    for ref_type, pattern in reference_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            references[ref_type] = matches
    return references

def identify_document_thread(doc_id, doc_type, upload_date):
    """Identify the document's thread based on type and temporal proximity"""
    return {
        "thread_id": f"thread_{doc_type}_{upload_date[:7]}",  # Group by month
        "position": "standalone",  # Will be updated when relationships are analyzed
        "thread_context": doc_type
    }

def extract_connected_entities(entities):
    """Extract connected entities from the entities dictionary"""
    connected = {}
    for entity_type, entity_list in entities.items():
        if entity_list:
            connected[entity_type] = [{"text": e["text"], "frequency": 1} for e in entity_list]
    return connected

def identify_broader_context(text, doc_type):
    """Identify the broader context this document belongs to"""
    contexts = {
        "receipt": ["purchase", "transaction", "expense"],
        "invoice": ["billing", "payment", "accounts_receivable"],
        "contract": ["agreement", "legal", "business_relationship"],
        "correspondence": ["communication", "customer_service", "inquiry"]
    }
    return contexts.get(doc_type, ["general"])

def extract_associated_ids(text):
    """Extract associated IDs from text"""
    id_patterns = {
        "transaction": r"(?i)transaction\s*#?\s*([A-Z0-9-]+)",
        "reference": r"(?i)ref\s*#?\s*([A-Z0-9-]+)",
        "account": r"(?i)account\s*#?\s*([A-Z0-9-]+)"
    }
    associated_ids = {}
    for id_type, pattern in id_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            associated_ids[id_type] = matches
    return associated_ids

def determine_document_purpose(text):
    """Determine the primary purpose of the document"""
    try:
        classifier = get_zero_shot_classifier()
        candidate_labels = [
            "meeting_notes", "contract", "correspondence", "report",
            "proposal", "invoice", "policy", "presentation",
            "agreement", "memo", "specification", "review"
        ]
        result = classifier(text, candidate_labels)
        return {
            "primary": result["labels"][0],
            "confidence": result["scores"][0],
            "secondary": result["labels"][1] if len(result["labels"]) > 1 else None
        }
    except Exception as e:
        print(f"Error determining document purpose: {str(e)}")
        return {"primary": "unknown", "confidence": 0.0, "secondary": None}

def determine_activity_type(text):
    """Determine the type of activity represented in the document"""
    activities = {
        "purchase": any(word in text.lower() for word in ["bought", "purchased", "order", "buy"]),
        "travel": any(word in text.lower() for word in ["flight", "hotel", "travel", "trip"]),
        "service": any(word in text.lower() for word in ["service", "consultation", "appointment"]),
        "subscription": any(word in text.lower() for word in ["subscription", "recurring", "monthly", "annual"])
    }
    return [k for k, v in activities.items() if v]

def determine_location_context(locations):
    """Determine the location context from extracted locations"""
    if not locations:
        return []
    return [loc["text"] for loc in locations[:3]]  # Return top 3 locations

def generate_time_contexts(upload_date, date_entities):
    """Generate time-based retrieval contexts"""
    contexts = []
    try:
        upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
        contexts.append(f"year_{upload_dt.year}")
        contexts.append(f"month_{upload_dt.year}_{upload_dt.month:02d}")
        contexts.append(f"quarter_{upload_dt.year}_Q{(upload_dt.month-1)//3 + 1}")
    except (ValueError, AttributeError):
        pass
    
    # Add extracted dates
    for date in date_entities:
        contexts.append(f"date_mentioned_{date['text']}")
    
    return contexts

def calculate_importance(text, entities):
    """Calculate document importance based on various factors"""
    importance_score = 0
    
    # Check for urgent indicators
    if any(word in text.lower() for word in ["urgent", "important", "priority", "asap"]):
        importance_score += 2
        
    # Check for monetary value
    if entities["money"]:
        importance_score += 1
        
    # Check for multiple involved parties
    if len(entities["people"]) + len(entities["organizations"]) > 2:
        importance_score += 1
        
    # Check for action items
    if any(word in text.lower() for word in ["required", "must", "deadline", "action"]):
        importance_score += 1
        
    return min(5, importance_score)  # Scale from 0-5

def calculate_completeness(text, doc_type):
    """Calculate completeness score based on document type requirements"""
    required_elements = {
        "receipt": ["total", "date", "vendor"],
        "invoice": ["total", "date", "vendor", "invoice number"],
        "contract": ["parties", "date", "terms", "signatures"],
        "correspondence": ["sender", "recipient", "date", "subject"]
    }
    
    elements = required_elements.get(doc_type, ["date", "content"])
    present_elements = sum(1 for element in elements if element.lower() in text.lower())
    total_elements = len(elements)
    
    return round(present_elements / total_elements, 2) if total_elements > 0 else 0.0

def has_verification_markers(text):
    """Check for verification markers in the document"""
    verification_indicators = [
        "verified", "approved", "confirmed", "authorized",
        "signature", "authenticated", "validated"
    ]
    return any(indicator in text.lower() for indicator in verification_indicators)

def extract_search_terms(text):
    """Extract primary search terms from the document"""
    # Get NLP model
    nlp = get_nlp()
    doc = nlp(text)
    
    # Extract important terms
    terms = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            terms.append(token.text)
    
    # Remove duplicates and limit
    return list(set(terms))[:10]

def determine_semantic_categories(doc_type, text):
    """Determine semantic categories for the document"""
    categories = set()
    
    # Add document type category
    categories.add(doc_type)
    
    # Check for business context
    if any(word in text.lower() for word in ["business", "corporate", "company", "enterprise"]):
        categories.add("business")
        
    # Check for personal context
    if any(word in text.lower() for word in ["personal", "individual", "private"]):
        categories.add("personal")
        
    # Check for financial context
    if any(word in text.lower() for word in ["payment", "invoice", "money", "cost"]):
        categories.add("financial")
        
    return list(categories)

def determine_temporal_markers(upload_date, text):
    """Determine temporal markers for the document"""
    markers = []
    try:
        upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
        markers.append(f"year_{upload_dt.year}")
        markers.append(f"month_{upload_dt.strftime('%B').lower()}")
        markers.append(f"quarter_Q{(upload_dt.month-1)//3 + 1}")
        
        # Add temporal context
        if "urgent" in text.lower() or "asap" in text.lower():
            markers.append("urgent")
        if "deadline" in text.lower():
            markers.append("deadline")
        if any(word in text.lower() for word in ["recurring", "monthly", "annual"]):
            markers.append("recurring")
            
    except (ValueError, AttributeError):
        pass
        
    return markers

def extract_key_topics(text):
    """Extract key topics using NLP and zero-shot classification"""
    try:
        nlp = get_nlp()
        doc = nlp(text)
        
        # Extract noun phrases
        key_phrases = [chunk.text for chunk in doc.noun_chunks 
                      if len(chunk.text.split()) >= 2][:15]  # Limit to 15 topics
        
        # Get topics using zero-shot classification
        classifier = get_zero_shot_classifier()
        candidate_topics = [
            "finance", "operations", "legal", "human_resources",
            "technology", "sales", "marketing", "strategy",
            "compliance", "research", "development", "customer_service"
        ]
        results = classifier(text, candidate_topics, multi_label=True)
        
        return {
            "key_phrases": key_phrases,
            "classified_topics": [
                {"topic": label, "confidence": score}
                for label, score in zip(results["labels"], results["scores"])
                if score > 0.3
            ]
        }
    except Exception as e:
        print(f"Error extracting key topics: {str(e)}")
        return {"key_phrases": [], "classified_topics": []}

def extract_action_items(text):
    """Extract action items and tasks from text"""
    try:
        action_patterns = [
            r"(?i)(?:need to|must|should|will|to-do|action item[s]?:).*?(?:\.|$)",
            r"(?i)(?:required|requested|pending|awaiting).*?(?:\.|$)",
            r"(?i)(?:follow[- ]up|follow[- ]through).*?(?:\.|$)"
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                action = match.group(0).strip()
                if len(action) > 10:  # Filter out very short matches
                    actions.append({
                        "action": action,
                        "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })
        
        return actions
    except Exception as e:
        print(f"Error extracting action items: {str(e)}")
        return []

def extract_dates(text, filter_type=None):
    """Extract dates with context based on type"""
    try:
        nlp = get_nlp()
        doc = nlp(text)
        
        date_patterns = {
            "deadline": r"(?i)(?:due|deadline|by|until|before)\s*(?:the\s*)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})",
            "effective": r"(?i)(?:effective|valid|starts?|begins?)\s*(?:from|on)?\s*(?:the\s*)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})",
            "expiration": r"(?i)(?:expires?|valid until|ends?|terminates?)\s*(?:on)?\s*(?:the\s*)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})",
            "review": r"(?i)(?:review|assess|evaluate)\s*(?:on|by)?\s*(?:the\s*)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})"
        }
        
        dates = []
        if filter_type and filter_type in date_patterns:
            matches = re.finditer(date_patterns[filter_type], text)
            for match in matches:
                dates.append({
                    "date": match.group(1),
                    "type": filter_type,
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        else:
            # Extract all date entities
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    dates.append({
                        "date": ent.text,
                        "type": "general",
                        "context": text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
                    })
        
        return dates
    except Exception as e:
        print(f"Error extracting dates: {str(e)}")
        return []

def extract_document_references(text, ref_type="related"):
    """Extract references to other documents"""
    try:
        reference_patterns = {
            "previous": r"(?i)(?:previous|prior|earlier|preceding)\s+(?:document|agreement|contract|version|revision)\s+(?:number|#|ref)?[\s:]*([\w-]+)",
            "related": r"(?i)(?:related|associated|linked|referenced)\s+(?:document|agreement|contract)\s+(?:number|#|ref)?[\s:]*([\w-]+)",
            "supersedes": r"(?i)(?:supersedes|replaces|updates)\s+(?:document|agreement|contract)\s+(?:number|#|ref)?[\s:]*([\w-]+)",
            "child": r"(?i)(?:attachment|appendix|exhibit|annex)\s+(?:[A-Z]|[0-9]+|#)*[\s:]*([\w-]+)"
        }
        
        if ref_type not in reference_patterns:
            return []
            
        references = []
        matches = re.finditer(reference_patterns[ref_type], text)
        for match in matches:
            references.append({
                "reference_id": match.group(1),
                "type": ref_type,
                "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
            
        return references
    except Exception as e:
        print(f"Error extracting document references: {str(e)}")
        return []

def generate_semantic_markers(text):
    """Generate semantic markers for improved retrieval"""
    try:
        # Get key phrases using NLP
        nlp = get_nlp()
        doc = nlp(text)
        
        # Extract important phrases
        phrases = [chunk.text for chunk in doc.noun_chunks 
                  if len(chunk.text.split()) >= 2][:10]
        
        # Generate markers
        markers = {
            "key_phrases": phrases,
            "document_length": len(text),
            "complexity_score": calculate_complexity(text),
            "formality_level": determine_formality(text),
            "sentiment": analyze_sentiment(text)
        }
        
        return markers
    except Exception as e:
        print(f"Error generating semantic markers: {str(e)}")
        return {}

def determine_confidentiality(text):
    """Determine document confidentiality level"""
    try:
        confidential_patterns = {
            "restricted": r"(?i)(restricted|confidential|private|sensitive)",
            "internal": r"(?i)(internal use|internal only|company use|not for distribution)",
            "public": r"(?i)(public|unclassified|unrestricted)"
        }
        
        levels = {}
        for level, pattern in confidential_patterns.items():
            matches = re.finditer(pattern, text)
            contexts = []
            for match in matches:
                contexts.append(text[max(0, match.start()-50):min(len(text), match.end()+50)])
            if contexts:
                levels[level] = contexts
        
        if not levels:
            return {"level": "unspecified", "confidence": 0.0}
            
        # Return highest restriction level found
        if "restricted" in levels:
            return {"level": "restricted", "confidence": 1.0, "contexts": levels["restricted"]}
        elif "internal" in levels:
            return {"level": "internal", "confidence": 0.8, "contexts": levels["internal"]}
        else:
            return {"level": "public", "confidence": 0.6, "contexts": levels.get("public", [])}
            
    except Exception as e:
        print(f"Error determining confidentiality: {str(e)}")
        return {"level": "unspecified", "confidence": 0.0}

def extract_business_unit(text):
    """Extract business unit references"""
    try:
        # Common business unit patterns
        unit_patterns = [
            r"(?i)(?:department|dept|division|unit|team)[\s:]+([A-Za-z\s&]+?)(?:\.|,|\n|$)",
            r"(?i)(?:from|to)[\s:]+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(?:department|dept|division|unit|team)"
        ]
        
        units = []
        for pattern in unit_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                unit = match.group(1).strip()
                if len(unit) > 2:  # Filter out very short matches
                    units.append({
                        "unit": unit,
                        "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })
        
        return units
    except Exception as e:
        print(f"Error extracting business unit: {str(e)}")
        return []

def extract_cost_center(text):
    """Extract cost center information"""
    try:
        # Cost center patterns
        patterns = [
            r"(?i)cost\s+center[\s#:]+([A-Z0-9-]+)",
            r"(?i)cc[\s#:]+([A-Z0-9-]+)",
            r"(?i)center\s+code[\s#:]+([A-Z0-9-]+)"
        ]
        
        centers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                centers.append({
                    "code": match.group(1),
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        
        return centers
    except Exception as e:
        print(f"Error extracting cost center: {str(e)}")
        return []

def identify_compliance_items(text):
    """Identify compliance-related items"""
    try:
        compliance_patterns = {
            "regulatory": r"(?i)(regulation|compliance|regulatory|requirement)s?",
            "legal": r"(?i)(law|statute|legal|legislation)",
            "policy": r"(?i)(policy|procedure|guideline|standard)",
            "certification": r"(?i)(certification|certified|accredited|iso)"
        }
        
        items = {}
        for category, pattern in compliance_patterns.items():
            matches = re.finditer(pattern, text)
            contexts = []
            for match in matches:
                contexts.append({
                    "text": match.group(0),
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
            if contexts:
                items[category] = contexts
        
        return items
    except Exception as e:
        print(f"Error identifying compliance items: {str(e)}")
        return {}

def extract_timeline_markers(text):
    """Extract timeline markers and milestones"""
    try:
        timeline_patterns = {
            "milestone": r"(?i)milestone[\s#:]+([^\n.]+)",
            "phase": r"(?i)phase[\s#:]+([^\n.]+)",
            "stage": r"(?i)stage[\s#:]+([^\n.]+)",
            "deadline": r"(?i)deadline[\s:]+([^\n.]+)"
        }
        
        markers = {}
        for marker_type, pattern in timeline_patterns.items():
            matches = re.finditer(pattern, text)
            items = []
            for match in matches:
                items.append({
                    "text": match.group(1).strip(),
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
            if items:
                markers[marker_type] = items
        
        return markers
    except Exception as e:
        print(f"Error extracting timeline markers: {str(e)}")
        return {}

def calculate_complexity(text):
    """Calculate document complexity score"""
    try:
        # Simple complexity metrics
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        metrics = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }
        
        # Calculate complexity score (0-1)
        complexity = min(1.0, (
            (metrics["avg_word_length"] / 10) * 0.3 +
            (min(metrics["avg_sentence_length"], 40) / 40) * 0.3 +
            (min(metrics["word_count"], 1000) / 1000) * 0.4
        ))
        
        return {
            "score": complexity,
            "metrics": metrics
        }
    except Exception as e:
        print(f"Error calculating complexity: {str(e)}")
        return {"score": 0.0, "metrics": {}}

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
    processed_count = len(state.processed_documents)
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
    documents_to_process = [doc for doc in documents if doc['id'] not in state.processed_documents]
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
    
    final_processed = len(state.processed_documents)
    print(f"\nProcessing Summary:")
    print(f"Total documents: {total}")
    print(f"Successfully processed: {final_processed}")
    print(f"Remaining: {total - final_processed}")
    print(f"Current status: {state.status}")
    
if __name__ == "__main__":
    main() 