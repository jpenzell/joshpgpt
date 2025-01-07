import requests
import json
import os
from datetime import datetime
import time
import traceback
from pathlib import Path
import concurrent.futures
from typing import List, Dict
from tqdm import tqdm

# Constants
BATCH_SIZE = 100  # Number of documents to fetch per batch
MAX_WORKERS = 4   # Number of concurrent downloads
RATE_LIMIT_DELAY = 0.1  # Delay between API calls to avoid rate limiting

# Constants for file paths
DATA_DIR = Path('data')
TOKENS_FILE = DATA_DIR / '.auth_success'
PROGRESS_FILE = DATA_DIR / 'fetch_progress.json'
DOCUMENTS_DIR = DATA_DIR / 'documents'
IMAGES_DIR = DATA_DIR / 'images'
CACHE_FILE = DATA_DIR / 'document_id_cache.json'

def ensure_directories():
    """Ensure all required directories exist"""
    for directory in [DATA_DIR, DOCUMENTS_DIR, IMAGES_DIR]:
        directory.mkdir(exist_ok=True)

def load_document_cache():
    """Load the document ID cache"""
    if CACHE_FILE.exists():
        with CACHE_FILE.open('r') as f:
            return json.load(f)
    return {}

def save_document_cache(cache):
    """Save the document ID cache"""
    with CACHE_FILE.open('w') as f:
        json.dump(cache, f)

def fetch_documents_batch(offset: int, limit: int, access_token: str) -> List[Dict]:
    """Fetch a batch of documents from the API"""
    url = f"https://api.shoeboxed.com/v2/documents"
    headers = {
        "Authorization": f"Bearer {access_token}",
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    params = {
        "limit": limit,
        "offset": offset
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching documents batch at offset {offset}: {str(e)}")
        return []

def download_document_batch(documents: List[Dict], access_token: str, doc_cache: Dict) -> None:
    """Download a batch of documents and their attachments in parallel"""
    def download_single_document(doc):
        try:
            doc_id = doc['id']
            if doc_id in doc_cache:
                return None  # Skip if already downloaded
                
            attachments = doc.get('attachments', [])
            if not attachments:
                return None
                
            doc_dir = DOCUMENTS_DIR / doc_id
            doc_dir.mkdir(exist_ok=True)
            
            # Save document metadata
            with (doc_dir / 'metadata.json').open('w') as f:
                json.dump(doc, f)
            
            # Download attachments
            for idx, attachment in enumerate(attachments):
                url = attachment.get('url')
                if not url:
                    continue
                    
                response = requests.get(
                    url, 
                    headers={"Authorization": f"Bearer {access_token}"},
                    stream=True
                )
                response.raise_for_status()
                
                ext = url.split('.')[-1].lower()
                file_path = doc_dir / f"attachment_{idx}.{ext}"
                
                with file_path.open('wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            doc_cache[doc_id] = {
                'downloaded_at': datetime.now().isoformat(),
                'path': str(doc_dir)
            }
            return doc_id
            
        except Exception as e:
            print(f"Error downloading document {doc.get('id')}: {str(e)}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_single_document, doc) for doc in documents]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    time.sleep(RATE_LIMIT_DELAY)  # Avoid rate limiting
            except Exception as e:
                print(f"Error processing document: {str(e)}")

def fetch_all_documents(access_token: str) -> None:
    """Fetch all documents in batches with parallel downloading"""
    doc_cache = load_document_cache()
    offset = 0
    total_processed = 0
    
    while True:
        print(f"\n📡 Fetching document batch starting at offset {offset}")
        
        batch = fetch_documents_batch(offset, BATCH_SIZE, access_token)
        if not batch:
            break
            
        print(f"\n📊 Batch Statistics:")
        print(f"   Total Documents: {len(batch)}")
        
        # Process batch in parallel
        download_document_batch(batch, access_token, doc_cache)
        
        total_processed += len(batch)
        print(f"   Running Total Processed: {total_processed}")
        
        # Save progress periodically
        if total_processed % 500 == 0:
            save_document_cache(doc_cache)
        
        if len(batch) < BATCH_SIZE:
            break
            
        offset += BATCH_SIZE
        time.sleep(RATE_LIMIT_DELAY)  # Avoid rate limiting
    
    # Save final cache
    save_document_cache(doc_cache)
    print(f"\n✅ Completed processing {total_processed} documents") 