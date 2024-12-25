import requests
import json
import os
from datetime import datetime
import time
import traceback

def load_tokens():
    """Load tokens from file"""
    try:
        with open('tokens.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception("No tokens found. Please run auth_flow.py first.")

def refresh_if_needed(tokens):
    """Check if token needs refresh and refresh if necessary"""
    expires_at = datetime.fromisoformat(tokens['expires_at'])
    if datetime.now() >= expires_at:
        from auth_flow import refresh_access_token
        return refresh_access_token(tokens['refresh_token'])
    return tokens

def download_image(url, access_token, save_path):
    """Download an image from Shoeboxed API"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def get_organization_id(access_token):
    """Get the organization ID for the authenticated user"""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Origin': 'https://api.shoeboxed.com',
        'Referer': 'https://api.shoeboxed.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        # Get user info first
        user_response = requests.get('https://api.shoeboxed.com/v2/user', headers=headers)
        print(f"User response status: {user_response.status_code}")
        print(f"User response body: {user_response.text}")
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            # Get the first account ID from the user's accounts
            if 'accounts' in user_data and len(user_data['accounts']) > 0:
                account_id = user_data['accounts'][0].get('id')
                if account_id:
                    print(f"Found account ID: {account_id}")
                    return account_id
        
        raise Exception(f"No account ID found in user data. User response: {user_response.text}")
        
    except Exception as e:
        print(f"Error getting account ID: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def load_progress():
    """Load the current progress of document fetching"""
    progress_file = 'fetch_progress.json'
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'last_offset': 0,
            'processed_document_ids': set(),
            'total_documents_processed': 0
        }

def save_progress(progress):
    """Save the current progress of document fetching"""
    progress_file = 'fetch_progress.json'
    # Convert set to list for JSON serialization
    if isinstance(progress.get('processed_document_ids'), set):
        progress['processed_document_ids'] = list(progress['processed_document_ids'])
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def fetch_documents(access_token, modified_since=None):
    """Enhanced fetch_documents with resumability"""
    # First get account ID
    account_id = get_organization_id(access_token)
    
    # Load existing progress
    progress = load_progress()
    
    # Get list of documents
    list_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Origin': 'https://api.shoeboxed.com',
        'Referer': 'https://api.shoeboxed.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    params = {
        'limit': 50,  # Number of documents per page
        'offset': progress.get('last_offset', 0),
        'include': 'attachments,metadata'
    }
    
    if modified_since:
        params['modified_since'] = modified_since
    
    all_documents = []
    processed_document_ids = set(progress.get('processed_document_ids', []))
    
    while True:
        print(f"Fetching document list with params: {params}")
        response = requests.get(list_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Failed to fetch document list. Status: {response.status_code}")
            raise Exception(f"Failed to fetch document list: {response.text}")
        
        data = response.json()
        doc_list = data.get('documents', [])
        
        # Fetch each document's details
        for doc in doc_list:
            doc_id = doc.get('id')
            if not doc_id or doc_id in processed_document_ids:
                continue
                
            print(f"Fetching details for document {doc_id}")
            doc_url = f"https://api.shoeboxed.com/v2/accounts/{account_id}/documents/{doc_id}"
            doc_response = requests.get(doc_url, headers=headers)
            
            if doc_response.status_code == 200:
                doc_data = doc_response.json()
                all_documents.append(doc_data)
                processed_document_ids.add(doc_id)
                
                # Save progress after each document
                progress = {
                    'last_offset': params['offset'],
                    'processed_document_ids': processed_document_ids,
                    'total_documents_processed': len(processed_document_ids)
                }
                save_progress(progress)
            else:
                print(f"Failed to fetch document {doc_id}. Status: {doc_response.status_code}")
            
            time.sleep(0.5)  # Be nice to the API
        
        # Check if there are more documents
        if len(doc_list) < params['limit']:
            break
            
        params['offset'] += params['limit']
        time.sleep(1)  # Be nice to the API
    
    return all_documents

def save_documents_and_images(documents, access_token):
    """Enhanced save method with progress tracking"""
    # Create base directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'documents_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f'{base_dir}/images', exist_ok=True)
    
    # Save metadata
    metadata_file = f'{base_dir}/metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(documents, f, indent=2)
    
    print(f"Saved metadata for {len(documents)} documents to {metadata_file}")
    
    # Download images with progress tracking
    total_images = 0
    image_progress_file = f'{base_dir}/image_download_progress.json'
    
    # Load existing image download progress if it exists
    try:
        with open(image_progress_file, 'r') as f:
            image_progress = json.load(f)
    except FileNotFoundError:
        image_progress = {'downloaded_doc_ids': []}
    
    for doc in documents:
        # Skip already downloaded images
        if doc['id'] in image_progress.get('downloaded_doc_ids', []):
            continue
        
        if 'attachment' in doc:
            # Get the PDF version if available, otherwise get the original image
            image_url = doc['attachment'].get('pdf', doc['attachment'].get('original'))
            if image_url:
                file_extension = 'pdf' if 'pdf' in doc['attachment'] else image_url.split('.')[-1]
                image_filename = f"{base_dir}/images/{doc['id']}.{file_extension}"
                
                print(f"Downloading image for document {doc['id']}...")
                if download_image(image_url, access_token, image_filename):
                    total_images += 1
                    image_progress['downloaded_doc_ids'].append(doc['id'])
                    
                    # Save image download progress after each successful download
                    with open(image_progress_file, 'w') as f:
                        json.dump(image_progress, f, indent=2)
                    
                    print(f"Successfully downloaded image to {image_filename}")
                else:
                    print(f"Failed to download image for document {doc['id']}")
                
                time.sleep(0.5)  # Be nice to the API
    
    print(f"\nDownload complete!")
    print(f"Total documents: {len(documents)}")
    print(f"Total images downloaded: {total_images}")
    print(f"All files saved in directory: {base_dir}")

def main():
    # Load tokens
    tokens = load_tokens()
    
    # Refresh token if needed
    tokens = refresh_if_needed(tokens)
    
    # Fetch documents
    print("Fetching documents...")
    documents = fetch_documents(tokens['access_token'])
    
    # Save documents and download images
    save_documents_and_images(documents, tokens['access_token'])

if __name__ == "__main__":
    main() 