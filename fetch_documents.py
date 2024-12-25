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
        # First try to get user info
        user_response = requests.get('https://api.shoeboxed.com/v2/user', headers=headers)
        if user_response.status_code == 200:
            user_data = user_response.json()
            print(f"User data: {json.dumps(user_data, indent=2)}")
        
        # Then get organizations
        org_response = requests.get('https://api.shoeboxed.com/v2/organizations', headers=headers)
        print(f"Organization response status: {org_response.status_code}")
        print(f"Organization response body: {org_response.text}")
        
        if org_response.status_code == 200:
            orgs = org_response.json()
            if isinstance(orgs, list) and len(orgs) > 0:
                return orgs[0].get('id')
            elif isinstance(orgs, dict) and 'organizations' in orgs:
                orgs_list = orgs['organizations']
                if len(orgs_list) > 0:
                    return orgs_list[0].get('id')
        
        raise Exception(f"No organization ID found. Response: {org_response.text}")
        
    except Exception as e:
        print(f"Error getting organization ID: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def fetch_documents(access_token, modified_since=None):
    """Fetch documents from Shoeboxed API"""
    # First get organization ID
    print("Getting organization ID...")
    org_id = get_organization_id(access_token)
    print(f"Using organization ID: {org_id}")
    
    # Get list of documents
    list_url = f"https://api.shoeboxed.com/v2/organizations/{org_id}/documents"
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
        'offset': 0,
        'include': 'attachments,metadata'  # Include all necessary data
    }
    
    if modified_since:
        params['modified_since'] = modified_since
    
    all_documents = []
    
    # Get list of document IDs
    while True:
        print(f"Fetching document list with params: {params}")
        response = requests.get(list_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Failed to fetch document list. Status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response body: {response.text}")
            raise Exception(f"Failed to fetch document list: {response.text}")
        
        data = response.json()
        doc_list = data.get('documents', [])
        
        # Fetch each document's details
        for doc in doc_list:
            doc_id = doc.get('id')
            if not doc_id:
                continue
                
            print(f"Fetching details for document {doc_id}")
            doc_url = f"https://api.shoeboxed.com/v2/organizations/{org_id}/documents/{doc_id}"
            doc_response = requests.get(doc_url, headers=headers)
            
            if doc_response.status_code == 200:
                doc_data = doc_response.json()
                all_documents.append(doc_data)
            else:
                print(f"Failed to fetch document {doc_id}. Status: {doc_response.status_code}")
                print(f"Response: {doc_response.text}")
            
            time.sleep(0.5)  # Be nice to the API
        
        # Check if there are more documents
        if len(doc_list) < params['limit']:
            break
            
        params['offset'] += params['limit']
        time.sleep(1)  # Be nice to the API
    
    return all_documents

def save_documents_and_images(documents, access_token):
    """Save documents metadata and download their images"""
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
    
    # Download images
    total_images = 0
    for doc in documents:
        if 'attachment' in doc:
            # Get the PDF version if available, otherwise get the original image
            image_url = doc['attachment'].get('pdf', doc['attachment'].get('original'))
            if image_url:
                file_extension = 'pdf' if 'pdf' in doc['attachment'] else image_url.split('.')[-1]
                image_filename = f"{base_dir}/images/{doc['id']}.{file_extension}"
                
                print(f"Downloading image for document {doc['id']}...")
                if download_image(image_url, access_token, image_filename):
                    total_images += 1
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