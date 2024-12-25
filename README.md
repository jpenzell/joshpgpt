# Shoeboxed Document Processing and Chat Application

This application integrates with Shoeboxed's API to process documents and provide a chat interface for interacting with the processed documents. It uses OpenAI's GPT-4 for OCR and document understanding, and Pinecone for vector storage and retrieval.

## Features

- OAuth2.0 authentication with Shoeboxed
- Document retrieval and processing from Shoeboxed
- OCR processing using GPT-4
- Vector embedding storage in Pinecone
- S3 storage for processed documents
- Interactive chat interface for document queries

## Prerequisites

- Python 3.8+
- Shoeboxed API credentials
- OpenAI API key
- Pinecone API key and environment
- AWS S3 credentials

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/shoeboxed-processor.git
cd shoeboxed-processor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```
SHOEBOXED_CLIENT_ID=your_client_id
SHOEBOXED_CLIENT_SECRET=your_client_secret
SHOEBOXED_REDIRECT_URI=http://localhost:8000/callback
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_bucket_name
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application and authenticate with your Shoeboxed account
2. Select documents to process from your Shoeboxed account
3. The application will download, process, and store the documents
4. Use the chat interface to ask questions about your processed documents

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)
