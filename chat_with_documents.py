import os
import openai
import json
from datetime import datetime
import textwrap
import time
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get embedding dimensions from environment
EMBEDDING_DIMENSIONS = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536'))

def init_pinecone():
    """Initialize Pinecone connection"""
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

def create_embedding(text):
    """Create embedding with retry logic and dimension verification"""
    max_retries = int(os.getenv('MAX_RETRIES', '3'))
    
    for attempt in range(max_retries):
        try:
            print(f"Creating embedding (attempt {attempt + 1}/{max_retries})")
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL'),
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

def search_documents(query, index, n=5):
    """Search for relevant documents"""
    query_embedding = create_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=n,
        include_metadata=True
    )
    return results.matches

def format_document_context(documents):
    """Format retrieved documents into context for the AI"""
    context = []
    for doc in documents:
        metadata = doc.metadata
        score = doc.score if hasattr(doc, 'score') else 'N/A'
        
        # Create a more structured context
        doc_info = {
            'relevance': score,
            'id': metadata.get('id', 'Unknown'),
            'category': metadata.get('category', 'Unknown'),
            'date': metadata.get('created', 'Unknown'),
            'total': metadata.get('total', 'N/A'),
            'tax': metadata.get('tax', 'N/A'),
            'text': metadata.get('text', 'No text available'),
            'processed_date': metadata.get('processed_date', 'Unknown')
        }
        
        # Format the document information
        context.append(
            f"Document (ID: {doc_info['id']}, Relevance: {doc_info['relevance']:.2f}):\n"
            f"Category: {doc_info['category']}\n"
            f"Date: {doc_info['date']}\n"
            f"Amount: ${doc_info['total']}\n"
            f"Tax: ${doc_info['tax']}\n"
            f"Processed: {doc_info['processed_date']}\n"
            f"Content: {doc_info['text']}\n"
            f"---"
        )
    
    return "\n".join(context)

def chat_with_documents(query):
    """Chat with documents using GPT-4o with enhanced context understanding"""
    try:
        # Initialize Pinecone
        index = init_pinecone()
        
        # Create query embedding
        query_embedding = create_embedding(query)
        if not query_embedding:
            return "Sorry, I couldn't process your question. Please try again."
        
        # Search Pinecone with improved parameters
        results = index.query(
            vector=query_embedding,
            top_k=10,  # Get more results for better context
            include_metadata=True
        )
        
        # Format context from results
        context = []
        for match in results.matches:
            metadata = match.metadata
            score = match.score if hasattr(match, 'score') else 'N/A'
            
            # Create a rich context including enhanced metadata
            doc_context = {
                'relevance': score,
                'document_type': metadata.get('doc_type', 'Unknown'),
                'confidence': metadata.get('doc_type_confidence', 'N/A'),
                'date': metadata.get('original_date', 'Unknown'),
                'text': metadata.get('text', 'No text available'),
                'summary': metadata.get('summary', ''),
                'entities': metadata.get('entities', {}),
                'topics': metadata.get('topics', []),
                'language': metadata.get('language', 'en'),
                'document_id': metadata.get('id', 'Unknown'),
                'chunk_info': {
                    'chunk_id': metadata.get('chunk_id', 'N/A'),
                    'total_chunks': metadata.get('total_chunks', 1),
                    'is_summary': metadata.get('is_summary', False)
                }
            }
            context.append(doc_context)
        
        # Create a detailed system message
        system_message = """You are an AI assistant analyzing personal documents and serving as a "second brain". 
        You have access to document metadata, text content, summaries, and entity information.
        
        When answering:
        1. Focus on the most relevant documents first (higher relevance scores)
        2. Use the enhanced metadata to provide rich context:
           - Reference people, organizations, and locations mentioned
           - Consider the document type and its confidence score
           - Use summaries for overview and full text for details
           - Note temporal relationships between documents
           - Highlight connections between different documents
        3. If you notice patterns or relationships between documents, explain them
        4. Be clear about the source and type of each document
        5. If information seems incomplete, explain what's missing
        6. Consider the document's language and context
        7. For multi-chunk documents, ensure you're providing complete context
        
        Remember: This is personal information spanning many years and document types. 
        Be thorough but respect privacy and sensitivity of the content."""
        
        # Create a conversation with GPT-4o
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Based on the following documents, please answer this question: {query}\n\nDocuments:\n{json.dumps(context, indent=2)}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in chat_with_documents: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}"

if __name__ == "__main__":
    chat_with_documents() 