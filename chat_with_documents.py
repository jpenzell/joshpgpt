import os
import openai
import json
from datetime import datetime
import textwrap
import time
from pinecone import Pinecone

# Initialize OpenAI API key (optional if set via environment variables)
openai.api_key = os.getenv('OPENAI_API_KEY')

def init_pinecone():
    """Initialize Pinecone connection"""
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not pinecone_api_key:
        raise ValueError("Please set PINECONE_API_KEY environment variable")
    
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(
        host=os.getenv('PINECONE_INDEX_HOST')
    )
    return index

def create_embedding(text):
    """Create embedding with retry logic and dimension verification"""
    max_retries = 3
    expected_dimension = 1536  # text-embedding-3-small dimension
    
    for attempt in range(max_retries):
        try:
            print(f"Creating embedding (attempt {attempt + 1}/{max_retries})")
            client = openai.OpenAI()
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL'),
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Verify embedding dimension
            if len(embedding) != expected_dimension:
                print(f"Warning: Unexpected embedding dimension {len(embedding)}, expected {expected_dimension}")
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
        include_metadata=True,
        namespace=os.getenv('PINECONE_INDEX_NAME')
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
    """Chat with documents using GPT-4o"""
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
            top_k=5,  # Get more results for better context
            include_metadata=True,
            namespace=os.getenv('PINECONE_INDEX_NAME')
        )
        
        # Format context from results
        context = []
        for match in results.matches:
            metadata = match.metadata
            score = match.score if hasattr(match, 'score') else 'N/A'
            
            # Create a rich context including metadata and relevance score
            doc_context = {
                'relevance': score,
                'category': metadata.get('category', 'Unknown'),
                'date': metadata.get('created', 'Unknown'),
                'text': metadata.get('text', 'No text available'),
                'total': metadata.get('total', 'N/A'),
                'tax': metadata.get('tax', 'N/A'),
                'document_id': metadata.get('id', 'Unknown')
            }
            context.append(doc_context)
        
        # Create a detailed system message
        system_message = """You are an AI assistant analyzing personal documents from Shoeboxed. 
        You have access to document metadata, text content, and relevance scores.
        
        When answering:
        1. Focus on the most relevant documents first (higher relevance scores)
        2. Include specific dates, amounts, and categories when available
        3. If you notice patterns or relationships between documents, mention them
        4. Be clear about which documents you're referencing
        5. If the information seems incomplete or unclear, say so
        6. Use the document metadata to provide context
        
        Remember: These are personal financial and business documents, so be precise and professional."""
        
        # Create a conversation with GPT-4o
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_CHAT_MODEL'),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Based on the following documents, please answer this question: {query}\n\nDocuments:\n{json.dumps(context, indent=2)}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in chat_with_documents: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}"

if __name__ == "__main__":
    chat_with_documents() 