import os
import openai
import json
from datetime import datetime
import textwrap
import time
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding_utils import EmbeddingManager
from typing import List, Dict, Optional, Union
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get embedding dimensions from environment
EMBEDDING_DIMENSIONS = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536'))

# Initialize embedding manager
embedding_manager = EmbeddingManager()

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

def create_embedding(text: str) -> Optional[List[float]]:
    """Create embedding with enhanced quality control"""
    max_retries = int(os.getenv('MAX_RETRIES', '3'))
    
    for attempt in range(max_retries):
        try:
            print(f"Creating embedding (attempt {attempt + 1}/{max_retries})")
            
            # Generate embedding
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text,
                dimensions=EMBEDDING_DIMENSIONS
            )
            
            embedding = response.data[0].embedding
            
            # Validate quality
            is_valid, quality_metrics = embedding_manager.validate_embedding_quality(embedding)
            if not is_valid:
                print(f"Warning: Low quality embedding detected: {quality_metrics}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            # Create metadata
            metadata = embedding_manager.create_embedding_metadata(
                content=text,
                model_name=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                dimensions=EMBEDDING_DIMENSIONS,
                content_type="query",
                processing_params={},
                quality_metrics=quality_metrics
            )
            
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

def semantic_search(
    query: str,
    index,
    top_k: int = 5,
    threshold: float = 0.7,
    search_type: str = "similarity"
) -> List[Dict]:
    """Enhanced semantic search with multiple search strategies"""
    try:
        # Create query embedding
        query_embedding = create_embedding(query)
        if not query_embedding:
            return []
        
        # Normalize the query embedding
        query_embedding = embedding_manager.normalize_embedding(query_embedding)
        
        # Prepare search parameters based on search type
        search_params = {
            "top_k": top_k,
            "include_metadata": True
        }
        
        if search_type == "similarity":
            # Standard similarity search
            results = index.query(
                vector=query_embedding,
                **search_params
            )
        elif search_type == "hybrid":
            # Hybrid search combining semantic and keyword matching
            results = index.query(
                vector=query_embedding,
                filter={
                    "text_match": {"$contains": query.lower()}
                },
                **search_params
            )
        elif search_type == "mmr":
            # Maximum Marginal Relevance search for diversity
            results = index.query(
                vector=query_embedding,
                sparse_vector=query.split(),  # Simple keyword representation
                **search_params
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Filter results by similarity threshold
        filtered_results = []
        for match in results.matches:
            if match.score >= threshold:
                filtered_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
        
        return filtered_results
        
    except Exception as e:
        print(f"Error in semantic search: {str(e)}")
        return []

def get_context_window(
    query: str,
    index,
    max_tokens: int = 3000,
    search_type: str = "similarity"
) -> str:
    """Get context window with enhanced search capabilities"""
    try:
        # Perform semantic search
        results = semantic_search(
            query=query,
            index=index,
            top_k=5,
            threshold=0.7,
            search_type=search_type
        )
        
        if not results:
            return "No relevant context found."
        
        # Build context window with metadata
        context_parts = []
        total_tokens = 0
        
        for result in results:
            # Extract text and metadata
            text = result["metadata"].get("text", "")
            source = result["metadata"].get("source", "Unknown")
            date = result["metadata"].get("date", "Unknown")
            score = result["score"]
            
            # Format context entry
            entry = f"\nSource: {source}\nDate: {date}\nRelevance: {score:.2f}\nContent:\n{text}\n"
            entry_tokens = len(entry.split())  # Simple token estimation
            
            if total_tokens + entry_tokens > max_tokens:
                break
                
            context_parts.append(entry)
            total_tokens += entry_tokens
        
        return "\n---\n".join(context_parts)
        
    except Exception as e:
        print(f"Error getting context window: {str(e)}")
        return "Error retrieving context."

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