"""
Simple RAG Pipeline Runner - Works with existing processed data
No complex imports - just uses what's already been processed.
"""

import os
import sys
import json
import pickle
import argparse

def load_processed_chunks():
    """Load the already processed chunks."""
    chunks_file = "data/processed/extracted_chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                return data
            return data.get('chunks', [])
    return []

def load_vector_store():
    """Load FAISS index if available."""
    try:
        import faiss
        import numpy as np
        
        index_file = "data/vector_store/faiss.index"
        metadata_file = "data/vector_store/metadata.pkl"
        
        if not os.path.exists(index_file):
            return None, None
        
        index = faiss.read_index(index_file)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✓ Loaded FAISS index with {index.ntotal} vectors")
        return index, metadata
    except Exception as e:
        print(f"⚠️  Could not load FAISS: {e}")
        return None, None

def simple_search(query, chunks, top_k=5):
    """Simple keyword-based search in chunks with improved scoring."""
    import re
    
    # Normalize query
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Remove common stop words
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    query_words = query_words - stop_words
    
    scored_chunks = []
    for chunk in chunks:
        # Support both 'text' and 'content' keys
        text = chunk.get('text', chunk.get('content', '')).lower()
        
        # Multiple scoring strategies
        exact_phrase_score = 10 if query_lower in text else 0
        word_match_score = sum(2 for word in query_words if word in text)  # Count each word match
        partial_match_score = sum(1 for word in query_words if any(word in token for token in text.split()))
        
        total_score = exact_phrase_score + word_match_score + partial_match_score
        
        if total_score > 0:
            scored_chunks.append((total_score, chunk))
    
    # Sort by score (highest first)
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    return [chunk for score, chunk in scored_chunks[:top_k]]

def call_llm(query, context_chunks, provider="groq"):
    """Call LLM API to generate answer."""
    api_key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        return f"⚠️  No API key found. Set {provider.upper()}_API_KEY environment variable."
    
    # Format context - support both 'text' and 'content' keys
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk.get('text', chunk.get('content', ''))[:500]}"
        for i, chunk in enumerate(context_chunks[:3])
    ])
    
    prompt = f"""You are a helpful assistant answering questions based on provided document context.

Context from document:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- Be direct and confident in your answer
- If the context contains relevant information, provide a clear answer
- Only say "no information available" if the context is truly unrelated to the question
- Keep your answer concise and accurate

Answer:"""
    
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        else:
            return f"⚠️  Provider '{provider}' not supported yet"
    
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Query Tool")
    parser.add_argument('--query', type=str, required=True, help='Your question')
    parser.add_argument('--provider', type=str, default='groq', choices=['groq', 'openai'],
                       help='LLM provider')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🤖 Simple RAG QA System")
    print("="*80)
    
    # Load data
    print("\n📚 Loading processed data...")
    chunks = load_processed_chunks()
    
    if not chunks:
        print("❌ No processed chunks found. Please run document processing first.")
        print("   File needed: data/processed/extracted_chunks.json")
        return
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Search
    print(f"\n🔍 Searching for: {args.query}")
    results = simple_search(args.query, chunks, top_k=args.top_k)
    
    if not results:
        print("❌ No relevant chunks found")
        return
    
    print(f"✓ Found {len(results)} relevant chunks")
    
    # Generate answer
    print(f"\n💭 Generating answer with {args.provider}...")
    answer = call_llm(args.query, results, provider=args.provider)
    
    # Display results
    print("\n" + "="*80)
    print("💡 ANSWER")
    print("="*80)
    print(f"\n{answer}\n")
    
    print("="*80)
    print("📚 SOURCES")
    print("="*80)
    for i, chunk in enumerate(results[:3], 1):
        text = chunk.get('text', chunk.get('content', ''))[:200]
        page = chunk.get('page', chunk.get('metadata', {}).get('page_number', 'N/A'))
        print(f"\n{i}. [Page {page}]")
        print(f"   {text}...")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
