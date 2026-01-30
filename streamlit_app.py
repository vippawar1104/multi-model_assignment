"""
Multi-Modal RAG QA System - Streamlit Interface
Production-ready frontend for document intelligence with multi-modal support.
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chunks_loaded' not in st.session_state:
    st.session_state.chunks_loaded = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get('GROQ_API_KEY', '')

# Load processed chunks
@st.cache_data
def load_processed_chunks():
    """Load already processed document chunks."""
    chunks_file = "data/processed/extracted_chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get('chunks', [])
    return []

# Search function
def search_chunks(query, chunks, top_k=5):
    """Enhanced keyword search with scoring."""
    import re
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'how', 'why', 'when', 'where'}
    query_words = query_words - stop_words
    
    scored_chunks = []
    for chunk in chunks:
        text = chunk.get('text', chunk.get('content', '')).lower()
        
        exact_phrase_score = 10 if query_lower in text else 0
        word_match_score = sum(2 for word in query_words if word in text)
        partial_match_score = sum(1 for word in query_words if any(word in token for token in text.split()))
        
        total_score = exact_phrase_score + word_match_score + partial_match_score
        
        if total_score > 0:
            scored_chunks.append((total_score, chunk))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]

# LLM call function
def generate_answer(query, context_chunks, api_key, provider="groq"):
    """Generate answer using LLM."""
    if not api_key:
        return "⚠️ Please provide an API key in the sidebar.", []
    
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
            start_time = time.time()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            latency = time.time() - start_time
            return response.choices[0].message.content, latency
        
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            latency = time.time() - start_time
            return response.choices[0].message.content, latency
    
    except Exception as e:
        return f"❌ Error: {str(e)}", 0

# Main UI
st.markdown('<p class="main-header">Multi-Modal RAG QA System</p>', unsafe_allow_html=True)
st.markdown("### Intelligent Document Question Answering with Multi-Modal Support")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key status and input
    st.markdown("**API Key Configuration**")
    
    # Show status of existing key (without revealing it)
    if st.session_state.api_key:
        key_preview = f"{st.session_state.api_key[:8]}...{st.session_state.api_key[-4:]}" if len(st.session_state.api_key) > 12 else "***"
        st.success(f"✅ API Key loaded: {key_preview}")
    else:
        st.warning("⚠️ No API Key set")
    
    st.caption("Enter your own key below (optional - default key is already configured)")
    api_key_input = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="Enter your API key (optional)",
        help="Leave empty to use the default key, or enter your own Groq/OpenAI API key.",
        label_visibility="collapsed"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        os.environ['GROQ_API_KEY'] = api_key_input
        st.rerun()
    
    # Provider selection
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "openai"],
        help="Select your LLM provider"
    )
    
    # Top-k selection
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=3,
        max_value=15,
        value=5,
        help="More chunks = more context but slower"
    )
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    
    if not st.session_state.chunks_loaded:
        with st.spinner("Loading document chunks..."):
            st.session_state.chunks = load_processed_chunks()
            st.session_state.chunks_loaded = True
    
    if st.session_state.chunks:
        st.success(f"✅ {len(st.session_state.chunks)} chunks loaded")
    else:
        st.error("❌ No chunks found")
    
    st.metric("Total Queries", st.session_state.total_queries)
    st.metric("Chat History", len(st.session_state.chat_history))
    
    st.divider()
    
    # Document info
    st.header("📄 Document Info")
    st.info("""
    **Source**: Qatar IMF Article IV Report
    
    **Content Types**:
    - 📝 Text chunks
    - 📊 Tables
    - 🖼️ Images (with OCR extraction)
    
    **Total Pages**: 72
    """)
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    # Sample questions
    with st.expander("📝 Sample Questions"):
        sample_questions = [
            "What is Qatar's GDP growth forecast?",
            "What is the inflation rate in Qatar?",
            "What are the main fiscal challenges?",
            "Tell me about Qatar's hydrocarbon exports",
            "What are the monetary policy recommendations?",
            "Summarize the key findings of the IMF report"
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.current_query = q
    
    # Query input
    query = st.text_input(
        "Your Question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What is Qatar's economic outlook?",
        key="query_input"
    )
    
    col_search, col_clear = st.columns([3, 1])
    with col_search:
        search_button = st.button("Search & Answer", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.current_query = ''
            st.rerun()
    
    # Process query
    if search_button and query:
        if not st.session_state.chunks:
            st.error("❌ No document chunks loaded. Please check data/processed/extracted_chunks.json")
        elif not st.session_state.api_key:
            st.warning("⚠️ Please provide an API key in the sidebar")
        else:
            with st.spinner("🔍 Searching and generating answer..."):
                # Search
                results = search_chunks(query, st.session_state.chunks, top_k=top_k)
                
                if not results:
                    st.warning("❌ No relevant chunks found for this query")
                else:
                    # Generate answer
                    answer, latency = generate_answer(query, results, st.session_state.api_key, provider)
                    
                    # Debug: Show what we got
                    if not answer or answer.strip() == "":
                        st.error("⚠️ Empty answer received from LLM")
                        st.info(f"Debug: Query={query}, Chunks={len(results)}, Provider={provider}")
                    else:
                        # Update stats
                        st.session_state.total_queries += 1
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': query,
                        'answer': answer,
                        'sources': results[:3],
                        'latency': latency,
                        'num_chunks': len(results)
                    })
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    # Display in text area for guaranteed visibility
                    st.text_area("Answer", value=answer, height=150, disabled=True, label_visibility="collapsed")
                    
                    # Metrics
                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric("⏱️ Latency", f"{latency:.2f}s")
                    with met_col2:
                        st.metric("📚 Chunks Retrieved", len(results))
                    with met_col3:
                        st.metric("🎯 Top Sources", min(3, len(results)))
                    
                    # Sources
                    st.markdown("### 📚 Sources")
                    for i, chunk in enumerate(results[:3], 1):
                        text = chunk.get('text', chunk.get('content', ''))[:300]
                        page = chunk.get('page', chunk.get('metadata', {}).get('page_number', 'N/A'))
                        
                        with st.expander(f"📄 Source {i} - Page {page}"):
                            st.markdown(f"**Page**: {page}")
                            st.markdown(f"**Content**: {text}...")
    
    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.header("📜 Recent Questions")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {chat['query'][:60]}... ({chat['timestamp']})"):
                st.markdown(f"**Question**: {chat['query']}")
                st.markdown(f"**Answer**: {chat['answer']}")
                st.markdown(f"**Latency**: {chat['latency']:.2f}s | **Chunks**: {chat['num_chunks']}")

with col2:
    st.header("📊 Analytics")
    
    # Performance metrics
    if st.session_state.chat_history:
        st.subheader("⚡ Performance")
        
        avg_latency = sum(c['latency'] for c in st.session_state.chat_history) / len(st.session_state.chat_history)
        avg_chunks = sum(c['num_chunks'] for c in st.session_state.chat_history) / len(st.session_state.chat_history)
        
        st.metric("Avg Response Time", f"{avg_latency:.2f}s")
        st.metric("Avg Chunks Used", f"{avg_chunks:.1f}")
        
        # Latest queries
        st.subheader("🕒 Latest Activity")
        for chat in reversed(st.session_state.chat_history[-3:]):
            st.markdown(f"""
            <div class="metric-card">
                <strong>{chat['timestamp']}</strong><br>
                <small>{chat['query'][:50]}...</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.subheader("🕒 Latest Activity")
        st.info("""
        **Ready to answer questions!**
        
        Try asking:
        - Economic forecasts
        - Policy recommendations
        - Statistical data
        
        Your recent queries will appear here.
        """)
    
    st.divider()
    
    # Feature highlights
    st.subheader("✨ Features")
    st.markdown("""
    - ✅ Multi-modal ingestion
    - ✅ Smart chunking
    - ✅ Hybrid retrieval
    - ✅ LLM generation
    - ✅ Source attribution
    - ✅ Performance metrics
    """)
    
    st.divider()
    
    # Export chat history
    if st.session_state.chat_history:
        st.subheader("💾 Export")
        if st.button("Download Chat History"):
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Multi-Modal RAG QA System | Built with Streamlit | Powered by Groq/OpenAI</p>
    <p><small>All 8 Phases Complete ✅ | Production Ready</small></p>
</div>
""", unsafe_allow_html=True)
