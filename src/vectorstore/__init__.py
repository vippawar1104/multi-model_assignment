"""
Vector store module for the Multi-Modal RAG system.
Handles embedding generation and vector storage.
"""

from typing import TYPE_CHECKING

# Try importing main components
try:
    from .embedding_generator import EmbeddingGenerator
    EMBEDDING_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EmbeddingGenerator not available: {e}")
    EMBEDDING_GENERATOR_AVAILABLE = False
    EmbeddingGenerator = None

try:
    from .faiss_store import FAISSVectorStore
    FAISS_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FAISSVectorStore not available: {e}")
    FAISS_STORE_AVAILABLE = False
    FAISSVectorStore = None

try:
    from .chroma_store import ChromaVectorStore
    CHROMA_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChromaVectorStore not available: {e}")
    CHROMA_STORE_AVAILABLE = False
    ChromaVectorStore = None

try:
    from .hybrid_store import HybridVectorStore
    HYBRID_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HybridVectorStore not available: {e}")
    HYBRID_STORE_AVAILABLE = False
    HybridVectorStore = None

__all__ = [
    "EmbeddingGenerator",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "HybridVectorStore",
]

__version__ = "0.1.0"
