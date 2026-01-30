"""
Retrieval module for the Multi-Modal RAG system.
Handles query processing, similarity search, and result ranking.
"""

from typing import TYPE_CHECKING

# Try importing main components
try:
    from .retriever import Retriever
    RETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Retriever not available: {e}")
    RETRIEVER_AVAILABLE = False
    Retriever = None

try:
    from .query_processor import QueryProcessor
    QUERY_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QueryProcessor not available: {e}")
    QUERY_PROCESSOR_AVAILABLE = False
    QueryProcessor = None

try:
    from .reranker import Reranker
    RERANKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Reranker not available: {e}")
    RERANKER_AVAILABLE = False
    Reranker = None

try:
    from .hybrid_search import HybridSearcher
    HYBRID_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HybridSearcher not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False
    HybridSearcher = None

__all__ = [
    "Retriever",
    "QueryProcessor",
    "Reranker",
    "HybridSearcher",
]

__version__ = "0.1.0"
