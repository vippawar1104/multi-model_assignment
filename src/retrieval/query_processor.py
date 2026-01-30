"""
Query processor for the Multi-Modal RAG system.
Handles query preprocessing, expansion, and optimization.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class QueryProcessor:
    """
    Process and optimize queries for better retrieval results.
    Includes query cleaning, expansion, and reformulation.
    """

    def __init__(
        self,
        min_query_length: int = 3,
        max_query_length: int = 512,
        expand_queries: bool = False,
        remove_stopwords: bool = False
    ):
        """
        Initialize query processor.

        Args:
            min_query_length: Minimum query length in characters
            max_query_length: Maximum query length in characters
            expand_queries: Whether to expand queries with synonyms
            remove_stopwords: Whether to remove common stopwords
        """
        self.min_query_length = min_query_length or get_config_value("retrieval.min_query_length", 3)
        self.max_query_length = max_query_length or get_config_value("retrieval.max_query_length", 512)
        self.expand_queries = expand_queries
        self.remove_stopwords = remove_stopwords
        
        # Common stopwords (simple set for now)
        self.stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "what", "when", "where", "who", "why"
        }
        
        logger.info(f"QueryProcessor initialized (min_length={self.min_query_length}, max_length={self.max_query_length})")

    def process(self, query: str) -> str:
        """
        Process a single query.

        Args:
            query: Raw query text

        Returns:
            Processed query
        """
        if not query:
            return ""
        
        # Clean query
        processed = self._clean_query(query)
        
        # Validate length
        if len(processed) < self.min_query_length:
            logger.warning(f"Query too short: '{processed}'")
            return processed
        
        if len(processed) > self.max_query_length:
            logger.warning(f"Query too long, truncating to {self.max_query_length} chars")
            processed = processed[:self.max_query_length]
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            processed = self._remove_stopwords(processed)
        
        # Expand query if enabled
        if self.expand_queries:
            processed = self._expand_query(processed)
        
        logger.debug(f"Processed query: '{query}' -> '{processed}'")
        return processed

    def process_batch(self, queries: List[str]) -> List[str]:
        """
        Process multiple queries.

        Args:
            queries: List of raw queries

        Returns:
            List of processed queries
        """
        return [self.process(q) for q in queries]

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Strip whitespace
        cleaned = query.strip()
        
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        cleaned = re.sub(r'[^a-zA-Z0-9\s\?\.\,\-\_]', '', cleaned)
        
        # Lowercase (optional - might want to preserve case for some use cases)
        # cleaned = cleaned.lower()
        
        return cleaned

    def _remove_stopwords(self, query: str) -> str:
        """Remove common stopwords from query."""
        words = query.lower().split()
        filtered_words = [w for w in words if w not in self.stopwords]
        
        # If all words were stopwords, return original
        if not filtered_words:
            return query
        
        return ' '.join(filtered_words)

    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms or related terms.
        
        Note: This is a placeholder. In production, you might use:
        - WordNet for synonyms
        - A language model for query expansion
        - Domain-specific expansion rules
        """
        # Simple expansion rules (placeholder)
        expansions = {
            "ML": "machine learning",
            "AI": "artificial intelligence",
            "DL": "deep learning",
            "NLP": "natural language processing",
            "CV": "computer vision"
        }
        
        expanded = query
        for abbr, full in expansions.items():
            # Replace abbreviation with full form
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, f"{abbr} {full}", expanded, flags=re.IGNORECASE)
        
        return expanded

    def extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """
        Extract key terms from query.

        Args:
            query: Query text
            top_k: Number of keywords to extract

        Returns:
            List of keywords
        """
        # Clean query
        cleaned = self._clean_query(query)
        
        # Split into words
        words = cleaned.lower().split()
        
        # Remove stopwords
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Simple frequency-based extraction (in production, use TF-IDF or similar)
        # For now, just return unique words, prioritizing longer words
        keywords = sorted(set(keywords), key=lambda x: len(x), reverse=True)
        
        return keywords[:top_k]

    def generate_variations(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate query variations for better coverage.

        Args:
            query: Original query
            max_variations: Maximum number of variations

        Returns:
            List of query variations including original
        """
        variations = [query]
        
        # Variation 1: Question format
        if not query.endswith('?'):
            variations.append(f"What is {query}?")
        
        # Variation 2: More specific
        keywords = self.extract_keywords(query, top_k=2)
        if keywords:
            variations.append(f"Explain {' and '.join(keywords)}")
        
        # Variation 3: Broader context
        variations.append(f"Information about {query}")
        
        return variations[:max_variations + 1]

    def detect_intent(self, query: str) -> str:
        """
        Detect query intent.

        Args:
            query: Query text

        Returns:
            Intent type ('question', 'definition', 'comparison', 'general')
        """
        query_lower = query.lower()
        
        # Question patterns
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(query_lower.startswith(q) for q in question_starters) or query.endswith('?'):
            return 'question'
        
        # Definition patterns
        if 'define' in query_lower or 'what is' in query_lower or 'meaning of' in query_lower:
            return 'definition'
        
        # Comparison patterns
        if 'compare' in query_lower or 'difference between' in query_lower or 'vs' in query_lower:
            return 'comparison'
        
        # Default
        return 'general'

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate query and provide feedback.

        Args:
            query: Query text

        Returns:
            Validation result with is_valid flag and suggestions
        """
        result = {
            "is_valid": True,
            "original": query,
            "processed": "",
            "issues": [],
            "suggestions": []
        }
        
        # Check if empty
        if not query or not query.strip():
            result["is_valid"] = False
            result["issues"].append("Query is empty")
            result["suggestions"].append("Please provide a query")
            return result
        
        # Process query
        processed = self.process(query)
        result["processed"] = processed
        
        # Check length
        if len(processed) < self.min_query_length:
            result["is_valid"] = False
            result["issues"].append(f"Query too short (minimum {self.min_query_length} characters)")
            result["suggestions"].append("Try adding more detail to your query")
        
        if len(processed) > self.max_query_length:
            result["issues"].append(f"Query too long (maximum {self.max_query_length} characters)")
            result["suggestions"].append("Try to make your query more concise")
        
        # Check if query is just stopwords
        if self._is_only_stopwords(processed):
            result["is_valid"] = False
            result["issues"].append("Query contains only common words")
            result["suggestions"].append("Try adding more specific terms")
        
        return result

    def _is_only_stopwords(self, query: str) -> bool:
        """Check if query contains only stopwords."""
        words = query.lower().split()
        if not words:
            return True
        return all(w in self.stopwords for w in words)


def process_query(query: str, clean: bool = True, expand: bool = False) -> str:
    """
    Convenience function to process a query.

    Args:
        query: Raw query text
        clean: Whether to clean the query
        expand: Whether to expand the query

    Returns:
        Processed query
    """
    processor = QueryProcessor(expand_queries=expand)
    if clean:
        return processor.process(query)
    return query
