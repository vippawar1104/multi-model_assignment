"""
Reranker for the Multi-Modal RAG system.
Improves initial retrieval results through reranking and scoring.
"""

from typing import List, Dict, Any, Optional
import re

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class Reranker:
    """
    Rerank retrieval results using various strategies.
    Supports relevance-based, diversity-based, and custom reranking.
    """

    def __init__(
        self,
        strategy: str = "relevance",
        diversity_penalty: float = 0.5,
        recency_weight: float = 0.0
    ):
        """
        Initialize reranker.

        Args:
            strategy: Reranking strategy ('relevance', 'diversity', 'hybrid')
            diversity_penalty: Penalty for similar results (0-1)
            recency_weight: Weight for recency in scoring (0-1)
        """
        self.strategy = strategy or get_config_value("retrieval.rerank_strategy", "relevance")
        self.diversity_penalty = diversity_penalty
        self.recency_weight = recency_weight
        
        logger.info(f"Reranker initialized with strategy={self.strategy}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on configured strategy.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not results:
            return results
        
        if self.strategy == "relevance":
            reranked = self._rerank_by_relevance(query, results)
        elif self.strategy == "diversity":
            reranked = self._rerank_by_diversity(query, results)
        elif self.strategy == "hybrid":
            reranked = self._rerank_hybrid(query, results)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using relevance")
            reranked = self._rerank_by_relevance(query, results)
        
        # Apply top_k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        logger.info(f"Reranked {len(results)} results to {len(reranked)} results")
        return reranked

    def _rerank_by_relevance(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank by relevance score (simple keyword matching).

        Args:
            query: Query text
            results: Results to rerank

        Returns:
            Reranked results
        """
        # Extract query keywords
        query_terms = set(query.lower().split())
        
        # Score each result
        scored_results = []
        for result in results:
            text = result.get("text", "").lower()
            
            # Count keyword matches
            keyword_score = sum(1 for term in query_terms if term in text)
            
            # Combine with original similarity score
            original_score = result.get("similarity_score", result.get("score", 0.0))
            combined_score = original_score + (keyword_score * 0.1)
            
            result["rerank_score"] = combined_score
            scored_results.append(result)
        
        # Sort by combined score
        reranked = sorted(scored_results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    def _rerank_by_diversity(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank to maximize diversity (Maximal Marginal Relevance).

        Args:
            query: Query text
            results: Results to rerank

        Returns:
            Diverse reranked results
        """
        if not results:
            return results
        
        # Start with highest scoring result
        reranked = [results[0]]
        remaining = results[1:]
        
        while remaining and len(reranked) < len(results):
            # Find result that maximizes: relevance - diversity_penalty * max_similarity_to_selected
            best_score = -float('inf')
            best_idx = 0
            
            for idx, candidate in enumerate(remaining):
                relevance = candidate.get("similarity_score", candidate.get("score", 0.0))
                
                # Calculate max similarity to already selected results
                max_sim = 0.0
                candidate_text = candidate.get("text", "")
                for selected in reranked:
                    selected_text = selected.get("text", "")
                    similarity = self._text_similarity(candidate_text, selected_text)
                    max_sim = max(max_sim, similarity)
                
                # MMR score
                mmr_score = relevance - (self.diversity_penalty * max_sim)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add best candidate to reranked list
            reranked.append(remaining.pop(best_idx))
        
        return reranked

    def _rerank_hybrid(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Hybrid reranking combining relevance and diversity.

        Args:
            query: Query text
            results: Results to rerank

        Returns:
            Reranked results
        """
        # First pass: relevance reranking
        relevance_ranked = self._rerank_by_relevance(query, results)
        
        # Second pass: apply diversity with lower penalty
        diverse_ranked = self._rerank_by_diversity(
            query,
            relevance_ranked[:min(20, len(relevance_ranked))]  # Only diversify top 20
        )
        
        # Combine with remaining results
        if len(relevance_ranked) > 20:
            diverse_ranked.extend(relevance_ranked[20:])
        
        return diverse_ranked

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (Jaccard similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Filter results by minimum score threshold.

        Args:
            results: Results to filter
            threshold: Minimum score

        Returns:
            Filtered results
        """
        filtered = [
            r for r in results
            if r.get("similarity_score", r.get("score", 0.0)) >= threshold
        ]
        logger.info(f"Filtered {len(results)} to {len(filtered)} results (threshold={threshold})")
        return filtered

    def deduplicate(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate results.

        Args:
            results: Results to deduplicate
            similarity_threshold: Threshold for considering duplicates

        Returns:
            Deduplicated results
        """
        if not results:
            return results
        
        unique_results = [results[0]]
        
        for candidate in results[1:]:
            candidate_text = candidate.get("text", "")
            
            # Check similarity with already selected results
            is_duplicate = False
            for selected in unique_results:
                selected_text = selected.get("text", "")
                similarity = self._text_similarity(candidate_text, selected_text)
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(candidate)
        
        logger.info(f"Deduplicated {len(results)} to {len(unique_results)} results")
        return unique_results

    def boost_by_metadata(
        self,
        results: List[Dict[str, Any]],
        boost_rules: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Boost scores based on metadata attributes.

        Args:
            results: Results to boost
            boost_rules: Dict of metadata_key -> boost_factor

        Returns:
            Results with boosted scores
        """
        boosted_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            original_score = result.get("similarity_score", result.get("score", 0.0))
            boost = 1.0
            
            # Apply boost rules
            for key, factor in boost_rules.items():
                if key in metadata and metadata[key]:
                    boost *= factor
            
            result["boosted_score"] = original_score * boost
            boosted_results.append(result)
        
        # Re-sort by boosted score
        boosted_results = sorted(boosted_results, key=lambda x: x.get("boosted_score", 0.0), reverse=True)
        return boosted_results


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    strategy: str = "relevance",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank results.

    Args:
        query: Query text
        results: Results to rerank
        strategy: Reranking strategy
        top_k: Number of results to return

    Returns:
        Reranked results
    """
    reranker = Reranker(strategy=strategy)
    return reranker.rerank(query, results, top_k=top_k)
