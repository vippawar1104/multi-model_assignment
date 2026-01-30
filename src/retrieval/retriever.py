"""
Main retriever implementation for the Multi-Modal RAG system.
Handles the complete retrieval pipeline from query to ranked results.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.vectorstore import EmbeddingGenerator, FAISSVectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Main retrieval class that orchestrates the complete retrieval pipeline.
    Supports dense retrieval, filtering, and result ranking.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store_path: Optional[str] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Pre-initialized vector store
            embedding_generator: Pre-initialized embedding generator
            vector_store_path: Path to load vector store from
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score threshold
            include_metadata: Whether to include metadata in results
        """
        self.top_k = top_k or get_config_value("retrieval.top_k", 5)
        self.score_threshold = score_threshold or get_config_value("retrieval.score_threshold", None)
        self.include_metadata = include_metadata
        
        # Initialize embedding generator
        if embedding_generator:
            self.embedding_generator = embedding_generator
        else:
            logger.info("Initializing embedding generator...")
            self.embedding_generator = EmbeddingGenerator()
        
        # Initialize or load vector store
        if vector_store:
            self.vector_store = vector_store
        elif vector_store_path:
            logger.info(f"Loading vector store from {vector_store_path}")
            embedding_dim = get_config_value("embeddings.dimension", 384)
            self.vector_store = FAISSVectorStore(dimension=embedding_dim)
            self.vector_store.load(vector_store_path)
        else:
            logger.warning("No vector store provided. Initialize with set_vector_store()")
            self.vector_store = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "average_latency_ms": 0.0
        }
        
        logger.info(f"Retriever initialized with top_k={self.top_k}")

    def set_vector_store(self, vector_store: FAISSVectorStore):
        """Set or update the vector store."""
        self.vector_store = vector_store
        logger.info("Vector store updated")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of results (overrides default)
            filters: Metadata filters to apply
            return_scores: Whether to include similarity scores

        Returns:
            List of retrieved documents with metadata
        """
        start_time = time.time()
        
        if not self.vector_store:
            logger.error("No vector store available")
            return []
        
        k = top_k or self.top_k
        
        # Step 1: Generate query embedding
        logger.debug(f"Generating embedding for query: {query[:100]}...")
        query_embedding = self.embedding_generator.generate_text_embeddings(query)
        
        # Step 2: Search vector store
        logger.debug(f"Searching for top {k} results...")
        indices, distances, metadata_list = self.vector_store.search(
            query_embedding,
            k=k,
            filter_metadata=filters
        )
        
        # Step 3: Format results
        results = []
        for idx, distance, metadata in zip(indices, distances, metadata_list):
            # Convert distance to similarity score (lower distance = higher similarity)
            # For L2 distance, we can use 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance)
            
            # Apply score threshold if set
            if self.score_threshold and similarity_score < self.score_threshold:
                continue
            
            result = {
                "text": metadata.get("text", ""),
                "similarity_score": similarity_score,
                "distance": distance,
                "index": idx
            }
            
            # Add metadata if requested
            if self.include_metadata:
                result["metadata"] = metadata
            
            # Add scores if requested
            if return_scores:
                result["score"] = similarity_score
            
            results.append(result)
        
        # Update statistics
        latency_ms = (time.time() - start_time) * 1000
        self.stats["total_queries"] += 1
        self.stats["total_results"] += len(results)
        self.stats["average_latency_ms"] = (
            (self.stats["average_latency_ms"] * (self.stats["total_queries"] - 1) + latency_ms)
            / self.stats["total_queries"]
        )
        
        logger.info(f"Retrieved {len(results)} results in {latency_ms:.2f}ms")
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            filters: Metadata filters to apply

        Returns:
            List of result lists, one per query
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        results = []
        
        for query in queries:
            query_results = self.retrieve(query, top_k=top_k, filters=filters)
            results.append(query_results)
        
        return results

    def retrieve_with_context(
        self,
        query: str,
        context_window: int = 1,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results with surrounding context chunks.

        Args:
            query: Query text
            context_window: Number of chunks before/after to include
            top_k: Number of results

        Returns:
            List of results with context
        """
        # Get initial results
        results = self.retrieve(query, top_k=top_k, return_scores=True)
        
        if context_window == 0:
            return results
        
        # TODO: Implement context window expansion
        # This would require chunk IDs to be sequential or have position info
        logger.warning("Context window expansion not yet implemented")
        return results

    def retrieve_multi_modal(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using multi-modal query (text + image).

        Args:
            text_query: Text query
            image_path: Path to query image
            text_weight: Weight for text embedding
            image_weight: Weight for image embedding
            top_k: Number of results

        Returns:
            List of retrieved results
        """
        if not text_query and not image_path:
            logger.error("Must provide either text_query or image_path")
            return []
        
        # Generate hybrid embedding
        query_embedding = self.embedding_generator.generate_hybrid_embeddings(
            text=text_query or "",
            image_path=image_path,
            text_weight=text_weight,
            image_weight=image_weight
        )
        
        # Search
        k = top_k or self.top_k
        indices, distances, metadata_list = self.vector_store.search(query_embedding, k=k)
        
        # Format results
        results = []
        for idx, distance, metadata in zip(indices, distances, metadata_list):
            similarity_score = 1.0 / (1.0 + distance)
            
            result = {
                "text": metadata.get("text", ""),
                "similarity_score": similarity_score,
                "distance": distance,
                "metadata": metadata if self.include_metadata else {}
            }
            results.append(result)
        
        logger.info(f"Multi-modal retrieval: {len(results)} results")
        return results

    def get_similar_chunks(
        self,
        chunk_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of results

        Returns:
            List of similar chunks
        """
        k = top_k or self.top_k
        
        # Search by chunk ID
        indices, distances, metadata_list = self.vector_store.search_by_id(chunk_id, k=k + 1)
        
        # Format results (skip first result which is the chunk itself)
        results = []
        for idx, distance, metadata in zip(indices[1:], distances[1:], metadata_list[1:]):
            similarity_score = 1.0 / (1.0 + distance)
            
            result = {
                "text": metadata.get("text", ""),
                "similarity_score": similarity_score,
                "metadata": metadata if self.include_metadata else {}
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} similar chunks")
        return results

    def filter_by_source(
        self,
        query: str,
        source: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results filtered by source document.

        Args:
            query: Query text
            source: Source document identifier
            top_k: Number of results

        Returns:
            Filtered results
        """
        return self.retrieve(
            query,
            top_k=top_k,
            filters={"source": source}
        )

    def filter_by_metadata(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results with custom metadata filters.

        Args:
            query: Query text
            metadata_filters: Dictionary of metadata key-value pairs
            top_k: Number of results

        Returns:
            Filtered results
        """
        return self.retrieve(
            query,
            top_k=top_k,
            filters=metadata_filters
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.stats.copy()
        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()
        return stats

    def reset_stats(self):
        """Reset retrieval statistics."""
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "average_latency_ms": 0.0
        }
        logger.info("Statistics reset")


def retrieve_documents(
    query: str,
    vector_store_path: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for quick retrieval.

    Args:
        query: Query text
        vector_store_path: Path to vector store
        top_k: Number of results

    Returns:
        Retrieved documents
    """
    retriever = Retriever(vector_store_path=vector_store_path, top_k=top_k)
    return retriever.retrieve(query)
