"""
Example: Complete Phase 6 Retrieval Pipeline
Demonstrates query processing, retrieval, and reranking with the vector store.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import Retriever, QueryProcessor, Reranker
from src.vectorstore import FAISSVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run complete retrieval pipeline example."""
    
    logger.info("=" * 80)
    logger.info("Phase 6: Retrieval Pipeline Example")
    logger.info("=" * 80)
    
    # Step 1: Initialize components
    logger.info("\nStep 1: Initializing retrieval components...")
    
    # Load vector store from Phase 5
    vector_store_path = Path("data/vector_store")
    if not vector_store_path.exists():
        logger.error(f"Vector store not found at {vector_store_path}")
        logger.error("Please run Phase 5 example first to create the vector store")
        return
    
    store = FAISSVectorStore(dimension=384)
    store.load(vector_store_path)
    logger.info(f"Loaded vector store with {len(store.metadata)} vectors")
    
    # Initialize query processor
    processor = QueryProcessor(expand_queries=True, remove_stopwords=False)
    logger.info("Query processor initialized")
    
    # Initialize retriever
    retriever = Retriever(vector_store=store, top_k=5)
    logger.info("Retriever initialized")
    
    # Initialize reranker
    reranker = Reranker(strategy="hybrid", diversity_penalty=0.3)
    logger.info("Reranker initialized")
    
    # Step 2: Test queries
    logger.info("\nStep 2: Testing different query types...")
    
    queries = [
        "What is deep learning and how does it work?",
        "Explain vector embeddings",
        "Tell me about RAG systems",
        "How does FAISS work?",
        "Compare ML and DL"
    ]
    
    for i, raw_query in enumerate(queries, 1):
        logger.info(f"\n--- Query {i}: '{raw_query}' ---")
        
        # Process query
        processed_query = processor.process(raw_query)
        logger.info(f"Processed query: '{processed_query}'")
        
        # Detect intent
        intent = processor.detect_intent(raw_query)
        logger.info(f"Detected intent: {intent}")
        
        # Extract keywords
        keywords = processor.extract_keywords(raw_query, top_k=3)
        logger.info(f"Key terms: {keywords}")
        
        # Retrieve results
        results = retriever.retrieve(processed_query, top_k=5)
        logger.info(f"Retrieved {len(results)} initial results")
        
        # Rerank results
        reranked = reranker.rerank(processed_query, results, top_k=3)
        logger.info(f"Reranked to top {len(reranked)} results:")
        
        for j, res in enumerate(reranked, 1):
            score = res.get('rerank_score', res.get('similarity_score', 0))
            source = res.get('metadata', {}).get('source', 'unknown')
            text = res.get('text', '')[:80]
            logger.info(f"  {j}. [{source}] Score: {score:.4f}")
            logger.info(f"     {text}...")
    
    # Step 3: Test filtering
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Testing filtered retrieval...")
    
    query = "machine learning"
    logger.info(f"\nQuery: '{query}'")
    
    # Filter by source
    logger.info("\nFiltering by source='ml_intro.pdf':")
    filtered_results = retriever.filter_by_source(query, source="ml_intro.pdf", top_k=3)
    for res in filtered_results:
        logger.info(f"  - {res.get('metadata', {}).get('source', 'unknown')}: {res['text'][:60]}...")
    
    # Step 4: Test metadata-based boosting
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Testing result boosting...")
    
    query = "What is deep learning?"
    results = retriever.retrieve(query, top_k=5)
    
    # Boost results from specific sources
    boost_rules = {
        "source": 1.5  # Boost any result with a source field
    }
    boosted = reranker.boost_by_metadata(results, boost_rules)
    
    logger.info(f"\nOriginal vs Boosted scores:")
    for i, (orig, boost) in enumerate(zip(results, boosted), 1):
        orig_score = orig.get('similarity_score', 0)
        boost_score = boost.get('boosted_score', 0)
        logger.info(f"  {i}. Original: {orig_score:.4f} -> Boosted: {boost_score:.4f}")
    
    # Step 5: Test deduplication
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Testing deduplication...")
    
    # Simulate duplicate results
    query = "vector embeddings"
    results = retriever.retrieve(query, top_k=5)
    logger.info(f"\nBefore deduplication: {len(results)} results")
    
    deduplicated = reranker.deduplicate(results, similarity_threshold=0.9)
    logger.info(f"After deduplication: {len(deduplicated)} results")
    
    # Step 6: Test query variations
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Testing query variations...")
    
    original_query = "deep learning"
    variations = processor.generate_variations(original_query, max_variations=3)
    
    logger.info(f"\nOriginal query: '{original_query}'")
    logger.info(f"Generated {len(variations)} variations:")
    for var in variations:
        logger.info(f"  - {var}")
    
    # Test each variation
    logger.info("\nRetrieving with each variation:")
    all_results = []
    for var in variations[:2]:  # Test first 2 variations
        results = retriever.retrieve(var, top_k=2)
        all_results.extend(results)
        logger.info(f"  '{var}': {len(results)} results")
    
    # Deduplicate combined results
    combined = reranker.deduplicate(all_results, similarity_threshold=0.8)
    logger.info(f"\nCombined and deduplicated: {len(combined)} unique results")
    
    # Step 7: Display retrieval statistics
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Retrieval statistics")
    
    stats = retriever.get_stats()
    logger.info(f"\nRetrieval Statistics:")
    logger.info(f"  Total queries: {stats['total_queries']}")
    logger.info(f"  Total results: {stats['total_results']}")
    logger.info(f"  Average latency: {stats['average_latency_ms']:.2f}ms")
    
    if 'vector_store' in stats:
        vs_stats = stats['vector_store']
        logger.info(f"\nVector Store:")
        logger.info(f"  Total vectors: {vs_stats.get('total_vectors', 0)}")
        logger.info(f"  Dimension: {vs_stats.get('dimension', 0)}")
        logger.info(f"  Index type: {vs_stats.get('index_type', 'unknown')}")
    
    # Step 8: Test batch retrieval
    logger.info("\n" + "=" * 80)
    logger.info("Step 8: Testing batch retrieval...")
    
    batch_queries = [
        "What is machine learning?",
        "Explain vector search",
        "How does RAG work?"
    ]
    
    logger.info(f"\nProcessing batch of {len(batch_queries)} queries:")
    batch_results = retriever.retrieve_batch(batch_queries, top_k=2)
    
    for query, results in zip(batch_queries, batch_results):
        logger.info(f"\n  Query: '{query}'")
        logger.info(f"  Results: {len(results)}")
        for res in results:
            logger.info(f"    - Score: {res['similarity_score']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Phase 6 Retrieval Pipeline Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
