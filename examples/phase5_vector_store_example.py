"""
Example: Complete Phase 5 Vector Store Pipeline
Demonstrates embedding generation and vector storage with real data.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import TextChunker
from src.vectorstore import EmbeddingGenerator, FAISSVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run complete vector store pipeline example."""
    
    logger.info("=" * 80)
    logger.info("Phase 5: Vector Store Pipeline Example")
    logger.info("=" * 80)
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence that focuses on "
                   "the development of algorithms and statistical models. These systems can learn "
                   "from and make predictions or decisions based on data.",
            "source": "ml_intro.pdf",
            "page": 1
        },
        {
            "id": "doc2",
            "text": "Deep learning is a specialized branch of machine learning that uses neural "
                   "networks with multiple layers. It has revolutionized fields like computer vision, "
                   "natural language processing, and speech recognition.",
            "source": "dl_guide.pdf",
            "page": 3
        },
        {
            "id": "doc3",
            "text": "Vector embeddings are numerical representations of text, images, or other data "
                   "in a high-dimensional space. They capture semantic meaning and enable similarity "
                   "search in AI applications.",
            "source": "embeddings.pdf",
            "page": 5
        },
        {
            "id": "doc4",
            "text": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search "
                   "and clustering of dense vectors. It can handle billions of vectors and provides "
                   "both CPU and GPU implementations.",
            "source": "faiss_docs.pdf",
            "page": 2
        },
        {
            "id": "doc5",
            "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with text "
                   "generation. It retrieves relevant documents from a vector store and uses them "
                   "to generate more accurate and informed responses.",
            "source": "rag_paper.pdf",
            "page": 1
        }
    ]
    
    # Step 1: Chunk documents
    logger.info("\nStep 1: Chunking documents...")
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk_text(doc["text"])
        # Add document metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{doc['id']}_chunk_{i}"
            chunk["doc_id"] = doc["id"]
            chunk["source"] = doc["source"]
            chunk["page"] = doc["page"]
            all_chunks.append(chunk)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Step 2: Generate embeddings
    logger.info("\nStep 2: Generating embeddings...")
    generator = EmbeddingGenerator()
    
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = generator.generate_text_embeddings(texts, show_progress=True)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    embedding_dim = embeddings.shape[1]
    
    # Step 3: Create vector store
    logger.info("\nStep 3: Creating vector store...")
    store = FAISSVectorStore(dimension=embedding_dim, index_type="flatl2")
    
    # Prepare metadata
    metadata = [
        {
            "text": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk.get("doc_id"),
            "source": chunk.get("source"),
            "page": chunk.get("page")
        }
        for chunk in all_chunks
    ]
    
    # Add vectors to store
    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
    store.add_vectors(embeddings, metadata=metadata, ids=chunk_ids)
    
    # Step 4: Test search
    logger.info("\nStep 4: Testing similarity search...")
    
    queries = [
        "What is deep learning?",
        "How do vector embeddings work?",
        "Tell me about RAG systems"
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n--- Query {i}: '{query}' ---")
        
        # Generate query embedding
        query_emb = generator.generate_text_embeddings(query)
        
        # Search
        indices, distances, results = store.search(query_emb, k=3)
        
        logger.info(f"Top 3 results:")
        for j, (idx, dist, meta) in enumerate(zip(indices, distances, results), 1):
            logger.info(f"  {j}. Distance: {dist:.4f}")
            logger.info(f"     Source: {meta['source']}, Page: {meta['page']}")
            logger.info(f"     Text: {meta['text'][:100]}...")
    
    # Step 5: Save vector store
    logger.info("\nStep 5: Saving vector store...")
    save_path = Path("data/vector_store")
    store.save(save_path)
    logger.info(f"Saved vector store to {save_path}")
    
    # Step 6: Load and test
    logger.info("\nStep 6: Loading vector store...")
    new_store = FAISSVectorStore(dimension=embedding_dim)
    new_store.load(save_path)
    
    # Test loaded store
    query_emb = generator.generate_text_embeddings("machine learning algorithms")
    indices, distances, results = new_store.search(query_emb, k=2)
    
    logger.info("Testing loaded store:")
    for idx, dist, meta in zip(indices, distances, results):
        logger.info(f"  Found: {meta['source']} (distance: {dist:.4f})")
    
    # Step 7: Show statistics
    logger.info("\nStep 7: Vector store statistics")
    stats = store.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Phase 5 Pipeline Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
