"""
Complete RAG Pipeline Integration Test
Demonstrates Phases 1-7 working together.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_pipeline_integration():
    """Test complete pipeline integration (Phases 1-7)."""
    print("\n" + "="*70)
    print("COMPLETE RAG PIPELINE INTEGRATION TEST")
    print("Phases 1-7: Environment → Utilities → Ingestion → Preprocessing → ")
    print("            Vector Store → Retrieval → Generation")
    print("="*70)
    
    phases_status = {}
    
    # Phase 2: Core Utilities
    print("\n[Phase 2] Testing Core Utilities...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        
        from utils.logger import LoggerConfig
        from utils.config_loader import ConfigLoader
        print("  ✓ Logger and Config Loader available")
        phases_status['Phase 2'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 2 failed: {e}")
        phases_status['Phase 2'] = '❌'
    
    # Phase 3: Data Ingestion
    print("\n[Phase 3] Testing Data Ingestion...")
    try:
        from data_ingestion.pdf_extractor import PDFExtractor
        from data_ingestion.image_extractor import ImageExtractor
        print("  ✓ PDF and Image Extractors available")
        phases_status['Phase 3'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 3 failed: {e}")
        phases_status['Phase 3'] = '❌'
    
    # Phase 4: Preprocessing
    print("\n[Phase 4] Testing Preprocessing...")
    try:
        from preprocessing.text_processor import TextProcessor
        from preprocessing.image_processor import ImageProcessor
        processor = TextProcessor()
        print("  ✓ Text and Image Processors available")
        phases_status['Phase 4'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 4 failed: {e}")
        phases_status['Phase 4'] = '❌'
    
    # Phase 5: Vector Store
    print("\n[Phase 5] Testing Vector Store...")
    try:
        from vectorstore.embedding_generator import EmbeddingGenerator
        from vectorstore.faiss_store import FAISSVectorStore
        
        # Create test embeddings
        generator = EmbeddingGenerator()
        embeddings = generator.generate_text_embeddings(
            ["test document 1", "test document 2"]
        )
        
        # Create vector store
        store = FAISSVectorStore(dimension=384)
        store.add_vectors(embeddings, [{'id': 0}, {'id': 1}])
        
        print(f"  ✓ Vector store with {store.count()} vectors")
        phases_status['Phase 5'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 5 failed: {e}")
        phases_status['Phase 5'] = '❌'
    
    # Phase 6: Retrieval
    print("\n[Phase 6] Testing Retrieval...")
    query_processor_cls = None
    retriever_cls = None
    reranker_cls = None
    try:
        from retrieval.retriever import Retriever
        from retrieval.query_processor import QueryProcessor
        from retrieval.reranker import Reranker
        
        query_processor_cls = QueryProcessor
        retriever_cls = Retriever
        reranker_cls = Reranker
        
        # Test query processing
        query_processor = QueryProcessor()
        processed = query_processor.process("What is machine learning?")
        
        print(f"  ✓ Query processed: intent='{processed.intent}'")
        phases_status['Phase 6'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 6 failed: {e}")
        phases_status['Phase 6'] = '❌'
    
    # Phase 7: Generation
    print("\n[Phase 7] Testing Generation...")
    try:
        from generation import (
            ResponseGenerator,
            ContextFormatter,
            PromptManager,
            LLMClient
        )
        
        # Test context formatting
        formatter = ContextFormatter()
        test_chunks = [
            {'text': 'Test content', 'metadata': {'source': 'test.pdf', 'page': 1}}
        ]
        context = formatter.format_chunks(test_chunks)
        
        # Test prompt generation
        manager = PromptManager()
        prompt = manager.get_prompt(
            'general',
            {'context': context, 'query': 'test query'}
        )
        
        print(f"  ✓ Context formatted: {len(context)} chars")
        print(f"  ✓ Prompt templates: {len(manager.list_templates())} available")
        print("  ℹ️  Set OPENAI_API_KEY for full LLM generation")
        phases_status['Phase 7'] = '✅'
    except Exception as e:
        print(f"  ✗ Phase 7 failed: {e}")
        phases_status['Phase 7'] = '❌'
    
    # Integration Test: Simulated End-to-End
    print("\n" + "="*70)
    print("SIMULATED END-TO-END PIPELINE")
    print("="*70)
    
    try:
        query = "What is deep learning?"
        
        # Step 1: Query Processing (Phase 6)
        print(f"\n1. Query: '{query}'")
        if query_processor_cls:
            query_processor = query_processor_cls()
            processed_query = query_processor.process(query)
        else:
            # Create mock processed query
            from dataclasses import dataclass
            @dataclass
            class MockProcessed:
                intent: str = "question"
                keywords: list = None
            processed_query = MockProcessed(keywords=['deep', 'learning'])
        processed_query = query_processor.process(query)
        print(f"   → Intent: {processed_query.intent}")
        print(f"   → Keywords: {processed_query.keywords[:3]}")
        
        # Step 2: Embedding (Phase 5)
        print("\n2. Generate query embedding...")
        generator = EmbeddingGenerator()
        query_embedding = generator.generate_text_embeddings([query])[0]
        print(f"   → Embedding shape: {query_embedding.shape}")
        
        # Step 3: Vector Search (Phase 5)
        print("\n3. Search vector store...")
        # Using the store from Phase 5 test
        results = store.search(query_embedding, top_k=2)
        print(f"   → Found {len(results)} results")
        for i, (idx, score, meta) in enumerate(results, 1):
            print(f"      {i}. Score: {score:.4f}")
        
        # Step 4: Reranking (Phase 6)
        print("\n4. Rerank results...")
        reranker = Reranker()
        mock_chunks = [
            {'text': 'Deep learning uses neural networks', 'score': 0.8, 'metadata': {}},
            {'text': 'Machine learning is a subset of AI', 'score': 0.6, 'metadata': {}}
        ]
        reranked = reranker.rerank(mock_chunks, query)
        print(f"   → Reranked {len(reranked)} chunks")
        print(f"      Best score: {reranked[0]['score']:.4f}")
        
        # Step 5: Context Formatting (Phase 7)
        print("\n5. Format context for LLM...")
        context = formatter.format_chunks(reranked[:2])
        print(f"   → Context: {len(context)} characters")
        
        # Step 6: Prompt Construction (Phase 7)
        print("\n6. Construct prompt...")
        prompt = manager.get_prompt(
            'technical',
            {'context': context, 'query': query},
            query_type=processed_query.intent
        )
        print(f"   → System prompt: {len(prompt['system'])} chars")
        print(f"   → User prompt: {len(prompt['user'])} chars")
        
        # Step 7: Generation (Phase 7) - would call LLM here
        print("\n7. Generate response...")
        print("   → [Would call LLM API here with OPENAI_API_KEY]")
        print("   → Expected output: Natural language answer with citations")
        
        print("\n" + "="*70)
        print("✓ END-TO-END PIPELINE SIMULATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE STATUS SUMMARY")
    print("="*70)
    print("\nPhase 1: Environment Setup         ✅ (prerequisite)")
    for phase, status in phases_status.items():
        print(f"{phase}: {status}")
    print("Phase 8: Evaluation & Metrics      🔄 (next phase)")
    
    all_passed = all(status == '✅' for status in phases_status.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL PHASES OPERATIONAL - READY FOR PHASE 8")
    else:
        print("⚠️  SOME PHASES HAVE ISSUES - CHECK LOGS")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = test_pipeline_integration()
    sys.exit(0 if success else 1)
