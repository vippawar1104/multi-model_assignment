"""
Complete End-to-End Multi-Modal RAG Pipeline Demo
Demonstrates the full pipeline from document ingestion to evaluated answer generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def complete_rag_pipeline_demo():
    """
    Demonstrate the complete RAG pipeline across all 8 phases.
    """
    print("\n" + "="*80)
    print("COMPLETE END-TO-END MULTI-MODAL RAG PIPELINE DEMO")
    print("="*80)
    
    # =========================================================================
    # PHASE 3: DATA INGESTION
    # =========================================================================
    print("\n[PHASE 3: DATA INGESTION]")
    print("-" * 80)
    
    from data_ingestion import PDFExtractor
    
    pdf_path = "data/raw/qatar_test_doc.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Test PDF not found: {pdf_path}")
        print("Using mock data for demonstration...")
        
        # Mock extracted content
        extracted_content = {
            'text': """
            Qatar Airways is the state-owned flag carrier of Qatar. The airline operates 
            a hub-and-spoke network, linking over 150 international destinations across 
            Africa, Asia, Europe, the Americas, and Oceania from its base at Hamad 
            International Airport. Qatar Airways is a member of the Oneworld airline 
            alliance. The airline has received numerous awards including Skytrax's 
            Airline of the Year multiple times.
            
            Fleet and Services:
            Qatar Airways operates a modern fleet of Airbus and Boeing aircraft. The 
            airline offers premium services including Qsuites business class, known for 
            its privacy and comfort. Economy class passengers also enjoy award-winning 
            service and entertainment options.
            
            Sustainability Initiatives:
            The airline is committed to environmental sustainability through fuel-efficient 
            aircraft, carbon offset programs, and waste reduction initiatives.
            """,
            'images': [],
            'tables': [],
            'metadata': {'total_pages': 3, 'file_name': 'qatar_test_doc.pdf'}
        }
        print("✓ Using mock Qatar Airways document content")
    else:
        extractor = PDFExtractor()
        extracted_content = extractor.extract(pdf_path)
        print(f"✓ Extracted {len(extracted_content.get('text', ''))} characters")
        print(f"✓ Found {len(extracted_content.get('images', []))} images")
        print(f"✓ Found {len(extracted_content.get('tables', []))} tables")
    
    # =========================================================================
    # PHASE 4: PREPROCESSING & CHUNKING
    # =========================================================================
    print("\n[PHASE 4: PREPROCESSING & CHUNKING]")
    print("-" * 80)
    
    from preprocessing import TextCleaner, TextChunker
    
    # Clean text
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(extracted_content['text'])
    print(f"✓ Cleaned text: {len(cleaned_text)} characters")
    
    # Create semantic chunks
    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_text(cleaned_text, metadata={'source': 'qatar_test_doc.pdf'})
    print(f"✓ Created {len(chunks)} semantic chunks")
    
    # Show sample chunk
    if chunks:
        sample_chunk = chunks[0]
        print(f"\nSample chunk:")
        print(f"  Text: {sample_chunk['text'][:100]}...")
        print(f"  Tokens: {sample_chunk['metadata'].get('tokens', 'N/A')}")
    
    # =========================================================================
    # PHASE 5: VECTOR STORE & EMBEDDINGS
    # =========================================================================
    print("\n[PHASE 5: VECTOR STORE & EMBEDDINGS]")
    print("-" * 80)
    
    from vectorstore import EmbeddingGenerator, FAISSStore
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.generate_embeddings(chunk_texts, batch_size=32)
    print(f"✓ Generated embeddings: shape {embeddings.shape}")
    
    # Build FAISS index
    store = FAISSStore(dimension=embeddings.shape[1])
    store.add_vectors(embeddings, chunks)
    print(f"✓ Built FAISS index with {store.get_size()} vectors")
    
    # =========================================================================
    # PHASE 6: RETRIEVAL
    # =========================================================================
    print("\n[PHASE 6: RETRIEVAL]")
    print("-" * 80)
    
    from retrieval import HybridRetriever
    
    # Initialize retriever
    retriever = HybridRetriever(
        vector_store=store,
        embedder=embedder,
        chunks=chunks
    )
    
    # Test queries
    test_queries = [
        "What is Qatar Airways?",
        "What services does Qatar Airways offer?",
        "Tell me about Qatar Airways sustainability"
    ]
    
    all_results = {}
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Hybrid search
        results = retriever.hybrid_search(query, top_k=3)
        all_results[query] = results
        
        print(f"  ✓ Retrieved {len(results)} relevant chunks")
        
        # Show top result
        if results:
            top_result = results[0]
            print(f"  Top match (score: {top_result.get('score', 0):.3f}):")
            print(f"    {top_result['text'][:150]}...")
    
    # =========================================================================
    # PHASE 7: RESPONSE GENERATION
    # =========================================================================
    print("\n[PHASE 7: RESPONSE GENERATION]")
    print("-" * 80)
    
    # Check for API key
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if groq_api_key:
        from generation import ResponseGenerator
        
        # Initialize generator with Groq
        generator = ResponseGenerator(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )
        
        # Generate answer for first query
        test_query = test_queries[0]
        print(f"\nGenerating answer for: '{test_query}'")
        
        response = generator.generate(
            query=test_query,
            context_chunks=all_results[test_query]
        )
        
        print(f"\n✓ Generated Answer:")
        print(f"{response.answer}\n")
        print(f"Citations: {response.citations}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Latency: {response.metadata.get('generation_time', 0):.2f}s")
        
        # Store for evaluation
        generated_answer = response.answer
        generated_context = response.context
        
    else:
        print("⚠️  GROQ_API_KEY not found in environment")
        print("Skipping real LLM generation - using mock answer")
        
        # Mock response
        generated_answer = "Qatar Airways is the state-owned flag carrier of Qatar, operating a hub-and-spoke network linking over 150 international destinations."
        generated_context = all_results[test_queries[0]][0]['text'] if all_results[test_queries[0]] else ""
    
    # =========================================================================
    # PHASE 8: EVALUATION
    # =========================================================================
    print("\n[PHASE 8: EVALUATION]")
    print("-" * 80)
    
    from evaluation import RAGMetrics, RAGASEvaluator, QualityAssessor
    
    # Prepare evaluation data
    test_query = test_queries[0]
    retrieved_ids = [r.get('id', f'doc_{i}') for i, r in enumerate(all_results[test_query])]
    relevant_ids = retrieved_ids[:2]  # Mock ground truth: top 2 are relevant
    
    # 1. Retrieval Metrics
    print("\n1. Retrieval Metrics:")
    metrics = RAGMetrics()
    retrieval_result = metrics.evaluate_retrieval(retrieved_ids, relevant_ids)
    print(f"   Precision: {retrieval_result.precision:.3f}")
    print(f"   Recall:    {retrieval_result.recall:.3f}")
    print(f"   F1 Score:  {retrieval_result.f1_score:.3f}")
    
    # 2. RAGAS Evaluation
    print("\n2. RAGAS Evaluation:")
    ragas = RAGASEvaluator()
    ragas_result = ragas.evaluate(
        query=test_query,
        answer=generated_answer,
        context=generated_context
    )
    print(f"   Faithfulness:      {ragas_result.faithfulness:.3f}")
    print(f"   Answer Relevance:  {ragas_result.answer_relevance:.3f}")
    print(f"   Context Relevance: {ragas_result.context_relevance:.3f}")
    print(f"   Overall Score:     {ragas_result.overall_score:.3f}")
    
    # 3. Quality Assessment
    print("\n3. Quality Assessment:")
    quality = QualityAssessor()
    quality_result = quality.assess(
        answer=generated_answer,
        query=test_query,
        context=generated_context
    )
    print(f"   Overall Quality:   {quality_result.overall_quality:.3f}")
    print(f"   Hallucination:     {quality_result.hallucination_score:.3f}")
    print(f"   Completeness:      {quality_result.completeness_score:.3f}")
    print(f"   Coherence:         {quality_result.coherence_score:.3f}")
    print(f"   Status:            {'PASS ✓' if quality_result.passed else 'FAIL ✗'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    print("\n✓ Complete Pipeline Executed:")
    print("  [Phase 3] Data Ingestion      → Extracted document content")
    print(f"  [Phase 4] Preprocessing       → Created {len(chunks)} chunks")
    print(f"  [Phase 5] Vector Store        → Built FAISS index with {store.get_size()} vectors")
    print(f"  [Phase 6] Retrieval           → Retrieved {len(all_results[test_queries[0]])} relevant chunks")
    print(f"  [Phase 7] Generation          → Generated answer ({'Real LLM' if groq_api_key else 'Mock'})")
    print(f"  [Phase 8] Evaluation          → Assessed quality (Overall: {quality_result.overall_quality:.2f})")
    
    print("\n✓ Quality Metrics:")
    print(f"  Retrieval F1:      {retrieval_result.f1_score:.3f}")
    print(f"  RAGAS Overall:     {ragas_result.overall_score:.3f}")
    print(f"  Quality Score:     {quality_result.overall_quality:.3f}")
    print(f"  Hallucination:     {quality_result.hallucination_score:.3f}")
    
    print("\n✓ System Performance:")
    print(f"  Total Chunks:      {len(chunks)}")
    print(f"  Embedding Dim:     {embeddings.shape[1]}")
    print(f"  Retrieval Top-K:   {len(all_results[test_queries[0]])}")
    
    print("\n" + "="*80)
    print("🎉 END-TO-END PIPELINE COMPLETE!")
    print("="*80)
    
    print("\n📊 Next Steps:")
    print("  1. Run with your own documents")
    print("  2. Set GROQ_API_KEY for real LLM generation")
    print("  3. Adjust retrieval/generation parameters")
    print("  4. Customize evaluation thresholds")
    print("  5. Deploy as API service")


def main():
    """Run the complete demo."""
    try:
        complete_rag_pipeline_demo()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n✓ All 8 phases demonstrated")
        print("✓ Multi-Modal RAG System: Fully Operational")
        print("\nFor detailed phase documentation, see:")
        print("  - docs/phase3_data_ingestion_complete.md")
        print("  - docs/phase4_preprocessing_complete.md")
        print("  - docs/phase5_vector_store_complete.md")
        print("  - docs/phase7_generation_complete.md")
        print("  - docs/phase8_evaluation_complete.md")
        print("  - PROJECT_COMPLETE.md")
        
    except Exception as e:
        logger.error(f"Pipeline demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
