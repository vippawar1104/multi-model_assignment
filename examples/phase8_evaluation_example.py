"""
Phase 8: Evaluation & Metrics - Complete Demo
Demonstrates all evaluation capabilities including RAGAS, quality assessment, and benchmarking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation import (
    RAGMetrics,
    MetricsConfig,
    RAGASEvaluator,
    RAGASConfig,
    QualityAssessor,
    QualityConfig,
    Benchmark,
    BenchmarkConfig
)
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def demo_retrieval_metrics():
    """Demonstrate retrieval metrics calculation."""
    print("\n" + "="*70)
    print("Demo 1: Retrieval Metrics")
    print("="*70)
    
    metrics = RAGMetrics()
    
    # Test data
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc6', 'doc7']
    
    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant:  {relevant}")
    
    # Calculate metrics
    precision = metrics.calculate_precision(retrieved, relevant)
    recall = metrics.calculate_recall(retrieved, relevant)
    f1 = metrics.calculate_f1(precision, recall)
    
    print(f"\n✓ Precision: {precision:.3f} (2 relevant / 5 retrieved)")
    print(f"✓ Recall:    {recall:.3f} (2 relevant / 4 total relevant)")
    print(f"✓ F1 Score:  {f1:.3f}")
    
    # MRR calculation
    retrieved_lists = [
        ['doc1', 'doc2', 'doc3'],
        ['doc4', 'doc1', 'doc5'],
        ['doc2', 'doc3', 'doc1']
    ]
    relevant_lists = [
        ['doc1'],
        ['doc1', 'doc5'],
        ['doc4']
    ]
    
    mrr = metrics.calculate_mrr(retrieved_lists, relevant_lists)
    print(f"✓ MRR:       {mrr:.3f}")
    
    # NDCG calculation
    relevance_scores = {
        'doc1': 1.0,
        'doc2': 0.5,
        'doc3': 0.8,
        'doc4': 0.3,
        'doc5': 0.1
    }
    ndcg = metrics.calculate_ndcg(retrieved, relevance_scores, k=5)
    print(f"✓ NDCG@5:    {ndcg:.3f}")


def demo_ragas_evaluation():
    """Demonstrate RAGAS evaluation."""
    print("\n" + "="*70)
    print("Demo 2: RAGAS Evaluation")
    print("="*70)
    
    evaluator = RAGASEvaluator()
    
    # Test case
    query = "What is deep learning?"
    context = "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn hierarchical representations."
    answer = "Deep learning is a type of machine learning using multi-layer neural networks to learn hierarchical data representations."
    
    print(f"\nQuery:   {query}")
    print(f"Context: {context[:80]}...")
    print(f"Answer:  {answer}")
    
    # Evaluate
    result = evaluator.evaluate(query, answer, context)
    
    print(f"\n✓ Faithfulness:      {result.faithfulness:.3f} ({'✓ PASS' if result.faithfulness >= 0.5 else '✗ FAIL'})")
    print(f"✓ Answer Relevance:  {result.answer_relevance:.3f} ({'✓ PASS' if result.answer_relevance >= 0.5 else '✗ FAIL'})")
    print(f"✓ Context Relevance: {result.context_relevance:.3f} ({'✓ PASS' if result.context_relevance >= 0.5 else '✗ FAIL'})")
    print(f"✓ Overall Score:     {result.overall_score:.3f}")
    
    print(f"\nStatus: {result.details.get('overall_status', 'N/A').upper()}")


def demo_quality_assessment():
    """Demonstrate quality assessment."""
    print("\n" + "="*70)
    print("Demo 3: Quality Assessment (Hallucination Detection)")
    print("="*70)
    
    assessor = QualityAssessor()
    
    # Good answer
    query = "What are neural networks?"
    context = "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes organized in layers."
    answer = "Neural networks are computing systems with interconnected nodes in layers, inspired by biological neural networks."
    
    print(f"\n[Test 1: Good Answer]")
    print(f"Query:   {query}")
    print(f"Answer:  {answer}")
    
    result = assessor.assess(answer, query, context)
    
    print(f"\n✓ Overall Quality:    {result.overall_quality:.3f}")
    print(f"✓ Hallucination:      {result.hallucination_score:.3f}")
    print(f"✓ Completeness:       {result.completeness_score:.3f}")
    print(f"✓ Coherence:          {result.coherence_score:.3f}")
    print(f"✓ Passed:             {result.passed}")
    
    if result.issues:
        print(f"\nIssues: {result.issues}")
    else:
        print("\nNo issues detected ✓")
    
    # Hallucinated answer
    hallucinated_answer = "Neural networks were invented in 2020 and are primarily used for cryptocurrency mining."
    
    print(f"\n[Test 2: Hallucinated Answer]")
    print(f"Answer:  {hallucinated_answer}")
    
    result2 = assessor.assess(hallucinated_answer, query, context)
    
    print(f"\n✓ Overall Quality:    {result2.overall_quality:.3f}")
    print(f"✓ Hallucination:      {result2.hallucination_score:.3f}")
    print(f"✓ Passed:             {result2.passed}")
    
    if result2.issues:
        print(f"\nIssues detected:")
        for issue in result2.issues:
            print(f"  - {issue}")


def demo_benchmark():
    """Demonstrate benchmarking."""
    print("\n" + "="*70)
    print("Demo 4: Benchmarking")
    print("="*70)
    
    benchmark = Benchmark(BenchmarkConfig(
        num_warmup_runs=1,
        num_test_runs=5
    ))
    
    # Mock RAG function
    import time
    import random
    
    def mock_rag_pipeline(query: str):
        """Simulate RAG pipeline."""
        time.sleep(random.uniform(0.1, 0.3))  # Simulate processing
        return f"Answer to: {query}"
    
    # Test cases
    test_cases = [
        {'query': 'What is AI?'},
        {'query': 'Explain machine learning'},
        {'query': 'What are neural networks?'},
        {'query': 'How does deep learning work?'},
        {'query': 'What is NLP?'}
    ]
    
    print(f"\nRunning benchmark with {len(test_cases)} test cases...")
    
    result = benchmark.run_benchmark(mock_rag_pipeline, test_cases)
    
    print(f"\n✓ Benchmark Results:")
    print(f"  Total Queries:     {result.total_queries}")
    print(f"  Successful:        {result.successful_queries}")
    print(f"  Failed:            {result.failed_queries}")
    print(f"  Avg Latency:       {result.avg_latency:.3f}s")
    print(f"  Min Latency:       {result.min_latency:.3f}s")
    print(f"  Max Latency:       {result.max_latency:.3f}s")
    print(f"  Throughput:        {result.throughput:.2f} queries/sec")
    print(f"  Success Rate:      {result.details['success_rate']:.1%}")
    print(f"  Latency P50:       {result.details['latency_p50']:.3f}s")
    print(f"  Latency P95:       {result.details['latency_p95']:.3f}s")
    print(f"  Latency P99:       {result.details['latency_p99']:.3f}s")


def demo_end_to_end_evaluation():
    """Demonstrate complete end-to-end evaluation."""
    print("\n" + "="*70)
    print("Demo 5: End-to-End RAG Pipeline Evaluation")
    print("="*70)
    
    metrics = RAGMetrics()
    
    # Simulated RAG pipeline results
    query = "What is machine learning?"
    retrieved_docs = ['doc1', 'doc2', 'doc3']
    relevant_docs = ['doc1', 'doc2', 'doc4']
    context = "Machine learning is a field of AI that enables computers to learn from data without explicit programming."
    answer = "Machine learning is an AI field where computers learn from data automatically without being explicitly programmed."
    citations = ['doc1', 'doc2']
    context_sources = ['doc1', 'doc2', 'doc3']
    latency = 1.5
    
    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(retrieved_docs)} docs")
    print(f"Answer: {answer}")
    
    # End-to-end evaluation
    result = metrics.evaluate_e2e(
        retrieved=retrieved_docs,
        relevant=relevant_docs,
        answer=answer,
        query=query,
        context=context,
        latency=latency,
        citations=citations,
        context_sources=context_sources
    )
    
    print(f"\n✓ Complete Metrics:")
    print(f"\n  Retrieval:")
    print(f"    Precision:        {result.precision:.3f}")
    print(f"    Recall:           {result.recall:.3f}")
    print(f"    F1 Score:         {result.f1_score:.3f}")
    print(f"    MRR:              {result.mrr:.3f}")
    
    print(f"\n  Generation:")
    print(f"    Relevance:        {result.answer_relevance:.3f}")
    print(f"    Completeness:     {result.answer_completeness:.3f}")
    print(f"    Citation Acc:     {result.citation_accuracy:.3f}")
    
    print(f"\n  Performance:")
    print(f"    Latency:          {result.latency:.2f}s")
    print(f"    Throughput:       {result.throughput:.2f} qps")


def main():
    """Run all evaluation demos."""
    print("\n" + "="*70)
    print("PHASE 8: EVALUATION & METRICS - COMPLETE DEMO")
    print("="*70)
    
    try:
        # Run all demos
        demo_retrieval_metrics()
        demo_ragas_evaluation()
        demo_quality_assessment()
        demo_benchmark()
        demo_end_to_end_evaluation()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print("\n✓ Phase 8 Evaluation Components:")
        print("  1. RAGMetrics - Retrieval & generation metrics")
        print("  2. RAGASEvaluator - Advanced RAGAS evaluation")
        print("  3. QualityAssessor - Hallucination detection & quality")
        print("  4. Benchmark - Performance benchmarking")
        
        print("\n✓ Metrics Implemented:")
        print("  • Precision, Recall, F1 Score")
        print("  • MRR (Mean Reciprocal Rank)")
        print("  • NDCG (Normalized Discounted Cumulative Gain)")
        print("  • MAP (Mean Average Precision)")
        print("  • Faithfulness (groundedness)")
        print("  • Answer & Context Relevance")
        print("  • Hallucination Detection")
        print("  • Quality & Completeness")
        print("  • Latency & Throughput")
        
        print("\n✓ Use Cases:")
        print("  • Evaluate retrieval quality")
        print("  • Detect hallucinations in answers")
        print("  • Assess answer relevance and completeness")
        print("  • Benchmark system performance")
        print("  • Compare different RAG configurations")
        
        print("\n" + "="*70)
        print("✓ Phase 8 Evaluation & Metrics Pipeline Complete!")
        print("="*70)
        
        print("\n🎉 ALL 8 PHASES COMPLETE!")
        print("📊 Multi-Modal RAG System: 100% Implemented")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
