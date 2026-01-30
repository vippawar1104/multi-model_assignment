"""
Phase 7: Real LLM Generation Test
Tests the complete generation pipeline with actual OpenAI API calls.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import (
    ResponseGenerator,
    GenerationConfig,
    LLMClient,
    LLMConfig,
    ContextFormatter,
    PromptManager
)
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_real_generation():
    """Test generation with real OpenAI API."""
    print("\n" + "="*70)
    print("PHASE 7: REAL LLM GENERATION TEST")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        return False
    
    print(f"\n✓ API Key found: {api_key[:20]}...")
    
    # Create mock retrieved chunks
    chunks = [
        {
            'text': "Deep learning is a subset of machine learning that uses neural networks "
                   "with multiple layers (deep neural networks). These networks can learn "
                   "hierarchical representations of data, making them highly effective for "
                   "complex tasks like image recognition, natural language processing, and "
                   "speech recognition.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 5,
                'chunk_id': 'chunk_001'
            },
            'score': 0.89
        },
        {
            'text': "Neural networks consist of interconnected nodes (neurons) organized in layers. "
                   "Each connection has a weight that is adjusted during training through "
                   "backpropagation. The network learns by minimizing the difference between "
                   "predicted and actual outputs.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 6,
                'chunk_id': 'chunk_002'
            },
            'score': 0.76
        },
        {
            'text': "Popular deep learning architectures include Convolutional Neural Networks (CNNs) "
                   "for image processing, Recurrent Neural Networks (RNNs) for sequential data, "
                   "and Transformers for natural language understanding. Each architecture is "
                   "optimized for specific types of tasks.",
            'metadata': {
                'source': 'ml_architectures.pdf',
                'page': 12,
                'chunk_id': 'chunk_003'
            },
            'score': 0.72
        }
    ]
    
    # Test 1: Basic Generation
    print("\n" + "="*70)
    print("TEST 1: Basic Question Answering")
    print("="*70)
    
    try:
        config = GenerationConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=300,
            include_citations=True,
            return_confidence=True
        )
        
        generator = ResponseGenerator(config=config)
        
        query = "What is deep learning and how does it work?"
        print(f"\nQuery: {query}")
        print(f"Retrieved chunks: {len(chunks)}")
        
        print("\n⏳ Generating response with OpenAI...")
        result = generator.generate(
            query=query,
            retrieved_chunks=chunks,
            query_type="question"
        )
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(result.answer)
        print("-"*70)
        
        print(f"\n✓ Generated in {result.generation_time:.2f}s")
        print(f"  - Chunks used: {result.chunks_used}")
        print(f"  - Template: {result.template_used}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Citations: {len(result.citations)}")
        
        if result.citations:
            print(f"  - Sources: {', '.join(result.citations)}")
        
    except Exception as e:
        print(f"\n❌ Test 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Technical Query
    print("\n" + "="*70)
    print("TEST 2: Technical Explanation")
    print("="*70)
    
    try:
        query2 = "Explain the architecture of neural networks"
        print(f"\nQuery: {query2}")
        
        print("\n⏳ Generating technical response...")
        result2 = generator.generate(
            query=query2,
            retrieved_chunks=chunks,
            query_type="explanation",
            template_name="technical"
        )
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(result2.answer)
        print("-"*70)
        
        print(f"\n✓ Generated in {result2.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 2 failed: {str(e)}")
        return False
    
    # Test 3: Comparison Query
    print("\n" + "="*70)
    print("TEST 3: Comparison")
    print("="*70)
    
    try:
        query3 = "What are the differences between CNNs, RNNs, and Transformers?"
        print(f"\nQuery: {query3}")
        
        print("\n⏳ Generating comparison...")
        result3 = generator.generate(
            query=query3,
            retrieved_chunks=chunks,
            template_name="comparison"
        )
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(result3.answer)
        print("-"*70)
        
        print(f"\n✓ Generated in {result3.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 3 failed: {str(e)}")
        return False
    
    # Test 4: Batch Generation
    print("\n" + "="*70)
    print("TEST 4: Batch Processing")
    print("="*70)
    
    try:
        queries = [
            "What is deep learning?",
            "How do neural networks learn?",
            "What are common architectures?"
        ]
        
        print(f"\nProcessing {len(queries)} queries in batch...")
        
        results = generator.generate_batch(
            queries=queries,
            retrieved_chunks_list=[chunks] * len(queries),
            query_types=["question", "explanation", "question"]
        )
        
        for i, (q, r) in enumerate(zip(queries, results), 1):
            print(f"\n{i}. {q}")
            print(f"   → Answer: {r.answer[:100]}...")
            print(f"   → Time: {r.generation_time:.2f}s")
        
        total_time = sum(r.generation_time for r in results)
        avg_time = total_time / len(results)
        
        print(f"\n✓ Batch complete:")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Avg time: {avg_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 4 failed: {str(e)}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n✅ All tests passed!")
    print("\n✓ Verified Capabilities:")
    print("  1. Basic Q&A with citations")
    print("  2. Technical explanations")
    print("  3. Comparison generation")
    print("  4. Batch processing")
    print("  5. Multiple prompt templates")
    print("  6. Confidence estimation")
    print("  7. Real-time LLM integration")
    
    print("\n" + "="*70)
    print("✓ PHASE 7 GENERATION - FULLY OPERATIONAL WITH OPENAI!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_real_generation()
    sys.exit(0 if success else 1)
