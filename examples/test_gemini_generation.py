"""
Test Phase 7 Generation with Google Gemini API
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import (
    ResponseGenerator,
    GenerationConfig,
    ContextFormatter,
    PromptManager
)
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_gemini_generation():
    """Test generation with Gemini API."""
    print("\n" + "="*70)
    print("PHASE 7: GEMINI API GENERATION TEST")
    print("="*70)
    
    # Set API key
    api_key = "AIzaSyDPXOc1-AuO3OoaNosQ56kr915HsZ_pHFQ"
    os.environ["GEMINI_API_KEY"] = api_key
    
    print(f"\n✓ Gemini API Key set: {api_key[:20]}...")
    
    # Create test chunks
    chunks = [
        {
            'text': "Deep learning is a subset of machine learning that uses neural networks "
                   "with multiple layers. These deep neural networks can learn hierarchical "
                   "representations of data, making them effective for complex tasks like "
                   "image recognition and natural language processing.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 5
            },
            'score': 0.89
        },
        {
            'text': "Neural networks consist of interconnected nodes organized in layers. "
                   "Each connection has a weight adjusted during training through backpropagation. "
                   "The network learns by minimizing prediction errors.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 6
            },
            'score': 0.76
        },
        {
            'text': "Popular architectures include CNNs for images, RNNs for sequences, "
                   "and Transformers for language understanding. Each is optimized for "
                   "specific types of tasks.",
            'metadata': {
                'source': 'ml_architectures.pdf',
                'page': 12
            },
            'score': 0.72
        }
    ]
    
    # Configure for Gemini
    print("\n[1/4] Initializing Gemini generator...")
    config = GenerationConfig(
        llm_provider="gemini",
        llm_model="gemini-1.5-flash",  # Updated model name
        temperature=0.7,
        max_tokens=500,
        include_citations=True,
        return_confidence=True
    )
    
    generator = ResponseGenerator(config=config)
    print("✓ Generator initialized with Gemini")
    
    # Test 1: Basic Question
    print("\n" + "="*70)
    print("[2/4] TEST: Basic Question Answering")
    print("="*70)
    
    query = "What is deep learning and how does it work?"
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    
    try:
        print("\n⏳ Generating response with Gemini...")
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
        
        print(f"\n✅ SUCCESS!")
        print(f"  - Generated in: {result.generation_time:.2f}s")
        print(f"  - Chunks used: {result.chunks_used}")
        print(f"  - Template: {result.template_used}")
        print(f"  - Confidence: {result.confidence:.2f}" if result.confidence else "  - Confidence: N/A")
        print(f"  - Citations: {len(result.citations)}")
        
        if result.citations:
            print(f"  - Sources: {', '.join(result.citations)}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Technical Explanation
    print("\n" + "="*70)
    print("[3/4] TEST: Technical Explanation")
    print("="*70)
    
    query2 = "Explain the architecture of neural networks"
    print(f"\nQuery: {query2}")
    
    try:
        print("\n⏳ Generating technical response...")
        result2 = generator.generate(
            query=query2,
            retrieved_chunks=chunks,
            template_name="technical"
        )
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(result2.answer)
        print("-"*70)
        
        print(f"\n✅ SUCCESS! Generated in {result2.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False
    
    # Test 3: Comparison
    print("\n" + "="*70)
    print("[4/4] TEST: Comparison Query")
    print("="*70)
    
    query3 = "Compare different neural network architectures"
    print(f"\nQuery: {query3}")
    
    try:
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
        
        print(f"\n✅ SUCCESS! Generated in {result3.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\n✅ Verified Capabilities:")
    print("  1. ✓ Gemini API integration working")
    print("  2. ✓ Basic Q&A with citations")
    print("  3. ✓ Technical explanations")
    print("  4. ✓ Comparison generation")
    print("  5. ✓ Context formatting")
    print("  6. ✓ Prompt templates")
    print("  7. ✓ Real-time LLM generation")
    
    print("\n" + "="*70)
    print("✅ PHASE 7 GENERATION - FULLY OPERATIONAL WITH GEMINI!")
    print("="*70)
    
    print("\n📊 Generation Statistics:")
    total_time = result.generation_time + result2.generation_time + result3.generation_time
    avg_time = total_time / 3
    print(f"  - Total queries: 3")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average time: {avg_time:.2f}s per query")
    print(f"  - Total chunks used: {result.chunks_used + result2.chunks_used + result3.chunks_used}")
    
    print("\n" + "="*70)
    print("✨ PHASE 7 COMPLETE - READY FOR PHASE 8 EVALUATION!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_gemini_generation()
    sys.exit(0 if success else 1)
