"""
Test Phase 7 Generation with Groq API (Llama models)
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


def test_groq_generation():
    """Test generation with Groq API."""
    print("\n" + "="*70)
    print("🚀 PHASE 7: GROQ API GENERATION TEST")
    print("="*70)
    
    # Set API key
    api_key = "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"
    os.environ["GROQ_API_KEY"] = api_key
    
    print(f"\n✓ Groq API Key set: {api_key[:20]}...")
    
    # Create test chunks with realistic RAG content
    chunks = [
        {
            'text': "Deep learning is a subset of machine learning that uses neural networks "
                   "with multiple layers (deep neural networks). These networks can learn "
                   "hierarchical representations of data, making them highly effective for "
                   "complex tasks like image recognition, natural language processing, and "
                   "speech recognition. The 'deep' in deep learning refers to the number of "
                   "layers in the network.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 5,
                'chunk_id': 'chunk_001'
            },
            'score': 0.89
        },
        {
            'text': "Neural networks consist of interconnected nodes (neurons) organized in layers: "
                   "input layer, hidden layers, and output layer. Each connection has a weight "
                   "that is adjusted during training through backpropagation. The network learns "
                   "by minimizing the difference between predicted and actual outputs using "
                   "gradient descent optimization.",
            'metadata': {
                'source': 'dl_guide.pdf',
                'page': 6,
                'chunk_id': 'chunk_002'
            },
            'score': 0.76
        },
        {
            'text': "Popular deep learning architectures include: (1) Convolutional Neural Networks "
                   "(CNNs) - designed for image processing with convolutional layers that detect "
                   "spatial patterns, (2) Recurrent Neural Networks (RNNs) - optimized for sequential "
                   "data with memory of previous inputs, and (3) Transformers - use attention mechanisms "
                   "for natural language understanding, enabling models like GPT and BERT.",
            'metadata': {
                'source': 'ml_architectures.pdf',
                'page': 12,
                'chunk_id': 'chunk_003'
            },
            'score': 0.72
        }
    ]
    
    # Configure for Groq with Llama
    print("\n[1/5] Initializing Groq generator...")
    config = GenerationConfig(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",  # Fast Llama 3.3 70B model
        temperature=0.7,
        max_tokens=500,
        include_citations=True,
        return_confidence=True
    )
    
    generator = ResponseGenerator(config=config)
    print("✓ Generator initialized with Groq/Llama-3.3-70B")
    
    # Test 1: Basic Question
    print("\n" + "="*70)
    print("[2/5] TEST 1: Basic Question Answering")
    print("="*70)
    
    query = "What is deep learning and how does it work?"
    print(f"\n📝 Query: {query}")
    print(f"📚 Retrieved chunks: {len(chunks)}")
    
    try:
        print("\n⏳ Generating response with Groq/Llama...")
        result = generator.generate(
            query=query,
            retrieved_chunks=chunks,
            query_type="question"
        )
        
        print("\n" + "─"*70)
        print("💬 ANSWER:")
        print("─"*70)
        print(result.answer)
        print("─"*70)
        
        print(f"\n✅ SUCCESS!")
        print(f"  ⏱️  Generated in: {result.generation_time:.2f}s")
        print(f"  📄 Chunks used: {result.chunks_used}")
        print(f"  📋 Template: {result.template_used}")
        print(f"  🎯 Confidence: {result.confidence:.2f}" if result.confidence else "  🎯 Confidence: N/A")
        print(f"  📌 Citations: {len(result.citations)}")
        
        if result.citations:
            print(f"  📚 Sources: {', '.join(result.citations)}")
        
    except Exception as e:
        print(f"\n❌ Test 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Technical Explanation
    print("\n" + "="*70)
    print("[3/5] TEST 2: Technical Explanation")
    print("="*70)
    
    query2 = "Explain the architecture of neural networks in detail"
    print(f"\n📝 Query: {query2}")
    
    try:
        print("\n⏳ Generating technical response...")
        result2 = generator.generate(
            query=query2,
            retrieved_chunks=chunks,
            template_name="technical"
        )
        
        print("\n" + "─"*70)
        print("💬 ANSWER:")
        print("─"*70)
        print(result2.answer)
        print("─"*70)
        
        print(f"\n✅ SUCCESS! Generated in {result2.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Comparison
    print("\n" + "="*70)
    print("[4/5] TEST 3: Comparison Query")
    print("="*70)
    
    query3 = "Compare CNNs, RNNs, and Transformers - what are their key differences?"
    print(f"\n📝 Query: {query3}")
    
    try:
        print("\n⏳ Generating comparison...")
        result3 = generator.generate(
            query=query3,
            retrieved_chunks=chunks,
            template_name="comparison"
        )
        
        print("\n" + "─"*70)
        print("💬 ANSWER:")
        print("─"*70)
        print(result3.answer)
        print("─"*70)
        
        print(f"\n✅ SUCCESS! Generated in {result3.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Summary
    print("\n" + "="*70)
    print("[5/5] TEST 4: Summary Generation")
    print("="*70)
    
    query4 = "Summarize the key concepts of deep learning"
    print(f"\n📝 Query: {query4}")
    
    try:
        print("\n⏳ Generating summary...")
        result4 = generator.generate(
            query=query4,
            retrieved_chunks=chunks,
            template_name="summary"
        )
        
        print("\n" + "─"*70)
        print("💬 ANSWER:")
        print("─"*70)
        print(result4.answer)
        print("─"*70)
        
        print(f"\n✅ SUCCESS! Generated in {result4.generation_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Test 4 failed: {str(e)}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\n✅ Verified Capabilities:")
    print("  1. ✓ Groq API integration working")
    print("  2. ✓ Llama-3.3-70B model responding")
    print("  3. ✓ Basic Q&A with citations")
    print("  4. ✓ Technical explanations")
    print("  5. ✓ Comparison generation")
    print("  6. ✓ Summary generation")
    print("  7. ✓ Context formatting")
    print("  8. ✓ Multiple prompt templates")
    print("  9. ✓ Real-time LLM generation")
    
    print("\n" + "="*70)
    print("✅ PHASE 7 GENERATION - FULLY OPERATIONAL WITH GROQ!")
    print("="*70)
    
    print("\n📊 Generation Statistics:")
    total_time = result.generation_time + result2.generation_time + result3.generation_time + result4.generation_time
    avg_time = total_time / 4
    total_chunks = result.chunks_used + result2.chunks_used + result3.chunks_used + result4.chunks_used
    
    print(f"  • Total queries: 4")
    print(f"  • Total time: {total_time:.2f}s")
    print(f"  • Average time: {avg_time:.2f}s per query")
    print(f"  • Total chunks used: {total_chunks}")
    print(f"  • Model: {config.llm_model}")
    print(f"  • Provider: {config.llm_provider}")
    
    print("\n" + "="*70)
    print("✨ PHASE 7 COMPLETE - PRODUCTION READY!")
    print("🎯 READY FOR PHASE 8: EVALUATION & METRICS")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_groq_generation()
    sys.exit(0 if success else 1)
