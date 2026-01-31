"""
Multi-Modal RAG QA System - Main Application
Production-ready end-to-end pipeline for document processing and question answering.

Usage:
    # Process a document
    python main.py --mode process --pdf data/raw/document.pdf
    
    # Query the system
    python main.py --mode query --query "What is machine learning?"
    
    # Interactive mode
    python main.py --mode interactive
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from loguru import logger
from src.data_ingestion import PDFExtractor
from src.preprocessing import TextCleaner, TextChunker
from src.vectorstore import EmbeddingGenerator, FAISSStore
from src.retrieval import HybridRetriever
from src.generation import ResponseGenerator
from src.evaluation import RAGMetrics, RAGASEvaluator, QualityAssessor

# Configure logger
logger.remove()
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(sys.stdout, level="INFO")


class MultiModalRAGSystem:
    """Complete Multi-Modal RAG QA System."""
    
    def __init__(
        self,
        vector_store_path: str = "data/vector_store/faiss_index",
        llm_provider: str = "groq",
        llm_model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_path: Path to save/load vector store
            llm_provider: LLM provider (groq, openai, gemini)
            llm_model: Model name
            api_key: API key for LLM provider
        """
        self.vector_store_path = vector_store_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key or os.environ.get(f"{llm_provider.upper()}_API_KEY")
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(chunk_size=300, overlap=50)
        self.embedder = EmbeddingGenerator()
        
        self.vector_store: Optional[FAISSStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[ResponseGenerator] = None
        self.chunks: List[Dict] = []
        
        logger.info("Multi-Modal RAG System initialized")
    
    def process_document(self, pdf_path: str) -> Dict:
        """
        Process a PDF document and add it to the knowledge base.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing results
        """
        logger.info(f"Processing document: {pdf_path}")
        
        # 1. Extract content
        logger.info("Step 1/5: Extracting content from PDF...")
        content = self.pdf_extractor.extract(pdf_path)
        
        text_length = len(content.get('text', ''))
        num_images = len(content.get('images', []))
        num_tables = len(content.get('tables', []))
        
        logger.info(f"Extracted: {text_length} chars, {num_images} images, {num_tables} tables")
        
        # 2. Clean text
        logger.info("Step 2/5: Cleaning text...")
        cleaned_text = self.text_cleaner.clean(content['text'])
        logger.info(f"Cleaned text: {len(cleaned_text)} chars")
        
        # 3. Create chunks
        logger.info("Step 3/5: Creating semantic chunks...")
        chunks = self.text_chunker.chunk_text(
            cleaned_text,
            metadata={
                'source': Path(pdf_path).name,
                'path': pdf_path
            }
        )
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        # 4. Generate embeddings
        logger.info("Step 4/5: Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(chunk_texts, batch_size=32)
        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        
        # 5. Add to vector store
        logger.info("Step 5/5: Adding to vector store...")
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISSStore(dimension=embeddings.shape[1])
            self.chunks = chunks
        else:
            # Add to existing store
            self.chunks.extend(chunks)
        
        self.vector_store.add_vectors(embeddings, chunks)
        logger.info(f"Vector store size: {self.vector_store.get_size()} vectors")
        
        # Save vector store
        self.save_vector_store()
        
        return {
            'status': 'success',
            'document': Path(pdf_path).name,
            'chunks_created': len(chunks),
            'total_chunks': len(self.chunks),
            'text_length': text_length,
            'images': num_images,
            'tables': num_tables
        }
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store.
        
        Returns:
            True if loaded successfully
        """
        try:
            logger.info(f"Loading vector store from: {self.vector_store_path}")
            self.vector_store = FAISSStore.load(self.vector_store_path)
            
            # Load chunks metadata
            import pickle
            chunks_path = f"{self.vector_store_path}_chunks.pkl"
            if os.path.exists(chunks_path):
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
            
            logger.info(f"Vector store loaded: {self.vector_store.get_size()} vectors")
            return True
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return False
    
    def save_vector_store(self):
        """Save vector store to disk."""
        logger.info(f"Saving vector store to: {self.vector_store_path}")
        self.vector_store.save(self.vector_store_path)
        
        # Save chunks metadata
        import pickle
        chunks_path = f"{self.vector_store_path}_chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved {len(self.chunks)} chunks to {chunks_path}")
    
    def initialize_retriever(self):
        """Initialize the retriever."""
        if self.vector_store is None:
            raise ValueError("No vector store available. Process a document first or load existing store.")
        
        logger.info("Initializing retriever...")
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            chunks=self.chunks
        )
        logger.info("Retriever initialized")
    
    def initialize_generator(self):
        """Initialize the LLM generator."""
        if not self.api_key:
            raise ValueError(f"No API key found for {self.llm_provider}. Set {self.llm_provider.upper()}_API_KEY environment variable.")
        
        logger.info(f"Initializing generator: {self.llm_provider}/{self.llm_model}")
        self.generator = ResponseGenerator(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.api_key
        )
        logger.info("Generator initialized")
    
    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        evaluate: bool = False
    ) -> Dict:
        """
        Answer a question using RAG pipeline.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            use_reranking: Whether to use reranking
            evaluate: Whether to evaluate the answer
            
        Returns:
            Answer with metadata
        """
        logger.info(f"Query: {query}")
        
        # Initialize components if needed
        if self.retriever is None:
            self.initialize_retriever()
        if self.generator is None:
            self.initialize_generator()
        
        # 1. Retrieve relevant chunks
        logger.info(f"Retrieving top-{top_k} chunks...")
        results = self.retriever.hybrid_search(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking
        )
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        # 2. Generate answer
        logger.info("Generating answer...")
        response = self.generator.generate(
            query=query,
            context_chunks=results
        )
        logger.info("Answer generated")
        
        # 3. Prepare result
        result = {
            'query': query,
            'answer': response.answer,
            'citations': response.citations,
            'confidence': response.confidence,
            'num_chunks_retrieved': len(results),
            'generation_time': response.metadata.get('generation_time', 0),
            'context_chunks': [
                {
                    'text': chunk['text'][:200] + '...',
                    'score': chunk.get('score', 0),
                    'source': chunk.get('metadata', {}).get('source', 'unknown')
                }
                for chunk in results[:3]  # Top 3 for display
            ]
        }
        
        # 4. Evaluate if requested
        if evaluate:
            logger.info("Evaluating answer quality...")
            
            # RAGAS evaluation
            ragas_eval = RAGASEvaluator()
            ragas_result = ragas_eval.evaluate(
                query=query,
                answer=response.answer,
                context=response.context
            )
            
            # Quality assessment
            quality_eval = QualityAssessor()
            quality_result = quality_eval.assess(
                answer=response.answer,
                query=query,
                context=response.context
            )
            
            result['evaluation'] = {
                'faithfulness': ragas_result.faithfulness,
                'answer_relevance': ragas_result.answer_relevance,
                'overall_ragas_score': ragas_result.overall_score,
                'quality_score': quality_result.overall_quality,
                'hallucination_score': quality_result.hallucination_score,
                'quality_passed': quality_result.passed,
                'issues': quality_result.issues
            }
            
            logger.info(f"Evaluation: RAGAS={ragas_result.overall_score:.3f}, Quality={quality_result.overall_quality:.3f}")
        
        return result


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Multi-Modal RAG QA System")
    parser.add_argument('--mode', choices=['process', 'query', 'interactive'], required=True,
                       help='Operation mode: process documents, query, or interactive')
    parser.add_argument('--pdf', type=str, help='Path to PDF document to process')
    parser.add_argument('--query', type=str, help='Question to ask')
    parser.add_argument('--provider', type=str, default='groq', 
                       choices=['groq', 'openai', 'gemini'],
                       help='LLM provider (default: groq)')
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                       help='LLM model name')
    parser.add_argument('--api-key', type=str, help='API key for LLM provider')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate answer quality')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    
    args = parser.parse_args()
    
    # Initialize system
    rag_system = MultiModalRAGSystem(
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=args.api_key
    )
    
    try:
        if args.mode == 'process':
            # Process document mode
            if not args.pdf:
                print("❌ Error: --pdf argument required for process mode")
                return
            
            print(f"\n📄 Processing document: {args.pdf}")
            print("=" * 80)
            
            result = rag_system.process_document(args.pdf)
            
            print(f"\n✅ Document processed successfully!")
            print(f"   Document: {result['document']}")
            print(f"   Chunks created: {result['chunks_created']}")
            print(f"   Total chunks in store: {result['total_chunks']}")
            print(f"   Text length: {result['text_length']} chars")
            print(f"   Images: {result['images']}")
            print(f"   Tables: {result['tables']}")
            print(f"\n💾 Vector store saved to: data/vector_store/faiss_index")
            
        elif args.mode == 'query':
            # Query mode
            if not args.query:
                print("❌ Error: --query argument required for query mode")
                return
            
            # Load existing vector store
            if not rag_system.load_vector_store():
                print("❌ Error: No vector store found. Process a document first.")
                return
            
            print(f"\n❓ Query: {args.query}")
            print("=" * 80)
            
            result = rag_system.answer_question(
                query=args.query,
                top_k=args.top_k,
                use_reranking=not args.no_rerank,
                evaluate=args.evaluate
            )
            
            print(f"\n✅ Answer:")
            print(f"{result['answer']}\n")
            print(f"📊 Metadata:")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Citations: {', '.join(result['citations'])}")
            print(f"   Chunks retrieved: {result['num_chunks_retrieved']}")
            print(f"   Generation time: {result['generation_time']:.2f}s")
            
            if args.evaluate and 'evaluation' in result:
                eval_data = result['evaluation']
                print(f"\n📈 Evaluation:")
                print(f"   Faithfulness: {eval_data['faithfulness']:.3f}")
                print(f"   Answer Relevance: {eval_data['answer_relevance']:.3f}")
                print(f"   RAGAS Score: {eval_data['overall_ragas_score']:.3f}")
                print(f"   Quality Score: {eval_data['quality_score']:.3f}")
                print(f"   Hallucination: {eval_data['hallucination_score']:.3f}")
                print(f"   Status: {'✅ PASS' if eval_data['quality_passed'] else '❌ FAIL'}")
                if eval_data['issues']:
                    print(f"   Issues: {', '.join(eval_data['issues'])}")
            
            print(f"\n📚 Top sources:")
            for i, chunk in enumerate(result['context_chunks'], 1):
                print(f"   {i}. [{chunk['source']}] (score: {chunk['score']:.3f})")
                print(f"      {chunk['text'][:150]}...")
            
        elif args.mode == 'interactive':
            # Interactive mode
            if not rag_system.load_vector_store():
                print("❌ Error: No vector store found. Process a document first.")
                return
            
            print("\n" + "=" * 80)
            print("🤖 Multi-Modal RAG QA System - Interactive Mode")
            print("=" * 80)
            print(f"Provider: {args.provider}/{args.model}")
            print(f"Vector store: {rag_system.vector_store.get_size()} chunks loaded")
            print("\nCommands:")
            print("  - Type your question")
            print("  - Type 'eval' to enable/disable evaluation")
            print("  - Type 'quit' or 'exit' to quit")
            print("=" * 80)
            
            evaluate = args.evaluate
            
            while True:
                try:
                    query = input("\n❓ Your question: ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 Goodbye!")
                        break
                    
                    if query.lower() == 'eval':
                        evaluate = not evaluate
                        print(f"✓ Evaluation {'enabled' if evaluate else 'disabled'}")
                        continue
                    
                    result = rag_system.answer_question(
                        query=query,
                        top_k=args.top_k,
                        use_reranking=not args.no_rerank,
                        evaluate=evaluate
                    )
                    
                    print(f"\n💡 Answer:")
                    print(f"{result['answer']}")
                    print(f"\n📊 Confidence: {result['confidence']:.2f} | Sources: {', '.join(result['citations'][:3])}")
                    
                    if evaluate and 'evaluation' in result:
                        eval_data = result['evaluation']
                        status = '✅ PASS' if eval_data['quality_passed'] else '❌ FAIL'
                        print(f"📈 Quality: {eval_data['quality_score']:.2f} | Hallucination: {eval_data['hallucination_score']:.2f} | {status}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    logger.error(f"Error in interactive mode: {str(e)}", exc_info=True)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
