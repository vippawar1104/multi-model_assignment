# Multi-Modal RAG QA System - Quick Reference Guide

## 🚀 Getting Started

### Installation
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key (for LLM generation)
export GROQ_API_KEY="your-groq-api-key"
```

### Quick Test
```bash
# Run complete pipeline demo
python examples/complete_pipeline_demo.py

# Run evaluation demo
python examples/phase8_evaluation_example.py
```

---

## 📚 Phase-by-Phase Examples

### Phase 3: Data Ingestion
```python
from src.data_ingestion import PDFExtractor

extractor = PDFExtractor()
content = extractor.extract("document.pdf")
# Returns: {'text': '...', 'images': [...], 'tables': [...]}
```

### Phase 4: Preprocessing
```python
from src.preprocessing import TextCleaner, TextChunker

cleaner = TextCleaner()
cleaned = cleaner.clean(text)

chunker = TextChunker(chunk_size=300, overlap=50)
chunks = chunker.chunk_text(cleaned)
# Returns: [{'text': '...', 'metadata': {...}}, ...]
```

### Phase 5: Vector Store
```python
from src.vectorstore import EmbeddingGenerator, FAISSStore

# Generate embeddings
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings([chunk['text'] for chunk in chunks])

# Build FAISS index
store = FAISSStore(dimension=384)
store.add_vectors(embeddings, chunks)
store.save("data/vector_store/faiss_index")
```

### Phase 6: Retrieval
```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever(vector_store=store, embedder=embedder, chunks=chunks)
results = retriever.hybrid_search("What is machine learning?", top_k=5)
# Returns: [{'text': '...', 'score': 0.95, ...}, ...]
```

### Phase 7: Generation
```python
from src.generation import ResponseGenerator

generator = ResponseGenerator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    api_key="your-api-key"
)

response = generator.generate(
    query="What is machine learning?",
    context_chunks=results
)

print(response.answer)
print(response.citations)
print(f"Confidence: {response.confidence}")
```

### Phase 8: Evaluation
```python
from src.evaluation import RAGMetrics, RAGASEvaluator, QualityAssessor

# Retrieval metrics
metrics = RAGMetrics()
result = metrics.evaluate_retrieval(retrieved=['doc1', 'doc2'], relevant=['doc1', 'doc3'])
print(f"Precision: {result.precision}")

# RAGAS evaluation
ragas = RAGASEvaluator()
result = ragas.evaluate(query="...", answer="...", context="...")
print(f"Faithfulness: {result.faithfulness}")

# Quality assessment
quality = QualityAssessor()
result = quality.assess(answer="...", query="...", context="...")
print(f"Hallucination: {result.hallucination_score}")
```

---

## 🔧 Common Tasks

### Process a New Document
```python
from src.data_ingestion import PDFExtractor
from src.preprocessing import TextChunker
from src.vectorstore import EmbeddingGenerator, FAISSStore

# 1. Extract
extractor = PDFExtractor()
content = extractor.extract("my_document.pdf")

# 2. Chunk
chunker = TextChunker()
chunks = chunker.chunk_text(content['text'])

# 3. Embed
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings([c['text'] for c in chunks])

# 4. Store
store = FAISSStore.load("data/vector_store/faiss_index")  # Load existing
store.add_vectors(embeddings, chunks)
store.save("data/vector_store/faiss_index")  # Save updated
```

### Ask a Question
```python
from src.vectorstore import FAISSStore, EmbeddingGenerator
from src.retrieval import HybridRetriever
from src.generation import ResponseGenerator

# 1. Load vector store
store = FAISSStore.load("data/vector_store/faiss_index")
embedder = EmbeddingGenerator()

# 2. Retrieve
retriever = HybridRetriever(store, embedder)
results = retriever.hybrid_search("Your question?", top_k=5)

# 3. Generate
generator = ResponseGenerator(provider="groq", api_key="...")
response = generator.generate("Your question?", results)

print(response.answer)
```

### Evaluate Results
```python
from src.evaluation import RAGMetrics, RAGASEvaluator, QualityAssessor

# Comprehensive evaluation
metrics = RAGMetrics()
ragas = RAGASEvaluator()
quality = QualityAssessor()

# Evaluate retrieval
retrieval_metrics = metrics.evaluate_retrieval(retrieved_docs, ground_truth_docs)

# Evaluate answer quality
ragas_result = ragas.evaluate(query, answer, context)
quality_result = quality.assess(answer, query, context)

# Print results
print(f"Retrieval F1: {retrieval_metrics.f1_score:.3f}")
print(f"RAGAS Score: {ragas_result.overall_score:.3f}")
print(f"Quality: {quality_result.overall_quality:.3f}")
print(f"Hallucination: {quality_result.hallucination_score:.3f}")
```

---

## 🎯 Configuration

### Vector Store Config
```python
from src.vectorstore import FAISSStore

store = FAISSStore(
    dimension=384,              # Embedding dimension
    metric='cosine',            # Distance metric
    index_type='Flat'           # Index type
)
```

### Retrieval Config
```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever(
    dense_weight=0.7,           # Weight for dense search
    sparse_weight=0.3,          # Weight for sparse search
    use_reranking=True,         # Enable reranking
    rerank_top_k=10             # Rerank top 10 results
)
```

### Generation Config
```python
from src.generation import ResponseGenerator

generator = ResponseGenerator(
    provider="groq",                    # Provider: openai, groq, gemini
    model="llama-3.3-70b-versatile",   # Model name
    temperature=0.1,                    # Lower = more focused
    max_tokens=1000,                    # Max response length
    enable_citations=True,              # Include citations
    estimate_confidence=True            # Include confidence score
)
```

### Evaluation Config
```python
from src.evaluation import RAGMetrics, RAGASEvaluator, QualityAssessor

# Metrics thresholds
metrics = RAGMetrics(MetricsConfig(
    precision_threshold=0.7,
    recall_threshold=0.6,
    f1_threshold=0.65
))

# RAGAS thresholds
ragas = RAGASEvaluator(RAGASConfig(
    faithfulness_threshold=0.7,
    answer_relevance_threshold=0.6
))

# Quality thresholds
quality = QualityAssessor(QualityConfig(
    min_context_overlap=0.3,
    min_completeness=0.5
))
```

---

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Precision**: Relevant retrieved / Total retrieved
- **Recall**: Relevant retrieved / Total relevant
- **F1 Score**: Harmonic mean of precision and recall
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Metrics
- **Faithfulness**: No hallucinations, grounded in context
- **Answer Relevance**: Relevance to query
- **Context Relevance**: Relevance of retrieved context
- **Completeness**: Query coverage
- **Coherence**: Answer structure and flow

### Quality Metrics
- **Hallucination Score**: Context overlap (higher = less hallucination)
- **Completeness**: Query term coverage
- **Coherence**: Sentence quality

### Interpretation
```
Metric              | Good    | Needs Improvement
--------------------|---------|------------------
Precision           | > 0.7   | < 0.5
Recall              | > 0.6   | < 0.4
F1 Score            | > 0.65  | < 0.5
Faithfulness        | > 0.7   | < 0.5
Answer Relevance    | > 0.6   | < 0.4
Hallucination Score | > 0.6   | < 0.3
```

---

## 🐛 Troubleshooting

### Low Retrieval Quality
```python
# Increase top_k
results = retriever.hybrid_search(query, top_k=10)  # Try 10 instead of 5

# Adjust hybrid weights
retriever = HybridRetriever(
    dense_weight=0.8,    # Favor semantic search
    sparse_weight=0.2
)

# Enable reranking
retriever = HybridRetriever(use_reranking=True)
```

### Hallucinated Answers
```python
# Lower temperature
generator = ResponseGenerator(temperature=0.0)  # More deterministic

# Increase context
results = retriever.hybrid_search(query, top_k=10)  # More context

# Use stricter prompt
generator.template = "technical"  # Use technical template
```

### Slow Response Times
```python
# Reduce top_k
results = retriever.hybrid_search(query, top_k=3)

# Disable reranking
retriever = HybridRetriever(use_reranking=False)

# Use faster model
generator = ResponseGenerator(model="llama-3.1-8b-instant")
```

---

## 📁 File Locations

### Data Files
```
data/raw/                    # Original documents
data/processed/              # Processed chunks
data/embeddings/             # Generated embeddings
data/vector_store/           # FAISS indices
```

### Configuration Files
```
configs/config.yaml          # Main config
configs/model_config.yaml    # LLM config
configs/retrieval_config.yaml # Retrieval config
configs/generation_config.yaml # Generation config
```

### Examples
```
examples/complete_pipeline_demo.py        # Full pipeline
examples/phase8_evaluation_example.py     # Evaluation demo
examples/test_groq_generation.py          # LLM testing
```

### Documentation
```
docs/phase3_data_ingestion_complete.md    # Data ingestion
docs/phase4_preprocessing_complete.md     # Preprocessing
docs/phase5_vector_store_complete.md      # Vector store
docs/phase7_generation_complete.md        # Generation
docs/phase8_evaluation_complete.md        # Evaluation
PROJECT_COMPLETE.md                       # Project summary
```

---

## 🔑 Environment Variables

```bash
# Required for LLM generation
export GROQ_API_KEY="gsk_..."           # Groq API key
export OPENAI_API_KEY="sk-..."          # OpenAI API key (optional)
export GOOGLE_API_KEY="..."             # Gemini API key (optional)

# Optional settings
export RAG_LOG_LEVEL="INFO"             # Logging level
export RAG_CONFIG_PATH="configs/"       # Config directory
```

---

## 🎓 Learning Resources

### Documentation
1. Read `PROJECT_COMPLETE.md` for full project overview
2. Check phase-specific docs in `docs/` folder
3. Review examples in `examples/` folder

### Testing
1. Run `python examples/complete_pipeline_demo.py`
2. Try `python examples/phase8_evaluation_example.py`
3. Experiment with different queries and documents

### Customization
1. Adjust configs in `configs/` folder
2. Modify prompts in `src/generation/prompt_manager.py`
3. Tune thresholds in evaluation modules

---

## 📞 Support

For detailed documentation, see:
- `PROJECT_COMPLETE.md` - Complete project overview
- `docs/` - Phase-specific documentation
- `examples/` - Working code examples

For issues or questions, refer to the comprehensive documentation in the `/docs` folder.

---

**Quick Start**: `python examples/complete_pipeline_demo.py`  
**Full Documentation**: `PROJECT_COMPLETE.md`  
**Status**: ✅ Production Ready - All 8 Phases Complete
