# Multi-Modal RAG QA System

**Status**: ALL 8 PHASES COMPLETE - PRODUCTION READY

A comprehensive Multi-Modal Retrieval Augmented Generation (RAG) system for question answering over documents containing text, images, tables, charts, audio, and video content.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

### Multi-Modal Data Ingestion
- PDF Processing: Extract text, images, tables, and charts
- Video Processing: Frame extraction and audio transcription
- Audio Processing: Speech-to-text with Whisper
- Image Processing: OCR, caption generation, and visual analysis

### Advanced Retrieval
- Hybrid Retrieval: Combines dense (semantic) and sparse (BM25) search
- Smart Query Routing: Automatically routes queries to appropriate retrievers
- Reranking: Cross-encoder reranking for improved relevance
- Metadata Filtering: Filter by document type, page, section, etc.

### Intelligent Generation
- Multiple LLM Support: OpenAI GPT-4, Groq Llama, Ollama (local)
- Context-Aware Responses: Leverages multi-modal context
- Source Attribution: Cites sources and page numbers
- Multi-Turn Conversations: Maintains conversation history

### Evaluation & Metrics
- RAGAS Integration: Automated evaluation with RAGAS framework
- Custom Metrics: Precision, Recall, F1, Answer Relevancy
- Benchmark Dataset: Comprehensive test suite
- Performance Tracking: Monitor and optimize system performance

## Architecture

```
┌─────────────────┐
│  Input Sources  │
│ (PDF/Video/Audio)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│   & Extraction  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ & Chunking      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │
│   Generation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  (ChromaDB)     │
└────────┬────────┘
         │
    Query│
         ▼
┌─────────────────┐
│     Hybrid      │
│    Retrieval    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reranking    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │
│  (LLM + RAG)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda
- (Optional) CUDA for GPU acceleration
- (Optional) Tesseract OCR for text extraction

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd multi-model_assignment
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR** (Optional)
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

6. **Download required models** (Optional)
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

## Quick Start

### 1. Process Documents
```python
from src.data_ingestion.pdf_extractor import PDFExtractor
from src.preprocessing.text_chunker import TextChunker
from src.vectorstore.embedding_generator import EmbeddingGenerator

# Extract content from PDF
extractor = PDFExtractor()
content = extractor.extract("data/raw/document.pdf")

# Chunk text
chunker = TextChunker()
chunks = chunker.chunk_text(content['text'])

# Generate embeddings
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(chunks)
```

### 2. Build Vector Store
```python
from src.vectorstore.vector_db import VectorDB

# Initialize vector store
vector_db = VectorDB(store_type="chromadb")

# Add documents
vector_db.add_documents(
    texts=chunks,
    embeddings=embeddings,
    metadata=metadata
)
```

### 3. Query the System
```python
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.response_generator import ResponseGenerator

# Initialize components
retriever = HybridRetriever(vector_db)
generator = ResponseGenerator()

# Query
question = "What are the main findings in the document?"
context = retriever.retrieve(question, top_k=5)
answer = generator.generate(question, context)

print(answer)
```

### 4. Run API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run Streamlit UI
```bash
streamlit run app.py
```

## Usage

### Command Line Interface

**Process a document:**
```bash
python main.py process --input data/raw/document.pdf --output data/processed
```

**Run the pipeline:**
```bash
python pipeline.py --config configs/config.yaml
```

**Query the system:**
```bash
python main.py query --question "What is the main topic?" --top_k 5
```

### Python API

```python
from pipeline import MultiModalRAGPipeline

# Initialize pipeline
pipeline = MultiModalRAGPipeline(config_path="configs/config.yaml")

# Process document
pipeline.process_document("data/raw/document.pdf")

# Query
result = pipeline.query("What are the key findings?")
print(result['answer'])
print(result['sources'])
```

### REST API

```bash
# Upload and process document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is discussed in the document?"}'
```

## Configuration

Configuration files are located in the `configs/` directory:

- **`config.yaml`**: Main configuration (paths, logging, ingestion)
- **`model_config.yaml`**: Model settings (embeddings, LLM, OCR)
- **`retrieval_config.yaml`**: Retrieval settings (vector store, hybrid search)
- **`generation_config.yaml`**: Generation settings (prompts, parameters)

See individual config files for detailed options.

## Evaluation

### Run Evaluation

```bash
python -m src.evaluation.evaluation_pipeline \
  --test_data data/benchmark.json \
  --output results/evaluation.json
```

### Metrics

- **Retrieval Metrics**: Precision@K, Recall@K, MRR
- **Generation Metrics**: Answer Relevancy, Faithfulness
- **RAGAS Metrics**: Context Precision, Context Recall

### Create Benchmark Dataset

```python
from src.evaluation.benchmark_dataset import BenchmarkDataset

dataset = BenchmarkDataset()
dataset.create_from_documents("data/processed")
dataset.save("data/benchmark.json")
```

## Project Structure

```
multi-model_assignment/
├── src/                          # Source code
│   ├── data_ingestion/          # Extract data from PDFs, videos, audio
│   ├── preprocessing/           # Process and chunk content
│   ├── vectorstore/             # Embeddings and vector database
│   ├── retrieval/               # Retrieval strategies
│   ├── generation/              # LLM integration and response generation
│   ├── evaluation/              # Metrics and evaluation
│   └── utils/                   # Utilities
├── configs/                      # Configuration files
├── data/                         # Data directory
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── main.py                       # CLI entry point
├── pipeline.py                   # End-to-end pipeline
├── api.py                        # REST API
└── requirements.txt             # Dependencies
```

## Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```

Run specific test:
```bash
pytest tests/test_retrieval.py -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- RAGAS for evaluation framework
- LangChain for LLM orchestration
- ChromaDB for vector storage
- HuggingFace for models and embeddings

## Contact

For questions or issues, please open an issue on GitHub.
