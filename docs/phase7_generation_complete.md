# Phase 7: Response Generation - Complete Documentation

## Overview
Phase 7 implements the response generation pipeline for the Multi-Modal RAG system. This phase takes retrieved context from Phase 6 and generates natural language answers using Large Language Models (LLMs).

## Components

### 1. LLMClient (`llm_client.py`)
**Purpose**: Unified interface for different LLM providers

**Features**:
- Multi-provider support (OpenAI, Groq, local models)
- Streaming and non-streaming generation
- Token counting and estimation
- Error handling and retry logic

**Usage**:
```python
from generation import LLMClient, LLMConfig

# Configure LLM
config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

client = LLMClient(config)

# Generate response
response = client.generate_with_system(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is machine learning?"
)
```

**Supported Providers**:
- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo
- **Groq**: Llama-3, Mixtral, Gemma
- **Local**: Any OpenAI-compatible endpoint

**Environment Variables**:
- `OPENAI_API_KEY`: For OpenAI models
- `GROQ_API_KEY`: For Groq models

### 2. ContextFormatter (`context_formatter.py`)
**Purpose**: Format retrieved chunks into LLM-ready context

**Features**:
- Smart truncation to fit token limits
- Metadata and score inclusion
- Multi-modal content formatting
- Token estimation

**Usage**:
```python
from generation import ContextFormatter, FormattingConfig

config = FormattingConfig(
    max_context_length=4000,
    include_metadata=True,
    include_scores=True,
    truncation_strategy="smart"
)

formatter = ContextFormatter(config)

# Format text chunks
context = formatter.format_chunks(
    chunks=retrieved_chunks,
    query=user_query,
    max_chunks=5
)

# Format with images
multimodal = formatter.format_with_images(
    text_chunks=text_chunks,
    image_chunks=image_chunks,
    max_images=3
)
```

**Truncation Strategies**:
- `smart`: Preserve sentence boundaries
- `head`: Keep beginning of context
- `tail`: Keep end of context

### 3. PromptManager (`prompt_manager.py`)
**Purpose**: Manage prompt templates and engineering

**Features**:
- Pre-built templates for different query types
- Few-shot learning support
- Dynamic variable substitution
- Template customization

**Built-in Templates**:
1. **general**: Standard Q&A
2. **technical**: Detailed technical explanations
3. **summary**: Summarization tasks
4. **comparison**: Compare and contrast
5. **multimodal**: Text + image understanding

**Usage**:
```python
from generation import PromptManager

manager = PromptManager()

# Get prompt for query
prompt = manager.get_prompt(
    template_name="technical",
    variables={'context': context, 'query': query}
)

# List available templates
templates = manager.list_templates()

# Add few-shot examples
manager.add_few_shot_examples(
    template_name="general",
    examples=[
        {'query': 'What is AI?', 'answer': 'AI is...'},
        {'query': 'How does ML work?', 'answer': 'ML works by...'}
    ]
)
```

### 4. ResponseGenerator (`response_generator.py`)
**Purpose**: Main orchestrator for complete generation pipeline

**Features**:
- End-to-end generation pipeline
- Streaming and batch processing
- Citation generation
- Confidence estimation
- Error handling with fallbacks

**Usage**:
```python
from generation import ResponseGenerator, GenerationConfig

config = GenerationConfig(
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
    include_citations=True,
    return_confidence=True
)

generator = ResponseGenerator(config=config)

# Generate response
result = generator.generate(
    query="What is deep learning?",
    retrieved_chunks=chunks,
    query_type="question"
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Citations: {result.citations}")
print(f"Time: {result.generation_time:.2f}s")
```

## Configuration

### Generation Config (`configs/generation_config.yaml`)
```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000

context:
  max_context_length: 4000
  max_chunks: 5
  include_metadata: true
  include_scores: false

prompts:
  default_template: "general"
  use_few_shot: false

citations:
  include_citations: true
  citation_format: "[{source}, p.{page}]"

response:
  return_confidence: true
  fallback_response: "I don't have enough information..."
```

## Generation Pipeline Flow

```
Retrieved Chunks (Phase 6)
    ↓
Context Formatting
    ↓
Template Selection (based on query type)
    ↓
Prompt Construction
    ↓
LLM API Call
    ↓
Response Post-Processing
    ↓
Citation Addition
    ↓
Confidence Estimation
    ↓
Final Answer (to Phase 8 Evaluation)
```

## Advanced Features

### 1. Streaming Generation
```python
for chunk in generator.generate_streaming(query, chunks):
    print(chunk, end='', flush=True)
```

### 2. Batch Processing
```python
results = generator.generate_batch(
    queries=["Query 1", "Query 2", "Query 3"],
    retrieved_chunks_list=[chunks1, chunks2, chunks3],
    query_types=["question", "summary", "comparison"]
)
```

### 3. Multi-Modal Generation
```python
# Format context with images
multimodal_context = formatter.format_with_images(
    text_chunks=text_chunks,
    image_chunks=image_chunks
)

# Use multimodal template
result = generator.generate(
    query=query,
    retrieved_chunks=text_chunks,
    template_name="multimodal"
)
```

### 4. Custom Prompts
```python
# Create custom prompt
custom_prompt = manager.create_custom_prompt(
    system_prompt="You are a domain expert...",
    context=context,
    query=query,
    additional_instructions="Use bullet points"
)
```

## Performance Optimization

### 1. Context Length Management
- Automatic truncation to fit model limits
- Smart chunking for long documents
- Token estimation before API calls

### 2. Caching
- Cache formatted contexts for repeated queries
- Reuse prompt templates
- Store common few-shot examples

### 3. Batch Processing
- Process multiple queries in parallel
- Reduce API overhead
- Optimize token usage

## Error Handling

### 1. API Failures
```python
try:
    result = generator.generate(query, chunks)
except Exception as e:
    # Fallback response returned
    print(f"Error: {e}")
```

### 2. Missing API Keys
- Graceful degradation without keys
- Clear warning messages
- Demo mode for testing

### 3. Token Limit Exceeded
- Automatic context truncation
- Smart chunking strategies
- Warning logs

## Testing

### Run Phase 7 Example
```bash
# Set API key (optional for demo)
export OPENAI_API_KEY="your-key-here"

# Run example
python examples/phase7_generation_example.py
```

### Test Individual Components
```python
# Test context formatting
from generation import ContextFormatter
formatter = ContextFormatter()
context = formatter.format_chunks(chunks)

# Test prompt templates
from generation import PromptManager
manager = PromptManager()
prompt = manager.get_prompt('technical', {'context': context, 'query': query})

# Test LLM client (requires API key)
from generation import LLMClient, LLMConfig
client = LLMClient(LLMConfig(provider="openai"))
response = client.generate_with_system("System prompt", "User prompt")
```

## Integration with Other Phases

### Input from Phase 6 (Retrieval)
```python
from retrieval import Retriever
from generation import ResponseGenerator

# Retrieve chunks
retriever = Retriever()
chunks = retriever.retrieve(query, top_k=5)

# Generate response
generator = ResponseGenerator()
result = generator.generate(query, chunks)
```

### Output to Phase 8 (Evaluation)
```python
from generation import ResponseGenerator
from evaluation import RAGASEvaluator

# Generate response
result = generator.generate(query, chunks)

# Evaluate response
evaluator = RAGASEvaluator()
scores = evaluator.evaluate(
    query=query,
    answer=result.answer,
    context=result.context_used
)
```

## Best Practices

### 1. Template Selection
- Use `general` for standard Q&A
- Use `technical` for detailed explanations
- Use `comparison` for contrasting concepts
- Use `multimodal` when images are relevant

### 2. Context Formatting
- Keep context concise (< 4000 chars)
- Include metadata for citations
- Use smart truncation to preserve meaning
- Limit to top 5 chunks for relevance

### 3. Prompt Engineering
- Clear system prompts with guidelines
- Few-shot examples for complex tasks
- Specify desired output format
- Include constraints (e.g., "answer only from context")

### 4. Citation Management
- Enable citations for factual queries
- Use consistent citation format
- Reference page numbers when available
- Link to original sources

### 5. Error Handling
- Always provide fallback responses
- Log errors for debugging
- Return confidence scores
- Handle API timeouts gracefully

## Troubleshooting

### Issue: "No API key found"
**Solution**: Set environment variable
```bash
export OPENAI_API_KEY="sk-..."
# or
export GROQ_API_KEY="gsk_..."
```

### Issue: Context too long
**Solution**: Reduce max_context_length or max_chunks
```python
config = GenerationConfig(
    max_context_length=2000,
    max_chunks=3
)
```

### Issue: Poor answer quality
**Solution**: 
1. Improve retrieval (Phase 6)
2. Use better prompt template
3. Add few-shot examples
4. Increase temperature for creativity

### Issue: Slow generation
**Solution**:
1. Reduce max_tokens
2. Use faster model (gpt-3.5 vs gpt-4)
3. Enable streaming for perceived speed
4. Batch similar queries

## Dependencies

```txt
# LLM Clients
openai>=1.0.0         # OpenAI API
groq>=0.4.0           # Groq API

# Utilities
loguru>=0.7.0         # Logging
pyyaml>=6.0           # Config loading
```

## Files Created

```
src/generation/
├── __init__.py              # Module exports
├── llm_client.py            # LLM API interface (~250 lines)
├── context_formatter.py     # Context formatting (~280 lines)
├── prompt_manager.py        # Prompt templates (~340 lines)
└── response_generator.py   # Main generator (~380 lines)

configs/
└── generation_config.yaml   # Generation settings

examples/
└── phase7_generation_example.py  # Complete demo (~400 lines)
```

**Total**: ~1,650 lines of production code

## Summary

Phase 7 provides a complete response generation pipeline with:

✅ **Multi-provider LLM support** (OpenAI, Groq, local)
✅ **Smart context formatting** with truncation
✅ **Template-based prompts** for different query types
✅ **Citation generation** with source tracking
✅ **Confidence estimation** for answers
✅ **Streaming and batch processing**
✅ **Multi-modal support** (text + images)
✅ **Robust error handling** with fallbacks

The generation system is ready for production use and integrates seamlessly with Phase 6 (Retrieval) and Phase 8 (Evaluation).

## Next Steps

Proceed to **Phase 8: Evaluation & Metrics** to:
- Implement answer quality assessment
- Add retrieval accuracy metrics
- Create RAGAS evaluation pipeline
- Build comprehensive benchmarking system
