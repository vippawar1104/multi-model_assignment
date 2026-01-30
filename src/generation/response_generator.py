"""
Response Generator - Main orchestrator for generating LLM responses.
Combines retrieval, context formatting, prompt engineering, and LLM calls.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
from loguru import logger

from .llm_client import LLMClient, LLMConfig
from .context_formatter import ContextFormatter, FormattingConfig
from .prompt_manager import PromptManager


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Context settings
    max_context_length: int = 4000
    max_chunks: int = 5
    include_metadata: bool = True
    include_scores: bool = False
    
    # Prompt settings
    default_template: str = "general"
    use_few_shot: bool = False
    
    # Citation settings
    include_citations: bool = True
    citation_format: str = "[{source}, p.{page}]"
    
    # Response settings
    stream_response: bool = False
    return_confidence: bool = False
    fallback_response: str = "I don't have enough information to answer that question accurately."


@dataclass
class GenerationResult:
    """Result from response generation."""
    answer: str
    query: str
    context_used: str
    chunks_used: int
    template_used: str
    generation_time: float
    citations: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    """
    Main class for generating responses in RAG pipeline.
    
    Orchestrates:
    1. Context formatting from retrieved chunks
    2. Prompt construction based on query type
    3. LLM API calls
    4. Response post-processing and citation
    """
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        llm_client: Optional[LLMClient] = None,
        context_formatter: Optional[ContextFormatter] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize response generator.
        
        Args:
            config: Generation configuration
            llm_client: Optional pre-configured LLM client
            context_formatter: Optional context formatter
            prompt_manager: Optional prompt manager
        """
        self.config = config or GenerationConfig()
        
        # Initialize components
        if llm_client:
            self.llm_client = llm_client
        else:
            llm_config = LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self.llm_client = LLMClient(llm_config)
        
        self.context_formatter = context_formatter or ContextFormatter(
            FormattingConfig(
                max_context_length=self.config.max_context_length,
                include_metadata=self.config.include_metadata,
                include_scores=self.config.include_scores
            )
        )
        
        self.prompt_manager = prompt_manager or PromptManager()
        
        logger.info("Initialized ResponseGenerator")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_type: Optional[str] = None,
        template_name: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate response for a query using retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: Chunks retrieved from vector store
            query_type: Type of query (for template selection)
            template_name: Specific template to use (overrides query_type)
            **kwargs: Additional generation parameters
        
        Returns:
            GenerationResult with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Format context from chunks
            context = self.context_formatter.format_chunks(
                chunks=retrieved_chunks,
                query=query,
                max_chunks=self.config.max_chunks
            )
            
            if not context:
                logger.warning("No context available for generation")
                return self._create_fallback_result(query, start_time)
            
            # Step 2: Select and format prompt
            template = template_name or self.config.default_template
            if query_type and not template_name:
                # Let prompt manager select template based on query type
                template = query_type
            
            prompt = self.prompt_manager.get_prompt(
                template_name=template,
                variables={'context': context, 'query': query},
                query_type=query_type
            )
            
            # Step 3: Generate response using LLM
            answer = self.llm_client.generate_with_system(
                system_prompt=prompt['system'],
                user_prompt=prompt['user'],
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens')
            )
            
            # Step 4: Post-process response
            if self.config.include_citations:
                answer, citations = self._add_citations(answer, retrieved_chunks)
            else:
                citations = []
            
            # Step 5: Calculate confidence if enabled
            confidence = None
            if self.config.return_confidence:
                confidence = self._estimate_confidence(answer, retrieved_chunks)
            
            generation_time = time.time() - start_time
            
            result = GenerationResult(
                answer=answer,
                query=query,
                context_used=context,
                chunks_used=len(retrieved_chunks),
                template_used=template,
                generation_time=generation_time,
                citations=citations,
                confidence=confidence,
                metadata={
                    'model': self.llm_client.config.model,
                    'provider': self.llm_client.config.provider,
                    'query_type': query_type
                }
            )
            
            logger.info(
                f"Generated response in {generation_time:.2f}s "
                f"using {len(retrieved_chunks)} chunks"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return self._create_error_result(query, str(e), start_time)
    
    def generate_streaming(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_type: Optional[str] = None,
        template_name: Optional[str] = None
    ):
        """
        Generate response with streaming.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            query_type: Query type
            template_name: Template name
        
        Yields:
            Text chunks as generated
        """
        # Format context
        context = self.context_formatter.format_chunks(
            chunks=retrieved_chunks,
            query=query,
            max_chunks=self.config.max_chunks
        )
        
        # Get prompt
        template = template_name or self.config.default_template
        prompt = self.prompt_manager.get_prompt(
            template_name=template,
            variables={'context': context, 'query': query},
            query_type=query_type
        )
        
        # Stream generation
        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]
        
        for chunk in self.llm_client.generate_streaming(messages):
            yield chunk
    
    def generate_batch(
        self,
        queries: List[str],
        retrieved_chunks_list: List[List[Dict[str, Any]]],
        query_types: Optional[List[str]] = None
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple queries.
        
        Args:
            queries: List of queries
            retrieved_chunks_list: List of chunk lists for each query
            query_types: Optional list of query types
        
        Returns:
            List of GenerationResults
        """
        results = []
        query_types = query_types or [None] * len(queries)
        
        for i, (query, chunks, qtype) in enumerate(
            zip(queries, retrieved_chunks_list, query_types), 1
        ):
            logger.debug(f"Generating response {i}/{len(queries)}")
            result = self.generate(query, chunks, query_type=qtype)
            results.append(result)
        
        logger.info(f"Generated {len(results)} responses")
        return results
    
    def _add_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        """
        Add citations to answer.
        
        Args:
            answer: Generated answer
            chunks: Source chunks
        
        Returns:
            Tuple of (answer with citations, list of citations)
        """
        citations = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'unknown')
            page = metadata.get('page', 'unknown')
            
            citation = self.config.citation_format.format(
                source=source,
                page=page
            )
            
            if citation not in citations:
                citations.append(citation)
        
        # Append citations to answer
        if citations:
            answer += "\n\nSources: " + ", ".join(citations)
        
        return answer, citations
    
    def _estimate_confidence(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate confidence in the answer.
        
        Args:
            answer: Generated answer
            chunks: Source chunks
        
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic-based confidence
        confidence = 0.5  # Base confidence
        
        # Higher confidence if answer is substantive
        if len(answer) > 50:
            confidence += 0.1
        
        # Higher confidence if multiple chunks support answer
        if len(chunks) >= 3:
            confidence += 0.1
        
        # Higher confidence based on chunk scores
        if chunks and 'score' in chunks[0]:
            avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks)
            confidence += min(avg_score, 0.3)
        
        # Lower confidence if answer contains uncertainty phrases
        uncertainty_phrases = [
            "i don't have enough information",
            "not sure",
            "unclear",
            "cannot determine",
            "insufficient"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence *= 0.5
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_fallback_result(
        self,
        query: str,
        start_time: float
    ) -> GenerationResult:
        """Create fallback result when no context available."""
        return GenerationResult(
            answer=self.config.fallback_response,
            query=query,
            context_used="",
            chunks_used=0,
            template_used="fallback",
            generation_time=time.time() - start_time,
            confidence=0.0
        )
    
    def _create_error_result(
        self,
        query: str,
        error_msg: str,
        start_time: float
    ) -> GenerationResult:
        """Create error result."""
        return GenerationResult(
            answer=f"Error generating response: {error_msg}",
            query=query,
            context_used="",
            chunks_used=0,
            template_used="error",
            generation_time=time.time() - start_time,
            metadata={'error': error_msg}
        )
    
    def update_config(self, **kwargs):
        """
        Update generation configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get generator statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'llm_model': self.llm_client.config.model,
            'llm_provider': self.llm_client.config.provider,
            'max_context_length': self.config.max_context_length,
            'max_chunks': self.config.max_chunks,
            'default_template': self.config.default_template,
            'available_templates': self.prompt_manager.list_templates()
        }
