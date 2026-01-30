"""
Context Formatter for preparing retrieved chunks for LLM input.
Handles formatting, truncation, and metadata inclusion.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FormattingConfig:
    """Configuration for context formatting."""
    max_context_length: int = 4000  # Max characters in context
    include_metadata: bool = True
    include_scores: bool = False
    chunk_separator: str = "\n\n---\n\n"
    metadata_template: str = "[Source: {source}, Page: {page}]"
    truncation_strategy: str = "smart"  # smart, tail, head


class ContextFormatter:
    """
    Formats retrieved chunks into context for LLM input.
    
    Features:
    - Smart truncation to fit token limits
    - Metadata formatting
    - Score inclusion
    - Multi-modal content handling
    """
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        """
        Initialize context formatter.
        
        Args:
            config: Formatting configuration
        """
        self.config = config or FormattingConfig()
        logger.info("Initialized ContextFormatter")
    
    def format_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of retrieved chunks with metadata
            query: Original query (for context)
            max_chunks: Maximum number of chunks to include
        
        Returns:
            Formatted context string
        """
        if not chunks:
            logger.warning("No chunks provided for formatting")
            return ""
        
        # Limit number of chunks if specified
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        formatted_chunks = []
        total_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            formatted_chunk = self._format_single_chunk(chunk, index=i)
            chunk_length = len(formatted_chunk)
            
            # Check if adding this chunk exceeds max length
            if total_length + chunk_length > self.config.max_context_length:
                if self.config.truncation_strategy == "smart":
                    # Try to fit partial chunk
                    remaining = self.config.max_context_length - total_length
                    if remaining > 100:  # Only if we have reasonable space
                        formatted_chunk = formatted_chunk[:remaining] + "..."
                        formatted_chunks.append(formatted_chunk)
                break
            
            formatted_chunks.append(formatted_chunk)
            total_length += chunk_length
        
        context = self.config.chunk_separator.join(formatted_chunks)
        
        logger.debug(
            f"Formatted {len(formatted_chunks)}/{len(chunks)} chunks, "
            f"{len(context)} characters"
        )
        
        return context
    
    def _format_single_chunk(self, chunk: Dict[str, Any], index: int) -> str:
        """
        Format a single chunk with metadata.
        
        Args:
            chunk: Chunk dictionary
            index: Chunk index in results
        
        Returns:
            Formatted chunk string
        """
        parts = []
        
        # Add metadata header if enabled
        if self.config.include_metadata:
            metadata_str = self._format_metadata(chunk.get('metadata', {}), index)
            if metadata_str:
                parts.append(metadata_str)
        
        # Add score if enabled
        if self.config.include_scores and 'score' in chunk:
            parts.append(f"Relevance: {chunk['score']:.4f}")
        
        # Add main content
        content = chunk.get('text', chunk.get('content', ''))
        parts.append(content)
        
        return "\n".join(parts)
    
    def _format_metadata(self, metadata: Dict[str, Any], index: int) -> str:
        """
        Format metadata for a chunk.
        
        Args:
            metadata: Metadata dictionary
            index: Chunk index
        
        Returns:
            Formatted metadata string
        """
        if not metadata:
            return f"[Chunk {index}]"
        
        # Try to use template
        try:
            formatted = self.config.metadata_template.format(**metadata)
            return f"[Chunk {index}] {formatted}"
        except KeyError:
            # Fallback to simple formatting
            meta_parts = [f"{k}: {v}" for k, v in metadata.items() 
                         if k not in ['embedding', 'id']]
            if meta_parts:
                return f"[Chunk {index}] " + ", ".join(meta_parts)
            return f"[Chunk {index}]"
    
    def format_with_images(
        self,
        text_chunks: List[Dict[str, Any]],
        image_chunks: List[Dict[str, Any]],
        max_images: int = 3
    ) -> Dict[str, Any]:
        """
        Format context with both text and images.
        
        Args:
            text_chunks: Text chunks
            image_chunks: Image chunks
            max_images: Maximum number of images to include
        
        Returns:
            Dictionary with formatted text and image references
        """
        # Format text context
        text_context = self.format_chunks(text_chunks)
        
        # Format image references
        image_refs = []
        for i, img_chunk in enumerate(image_chunks[:max_images], 1):
            metadata = img_chunk.get('metadata', {})
            image_refs.append({
                'index': i,
                'path': img_chunk.get('path', ''),
                'caption': metadata.get('caption', f'Image {i}'),
                'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page', 'unknown')
            })
        
        return {
            'text_context': text_context,
            'images': image_refs,
            'image_count': len(image_refs)
        }
    
    def truncate_context(
        self,
        context: str,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> str:
        """
        Truncate context to fit length limit.
        
        Args:
            context: Context string
            max_length: Maximum length (uses config if not provided)
            strategy: Truncation strategy (uses config if not provided)
        
        Returns:
            Truncated context
        """
        max_len = max_length or self.config.max_context_length
        strat = strategy or self.config.truncation_strategy
        
        if len(context) <= max_len:
            return context
        
        if strat == "head":
            # Keep beginning
            return context[:max_len] + "..."
        
        elif strat == "tail":
            # Keep ending
            return "..." + context[-max_len:]
        
        elif strat == "smart":
            # Try to keep complete sentences
            truncated = context[:max_len]
            
            # Find last sentence boundary
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > max_len * 0.8:  # If we're keeping most of it
                return truncated[:boundary + 1]
            else:
                return truncated + "..."
        
        return context[:max_len] + "..."
    
    def add_query_context(
        self,
        formatted_context: str,
        query: str,
        include_query: bool = True
    ) -> str:
        """
        Add query information to context.
        
        Args:
            formatted_context: Already formatted context
            query: User query
            include_query: Whether to include query in output
        
        Returns:
            Context with query information
        """
        if not include_query:
            return formatted_context
        
        query_section = f"User Question: {query}\n\nRelevant Information:\n"
        return query_section + formatted_context
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate
        
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        return len(text) // 4
    
    def split_for_token_limit(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> List[List[Dict[str, Any]]]:
        """
        Split chunks into groups that fit token limit.
        
        Args:
            chunks: List of chunks
            max_tokens: Maximum tokens per group
        
        Returns:
            List of chunk groups
        """
        groups = []
        current_group = []
        current_tokens = 0
        
        for chunk in chunks:
            content = chunk.get('text', chunk.get('content', ''))
            chunk_tokens = self.estimate_tokens(content)
            
            if current_tokens + chunk_tokens > max_tokens and current_group:
                # Start new group
                groups.append(current_group)
                current_group = [chunk]
                current_tokens = chunk_tokens
            else:
                current_group.append(chunk)
                current_tokens += chunk_tokens
        
        if current_group:
            groups.append(current_group)
        
        logger.debug(f"Split {len(chunks)} chunks into {len(groups)} groups")
        return groups
