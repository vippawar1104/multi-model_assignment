"""
Generation module for Multi-Modal RAG System.
Handles response generation using LLMs with retrieved context.
"""

from typing import Optional

# Conditional imports for graceful degradation
try:
    from .llm_client import LLMClient, LLMConfig
    HAS_LLM_CLIENT = True
except ImportError as e:
    HAS_LLM_CLIENT = False
    _llm_client_error = str(e)

try:
    from .context_formatter import ContextFormatter, FormattingConfig
    HAS_CONTEXT_FORMATTER = True
except ImportError as e:
    HAS_CONTEXT_FORMATTER = False
    _context_formatter_error = str(e)

try:
    from .prompt_manager import PromptManager, PromptTemplate
    HAS_PROMPT_MANAGER = True
except ImportError as e:
    HAS_PROMPT_MANAGER = False
    _prompt_manager_error = str(e)

try:
    from .response_generator import ResponseGenerator, GenerationConfig, GenerationResult
    HAS_RESPONSE_GENERATOR = True
except ImportError as e:
    HAS_RESPONSE_GENERATOR = False
    _response_generator_error = str(e)

__all__ = [
    'LLMClient',
    'LLMConfig',
    'ContextFormatter',
    'FormattingConfig',
    'PromptManager',
    'PromptTemplate',
    'ResponseGenerator',
    'GenerationConfig',
    'GenerationResult',
]


def check_dependencies() -> dict:
    """Check which generation components are available."""
    return {
        'llm_client': HAS_LLM_CLIENT,
        'context_formatter': HAS_CONTEXT_FORMATTER,
        'prompt_manager': HAS_PROMPT_MANAGER,
        'response_generator': HAS_RESPONSE_GENERATOR,
    }
