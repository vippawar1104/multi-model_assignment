"""
LLM Client for interacting with various language models.
Supports OpenAI, Groq, and local models.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "openai"  # openai, groq, gemini, local
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 30


class LLMClient:
    """
    Unified client for interacting with different LLM providers.
    
    Supports:
    - OpenAI (GPT-3.5, GPT-4)
    - Groq (Llama, Mixtral)
    - Local models (via API-compatible endpoints)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.client = None
        self._initialize_client()
        
        logger.info(f"Initialized LLM client: {self.config.provider}/{self.config.model}")
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            
        elif provider == "groq":
            if not HAS_GROQ:
                raise ImportError("groq package not installed. Run: pip install groq")
            
            api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("No Groq API key found. Set GROQ_API_KEY environment variable.")
            
            self.client = Groq(api_key=api_key)
            
        elif provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            
            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("No Gemini API key found. Set GEMINI_API_KEY environment variable.")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.config.model)
            
        elif provider == "local":
            if not HAS_OPENAI:
                raise ImportError("openai package needed for local models. Run: pip install openai")
            
            # Use OpenAI-compatible client with custom base URL
            self.client = openai.OpenAI(
                api_key=self.config.api_key or "local",
                base_url=self.config.api_base or "http://localhost:8000/v1",
                timeout=self.config.timeout
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            **kwargs: Additional parameters for the API
        
        Returns:
            Generated text response
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")
        
        # Use provided values or fall back to config
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            # Handle Gemini differently
            if self.config.provider == "gemini":
                # Combine messages into a single prompt for Gemini
                prompt = ""
                for msg in messages:
                    if msg['role'] == 'system':
                        prompt += f"{msg['content']}\n\n"
                    elif msg['role'] == 'user':
                        prompt += f"{msg['content']}"
                
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temp,
                        max_output_tokens=max_tok,
                    )
                )
                generated_text = response.text
            else:
                # OpenAI/Groq style
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=kwargs.get('top_p', self.config.top_p),
                    frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
                    presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
                )
                generated_text = response.choices[0].message.content
            
            logger.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response with system and user prompts.
        
        Args:
            system_prompt: System message setting context/role
            user_prompt: User message with the query
            **kwargs: Additional parameters for generate()
        
        Returns:
            Generated text response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate(messages, **kwargs)
    
    def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate response with streaming.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they're generated
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")
        
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                top_p=kwargs.get('top_p', self.config.top_p),
                frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
                presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        # For production, use tiktoken library
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model info
        """
        return {
            'provider': self.config.provider,
            'model': self.config.model,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'has_client': self.client is not None
        }
