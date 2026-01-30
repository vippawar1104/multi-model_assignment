"""
Prompt Manager for handling prompt templates and engineering.
Manages system prompts, few-shot examples, and dynamic prompt construction.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
from loguru import logger


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    name: str
    system_prompt: str
    user_template: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    variables: Optional[List[str]] = None


class PromptManager:
    """
    Manages prompt templates for different query types.
    
    Features:
    - Template-based prompt generation
    - Few-shot learning examples
    - Query type specific prompts
    - Dynamic variable substitution
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering questions based on provided context.

Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
3. Be concise and accurate
4. Cite sources when possible
5. If there are multiple perspectives, present them fairly"""

    DEFAULT_USER_TEMPLATE = """Context:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            config_path: Path to prompt configuration file (YAML)
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
        
        if config_path and config_path.exists():
            self._load_templates_from_config(config_path)
        
        logger.info(f"Initialized PromptManager with {len(self.templates)} templates")
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # General QA template
        self.templates['general'] = PromptTemplate(
            name='general',
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            user_template=self.DEFAULT_USER_TEMPLATE,
            variables=['context', 'query']
        )
        
        # Technical/detailed template
        self.templates['technical'] = PromptTemplate(
            name='technical',
            system_prompt="""You are a technical expert assistant. Provide detailed, accurate answers based on the context.

Guidelines:
1. Be precise and technical when appropriate
2. Include relevant details and specifications
3. Use proper terminology
4. Explain complex concepts clearly
5. Only use information from the provided context""",
            user_template="""Technical Context:
{context}

Technical Question: {query}

Provide a detailed technical answer:""",
            variables=['context', 'query']
        )
        
        # Summarization template
        self.templates['summary'] = PromptTemplate(
            name='summary',
            system_prompt="""You are a skilled summarizer. Create concise, accurate summaries of the provided information.

Guidelines:
1. Capture key points
2. Be concise but complete
3. Maintain accuracy
4. Use clear language
5. Organize information logically""",
            user_template="""Information to summarize:
{context}

Focus: {query}

Provide a clear summary:""",
            variables=['context', 'query']
        )
        
        # Comparison template
        self.templates['comparison'] = PromptTemplate(
            name='comparison',
            system_prompt="""You are an analytical assistant specialized in comparing and contrasting concepts.

Guidelines:
1. Identify key similarities and differences
2. Use structured format (tables/lists when helpful)
3. Be objective and balanced
4. Base comparisons only on provided context
5. Highlight important distinctions""",
            user_template="""Context for comparison:
{context}

Comparison request: {query}

Provide a structured comparison:""",
            variables=['context', 'query']
        )
        
        # Multi-modal template
        self.templates['multimodal'] = PromptTemplate(
            name='multimodal',
            system_prompt="""You are an AI assistant capable of understanding both text and image content.

Guidelines:
1. Consider both textual and visual information
2. Reference specific images when relevant
3. Integrate information from multiple sources
4. Be clear about which information comes from images vs text
5. Only use provided context and images""",
            user_template="""Text Context:
{context}

Images Referenced: {image_count} images available
{image_info}

Question: {query}

Answer considering both text and visual information:""",
            variables=['context', 'query', 'image_count', 'image_info']
        )
    
    def _load_templates_from_config(self, config_path: Path):
        """
        Load templates from YAML configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for template_data in config.get('templates', []):
                template = PromptTemplate(**template_data)
                self.templates[template.name] = template
            
            logger.info(f"Loaded {len(config.get('templates', []))} templates from config")
        
        except Exception as e:
            logger.error(f"Failed to load templates from {config_path}: {str(e)}")
    
    def get_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any],
        query_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate prompt from template.
        
        Args:
            template_name: Name of template to use
            variables: Variables to substitute in template
            query_type: Query type for template selection (overrides template_name)
        
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Use query type to select template if provided
        if query_type:
            template_name = self._select_template_for_query_type(query_type)
        
        template = self.templates.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found, using 'general'")
            template = self.templates['general']
        
        try:
            user_prompt = template.user_template.format(**variables)
            
            return {
                'system': template.system_prompt,
                'user': user_prompt
            }
        
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            raise ValueError(f"Missing required variable: {e}")
    
    def _select_template_for_query_type(self, query_type: str) -> str:
        """
        Select appropriate template based on query type.
        
        Args:
            query_type: Type of query (from query processor)
        
        Returns:
            Template name
        """
        type_mapping = {
            'question': 'general',
            'definition': 'technical',
            'comparison': 'comparison',
            'summary': 'summary',
            'explanation': 'technical',
            'general': 'general',
            'multimodal': 'multimodal'
        }
        
        return type_mapping.get(query_type, 'general')
    
    def add_few_shot_examples(
        self,
        template_name: str,
        examples: List[Dict[str, str]]
    ):
        """
        Add few-shot examples to a template.
        
        Args:
            template_name: Template to add examples to
            examples: List of example dicts with 'query' and 'answer'
        """
        if template_name not in self.templates:
            logger.warning(f"Template '{template_name}' not found")
            return
        
        self.templates[template_name].few_shot_examples = examples
        logger.debug(f"Added {len(examples)} examples to '{template_name}'")
    
    def format_with_examples(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Format prompt with few-shot examples.
        
        Args:
            template_name: Template name
            variables: Variables for prompt
        
        Returns:
            Formatted prompt with examples
        """
        template = self.templates.get(template_name, self.templates['general'])
        
        # Get base prompt
        prompt = self.get_prompt(template_name, variables)
        
        # Add few-shot examples if available
        if template.few_shot_examples:
            examples_text = "\n\nExamples:\n"
            for i, example in enumerate(template.few_shot_examples, 1):
                examples_text += f"\nExample {i}:\n"
                examples_text += f"Q: {example['query']}\n"
                examples_text += f"A: {example['answer']}\n"
            
            prompt['user'] = examples_text + "\n---\n" + prompt['user']
        
        return prompt
    
    def create_custom_prompt(
        self,
        system_prompt: str,
        context: str,
        query: str,
        additional_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create a custom prompt without using templates.
        
        Args:
            system_prompt: System prompt
            context: Context information
            query: User query
            additional_instructions: Optional additional instructions
        
        Returns:
            Prompt dictionary
        """
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        if additional_instructions:
            user_prompt += f"\n\nAdditional Instructions:\n{additional_instructions}"
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }
    
    def list_templates(self) -> List[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a template.
        
        Args:
            template_name: Name of template
        
        Returns:
            Template information dictionary
        """
        template = self.templates.get(template_name)
        if not template:
            return None
        
        return {
            'name': template.name,
            'variables': template.variables,
            'has_examples': bool(template.few_shot_examples),
            'example_count': len(template.few_shot_examples) if template.few_shot_examples else 0
        }
