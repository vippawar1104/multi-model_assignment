"""
Preprocessing module for the Multi-Modal RAG system.
Contains processors for text, images, tables, and charts.
"""

# Import only the basic modules that work
from .text_chunker import TextChunker, chunk_text_simple, chunk_file_simple

# Conditional imports for modules with complex dependencies
try:
    from .table_processor import TableProcessor, process_table_simple, convert_table_to_markdown
    _table_available = True
except (ImportError, ValueError) as e:
    print(f"Warning: Table processor not available: {e}")
    _table_available = False

try:
    from .image_processor import ImageProcessor, process_image_simple, extract_image_features_simple
    _image_available = True
except (ImportError, ValueError) as e:
    print(f"Warning: Image processor not available: {e}")
    _image_available = False

try:
    from .chart_processor import ChartProcessor, process_chart_simple, detect_chart_type_simple
    _chart_available = True
except (ImportError, ValueError) as e:
    print(f"Warning: Chart processor not available: {e}")
    _chart_available = False

__all__ = [
    # Text processing (always available)
    "TextChunker",
    "chunk_text_simple",
    "chunk_file_simple",
]

if _table_available:
    __all__.extend([
        "TableProcessor",
        "process_table_simple",
        "convert_table_to_markdown"
    ])

if _image_available:
    __all__.extend([
        "ImageProcessor",
        "process_image_simple",
        "extract_image_features_simple"
    ])

if _chart_available:
    __all__.extend([
        "ChartProcessor",
        "process_chart_simple",
        "detect_chart_type_simple"
    ])
