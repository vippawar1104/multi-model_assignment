"""
Evaluation module for Multi-Modal RAG System.
Handles metrics, benchmarking, and quality assessment.
"""

from typing import Optional

# Conditional imports for graceful degradation
try:
    from .metrics import RAGMetrics, MetricsConfig
    HAS_METRICS = True
except ImportError as e:
    HAS_METRICS = False
    _metrics_error = str(e)

try:
    from .ragas_evaluator import RAGASEvaluator, RAGASConfig
    HAS_RAGAS = True
except ImportError as e:
    HAS_RAGAS = False
    _ragas_error = str(e)

try:
    from .benchmark import Benchmark, BenchmarkConfig
    HAS_BENCHMARK = True
except ImportError as e:
    HAS_BENCHMARK = False
    _benchmark_error = str(e)

try:
    from .quality_assessor import QualityAssessor, QualityConfig
    HAS_QUALITY = True
except ImportError as e:
    HAS_QUALITY = False
    _quality_error = str(e)

__all__ = [
    'RAGMetrics',
    'MetricsConfig',
    'RAGASEvaluator',
    'RAGASConfig',
    'Benchmark',
    'BenchmarkConfig',
    'QualityAssessor',
    'QualityConfig',
]


def check_dependencies() -> dict:
    """Check which evaluation components are available."""
    return {
        'metrics': HAS_METRICS,
        'ragas': HAS_RAGAS,
        'benchmark': HAS_BENCHMARK,
        'quality': HAS_QUALITY,
    }
