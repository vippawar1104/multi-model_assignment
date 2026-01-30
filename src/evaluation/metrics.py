"""
RAG Metrics - Comprehensive evaluation metrics for RAG systems.
Includes retrieval metrics, generation metrics, and end-to-end evaluation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    calculate_retrieval_metrics: bool = True
    calculate_generation_metrics: bool = True
    calculate_e2e_metrics: bool = True
    relevance_threshold: float = 0.5


@dataclass
class MetricsResult:
    """Results from metrics evaluation."""
    # Retrieval metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg: Optional[float] = None  # Normalized Discounted Cumulative Gain
    map_score: Optional[float] = None  # Mean Average Precision
    
    # Generation metrics
    answer_relevance: Optional[float] = None
    answer_completeness: Optional[float] = None
    citation_accuracy: Optional[float] = None
    
    # End-to-end metrics
    latency: Optional[float] = None
    throughput: Optional[float] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGMetrics:
    """
    Comprehensive metrics calculator for RAG systems.
    
    Features:
    - Retrieval metrics (precision, recall, F1, MRR, NDCG)
    - Generation metrics (relevance, completeness, citations)
    - End-to-end performance metrics
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        logger.info("Initialized RAGMetrics")
    
    def calculate_precision(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate precision: relevant retrieved / total retrieved.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
        
        Returns:
            Precision score (0-1)
        """
        if not retrieved:
            return 0.0
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved)
        
        return precision
    
    def calculate_recall(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate recall: relevant retrieved / total relevant.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
        
        Returns:
            Recall score (0-1)
        """
        if not relevant:
            return 0.0
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set & relevant_set)
        recall = true_positives / len(relevant)
        
        return recall
    
    def calculate_f1(
        self,
        precision: float,
        recall: float
    ) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_mrr(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_lists: List of retrieved document lists for each query
            relevant_lists: List of relevant document lists for each query
        
        Returns:
            MRR score (0-1)
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return float(mrr)
    
    def calculate_ndcg(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved: List of retrieved document IDs
            relevance_scores: Dict mapping doc IDs to relevance scores
            k: Cut-off rank (None for all)
        
        Returns:
            NDCG score (0-1)
        """
        if not retrieved:
            return 0.0
        
        k = k or len(retrieved)
        retrieved = retrieved[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved, 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_scores, 1))
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return float(ndcg)
    
    def calculate_map(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]]
    ) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            retrieved_lists: List of retrieved document lists
            relevant_lists: List of relevant document lists
        
        Returns:
            MAP score (0-1)
        """
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            if not relevant:
                continue
            
            relevant_set = set(relevant)
            precisions = []
            num_relevant = 0
            
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    num_relevant += 1
                    precision = num_relevant / i
                    precisions.append(precision)
            
            if precisions:
                avg_precision = np.mean(precisions)
                average_precisions.append(avg_precision)
        
        map_score = np.mean(average_precisions) if average_precisions else 0.0
        return float(map_score)
    
    def evaluate_retrieval(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation.
        
        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            relevance_scores: Optional relevance scores for NDCG
        
        Returns:
            Dictionary of retrieval metrics
        """
        precision = self.calculate_precision(retrieved, relevant)
        recall = self.calculate_recall(retrieved, relevant)
        f1 = self.calculate_f1(precision, recall)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        # Add MRR
        mrr = self.calculate_mrr([retrieved], [relevant])
        metrics['mrr'] = mrr
        
        # Add NDCG if relevance scores provided
        if relevance_scores:
            ndcg = self.calculate_ndcg(retrieved, relevance_scores)
            metrics['ndcg'] = ndcg
        
        logger.debug(f"Retrieval metrics: {metrics}")
        return metrics
    
    def evaluate_answer_relevance(
        self,
        answer: str,
        query: str,
        context: str
    ) -> float:
        """
        Evaluate answer relevance to query and context.
        Simple heuristic-based implementation.
        
        Args:
            answer: Generated answer
            query: User query
            context: Retrieved context
        
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword overlap heuristic
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Score based on query term coverage
        query_coverage = len(query_words & answer_words) / len(query_words) if query_words else 0
        
        # Score based on context usage
        context_usage = len(context_words & answer_words) / len(answer_words) if answer_words else 0
        
        # Combined score
        relevance = 0.6 * query_coverage + 0.4 * context_usage
        
        return min(relevance, 1.0)
    
    def evaluate_citation_accuracy(
        self,
        citations: List[str],
        context_sources: List[str]
    ) -> float:
        """
        Evaluate accuracy of citations.
        
        Args:
            citations: Citations in the answer
            context_sources: Actual sources used
        
        Returns:
            Citation accuracy (0-1)
        """
        if not citations:
            return 0.0 if context_sources else 1.0
        
        citation_set = set(citations)
        source_set = set(context_sources)
        
        # Accuracy = correct citations / total citations
        correct = len(citation_set & source_set)
        accuracy = correct / len(citations)
        
        return accuracy
    
    def evaluate_generation(
        self,
        answer: str,
        query: str,
        context: str,
        citations: List[str],
        context_sources: List[str]
    ) -> Dict[str, float]:
        """
        Comprehensive generation evaluation.
        
        Args:
            answer: Generated answer
            query: User query
            context: Retrieved context
            citations: Answer citations
            context_sources: Actual sources
        
        Returns:
            Dictionary of generation metrics
        """
        relevance = self.evaluate_answer_relevance(answer, query, context)
        citation_acc = self.evaluate_citation_accuracy(citations, context_sources)
        
        # Completeness based on answer length relative to context
        completeness = min(len(answer) / max(len(context) * 0.3, 50), 1.0)
        
        metrics = {
            'answer_relevance': relevance,
            'answer_completeness': completeness,
            'citation_accuracy': citation_acc,
        }
        
        logger.debug(f"Generation metrics: {metrics}")
        return metrics
    
    def evaluate_e2e(
        self,
        retrieved: List[str],
        relevant: List[str],
        answer: str,
        query: str,
        context: str,
        latency: float,
        citations: Optional[List[str]] = None,
        context_sources: Optional[List[str]] = None
    ) -> MetricsResult:
        """
        End-to-end evaluation of RAG pipeline.
        
        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            answer: Generated answer
            query: User query
            context: Retrieved context
            latency: Total latency in seconds
            citations: Answer citations
            context_sources: Actual sources
        
        Returns:
            Complete metrics result
        """
        result = MetricsResult()
        
        # Retrieval metrics
        if self.config.calculate_retrieval_metrics:
            retrieval = self.evaluate_retrieval(retrieved, relevant)
            result.precision = retrieval['precision']
            result.recall = retrieval['recall']
            result.f1_score = retrieval['f1_score']
            result.mrr = retrieval['mrr']
            result.ndcg = retrieval.get('ndcg')
        
        # Generation metrics
        if self.config.calculate_generation_metrics and citations and context_sources:
            generation = self.evaluate_generation(
                answer, query, context, citations, context_sources
            )
            result.answer_relevance = generation['answer_relevance']
            result.answer_completeness = generation['answer_completeness']
            result.citation_accuracy = generation['citation_accuracy']
        
        # Performance metrics
        if self.config.calculate_e2e_metrics:
            result.latency = latency
            result.throughput = 1.0 / latency if latency > 0 else 0
        
        logger.info(f"E2E evaluation complete: {result}")
        return result
