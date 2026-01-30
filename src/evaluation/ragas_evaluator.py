"""
RAGAS Evaluator - Advanced RAG evaluation using RAGAS framework concepts.
Implements faithfulness, answer relevance, and context relevance metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class RAGASConfig:
    """Configuration for RAGAS evaluation."""
    evaluate_faithfulness: bool = True
    evaluate_answer_relevance: bool = True
    evaluate_context_relevance: bool = True
    evaluate_context_recall: bool = True
    min_score_threshold: float = 0.5


@dataclass
class RAGASResult:
    """RAGAS evaluation results."""
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    details: Dict[str, Any] = None


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG systems.
    
    Metrics:
    - Faithfulness: Answer groundedness in context
    - Answer Relevance: Relevance to user query
    - Context Relevance: Retrieved context relevance
    - Context Recall: Coverage of ground truth in context
    """
    
    def __init__(self, config: Optional[RAGASConfig] = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: RAGAS configuration
        """
        self.config = config or RAGASConfig()
        logger.info("Initialized RAGASEvaluator")
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate if answer is grounded in context (no hallucinations).
        
        Args:
            answer: Generated answer
            context: Retrieved context
        
        Returns:
            Faithfulness score (0-1)
        """
        # Extract claims from answer (simple sentence splitting)
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        # Check how many claims are supported by context
        context_lower = context.lower()
        supported_count = 0
        
        for sentence in answer_sentences:
            # Simple keyword overlap check
            words = set(sentence.lower().split())
            # Remove common words
            words = words - {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            
            if not words:
                continue
            
            # Check if significant words appear in context
            overlap = sum(1 for w in words if w in context_lower)
            if overlap / len(words) > 0.5:  # More than 50% overlap
                supported_count += 1
        
        faithfulness = supported_count / len(answer_sentences)
        
        logger.debug(f"Faithfulness: {faithfulness:.2f} ({supported_count}/{len(answer_sentences)} claims supported)")
        return faithfulness
    
    def evaluate_answer_relevance(
        self,
        answer: str,
        query: str
    ) -> float:
        """
        Evaluate relevance of answer to query.
        
        Args:
            answer: Generated answer
            query: User query
        
        Returns:
            Answer relevance score (0-1)
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_words = query_words - stopwords
        answer_words = answer_words - stopwords
        
        if not query_words:
            return 0.5  # Neutral if no content words in query
        
        # Calculate overlap
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words)
        
        # Bonus for answer length (not too short, not too long)
        length_score = min(len(answer) / 200, 1.0)  # Optimal around 200 chars
        if len(answer) > 500:
            length_score *= 0.9  # Slight penalty for very long answers
        
        # Combined score
        total_relevance = 0.7 * relevance + 0.3 * length_score
        
        logger.debug(f"Answer relevance: {total_relevance:.2f}")
        return min(total_relevance, 1.0)
    
    def evaluate_context_relevance(
        self,
        context: str,
        query: str
    ) -> float:
        """
        Evaluate relevance of retrieved context to query.
        
        Args:
            context: Retrieved context
            query: User query
        
        Returns:
            Context relevance score (0-1)
        """
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_words = query_words - stopwords
        
        if not query_words:
            return 0.5
        
        # Calculate term coverage
        coverage = len(query_words & context_words) / len(query_words)
        
        # Factor in context density (relevant terms / total terms)
        if context_words:
            density = len(query_words & context_words) / len(context_words)
        else:
            density = 0
        
        # Combined score (favor coverage more)
        relevance = 0.7 * coverage + 0.3 * min(density * 20, 1.0)
        
        logger.debug(f"Context relevance: {relevance:.2f}")
        return min(relevance, 1.0)
    
    def evaluate_context_recall(
        self,
        context: str,
        ground_truth: str
    ) -> float:
        """
        Evaluate how well context covers ground truth information.
        
        Args:
            context: Retrieved context
            ground_truth: Ground truth answer/information
        
        Returns:
            Context recall score (0-1)
        """
        gt_words = set(ground_truth.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        gt_words = gt_words - stopwords
        
        if not gt_words:
            return 1.0  # Perfect recall if no ground truth
        
        # How many ground truth terms appear in context
        recall = len(gt_words & context_words) / len(gt_words)
        
        logger.debug(f"Context recall: {recall:.2f}")
        return recall
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> RAGASResult:
        """
        Comprehensive RAGAS evaluation.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth for recall
        
        Returns:
            RAGAS evaluation result
        """
        result = RAGASResult(details={})
        scores = []
        
        # Faithfulness
        if self.config.evaluate_faithfulness:
            result.faithfulness = self.evaluate_faithfulness(answer, context)
            scores.append(result.faithfulness)
            result.details['faithfulness_status'] = 'pass' if result.faithfulness >= self.config.min_score_threshold else 'fail'
        
        # Answer Relevance
        if self.config.evaluate_answer_relevance:
            result.answer_relevance = self.evaluate_answer_relevance(answer, query)
            scores.append(result.answer_relevance)
            result.details['answer_relevance_status'] = 'pass' if result.answer_relevance >= self.config.min_score_threshold else 'fail'
        
        # Context Relevance
        if self.config.evaluate_context_relevance:
            result.context_relevance = self.evaluate_context_relevance(context, query)
            scores.append(result.context_relevance)
            result.details['context_relevance_status'] = 'pass' if result.context_relevance >= self.config.min_score_threshold else 'fail'
        
        # Context Recall (if ground truth provided)
        if self.config.evaluate_context_recall and ground_truth:
            result.context_recall = self.evaluate_context_recall(context, ground_truth)
            scores.append(result.context_recall)
            result.details['context_recall_status'] = 'pass' if result.context_recall >= self.config.min_score_threshold else 'fail'
        
        # Overall score (average of all metrics)
        if scores:
            result.overall_score = sum(scores) / len(scores)
            result.details['overall_status'] = 'pass' if result.overall_score >= self.config.min_score_threshold else 'fail'
        
        logger.info(f"RAGAS evaluation: overall={result.overall_score:.2f}")
        return result
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[RAGASResult]:
        """
        Evaluate multiple query-answer pairs.
        
        Args:
            queries: List of queries
            answers: List of generated answers
            contexts: List of retrieved contexts
            ground_truths: Optional list of ground truths
        
        Returns:
            List of RAGAS results
        """
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        
        results = []
        for i, (q, a, c, gt) in enumerate(zip(queries, answers, contexts, ground_truths), 1):
            logger.debug(f"Evaluating query {i}/{len(queries)}")
            result = self.evaluate(q, a, c, gt)
            results.append(result)
        
        logger.info(f"Batch evaluation complete: {len(results)} queries evaluated")
        return results
    
    def get_aggregate_scores(
        self,
        results: List[RAGASResult]
    ) -> Dict[str, float]:
        """
        Calculate aggregate scores across multiple evaluations.
        
        Args:
            results: List of RAGAS results
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {}
        
        aggregates = {}
        
        # Average each metric
        metrics = ['faithfulness', 'answer_relevance', 'context_relevance', 'context_recall', 'overall_score']
        
        for metric in metrics:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
            if values:
                aggregates[f'avg_{metric}'] = sum(values) / len(values)
                aggregates[f'min_{metric}'] = min(values)
                aggregates[f'max_{metric}'] = max(values)
        
        return aggregates
