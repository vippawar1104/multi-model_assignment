"""
Quality Assessor - Assess answer quality including hallucination detection.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    check_hallucinations: bool = True
    check_completeness: bool = True
    check_coherence: bool = True
    min_quality_score: float = 0.6


@dataclass
class QualityResult:
    """Quality assessment result."""
    overall_quality: float
    hallucination_score: Optional[float] = None
    completeness_score: Optional[float] = None
    coherence_score: Optional[float] = None
    issues: List[str] = None
    passed: bool = False


class QualityAssessor:
    """
    Assess quality of generated answers.
    
    Features:
    - Hallucination detection
    - Completeness checking
    - Coherence analysis
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality assessor."""
        self.config = config or QualityConfig()
        logger.info("Initialized QualityAssessor")
    
    def detect_hallucinations(
        self,
        answer: str,
        context: str
    ) -> tuple[float, List[str]]:
        """
        Detect potential hallucinations in answer.
        
        Args:
            answer: Generated answer
            context: Source context
        
        Returns:
            Tuple of (score, list of issues)
        """
        issues = []
        
        # Extract factual statements
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if not sentences:
            return 1.0, []
        
        context_lower = context.lower()
        supported = 0
        
        for sentence in sentences:
            # Check for specific claims
            words = set(sentence.lower().split())
            words = words - {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are'}
            
            if not words:
                supported += 1  # Empty/filler sentence
                continue
            
            # Calculate word overlap
            overlap = sum(1 for w in words if w in context_lower)
            overlap_ratio = overlap / len(words) if words else 0
            
            if overlap_ratio < 0.3:
                issues.append(f"Low context support: '{sentence[:50]}...'")
            else:
                supported += 1
        
        score = supported / len(sentences) if sentences else 1.0
        
        return score, issues
    
    def assess_completeness(
        self,
        answer: str,
        query: str
    ) -> float:
        """
        Assess if answer adequately addresses query.
        
        Args:
            answer: Generated answer
            query: User query
        
        Returns:
            Completeness score (0-1)
        """
        # Check answer length (not too short)
        if len(answer) < 50:
            return 0.3
        
        # Extract key terms from query
        query_words = set(query.lower().split())
        query_words = query_words - {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        
        if not query_words:
            return 0.5
        
        # Check coverage of query terms
        answer_lower = answer.lower()
        covered = sum(1 for w in query_words if w in answer_lower)
        coverage = covered / len(query_words)
        
        # Factor in answer structure (multiple sentences good)
        sentences = len([s for s in re.split(r'[.!?]', answer) if s.strip()])
        structure_score = min(sentences / 3, 1.0)  # Optimal: 3+ sentences
        
        completeness = 0.7 * coverage + 0.3 * structure_score
        return min(completeness, 1.0)
    
    def assess_coherence(
        self,
        answer: str
    ) -> float:
        """
        Assess coherence and readability of answer.
        
        Args:
            answer: Generated answer
        
        Returns:
            Coherence score (0-1)
        """
        if not answer:
            return 0.0
        
        # Check for complete sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check average sentence length (not too short or too long)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 1.0 if 10 <= avg_length <= 30 else 0.7
        
        # Check for transition/connector words
        connectors = ['however', 'moreover', 'furthermore', 'additionally', 'therefore', 'thus', 'hence', 'also']
        has_connectors = any(conn in answer.lower() for conn in connectors)
        connector_score = 1.0 if has_connectors else 0.8
        
        # Check for proper capitalization and punctuation
        proper_caps = answer[0].isupper() if answer else False
        ends_properly = answer[-1] in '.!?' if answer else False
        format_score = 1.0 if (proper_caps and ends_properly) else 0.8
        
        coherence = (length_score + connector_score + format_score) / 3
        return coherence
    
    def assess(
        self,
        answer: str,
        query: str,
        context: str
    ) -> QualityResult:
        """
        Comprehensive quality assessment.
        
        Args:
            answer: Generated answer
            query: User query
            context: Source context
        
        Returns:
            Quality assessment result
        """
        scores = []
        all_issues = []
        
        # Hallucination detection
        hallucination_score = None
        if self.config.check_hallucinations:
            hallucination_score, issues = self.detect_hallucinations(answer, context)
            scores.append(hallucination_score)
            all_issues.extend(issues)
        
        # Completeness
        completeness_score = None
        if self.config.check_completeness:
            completeness_score = self.assess_completeness(answer, query)
            scores.append(completeness_score)
            if completeness_score < 0.5:
                all_issues.append("Answer may be incomplete")
        
        # Coherence
        coherence_score = None
        if self.config.check_coherence:
            coherence_score = self.assess_coherence(answer)
            scores.append(coherence_score)
            if coherence_score < 0.6:
                all_issues.append("Answer may lack coherence")
        
        # Overall quality
        overall_quality = sum(scores) / len(scores) if scores else 0.0
        passed = overall_quality >= self.config.min_quality_score
        
        result = QualityResult(
            overall_quality=overall_quality,
            hallucination_score=hallucination_score,
            completeness_score=completeness_score,
            coherence_score=coherence_score,
            issues=all_issues,
            passed=passed
        )
        
        logger.info(f"Quality assessment: {overall_quality:.2f} ({'PASS' if passed else 'FAIL'})")
        return result
