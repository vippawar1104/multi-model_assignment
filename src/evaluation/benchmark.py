"""
Benchmark - End-to-end benchmarking framework for RAG systems.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
from loguru import logger


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_warmup_runs: int = 2
    num_test_runs: int = 10
    measure_latency: bool = True
    measure_throughput: bool = True
    measure_accuracy: bool = True


@dataclass
class BenchmarkResult:
    """Benchmarking results."""
    avg_latency: float
    min_latency: float
    max_latency: float
    throughput: float
    avg_accuracy: Optional[float] = None
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    End-to-end benchmarking for RAG systems.
    
    Features:
    - Latency measurement
    - Throughput calculation
    - Accuracy tracking
    - Performance profiling
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark."""
        self.config = config or BenchmarkConfig()
        logger.info("Initialized Benchmark")
    
    def measure_latency(
        self,
        func: callable,
        *args,
        **kwargs
    ) -> tuple[Any, float]:
        """
        Measure function execution time.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, latency in seconds)
        """
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        
        return result, latency
    
    def run_benchmark(
        self,
        func: callable,
        test_cases: List[Dict[str, Any]],
        accuracy_func: Optional[callable] = None
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark.
        
        Args:
            func: Function to benchmark (e.g., RAG pipeline)
            test_cases: List of test case dictionaries
            accuracy_func: Optional function to calculate accuracy
        
        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {len(test_cases)} test cases")
        
        latencies = []
        accuracies = []
        successful = 0
        failed = 0
        
        # Warmup runs
        if self.config.num_warmup_runs > 0:
            logger.info(f"Running {self.config.num_warmup_runs} warmup iterations")
            for i in range(min(self.config.num_warmup_runs, len(test_cases))):
                try:
                    func(**test_cases[i])
                except Exception as e:
                    logger.warning(f"Warmup run {i+1} failed: {e}")
        
        # Benchmark runs
        logger.info("Running benchmark iterations")
        for i, test_case in enumerate(test_cases, 1):
            try:
                result, latency = self.measure_latency(func, **test_case)
                latencies.append(latency)
                successful += 1
                
                # Calculate accuracy if function provided
                if accuracy_func and self.config.measure_accuracy:
                    accuracy = accuracy_func(result, test_case)
                    accuracies.append(accuracy)
                
                logger.debug(f"Test {i}/{len(test_cases)}: latency={latency:.3f}s")
                
            except Exception as e:
                logger.error(f"Test {i}/{len(test_cases)} failed: {e}")
                failed += 1
        
        # Calculate metrics
        if not latencies:
            logger.error("No successful benchmark runs")
            return BenchmarkResult(
                avg_latency=0,
                min_latency=0,
                max_latency=0,
                throughput=0,
                total_queries=len(test_cases),
                successful_queries=0,
                failed_queries=failed
            )
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = len(latencies) / sum(latencies) if sum(latencies) > 0 else 0
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        
        result = BenchmarkResult(
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            throughput=throughput,
            avg_accuracy=avg_accuracy,
            total_queries=len(test_cases),
            successful_queries=successful,
            failed_queries=failed,
            details={
                'latency_std': self._calculate_std(latencies),
                'latency_p50': self._calculate_percentile(latencies, 50),
                'latency_p95': self._calculate_percentile(latencies, 95),
                'latency_p99': self._calculate_percentile(latencies, 99),
                'success_rate': successful / len(test_cases) if test_cases else 0
            }
        )
        
        logger.info(f"Benchmark complete: {successful}/{len(test_cases)} successful")
        logger.info(f"Avg latency: {avg_latency:.3f}s, Throughput: {throughput:.2f} qps")
        
        return result
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def compare_benchmarks(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        Compare two benchmark results.
        
        Args:
            baseline: Baseline benchmark result
            current: Current benchmark result
        
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        # Latency comparison
        latency_change = ((current.avg_latency - baseline.avg_latency) / baseline.avg_latency * 100
                         if baseline.avg_latency > 0 else 0)
        comparison['latency_change_pct'] = latency_change
        comparison['latency_improved'] = latency_change < 0
        
        # Throughput comparison
        throughput_change = ((current.throughput - baseline.throughput) / baseline.throughput * 100
                            if baseline.throughput > 0 else 0)
        comparison['throughput_change_pct'] = throughput_change
        comparison['throughput_improved'] = throughput_change > 0
        
        # Accuracy comparison
        if baseline.avg_accuracy and current.avg_accuracy:
            accuracy_change = ((current.avg_accuracy - baseline.avg_accuracy) / baseline.avg_accuracy * 100)
            comparison['accuracy_change_pct'] = accuracy_change
            comparison['accuracy_improved'] = accuracy_change > 0
        
        logger.info(f"Benchmark comparison: latency {latency_change:+.1f}%, throughput {throughput_change:+.1f}%")
        return comparison
