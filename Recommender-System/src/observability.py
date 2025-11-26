"""
Production-grade observability using Prometheus metrics.
Tracks latencies, throughput, errors, and system health.
"""
import time
import logging
import os
from datetime import datetime
from functools import wraps
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
from pathlib import Path

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    CollectorRegistry,
    REGISTRY,
)

logger = logging.getLogger(__name__)

# ============================================================================
# LATENCY METRICS FILE LOGGER
# ============================================================================

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LATENCY_LOG_FILE = LOGS_DIR / "latency_metrics.log"

# Configure dedicated file handler for latency metrics
latency_logger = logging.getLogger("latency_metrics")
latency_logger.setLevel(logging.INFO)
latency_logger.propagate = False  # Don't propagate to root logger

# File handler for latency metrics
latency_file_handler = logging.FileHandler(LATENCY_LOG_FILE, mode='a')
latency_file_handler.setLevel(logging.INFO)

# Format: timestamp | metric_type | component | duration | details
latency_formatter = logging.Formatter(
    '%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
latency_file_handler.setFormatter(latency_formatter)
latency_logger.addHandler(latency_file_handler)


def log_latency_to_file(
    metric_type: str,
    component: str,
    duration_ms: float,
    **kwargs
):
    """
    Log latency metrics to a dedicated file for quick access.
    
    Args:
        metric_type: Type of metric (retrieval, llm_call, entity_extraction, etc.)
        component: Component name (agent name, retriever type, etc.)
        duration_ms: Duration in milliseconds
        **kwargs: Additional metadata to log
    """
    # Format additional details
    details = " | ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
    
    # Log message format: metric_type | component | duration_ms | details
    log_message = f"{metric_type:20s} | {component:25s} | {duration_ms:8.2f}ms"
    if details:
        log_message += f" | {details}"
    
    latency_logger.info(log_message)


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

# Request Metrics
REQUEST_COUNT = Counter(
    'restaurant_search_requests_total',
    'Total number of search requests',
    ['agent', 'status']  # labels: query_understanding, retrieval, response_generation
)

REQUEST_DURATION = Histogram(
    'restaurant_search_request_duration_seconds',
    'Request processing duration in seconds',
    ['agent'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# LLM Metrics
LLM_CALL_COUNT = Counter(
    'llm_calls_total',
    'Total number of LLM API calls',
    ['agent', 'model', 'status']
)

LLM_LATENCY = Histogram(
    'llm_call_duration_seconds',
    'LLM API call duration in seconds',
    ['agent', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
)

LLM_TOKEN_USAGE = Counter(
    'llm_tokens_total',
    'Total tokens used in LLM calls',
    ['agent', 'token_type']  # token_type: prompt, completion, total
)

LLM_COST = Counter(
    'llm_cost_usd_total',
    'Estimated LLM API cost in USD',
    ['agent', 'model']
)

# Retrieval Metrics
RETRIEVAL_LATENCY = Histogram(
    'retrieval_duration_seconds',
    'Document retrieval duration in seconds',
    ['retriever_type'],  # semantic, bm25, hybrid
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)

RETRIEVAL_COUNT = Counter(
    'retrieval_operations_total',
    'Total number of retrieval operations',
    ['retriever_type', 'status']
)

DOCUMENTS_RETRIEVED = Histogram(
    'documents_retrieved_count',
    'Number of documents retrieved per query',
    ['retriever_type'],
    buckets=(0, 1, 5, 10, 20, 50, 100)
)

# Entity Extraction Metrics
ENTITY_EXTRACTION_COUNT = Counter(
    'entity_extraction_total',
    'Total entity extraction attempts',
    ['status']  # success, failure, fallback
)

ENTITY_EXTRACTION_LATENCY = Histogram(
    'entity_extraction_duration_seconds',
    'Entity extraction duration in seconds',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

ENTITIES_EXTRACTED = Histogram(
    'entities_per_query',
    'Number of entities extracted per query',
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
)

# Filter & Ranking Metrics
FILTER_LATENCY = Histogram(
    'filter_duration_seconds',
    'Entity-based filtering duration in seconds',
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

DOCUMENTS_FILTERED = Counter(
    'documents_filtered_total',
    'Documents filtered out by entity matching',
    ['filter_type']  # cuisine, location, price, amenities
)

RERANK_LATENCY = Histogram(
    'rerank_duration_seconds',
    'Result re-ranking duration in seconds',
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

# Response Generation Metrics
RESPONSE_GENERATION_LATENCY = Histogram(
    'response_generation_duration_seconds',
    'Response generation duration in seconds',
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
)

RESPONSE_LENGTH = Histogram(
    'response_length_characters',
    'Response length in characters',
    buckets=(50, 100, 200, 500, 1000, 2000, 5000)
)

# Error Metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['component', 'error_type']
)

# System Metrics
ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of requests currently being processed',
    ['agent']
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Cache operations',
    ['operation', 'result']  # operation: hit, miss, set; result: success, failure
)

VECTOR_STORE_SIZE = Gauge(
    'vector_store_documents_total',
    'Total number of documents in vector store'
)

# System Info
SYSTEM_INFO = Info(
    'restaurant_search_system',
    'System information'
)


# ============================================================================
# DECORATORS & CONTEXT MANAGERS
# ============================================================================

def track_time(metric: Histogram, labels: Optional[dict] = None):
    """
    Decorator to track execution time of a function.
    
    Usage:
        @track_time(LLM_LATENCY, {'agent': 'query_understanding', 'model': 'gpt-4'})
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                raise
        return wrapper
    return decorator


@contextmanager
def track_operation(
    metric: Histogram,
    counter: Optional[Counter] = None,
    labels: Optional[dict] = None,
    counter_labels: Optional[dict] = None
):
    """
    Context manager to track operation duration and optionally count occurrences.
    
    Usage:
        with track_operation(
            RETRIEVAL_LATENCY, 
            RETRIEVAL_COUNT,
            labels={'retriever_type': 'hybrid'},
            counter_labels={'retriever_type': 'hybrid', 'status': 'success'}
        ):
            # your operation
            ...
    """
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        
        # Record duration
        if labels:
            metric.labels(**labels).observe(duration)
        else:
            metric.observe(duration)
        
        # Increment counter on success
        if counter and counter_labels:
            counter.labels(**counter_labels).inc()
            
    except Exception as e:
        duration = time.time() - start_time
        
        # Record duration even on failure
        if labels:
            metric.labels(**labels).observe(duration)
        else:
            metric.observe(duration)
        
        # Increment error counter
        if counter and counter_labels:
            error_labels = counter_labels.copy()
            error_labels['status'] = 'error'
            counter.labels(**error_labels).inc()
        
        raise


@contextmanager
def track_active_requests(agent: str):
    """Track number of active requests for an agent."""
    ACTIVE_REQUESTS.labels(agent=agent).inc()
    try:
        yield
    finally:
        ACTIVE_REQUESTS.labels(agent=agent).dec()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_llm_call(
    agent: str,
    model: str,
    duration: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True
):
    """
    Record metrics for an LLM API call.
    
    Args:
        agent: Name of the agent making the call
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        duration: Call duration in seconds
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        success: Whether the call succeeded
    """
    status = 'success' if success else 'error'
    
    LLM_CALL_COUNT.labels(agent=agent, model=model, status=status).inc()
    LLM_LATENCY.labels(agent=agent, model=model).observe(duration)
    
    # Log latency to file
    duration_ms = duration * 1000
    log_latency_to_file(
        metric_type="llm_call",
        component=f"{agent}/{model}",
        duration_ms=duration_ms,
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens if success else None
    )
    
    if success:
        total_tokens = prompt_tokens + completion_tokens
        LLM_TOKEN_USAGE.labels(agent=agent, token_type='prompt').inc(prompt_tokens)
        LLM_TOKEN_USAGE.labels(agent=agent, token_type='completion').inc(completion_tokens)
        LLM_TOKEN_USAGE.labels(agent=agent, token_type='total').inc(total_tokens)
        
        # Estimate cost (approximate pricing)
        cost = estimate_llm_cost(model, prompt_tokens, completion_tokens)
        LLM_COST.labels(agent=agent, model=model).inc(cost)


def estimate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate LLM API cost based on model and token usage.
    Prices are approximate and should be updated based on actual pricing.
    """
    # Prices per 1K tokens (as of 2024)
    pricing = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
        'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},
        'gpt-3.5-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
    }
    
    model_pricing = pricing.get(model, {'prompt': 0.01, 'completion': 0.03})
    
    prompt_cost = (prompt_tokens / 1000) * model_pricing['prompt']
    completion_cost = (completion_tokens / 1000) * model_pricing['completion']
    
    return prompt_cost + completion_cost


def record_retrieval(
    retriever_type: str,
    duration: float,
    num_documents: int,
    success: bool = True
):
    """
    Record retrieval operation metrics.
    
    Args:
        retriever_type: Type of retriever (semantic, bm25, hybrid)
        duration: Retrieval duration in seconds
        num_documents: Number of documents retrieved
        success: Whether retrieval succeeded
    """
    status = 'success' if success else 'error'
    
    RETRIEVAL_LATENCY.labels(retriever_type=retriever_type).observe(duration)
    RETRIEVAL_COUNT.labels(retriever_type=retriever_type, status=status).inc()
    
    # Log latency to file
    duration_ms = duration * 1000
    log_latency_to_file(
        metric_type="retrieval",
        component=retriever_type,
        duration_ms=duration_ms,
        status=status,
        num_documents=num_documents
    )
    
    if success:
        DOCUMENTS_RETRIEVED.labels(retriever_type=retriever_type).observe(num_documents)


def record_entity_extraction(
    duration: float,
    num_entities: int,
    success: bool = True,
    fallback: bool = False
):
    """Record entity extraction metrics."""
    if fallback:
        status = 'fallback'
    elif success:
        status = 'success'
    else:
        status = 'failure'
    
    ENTITY_EXTRACTION_COUNT.labels(status=status).inc()
    ENTITY_EXTRACTION_LATENCY.observe(duration)
    
    # Log latency to file
    duration_ms = duration * 1000
    log_latency_to_file(
        metric_type="entity_extraction",
        component="query_understanding",
        duration_ms=duration_ms,
        status=status,
        num_entities=num_entities
    )
    
    if success:
        ENTITIES_EXTRACTED.observe(num_entities)


def record_filtering(duration: float, filtered_by: dict):
    """
    Record filtering operation metrics.
    
    Args:
        duration: Filtering duration in seconds
        filtered_by: Dict of filter types and counts, e.g., {'cuisine': 5, 'location': 3}
    """
    FILTER_LATENCY.observe(duration)
    
    for filter_type, count in filtered_by.items():
        DOCUMENTS_FILTERED.labels(filter_type=filter_type).inc(count)


def record_error(component: str, error_type: str):
    """Record an error occurrence."""
    ERROR_COUNT.labels(component=component, error_type=error_type).inc()


def record_request(agent: str, duration: float, status: str = 'success'):
    """
    Record a completed request.
    
    Args:
        agent: Agent that processed the request
        duration: Request duration in seconds
        status: Request status (success, error, timeout)
    """
    REQUEST_COUNT.labels(agent=agent, status=status).inc()
    REQUEST_DURATION.labels(agent=agent).observe(duration)


def update_vector_store_size(size: int):
    """Update the vector store size gauge."""
    VECTOR_STORE_SIZE.set(size)


def set_system_info(version: str, environment: str, **kwargs):
    """
    Set system information.
    
    Args:
        version: Application version
        environment: Environment (dev, staging, production)
        **kwargs: Additional info fields
    """
    info = {
        'version': version,
        'environment': environment,
        **kwargs
    }
    SYSTEM_INFO.info(info)


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus format.
    
    Returns:
        Metrics in Prometheus exposition format
    """
    return generate_latest(REGISTRY)


def get_metrics_text() -> str:
    """Get metrics as text string."""
    return get_metrics().decode('utf-8')


# ============================================================================
# HEALTH CHECK
# ============================================================================

class HealthChecker:
    """Simple health check tracker."""
    
    def __init__(self):
        self.last_request_time = time.time()
        self.error_count = 0
        self.success_count = 0
    
    def record_success(self):
        """Record a successful request."""
        self.last_request_time = time.time()
        self.success_count += 1
    
    def record_error(self):
        """Record a failed request."""
        self.error_count += 1
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        # System is unhealthy if:
        # 1. No requests in last 5 minutes (assuming some traffic)
        # 2. Error rate > 50%
        
        time_since_last = time.time() - self.last_request_time
        total_requests = self.success_count + self.error_count
        
        if total_requests == 0:
            return True  # No traffic yet
        
        error_rate = self.error_count / total_requests
        
        return error_rate < 0.5
    
    def get_stats(self) -> dict:
        """Get health stats."""
        total = self.success_count + self.error_count
        return {
            'success_count': self.success_count,
            'error_count': self.error_count,
            'total_requests': total,
            'error_rate': self.error_count / total if total > 0 else 0,
            'last_request_seconds_ago': time.time() - self.last_request_time,
            'healthy': self.is_healthy()
        }


# Global health checker instance
health_checker = HealthChecker()


# ============================================================================
# LOGGING INTEGRATION
# ============================================================================

def log_metrics_summary():
    """Log a summary of current metrics (useful for debugging)."""
    logger.info("=== Metrics Summary ===")
    logger.info(f"Health: {health_checker.get_stats()}")
    # Add more summary logging as needed


def get_latest_latency_metrics(num_lines: int = 50) -> str:
    """
    Get the latest latency metrics from the log file.
    
    Args:
        num_lines: Number of latest entries to return
        
    Returns:
        Formatted string with latest metrics
    """
    try:
        if not LATENCY_LOG_FILE.exists():
            return "No latency metrics logged yet."
        
        with open(LATENCY_LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        # Get last N lines
        latest_lines = lines[-num_lines:] if len(lines) > num_lines else lines
        
        result = f"=== Latest {len(latest_lines)} Latency Metrics ===\n"
        result += "Timestamp           | Metric Type          | Component                  | Duration    | Details\n"
        result += "-" * 120 + "\n"
        result += "".join(latest_lines)
        
        return result
    except Exception as e:
        logger.error(f"Error reading latency metrics: {e}")
        return f"Error reading metrics: {e}"


def get_latency_summary() -> Dict[str, Any]:
    """
    Calculate summary statistics from the latency log file.
    
    Returns:
        Dictionary with average, min, max latencies by metric type
    """
    try:
        if not LATENCY_LOG_FILE.exists():
            return {"message": "No latency metrics logged yet."}
        
        with open(LATENCY_LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        # Parse metrics
        from collections import defaultdict
        metrics_by_type = defaultdict(list)
        
        for line in lines:
            try:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    metric_type = parts[1].strip()
                    duration_str = parts[3].strip().replace('ms', '')
                    duration = float(duration_str)
                    metrics_by_type[metric_type].append(duration)
            except (ValueError, IndexError):
                continue
        
        # Calculate statistics
        summary = {}
        for metric_type, durations in metrics_by_type.items():
            if durations:
                summary[metric_type] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "latest_ms": durations[-1]
                }
        
        return summary
    except Exception as e:
        logger.error(f"Error calculating latency summary: {e}")
        return {"error": str(e)}


def clear_latency_log():
    """Clear the latency metrics log file."""
    try:
        if LATENCY_LOG_FILE.exists():
            LATENCY_LOG_FILE.unlink()
            latency_logger.info("Latency log cleared")
        logger.info("Latency metrics log cleared")
    except Exception as e:
        logger.error(f"Error clearing latency log: {e}")


if __name__ == "__main__":
    # Example usage
    set_system_info(version='1.0.0', environment='production')
    
    # Simulate some operations
    record_llm_call('query_understanding', 'gpt-4', 1.5, 100, 50, success=True)
    record_retrieval('hybrid', 0.3, 10, success=True)
    record_entity_extraction(0.5, 3, success=True)
    
    # Print metrics
    print(get_metrics_text())

