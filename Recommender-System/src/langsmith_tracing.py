"""
LangSmith Observability Integration for the Restaurant Recommender System.

This module provides comprehensive tracing capabilities for:
- LLM calls (query understanding, response generation)
- RAG operations (retrieval, hybrid search)
- Data ingestion (document loading, embedding generation)

LangSmith Dashboard: https://smith.langchain.com/
"""
import os
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime
from contextlib import contextmanager

from langsmith import Client, traceable
from langsmith.run_trees import RunTree
from langsmith.run_helpers import get_current_run_tree, as_runnable

from src.config import LANGSMITH_CONFIG

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])


class LangSmithTracer:
    """
    Centralized LangSmith tracing manager.
    
    Provides:
    - Automatic trace initialization
    - Custom run tracking for non-LangChain operations
    - Metadata enrichment
    - Error handling with trace context
    """
    
    _instance: Optional['LangSmithTracer'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'LangSmithTracer':
        """Singleton pattern for tracer."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LangSmith client and configuration."""
        if self._initialized:
            return
        
        self.enabled = LANGSMITH_CONFIG["tracing_enabled"]
        self.project_name = LANGSMITH_CONFIG["project_name"]
        self.default_tags = LANGSMITH_CONFIG["default_tags"]
        self.default_metadata = LANGSMITH_CONFIG["default_metadata"]
        self.client: Optional[Client] = None
        
        if self.enabled and LANGSMITH_CONFIG["api_key"]:
            try:
                # Set environment variables for LangChain auto-tracing
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_CONFIG["api_key"]
                os.environ["LANGCHAIN_PROJECT"] = self.project_name
                os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_CONFIG["endpoint"]
                
                # Initialize client
                self.client = Client(
                    api_key=LANGSMITH_CONFIG["api_key"],
                    api_url=LANGSMITH_CONFIG["endpoint"]
                )
                
                logger.info(f"LangSmith tracing initialized for project: {self.project_name}")
                logger.info(f"   Dashboard: https://smith.langchain.com/o/default/projects")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False
        else:
            logger.info("LangSmith tracing is disabled (no API key or tracing disabled)")
        
        self._initialized = True
    
    def is_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled and configured."""
        return self.enabled and self.client is not None
    
    def get_enriched_metadata(self, custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get metadata enriched with default values and timestamp."""
        metadata = {
            **self.default_metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if custom_metadata:
            metadata.update(custom_metadata)
        return metadata
    
    def get_tags(self, custom_tags: Optional[List[str]] = None) -> List[str]:
        """Get tags combined with default tags."""
        tags = list(self.default_tags)
        if custom_tags:
            tags.extend(custom_tags)
        return tags


# Global tracer instance
_tracer: Optional[LangSmithTracer] = None


def get_tracer() -> LangSmithTracer:
    """Get the global LangSmith tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangSmithTracer()
    return _tracer


def initialize_langsmith_tracing():
    """
    Initialize LangSmith tracing for the application.
    Call this at application startup.
    """
    tracer = get_tracer()
    if tracer.is_enabled():
        logger.info("LangSmith observability is active")
        logger.info(f"   Project: {tracer.project_name}")
        logger.info(f"   View traces: https://smith.langchain.com/")
    return tracer


# =============================================================================
# TRACING DECORATORS
# =============================================================================

def trace_llm_call(
    name: str,
    run_type: str = "llm",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace LLM calls with LangSmith.
    
    Args:
        name: Name of the LLM operation (e.g., "query_understanding", "response_generation")
        run_type: Type of run (llm, chain, tool, etc.)
        tags: Additional tags for filtering
        metadata: Additional metadata to include
    
    Example:
        @trace_llm_call("query_understanding", tags=["entity-extraction"])
        def extract_entities(query: str) -> Dict:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            # Use LangSmith traceable decorator
            enriched_metadata = tracer.get_enriched_metadata(metadata)
            enriched_tags = tracer.get_tags(tags)
            
            traced_func = traceable(
                name=name,
                run_type=run_type,
                tags=enriched_tags,
                metadata=enriched_metadata
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


def trace_retrieval(
    name: str = "retrieval",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace RAG retrieval operations.
    
    Args:
        name: Name of the retrieval operation
        tags: Additional tags for filtering
        metadata: Additional metadata to include
    
    Example:
        @trace_retrieval("hybrid_search", tags=["semantic", "bm25"])
        def hybrid_search(query: str) -> List[Document]:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            enriched_metadata = tracer.get_enriched_metadata({
                "operation_type": "retrieval",
                **(metadata or {})
            })
            enriched_tags = tracer.get_tags(["retrieval"] + (tags or []))
            
            traced_func = traceable(
                name=name,
                run_type="retriever",
                tags=enriched_tags,
                metadata=enriched_metadata
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


def trace_chain(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace chain operations (multi-step workflows).
    
    Args:
        name: Name of the chain operation
        tags: Additional tags for filtering
        metadata: Additional metadata to include
    
    Example:
        @trace_chain("agent_workflow", tags=["multi-agent"])
        def process_query(query: str) -> Dict:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            enriched_metadata = tracer.get_enriched_metadata({
                "operation_type": "chain",
                **(metadata or {})
            })
            enriched_tags = tracer.get_tags(["chain"] + (tags or []))
            
            traced_func = traceable(
                name=name,
                run_type="chain",
                tags=enriched_tags,
                metadata=enriched_metadata
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


def trace_embedding(
    name: str = "embedding_generation",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace embedding generation operations.
    
    Args:
        name: Name of the embedding operation
        tags: Additional tags for filtering
        metadata: Additional metadata to include
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            enriched_metadata = tracer.get_enriched_metadata({
                "operation_type": "embedding",
                **(metadata or {})
            })
            enriched_tags = tracer.get_tags(["embedding"] + (tags or []))
            
            traced_func = traceable(
                name=name,
                run_type="embedding",
                tags=enriched_tags,
                metadata=enriched_metadata
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


def trace_data_ingestion(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace data ingestion operations.
    
    Args:
        name: Name of the ingestion operation
        tags: Additional tags for filtering
        metadata: Additional metadata to include
    
    Example:
        @trace_data_ingestion("load_restaurant_data", tags=["json"])
        def load_data(path: str) -> List[Dict]:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            enriched_metadata = tracer.get_enriched_metadata({
                "operation_type": "data_ingestion",
                **(metadata or {})
            })
            enriched_tags = tracer.get_tags(["data-ingestion"] + (tags or []))
            
            traced_func = traceable(
                name=name,
                run_type="tool",
                tags=enriched_tags,
                metadata=enriched_metadata
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


# =============================================================================
# CONTEXT MANAGERS FOR MANUAL TRACING
# =============================================================================

@contextmanager
def trace_run(
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for manual run tracing.
    
    Use when you need more control over the trace lifecycle.
    
    Example:
        with trace_run("custom_operation", inputs={"query": query}) as run:
            result = do_something(query)
            run.end(outputs={"result": result})
    
    Args:
        name: Name of the run
        run_type: Type of run (chain, llm, tool, retriever, embedding)
        inputs: Input data to log
        tags: Tags for filtering
        metadata: Additional metadata
    
    Yields:
        RunTree object for manual control (or None if tracing disabled)
    """
    tracer = get_tracer()
    
    if not tracer.is_enabled():
        yield None
        return
    
    enriched_metadata = tracer.get_enriched_metadata(metadata)
    enriched_tags = tracer.get_tags(tags)
    
    run_tree = RunTree(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        tags=enriched_tags,
        extra={"metadata": enriched_metadata}
    )
    
    try:
        run_tree.post()
        yield run_tree
    except Exception as e:
        run_tree.end(error=str(e))
        run_tree.patch()
        raise
    else:
        run_tree.patch()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_feedback(
    run_id: str,
    key: str,
    score: Optional[float] = None,
    value: Optional[str] = None,
    comment: Optional[str] = None
) -> bool:
    """
    Log feedback for a specific run in LangSmith.
    
    Useful for tracking user satisfaction, correctness, etc.
    
    Args:
        run_id: The ID of the run to provide feedback for
        key: Feedback key (e.g., "user_rating", "correctness")
        score: Numeric score (0-1 for normalized, or custom range)
        value: Categorical value
        comment: Additional comment
    
    Returns:
        True if feedback was logged successfully
    """
    tracer = get_tracer()
    
    if not tracer.is_enabled() or not tracer.client:
        return False
    
    try:
        tracer.client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            value=value,
            comment=comment
        )
        logger.debug(f"Logged feedback for run {run_id}: {key}={score or value}")
        return True
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")
        return False


def create_dataset(
    name: str,
    description: str = "",
    examples: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Create a dataset in LangSmith for evaluation.
    
    Args:
        name: Dataset name
        description: Dataset description
        examples: List of example dicts with 'inputs' and 'outputs' keys
    
    Returns:
        Dataset ID if created successfully, None otherwise
    """
    tracer = get_tracer()
    
    if not tracer.is_enabled() or not tracer.client:
        return None
    
    try:
        dataset = tracer.client.create_dataset(
            dataset_name=name,
            description=description
        )
        
        if examples:
            for example in examples:
                tracer.client.create_example(
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    dataset_id=dataset.id
                )
        
        logger.info(f"Created dataset '{name}' with {len(examples or [])} examples")
        return str(dataset.id)
    except Exception as e:
        logger.warning(f"Failed to create dataset: {e}")
        return None


def get_trace_url(run_id: str) -> str:
    """
    Get the LangSmith dashboard URL for a specific run.
    
    Args:
        run_id: The run ID
    
    Returns:
        URL to view the run in LangSmith dashboard
    """
    tracer = get_tracer()
    return f"https://smith.langchain.com/o/default/projects/p/{tracer.project_name}/r/{run_id}"


# =============================================================================
# CALLBACKS FOR LANGCHAIN INTEGRATION
# =============================================================================

def get_langsmith_callbacks() -> List[Any]:
    """
    Get LangSmith callbacks for LangChain operations.
    
    Returns:
        List of callback handlers (empty if tracing disabled)
    """
    tracer = get_tracer()
    
    if not tracer.is_enabled():
        return []
    
    # LangChain auto-traces when LANGCHAIN_TRACING_V2 is set
    # Return empty list as callbacks are handled by environment variables
    return []


if __name__ == "__main__":
    # Test the tracing setup
    import time
    
    # Initialize tracing
    tracer = initialize_langsmith_tracing()
    
    print(f"\nLangSmith Tracing Status:")
    print(f"  Enabled: {tracer.is_enabled()}")
    print(f"  Project: {tracer.project_name}")
    
    if tracer.is_enabled():
        # Example traced function
        @trace_llm_call("test_operation", tags=["test"])
        def test_function(input_text: str) -> str:
            time.sleep(0.1)  # Simulate work
            return f"Processed: {input_text}"
        
        result = test_function("Hello, LangSmith!")
        print(f"\n  Test result: {result}")
        print(f"  Trace sent to LangSmith!")
        print(f"  View traces: https://smith.langchain.com/")

