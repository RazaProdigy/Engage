"""
REST API for Restaurant Search System using FastAPI.
Provides production-ready endpoints with validation, error handling, and observability.

Observability:
- Prometheus metrics: /metrics endpoint
- LangSmith tracing: Automatic tracing of all LLM calls and RAG operations
- Health checks: /health endpoint
- Latency metrics: /latency/recent and /latency/summary endpoints
"""
import logging
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from langsmith import traceable

from src.rag_system import RestaurantRAGSystem
from src.agents import RestaurantSearchAgentWorkflow
from src.config import OPENAI_API_KEY, LANGSMITH_CONFIG
from src.observability import (
    get_metrics,
    health_checker,
    record_request,
    track_active_requests,
    set_system_info,
    get_latest_latency_metrics,
    get_latency_summary,
    clear_latency_log
)
from src.langsmith_tracing import initialize_langsmith_tracing, get_tracer

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for restaurant search."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query",
        example="Find Italian restaurants in Downtown Dubai with outdoor seating"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation context"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Find Italian restaurants in Downtown Dubai with outdoor seating",
                "top_k": 5,
                "session_id": "user-123-session-456"
            }
        }


class Restaurant(BaseModel):
    """Restaurant information model."""
    id: int
    name: str
    cuisine: str
    location: str
    price_range: str
    rating: float
    review_count: int
    description: str
    amenities: str
    attributes: str
    opening_hours: str
    relevance_score: Optional[float] = None


class ExtractedEntities(BaseModel):
    """Extracted entities from query."""
    cuisine: Optional[str] = None
    location: Optional[str] = None
    price_range: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    rating_min: Optional[float] = None
    amenities: Optional[List[str]] = None
    attributes: Optional[List[str]] = None


class SearchResponse(BaseModel):
    """Response model for restaurant search."""
    success: bool = Field(..., description="Whether the search was successful")
    query: str = Field(..., description="Original search query")
    response: str = Field(..., description="Natural language response")
    restaurants: List[Restaurant] = Field(..., description="List of matching restaurants")
    total_found: int = Field(..., description="Total number of restaurants found")
    extracted_entities: Optional[ExtractedEntities] = Field(
        None, 
        description="Entities extracted from the query"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "Italian restaurants in Downtown",
                "response": "I found 3 Italian restaurants in Downtown Dubai...",
                "restaurants": [
                    {
                        "id": 1,
                        "name": "Bella Italia",
                        "cuisine": "Italian",
                        "location": "Downtown Dubai",
                        "price_range": "AED 100-150",
                        "rating": 4.5,
                        "review_count": 120,
                        "description": "Authentic Italian cuisine...",
                        "amenities": "Outdoor Seating, WiFi",
                        "attributes": "Romantic, Family Friendly",
                        "opening_hours": "11:00 - 23:00",
                        "relevance_score": 0.95
                    }
                ],
                "total_found": 3,
                "extracted_entities": {
                    "cuisine": "Italian",
                    "location": "Downtown Dubai"
                },
                "processing_time_ms": 1234.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    detail: Optional[str] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    success_count: int
    error_count: int
    total_requests: int
    error_rate: float
    last_request_seconds_ago: float
    healthy: bool
    vector_store_initialized: bool
    ensemble_retriever_ready: bool


class RestaurantListResponse(BaseModel):
    """Response for listing all restaurants."""
    success: bool = True
    total: int
    restaurants: List[Restaurant]


# ============================================================================
# APPLICATION STATE
# ============================================================================

class AppState:
    """Global application state."""
    rag_system: Optional[RestaurantRAGSystem] = None
    agent_workflow: Optional[RestaurantSearchAgentWorkflow] = None
    sessions: Dict[str, List[tuple]] = {}  # session_id -> chat_history
    initialized: bool = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    # Startup
    logger.info("Starting Restaurant Search API...")
    
    # Initialize LangSmith tracing
    langsmith_tracer = initialize_langsmith_tracing()
    if langsmith_tracer.is_enabled():
        logger.info("LangSmith observability enabled")
        logger.info(f"   Project: {langsmith_tracer.project_name}")
        logger.info(f"   Dashboard: https://smith.langchain.com/")
    else:
        logger.info("LangSmith tracing disabled (no API key configured)")
    
    # Initialize system info (Prometheus)
    set_system_info(
        version='1.0.0',
        environment='production',
        api_type='rest',
        langsmith_enabled=str(langsmith_tracer.is_enabled()),
        langsmith_project=langsmith_tracer.project_name if langsmith_tracer.is_enabled() else 'N/A'
    )
    
    # Initialize RAG system
    try:
        logger.info("Initializing RAG system...")
        app_state.rag_system = RestaurantRAGSystem(OPENAI_API_KEY)
        app_state.rag_system.initialize_pipeline(force_rebuild=False)
        
        # Initialize agent workflow
        logger.info("Initializing agent workflow...")
        app_state.agent_workflow = RestaurantSearchAgentWorkflow(
            OPENAI_API_KEY, 
            app_state.rag_system
        )
        
        app_state.initialized = True
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        app_state.initialized = False
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Restaurant Search API...")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Restaurant Search API",
    description="Production-ready REST API for intelligent restaurant search using RAG and LLM",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Restaurant Search API",
        "version": "1.0.0",
        "status": "running" if app_state.initialized else "initializing",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.post(
    "/search",
    response_model=SearchResponse,
    responses={
        200: {"description": "Successful search"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    tags=["Search"]
)
@traceable(name="api_search_restaurants", run_type="chain", tags=["api", "search"])
async def search_restaurants(request: SearchRequest):
    """
    Search for restaurants using natural language query.
    
    This endpoint uses RAG (Retrieval-Augmented Generation) with hybrid search
    (semantic + BM25) and entity extraction to find the best matching restaurants.
    
    LangSmith traces this entire API call including:
    - Query processing
    - Entity extraction
    - Hybrid search
    - Response generation
    
    **Example queries:**
    - "Find Italian restaurants in Downtown Dubai with outdoor seating"
    - "I want a romantic dinner with great views under AED 200"
    - "Show me budget-friendly vegetarian restaurants near Dubai Marina"
    """
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initializing"
        )
    
    start_time = time.time()
    success = False
    
    try:
        with track_active_requests('api_search'):
            # Get or create session
            session_id = request.session_id or "default"
            chat_history = app_state.sessions.get(session_id, [])
            
            # Process query
            result = app_state.agent_workflow.process_query(
                request.query,
                chat_history
            )
            
            # Update session history
            chat_history.append(("human", request.query))
            chat_history.append(("ai", result["response"]))
            app_state.sessions[session_id] = chat_history[-20:]  # Keep last 10 turns
            
            # Build response
            restaurants = []
            for r in result.get("restaurants", [])[:request.top_k]:
                restaurants.append(Restaurant(
                    id=r["id"],
                    name=r["name"],
                    cuisine=r["cuisine"],
                    location=r["location"],
                    price_range=r["price_range"],
                    rating=r["rating"],
                    review_count=r["review_count"],
                    description=r["description"],
                    amenities=r["amenities"],
                    attributes=r["attributes"],
                    opening_hours=r["opening_hours"],
                    relevance_score=r.get("relevance_score")
                ))
            
            # Extract entities if available
            extracted_entities = None
            if result.get("extracted_entities"):
                entities = result["extracted_entities"]
                extracted_entities = ExtractedEntities(
                    cuisine=entities.get("cuisine"),
                    location=entities.get("location"),
                    price_range=entities.get("price_range"),
                    price_min=entities.get("price_min"),
                    price_max=entities.get("price_max"),
                    rating_min=entities.get("rating_min"),
                    amenities=entities.get("amenities"),
                    attributes=entities.get("attributes")
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            success = True
            health_checker.record_success()
            
            return SearchResponse(
                success=True,
                query=request.query,
                response=result["response"],
                restaurants=restaurants,
                total_found=result.get("total_found", len(restaurants)),
                extracted_entities=extracted_entities,
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        logger.error(f"Error processing search: {e}", exc_info=True)
        health_checker.record_error()
        
        processing_time = (time.time() - start_time) * 1000
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                detail="An error occurred while processing your search"
            ).dict()
        )
    finally:
        duration = time.time() - start_time
        record_request('api_search', duration, 'success' if success else 'error')


@app.get(
    "/restaurants",
    response_model=RestaurantListResponse,
    tags=["Restaurants"]
)
async def list_restaurants(
    skip: int = Query(0, ge=0, description="Number of restaurants to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of restaurants to return"),
    cuisine: Optional[str] = Query(None, description="Filter by cuisine"),
    location: Optional[str] = Query(None, description="Filter by location")
):
    """
    List all restaurants with optional filtering.
    
    Supports pagination and filtering by cuisine or location.
    """
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initializing"
        )
    
    try:
        # Get all restaurants
        all_restaurants = app_state.rag_system.restaurants_data
        
        # Apply filters
        filtered = all_restaurants
        if cuisine:
            filtered = [r for r in filtered if r["cuisine"].lower() == cuisine.lower()]
        if location:
            filtered = [r for r in filtered if location.lower() in r["location"].lower()]
        
        # Apply pagination
        paginated = filtered[skip:skip + limit]
        
        # Convert to response model
        restaurants = [
            Restaurant(
                id=r["id"],
                name=r["name"],
                cuisine=r["cuisine"],
                location=r["location"],
                price_range=r["price_range"],
                rating=r["rating"],
                review_count=r["review_count"],
                description=r["description"],
                amenities=r["amenities"],
                attributes=r["attributes"],
                opening_hours=r["opening_hours"]
            )
            for r in paginated
        ]
        
        return RestaurantListResponse(
            success=True,
            total=len(filtered),
            restaurants=restaurants
        )
        
    except Exception as e:
        logger.error(f"Error listing restaurants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/restaurants/{restaurant_id}",
    response_model=Restaurant,
    tags=["Restaurants"]
)
async def get_restaurant(
    restaurant_id: int = Path(..., description="Restaurant ID", ge=1)
):
    """Get detailed information about a specific restaurant."""
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initializing"
        )
    
    try:
        restaurant = app_state.rag_system.get_restaurant_by_id(restaurant_id)
        
        if not restaurant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Restaurant with ID {restaurant_id} not found"
            )
        
        return Restaurant(
            id=restaurant["id"],
            name=restaurant["name"],
            cuisine=restaurant["cuisine"],
            location=restaurant["location"],
            price_range=restaurant["price_range"],
            rating=restaurant["rating"],
            review_count=restaurant["review_count"],
            description=restaurant["description"],
            amenities=restaurant["amenities"],
            attributes=restaurant["attributes"],
            opening_hours=restaurant["opening_hours"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting restaurant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete(
    "/sessions/{session_id}",
    tags=["Sessions"]
)
async def clear_session(
    session_id: str = Path(..., description="Session ID to clear")
):
    """Clear conversation history for a session."""
    if session_id in app_state.sessions:
        del app_state.sessions[session_id]
        return {"success": True, "message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"]
)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
    - 200: Service is healthy
    - 503: Service is unhealthy
    """
    stats = health_checker.get_stats()
    
    health_response = HealthResponse(
        status="healthy" if stats["healthy"] else "unhealthy",
        success_count=stats["success_count"],
        error_count=stats["error_count"],
        total_requests=stats["total_requests"],
        error_rate=stats["error_rate"],
        last_request_seconds_ago=stats["last_request_seconds_ago"],
        healthy=stats["healthy"],
        vector_store_initialized=app_state.rag_system is not None and 
                                 app_state.rag_system.vectorstore is not None,
        ensemble_retriever_ready=app_state.rag_system is not None and 
                                app_state.rag_system.ensemble_retriever is not None
    )
    
    status_code = status.HTTP_200_OK if stats["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content=health_response.dict()
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["Monitoring"]
)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes metrics in Prometheus format for scraping.
    """
    return get_metrics().decode('utf-8')


@app.get(
    "/stats",
    tags=["Monitoring"]
)
async def get_stats():
    """Get system statistics including LangSmith observability status."""
    tracer = get_tracer()
    
    if not app_state.initialized:
        return {
            "initialized": False,
            "message": "System is still initializing",
            "langsmith": {
                "enabled": tracer.is_enabled(),
                "project": tracer.project_name if tracer.is_enabled() else None
            }
        }
    
    return {
        "initialized": True,
        "total_restaurants": len(app_state.rag_system.restaurants_data),
        "vector_store_documents": len(app_state.rag_system.documents),
        "active_sessions": len(app_state.sessions),
        "ensemble_retriever_ready": app_state.rag_system.ensemble_retriever is not None,
        "health": health_checker.get_stats(),
        "observability": {
            "prometheus_metrics": "/metrics",
            "langsmith": {
                "enabled": tracer.is_enabled(),
                "project": tracer.project_name if tracer.is_enabled() else None,
                "dashboard": "https://smith.langchain.com/" if tracer.is_enabled() else None
            }
        }
    }


@app.get(
    "/langsmith/status",
    tags=["Monitoring"]
)
async def get_langsmith_status():
    """
    Get LangSmith observability status and configuration.
    
    Returns information about:
    - Whether LangSmith tracing is enabled
    - Current project name
    - Dashboard URL
    - Default tags and metadata
    """
    tracer = get_tracer()
    
    return {
        "enabled": tracer.is_enabled(),
        "project": tracer.project_name if tracer.is_enabled() else None,
        "dashboard_url": "https://smith.langchain.com/" if tracer.is_enabled() else None,
        "default_tags": tracer.default_tags if tracer.is_enabled() else [],
        "default_metadata": tracer.default_metadata if tracer.is_enabled() else {},
        "traces_include": [
            "LLM calls (query understanding, response generation)",
            "RAG operations (hybrid search, semantic search)",
            "Data ingestion (document loading, vector store building)",
            "Agent workflows (multi-agent processing)"
        ] if tracer.is_enabled() else []
    }


@app.get(
    "/latency/recent",
    response_class=PlainTextResponse,
    tags=["Monitoring"]
)
async def get_recent_latency_metrics(
    num_lines: int = Query(50, ge=1, le=500, description="Number of recent log entries to return")
):
    """
    Get recent latency metrics from the log file.
    
    Shows the most recent latency measurements for retrieval and LLM calls,
    making it easy to quickly check performance without querying Prometheus.
    
    **Example response:**
    ```
    2024-01-15 10:30:45 | llm_call | query_understanding/gpt-4 | 1234.56ms | status=success | prompt_tokens=150
    2024-01-15 10:30:46 | retrieval | hybrid | 234.12ms | status=success | num_documents=10
    ```
    """
    return get_latest_latency_metrics(num_lines)


@app.get(
    "/latency/summary",
    tags=["Monitoring"]
)
async def get_latency_statistics():
    """
    Get latency summary statistics.
    
    Returns aggregated statistics (average, min, max) for all metric types,
    providing a quick overview of system performance.
    
    **Metrics included:**
    - `llm_call`: LLM API call latencies
    - `retrieval`: Document retrieval latencies
    - `entity_extraction`: Entity extraction latencies
    
    **Example response:**
    ```json
    {
      "llm_call": {
        "count": 100,
        "avg_ms": 1250.5,
        "min_ms": 450.2,
        "max_ms": 3200.8,
        "latest_ms": 1180.3
      },
      "retrieval": {
        "count": 100,
        "avg_ms": 125.3,
        "min_ms": 45.1,
        "max_ms": 450.2,
        "latest_ms": 110.5
      }
    }
    ```
    """
    return get_latency_summary()


@app.delete(
    "/latency/clear",
    tags=["Monitoring"]
)
async def clear_latency_metrics_log():
    """
    Clear the latency metrics log file.
    
    Useful for starting fresh or when the log file becomes too large.
    This only clears the file-based log; Prometheus metrics are unaffected.
    """
    try:
        clear_latency_log()
        return {
            "success": True,
            "message": "Latency metrics log cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing latency log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear latency log: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            success=False,
            error="Not found",
            error_type="NotFoundError",
            detail=str(exc.detail) if hasattr(exc, 'detail') else "The requested resource was not found"
        ).dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            error_type="InternalServerError",
            detail="An unexpected error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

