# REST API Documentation

## Overview

The Restaurant Search System provides a production-ready REST API built with **FastAPI**. The API offers intelligent restaurant search using RAG (Retrieval-Augmented Generation), natural language processing, and hybrid search capabilities.

## Quick Start

### Starting the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY='your-api-key-here'

# Start the server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload

# Or run directly
python src/api.py
```

The API will be available at: **http://localhost:8080**

### Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

## Base URL

```
http://localhost:8080
```

For production, replace with your deployed URL.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get API information and status.

**Response:**
```json
{
  "service": "Restaurant Search API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

---

### 2. Search Restaurants

**POST** `/search`

Search for restaurants using natural language queries.

**Request Body:**
```json
{
  "query": "Find Italian restaurants in Downtown Dubai with outdoor seating",
  "top_k": 5,
  "session_id": "user-123-session-456"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language search query (1-500 chars) |
| `top_k` | integer | No | Number of results (1-20, default: 5) |
| `session_id` | string | No | Session ID for conversation context |

**Response (200 OK):**
```json
{
  "success": true,
  "query": "Find Italian restaurants in Downtown Dubai with outdoor seating",
  "response": "I found 3 excellent Italian restaurants in Downtown Dubai that offer outdoor seating...",
  "restaurants": [
    {
      "id": 1,
      "name": "Bella Italia",
      "cuisine": "Italian",
      "location": "Downtown Dubai",
      "price_range": "AED 100-150",
      "rating": 4.5,
      "review_count": 120,
      "description": "Authentic Italian cuisine with traditional recipes...",
      "amenities": "Outdoor Seating, WiFi, Valet Parking",
      "attributes": "Romantic, Family Friendly",
      "opening_hours": "11:00 - 23:00",
      "relevance_score": 0.95
    }
  ],
  "total_found": 3,
  "extracted_entities": {
    "cuisine": "Italian",
    "location": "Downtown Dubai",
    "amenities": ["outdoor seating"]
  },
  "processing_time_ms": 1234.5
}
```

**Error Responses:**

```json
// 400 Bad Request
{
  "success": false,
  "error": "Query must be between 1 and 500 characters",
  "error_type": "ValidationError"
}

// 503 Service Unavailable
{
  "success": false,
  "error": "System is still initializing",
  "error_type": "ServiceUnavailable"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find Italian restaurants in Downtown Dubai",
    "top_k": 5
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8080/search",
    json={
        "query": "Find Italian restaurants with outdoor seating",
        "top_k": 3
    }
)

data = response.json()
print(f"Found {data['total_found']} restaurants")
for restaurant in data['restaurants']:
    print(f"- {restaurant['name']} ({restaurant['rating']}⭐)")
```

---

### 3. List Restaurants

**GET** `/restaurants`

Get a paginated list of all restaurants with optional filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | integer | 0 | Number of restaurants to skip |
| `limit` | integer | 50 | Number of restaurants to return (max 100) |
| `cuisine` | string | - | Filter by cuisine type |
| `location` | string | - | Filter by location |

**Response:**
```json
{
  "success": true,
  "total": 150,
  "restaurants": [
    {
      "id": 1,
      "name": "Bella Italia",
      "cuisine": "Italian",
      "location": "Downtown Dubai",
      "price_range": "AED 100-150",
      "rating": 4.5,
      "review_count": 120,
      "description": "...",
      "amenities": "...",
      "attributes": "...",
      "opening_hours": "11:00 - 23:00"
    }
  ]
}
```

**Examples:**
```bash
# Get first 10 restaurants
curl "http://localhost:8080/restaurants?limit=10"

# Get Italian restaurants
curl "http://localhost:8080/restaurants?cuisine=Italian"

# Get restaurants in Downtown Dubai
curl "http://localhost:8080/restaurants?location=Downtown%20Dubai"

# Pagination (skip 20, get next 10)
curl "http://localhost:8080/restaurants?skip=20&limit=10"
```

---

### 4. Get Restaurant by ID

**GET** `/restaurants/{restaurant_id}`

Get detailed information about a specific restaurant.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `restaurant_id` | integer | Restaurant ID (must be >= 1) |

**Response (200 OK):**
```json
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
  "opening_hours": "11:00 - 23:00"
}
```

**Error Response (404 Not Found):**
```json
{
  "success": false,
  "error": "Restaurant with ID 999 not found",
  "error_type": "NotFoundError"
}
```

**Example:**
```bash
curl "http://localhost:8080/restaurants/1"
```

---

### 5. Clear Session

**DELETE** `/sessions/{session_id}`

Clear conversation history for a specific session.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID to clear |

**Response:**
```json
{
  "success": true,
  "message": "Session user-123 cleared"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8080/sessions/user-123"
```

---

### 6. Health Check

**GET** `/health`

Check service health status. Used by load balancers and orchestrators.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "success_count": 150,
  "error_count": 5,
  "total_requests": 155,
  "error_rate": 0.032,
  "last_request_seconds_ago": 2.5,
  "healthy": true,
  "vector_store_initialized": true,
  "ensemble_retriever_ready": true
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "healthy": false,
  "error_rate": 0.52,
  ...
}
```

---

### 7. Prometheus Metrics

**GET** `/metrics`

Expose Prometheus-compatible metrics for monitoring.

**Response (Plain Text):**
```
# HELP restaurant_search_requests_total Total number of search requests
# TYPE restaurant_search_requests_total counter
restaurant_search_requests_total{agent="api_search",status="success"} 150.0

# HELP llm_calls_total Total number of LLM API calls
# TYPE llm_calls_total counter
llm_calls_total{agent="query_understanding",model="gpt-4",status="success"} 45.0
...
```

---

### 8. Statistics

**GET** `/stats`

Get system statistics and information.

**Response:**
```json
{
  "initialized": true,
  "total_restaurants": 150,
  "vector_store_documents": 150,
  "active_sessions": 3,
  "ensemble_retriever_ready": true,
  "health": {
    "success_count": 150,
    "error_count": 5,
    "total_requests": 155,
    "error_rate": 0.032,
    "healthy": true
  }
}
```

## Features

### 1. Session Management

Maintain conversation context across multiple requests using `session_id`:

```python
# First query in session
response1 = requests.post("/search", json={
    "query": "Find Italian restaurants",
    "session_id": "user-123"
})

# Follow-up query (uses context from previous query)
response2 = requests.post("/search", json={
    "query": "Which one has outdoor seating?",
    "session_id": "user-123"
})
```

### 2. Natural Language Understanding

The API understands complex natural language queries:

- "Find romantic Italian restaurants under AED 200"
- "I want vegetarian options near Dubai Marina"
- "Show me highly-rated places with WiFi for business lunch"
- "Budget-friendly Chinese food with delivery"

### 3. Entity Extraction

Automatically extracts structured entities:
- Cuisine type
- Location
- Price range
- Amenities (WiFi, parking, outdoor seating, etc.)
- Attributes (romantic, family-friendly, etc.)
- Rating requirements

### 4. Hybrid Search

Combines multiple search strategies:
- **Semantic search**: Understands meaning and context
- **BM25**: Exact keyword matching
- **Entity filtering**: Hard and soft constraints
- **Re-ranking**: Based on ratings, reviews, attributes

### 5. CORS Support

Configured for cross-origin requests, suitable for web applications.

### 6. Error Handling

Comprehensive error handling with detailed error responses:
- Validation errors
- Not found errors
- Service unavailable
- Internal server errors

## Production Deployment

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:
```bash
docker build -t restaurant-search-api .
docker run -p 8080:8080 -e OPENAI_API_KEY=your-key restaurant-search-api
```

### Using Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: restaurant-search-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: restaurant-search-api
  template:
    metadata:
      labels:
        app: restaurant-search-api
    spec:
      containers:
      - name: api
        image: restaurant-search-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: restaurant-search-api
spec:
  selector:
    app: restaurant-search-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Rate Limiting

For production, add rate limiting using `slowapi`:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/search")
@limiter.limit("10/minute")
async def search_restaurants(request: Request, search_req: SearchRequest):
    ...
```

## Authentication

For production, add API key authentication:

```python
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search_restaurants(request: SearchRequest):
    ...
```

## Monitoring

The API integrates with Prometheus for monitoring:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'restaurant_api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

Key metrics:
- Request rate and latency
- Error rates
- LLM call costs
- Active sessions
- System health

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_search():
    response = client.post("/search", json={
        "query": "Italian restaurants",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert len(data["restaurants"]) <= 5
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p search.json -T application/json \
   http://localhost:8080/search

# Using vegeta
echo "POST http://localhost:8080/search" | \
  vegeta attack -body search.json -rate=50 -duration=30s | \
  vegeta report
```

## Best Practices

1. **Always use HTTPS** in production
2. **Implement authentication** and authorization
3. **Add rate limiting** to prevent abuse
4. **Monitor** with Prometheus/Grafana
5. **Use health checks** in load balancers
6. **Set up logging** with correlation IDs
7. **Version your API** (e.g., `/v1/search`)
8. **Document breaking changes**
9. **Implement caching** for common queries
10. **Use async/await** for I/O operations

## Troubleshooting

### API not starting
```bash
# Check if port is already in use
lsof -i :8080

# Check logs
tail -f logs/restaurant_search.log
```

### Slow responses
- Check Prometheus metrics for bottlenecks
- Monitor LLM latency
- Verify vector store is loaded
- Check network latency to OpenAI API

### High error rates
- Check `/health` endpoint
- Review error logs
- Verify OpenAI API key is valid
- Check system resources (memory, CPU)

## Support

For issues and questions:
- Check `/docs` for interactive API documentation
- Review logs in `logs/restaurant_search.log`
- Monitor metrics at `/metrics`
- Check health at `/health`

---

**API Version**: 1.0.0  
**Framework**: FastAPI 0.104+  
**Python**: 3.12+

