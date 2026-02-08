# Production Observability Guide

## Overview

The Restaurant Search System includes production-grade observability using **Prometheus metrics**. This enables real-time monitoring of performance, errors, and system health.

## Quick Start

### Starting the Application with Metrics

```bash
# Start with metrics server on default port (8000)
python -m src.main --mode interactive

# Start with custom metrics port
python -m src.main --mode interactive --metrics-port 9090

# Start without metrics server
python -m src.main --mode interactive --no-metrics
```

### Accessing Metrics

Once the application is running:

- **Metrics Endpoint**: http://localhost:8000/metrics
- **Health Check**: http://localhost:8000/health

## Metrics Collected

### 1. Request Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `restaurant_search_requests_total` | Counter | Total number of search requests | `agent`, `status` |
| `restaurant_search_request_duration_seconds` | Histogram | Request processing duration | `agent` |
| `active_requests` | Gauge | Currently processing requests | `agent` |

**Example:**
```promql
# Request rate by agent
rate(restaurant_search_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(restaurant_search_request_duration_seconds_bucket[5m]))

# Error rate
rate(restaurant_search_requests_total{status="error"}[5m]) / rate(restaurant_search_requests_total[5m])
```

### 2. LLM Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `llm_calls_total` | Counter | Total LLM API calls | `agent`, `model`, `status` |
| `llm_call_duration_seconds` | Histogram | LLM API call duration | `agent`, `model` |
| `llm_tokens_total` | Counter | Total tokens used | `agent`, `token_type` |
| `llm_cost_usd_total` | Counter | Estimated API cost in USD | `agent`, `model` |

**Example:**
```promql
# LLM call rate
rate(llm_calls_total[5m])

# Token usage per minute
rate(llm_tokens_total{token_type="total"}[1m]) * 60

# Estimated hourly cost
rate(llm_cost_usd_total[1h]) * 3600
```

### 3. Retrieval Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `retrieval_operations_total` | Counter | Total retrieval operations | `retriever_type`, `status` |
| `retrieval_duration_seconds` | Histogram | Retrieval duration | `retriever_type` |
| `documents_retrieved_count` | Histogram | Documents per query | `retriever_type` |

**Example:**
```promql
# Retrieval latency P99
histogram_quantile(0.99, rate(retrieval_duration_seconds_bucket{retriever_type="hybrid"}[5m]))

# Average documents retrieved
rate(documents_retrieved_count_sum[5m]) / rate(documents_retrieved_count_count[5m])
```

### 4. Entity Extraction Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `entity_extraction_total` | Counter | Entity extraction attempts | `status` |
| `entity_extraction_duration_seconds` | Histogram | Extraction duration | - |
| `entities_per_query` | Histogram | Entities extracted per query | - |

**Example:**
```promql
# Extraction success rate
rate(entity_extraction_total{status="success"}[5m]) / rate(entity_extraction_total[5m])

# Fallback rate (when extraction fails)
rate(entity_extraction_total{status="fallback"}[5m]) / rate(entity_extraction_total[5m])
```

### 5. Filter & Ranking Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `filter_duration_seconds` | Histogram | Filtering duration | - |
| `documents_filtered_total` | Counter | Documents filtered out | `filter_type` |
| `rerank_duration_seconds` | Histogram | Re-ranking duration | - |

**Example:**
```promql
# Documents filtered by cuisine
rate(documents_filtered_total{filter_type="cuisine"}[5m])

# Average filter latency
rate(filter_duration_seconds_sum[5m]) / rate(filter_duration_seconds_count[5m])
```

### 6. Error Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `errors_total` | Counter | Total errors | `component`, `error_type` |

**Example:**
```promql
# Error rate by component
rate(errors_total[5m])

# Errors per hour
increase(errors_total[1h])
```

### 7. System Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `vector_store_documents_total` | Gauge | Documents in vector store | - |
| `cache_operations_total` | Counter | Cache operations | `operation`, `result` |
| `restaurant_search_system_info` | Info | System information | `version`, `environment` |

## Integration with Prometheus

### 1. Prometheus Configuration

Create a `prometheus.yml` file:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'restaurant_search'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 2. Run Prometheus

```bash
# Using Docker
docker run -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# Or download binary
./prometheus --config.file=prometheus.yml
```

Access Prometheus UI at: http://localhost:9090

## Grafana Dashboards

### Sample Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Restaurant Search System",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(restaurant_search_requests_total[5m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(restaurant_search_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(restaurant_search_requests_total{status=\"error\"}[5m])"
        }]
      },
      {
        "title": "LLM Cost (hourly)",
        "targets": [{
          "expr": "rate(llm_cost_usd_total[1h]) * 3600"
        }]
      }
    ]
  }
}
```

## Alerting Rules

### Critical Alerts

```yaml
groups:
  - name: restaurant_search_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(restaurant_search_requests_total{status="error"}[5m]) / rate(restaurant_search_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(restaurant_search_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency"
          description: "P95 latency is {{ $value }}s"
      
      # LLM failures
      - alert: LLMFailures
        expr: rate(llm_calls_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM API failures"
          description: "LLM error rate: {{ $value }}/s"
      
      # High cost
      - alert: HighLLMCost
        expr: rate(llm_cost_usd_total[1h]) * 3600 > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High LLM costs"
          description: "Hourly cost: ${{ $value }}"
```

## Health Checks

### Health Endpoint Response

```json
{
  "success_count": 150,
  "error_count": 5,
  "total_requests": 155,
  "error_rate": 0.032,
  "last_request_seconds_ago": 2.5,
  "healthy": true
}
```

### Using Health Checks

```bash
# Check health
curl http://localhost:8000/health

# With jq
curl -s http://localhost:8000/health | jq '.healthy'

# In Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Production Best Practices

### 1. Resource Monitoring

```promql
# Monitor active requests
active_requests

# Vector store size
vector_store_documents_total

# Token usage trends
rate(llm_tokens_total[1h])
```

### 2. Performance Optimization

Monitor these metrics to identify bottlenecks:

- **Slow retrievals**: `retrieval_duration_seconds > 1s`
- **Slow LLM calls**: `llm_call_duration_seconds > 5s`
- **High filter time**: `filter_duration_seconds > 0.5s`

### 3. Cost Optimization

```promql
# Daily cost projection
rate(llm_cost_usd_total[1h]) * 3600 * 24

# Token efficiency (requests per 1K tokens)
rate(restaurant_search_requests_total[1h]) / (rate(llm_tokens_total{token_type="total"}[1h]) / 1000)
```

### 4. SLO/SLI Tracking

**Service Level Indicators (SLIs):**
- Availability: % of successful requests
- Latency: P95 request duration < 5s
- Error Rate: < 1%

**Example SLO Queries:**
```promql
# Availability (target: 99.9%)
sum(rate(restaurant_search_requests_total{status="success"}[30d])) / 
sum(rate(restaurant_search_requests_total[30d]))

# Latency SLO (target: 95% under 5s)
histogram_quantile(0.95, rate(restaurant_search_request_duration_seconds_bucket[30d])) < 5

# Error budget remaining
1 - (sum(rate(restaurant_search_requests_total{status="error"}[30d])) / sum(rate(restaurant_search_requests_total[30d]))) / 0.001
```

## Troubleshooting

### Common Issues

**1. Metrics not appearing**
```bash
# Check if metrics server is running
curl http://localhost:8000/metrics

# Check firewall
sudo netstat -tulpn | grep 8000
```

**2. High memory usage**
- Vector store is loaded in memory
- Monitor with: `vector_store_documents_total`
- Consider pagination for large datasets

**3. Prometheus cannot scrape**
- Verify network connectivity
- Check `prometheus.yml` configuration
- Ensure application is running

### Debug Mode

Enable debug logging for metrics:
```python
import logging
logging.getLogger('src.observability').setLevel(logging.DEBUG)
```

## Example: Complete Monitoring Stack

```bash
# 1. Start application with metrics
python -m src.main --mode interactive --metrics-port 8000

# 2. Start Prometheus
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# 3. Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# 4. Access dashboards
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
# Metrics:    http://localhost:8000/metrics
```

## API Reference

See `src/observability.py` for detailed API documentation.

### Key Functions

- `record_llm_call()` - Track LLM API calls
- `record_retrieval()` - Track retrieval operations
- `record_entity_extraction()` - Track entity extraction
- `record_error()` - Track errors
- `track_active_requests()` - Context manager for active requests
- `get_metrics()` - Get Prometheus-formatted metrics

---

## LangSmith Integration (Optional)

For LLM-specific tracing and debugging, you can enable LangSmith:

### Setup

```bash
# Add to .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=restaurant-recommender
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### What Gets Traced

| Operation | Run Type | Description |
|-----------|----------|-------------|
| Query Understanding | `llm` | Entity extraction from user queries |
| Response Generation | `llm` | Personalized recommendation generation |
| Hybrid Search | `retriever` | Semantic + BM25 retrieval operations |
| Data Ingestion | `chain` | Document loading and vector store building |

### Benefits

- **Detailed Traces**: Per-request LLM call analysis
- **Token Usage**: Automatic tracking of prompt/completion tokens
- **Cost Tracking**: Built-in LLM cost estimation
- **Debugging**: Trace through multi-agent workflows

### Dashboard

Access traces at: https://smith.langchain.com/

---

**For production deployments**, integrate with your existing monitoring infrastructure (Datadog, New Relic, etc.) using Prometheus exporters.

