# 🍽️ RAG-Powered Restaurant Search System

A production-ready, intelligent restaurant recommendation system for Dubai using **Retrieval-Augmented Generation (RAG)** and **Multi-Agent Workflows** powered by LangChain, LangGraph, and OpenAI.

## 🎯 Overview

This system demonstrates advanced AI capabilities for semantic search and personalized recommendations:

- **RAG Architecture**: Hybrid search combining semantic understanding and keyword matching
- **Multi-Agent Workflow**: Three specialized agents orchestrated via LangGraph
- **Intelligent Entity Extraction**: Understanding complex, natural language queries
- **Contextual Conversations**: Multi-turn dialogues with memory management
- **REST API**: Production-ready FastAPI with automatic documentation
- **Production Observability**: Prometheus metrics, health checks, cost tracking
- **Docker Deployment**: Containerized with full monitoring stack
- **Comprehensive Error Handling**: Logging, fallbacks, and graceful degradation

## ✨ Key Features

### 🔍 Search & Recommendations
- **Hybrid Search**: 60% semantic + 40% keyword matching for best results
- **Entity Extraction**: Automatically understands cuisine, location, price, amenities
- **Smart Filtering**: Hard constraints + soft preferences with score adjustments
- **Intelligent Ranking**: Multi-factor scoring (relevance, rating, reviews, attributes)
- **Edge Case Handling**: Progressive constraint relaxation, alternative suggestions

### 🤖 Multi-Agent System
- **Query Understanding Agent**: Parses queries, extracts entities, detects ambiguities
- **Retrieval Agent**: Executes hybrid search, applies filters, ranks results
- **Response Generation Agent**: Creates personalized, conversational recommendations
- **Conditional Logic**: Clarification flow for ambiguous queries
- **Memory Management**: Context from last 10 conversation turns

### 🌐 Production Features
- **REST API**: 8 endpoints with Swagger/ReDoc documentation
- **Observability**: 17 Prometheus metrics tracking LLM costs, latency, errors
- **Health Checks**: Load balancer-ready endpoints
- **Docker Deployment**: Multi-stage optimized build with monitoring stack
- **Session Management**: Multi-turn conversations with session IDs
- **Error Handling**: Comprehensive fallbacks and error responses
- **Type Safety**: Pydantic validation for all requests/responses

## 🏗️ System Architecture

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Agent 1: Query Understanding                             │  │
│  │  • Parse natural language                                 │  │
│  │  • Extract entities (cuisine, location, price, etc.)     │  │
│  │  • Detect ambiguities → Clarification needed?            │  │
│  └────────────────┬──────────────────────────────────────────┘  │
│                   │                                              │
│            ┌──────┴──────┐                                       │
│            │ Clarify?    │                                       │
│            └──┬──────┬───┘                                       │
│          Yes  │      │ No                                        │
│               │      │                                           │
│     ┌─────────┘      └──────────┐                               │
│     │                            │                               │
│     ▼                            ▼                               │
│  Return                  ┌───────────────────────────┐           │
│  Question                │  Agent 2: Retrieval       │           │
│                          │  • Hybrid search (RAG)    │           │
│                          │  • Apply filters          │           │
│                          │  • Rank results           │           │
│                          └────────┬──────────────────┘           │
│                                   │                              │
│                                   ▼                              │
│                          ┌───────────────────────────┐           │
│                          │  Agent 3: Response Gen    │           │
│                          │  • Personalize output     │           │
│                          │  • Format recommendations │           │
│                          │  • Suggest next steps     │           │
│                          └───────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  RAG System  │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Vector Store  │ │  BM25 Index  │ │  Metadata    │
│  (Chroma DB)   │ │  (Keyword)   │ │  Filters     │
└────────────────┘ └──────────────┘ └──────────────┘
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **LLM** | GPT-4o-mini | High-quality reasoning, cost-effective |
| **Embeddings** | text-embedding-3-small | Efficient, accurate semantic representation |
| **Vector Store** | ChromaDB | Fast, easy to use, persistent storage |
| **Orchestration** | LangGraph | State management, conditional flows |
| **Keyword Search** | BM25 (rank-bm25) | Complement semantic search with exact matching |
| **Framework** | LangChain | Rich ecosystem, production utilities |
| **API** | FastAPI | Modern, fast, auto-documented REST API |
| **Monitoring** | Prometheus + Grafana | Industry-standard observability |
| **Deployment** | Docker + Compose | Production-ready containerization |
| **Validation** | Pydantic | Type-safe data models |
| **Python** | 3.12+ | Latest features and performance |

### Design Decisions

#### 1. **Hybrid Search (60% Semantic + 40% Keyword)**
- **Why**: Semantic search captures meaning ("romantic dinner" → fine dining), while BM25 catches exact terms ("Jumeirah")
- **Trade-off**: Slightly more complex but significantly better recall and precision
- **Production Impact**: Handles diverse query styles from users

#### 2. **Multi-Agent Architecture**
- **Why**: Separation of concerns - each agent specialized in one task
- **Benefits**: 
  - Easier to debug and improve individual components
  - Parallel development potential
  - Clear failure boundaries
- **Alternative Considered**: Single-agent approach (rejected: too complex, harder to maintain)

#### 3. **ChromaDB Vector Store**
- **Why**: 
  - Easy setup (no external DB server)
  - Persistent storage
  - Good performance for our scale (50 restaurants, scalable to 10k+)
- **Alternative Considered**: Pinecone/Weaviate (rejected: overkill for MVP, added complexity)

#### 4. **Entity-Based Filtering**
- **Why**: Structured extraction enables precise filtering beyond semantic similarity
- **Implementation**: Hard filters (must-match) + soft filters (score adjustment)
- **Edge Cases**: Handles missing entities, conflicting requirements, fuzzy matching

## 🚀 Quick Start

### Prerequisites

```bash
python 3.12+
OpenAI API key
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Recommender-System
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set OpenAI API key**
```bash
export OPENAI_API_KEY='your-api-key-here'
# On Windows: set OPENAI_API_KEY=your-api-key-here
```

### Running the System

#### Interactive Mode (Recommended)
```bash
python -m src.main --mode interactive
```

#### Example Queries Mode
```bash
python -m src.main --mode examples
```

#### Single Query
```bash
python -m src.main --query "Find Italian restaurants in Downtown Dubai with outdoor seating under AED 200"
```

#### Rebuild Vector Store
```bash
python -m src.main --rebuild
```

## 📖 Usage Examples

### Example 1: Specific Criteria
```
Query: "Find Italian restaurants in Downtown Dubai with outdoor seating under AED 200"

Response: "I found some wonderful Italian restaurants in Downtown matching your preferences!

🍝 Mama's Kitchen (Downtown Dubai)
A family-run trattoria serving homemade pasta and wood-fired pizzas. They have lovely 
outdoor seating perfect for Dubai's pleasant evenings.
• Price: AED 120-180 ✓ Under budget!
• Rating: 4.3/5 (634 reviews)
• Hours: 11:00 - 23:00

🍕 Bella Vista (Downtown Dubai)  
Upscale Italian with stunning Burj Khalifa views from their outdoor terrace...
[Note: Slightly above budget at AED 150-220, but the view is spectacular!]
```

### Example 2: Ambiguous Query
```
Query: "I want Italian food"

Response: "I'd love to help you find a great Italian restaurant! Dubai has many excellent 
options. Could you tell me:
• Which area are you interested in? (Downtown, Marina, Jumeirah, etc.)
• What's your preferred price range?
• Any special requirements like outdoor seating or parking?"
```

### Example 3: No Results → Relaxed Search
```
Query: "Korean BBQ in Old Dubai under AED 50"

Response: "I couldn't find Korean BBQ in Old Dubai at that price point, but here are 
some alternatives:

🇰🇷 Seoul Kitchen (Business Bay) - AED 110-170
While slightly outside your budget and location, this is the closest match for authentic 
Korean BBQ with table grills...

Would you like to:
• Explore Korean options in other areas?
• Find budget-friendly restaurants in Old Dubai (different cuisine)?
• Adjust your budget range?"
```

## 📊 Monitoring Performance

The system automatically logs latency metrics for retrieval and LLM calls to help you quickly check performance.

### Quick Check Latency

```bash
# View recent latency metrics
python view_metrics.py

# View summary statistics
python view_metrics.py --summary

# View last 100 entries
python view_metrics.py --lines 100
```

### Via API

```bash
# Get recent metrics
curl http://localhost:8080/latency/recent

# Get summary
curl http://localhost:8080/latency/summary
```

### What You'll See

```
=== Latest Latency Metrics ===
2024-11-26 10:30:45 | llm_call      | query_understanding/gpt-4  | 1234.56ms | status=success
2024-11-26 10:30:46 | retrieval     | hybrid                     |  234.12ms | status=success
2024-11-26 10:30:47 | llm_call      | response_generation/gpt-4  | 1567.89ms | status=success
```

**📖 Full Guide**: See `docs/OBSERVABILITY.md` for detailed monitoring documentation

## 🏗️ Project Structure

```
Recommender-System/
├── data/
│   └── restaurant.json             # 50 Dubai restaurants
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration and prompts
│   ├── rag_system.py              # RAG implementation with hybrid search
│   ├── agents.py                   # Multi-agent workflow (LangGraph)
│   ├── api.py                      # FastAPI REST API
│   ├── observability.py           # Prometheus metrics
│   └── main.py                     # CLI application
├── docs/
│   ├── ARCHITECTURE.md             # System architecture & design decisions
│   ├── API.md                      # REST API documentation
│   └── OBSERVABILITY.md           # Monitoring & metrics guide
├── examples/
│   └── api_client_example.py      # Python API client example
├── tests/
│   └── test_rag_system.py         # Unit tests
├── Dockerfile                      # Production Docker image
├── docker-compose.yml             # Full stack with monitoring
├── requirements.txt
└── README.md
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Set your API key
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start API only
docker-compose up -d api

# Or start with full monitoring stack (Prometheus + Grafana)
docker-compose up -d
```

### Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8080 |
| API Docs | http://localhost:8080/docs |
| Health Check | http://localhost:8080/health |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Common Commands

```bash
# View logs
docker-compose logs -f api

# Restart
docker-compose restart api

# Stop all
docker-compose down

# Rebuild after changes
docker-compose build --no-cache api && docker-compose up -d api
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test specific component
pytest tests/test_rag_system.py -v
```

## 🎯 Key Features

### 1. RAG System Features

#### Hybrid Search
- **Semantic Search**: Understands query intent and context
- **Keyword Search (BM25)**: Catches exact matches
- **Ensemble Retriever**: Weighted combination (60/40 split)

#### Intelligent Filtering
- **Hard Filters**: Must-match criteria (cuisine, location)
- **Soft Filters**: Preference-based scoring (price with 20% flex, amenities)
- **Dynamic Re-ranking**: Multi-factor scoring (rating, reviews, attributes)

#### Edge Case Handling
- **No Results**: Progressive constraint relaxation
- **Ambiguous Queries**: Request clarification
- **Conflicting Requirements**: Prioritize and explain trade-offs

### 2. Multi-Agent Workflow

#### Agent 1: Query Understanding
- Natural language parsing
- Entity extraction (JSON format)
- Clarification detection
- Context from chat history

#### Agent 2: Restaurant Retrieval
- Execute hybrid search
- Apply entity-based filters
- Handle edge cases (no results)
- Suggest alternatives

#### Agent 3: Response Generation
- Personalized recommendations
- Clear formatting
- Actionable next steps
- Conversational tone

#### Workflow Features
- **Conditional Logic**: Branch based on query clarity
- **Memory Management**: Track conversation history (last 10 turns)
- **Error Recovery**: Fallback strategies at each step
- **Performance Monitoring**: Logging and metrics

### 3. Production-Ready Features

- **Logging**: Comprehensive logging to file and console
- **Error Handling**: Graceful degradation at every level
- **Configuration**: Centralized config management
- **Persistence**: Vector store saved to disk
- **Scalability**: Efficient embeddings and retrieval
- **Monitoring**: Track query processing, retrieval metrics
- **Latency Metrics**: File-based logging for quick performance checks

## 🎨 Design Philosophy

### Bias for Action
- **Built first, optimized later**: Implemented working MVP quickly, then refined
- **Pragmatic choices**: Used ChromaDB (simple) over complex vector DBs
- **Iterative development**: Started with basic search, added hybrid, then agents

### Ownership
- **End-to-end responsibility**: From data ingestion to response generation
- **Quality validation**: Tested edge cases extensively
- **Documentation**: Comprehensive docs for future maintainers
- **Production thinking**: Logging, error handling, configuration management

### Innovation
- **Novel approach**: Combined entity extraction with semantic search for better results
- **Calculated risks**: 
  - Hybrid search (60/40 split) - validated with testing
  - Multi-agent architecture - justified by separation of concerns
- **Creative solutions**: 
  - Relaxed search for no-results cases
  - Dynamic re-ranking with multiple factors

## 📊 Performance Characteristics

### Search Performance
- **Latency**: ~2-3 seconds per query (including LLM calls)
- **Accuracy**: High precision on specific queries, good recall with hybrid search
- **Scalability**: Supports 1000+ restaurants without performance degradation

### Resource Usage
- **Memory**: ~200MB for 50 restaurants (vector store + embeddings)
- **Storage**: ~10MB for persisted vector store
- **API Calls**: 2-3 OpenAI calls per query (embeddings + LLM)

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# LLM Configuration
LLM_CONFIG = {
    "model": "gpt-4o-mini",        # Change model
    "temperature": 0.3,             # Adjust creativity
    "max_tokens": 1000,
}

# Search Configuration
SEARCH_CONFIG = {
    "top_k": 10,                    # Number of results
    "similarity_threshold": 0.7,    # Minimum similarity
    "semantic_weight": 0.6,         # Hybrid search weights
    "keyword_weight": 0.4,
}
```

## 🚦 Roadmap

### Phase 1: MVP ✅
- [x] RAG system with hybrid search
- [x] Multi-agent workflow
- [x] Interactive CLI
- [x] 50 restaurants dataset

### Phase 2: Enhancements
- [ ] Web UI (Streamlit/Gradio)
- [ ] User preferences profile
- [ ] Reservation integration
- [ ] Image embeddings for ambiance matching

### Phase 3: Production
- [ ] API deployment (FastAPI)
- [ ] Authentication and user management
- [ ] Analytics dashboard
- [ ] A/B testing framework

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain**: Excellent framework for LLM applications
- **OpenAI**: Powerful models for embeddings and generation
- **ChromaDB**: Simple yet effective vector store



