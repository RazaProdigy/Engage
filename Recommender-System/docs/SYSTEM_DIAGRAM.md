# System Architecture Diagram

This document provides a visual representation of the RAG-based Restaurant Search System architecture. Since we can't embed images directly, this provides ASCII diagrams and descriptions suitable for visualization tools.

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              USER                                    │
│                      (CLI / Future: Web API)                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ Natural Language Query
                                 │ "Find Italian restaurants in Downtown
                                 │  with outdoor seating under AED 200"
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MAIN APPLICATION                                │
│                    (src/main.py)                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  RestaurantSearchApp                                          │  │
│  │  • Session management                                         │  │
│  │  • Conversation history (last 10 turns)                       │  │
│  │  • Request routing                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH AGENT WORKFLOW                            │
│                      (src/agents.py)                                 │
│                                                                      │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                    AGENT STATE                                 ║  │
│  ║  • query: str                                                  ║  │
│  ║  • chat_history: List[tuple]                                   ║  │
│  ║  • extracted_entities: Dict                                    ║  │
│  ║  • retrieved_restaurants: List[Dict]                           ║  │
│  ║  • final_response: str                                         ║  │
│  ║  • needs_clarification: bool                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  START                                                       │    │
│  └───────────────────────┬──────────────────────────────────────┘    │
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  AGENT 1: Query Understanding                               │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  • Parse natural language query                     │    │    │
│  │  │  • Extract entities via LLM:                        │    │    │
│  │  │    - cuisine (Italian)                              │    │    │
│  │  │    - location (Downtown Dubai)                      │    │    │
│  │  │    - price_max (200)                                │    │    │
│  │  │    - amenities ([outdoor seating])                  │    │    │
│  │  │  • Use chat history for context                     │    │    │
│  │  │  • Detect ambiguity                                 │    │    │
│  │  │  Technology: GPT-4o-mini with structured prompts    │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └───────────────────────┬──────────────────────────────────────┘    │
│                          │                                           │
│                          ▼                                           │
│                   ┌─────────────┐                                    │
│                   │ Clarify?    │ (Conditional Branch)               │
│                   └──────┬──┬───┘                                    │
│                          │  │                                        │
│                      Yes │  │ No                                     │
│                          │  │                                        │
│        ┌─────────────────┘  └────────────┐                          │
│        │                                  │                          │
│        ▼                                  ▼                          │
│  ┌──────────┐                  ┌─────────────────────────────────┐  │
│  │ Return   │                  │  AGENT 2: Retrieval             │  │
│  │ Question │                  │  ┌─────────────────────────────┐│  │
│  │ to User  │                  │  │ • Build search query        ││  │
│  └────┬─────┘                  │  │ • Execute hybrid search:    ││  │
│       │                        │  │   - Semantic (60%)          ││  │
│       │                        │  │   - Keyword BM25 (40%)      ││  │
│       │                        │  │ • Apply entity filters:     ││  │
│       │                        │  │   - Hard (cuisine, location)││  │
│       │                        │  │   - Soft (price, amenities) ││  │
│       │                        │  │ • Multi-factor re-ranking   ││  │
│       │                        │  │ • Progressive relaxation    ││  │
│       │                        │  │   if no results             ││  │
│       │                        │  └─────────────────────────────┘│  │
│       │                        └──────────────┬──────────────────┘  │
│       │                                       │                     │
│       │                                       ▼                     │
│       │                        ┌─────────────────────────────────┐  │
│       │                        │  AGENT 3: Response Generation   │  │
│       │                        │  ┌─────────────────────────────┐│  │
│       │                        │  │ • Format restaurant data    ││  │
│       │                        │  │ • Generate via LLM:         ││  │
│       │                        │  │   - Warm, personalized tone ││  │
│       │                        │  │   - Top 3-5 recommendations ││  │
│       │                        │  │   - Key features highlighted││  │
│       │                        │  │   - Practical details       ││  │
│       │                        │  │   - Next steps/alternatives ││  │
│       │                        │  │ • Use chat history context  ││  │
│       │                        │  │ Technology: GPT-4o-mini     ││  │
│       │                        │  └─────────────────────────────┘│  │
│       │                        └──────────────┬──────────────────┘  │
│       │                                       │                     │
│       └───────────────────────────────────────┘                     │
│                                               │                     │
│                                               ▼                     │
│                                         ┌──────────┐                │
│                                         │   END    │                │
│                                         └──────────┘                │
│                                                                      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ Calls
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM                                   │
│                      (src/rag_system.py)                             │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  RestaurantRAGSystem                                          │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  Data Pipeline                                       │     │  │
│  │  │  1. Load JSON (50 restaurants)                       │     │  │
│  │  │  2. Create rich text representations                 │     │  │
│  │  │  3. Generate embeddings (text-embedding-3-small)     │     │  │
│  │  │  4. Store in vector DB                               │     │  │
│  │  └─────────────────────────────────────────────────────┘     │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  Hybrid Search Engine                                │     │  │
│  │  │                                                       │     │  │
│  │  │  Semantic Retriever (60%) ────┐                      │     │  │
│  │  │  • Vector similarity search    │                      │     │  │
│  │  │  • Cosine distance            │                      │     │  │
│  │  │  • Returns top-k results      │                      │     │  │
│  │  │                               │                      │     │  │
│  │  │  BM25 Retriever (40%) ────────┤→ Ensemble Combiner   │     │  │
│  │  │  • Keyword-based search       │  • Weighted merge    │     │  │
│  │  │  • TF-IDF scoring            │  • Deduplication     │     │  │
│  │  │  • Returns top-k results      │  • Score fusion      │     │  │
│  │  │                               │                      │     │  │
│  │  └─────────────────────────────────────────────────────┘     │  │
│  │                                  │                            │  │
│  │                                  ▼                            │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  Entity-Based Filtering                             │     │  │
│  │  │                                                       │     │  │
│  │  │  Hard Filters (Must Match):                          │     │  │
│  │  │  • Cuisine = Italian ✓                               │     │  │
│  │  │  • Location in [Downtown, DIFC, Business Bay] ✓      │     │  │
│  │  │                                                       │     │  │
│  │  │  Soft Filters (Score Adjustment):                    │     │  │
│  │  │  • Price ≤ 220 (20% flex) → Score × overlap%         │     │  │
│  │  │  • Has outdoor seating → Score × 1.15                │     │  │
│  │  │  • Rating ≥ 4.0 → Score × 1.1                        │     │  │
│  │  └─────────────────────────────────────────────────────┘     │  │
│  │                                  │                            │  │
│  │                                  ▼                            │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  Multi-Factor Re-Ranking                            │     │  │
│  │  │                                                       │     │  │
│  │  │  Score = base_similarity                             │     │  │
│  │  │        × (1 + rating/5 × 0.2)                        │     │  │
│  │  │        × (1 + log10(reviews)/4 × 0.1)                │     │  │
│  │  │        × (1.15 ^ matching_attributes)                │     │  │
│  │  └─────────────────────────────────────────────────────┘     │  │
│  │                                  │                            │  │
│  │                                  ▼                            │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  Edge Case Handling                                  │     │  │
│  │  │                                                       │     │  │
│  │  │  if no_results:                                      │     │  │
│  │  │    • Remove price constraint                         │     │  │
│  │  │    • Retry search                                    │     │  │
│  │  │    • If still none, broaden location                 │     │  │
│  │  │    • Add explanatory note                            │     │  │
│  │  └─────────────────────────────────────────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ Uses
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Vector Store      │  │   BM25 Index    │  │  Source Data    ││
│  │   (ChromaDB)        │  │   (In-Memory)   │  │     (JSON)      ││
│  │                     │  │                 │  │                 ││
│  │  • 50 documents     │  │  • Token index  │  │  • 50 restaurants│
│  │  • Embeddings       │  │  • TF-IDF       │  │  • Full metadata││
│  │  • Metadata         │  │  • Fast lookup  │  │  • Ground truth ││
│  │  • Persistent       │  │                 │  │                 ││
│  │                     │  │                 │  │                 ││
│  │  Storage:           │  │  Rebuild:       │  │  Format:        ││
│  │  ./data/vectorstore │  │  On startup     │  │  restaurants.json│
│  └─────────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Powered by
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                              │
│                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   OpenAI API        │  │   LangChain Ecosystem               │  │
│  │                     │  │                                     │  │
│  │  • GPT-4o-mini      │  │  • LangChain Core                   │  │
│  │    (LLM)            │  │  • LangGraph (Workflow)             │  │
│  │  • text-embedding-  │  │  • Document Loaders                 │  │
│  │    3-small          │  │  • Retrievers                       │  │
│  │    (Embeddings)     │  │  • Memory Management                │  │
│  └─────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### Request Processing Flow

```
User Query: "Find Italian restaurants in Downtown with outdoor seating under AED 200"
     │
     ├─► Agent 1: Query Understanding
     │   ├─► LLM Call (GPT-4o-mini)
     │   │   Input: Query + Chat History + System Prompt
     │   │   Output: JSON entities
     │   │
     │   ├─► Post-Processing
     │   │   • Normalize price ranges
     │   │   • Expand location aliases
     │   │   • Convert to structured format
     │   │
     │   └─► Output:
     │       {
     │         "cuisine": "Italian",
     │         "location": "Downtown Dubai",
     │         "price_max": 200,
     │         "amenities": ["outdoor seating"]
     │       }
     │
     ├─► Conditional Check
     │   └─► No clarification needed → Continue
     │
     ├─► Agent 2: Retrieval
     │   │
     │   ├─► Build Search Query
     │   │   "Italian cuisine Downtown Dubai outdoor seating"
     │   │
     │   ├─► Hybrid Search
     │   │   ├─► Semantic Search (60%)
     │   │   │   ├─► Generate query embedding
     │   │   │   ├─► Vector similarity search (ChromaDB)
     │   │   │   └─► Get top 10 by cosine distance
     │   │   │
     │   │   ├─► Keyword Search (40%)
     │   │   │   ├─► Tokenize query
     │   │   │   ├─► BM25 scoring
     │   │   │   └─► Get top 10 by BM25 score
     │   │   │
     │   │   └─► Ensemble Combine
     │   │       ├─► Weighted merge (0.6, 0.4)
     │   │       ├─► Deduplicate
     │   │       └─► 15 combined results
     │   │
     │   ├─► Apply Entity Filters
     │   │   ├─► Hard filters: cuisine = Italian ✓, location = Downtown ✓
     │   │   ├─► Soft filters: price ≤ 220 ✓, outdoor seating ✓
     │   │   └─► 5 restaurants pass filters
     │   │
     │   ├─► Re-Rank
     │   │   ├─► Factor in rating (4.3 → boost 17%)
     │   │   ├─► Factor in reviews (634 → boost 7%)
     │   │   ├─► Factor in attributes (family-friendly → boost 15%)
     │   │   └─► Final ranked list
     │   │
     │   └─► Output:
     │       [
     │         {"name": "Mama's Kitchen", "score": 0.92, ...},
     │         {"name": "Bella Vista", "score": 0.87, ...},
     │         ...
     │       ]
     │
     ├─► Agent 3: Response Generation
     │   │
     │   ├─► Format Restaurant Data
     │   │   └─► Convert to readable text format
     │   │
     │   ├─► LLM Call (GPT-4o-mini)
     │   │   Input: Query + Entities + Restaurants + System Prompt
     │   │   Output: Natural language response
     │   │
     │   └─► Output:
     │       "I found some wonderful Italian restaurants in Downtown matching
     │        your preferences!
     │        
     │        🍝 Mama's Kitchen (Downtown Dubai)
     │        A family-run trattoria serving homemade pasta...
     │        • Price: AED 120-180 ✓ Under budget!
     │        • Rating: 4.3/5 (634 reviews)
     │        • Outdoor seating available ✓
     │        ..."
     │
     └─► Return to User
         └─► Update chat history
```

## Component Interaction Diagram

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │
       │ 1. Query
       ▼
┌─────────────────────┐
│  Main Application   │
│  • Session mgmt     │
│  • History tracking │
└──────┬──────────────┘
       │
       │ 2. Invoke workflow
       ▼
┌─────────────────────────────────────────────┐
│          LangGraph Workflow                 │
│  ┌───────────┐  ┌───────────┐  ┌─────────┐ │
│  │  Agent 1  │→ │  Agent 2  │→ │ Agent 3 │ │
│  └─────┬─────┘  └─────┬─────┘  └────┬────┘ │
└────────┼──────────────┼─────────────┼──────┘
         │              │             │
         │ 3. Extract   │ 4. Search   │ 6. Generate
         │ entities     │ & filter    │ response
         │              │             │
         │              ▼             │
         │      ┌──────────────┐     │
         │      │  RAG System  │     │
         │      │  • Hybrid    │     │
         │      │    search    │     │
         │      │  • Filtering │     │
         │      │  • Ranking   │     │
         │      └───────┬──────┘     │
         │              │             │
         │              │ 5. Results  │
         ▼              ▼             ▼
    ┌────────────────────────────────────┐
    │         OpenAI API                 │
    │  • Embeddings (Agent 2)            │
    │  • LLM calls (Agent 1 & 3)         │
    └────────────────────────────────────┘
```

## Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Python 3.12+  │  CLI (argparse)  │  Logging (stdlib)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  LangGraph 0.0.20  │  State Management  │  Conditional Flow │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  LangChain 0.1.0        │  OpenAI GPT-4o-mini               │
│  Prompt Templates       │  JSON Mode                        │
│  Memory Management      │  Structured Outputs               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Semantic Search         │  Keyword Search                   │
│  • text-embedding-3-small│  • BM25 (rank-bm25)              │
│  • ChromaDB 0.4.22       │  • TF-IDF scoring                │
│  • Cosine similarity     │  • Token-based                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  JSON (restaurants)  │  ChromaDB (vectors)  │  Logs (text)  │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Architecture (Future)

```
                    ┌─────────────┐
                    │   Client    │
                    │  (Browser)  │
                    └──────┬──────┘
                           │
                           │ HTTPS
                           ▼
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ FastAPI  │   │ FastAPI  │   │ FastAPI  │
        │ Instance │   │ Instance │   │ Instance │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Redis Cache   │
                   │  (Embeddings)  │
                   └────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │   Pinecone     │
                   │ (Vector Store) │
                   └────────────────┘
```

## Summary

This architecture demonstrates:

1. **Modularity**: Clear separation between agents, RAG system, and data
2. **Scalability**: Each component can be scaled independently
3. **Maintainability**: Well-defined interfaces and responsibilities
4. **Extensibility**: Easy to add new agents, data sources, or features
5. **Production-Ready**: Comprehensive error handling, logging, and monitoring hooks

The design prioritizes:
- **Speed**: Hybrid search for fast, accurate retrieval
- **Quality**: Multi-agent architecture for specialized tasks
- **Reliability**: Error handling and fallbacks at every level
- **Flexibility**: Configuration-driven, easy to customize

