"""
Configuration settings for the RAG-based restaurant search system.
"""
import os
from typing import Dict, Any

# =============================================================================
# LANGSMITH OBSERVABILITY CONFIGURATION
# =============================================================================
LANGSMITH_CONFIG: Dict[str, Any] = {
    "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
    "api_key": os.getenv("LANGCHAIN_API_KEY", ""),
    "project_name": os.getenv("LANGCHAIN_PROJECT", "restaurant-recommender"),
    "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    # Trace tags for filtering in LangSmith dashboard
    "default_tags": ["restaurant-search", "rag-system"],
    # Metadata to include in all traces
    "default_metadata": {
        "app_version": "1.0.0",
        "system": "restaurant-recommender"
    }
}

# LLM Configuration
LLM_CONFIG: Dict[str, Any] = {
    "model": "gpt-4o-mini",  # Using GPT-4 for high-quality responses
    "temperature": 0.3,  # Lower temperature for more consistent results
    "max_tokens": 1000,
}

# Embedding Configuration
EMBEDDING_CONFIG: Dict[str, Any] = {
    "model": "text-embedding-3-small",  # OpenAI's efficient embedding model
    "chunk_size": 500,  # Characters per chunk
    "chunk_overlap": 50,  # Overlap between chunks for context preservation
}

# Vector Store Configuration
VECTOR_STORE_CONFIG: Dict[str, Any] = {
    "collection_name": "restaurants",
    "distance_metric": "cosine",
    "persist_directory": "./data/vectorstore",
}

# Hybrid Search Configuration
SEARCH_CONFIG: Dict[str, Any] = {
    "top_k": 10,  # Number of results to retrieve
    "similarity_threshold": 0.7,  # Minimum similarity score
    "semantic_weight": 0.6,  # Weight for semantic search (40% for keyword)
    "keyword_weight": 0.4,
}

# Agent Configuration
AGENT_CONFIG: Dict[str, Any] = {
    "max_iterations": 5,  # Maximum agent iterations
    "early_stopping": "generate",  # Stop early if answer is generated
    "memory_key": "chat_history",  # Key for conversation memory
    "return_intermediate_steps": True,
}

# Data Configuration
DATA_CONFIG: Dict[str, Any] = {
    "restaurant_data_path": "./data/restaurant.json",
    "metadata_fields": [
        "cuisine",
        "location",
        "price_range",
        "amenities",
        "attributes",
        "rating",
    ],
}

# System Prompts
SYSTEM_PROMPTS: Dict[str, str] = {
    "query_understanding": """You are a query understanding specialist for a restaurant search system.
Your task is to extract structured information from user queries.

Extract the following entities when present:
- cuisine: Type of food (e.g., Italian, Chinese, Indian)
- location: Area or neighborhood (e.g., Downtown Dubai, Marina)
- price_range: Budget constraints (extract min/max from AED amounts)
- amenities: Special features (e.g., outdoor seating, parking, live music)
- attributes: Dining style (e.g., fine dining, casual, family-friendly)
- dietary: Special dietary needs (e.g., vegetarian, vegan, halal)
- ambiance: Atmosphere preferences (e.g., romantic, lively, quiet)
- rating_min: Minimum rating requirement

Respond in JSON format with extracted entities. Set null for missing fields.
Handle ambiguous queries by asking clarifying questions.
""",
    "retrieval": """You are a restaurant retrieval specialist.
Your task is to filter and rank restaurants based on extracted criteria.

Apply filters strictly but rank results by relevance:
1. Hard filters: Must match cuisine, location constraints
2. Soft filters: Price range (allow 20% flexibility), amenities (partial match ok)
3. Ranking factors: Rating, review count, attribute matches

Return top matches with confidence scores and reasoning.
""",
    "response_generation": """You are a friendly restaurant recommendation assistant.
Create personalized, engaging responses based on retrieved restaurants.

Guidelines:
- Start with a warm greeting if first interaction
- Present 3-5 restaurants max, ordered by relevance
- Highlight key features matching user preferences
- Include practical details: location, price, signature dishes
- Add personality: use descriptive language for ambiance
- End with helpful next steps or alternatives

IMPORTANT - If no exact matches were found (indicated in the context):
- First acknowledge that no restaurants match the exact criteria
- Clearly explain what was not available (e.g., "I couldn't find any Italian restaurants in Downtown Dubai")
- Then suggest the nearby alternatives being shown
- Be transparent about why these are alternatives (e.g., "These are in nearby areas like Sharjah")
- Maintain a helpful, positive tone while being honest about the limitations

If no perfect matches but still reasonable:
- Suggest close alternatives with explanations
- Ask if user wants to adjust criteria
- Provide broader options

Maintain conversational tone throughout but prioritize honesty and transparency.
""",
}

# Price range mapping for semantic understanding
PRICE_RANGES: Dict[str, tuple] = {
    "budget": (0, 100),
    "cheap": (0, 100),
    "affordable": (50, 150),
    "moderate": (100, 200),
    "mid-range": (100, 200),
    "expensive": (200, 300),
    "upscale": (200, 350),
    "luxury": (300, 500),
    "fine dining": (250, 500),
}

# Location aliases for better matching
LOCATION_ALIASES: Dict[str, list] = {
    "downtown": ["Downtown Dubai", "DIFC", "Business Bay"],
    "beach": ["Jumeirah Beach", "JBR", "Palm Jumeirah"],
    "marina": ["Dubai Marina", "Marina"],
    "old dubai": ["Deira", "Bur Dubai", "Old Dubai"],
    "jumeirah": ["Jumeirah", "Jumeirah Lake Towers"],
}

# Environment variables (with defaults for demo)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Logging Configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "./logs/restaurant_search.log",
}

