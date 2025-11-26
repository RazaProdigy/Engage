"""
RAG System Implementation with Vector Database and Hybrid Search.
Handles document ingestion, embedding generation, and semantic retrieval.
"""
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.config import (
    EMBEDDING_CONFIG,
    VECTOR_STORE_CONFIG,
    SEARCH_CONFIG,
    DATA_CONFIG,
)
from src.observability import (
    record_retrieval,
    record_filtering,
    record_error,
    update_vector_store_size,
    RERANK_LATENCY,
    FILTER_LATENCY
)

logger = logging.getLogger(__name__)


class RestaurantRAGSystem:
    """
    Production-ready RAG system for restaurant search.
    
    Features:
    - Hybrid search combining semantic and keyword matching
    - Metadata filtering for structured queries
    - Dynamic re-ranking based on multiple factors
    - Edge case handling (no results, ambiguous queries)
    """
    
    def __init__(self, api_key: str):
        """Initialize the RAG system with embeddings and vector store."""
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_CONFIG["model"],
            openai_api_key=api_key
        )
        self.vectorstore: Optional[Chroma] = None
        self.documents: List[Document] = []
        self.restaurants_data: List[Dict[str, Any]] = []
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        
        logger.info("RAG System initialized")
    
    def load_restaurant_data(self, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load restaurant data from JSON file."""
        if data_path is None:
            data_path = DATA_CONFIG["restaurant_data_path"]
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.restaurants_data = json.load(f)
            logger.info(f"Loaded {len(self.restaurants_data)} restaurants")
            return self.restaurants_data
        except Exception as e:
            logger.error(f"Error loading restaurant data: {e}")
            raise
    
    def create_documents(self) -> List[Document]:
        """
        Create LangChain documents from restaurant data.
        Each restaurant becomes a document with rich metadata.
        """
        documents = []
        
        for restaurant in self.restaurants_data:
            # Create rich text representation for embedding
            content = self._create_restaurant_text(restaurant)
            
            # Extract metadata for filtering
            metadata = {
                "id": restaurant["id"],
                "name": restaurant["name"],
                "cuisine": restaurant["cuisine"],
                "location": restaurant["location"],
                "price_range": restaurant["price_range"],
                "amenities": restaurant["amenities"],
                "attributes": restaurant["attributes"],
                "rating": restaurant["rating"],
                "review_count": restaurant["review_count"],
                "coordinates": restaurant["coordinates"],
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        self.documents = documents
        logger.info(f"Created {len(documents)} documents")
        return documents
    
    def _create_restaurant_text(self, restaurant: Dict[str, Any]) -> str:
        """Create rich text representation for embedding generation."""
        text_parts = [
            f"Restaurant: {restaurant['name']}",
            f"Cuisine: {restaurant['cuisine']}",
            f"Location: {restaurant['location']}",
            f"Price Range: {restaurant['price_range']}",
            f"Rating: {restaurant['rating']}/5.0 ({restaurant['review_count']} reviews)",
            f"Description: {restaurant['description']}",
            f"Amenities: {restaurant['amenities']}",
            f"Attributes: {restaurant['attributes']}",
            f"Hours: {restaurant['opening_hours']}",
        ]
        return "\n".join(text_parts)
    
    def build_vector_store(self, persist: bool = True) -> Chroma:
        """
        Build vector store with embeddings.
        Supports persistence for production use.
        """
        if not self.documents:
            raise ValueError("No documents available. Call create_documents first.")
        
        persist_dir = VECTOR_STORE_CONFIG["persist_directory"] if persist else None
        
        # Create vector store with embeddings
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name=VECTOR_STORE_CONFIG["collection_name"],
            persist_directory=persist_dir,
        )
        
        logger.info(f"Vector store built with {len(self.documents)} documents")
        return self.vectorstore
    
    def load_vector_store(self) -> Chroma:
        """Load existing vector store from disk."""
        persist_dir = VECTOR_STORE_CONFIG["persist_directory"]
        
        if not Path(persist_dir).exists():
            raise ValueError(f"Vector store not found at {persist_dir}")
        
        self.vectorstore = Chroma(
            collection_name=VECTOR_STORE_CONFIG["collection_name"],
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        
        logger.info("Vector store loaded from disk")
        return self.vectorstore
    
    def setup_hybrid_retriever(self) -> EnsembleRetriever:
        """
        Setup hybrid retriever combining semantic and keyword search.
        
        Benefits:
        - Semantic search: Captures meaning and context
        - BM25: Catches exact keyword matches
        - Ensemble: Best of both worlds
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if not self.documents:
            raise ValueError("Documents not created")
        
        # Semantic retriever (vector similarity)
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": SEARCH_CONFIG["top_k"]}
        )
        
        # Keyword retriever (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = SEARCH_CONFIG["top_k"]
        
        # Ensemble retriever with weighted combination
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, self.bm25_retriever],
            weights=[SEARCH_CONFIG["semantic_weight"], SEARCH_CONFIG["keyword_weight"]]
        )
        
        logger.info("Hybrid retriever setup complete")
        return self.ensemble_retriever
    
    def semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform semantic search with optional metadata filtering.
        
        Args:
            query: Natural language search query
            filters: Metadata filters (cuisine, location, etc.)
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        k = top_k or SEARCH_CONFIG["top_k"]
        
        # Perform similarity search
        if filters:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filters
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Semantic search returned {len(results)} results")
        return results
    
    def hybrid_search(
        self,
        query: str,
        extracted_entities: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Advanced hybrid search with entity-based filtering and re-ranking.
        
        Args:
            query: Natural language query
            extracted_entities: Structured entities from query understanding
            top_k: Number of results
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        k = top_k or SEARCH_CONFIG["top_k"]
        
        # Step 1: Get results using ensemble retriever (automatically combines semantic + BM25)
        retrieval_start = time.time()
        try:
            if self.ensemble_retriever:
                # Ensemble retriever already handles combination, weighting, and deduplication
                docs = self.ensemble_retriever.get_relevant_documents(query)
                # Convert to (doc, score) format with default scores based on rank
                results = [(doc, 1.0 / (1.0 + i * 0.1)) for i, doc in enumerate(docs[:k*2])]
                retriever_type = 'hybrid'
            else:
                # Fallback to semantic only if ensemble not set up
                logger.warning("Ensemble retriever not initialized, using semantic search only")
                results = self.vectorstore.similarity_search_with_score(query, k=k*2)
                retriever_type = 'semantic'
            
            retrieval_duration = time.time() - retrieval_start
            record_retrieval(retriever_type, retrieval_duration, len(results), success=True)
            
        except Exception as e:
            retrieval_duration = time.time() - retrieval_start
            record_retrieval('hybrid', retrieval_duration, 0, success=False)
            record_error('retrieval', type(e).__name__)
            raise
        
        # Step 2: Apply entity-based filtering if available
        filter_start = time.time()
        initial_count = len(results)
        if extracted_entities:
            results = self._apply_entity_filters(results, extracted_entities)
        filter_duration = time.time() - filter_start
        
        if filter_duration > 0.001:  # Only record if measurable
            FILTER_LATENCY.observe(filter_duration)
        
        # Step 3: Re-rank based on multiple factors
        rerank_start = time.time()
        ranked_results = self._rerank_results(results, extracted_entities)
        rerank_duration = time.time() - rerank_start
        
        if rerank_duration > 0.001:
            RERANK_LATENCY.observe(rerank_duration)
        
        # Step 4: Return top k results
        return ranked_results[:k]
    
    def _apply_entity_filters(
        self,
        results: List[Tuple[Document, float]],
        entities: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """Apply hard and soft filters based on extracted entities."""
        filtered = []
        
        logger.info(f"🔍 Applying filters with entities: {entities}")
        
        
        for doc, score in results:
            metadata = doc.metadata
            
            # Hard filters (must match)
            if entities.get("cuisine"):
                if metadata["cuisine"].lower() != entities["cuisine"].lower():
                    logger.debug(f"❌ Filtered out {metadata['name']} - cuisine mismatch")
                    continue
            
            if entities.get("location"):
                location_match = self._location_matches(
                    metadata["location"],
                    entities["location"]
                )
                if not location_match:
                    logger.info(f"❌ Filtered out {metadata['name']} - location mismatch: {metadata['location']} != {entities['location']}")
                    print(f"❌ Filtered out {metadata['name']} - location: {metadata['location']} != {entities['location']}")
                    continue
                else:
                    logger.info(f"✅ Location match: {metadata['name']} - {metadata['location']} matches {entities['location']}")
                    print(f"✅ Location match: {metadata['name']} - {metadata['location']} matches {entities['location']}")
            
            # Soft filters (affect score but don't eliminate)
            adjusted_score = score
            
            # Price range filter (with flexibility)
            if entities.get("price_min") or entities.get("price_max"):
                price_match_score = self._calculate_price_match(
                    metadata["price_range"],
                    entities.get("price_min"),
                    entities.get("price_max")
                )
                adjusted_score *= price_match_score
            
            # Rating filter
            if entities.get("rating_min"):
                if metadata["rating"] >= entities["rating_min"]:
                    adjusted_score *= 1.1  # Boost for meeting rating requirement
                else:
                    adjusted_score *= 0.7  # Penalize but don't eliminate
            
            # Amenities filter
            if entities.get("amenities"):
                amenity_match = self._check_amenities(
                    metadata["amenities"],
                    entities["amenities"]
                )
                if amenity_match:
                    adjusted_score *= 1.15
            
            filtered.append((doc, adjusted_score))
        
        return filtered
    
    def _location_matches(self, restaurant_location: str, query_location: str) -> bool:
        """Check if locations match (with fuzzy matching)."""
        restaurant_location = restaurant_location.lower()
        query_location = query_location.lower()
        
        print(f"  🗺️ Comparing: '{restaurant_location}' vs '{query_location}'")
        
        # Direct match
        if query_location in restaurant_location or restaurant_location in query_location:
            print(f"    ✅ Direct match!")
            return True
        
        # Check aliases (from config)
        from src.config import LOCATION_ALIASES
        for alias, locations in LOCATION_ALIASES.items():
            if query_location in alias or alias in query_location:
                match = any(loc.lower() in restaurant_location for loc in locations)
                print(f"    📍 Alias '{alias}' matched query, checking locations {locations}: {match}")
                return match
        
        print(f"    ❌ No match")
        return False
    
    def _calculate_price_match(
        self,
        price_range: str,
        min_price: Optional[float],
        max_price: Optional[float]
    ) -> float:
        """Calculate price match score with flexibility."""
        # Extract numeric values from "AED 100 - 200" format
        try:
            parts = price_range.replace("AED", "").strip().split("-")
            rest_min = float(parts[0].strip())
            rest_max = float(parts[1].strip())
            
            # Calculate overlap percentage
            if min_price and max_price:
                overlap_start = max(rest_min, min_price)
                overlap_end = min(rest_max, max_price)
                
                if overlap_end < overlap_start:
                    # Allow 20% flexibility
                    if rest_min <= max_price * 1.2 and rest_max >= min_price * 0.8:
                        return 0.85  # Partial match
                    return 0.5  # Out of range but don't eliminate
                else:
                    overlap = overlap_end - overlap_start
                    total_range = max_price - min_price
                    return 0.8 + (overlap / total_range) * 0.2
            
            return 1.0
        except:
            return 0.9  # Default if parsing fails
    
    def _check_amenities(self, restaurant_amenities: str, required_amenities: List[str]) -> bool:
        """Check if restaurant has required amenities."""
        restaurant_amenities = restaurant_amenities.lower()
        return any(amenity.lower() in restaurant_amenities for amenity in required_amenities)
    
    def _rerank_results(
        self,
        results: List[Tuple[Document, float]],
        entities: Optional[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank results based on multiple factors:
        - Semantic similarity score
        - Rating and review count
        - Attribute matches
        - Popularity signals
        """
        reranked = []
        
        for doc, score in results:
            metadata = doc.metadata
            
            # Base score from semantic similarity
            final_score = score
            
            # Factor 1: Rating (normalized to 0-1)
            rating_boost = metadata["rating"] / 5.0
            final_score *= (1 + rating_boost * 0.2)
            
            # Factor 2: Review count (logarithmic scale)
            import math
            review_boost = math.log10(max(metadata["review_count"], 1)) / 4
            final_score *= (1 + review_boost * 0.1)
            
            # Factor 3: Attribute matches
            if entities and entities.get("attributes"):
                restaurant_attrs = metadata["attributes"].lower()
                for attr in entities["attributes"]:
                    if attr.lower() in restaurant_attrs:
                        final_score *= 1.15
            
            reranked.append((doc, final_score))
        
        # Sort by final score (higher is better)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def get_restaurant_by_id(self, restaurant_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full restaurant data by ID."""
        for restaurant in self.restaurants_data:
            if restaurant["id"] == restaurant_id:
                return restaurant
        return None
    
    def initialize_pipeline(self, force_rebuild: bool = False) -> None:
        """
        Initialize the complete RAG pipeline.
        
        Args:
            force_rebuild: If True, rebuild vector store even if it exists
        """
        # Load data
        self.load_restaurant_data()
        
        # Create documents
        self.create_documents()
        
        # Build or load vector store
        persist_dir = Path(VECTOR_STORE_CONFIG["persist_directory"])
        
        if force_rebuild or not persist_dir.exists():
            logger.info("Building new vector store...")
            self.build_vector_store(persist=True)
        else:
            logger.info("Loading existing vector store...")
            self.load_vector_store()
        
        # Setup hybrid retriever (semantic + BM25)
        self.setup_hybrid_retriever()
        
        # Update observability metrics
        update_vector_store_size(len(self.documents))
        
        logger.info("RAG pipeline initialized successfully")

