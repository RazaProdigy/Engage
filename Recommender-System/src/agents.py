"""
Multi-Agent Workflow using LangGraph.
Implements three specialized agents with conditional logic and memory management.
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional, Annotated, TypedDict, Sequence
from operator import add

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END

from src.config import LLM_CONFIG, SYSTEM_PROMPTS, AGENT_CONFIG, PRICE_RANGES
from src.rag_system import RestaurantRAGSystem
from src.observability import (
    record_llm_call,
    record_entity_extraction,
    record_request,
    record_error,
    track_active_requests,
    RESPONSE_LENGTH,
    REQUEST_COUNT,
    health_checker
)

logger = logging.getLogger(__name__)


# Define the state structure for the agent workflow
class AgentState(TypedDict):
    """State shared across agents in the workflow."""
    query: str
    chat_history: Annotated[Sequence[tuple], add]
    extracted_entities: Optional[Dict[str, Any]]
    retrieved_restaurants: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]
    iteration_count: int


class QueryUnderstandingAgent:
    """
    Agent 1: Query Understanding & Entity Extraction
    
    Responsibilities:
    - Parse natural language queries
    - Extract structured entities (cuisine, location, price, etc.)
    - Detect ambiguous queries requiring clarification
    - Handle multi-intent queries
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPTS["query_understanding"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ])
        self.chain = self.prompt | self.llm
        
    def extract_entities(self, state: AgentState) -> AgentState:
        """
        Extract structured entities from user query.
        Handles edge cases like ambiguous or incomplete queries.
        """
        query = state["query"]
        chat_history = state.get("chat_history", [])
        
        logger.info(f"Extracting entities from query: {query}")
        
        start_time = time.time()
        fallback_used = False
        
        try:
            # Build context from chat history
            history_messages = []
            for msg_type, content in chat_history[-3:]:  # Last 3 turns
                if msg_type == "human":
                    history_messages.append(HumanMessage(content=content))
                else:
                    history_messages.append(AIMessage(content=content))
            
            # Call LLM for entity extraction
            llm_start = time.time()
            response = self.chain.invoke({
                "query": query,
                "chat_history": history_messages
            })
            llm_duration = time.time() - llm_start
            
            # Record LLM metrics
            record_llm_call(
                agent='query_understanding',
                model=LLM_CONFIG["model"],
                duration=llm_duration,
                prompt_tokens=getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0),
                completion_tokens=getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0),
                success=True
            )
            
            # Parse response as JSON
            entities = self._parse_llm_response(response.content)
            
            # Post-process entities
            entities = self._post_process_entities(entities)
            
            # Check if clarification needed
            needs_clarification, clarification_q = self._check_clarification_needed(entities, query)
            
            state["extracted_entities"] = entities
            state["needs_clarification"] = needs_clarification
            state["clarification_question"] = clarification_q
            
            logger.info(f"Extracted entities: {entities}")
            
            # Record entity extraction metrics
            duration = time.time() - start_time
            num_entities = len([v for v in entities.values() if v is not None])
            record_entity_extraction(duration, num_entities, success=True, fallback=False)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            record_error('query_understanding', type(e).__name__)
            
            # Record failed LLM call
            llm_duration = time.time() - start_time
            record_llm_call(
                agent='query_understanding',
                model=LLM_CONFIG["model"],
                duration=llm_duration,
                success=False
            )
            
            # Fallback: try to extract basic entities
            state["extracted_entities"] = self._fallback_extraction(query)
            state["needs_clarification"] = False
            fallback_used = True
            
            # Record fallback usage
            duration = time.time() - start_time
            record_entity_extraction(duration, 0, success=True, fallback=True)
        
        return state
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured entities."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end]
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end]
            
            entities = json.loads(response.strip())
            return entities
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, attempting fallback")
            return {}
    
    def _post_process_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process extracted entities for better matching.
        - Normalize price ranges
        - Expand location aliases
        - Handle synonyms
        """
        # Price range processing
        if entities.get("price_range"):
            price_str = entities["price_range"].lower()
            
            # Check semantic price ranges
            for key, (min_val, max_val) in PRICE_RANGES.items():
                if key in price_str:
                    entities["price_min"] = min_val
                    entities["price_max"] = max_val
                    break
            
            # Try to extract numeric values
            if "aed" in price_str or "dirham" in price_str:
                import re
                numbers = re.findall(r'\d+', price_str)
                if len(numbers) >= 1:
                    entities["price_max"] = int(numbers[0])
                if len(numbers) >= 2:
                    entities["price_min"] = int(numbers[0])
                    entities["price_max"] = int(numbers[1])
        
        # Convert amenities to list if string
        if entities.get("amenities") and isinstance(entities["amenities"], str):
            entities["amenities"] = [a.strip() for a in entities["amenities"].split(",")]
        
        # Convert attributes to list if string
        if entities.get("attributes") and isinstance(entities["attributes"], str):
            entities["attributes"] = [a.strip() for a in entities["attributes"].split(",")]
        
        return entities
    
    def _check_clarification_needed(
        self,
        entities: Dict[str, Any],
        query: str
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if clarification is needed from user.
        
        Cases requiring clarification:
        - Very generic queries without specific criteria
        - Ambiguous location or cuisine
        - Conflicting requirements
        """
        # Too generic
        if not any(entities.get(k) for k in ["cuisine", "location", "price_range", "attributes"]):
            if len(query.split()) < 5:  # Very short query
                return True, "I'd love to help you find a restaurant! Could you tell me what type of cuisine you're interested in, or which area of Dubai you prefer?"
        
        # Ambiguous location
        if entities.get("location"):
            location = entities["location"].lower()
            if location in ["dubai", "uae"]:  # Too broad
                return True, "Dubai has many great areas! Are you looking for somewhere specific like Downtown, Marina, Jumeirah, or another neighborhood?"
        
        return False, None
    
    def _fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback extraction using simple keyword matching."""
        query_lower = query.lower()
        entities = {}
        
        # Common cuisines
        cuisines = ["italian", "chinese", "indian", "japanese", "french", "arabic", 
                   "thai", "mexican", "american", "lebanese", "greek", "korean"]
        for cuisine in cuisines:
            if cuisine in query_lower:
                entities["cuisine"] = cuisine.capitalize()
                break
        
        # Common locations
        locations = ["downtown", "marina", "jumeirah", "deira", "bur dubai", 
                    "jbr", "difc", "business bay"]
        for location in locations:
            if location in query_lower:
                entities["location"] = location
                break
        
        # Price indicators
        if "cheap" in query_lower or "budget" in query_lower:
            entities["price_max"] = 100
        elif "expensive" in query_lower or "luxury" in query_lower or "fine dining" in query_lower:
            entities["price_min"] = 200
        
        # Amenities
        if "outdoor" in query_lower:
            entities["amenities"] = ["outdoor seating"]
        if "parking" in query_lower or "valet" in query_lower:
            entities["amenities"] = entities.get("amenities", []) + ["valet parking"]
        
        return entities


class RetrievalAgent:
    """
    Agent 2: Restaurant Retrieval & Filtering
    
    Responsibilities:
    - Query the RAG system with extracted entities
    - Apply intelligent filtering and ranking
    - Handle no-results scenarios
    - Suggest alternatives when needed
    """
    
    def __init__(self, rag_system: RestaurantRAGSystem):
        self.rag_system = rag_system
    
    def retrieve_restaurants(self, state: AgentState) -> AgentState:
        """
        Retrieve and filter restaurants based on extracted entities.
        """
        entities = state.get("extracted_entities", {})
        query = state["query"]
        
        logger.info(f"Retrieving restaurants for entities: {entities}")
        
        try:
            # Perform hybrid search
            results = self.rag_system.hybrid_search(
                query=query,
                extracted_entities=entities,
                top_k=10
            )

            logger.info(f"Results FROM HYBRID SEARCH: {results}")
            print(f"Results FROM HYBRID SEARCH: {results}")
            
            # Convert to restaurant dictionaries with scores
            restaurants = []
            for doc, score in results:
                restaurant_id = doc.metadata["id"]
                restaurant = self.rag_system.get_restaurant_by_id(restaurant_id)
                if restaurant:
                    restaurant["relevance_score"] = float(score)
                    restaurants.append(restaurant)
            
            # Handle no results - try relaxing constraints
            if not restaurants:
                logger.warning("No results found, trying relaxed search")
                restaurants = self._relaxed_search(query, entities)
            
            state["retrieved_restaurants"] = restaurants
            logger.info(f"Retrieved {len(restaurants)} restaurants")
            
        except Exception as e:
            logger.error(f"Error retrieving restaurants: {e}")
            state["retrieved_restaurants"] = []
        
        return state
    
    def _relaxed_search(
        self,
        query: str,
        entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform relaxed search when no results found.
        Progressively remove constraints.
        """
        # Try without price constraint
        relaxed_entities = entities.copy()
        relaxed_entities.pop("price_min", None)
        relaxed_entities.pop("price_max", None)
        
        results = self.rag_system.hybrid_search(
            query=query,
            extracted_entities=relaxed_entities,
            top_k=5
        )
        
        if results:
            restaurants = []
            for doc, score in results:
                restaurant_id = doc.metadata["id"]
                restaurant = self.rag_system.get_restaurant_by_id(restaurant_id)
                if restaurant:
                    restaurant["relevance_score"] = float(score)
                    restaurant["note"] = "Price range adjusted"
                    restaurants.append(restaurant)
            return restaurants
        
        # Try with just cuisine or location
        if entities.get("cuisine"):
            minimal_entities = {"cuisine": entities["cuisine"]}
            results = self.rag_system.hybrid_search(
                query=entities["cuisine"],
                extracted_entities=minimal_entities,
                top_k=5
            )
            
            if results:
                restaurants = []
                for doc, score in results:
                    restaurant_id = doc.metadata["id"]
                    restaurant = self.rag_system.get_restaurant_by_id(restaurant_id)
                    if restaurant:
                        restaurant["relevance_score"] = float(score)
                        restaurant["note"] = "Broader search applied"
                        restaurants.append(restaurant)
                return restaurants
        
        return []


class ResponseGenerationAgent:
    """
    Agent 3: Response Generation with Personalization
    
    Responsibilities:
    - Generate engaging, personalized responses
    - Format restaurant information clearly
    - Provide relevant recommendations
    - Handle multi-turn conversation context
    - Suggest follow-up actions
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPTS["response_generation"]),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Query: {query}\n\nExtracted Preferences: {entities}\n\nMatching Restaurants: {restaurants}\n\nGenerate a personalized recommendation response."),
        ])
        self.chain = self.prompt | self.llm
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response with personalized recommendations."""
        query = state["query"]
        entities = state.get("extracted_entities", {})
        restaurants = state.get("retrieved_restaurants", [])
        chat_history = state.get("chat_history", [])
        
        logger.info(f"Generating response for {len(restaurants)} restaurants")
        
        start_time = time.time()
        
        try:
            # Build context from chat history
            history_messages = []
            for msg_type, content in chat_history[-3:]:
                if msg_type == "human":
                    history_messages.append(HumanMessage(content=content))
                else:
                    history_messages.append(AIMessage(content=content))
            
            # Format restaurant data
            restaurants_text = self._format_restaurants(restaurants[:5])  # Top 5
            
            # Generate response
            llm_start = time.time()
            response = self.chain.invoke({
                "query": query,
                "entities": json.dumps(entities, indent=2),
                "restaurants": restaurants_text,
                "chat_history": history_messages
            })
            llm_duration = time.time() - llm_start
            
            # Record LLM metrics
            record_llm_call(
                agent='response_generation',
                model=LLM_CONFIG["model"],
                duration=llm_duration,
                prompt_tokens=getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0),
                completion_tokens=getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0),
                success=True
            )
            
            state["final_response"] = response.content
            
            # Record response metrics
            RESPONSE_LENGTH.observe(len(response.content))
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["final_response"] = self._fallback_response(restaurants)
        
        return state
    
    def _format_restaurants(self, restaurants: List[Dict[str, Any]]) -> str:
        """Format restaurant data for LLM processing."""
        if not restaurants:
            return "No restaurants found matching the criteria."
        
        formatted = []
        for i, rest in enumerate(restaurants, 1):
            text = f"""
Restaurant {i}: {rest['name']}
- Cuisine: {rest['cuisine']}
- Location: {rest['location']}
- Price Range: {rest['price_range']}
- Rating: {rest['rating']}/5.0 ({rest['review_count']} reviews)
- Description: {rest['description'][:200]}...
- Amenities: {rest['amenities']}
- Attributes: {rest['attributes']}
- Hours: {rest['opening_hours']}
"""
            if rest.get("note"):
                text += f"- Note: {rest['note']}\n"
            
            formatted.append(text.strip())
        
        return "\n\n".join(formatted)
    
    def _fallback_response(self, restaurants: List[Dict[str, Any]]) -> str:
        """Generate simple fallback response if LLM fails."""
        if not restaurants:
            return "I couldn't find any restaurants matching your criteria. Could you try adjusting your preferences or provide more details?"
        
        response = "Here are some restaurants I found for you:\n\n"
        for rest in restaurants[:3]:
            response += f"**{rest['name']}** ({rest['cuisine']} • {rest['location']})\n"
            response += f"Price: {rest['price_range']} • Rating: {rest['rating']}/5.0\n"
            response += f"{rest['description'][:150]}...\n\n"
        
        return response


class RestaurantSearchAgentWorkflow:
    """
    Multi-agent workflow orchestrator using LangGraph.
    
    Architecture:
    1. Query Understanding Agent → Extract entities
    2. Conditional: Check if clarification needed
    3. Retrieval Agent → Search and filter
    4. Response Generation Agent → Create personalized response
    
    Features:
    - Conditional branching based on query clarity
    - Memory management for multi-turn conversations
    - Error handling and fallback strategies
    """
    
    def __init__(self, api_key: str, rag_system: RestaurantRAGSystem):
        self.api_key = api_key
        self.rag_system = rag_system
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            openai_api_key=api_key
        )
        
        # Initialize agents
        self.query_agent = QueryUnderstandingAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(rag_system)
        self.response_agent = ResponseGenerationAgent(self.llm)
        
        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key=AGENT_CONFIG["memory_key"],
            return_messages=True
        )
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Agent workflow initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with conditional logic."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand_query", self.query_agent.extract_entities)
        workflow.add_node("retrieve_restaurants", self.retrieval_agent.retrieve_restaurants)
        workflow.add_node("generate_response", self.response_agent.generate_response)
        
        # Define edges with conditional logic
        workflow.set_entry_point("understand_query")
        
        # Conditional edge: check if clarification needed
        workflow.add_conditional_edges(
            "understand_query",
            self._should_clarify,
            {
                "clarify": END,  # Return clarification question
                "retrieve": "retrieve_restaurants"
            }
        )
        
        # Continue to response generation
        workflow.add_edge("retrieve_restaurants", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _should_clarify(self, state: AgentState) -> str:
        """Decision function: should we ask for clarification?"""
        if state.get("needs_clarification", False):
            return "clarify"
        return "retrieve"
    
    def process_query(
        self,
        query: str,
        chat_history: Optional[List[tuple]] = None
    ) -> Dict[str, Any]:
        """
        Process user query through the agent workflow.
        
        Args:
            query: User's natural language query
            chat_history: Previous conversation turns
            
        Returns:
            Dict containing response and metadata
        """
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "extracted_entities": None,
            "retrieved_restaurants": None,
            "final_response": None,
            "needs_clarification": False,
            "clarification_question": None,
            "iteration_count": 0,
        }
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Handle clarification case
            if final_state.get("needs_clarification"):
                return {
                    "response": final_state["clarification_question"],
                    "type": "clarification",
                    "entities": final_state.get("extracted_entities"),
                }
            
            # Normal response
            return {
                "response": final_state["final_response"],
                "type": "recommendation",
                "entities": final_state.get("extracted_entities"),
                "restaurants": final_state.get("retrieved_restaurants", [])[:5],
                "total_found": len(final_state.get("retrieved_restaurants", [])),
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Could you please try rephrasing your query?",
                "type": "error",
                "error": str(e)
            }

