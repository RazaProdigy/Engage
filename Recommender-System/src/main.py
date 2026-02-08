"""
Main application entry point for the RAG-based restaurant search system.
Provides CLI interface and example usage.

Observability:
- Prometheus metrics: /metrics endpoint
- LangSmith tracing: Automatic tracing of LLM calls, RAG operations, and data ingestion
- Health checks: /health endpoint
"""
import os
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from src.rag_system import RestaurantRAGSystem
from src.agents import RestaurantSearchAgentWorkflow
from src.config import OPENAI_API_KEY, LOGGING_CONFIG, LANGSMITH_CONFIG
from src.observability import (
    get_metrics,
    set_system_info,
    record_request,
    health_checker,
    track_active_requests
)
from src.langsmith_tracing import initialize_langsmith_tracing, get_tracer

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGGING_CONFIG["file"], mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# METRICS HTTP SERVER
# ============================================================================

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/metrics':
            # Return Prometheus metrics
            metrics_data = get_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(metrics_data)
        elif self.path == '/health':
            # Return health check
            health_stats = health_checker.get_stats()
            status_code = 200 if health_stats['healthy'] else 503
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            import json
            self.wfile.write(json.dumps(health_stats).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_metrics_server(port: int = 8000):
    """Start metrics server in background thread."""
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Metrics server started on http://0.0.0.0:{port}")
    logger.info(f"   - Metrics: http://localhost:{port}/metrics")
    logger.info(f"   - Health:  http://localhost:{port}/health")
    return server


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class RestaurantSearchApp:
    """
    Main application class orchestrating the restaurant search system.
    
    Features:
    - Interactive CLI for testing
    - Multi-turn conversation support
    - Session management
    - Performance monitoring
    """
    
    def __init__(self, api_key: str, rebuild_index: bool = False):
        """
        Initialize the restaurant search application.
        
        Args:
            api_key: OpenAI API key
            rebuild_index: Whether to rebuild the vector store
        """
        self.api_key = api_key
        
        logger.info("Initializing Restaurant Search System...")
        
        # Initialize RAG system
        self.rag_system = RestaurantRAGSystem(api_key)
        self.rag_system.initialize_pipeline(force_rebuild=rebuild_index)
        
        # Initialize agent workflow
        self.agent_workflow = RestaurantSearchAgentWorkflow(api_key, self.rag_system)
        
        # Conversation state
        self.chat_history: List[tuple] = []
        
        logger.info("System initialization complete")
    
    def search(self, query: str) -> dict:
        """
        Process a search query and return results.
        
        Args:
            query: Natural language search query
            
        Returns:
            Dict containing response and metadata
        """
        logger.info(f"Processing query: {query}")
        
        start_time = time.time()
        success = False
        
        try:
            with track_active_requests('restaurant_search'):
                # Process through agent workflow
                result = self.agent_workflow.process_query(query, self.chat_history)
                
                # Update chat history
                self.chat_history.append(("human", query))
                self.chat_history.append(("ai", result["response"]))
                
                # Keep only last 10 turns
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
                
                success = True
                health_checker.record_success()
                return result
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            health_checker.record_error()
            raise
        finally:
            duration = time.time() - start_time
            status = 'success' if success else 'error'
            record_request('restaurant_search', duration, status)
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def interactive_mode(self):
        """Run interactive CLI mode."""
        print("\n" + "="*70)
        print("Restaurant Search System - Interactive Mode")
        print("="*70)
        print("\nWelcome! I can help you find the perfect restaurant in Dubai.")
        print("Try queries like:")
        print("  - Find Italian restaurants in Downtown Dubai with outdoor seating")
        print("  - I want a romantic fine dining experience under AED 300")
        print("  - Show me budget-friendly Indian restaurants")
        print("\nType 'quit' or 'exit' to end, 'clear' to reset conversation.\n")
        
        while True:
            try:
                # Get user input
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit']:
                    print("\nThank you for using Restaurant Search! Goodbye!\n")
                    break
                
                if query.lower() == 'clear':
                    self.clear_history()
                    print("Conversation history cleared!")
                    continue
                
                # Process query
                print("\nSearching...\n")
                result = self.search(query)
                
                # Display response
                print(f"Assistant: {result['response']}\n")
                
                # Display metadata (if recommendation)
                if result['type'] == 'recommendation' and result.get('restaurants'):
                    print(f"Found {result['total_found']} matching restaurants")
                    print(f"Showing top {len(result['restaurants'])} recommendations\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!\n")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n[ERROR] Error: {e}\n")


def run_examples(app: RestaurantSearchApp):
    """Run example queries to demonstrate system capabilities."""
    print("\n" + "="*70)
    print("Running Example Queries")
    print("="*70 + "\n")
    
    examples = [
        "Find Italian restaurants in Downtown Dubai with outdoor seating under AED 200",
        "I want a romantic fine dining experience with great views",
        "Show me budget-friendly vegetarian options",
        "What are the best Japanese restaurants in Jumeirah?",
        "I need a place for a business lunch in DIFC",
    ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n{'-'*70}")
        print(f"Example {i}: {query}")
        print('-'*70)
        
        result = app.search(query)
        print(f"\n{result['response']}\n")
        
        if result.get('restaurants'):
            print(f"Matched {result['total_found']} restaurants")
        
        # Clear history between examples
        app.clear_history()
        
        # Small pause for readability
        import time
        time.sleep(1)


def main():
    """Main entry point."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    
    if not api_key:
        print("\n[ERROR] Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'\n")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Restaurant Search System")
    parser.add_argument(
        "--mode",
        choices=["interactive", "examples", "both"],
        default="interactive",
        help="Run mode (default: interactive)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from scratch"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (non-interactive)"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port for Prometheus metrics server (default: 8000)"
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics server"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize LangSmith tracing for observability
        langsmith_tracer = initialize_langsmith_tracing()
        if langsmith_tracer.is_enabled():
            logger.info("LangSmith observability enabled")
            logger.info(f"   Project: {langsmith_tracer.project_name}")
            logger.info(f"   Dashboard: https://smith.langchain.com/")
        else:
            logger.info("LangSmith tracing disabled (no API key configured)")
        
        # Initialize system info for Prometheus observability
        set_system_info(
            version='1.0.0',
            environment=os.getenv('ENVIRONMENT', 'development'),
            python_version=sys.version.split()[0],
            langsmith_enabled=str(langsmith_tracer.is_enabled()),
            langsmith_project=langsmith_tracer.project_name if langsmith_tracer.is_enabled() else 'N/A'
        )
        
        # Start metrics server (unless disabled)
        if not args.no_metrics:
            try:
                start_metrics_server(port=args.metrics_port)
            except Exception as e:
                logger.warning(f"Could not start metrics server: {e}")
        
        # Initialize application
        app = RestaurantSearchApp(api_key, rebuild_index=args.rebuild)
        
        # Single query mode
        if args.query:
            result = app.search(args.query)
            print(f"\n{result['response']}\n")
            return
        
        # Run modes
        if args.mode in ["examples", "both"]:
            run_examples(app)
        
        if args.mode in ["interactive", "both"]:
            app.interactive_mode()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n[ERROR] Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

