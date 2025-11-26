"""
Example client for the Restaurant Search API.
Demonstrates how to use the REST API endpoints.
"""
import requests
import json
from typing import Optional


class RestaurantSearchClient:
    """Client for Restaurant Search API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize the client."""
        self.base_url = base_url.rstrip('/')
        self.session_id: Optional[str] = None
    
    def search(self, query: str, top_k: int = 5, session_id: Optional[str] = None) -> dict:
        """
        Search for restaurants.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            session_id: Optional session ID for conversation context
            
        Returns:
            Search response dict
        """
        url = f"{self.base_url}/search"
        
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        if session_id or self.session_id:
            payload["session_id"] = session_id or self.session_id
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def list_restaurants(
        self, 
        skip: int = 0, 
        limit: int = 50,
        cuisine: Optional[str] = None,
        location: Optional[str] = None
    ) -> dict:
        """
        List all restaurants with optional filtering.
        
        Args:
            skip: Number to skip (pagination)
            limit: Number to return
            cuisine: Filter by cuisine
            location: Filter by location
            
        Returns:
            List of restaurants
        """
        url = f"{self.base_url}/restaurants"
        
        params = {"skip": skip, "limit": limit}
        if cuisine:
            params["cuisine"] = cuisine
        if location:
            params["location"] = location
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_restaurant(self, restaurant_id: int) -> dict:
        """
        Get a specific restaurant by ID.
        
        Args:
            restaurant_id: Restaurant ID
            
        Returns:
            Restaurant details
        """
        url = f"{self.base_url}/restaurants/{restaurant_id}"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def clear_session(self, session_id: Optional[str] = None):
        """Clear conversation history for a session."""
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID provided")
        
        url = f"{self.base_url}/sessions/{sid}"
        response = requests.delete(url)
        response.raise_for_status()
        
        return response.json()
    
    def health(self) -> dict:
        """Check API health."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()
    
    def stats(self) -> dict:
        """Get system statistics."""
        url = f"{self.base_url}/stats"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()


def main():
    """Demo the API client."""
    print("\n" + "="*70)
    print("🍽️  Restaurant Search API Client Demo")
    print("="*70 + "\n")
    
    # Initialize client
    client = RestaurantSearchClient("http://localhost:8080")
    
    # Check health
    print("1. Checking API health...")
    try:
        health = client.health()
        print(f"   ✅ API is {'healthy' if health['healthy'] else 'unhealthy'}")
        print(f"   📊 Total requests: {health['total_requests']}")
        print(f"   ✅ Success: {health['success_count']}")
        print(f"   ❌ Errors: {health['error_count']}\n")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API is not running: {e}")
        print("   💡 Start the API with: python src/api.py\n")
        return
    
    # Get stats
    print("2. Getting system stats...")
    stats = client.stats()
    print(f"   📚 Total restaurants: {stats['total_restaurants']}")
    print(f"   🗂️  Vector store docs: {stats['vector_store_documents']}")
    print(f"   👥 Active sessions: {stats['active_sessions']}\n")
    
    # Example 1: Basic search
    print("3. Example 1: Basic Search")
    print("   Query: 'Find Italian restaurants in Downtown Dubai'\n")
    
    result = client.search(
        query="Find Italian restaurants in Downtown Dubai",
        top_k=3,
        session_id="demo-session"
    )
    
    print(f"   🎯 Response: {result['response'][:100]}...")
    print(f"   📊 Found: {result['total_found']} restaurants")
    print(f"   ⚡ Processing time: {result['processing_time_ms']:.2f}ms")
    
    if result.get('extracted_entities'):
        entities = result['extracted_entities']
        print(f"   🏷️  Extracted entities:")
        if entities.get('cuisine'):
            print(f"      - Cuisine: {entities['cuisine']}")
        if entities.get('location'):
            print(f"      - Location: {entities['location']}")
    
    print(f"\n   🍽️  Top {len(result['restaurants'])} restaurants:")
    for i, restaurant in enumerate(result['restaurants'], 1):
        print(f"      {i}. {restaurant['name']} ({restaurant['rating']}⭐, {restaurant['price_range']})")
        if restaurant.get('relevance_score'):
            print(f"         Relevance: {restaurant['relevance_score']:.2f}")
    print()
    
    # Example 2: Follow-up query with session
    print("4. Example 2: Follow-up Query (using session context)")
    print("   Query: 'Which one has outdoor seating?'\n")
    
    client.session_id = "demo-session"
    result2 = client.search(
        query="Which one has outdoor seating?",
        top_k=3
    )
    
    print(f"   🎯 Response: {result2['response'][:100]}...")
    print(f"   📊 Found: {result2['total_found']} restaurants\n")
    
    # Example 3: List restaurants
    print("5. Example 3: List All Italian Restaurants")
    
    restaurants_list = client.list_restaurants(
        cuisine="Italian",
        limit=5
    )
    
    print(f"   📊 Total Italian restaurants: {restaurants_list['total']}")
    print(f"   🍽️  Showing {len(restaurants_list['restaurants'])} restaurants:")
    for i, r in enumerate(restaurants_list['restaurants'], 1):
        print(f"      {i}. {r['name']} - {r['location']} ({r['rating']}⭐)")
    print()
    
    # Example 4: Get specific restaurant
    if restaurants_list['restaurants']:
        print("6. Example 4: Get Restaurant Details")
        first_id = restaurants_list['restaurants'][0]['id']
        
        restaurant = client.get_restaurant(first_id)
        
        print(f"   🍽️  {restaurant['name']}")
        print(f"   🍝 Cuisine: {restaurant['cuisine']}")
        print(f"   📍 Location: {restaurant['location']}")
        print(f"   💰 Price: {restaurant['price_range']}")
        print(f"   ⭐ Rating: {restaurant['rating']} ({restaurant['review_count']} reviews)")
        print(f"   🕐 Hours: {restaurant['opening_hours']}")
        print(f"   🎁 Amenities: {restaurant['amenities']}")
        print(f"   📝 {restaurant['description'][:100]}...\n")
    
    # Example 5: Complex query
    print("7. Example 5: Complex Query")
    print("   Query: 'romantic dinner with great views under 200 AED'\n")
    
    result3 = client.search(
        query="romantic dinner with great views under 200 AED",
        top_k=3
    )
    
    print(f"   🎯 Response: {result3['response'][:150]}...")
    print(f"   📊 Found: {result3['total_found']} restaurants")
    
    if result3.get('extracted_entities'):
        entities = result3['extracted_entities']
        print(f"   🏷️  Extracted entities:")
        if entities.get('attributes'):
            print(f"      - Attributes: {', '.join(entities['attributes'])}")
        if entities.get('price_max'):
            print(f"      - Price max: AED {entities['price_max']}")
    print()
    
    # Clear session
    print("8. Clearing session...")
    clear_result = client.clear_session()
    print(f"   ✅ {clear_result['message']}\n")
    
    print("="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\nTry the interactive docs at: http://localhost:8080/docs\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

