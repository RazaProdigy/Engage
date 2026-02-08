"""
Unit tests for RAG system components.
"""
import pytest
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RestaurantRAGSystem
from src.config import DATA_CONFIG


class TestRestaurantRAGSystem:
    """Test suite for RAG system."""
    
    @pytest.fixture
    def sample_restaurant_data(self, tmp_path):
        """Create sample restaurant data for testing."""
        data = [
            {
                "id": 1,
                "name": "Test Italian Restaurant",
                "cuisine": "Italian",
                "location": "Downtown Dubai",
                "price_range": "AED 100 - 150",
                "description": "Authentic Italian cuisine with pasta and pizza",
                "amenities": "Outdoor Seating, Parking",
                "attributes": "Family-Friendly, Casual Dining",
                "opening_hours": "11:00 - 23:00",
                "coordinates": "25.197197, 55.274376",
                "rating": 4.5,
                "review_count": 200
            },
            {
                "id": 2,
                "name": "Test Chinese Restaurant",
                "cuisine": "Chinese",
                "location": "Marina",
                "price_range": "AED 80 - 120",
                "description": "Traditional Chinese dishes and dim sum",
                "amenities": "Takeaway, Delivery",
                "attributes": "Casual Dining",
                "opening_hours": "10:00 - 22:00",
                "coordinates": "25.077481, 55.139389",
                "rating": 4.2,
                "review_count": 150
            }
        ]
        
        # Write to temporary file
        data_file = tmp_path / "test_restaurants.json"
        with open(data_file, 'w') as f:
            json.dump(data, f)
        
        return str(data_file)
    
    def test_load_restaurant_data(self, sample_restaurant_data):
        """Test loading restaurant data from JSON."""
        # Note: This test requires an API key to initialize embeddings
        # In a real test suite, we'd mock the embeddings
        
        # Skip if no API key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        restaurants = rag_system.load_restaurant_data(sample_restaurant_data)
        
        assert len(restaurants) == 2
        assert restaurants[0]["name"] == "Test Italian Restaurant"
        assert restaurants[1]["cuisine"] == "Chinese"
    
    def test_create_documents(self, sample_restaurant_data):
        """Test document creation from restaurant data."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        rag_system.load_restaurant_data(sample_restaurant_data)
        documents = rag_system.create_documents()
        
        assert len(documents) == 2
        assert "Test Italian Restaurant" in documents[0].page_content
        assert documents[0].metadata["cuisine"] == "Italian"
        assert documents[0].metadata["rating"] == 4.5
    
    def test_create_restaurant_text(self, sample_restaurant_data):
        """Test rich text creation for embeddings."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        rag_system.load_restaurant_data(sample_restaurant_data)
        
        restaurant = rag_system.restaurants_data[0]
        text = rag_system._create_restaurant_text(restaurant)
        
        # Check all important fields are included
        assert "Test Italian Restaurant" in text
        assert "Italian" in text
        assert "Downtown Dubai" in text
        assert "AED 100 - 150" in text
        assert "Authentic Italian cuisine" in text
        assert "Outdoor Seating" in text
        assert "4.5" in text
    
    def test_location_matches(self, sample_restaurant_data):
        """Test location matching with fuzzy logic."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        
        # Exact match
        assert rag_system._location_matches("Downtown Dubai", "Downtown Dubai")
        
        # Partial match
        assert rag_system._location_matches("Downtown Dubai", "Downtown")
        
        # Case insensitive
        assert rag_system._location_matches("Downtown Dubai", "downtown")
        
        # No match
        assert not rag_system._location_matches("Marina", "Downtown")
    
    def test_calculate_price_match(self, sample_restaurant_data):
        """Test price matching with flexibility."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        
        # Perfect overlap
        score = rag_system._calculate_price_match("AED 100 - 150", 100, 150)
        assert score >= 0.9
        
        # Partial overlap
        score = rag_system._calculate_price_match("AED 100 - 150", 80, 120)
        assert 0.7 <= score <= 1.0
        
        # No overlap but within flex
        score = rag_system._calculate_price_match("AED 100 - 150", 150, 200)
        assert score >= 0.5
    
    def test_check_amenities(self, sample_restaurant_data):
        """Test amenity checking."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        rag_system = RestaurantRAGSystem(os.getenv("OPENAI_API_KEY"))
        
        # Has amenity
        assert rag_system._check_amenities(
            "Outdoor Seating, Parking, Bar",
            ["outdoor seating"]
        )
        
        # Multiple amenities, one matches
        assert rag_system._check_amenities(
            "Outdoor Seating, Parking",
            ["parking", "wifi"]
        )
        
        # No match
        assert not rag_system._check_amenities(
            "Outdoor Seating",
            ["wifi", "valet"]
        )


class TestConfigurationLoading:
    """Test configuration and setup."""
    
    def test_data_path_exists(self):
        """Test that default data path is configured."""
        from src.config import DATA_CONFIG
        assert "restaurant_data_path" in DATA_CONFIG
    
    def test_system_prompts_exist(self):
        """Test that system prompts are configured."""
        from src.config import SYSTEM_PROMPTS
        assert "query_understanding" in SYSTEM_PROMPTS
        assert "retrieval" in SYSTEM_PROMPTS
        assert "response_generation" in SYSTEM_PROMPTS
    
    def test_price_ranges_configured(self):
        """Test price range mappings."""
        from src.config import PRICE_RANGES
        assert "budget" in PRICE_RANGES
        assert "luxury" in PRICE_RANGES
        assert isinstance(PRICE_RANGES["budget"], tuple)


class TestEntityProcessing:
    """Test entity extraction and processing."""
    
    def test_price_range_extraction(self):
        """Test semantic price range extraction."""
        from src.config import PRICE_RANGES
        
        # Test that ranges are properly defined
        assert PRICE_RANGES["budget"][1] < PRICE_RANGES["luxury"][0]
        assert PRICE_RANGES["moderate"][0] < PRICE_RANGES["moderate"][1]


# Integration test example
class TestIntegration:
    """Integration tests for the full system."""
    
    @pytest.mark.integration
    def test_full_pipeline(self, sample_restaurant_data, tmp_path):
        """Test complete RAG pipeline."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # This would test the complete flow:
        # 1. Load data
        # 2. Create embeddings
        # 3. Build vector store
        # 4. Search
        # 5. Retrieve results
        
        # Note: Full implementation would go here
        # Skipped for brevity and to avoid long-running tests
        pass


if __name__ == "__main__":
    # Run tests with: pytest tests/test_rag_system.py -v
    pytest.main([__file__, "-v"])

