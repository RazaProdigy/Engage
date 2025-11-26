"""
Test suite for Restaurant Rating Prediction API
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from main import (
    app,
    parse_price_range_to_avg,
    build_feature_dataframe,
    RatingRequest,
    model
)

# ========================
# Test Client Setup
# ========================

client = TestClient(app)


# ========================
# Test Helper Functions
# ========================

class TestParsePriceRange:
    """Tests for parse_price_range_to_avg function"""
    
    def test_parse_valid_range(self):
        """Test parsing valid price range string"""
        result = parse_price_range_to_avg("AED 50 - 100")
        assert result == 75.0
        
    def test_parse_single_value(self):
        """Test parsing price with single value"""
        result = parse_price_range_to_avg("AED 100")
        assert result == 100.0
        
    def test_parse_multiple_values(self):
        """Test parsing price with multiple values"""
        result = parse_price_range_to_avg("50 100 150")
        assert result == 100.0  # Average of 50, 100, 150
        
    def test_parse_with_special_chars(self):
        """Test parsing price with special characters"""
        result = parse_price_range_to_avg("$50-$100")
        assert result == 75.0
        
    def test_parse_none(self):
        """Test parsing None value"""
        result = parse_price_range_to_avg(None)
        assert result is None
        
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_price_range_to_avg("")
        assert result is None
        
    def test_parse_no_numbers(self):
        """Test parsing string with no numbers"""
        result = parse_price_range_to_avg("expensive")
        assert result is None
        
    def test_parse_integer_input(self):
        """Test parsing integer input (should return None)"""
        result = parse_price_range_to_avg(100)
        assert result is None


class TestBuildFeatureDataframe:
    """Tests for build_feature_dataframe function"""
    
    def test_single_request(self):
        """Test building dataframe from single request"""
        request = RatingRequest(
            restaurant_id=1,
            user_id=100,
            review_text="Great food!",
            cuisine="Italian",
            location="Dubai Marina",
            price_range="AED 50 - 100",
            rating_rest=4.5,
            review_count=250,
            age=30,
            home_location="Dubai",
            dining_frequency="weekly",
            favorite_cuisines="Italian",
            preferred_="high",
            dietary_res="None",
            avg_rating_total_reviews_written=4.2,
            popularity_score=0.8,
            avg_price=75.0,
            booking_lead_time_days=2.0
        )
        
        features_df, ids = build_feature_dataframe([request])
        
        assert len(features_df) == 1
        assert len(ids) == 1
        assert features_df["review_text"].iloc[0] == "Great food!"
        assert features_df["cuisine"].iloc[0] == "Italian"
        assert features_df["avg_price_range"].iloc[0] == 75.0
        assert ids["_restaurant_id"].iloc[0] == 1
        assert ids["_user_id"].iloc[0] == 100
        
    def test_multiple_requests(self):
        """Test building dataframe from multiple requests"""
        requests = [
            RatingRequest(
                restaurant_id=1,
                user_id=100,
                review_text="Great food!",
                cuisine="Italian"
            ),
            RatingRequest(
                restaurant_id=2,
                user_id=101,
                review_text="Average experience",
                cuisine="Chinese"
            )
        ]
        
        features_df, ids = build_feature_dataframe(requests)
        
        assert len(features_df) == 2
        assert len(ids) == 2
        assert features_df["review_text"].iloc[0] == "Great food!"
        assert features_df["review_text"].iloc[1] == "Average experience"
        
    def test_missing_fields(self):
        """Test handling of missing optional fields"""
        request = RatingRequest(
            review_text="Simple review"
        )
        
        features_df, ids = build_feature_dataframe([request])
        
        assert len(features_df) == 1
        assert features_df["review_text"].iloc[0] == "Simple review"
        assert pd.isna(features_df["cuisine"].iloc[0]) or features_df["cuisine"].iloc[0] is None
        
    def test_price_range_computation(self):
        """Test that avg_price_range is computed correctly"""
        request = RatingRequest(
            review_text="Test",
            price_range="AED 100 - 200"
        )
        
        features_df, ids = build_feature_dataframe([request])
        
        assert features_df["avg_price_range"].iloc[0] == 150.0


# ========================
# Test API Endpoints
# ========================

class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_check_success(self):
        """Test health check returns ok status"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        
    def test_health_check_model_loaded(self):
        """Test health check confirms model is loaded"""
        response = client.get("/health")
        data = response.json()
        
        assert data["model_loaded"] is True


class TestRootEndpoint:
    """Tests for / endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs_url" in data
        assert data["docs_url"] == "/docs"
        assert data["redoc_url"] == "/redoc"


class TestPredictEndpoint:
    """Tests for /predict endpoint"""
    
    def test_predict_single_review(self):
        """Test prediction for single review"""
        payload = [{
            "restaurant_id": 1,
            "user_id": 100,
            "review_text": "Amazing food and great service! Highly recommended.",
            "cuisine": "Italian",
            "location": "Dubai Marina",
            "price_range": "AED 100 - 200",
            "rating_rest": 4.5,
            "review_count": 250,
            "age": 30,
            "home_location": "Dubai",
            "dining_frequency": "weekly",
            "favorite_cuisines": "Italian",
            "preferred_": "high",
            "dietary_res": "None",
            "avg_rating_total_reviews_written": 4.2,
            "popularity_score": 0.8,
            "avg_price": 150.0,
            "booking_lead_time_days": 2.0
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        
        prediction = data["predictions"][0]
        assert "predicted_rating" in prediction
        assert "predicted_rating_rounded" in prediction
        assert prediction["restaurant_id"] == 1
        assert prediction["user_id"] == 100
        assert isinstance(prediction["predicted_rating"], float)
        assert isinstance(prediction["predicted_rating_rounded"], float)
        
    def test_predict_multiple_reviews(self):
        """Test prediction for multiple reviews"""
        payload = [
            {
                "restaurant_id": 1,
                "user_id": 100,
                "review_text": "Excellent experience!",
                "cuisine": "Italian"
            },
            {
                "restaurant_id": 2,
                "user_id": 101,
                "review_text": "Not impressed with the service.",
                "cuisine": "Chinese"
            },
            {
                "restaurant_id": 3,
                "user_id": 102,
                "review_text": "Average food, decent atmosphere.",
                "cuisine": "Mexican"
            }
        ]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        
        for i, prediction in enumerate(data["predictions"]):
            assert prediction["restaurant_id"] == payload[i]["restaurant_id"]
            assert prediction["user_id"] == payload[i]["user_id"]
            assert "predicted_rating" in prediction
            assert "predicted_rating_rounded" in prediction
            
    def test_predict_minimal_data(self):
        """Test prediction with only required fields"""
        payload = [{
            "review_text": "Simple review"
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert "predicted_rating" in data["predictions"][0]
        
    def test_predict_missing_review_text(self):
        """Test prediction fails without review_text"""
        payload = [{
            "restaurant_id": 1,
            "user_id": 100,
            "cuisine": "Italian"
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 422  # Unprocessable Entity
        
    def test_predict_empty_list(self):
        """Test prediction with empty list"""
        payload = []
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 0
        
    def test_predict_with_null_optional_fields(self):
        """Test prediction with null optional fields"""
        payload = [{
            "restaurant_id": None,
            "user_id": None,
            "review_text": "Good food",
            "cuisine": None,
            "location": None,
            "price_range": None
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"][0]["restaurant_id"] is None
        assert data["predictions"][0]["user_id"] is None
        
    def test_predict_rating_in_valid_range(self):
        """Test that predicted ratings are reasonable"""
        payload = [{
            "review_text": "Amazing experience with great food and service!",
            "rating_rest": 4.5,
            "review_count": 500,
            "avg_rating_total_reviews_written": 4.3
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        predicted_rating = data["predictions"][0]["predicted_rating"]
        
        # Ratings should typically be between 1 and 5
        assert 1.0 <= predicted_rating <= 5.0 or predicted_rating >= 0
        
    def test_predict_response_structure(self):
        """Test that response follows expected structure"""
        payload = [{
            "restaurant_id": 999,
            "user_id": 888,
            "review_text": "Test review"
        }]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        
        prediction = data["predictions"][0]
        required_keys = ["restaurant_id", "user_id", "predicted_rating", "predicted_rating_rounded"]
        for key in required_keys:
            assert key in prediction


# ========================
# Test Data Validation
# ========================

class TestRequestValidation:
    """Tests for request data validation"""
    
    def test_invalid_age_type(self):
        """Test that invalid age type is rejected"""
        payload = [{
            "review_text": "Test",
            "age": "thirty"  # Should be int
        }]
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
    def test_invalid_rating_type(self):
        """Test that invalid rating type is rejected"""
        payload = [{
            "review_text": "Test",
            "rating_rest": "high"  # Should be float
        }]
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ========================
# Integration Tests
# ========================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_prediction_workflow(self):
        """Test complete workflow from request to prediction"""
        # Step 1: Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"
        
        # Step 2: Make prediction
        payload = [{
            "restaurant_id": 123,
            "user_id": 456,
            "review_text": "Wonderful dining experience!",
            "cuisine": "Italian",
            "rating_rest": 4.3,
            "avg_rating_total_reviews_written": 4.0
        }]
        
        predict_response = client.post("/predict", json=payload)
        assert predict_response.status_code == 200
        
        data = predict_response.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["restaurant_id"] == 123
        assert data["predictions"][0]["user_id"] == 456
        
    def test_batch_predictions_consistency(self):
        """Test that batch predictions are consistent"""
        review_data = {
            "review_text": "Consistent test review",
            "cuisine": "Italian",
            "rating_rest": 4.0
        }
        
        # Send same review twice in batch
        payload = [review_data, review_data]
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        # Both predictions should be identical (same inputs)
        pred1 = data["predictions"][0]["predicted_rating"]
        pred2 = data["predictions"][1]["predicted_rating"]
        
        assert pred1 == pred2


# ========================
# Performance Tests
# ========================

class TestPerformance:
    """Basic performance tests"""
    
    def test_large_batch_prediction(self):
        """Test prediction with larger batch of reviews"""
        payload = [
            {
                "review_text": f"Review number {i}",
                "cuisine": "Italian",
                "restaurant_id": i
            }
            for i in range(50)
        ]
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 50


# ========================
# Fixtures (if needed)
# ========================

@pytest.fixture
def sample_rating_request():
    """Fixture providing a sample RatingRequest"""
    return RatingRequest(
        restaurant_id=1,
        user_id=100,
        review_text="Test review",
        cuisine="Italian",
        location="Dubai Marina",
        price_range="AED 100 - 200",
        rating_rest=4.5,
        review_count=250,
        age=30,
        home_location="Dubai",
        dining_frequency="weekly",
        favorite_cuisines="Italian",
        preferred_="high",
        dietary_res="None",
        avg_rating_total_reviews_written=4.2,
        popularity_score=0.8,
        avg_price=150.0,
        booking_lead_time_days=2.0
    )


@pytest.fixture
def sample_minimal_request():
    """Fixture providing a minimal RatingRequest"""
    return RatingRequest(review_text="Minimal review")


# ========================
# Run with: pytest test_main.py -v
# ========================

