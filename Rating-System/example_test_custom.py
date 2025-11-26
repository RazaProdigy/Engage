"""
Simple Example: Custom API Testing Script

This is a standalone example showing how to test the API programmatically.
Customize this script for your specific testing needs.

Usage:
    1. Start the API: uvicorn main:app --reload
    2. Run this script: python example_test_custom.py
"""

import requests
import json
from typing import Dict, List, Any


# Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy: {data}")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        print(f"  Start server with: uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error checking API health: {e}")
        return False


def predict_rating(review_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a single review for rating prediction
    
    Args:
        review_data: Dictionary containing review information
        
    Returns:
        Prediction response as dictionary
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=[review_data],  # API expects a list
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def predict_batch(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Send multiple reviews for batch prediction
    
    Args:
        reviews: List of review dictionaries
        
    Returns:
        Batch prediction response as dictionary
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=reviews,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def example_1_minimal():
    """Example 1: Minimal request with only required fields"""
    print("\n" + "=" * 70)
    print("Example 1: Minimal Request")
    print("=" * 70)
    
    review = {
        "review_text": "The food was absolutely amazing! Best meal I've had in months."
    }
    
    print(f"\nSending review: \"{review['review_text']}\"")
    result = predict_rating(review)
    
    if result:
        pred = result["predictions"][0]
        print(f"\n✓ Predicted Rating: {pred['predicted_rating']:.4f}")
        print(f"✓ Rounded Rating: {pred['predicted_rating_rounded']}")


def example_2_detailed():
    """Example 2: Detailed request with multiple fields"""
    print("\n" + "=" * 70)
    print("Example 2: Detailed Request")
    print("=" * 70)
    
    review = {
        "restaurant_id": 1,
        "user_id": 100,
        "review_text": "Outstanding service and delicious food. The ambiance was perfect "
                       "for a romantic dinner. Prices are reasonable for the quality.",
        "cuisine": "Italian",
        "location": "Dubai Marina",
        "price_range": "AED 150 - 250",
        "rating_rest": 4.6,
        "review_count": 320,
        "age": 28,
        "home_location": "Dubai",
        "dining_frequency": "weekly",
        "favorite_cuisines": "Italian, Mediterranean",
        "preferred_": "high",
        "dietary_res": "None",
        "avg_rating_total_reviews_written": 4.3,
        "popularity_score": 0.85,
        "avg_price": 200.0,
        "booking_lead_time_days": 3.0
    }
    
    print(f"\nSending detailed review for Restaurant #{review['restaurant_id']}")
    print(f"Review: \"{review['review_text'][:60]}...\"")
    result = predict_rating(review)
    
    if result:
        pred = result["predictions"][0]
        print(f"\n✓ Restaurant ID: {pred['restaurant_id']}")
        print(f"✓ User ID: {pred['user_id']}")
        print(f"✓ Predicted Rating: {pred['predicted_rating']:.4f}")
        print(f"✓ Rounded Rating: {pred['predicted_rating_rounded']}")


def example_3_batch():
    """Example 3: Batch prediction for multiple reviews"""
    print("\n" + "=" * 70)
    print("Example 3: Batch Prediction")
    print("=" * 70)
    
    reviews = [
        {
            "restaurant_id": 1,
            "review_text": "Absolutely fantastic! Will definitely come back!",
            "cuisine": "Italian",
            "rating_rest": 4.7
        },
        {
            "restaurant_id": 2,
            "review_text": "Decent food but service was very slow.",
            "cuisine": "Chinese",
            "rating_rest": 3.2
        },
        {
            "restaurant_id": 3,
            "review_text": "Terrible experience. Food was cold and tasteless.",
            "cuisine": "Mexican",
            "rating_rest": 2.1
        },
        {
            "restaurant_id": 4,
            "review_text": "Pretty good, nothing exceptional but satisfying.",
            "cuisine": "Japanese",
            "rating_rest": 3.8
        }
    ]
    
    print(f"\nSending {len(reviews)} reviews for batch prediction...")
    result = predict_batch(reviews)
    
    if result:
        print(f"\n{'#':<3} {'Restaurant ID':<15} {'Predicted':<12} {'Rounded':<10}")
        print("-" * 70)
        for i, pred in enumerate(result["predictions"], 1):
            print(f"{i:<3} {pred['restaurant_id']:<15} "
                  f"{pred['predicted_rating']:<12.4f} "
                  f"{pred['predicted_rating_rounded']:<10}")


def example_4_sentiment_comparison():
    """Example 4: Compare predictions for different sentiment levels"""
    print("\n" + "=" * 70)
    print("Example 4: Sentiment Comparison")
    print("=" * 70)
    
    reviews = [
        {"review_text": "Perfect! Amazing! Best ever!", "sentiment": "Very Positive"},
        {"review_text": "Good food, nice service.", "sentiment": "Positive"},
        {"review_text": "It was okay, nothing special.", "sentiment": "Neutral"},
        {"review_text": "Not great, disappointed.", "sentiment": "Negative"},
        {"review_text": "Terrible! Worst experience!", "sentiment": "Very Negative"}
    ]
    
    print("\nComparing predictions for different sentiments...\n")
    print(f"{'Sentiment':<20} {'Review Text':<40} {'Predicted':<10}")
    print("-" * 70)
    
    for review in reviews:
        result = predict_rating({"review_text": review["review_text"]})
        if result:
            pred = result["predictions"][0]
            print(f"{review['sentiment']:<20} "
                  f"{review['review_text'][:38]:<40} "
                  f"{pred['predicted_rating_rounded']:<10.2f}")


def example_5_error_handling():
    """Example 5: Handling errors and invalid requests"""
    print("\n" + "=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)
    
    # Test 1: Missing required field
    print("\nTest 1: Missing required field (review_text)")
    invalid_review = {"cuisine": "Italian"}
    result = predict_rating(invalid_review)
    if result is None:
        print("✓ Correctly handled invalid request")
    
    # Test 2: Invalid data type
    print("\nTest 2: Invalid data type (age as string)")
    invalid_review = {
        "review_text": "Good food",
        "age": "thirty"  # Should be int
    }
    result = predict_rating(invalid_review)
    if result is None:
        print("✓ Correctly handled invalid data type")
    
    # Test 3: Valid request after errors
    print("\nTest 3: Valid request after handling errors")
    valid_review = {"review_text": "Everything was perfect!"}
    result = predict_rating(valid_review)
    if result:
        print(f"✓ Successfully recovered and processed valid request")
        print(f"  Predicted rating: {result['predictions'][0]['predicted_rating_rounded']}")


def main():
    """Main function to run all examples"""
    print("\n" + "=" * 70)
    print("  Custom API Testing Examples")
    print("  Restaurant Rating Prediction API")
    print("=" * 70)
    
    # Check if API is running
    if not check_api_health():
        print("\n⚠️  Please start the API server first:")
        print("    uvicorn main:app --reload")
        return
    
    # Run examples
    try:
        example_1_minimal()
        example_2_detailed()
        example_3_batch()
        example_4_sentiment_comparison()
        example_5_error_handling()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        print("\nCustomization Tips:")
        print("  - Modify the review_data dictionaries to test your scenarios")
        print("  - Add more fields based on your requirements")
        print("  - Create custom functions for your specific use cases")
        print("  - Handle responses according to your application needs")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")


if __name__ == "__main__":
    main()


# ==============================================================================
# Additional Helper Functions (for your custom use)
# ==============================================================================

def create_review_from_template(
    text: str,
    cuisine: str = None,
    rating_rest: float = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Helper function to create a review dictionary with common fields
    
    Args:
        text: Review text (required)
        cuisine: Restaurant cuisine type
        rating_rest: Restaurant's aggregate rating
        **kwargs: Any additional fields
        
    Returns:
        Review dictionary ready for API submission
    """
    review = {"review_text": text}
    
    if cuisine:
        review["cuisine"] = cuisine
    if rating_rest:
        review["rating_rest"] = rating_rest
    
    review.update(kwargs)
    return review


def compare_predictions(reviews: List[Dict[str, Any]]) -> None:
    """
    Compare predictions for multiple reviews side by side
    
    Args:
        reviews: List of review dictionaries
    """
    result = predict_batch(reviews)
    
    if result:
        print(f"\n{'Review':<50} {'Predicted Rating':<20}")
        print("-" * 70)
        for review, pred in zip(reviews, result["predictions"]):
            text = review['review_text'][:47] + "..."
            rating = pred['predicted_rating_rounded']
            print(f"{text:<50} {rating:<20.2f}")


def test_with_variations(base_review: Dict[str, Any], variations: Dict[str, List]) -> None:
    """
    Test how different field values affect predictions
    
    Args:
        base_review: Base review dictionary
        variations: Dictionary of field names to lists of values to test
    """
    for field, values in variations.items():
        print(f"\nTesting variations in '{field}':")
        for value in values:
            test_review = base_review.copy()
            test_review[field] = value
            result = predict_rating(test_review)
            if result:
                pred = result["predictions"][0]
                print(f"  {field}={value}: {pred['predicted_rating_rounded']:.2f}")

