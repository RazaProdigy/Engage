"""
Manual API Testing Script
Run this script while the FastAPI server is running to test the API manually.

Usage:
    1. Start the server: uvicorn main:app --reload
    2. Run this script: python test_api_manual.py
"""

import requests
import json
from typing import List, Dict, Any


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_response(response: requests.Response):
    """Pretty print the response"""
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_health_check():
    """Test the health check endpoint"""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint"""
    print_section("2. Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_prediction_minimal():
    """Test prediction with minimal data"""
    print_section("3. Single Prediction - Minimal Data")
    
    payload = [{
        "review_text": "Great food and excellent service!"
    }]
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_prediction_full():
    """Test prediction with full data"""
    print_section("4. Single Prediction - Full Data")
    
    payload = [{
        "restaurant_id": 1,
        "user_id": 100,
        "review_text": "Outstanding dining experience! The pasta was perfectly cooked, "
                       "the wine selection was excellent, and the service was top-notch. "
                       "The ambiance was romantic and the prices were reasonable. "
                       "Will definitely come back!",
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
    }]
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction with multiple reviews"""
    print_section("5. Batch Prediction - Multiple Reviews")
    
    payload = [
        {
            "restaurant_id": 1,
            "user_id": 100,
            "review_text": "Absolutely fantastic! Best Italian food I've had in years. "
                          "The atmosphere was cozy and the staff was very attentive.",
            "cuisine": "Italian",
            "location": "Dubai Marina",
            "rating_rest": 4.7,
            "review_count": 450,
            "age": 35,
            "avg_rating_total_reviews_written": 4.5
        },
        {
            "restaurant_id": 2,
            "user_id": 101,
            "review_text": "Decent food but nothing special. Service was slow and "
                          "the place was too crowded. Probably won't return.",
            "cuisine": "Chinese",
            "location": "Downtown Dubai",
            "rating_rest": 3.2,
            "review_count": 180,
            "age": 42,
            "avg_rating_total_reviews_written": 3.8
        },
        {
            "restaurant_id": 3,
            "user_id": 102,
            "review_text": "Terrible experience. Food was cold, service was rude, "
                          "and the place was dirty. Would not recommend to anyone.",
            "cuisine": "Mexican",
            "location": "JBR",
            "rating_rest": 2.1,
            "review_count": 95,
            "age": 29,
            "avg_rating_total_reviews_written": 3.5
        },
        {
            "restaurant_id": 4,
            "user_id": 103,
            "review_text": "Good value for money. The portions were generous and "
                          "the taste was authentic. Will come again with family.",
            "cuisine": "Indian",
            "location": "Bur Dubai",
            "rating_rest": 4.1,
            "review_count": 230,
            "age": 31,
            "avg_rating_total_reviews_written": 4.0
        },
        {
            "restaurant_id": 5,
            "user_id": 104,
            "review_text": "Average experience. Nothing to complain about but "
                          "nothing extraordinary either. Fair prices.",
            "cuisine": "Lebanese",
            "location": "Deira",
            "rating_rest": 3.5,
            "review_count": 310,
            "age": 27,
            "avg_rating_total_reviews_written": 3.7
        }
    ]
    
    print(f"Request Payload: {len(payload)} reviews")
    print(f"Review texts: {[r['review_text'][:50] + '...' for r in payload]}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print_response(response)
        
        if response.status_code == 200:
            results = response.json()
            print("\n" + "-" * 70)
            print("Summary of Predictions:")
            print("-" * 70)
            for i, pred in enumerate(results["predictions"], 1):
                print(f"Review {i}:")
                print(f"  Restaurant ID: {pred['restaurant_id']}")
                print(f"  User ID: {pred['user_id']}")
                print(f"  Predicted Rating: {pred['predicted_rating']:.4f}")
                print(f"  Rounded Rating: {pred['predicted_rating_rounded']}")
                print()
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_different_sentiments():
    """Test predictions for reviews with different sentiments"""
    print_section("6. Testing Different Review Sentiments")
    
    payload = [
        {
            "restaurant_id": 10,
            "review_text": "Amazing! Perfect! Excellent! Best ever! Highly recommended!",
            "cuisine": "Italian",
            "rating_rest": 4.5
        },
        {
            "restaurant_id": 11,
            "review_text": "Good food, nice place, decent service.",
            "cuisine": "Italian",
            "rating_rest": 4.0
        },
        {
            "restaurant_id": 12,
            "review_text": "It was okay. Nothing special. Average.",
            "cuisine": "Italian",
            "rating_rest": 3.5
        },
        {
            "restaurant_id": 13,
            "review_text": "Not good. Disappointing. Could be better.",
            "cuisine": "Italian",
            "rating_rest": 3.0
        },
        {
            "restaurant_id": 14,
            "review_text": "Terrible! Awful! Worst experience ever! Never again!",
            "cuisine": "Italian",
            "rating_rest": 2.0
        }
    ]
    
    print("Testing reviews with different sentiments (same cuisine):")
    for r in payload:
        print(f"  - \"{r['review_text'][:50]}...\"")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            results = response.json()
            print("\nPrediction Results:")
            print("-" * 70)
            for i, (req, pred) in enumerate(zip(payload, results["predictions"]), 1):
                sentiment = req['review_text'][:40] + "..."
                print(f"{i}. {sentiment}")
                print(f"   Predicted: {pred['predicted_rating_rounded']:.2f} | "
                      f"Restaurant Rating: {req['rating_rest']:.1f}")
        else:
            print_response(response)
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid data"""
    print_section("7. Error Handling - Invalid Data")
    
    # Test 1: Missing required field
    print("\nTest 7.1: Missing review_text (required field)")
    payload = [{
        "restaurant_id": 1,
        "cuisine": "Italian"
    }]
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Expected: 422 (Validation Error)")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Invalid data type
    print("\nTest 7.2: Invalid data type (age as string)")
    payload = [{
        "review_text": "Good food",
        "age": "thirty"  # Should be int
    }]
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Expected: 422 (Validation Error)")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def run_all_tests():
    """Run all tests and provide summary"""
    print("\n" + "=" * 70)
    print("  RESTAURANT RATING PREDICTION API - MANUAL TEST SUITE")
    print("=" * 70)
    print(f"\nAPI Base URL: {BASE_URL}")
    print("Make sure the server is running: uvicorn main:app --reload\n")
    
    results = {}
    
    # Run tests
    results["Health Check"] = test_health_check()
    results["Root Endpoint"] = test_root_endpoint()
    results["Single Prediction (Minimal)"] = test_single_prediction_minimal()
    results["Single Prediction (Full)"] = test_single_prediction_full()
    results["Batch Prediction"] = test_batch_prediction()
    results["Different Sentiments"] = test_different_sentiments()
    
    # Error handling doesn't return bool
    test_error_handling()
    
    # Print summary
    print_section("TEST SUMMARY")
    print(f"\n{'Test Name':<35} {'Status':<10}")
    print("-" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<35} {status:<10}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")

