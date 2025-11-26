# API Testing Guide - Restaurant Rating Prediction API

This guide provides examples of how to send requests to the Restaurant Rating Prediction API using various tools and methods.

## Table of Contents
1. [Running the Tests](#running-the-tests)
2. [API Endpoints](#api-endpoints)
3. [Testing with cURL](#testing-with-curl)
4. [Testing with Python Requests](#testing-with-python-requests)
5. [Testing with Postman](#testing-with-postman)
6. [Testing with HTTPie](#testing-with-httpie)
7. [Example Payloads](#example-payloads)

---

## Running the Tests

### Install Dependencies

```bash
pip install pytest pytest-cov httpx
```

### Run All Tests

```bash
# Run all tests
pytest test_main.py -v

# Run with coverage report
pytest test_main.py -v --cov=main --cov-report=html

# Run specific test class
pytest test_main.py::TestPredictEndpoint -v

# Run specific test
pytest test_main.py::TestPredictEndpoint::test_predict_single_review -v

# Run tests matching a keyword
pytest test_main.py -k "predict" -v
```

---

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Check if the API and model are loaded correctly

### 2. Root
- **Endpoint**: `GET /`
- **Description**: Get API information and documentation links

### 3. Predict Ratings
- **Endpoint**: `POST /predict`
- **Description**: Predict ratings for one or more restaurant reviews

---

## Testing with cURL

### 1. Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. Root Endpoint

```bash
curl -X GET http://localhost:8000/
```

**Expected Response:**
```json
{
  "message": "Restaurant Rating Prediction API",
  "docs_url": "/docs",
  "redoc_url": "/redoc"
}
```

### 3. Single Review Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "restaurant_id": 1,
      "user_id": 100,
      "review_text": "Amazing food and excellent service! The ambiance was perfect.",
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
    }
  ]'
```

### 4. Multiple Review Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "restaurant_id": 1,
      "user_id": 100,
      "review_text": "Great experience!",
      "cuisine": "Italian"
    },
    {
      "restaurant_id": 2,
      "user_id": 101,
      "review_text": "Disappointing meal.",
      "cuisine": "Chinese"
    }
  ]'
```

### 5. Minimal Request (Only Required Fields)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "review_text": "Simple but tasty food."
    }
  ]'
```

---

## Testing with Python Requests

### Setup

```python
import requests
import json

BASE_URL = "http://localhost:8000"
```

### 1. Health Check

```python
response = requests.get(f"{BASE_URL}/health")
print(response.json())
```

### 2. Single Prediction

```python
payload = [{
    "restaurant_id": 1,
    "user_id": 100,
    "review_text": "Outstanding food! Best Italian restaurant in Dubai.",
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

response = requests.post(f"{BASE_URL}/predict", json=payload)
print(json.dumps(response.json(), indent=2))
```

### 3. Batch Predictions

```python
payload = [
    {
        "restaurant_id": 1,
        "user_id": 100,
        "review_text": "Excellent food, great service!",
        "cuisine": "Italian",
        "rating_rest": 4.5
    },
    {
        "restaurant_id": 2,
        "user_id": 101,
        "review_text": "Average experience, nothing special.",
        "cuisine": "Chinese",
        "rating_rest": 3.2
    },
    {
        "restaurant_id": 3,
        "user_id": 102,
        "review_text": "Terrible service, cold food.",
        "cuisine": "Mexican",
        "rating_rest": 2.8
    }
]

response = requests.post(f"{BASE_URL}/predict", json=payload)
results = response.json()

for i, pred in enumerate(results["predictions"]):
    print(f"\nReview {i+1}:")
    print(f"  Restaurant ID: {pred['restaurant_id']}")
    print(f"  Predicted Rating: {pred['predicted_rating']:.2f}")
    print(f"  Rounded Rating: {pred['predicted_rating_rounded']}")
```

### 4. Error Handling

```python
def predict_rating(review_data):
    try:
        response = requests.post(f"{BASE_URL}/predict", json=[review_data])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
    return None

# Test with valid data
result = predict_rating({
    "review_text": "Great food!",
    "cuisine": "Italian"
})
print(result)

# Test with invalid data (missing review_text)
result = predict_rating({
    "cuisine": "Italian"
})
```

### 5. Complete Testing Script

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("=" * 50)
    print("Testing Restaurant Rating Prediction API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 3: Single Prediction
    print("\n3. Testing Single Prediction...")
    payload = [{
        "restaurant_id": 1,
        "user_id": 100,
        "review_text": "Amazing experience! The food was delicious and service was impeccable.",
        "cuisine": "Italian",
        "rating_rest": 4.5,
        "review_count": 200
    }]
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 4: Batch Prediction
    print("\n4. Testing Batch Prediction...")
    payload = [
        {"review_text": "Excellent!", "cuisine": "Italian"},
        {"review_text": "Not bad", "cuisine": "Chinese"},
        {"review_text": "Disappointing", "cuisine": "Mexican"}
    ]
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    results = response.json()
    for i, pred in enumerate(results["predictions"]):
        print(f"  Review {i+1} - Predicted Rating: {pred['predicted_rating']:.2f}")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_api()
```

---

## Testing with Postman

### Setup Collection

1. Create a new collection named "Restaurant Rating API"
2. Set base URL as variable: `{{base_url}}` = `http://localhost:8000`

### Request 1: Health Check

- **Method**: GET
- **URL**: `{{base_url}}/health`
- **Test Script**:
```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Model is loaded", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.model_loaded).to.eql(true);
});
```

### Request 2: Predict Rating

- **Method**: POST
- **URL**: `{{base_url}}/predict`
- **Headers**: 
  - `Content-Type: application/json`
- **Body** (raw JSON):
```json
[
  {
    "restaurant_id": 1,
    "user_id": 100,
    "review_text": "Amazing food and excellent service!",
    "cuisine": "Italian",
    "location": "Dubai Marina",
    "rating_rest": 4.5,
    "review_count": 250,
    "age": 30
  }
]
```
- **Test Script**:
```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has predictions", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.predictions).to.be.an('array');
    pm.expect(jsonData.predictions.length).to.be.above(0);
});

pm.test("Prediction has required fields", function () {
    var jsonData = pm.response.json();
    var pred = jsonData.predictions[0];
    pm.expect(pred).to.have.property('predicted_rating');
    pm.expect(pred).to.have.property('predicted_rating_rounded');
    pm.expect(pred).to.have.property('restaurant_id');
    pm.expect(pred).to.have.property('user_id');
});
```

---

## Testing with HTTPie

HTTPie provides a simpler syntax for HTTP requests.

### Install HTTPie
```bash
pip install httpie
```

### 1. Health Check
```bash
http GET http://localhost:8000/health
```

### 2. Single Prediction
```bash
http POST http://localhost:8000/predict \
  restaurant_id:=1 \
  user_id:=100 \
  review_text="Amazing food and service!" \
  cuisine="Italian" \
  rating_rest:=4.5
```

### 3. Batch Prediction (from file)
Create a file `batch_request.json`:
```json
[
  {
    "review_text": "Great experience!",
    "cuisine": "Italian"
  },
  {
    "review_text": "Not impressed",
    "cuisine": "Chinese"
  }
]
```

Then run:
```bash
http POST http://localhost:8000/predict < batch_request.json
```

---

## Example Payloads

### Minimal Payload (Only Required Fields)
```json
[
  {
    "review_text": "The food was okay."
  }
]
```

### Basic Payload (Common Fields)
```json
[
  {
    "restaurant_id": 1,
    "user_id": 100,
    "review_text": "Great food and atmosphere!",
    "cuisine": "Italian",
    "location": "Dubai Marina",
    "rating_rest": 4.3
  }
]
```

### Complete Payload (All Fields)
```json
[
  {
    "restaurant_id": 1,
    "user_id": 100,
    "review_text": "Outstanding dining experience! The pasta was perfectly cooked and the service was exceptional. Highly recommend!",
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
]
```

### Batch Payload (Multiple Reviews)
```json
[
  {
    "restaurant_id": 1,
    "user_id": 100,
    "review_text": "Excellent! Best pizza in town.",
    "cuisine": "Italian",
    "rating_rest": 4.7
  },
  {
    "restaurant_id": 2,
    "user_id": 101,
    "review_text": "Average food, slow service.",
    "cuisine": "Chinese",
    "rating_rest": 3.2
  },
  {
    "restaurant_id": 3,
    "user_id": 102,
    "review_text": "Fantastic tacos and great ambiance!",
    "cuisine": "Mexican",
    "rating_rest": 4.5
  }
]
```

---

## Running the API Server

Before testing, make sure the API server is running:

```bash
# Install dependencies
pip install fastapi uvicorn joblib pandas numpy scikit-learn xgboost

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs (Swagger UI): http://localhost:8000/docs
- Alternative Docs (ReDoc): http://localhost:8000/redoc

---

## Expected Response Format

All successful `/predict` requests return:

```json
{
  "predictions": [
    {
      "restaurant_id": 1,
      "user_id": 100,
      "predicted_rating": 4.234567,
      "predicted_rating_rounded": 4.23
    }
  ]
}
```

---

## Error Responses

### 422 Unprocessable Entity (Validation Error)
```json
{
  "detail": [
    {
      "loc": ["body", 0, "review_text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

---

## Tips for Testing

1. **Start with health check**: Always verify the API is running with `/health`
2. **Test incrementally**: Start with minimal payloads, then add more fields
3. **Use interactive docs**: Visit http://localhost:8000/docs for built-in testing
4. **Check logs**: Monitor server logs for detailed error messages
5. **Validate JSON**: Ensure your JSON payload is properly formatted
6. **Test edge cases**: Try empty strings, null values, extreme numbers
7. **Batch testing**: Test with various batch sizes (1, 10, 50, 100 reviews)

---

## Continuous Integration

For CI/CD pipelines, run tests with:

```bash
# Run tests and generate XML report for CI tools
pytest test_main.py --junitxml=test-results.xml --cov=main --cov-report=xml

# Run with verbose output and stop on first failure
pytest test_main.py -v -x

# Run tests in parallel (install pytest-xdist)
pip install pytest-xdist
pytest test_main.py -n auto
```

---

## Additional Resources

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pytest Documentation: https://docs.pytest.org/
- Python Requests: https://docs.python-requests.org/
- HTTPie: https://httpie.io/docs
- Postman: https://learning.postman.com/

