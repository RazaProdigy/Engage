# Testing Quick Reference

Quick commands and examples for testing the Restaurant Rating Prediction API.

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements-test.txt

# 2. Start the API server (in one terminal)
uvicorn main:app --reload

# 3. Run tests (in another terminal)
pytest test_main.py -v
```

---

## 📝 Common Commands

### Run Tests

```bash
# All tests
pytest test_main.py -v

# With coverage
pytest test_main.py --cov=main --cov-report=html

# Specific test
pytest test_main.py::TestPredictEndpoint::test_predict_single_review -v

# Stop on first failure
pytest test_main.py -x

# Run in parallel (faster)
pytest test_main.py -n auto
```

### Using Test Scripts

```bash
# Windows
run_tests.bat all         # Run all tests
run_tests.bat coverage    # With coverage report
run_tests.bat manual      # Manual interactive tests

# Linux/Mac
chmod +x run_tests.sh     # Make executable (first time only)
./run_tests.sh all        # Run all tests
./run_tests.sh coverage   # With coverage report
./run_tests.sh manual     # Manual interactive tests
```

### Manual Testing

```bash
# Interactive test script
python test_api_manual.py

# Or visit interactive docs
# Start server, then open: http://localhost:8000/docs
```

---

## 🔧 Quick API Tests

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict rating
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"review_text": "Amazing food!", "cuisine": "Italian"}]'
```

### Using Python

```python
import requests

# Health check
requests.get("http://localhost:8000/health").json()

# Predict
payload = [{"review_text": "Great!", "cuisine": "Italian"}]
requests.post("http://localhost:8000/predict", json=payload).json()
```

---

## 📊 Test Files

| File | Purpose |
|------|---------|
| `test_main.py` | Automated pytest suite (40+ tests) |
| `test_api_manual.py` | Interactive manual testing script |
| `test_requests_guide.md` | Complete API request examples |
| `README_TESTING.md` | Full testing documentation |
| `pytest.ini` | Pytest configuration |
| `requirements-test.txt` | Testing dependencies |

---

## 🎯 Example Requests

### Minimal Request
```json
[{"review_text": "Good food"}]
```

### Full Request
```json
[{
  "restaurant_id": 1,
  "user_id": 100,
  "review_text": "Amazing experience!",
  "cuisine": "Italian",
  "location": "Dubai Marina",
  "price_range": "AED 100-200",
  "rating_rest": 4.5,
  "review_count": 250,
  "age": 30,
  "avg_rating_total_reviews_written": 4.2
}]
```

### Expected Response
```json
{
  "predictions": [{
    "restaurant_id": 1,
    "user_id": 100,
    "predicted_rating": 4.234567,
    "predicted_rating_rounded": 4.23
  }]
}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start server: `uvicorn main:app --reload` |
| Module not found | Install: `pip install -r requirements-test.txt` |
| Model file missing | Ensure `rating_model.joblib` exists |
| Import errors | Run from project root directory |

---

## 📍 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | API info |
| POST | `/predict` | Predict ratings |
| GET | `/docs` | Interactive docs (Swagger) |
| GET | `/redoc` | Alternative docs (ReDoc) |

---

## 💡 Pro Tips

- **Use interactive docs**: Visit http://localhost:8000/docs for visual API testing
- **Run with coverage**: Add `--cov=main --cov-report=html` to see what's tested
- **Debug specific tests**: Use `-k "keyword"` to run matching tests
- **Speed up tests**: Use `-n auto` for parallel execution
- **See print statements**: Add `-s` flag to pytest

---

## 📚 More Information

- Full guide: `README_TESTING.md`
- Request examples: `test_requests_guide.md`
- Run manual tests: `python test_api_manual.py`

---

**Need Help?**

1. Check `README_TESTING.md` for detailed documentation
2. Run `python test_api_manual.py` to see the API in action
3. Visit http://localhost:8000/docs for interactive API exploration

