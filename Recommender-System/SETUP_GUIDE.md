# Setup Guide

Complete setup instructions for the RAG-based Restaurant Search System.

## Prerequisites

- **Python**: 3.12 or higher
- **OpenAI API Key**: Required for embeddings and LLM
- **Git**: For version control (optional)

## Step-by-Step Installation

### 1. Environment Setup

#### Windows
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation (you should see (venv) in prompt)
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in prompt)
```

### 2. Install Dependencies

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Expected installation time**: 2-3 minutes

### 3. Configure OpenAI API Key

#### Option A: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

#### Option B: .env File

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Verify Installation

```bash
# Test import
python -c "from src.rag_system import RestaurantRAGSystem; print('✓ Installation successful!')"
```

## First Run

### Initialize Vector Store

On first run, the system will automatically:
1. Load restaurant data (50 restaurants)
2. Generate embeddings (~30 seconds)
3. Build vector store
4. Save to disk for future use

```bash
# First run - builds vector store
python -m src.main --mode interactive

# Subsequent runs - loads from disk (much faster)
python -m src.main --mode interactive
```

### Force Rebuild (if needed)

```bash
python -m src.main --rebuild --mode interactive
```

## Usage Modes

### 1. Interactive Mode (Recommended for Testing)

```bash
python -m src.main --mode interactive
```

**Features**:
- Multi-turn conversations
- Type queries naturally
- Commands: `quit`, `exit`, `clear`

**Example Session**:
```
🔍 You: Find Italian restaurants in Downtown with outdoor seating

🤖 Assistant: [Personalized recommendations]

🔍 You: What about under AED 150?

🤖 Assistant: [Refined results based on conversation context]
```

### 2. Example Queries Mode

```bash
python -m src.main --mode examples
```

Runs 5 pre-configured example queries to demonstrate capabilities.

### 3. Single Query Mode

```bash
python -m src.main --query "Find Japanese restaurants in Marina"
```

Useful for:
- Quick tests
- Automation
- API integration

### 4. Both Modes

```bash
python -m src.main --mode both
```

Runs examples first, then enters interactive mode.

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"

**Solution**:
```bash
# Check if set
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows CMD
echo $env:OPENAI_API_KEY  # Windows PowerShell

# If not set, follow Step 3 above
```

### Issue: "Module not found"

**Solution**:
```bash
# Ensure virtual environment is activated
# You should see (venv) in your prompt

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Vector store not found"

**Solution**:
```bash
# Rebuild vector store
python -m src.main --rebuild --mode interactive
```

### Issue: Slow first run

**Expected**: First run takes 30-60 seconds to build vector store.

**Solution**: Subsequent runs load from disk and start immediately.

### Issue: API rate limits

**Solution**:
- OpenAI free tier: 3 RPM (requests per minute)
- Upgrade to paid tier for higher limits
- Add delay between queries if hitting limits

## Project Structure Quick Reference

```
Recommender-System/
├── data/
│   ├── restaurants_data.json      # 50 restaurants (modify to add more)
│   └── vectorstore/                # Auto-generated, safe to delete to rebuild
├── src/
│   ├── config.py                   # Adjust settings here
│   ├── rag_system.py              # Core RAG logic
│   ├── agents.py                   # Multi-agent workflow
│   └── main.py                     # Application entry point
├── docs/                           # Comprehensive documentation
├── tests/                          # Test suite
└── logs/                           # Auto-generated logs
```

## Configuration Customization

Edit `src/config.py` to customize:

### Change LLM Model
```python
LLM_CONFIG = {
    "model": "gpt-4",  # Change to gpt-4 for higher quality
    "temperature": 0.3,
}
```

### Adjust Search Parameters
```python
SEARCH_CONFIG = {
    "top_k": 15,  # Return more results
    "semantic_weight": 0.7,  # More semantic, less keyword
}
```

### Modify System Prompts
```python
SYSTEM_PROMPTS = {
    "response_generation": """Your custom prompt here""",
}
```

## Performance Tips

### Speed Optimization
1. **Keep vector store**: Don't rebuild unless data changes
2. **Use GPT-4o-mini**: Faster and cheaper than GPT-4
3. **Reduce top_k**: Fewer results = faster search

### Cost Optimization
```python
# Estimated costs per query:
# - Embeddings: $0.0001
# - LLM calls (2-3): $0.002
# Total: ~$0.002 per query
# 
# 1000 queries = $2
```

## Development Workflow

### Adding New Restaurants

1. Edit `data/restaurants_data.json`
2. Add new restaurant objects (follow existing format)
3. Rebuild vector store:
   ```bash
   python -m src.main --rebuild
   ```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_rag_system.py -v

# With coverage
pytest --cov=src tests/
```

### Checking Logs

```bash
# View recent logs
tail -f logs/restaurant_search.log

# Search logs
grep "ERROR" logs/restaurant_search.log
```

## Next Steps

1. ✅ Complete setup following this guide
2. ✅ Run interactive mode and test with queries
3. ✅ Read `README.md` for architecture overview
4. ✅ Explore `docs/ARCHITECTURE.md` for technical details
5. ✅ Review `docs/DESIGN_DECISIONS.md` for rationale
6. ✅ Customize `src/config.py` for your needs

## Getting Help

- Check `docs/` directory for detailed documentation
- Review example queries in `src/main.py`
- Inspect logs in `logs/restaurant_search.log`
- Run with `--help` flag: `python -m src.main --help`

## Common Use Cases

### 1. Demo/Presentation
```bash
python -m src.main --mode examples
```

### 2. Testing Changes
```bash
python -m src.main --query "test query" --rebuild
```

### 3. Interactive Exploration
```bash
python -m src.main --mode interactive
```

### 4. Production API (Future)
```python
from src.main import RestaurantSearchApp

app = RestaurantSearchApp(api_key="...")
result = app.search("Find Italian restaurants")
```

---

**You're all set! 🎉**

Start with: `python -m src.main --mode interactive`

