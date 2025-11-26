# Quick Start Guide - 5 Minutes to Running

Get the restaurant search system running in 5 minutes!

## 1. Setup (2 minutes)

```bash
# Clone/navigate to project
cd Recommender-System

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure API Key (1 minute)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"     # macOS/Linux
$env:OPENAI_API_KEY="sk-your-key-here"       # Windows PowerShell
```

Get your API key from: https://platform.openai.com/api-keys

## 3. Run (2 minutes)

```bash
# Start interactive mode
python -m src.main --mode interactive
```

**First run**: Builds vector store (~30 seconds)
**Subsequent runs**: Instant start (loads from disk)

## 4. Try These Queries

```
Find Italian restaurants in Downtown Dubai with outdoor seating under AED 200

Show me budget-friendly vegetarian options

I want a romantic fine dining experience

What are the best Japanese restaurants?

Find a place for a business lunch in DIFC
```

## That's It! 🎉

**Next Steps:**
- Try `--mode examples` to see demo queries
- Read `README.md` for full documentation
- Check `docs/` for architecture details
- Customize `src/config.py` for your needs

## Common Issues

**"No module named src"**
→ Make sure you're in the project root directory

**"OPENAI_API_KEY not set"**
→ Run the export/set command from step 2

**Slow first run**
→ Normal! Building vector store takes ~30s first time

## Quick Commands Reference

```bash
# Interactive mode
python -m src.main --mode interactive

# Run examples
python -m src.main --mode examples

# Single query
python -m src.main --query "your query here"

# Rebuild vector store
python -m src.main --rebuild

# Get help
python -m src.main --help
```

---

**Need more details?** Check `SETUP_GUIDE.md` or `README.md`

