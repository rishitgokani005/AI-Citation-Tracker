# AI Citation Tracker

AI Citation Tracker is a Generative Engine Optimization (GEO) research tool that tracks, measures, and analyzes brand visibility, mention share, placement position, and source citations across LLM recommendations (Google Gemini).

## Features

- **Gemini LLM Integration**: Query Google Gemini API with rate limiting, retry logic, and exponential backoff.
- **Brand & Citation Parser**: Automatically parses raw responses for brand mentions, appearance order rank, sentiment classification, and URL/domain citations.
- **SQLite Database Persistence**: Structured relational schema storing raw responses, brand mentions, and citations.
- **Pandas Analytics Engine**: Computes brand mention share (%), average placement position, sentiment breakdown, and funnel stage distribution.
- **Interactive Streamlit Dashboard**: Displays interactive Plotly visualizations, domain citation tables, and auto-generated GEO insights.

---

## Project Structure

```
AI Citation Tracker/
├── .env.example              # Environment variables template
├── .env                      # Local environment configuration (not committed)
├── config.py                 # Core configuration file
├── prompts.json              # Bank of 15 prompts across 5 sales funnel stages
├── brands.json               # Seed list of target brands
├── requirements.txt          # Python dependencies
├── db/
│   └── schema.sql            # SQLite database schema definition
├── src/
│   ├── query_engine.py       # API query handler for Google Gemini
│   ├── db_manager.py         # SQLite CRUD operations
│   ├── response_parser.py    # Regex & keyword parser for mentions/citations
│   ├── analysis.py           # Pandas analytics engine & insight generator
│   └── run_pipeline.py       # Main CLI pipeline orchestration script
├── dashboard/
│   └── app.py                # Streamlit dashboard UI
└── tests/
    └── test_tracker.py       # Pytest unit & integration test suite
```

---

## How to Run the Project

### 1. Prerequisites & Virtual Environment

Make sure you have Python 3.10+ installed.

Activate your virtual environment (or create one):
```powershell
# Windows PowerShell
.\venv\Scripts\activate
```
*(If virtual environment does not exist yet, create it with `python -m venv venv`)*

### 2. Install Dependencies

Install the required Python libraries:
```powershell
pip install -r requirements.txt
```

### 3. Set Up Gemini API Key

1. Get a free API Key from [Google AI Studio](https://aistudio.google.com/).
2. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Open `.env` and set your API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   MODEL_NAME=gemini-3.1-flash-lite
   ENABLE_GROUNDING=false
   ```

---

### 4. Run the Data Pipeline

To query Gemini for all 15 prompts, parse responses, and store results in SQLite:

```powershell
python src/run_pipeline.py
```

- To force re-running all prompts and overwrite existing records in the database:
  ```powershell
  python src/run_pipeline.py --force
  ```

---

### 5. Launch the Streamlit Dashboard

Start the interactive web dashboard to view charts, metrics, and insights:

```powershell
streamlit run dashboard/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

### 6. Run Tests

To verify that database CRUD, response parsing, query engine error handling, and metrics analysis work properly:

```powershell
pytest tests/test_tracker.py
```

## Security

- Never commit `.env`, API keys, database files, or virtual environments.
- The repository includes `.env.example` with placeholders only.
- If an API key is exposed, revoke it in Google AI Studio and create a replacement.
