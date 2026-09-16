import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file (for local development)
load_dotenv(BASE_DIR / ".env")

# API Configuration
# Prefer os.environ (works locally via .env, and on most platforms).
# Fall back to Streamlit's st.secrets, since Streamlit Cloud secrets aren't
# always mirrored into os.environ before this module is imported.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        # Not running inside Streamlit (e.g. run_pipeline.py from the CLI
        # with no secrets.toml present) — just keep the empty string and
        # let get_api_key_or_raise() report it clearly.
        pass

# Using model identifier. Normalizes human-friendly strings like "Gemini 3.1 Flash Lite" to valid API ID "gemini-3.1-flash-lite".
raw_model = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
MODEL_NAME = raw_model.strip().lower().replace(" ", "-")

# Grounding Configuration
# When enabled, Gemini calls include the Google Search tool so the model can
# actually browse the web and return real source URLs via grounding_metadata,
# instead of only whatever URLs it happens to mention in plain text.
ENABLE_GROUNDING = os.getenv("ENABLE_GROUNDING", "true").lower() in ("1", "true", "yes")

# Niche & Data Configuration
NICHE_NAME = "CRM tools for freelancers"
PROMPTS_PATH = BASE_DIR / "prompts.json"
BRANDS_PATH = BASE_DIR / "brands.json"
DB_PATH = BASE_DIR / "db" / "citation_tracker.db"
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"

# Rate Limit Delay between API calls (in seconds)
REQUEST_DELAY_SECONDS = 2.0


def validate_api_key() -> bool:
    """Checks if GOOGLE_API_KEY is properly configured."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        return False
    return True


def get_api_key_or_raise() -> str:
    """Returns GOOGLE_API_KEY or raises a user-friendly ValueError."""
    if not validate_api_key():
        raise ValueError(
            "Missing or invalid GOOGLE_API_KEY in .env file.\n"
            "Please create a free API key at Google AI Studio (https://aistudio.google.com/)\n"
            "and set GOOGLE_API_KEY in your .env file."
        )
    return GOOGLE_API_KEY