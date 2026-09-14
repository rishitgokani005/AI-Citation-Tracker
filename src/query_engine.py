import sys
from pathlib import Path

# Add project root directory to sys.path so config can be imported from src/ scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import logging
from typing import Dict, Any, List
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_grounding_citations(response: Any) -> List[Dict[str, Any]]:
    """Pulls real source URLs out of a grounded Gemini response.

    When the Google Search tool is enabled, a grounded response carries a
    grounding_metadata object on its first candidate, containing
    grounding_chunks -> web -> {uri, title} for each source the model
    actually consulted. This is the ONLY reliable source of real citations;
    anything scraped from the plain text is a fallback, not a substitute.

    Returns:
        list of {"url": Optional[str], "title": Optional[str]}
    """
    citations: List[Dict[str, Any]] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return citations

        grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
        if not grounding_metadata:
            return citations

        chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            uri = getattr(web, "uri", None)
            title = getattr(web, "title", None)
            if uri or title:
                citations.append({"url": uri, "title": title})
    except Exception as e:
        logger.warning(f"Failed to parse grounding metadata: {e}")

    return citations


def _is_daily_quota_exhausted(exception: Exception) -> bool:
    """Detects whether a 429 is a PER-DAY quota violation (RPD), as opposed to
    a per-minute (RPM) or per-token (TPM) limit.

    Retrying a per-minute limit makes sense — waiting a few seconds can help.
    Retrying a per-day limit does NOT — the quota won't reset until the next
    day, so every retry just wastes time and (if any daily budget remains)
    burns through it for no benefit. We detect this by looking for the
    quotaId Google includes in the 429 error body, which contains "PerDay"
    for daily limits (e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier").
    """
    message = str(exception)
    if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
        return False
    # Google's quotaId naming convention marks daily limits explicitly.
    daily_markers = ("PerDay", "per_day", "RequestsPerDay", "daily")
    return any(marker in message for marker in daily_markers)


def query_gemini(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Queries the Google Gemini API with exponential backoff and rate limiting.

    When config.ENABLE_GROUNDING is True, the Google Search tool is attached
    so the model can browse the web and return real, structured citations.

    If a 429 is identified as a daily-quota (RPD) violation, retries are
    skipped immediately rather than burning through backoff delays and any
    remaining quota on a limit that won't reset until the next day.

    Returns:
        dict: {
            "raw_text": str,
            "citations": [{"url": str|None, "title": str|None}, ...],  # grounded sources, may be empty
            "grounded": bool,  # whether grounding was actually attempted
            "daily_quota_exhausted": bool,  # True if this failure was a same-day RPD limit
            "error": str  # present only on failure
        }
    """
    try:
        api_key = config.get_api_key_or_raise()

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "The google-genai package is not installed. "
                "Please run 'pip install google-genai'."
            )

        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Gemini configuration error: {e}")
        return {
            "raw_text": "",
            "citations": [],
            "grounded": False,
            "daily_quota_exhausted": False,
            "error": str(e)
        }

    generate_config = None
    if config.ENABLE_GROUNDING:
        generate_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Querying Gemini (Attempt {attempt}/{max_retries})...")
            # Rate limit politeness pause before calling
            time.sleep(config.REQUEST_DELAY_SECONDS)

            if generate_config is not None:
                response = client.models.generate_content(
                    model=config.MODEL_NAME,
                    contents=prompt,
                    config=generate_config,
                )
            else:
                response = client.models.generate_content(
                    model=config.MODEL_NAME,
                    contents=prompt
                )

            raw_text = ""
            if response:
                try:
                    raw_text = response.text or ""
                except Exception:
                    # Fallback: pull text out of candidates/parts if .text isn't populated
                    try:
                        candidates = getattr(response, "candidates", None) or []
                        parts_text = []
                        for cand in candidates:
                            content = getattr(cand, "content", None)
                            parts = getattr(content, "parts", []) if content else []
                            for part in parts:
                                if hasattr(part, "text") and part.text:
                                    parts_text.append(part.text)
                        raw_text = "".join(parts_text)
                    except Exception:
                        raw_text = ""

            if not raw_text:
                logger.warning("Gemini returned empty text response.")

            grounding_citations = (
                _extract_grounding_citations(response) if config.ENABLE_GROUNDING else []
            )

            return {
                "raw_text": raw_text,
                "citations": grounding_citations,
                "grounded": config.ENABLE_GROUNDING,
                "daily_quota_exhausted": False
            }

        except Exception as e:
            last_exception = e

            if _is_daily_quota_exhausted(e):
                logger.error(
                    f"Daily quota (RPD) exhausted on attempt {attempt} — "
                    f"skipping remaining retries, this won't reset until tomorrow."
                )
                return {
                    "raw_text": "",
                    "citations": [],
                    "grounded": config.ENABLE_GROUNDING,
                    "daily_quota_exhausted": True,
                    "error": str(e)
                }

            logger.warning(f"Gemini query attempt {attempt} failed: {e}")
            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)

    logger.error(f"All {max_retries} attempts to query Gemini failed.")
    return {
        "raw_text": "",
        "citations": [],
        "grounded": config.ENABLE_GROUNDING,
        "daily_quota_exhausted": False,
        "error": str(last_exception)
    }

