import sys
from pathlib import Path

# Add project root directory to sys.path so config can be imported from src/ scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import logging
from typing import List, Dict, Any

import config
import query_engine
import response_parser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("run_pipeline")


def load_json_file(file_path) -> List[Any]:
    """Loads and parses a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(platform: str = "gemini", force: bool = False) -> Dict[str, Any]:
    """Orchestrates prompt reading, API querying, parsing, and database persistence.

    Args:
        platform: Target LLM platform (default: "gemini").
        force: If True, re-query and overwrite existing DB records for all prompts.

    Returns:
        Dict summary of pipeline execution stats.
    """
    logger.info(f"Starting AI Citation Tracker pipeline for platform: '{platform}'")
    logger.info(f"Grounding enabled: {config.ENABLE_GROUNDING}")

    # 1. Initialize Database
# (No DB initialization needed; pipeline runs fresh each execution)

    # 2. Load Prompts & Brands
    prompts = load_json_file(config.PROMPTS_PATH)
    brands = load_json_file(config.BRANDS_PATH)
    logger.info(f"Loaded {len(prompts)} prompts and {len(brands)} seed brands.")

    stats = {
        "total_prompts": len(prompts),
        "processed": 0,
        "skipped": 0,
        "api_errors": 0,
        "total_mentions": 0,
        "total_citations": 0,
        "grounded_citations": 0,
        "fallback_citations": 0
    }

    # 3. Iterate over prompts
    for p in prompts:
        prompt_id = p["id"]
        prompt_text = p["text"]
        stage = p.get("stage", "unknown")

        logger.info(f"--- [Prompt {prompt_id}/15] ({stage}) ---")

        # No existing record check – always process prompts fresh
        # Query API
        query_result = query_engine.query_gemini(prompt_text)
        raw_text = query_result.get("raw_text", "")
        error = query_result.get("error")
        grounding_citations = query_result.get("citations", [])
        was_grounded = query_result.get("grounded", False)

        if error or not raw_text:
            logger.error(f"Failed to query prompt {prompt_id}: {error or 'Empty response'}")
            stats["api_errors"] += 1
            continue

        # Parse Response — pass grounding citations through so the parser
        # uses real sources instead of falling back to regex-scraped text
        parsed = response_parser.parse_response(
            raw_text, brands, grounding_citations=grounding_citations
        )
        mentions = parsed["mentions"]
        citations = parsed["citations"]

        # Track whether citations came from grounding or the regex fallback,
        # useful for sanity-checking how often grounding actually returns sources
        if was_grounded and grounding_citations:
            stats["grounded_citations"] += len(citations)
        else:
            stats["fallback_citations"] += len(citations)

        # Update overall stats (no DB writes)
        stats["total_mentions"] += len(mentions)
        stats["total_citations"] += len(citations)

# Citations are not persisted to a database; they are counted in stats only.

        stats["processed"] += 1
        logger.info(f"Saved prompt {prompt_id}: Found {len(mentions)} mentions, {len(citations)} citations.")

    # Print Pipeline Summary
    logger.info("==========================================")
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("==========================================")
    logger.info(f"Total Prompts:      {stats['total_prompts']}")
    logger.info(f"New Processed:      {stats['processed']}")
    logger.info(f"Skipped (Cached):   {stats['skipped']}")
    logger.info(f"API Errors:         {stats['api_errors']}")
    logger.info(f"Brand Mentions:     {stats['total_mentions']}")
    logger.info(f"Citations Extracted:{stats['total_citations']}")
    logger.info(f"  - From grounding: {stats['grounded_citations']}")
    logger.info(f"  - Fallback regex: {stats['fallback_citations']}")
    logger.info("==========================================")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Citation Tracker Pipeline")
    parser.add_argument("--platform", type=str, default="gemini", help="Target LLM platform (default: gemini)")
    parser.add_argument("--force", action="store_true", help="Force re-run all prompts even if already in DB")
    args = parser.parse_args()

    run_pipeline(platform=args.platform, force=args.force)