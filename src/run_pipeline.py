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
import db_manager
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
        force: If True, clears the database before re-running all prompts.

    Returns:
        Dict summary of pipeline execution stats.
    """
    logger.info(f"Starting AI Citation Tracker pipeline for platform: '{platform}'")
    logger.info(f"Grounding enabled: {config.ENABLE_GROUNDING}")

    # 1. Initialize DB schema (creates tables if they don't exist yet)
    db_manager.init_db()

    # 2. Clear database if --force flag is set
    if force:
        logger.info("--force flag detected: clearing existing database records before re-running.")
        db_manager.clear_database()
        logger.info("Database cleared successfully.")

    # 3. Load Prompts & Brands
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

    # 4. Iterate over prompts
    for p in prompts:
        prompt_id = p["id"]
        prompt_text = p["text"]
        stage = p.get("stage", "unknown")

        logger.info(f"--- [Prompt {prompt_id}/{len(prompts)}] ({stage}) ---")

        # Query Gemini API
        query_result = query_engine.query_gemini(prompt_text)
        raw_text = query_result.get("raw_text", "")
        error = query_result.get("error")
        grounding_citations = query_result.get("citations", [])
        was_grounded = query_result.get("grounded", False)

        if error or not raw_text:
            logger.error(f"Failed to query prompt {prompt_id}: {error or 'Empty response'}")
            stats["api_errors"] += 1
            continue

        # Parse response for brand mentions and citations
        parsed = response_parser.parse_response(
            raw_text, brands, grounding_citations=grounding_citations
        )
        mentions = parsed["mentions"]
        citations = parsed["citations"]

        # Persist response to database
        response_id = db_manager.insert_response(
            prompt_id=prompt_id,
            platform=platform,
            raw_text=raw_text
        )

        # Persist mentions to database
        for mention in mentions:
            db_manager.insert_mention(
                response_id=response_id,
                brand_name=mention["brand_name"],
                position=mention["position"],
                sentiment=mention["sentiment"]
            )

        # Persist citations to database
        for citation in citations:
            db_manager.insert_citation(
                response_id=response_id,
                source_url=citation.get("source_url"),
                source_domain=citation.get("source_domain", "")
            )

        # Track citation source type
        if was_grounded and grounding_citations:
            stats["grounded_citations"] += len(citations)
        else:
            stats["fallback_citations"] += len(citations)

        stats["total_mentions"] += len(mentions)
        stats["total_citations"] += len(citations)
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
    parser.add_argument("--force", action="store_true", help="Clear DB and re-run all prompts fresh")
    args = parser.parse_args()

    run_pipeline(platform=args.platform, force=args.force)