import os
import sys
import tempfile
from pathlib import Path
import pytest
import pandas as pd

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db_manager
import response_parser
import analysis
import query_engine
import data_manager



@pytest.fixture
def temp_db():
    """Fixture providing a clean temporary SQLite database."""
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "test_citation_tracker.db"
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

    db_manager.init_db(db_path=db_path, schema_path=schema_path)
    yield db_path
    temp_dir.cleanup()


def test_database_init_and_crud(temp_db):
    """Verifies table creation and CRUD operations in db_manager."""
    # Insert response
    resp_id = db_manager.insert_response(prompt_id=1, platform="gemini", raw_text="HubSpot is great for freelancers.", db_path=temp_db)
    assert resp_id > 0

    # Insert mention
    mention_id = db_manager.insert_mention(response_id=resp_id, brand_name="HubSpot", position=1, sentiment="recommended", db_path=temp_db)
    assert mention_id > 0

    # Insert citation
    citation_id = db_manager.insert_citation(response_id=resp_id, source_url="https://www.hubspot.com/pricing", source_domain="hubspot.com", db_path=temp_db)
    assert citation_id > 0

    # Retrieve data
    responses = db_manager.fetch_all_responses(db_path=temp_db)
    mentions = db_manager.fetch_all_mentions(db_path=temp_db)
    citations = db_manager.fetch_all_citations(db_path=temp_db)

    assert len(responses) == 1
    assert responses[0]["platform"] == "gemini"
    assert len(mentions) == 1
    assert mentions[0]["brand_name"] == "HubSpot"
    assert len(citations) == 1
    assert citations[0]["source_domain"] == "hubspot.com"


def test_insert_citation_with_null_url(temp_db):
    """Verifies citations with no resolvable URL (title-only grounding source) still insert cleanly."""
    resp_id = db_manager.insert_response(prompt_id=1, platform="gemini", raw_text="Bonsai is solid.", db_path=temp_db)

    citation_id = db_manager.insert_citation(
        response_id=resp_id, source_url=None, source_domain="g2 reviews: bonsai", db_path=temp_db
    )
    assert citation_id > 0

    citations = db_manager.fetch_all_citations(db_path=temp_db)
    assert len(citations) == 1
    assert citations[0]["source_url"] is None
    assert citations[0]["source_domain"] == "g2 reviews: bonsai"


def test_response_parser():
    """Verifies brand detection, position ordering, sentiment heuristics, and URL extraction."""
    sample_text = (
        "For freelance client management, Bonsai is top choice and highly recommended. "
        "HubSpot is also popular but expensive and complex. "
        "Check reviews at https://www.capterra.com/p/bonsai."
    )
    brand_list = ["HubSpot", "Bonsai", "Zoho CRM"]

    parsed = response_parser.parse_response(sample_text, brand_list)

    mentions = parsed["mentions"]
    citations = parsed["citations"]

    assert len(mentions) == 2

    # Bonsai appears first
    assert mentions[0]["brand_name"] == "Bonsai"
    assert mentions[0]["position"] == 1
    assert mentions[0]["sentiment"] == "recommended"

    # HubSpot appears second
    assert mentions[1]["brand_name"] == "HubSpot"
    assert mentions[1]["position"] == 2
    assert mentions[1]["sentiment"] == "negative"

    # Citations (regex fallback, since no grounding_citations passed)
    assert len(citations) == 1
    assert citations[0]["source_domain"] == "capterra.com"
    assert citations[0]["source_url"] == "https://www.capterra.com/p/bonsai"


def test_response_parser_prefers_grounding_citations():
    """Verifies structured grounding citations are used instead of regex-scraped text URLs."""
    sample_text = (
        "Bonsai is a great pick for freelancers. See https://example-fake-link.com for details."
    )
    brand_list = ["Bonsai"]

    grounding_citations = [
        {"url": "https://www.g2.com/products/bonsai/reviews", "title": "Bonsai Reviews - G2"},
        {"url": "https://www.capterra.com/p/bonsai", "title": "Bonsai - Capterra"},
    ]

    parsed = response_parser.parse_response(sample_text, brand_list, grounding_citations=grounding_citations)
    citations = parsed["citations"]

    # Should use the two grounded sources, NOT the regex-scraped example-fake-link.com
    domains = {c["source_domain"] for c in citations}
    assert domains == {"g2.com", "capterra.com"}
    assert "example-fake-link.com" not in domains


def test_response_parser_grounding_citation_title_only():
    """Verifies a grounding source with no URI falls back to using its title as the domain."""
    grounding_citations = [{"url": None, "title": "Freelancer CRM Buying Guide 2026"}]

    parsed = response_parser.parse_response("Bonsai is solid.", ["Bonsai"], grounding_citations=grounding_citations)
    citations = parsed["citations"]

    assert len(citations) == 1
    assert citations[0]["source_url"] is None
    assert citations[0]["source_domain"] == "freelancer crm buying guide 2026"


def test_response_parser_empty_text():
    """Verifies an empty raw_text returns empty mentions and citations without error."""
    parsed = response_parser.parse_response("", ["HubSpot", "Bonsai"])
    assert parsed["mentions"] == []
    assert parsed["citations"] == []


def test_response_parser_no_brand_mentions():
    """Verifies text with none of the target brands returns zero mentions, not an error."""
    sample_text = "Freelancers often struggle to track client communication manually."
    parsed = response_parser.parse_response(sample_text, ["HubSpot", "Bonsai", "Zoho CRM"])
    assert parsed["mentions"] == []
    # No URLs and no grounding citations provided -> empty citations too
    assert parsed["citations"] == []


def test_analysis_metrics(temp_db):
    """Verifies pandas analysis metrics computation on synthetic database data."""
    # Seed DB with two responses
    resp1 = db_manager.insert_response(prompt_id=1, platform="gemini", raw_text="HubSpot vs Zoho", db_path=temp_db)
    resp2 = db_manager.insert_response(prompt_id=4, platform="gemini", raw_text="HubSpot review", db_path=temp_db)

    db_manager.insert_mention(resp1, "HubSpot", 1, "recommended", db_path=temp_db)
    db_manager.insert_mention(resp1, "Zoho CRM", 2, "neutral", db_path=temp_db)
    db_manager.insert_mention(resp2, "HubSpot", 1, "recommended", db_path=temp_db)

    db_manager.insert_citation(resp1, "https://g2.com/hubspot", "g2.com", db_path=temp_db)

    df_share = analysis.get_mention_share(db_path=temp_db)
    assert not df_share.empty

    hubspot_share = df_share[df_share["brand_name"] == "HubSpot"].iloc[0]
    assert hubspot_share["mention_count"] == 2
    assert hubspot_share["mention_share_pct"] == 100.0

    df_pos = analysis.get_average_position(db_path=temp_db)
    hubspot_pos = df_pos[df_pos["brand_name"] == "HubSpot"].iloc[0]
    assert hubspot_pos["avg_position"] == 1.0

    insights = analysis.generate_insights(db_path=temp_db)
    assert len(insights) >= 3
    assert "HubSpot" in insights[0]










def test_data_manager_prompt_crud_and_validation():
    """Verifies prompt auto-increment ID, stage validation, duplicate rejection, and atomic writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_path = Path(temp_dir) / "test_prompts.json"

        # Load empty file
        assert data_manager.load_prompts(prompts_path) == []

        # Add prompt 1
        p1 = data_manager.add_prompt("Best CRM software for freelancers?", "research", prompts_path)
        assert p1["id"] == 1
        assert p1["text"] == "Best CRM software for freelancers?"
        assert p1["stage"] == "research"

        # Add prompt 2 (ID auto-increment)
        p2 = data_manager.add_prompt("HubSpot vs Zoho for freelancers", "comparison", prompts_path)
        assert p2["id"] == 2

        # Verify load
        loaded = data_manager.load_prompts(prompts_path)
        assert len(loaded) == 2

        # Rejects duplicate prompt (case-insensitive, trimmed)
        with pytest.raises(ValueError, match="already exists"):
            data_manager.add_prompt("  best CRM software for freelancers?  ", "research", prompts_path)

        # Rejects invalid stage
        with pytest.raises(ValueError, match="Invalid stage"):
            data_manager.add_prompt("New prompt text", "invalid_stage_name", prompts_path)

        # Rejects empty prompt text
        with pytest.raises(ValueError, match="cannot be empty"):
            data_manager.add_prompt("   ", "research", prompts_path)


def test_data_manager_brand_crud_and_validation():
    """Verifies brand insertion, duplicate brand rejection, empty name handling, and atomic writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        brands_path = Path(temp_dir) / "test_brands.json"

        # Load empty file
        assert data_manager.load_brands(brands_path) == []

        # Add brand 1
        b1 = data_manager.add_brand("HubSpot", brands_path)
        assert b1 == "HubSpot"

        # Add brand 2
        b2 = data_manager.add_brand("Bonsai", brands_path)
        assert b2 == "Bonsai"

        # Verify load
        loaded = data_manager.load_brands(brands_path)
        assert loaded == ["HubSpot", "Bonsai"]

        # Rejects duplicate brand (case-insensitive, trimmed)
        with pytest.raises(ValueError, match="already exists"):
            data_manager.add_brand("  hubspot  ", brands_path)

        # Rejects empty brand name
        with pytest.raises(ValueError, match="cannot be empty"):
            data_manager.add_brand("   ", brands_path)


def test_data_manager_empty_and_corrupted_files():
    """Verifies graceful handling of empty and corrupted JSON files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_path = Path(temp_dir) / "corrupted_prompts.json"

        # Write invalid JSON
        prompts_path.write_text("invalid json content {", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted or invalid JSON"):
            data_manager.load_prompts(prompts_path)

        # Write empty content
        prompts_path.write_text("", encoding="utf-8")
        assert data_manager.load_prompts(prompts_path) == []


def test_data_manager_update_and_delete_prompt():
    """Verifies update_prompt and delete_prompt logic and validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_path = Path(temp_dir) / "test_prompts.json"

        p1 = data_manager.add_prompt("Original text 1", "research", prompts_path)
        p2 = data_manager.add_prompt("Original text 2", "comparison", prompts_path)

        # Update prompt 1
        updated = data_manager.update_prompt(p1["id"], "Updated text 1", "vendor_selection", prompts_path)
        assert updated["text"] == "Updated text 1"
        assert updated["stage"] == "vendor_selection"

        # Rejects conflict with existing prompt
        with pytest.raises(ValueError, match="already exists"):
            data_manager.update_prompt(p1["id"], "Original text 2", "research", prompts_path)

        # Rejects non-existent ID
        with pytest.raises(ValueError, match="not found"):
            data_manager.update_prompt(999, "Text", "research", prompts_path)

        # Delete prompt 1
        assert data_manager.delete_prompt(p1["id"], prompts_path) is True
        loaded = data_manager.load_prompts(prompts_path)
        assert len(loaded) == 1
        assert loaded[0]["id"] == p2["id"]

        # Deleting non-existent ID raises ValueError
        with pytest.raises(ValueError, match="not found"):
            data_manager.delete_prompt(p1["id"], prompts_path)


def test_data_manager_update_and_delete_brand():
    """Verifies update_brand and delete_brand logic and validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        brands_path = Path(temp_dir) / "test_brands.json"

        data_manager.add_brand("HubSpot", brands_path)
        data_manager.add_brand("Zoho CRM", brands_path)

        # Rename brand
        renamed = data_manager.update_brand("HubSpot", "HubSpot Pro", brands_path)
        assert renamed == "HubSpot Pro"

        # Rejects duplicate name
        with pytest.raises(ValueError, match="already exists"):
            data_manager.update_brand("HubSpot Pro", "Zoho CRM", brands_path)

        # Rejects non-existent brand
        with pytest.raises(ValueError, match="not found"):
            data_manager.update_brand("Salesforce", "Salesforce CRM", brands_path)

        # Delete brand
        assert data_manager.delete_brand("HubSpot Pro", brands_path) is True
        loaded = data_manager.load_brands(brands_path)
        assert loaded == ["Zoho CRM"]

        # Deleting non-existent brand raises ValueError
        with pytest.raises(ValueError, match="not found"):
            data_manager.delete_brand("HubSpot Pro", brands_path)


def test_data_manager_db_helpers(temp_db):
    """Verifies prompt_has_responses and brand_has_mentions database helpers."""
    # Initially no responses or mentions
    assert data_manager.prompt_has_responses(1, db_path=temp_db) is False
    assert data_manager.brand_has_mentions("HubSpot", db_path=temp_db) is False

    # Insert response and mention
    resp_id = db_manager.insert_response(prompt_id=1, platform="gemini", raw_text="HubSpot is good.", db_path=temp_db)
    db_manager.insert_mention(response_id=resp_id, brand_name="HubSpot", position=1, sentiment="recommended", db_path=temp_db)

    # Verify helpers return True
    assert data_manager.prompt_has_responses(1, db_path=temp_db) is True
    assert data_manager.brand_has_mentions("HubSpot", db_path=temp_db) is True
    assert data_manager.brand_has_mentions("hubspot", db_path=temp_db) is True
    assert data_manager.prompt_has_responses(99, db_path=temp_db) is False
    assert data_manager.brand_has_mentions("UnknownBrand", db_path=temp_db) is False


def test_data_manager_delete_all():
    """Verifies delete_all_prompts and delete_all_brands logic."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_path = Path(temp_dir) / "test_prompts.json"
        brands_path = Path(temp_dir) / "test_brands.json"

        data_manager.add_prompt("Test prompt 1", "research", prompts_path)
        data_manager.add_prompt("Test prompt 2", "comparison", prompts_path)
        assert len(data_manager.load_prompts(prompts_path)) == 2

        data_manager.delete_all_prompts(prompts_path)
        assert data_manager.load_prompts(prompts_path) == []

        data_manager.add_brand("Brand A", brands_path)
        data_manager.add_brand("Brand B", brands_path)
        assert len(data_manager.load_brands(brands_path)) == 2

        data_manager.delete_all_brands(brands_path)
        assert data_manager.load_brands(brands_path) == []