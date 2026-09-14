import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

POSITIVE_KEYWORDS = {
    "best", "recommend", "recommended", "top choice", "top", "excellent", "ideal",
    "favorite", "favourite", "great", "standout", "highly rated", "leading", "popular",
    "perfect", "robust", "powerful", "user-friendly", "intuitive", "effective"
}

NEGATIVE_KEYWORDS = {
    "avoid", "not ideal", "expensive", "steep", "clunky", "outdated", "complex",
    "lacks", "poor", "drawback", "limited", "hard to use", "frustrating", "confusing",
    "buggy", "overpriced", "slow", "restrictive"
}


def extract_domain(url_or_text: str) -> str:
    """Extracts clean domain name from a URL or text string."""
    if not url_or_text:
        return ""
    # Ensure scheme for urlparse if starting with www or domain pattern
    url_to_parse = url_or_text
    if not url_to_parse.startswith("http://") and not url_to_parse.startswith("https://"):
        url_to_parse = "https://" + url_to_parse

    parsed = urlparse(url_to_parse)
    domain = parsed.netloc or parsed.path
    # Strip www.
    if domain.startswith("www."):
        domain = domain[4:]
    # Strip trailing path/port
    domain = domain.split(":")[0].split("/")[0].lower()
    return domain


def extract_citations(text: str) -> List[Dict[str, Optional[str]]]:
    """Fallback: extracts explicit URLs and cited domains from raw text via regex.

    This is ONLY reliable when the model happens to print a real URL in its
    text. Prefer citations from Gemini's grounding_metadata (see
    normalize_grounding_citations) whenever they are available — this
    regex pass exists purely for ungrounded runs or edge cases where
    grounding returns nothing for a given prompt.
    """
    if not text:
        return []

    citations = []
    seen_urls = set()

    # 1. Regex for HTTP/HTTPS URLs
    url_pattern = r'https?://[^\s()<>"]+'
    urls = re.findall(url_pattern, text)

    for url in urls:
        # Clean trailing punctuation
        cleaned_url = url.rstrip(".,;:!)]}")
        if cleaned_url not in seen_urls:
            seen_urls.add(cleaned_url)
            domain = extract_domain(cleaned_url)
            if domain:
                citations.append({
                    "source_url": cleaned_url,
                    "source_domain": domain
                })

    # 2. Extract markdown link domains if not captured by plain URL regex
    md_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    md_matches = re.findall(md_pattern, text)
    for text_label, link in md_matches:
        if link.startswith("http") and link not in seen_urls:
            seen_urls.add(link)
            domain = extract_domain(link)
            if domain:
                citations.append({
                    "source_url": link,
                    "source_domain": domain
                })

    return citations


def normalize_grounding_citations(
    grounding_citations: List[Dict[str, Any]]
) -> List[Dict[str, Optional[str]]]:
    """Converts query_engine's grounding citations into DB-ready citation records.

    Args:
        grounding_citations: list of {"url": str|None, "title": str|None}
            as returned by query_engine._extract_grounding_citations.

    Returns:
        list of {"source_url": str|None, "source_domain": str}
    """
    normalized = []
    seen = set()

    for item in grounding_citations or []:
        url = item.get("url")
        title = item.get("title")

        if url:
            domain = extract_domain(url)
            key = url
        elif title:
            # Grounding sometimes returns a source title without a resolvable
            # URI. Fall back to the title itself so the source isn't dropped
            # entirely from the citations table.
            domain = title.strip().lower()
            key = f"title::{domain}"
        else:
            continue

        if not domain or key in seen:
            continue
        seen.add(key)

        normalized.append({
            "source_url": url,
            "source_domain": domain
        })

    return normalized


def classify_sentiment(context_window: str) -> str:
    """Classifies sentiment of a brand mention within its contextual text window."""
    text_lower = context_window.lower()

    pos_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    if pos_score > neg_score:
        return "recommended"
    elif neg_score > pos_score:
        return "negative"
    else:
        return "neutral"


def parse_response(
    raw_text: str,
    brand_list: List[str],
    grounding_citations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Parses raw text response to extract brand mentions, positions, sentiment, and citations.

    Args:
        raw_text: Full response string from Gemini/LLM.
        brand_list: List of seed brand names (e.g., ["HubSpot", "Zoho CRM", ...])
        grounding_citations: Optional structured citations from query_engine
            (real sources the model actually consulted, via Google Search
            grounding). When provided and non-empty, these are used instead
            of regex-scraping the raw text, since they're trustworthy and
            the text-scraped ones aren't.

    Returns:
        dict: {
            "mentions": [{"brand_name": str, "position": int, "sentiment": str}],
            "citations": [{"source_url": str, "source_domain": str}]
        }
    """
    if not raw_text:
        return {"mentions": [], "citations": []}

    detected_mentions = []
    brand_positions = []

    # Find first index of appearance for each brand
    text_lower = raw_text.lower()
    for brand in brand_list:
        brand_lower = brand.lower()

        # Word boundary or flexible substring regex
        # For multi-word brands like "Zoho CRM", handle gracefully
        pattern = r'\b' + re.escape(brand_lower) + r'\b'
        match = re.search(pattern, text_lower)

        if match:
            first_idx = match.start()
            brand_positions.append((first_idx, brand))

    # Sort brands by appearance order in the text
    brand_positions.sort(key=lambda x: x[0])

    # Assign position order (1-indexed) and evaluate sentiment
    for rank, (idx, brand) in enumerate(brand_positions, start=1):
        # Extract sentence-focused context window (approx sentence boundary or 80 chars)
        sub_before = raw_text[max(0, idx - 100):idx]
        period_idx = max(sub_before.rfind('.'), sub_before.rfind('\n'))
        if period_idx != -1:
            start_pos = idx - len(sub_before) + period_idx + 1
        else:
            start_pos = max(0, idx - 80)

        sub_after = raw_text[idx + len(brand):idx + len(brand) + 100]
        period_idx_after = min([p for p in [sub_after.find('.'), sub_after.find('\n')] if p != -1], default=-1)
        if period_idx_after != -1:
            end_pos = idx + len(brand) + period_idx_after + 1
        else:
            end_pos = min(len(raw_text), idx + len(brand) + 80)

        context_window = raw_text[start_pos:end_pos]

        sentiment = classify_sentiment(context_window)

        detected_mentions.append({
            "brand_name": brand,
            "position": rank,
            "sentiment": sentiment
        })

    # Citations: prefer real grounding sources; fall back to regex over text
    if grounding_citations:
        citations = normalize_grounding_citations(grounding_citations)
    else:
        citations = extract_citations(raw_text)

    return {
        "mentions": detected_mentions,
        "citations": citations
    }