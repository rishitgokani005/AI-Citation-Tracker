import sys
from pathlib import Path

# Add project root directory to sys.path so config can be imported from src/ scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import sqlite3
from typing import Dict, Any, List, Optional
import pandas as pd
import config
import db_manager


TupleData = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def load_raw_data(db_path=None) -> TupleData:
    """Loads all tables from SQLite into Pandas DataFrames and joins prompt stage data."""
    conn = db_manager.get_connection(db_path)

    # Load responses table
    df_responses = pd.read_sql_query("SELECT * FROM responses", conn)

    # Load prompts.json to map prompt_id to stage and prompt text
    with open(config.PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    df_prompts = pd.DataFrame(prompts)

    if not df_responses.empty and not df_prompts.empty:
        df_responses = df_responses.merge(
            df_prompts[["id", "text", "stage"]],
            left_on="prompt_id",
            right_on="id",
            how="left"
        )
        if "id_x" in df_responses.columns:
            df_responses = df_responses.rename(columns={"id_x": "id"}).drop(columns=["id_y"], errors="ignore")
    else:
        df_responses["text"] = ""
        df_responses["stage"] = "unknown"

    # Load mentions table
    df_mentions = pd.read_sql_query("SELECT * FROM mentions", conn)

    # Load citations table
    df_citations = pd.read_sql_query("SELECT * FROM citations", conn)

    conn.close()
    return df_responses, df_mentions, df_citations



def get_filtered_data(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> TupleData:
    """Helper to fetch and filter DataFrames by funnel stage or platform."""
    df_responses, df_mentions, df_citations = load_raw_data(db_path)

    if df_responses.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Filter responses
    if platform_filter and platform_filter != "All":
        df_responses = df_responses[df_responses["platform"] == platform_filter]

    if stage_filter and stage_filter != "All":
        df_responses = df_responses[df_responses["stage"] == stage_filter]

    valid_response_ids = set(df_responses["id"])

    # Filter mentions and citations based on filtered response_ids
    if not df_mentions.empty:
        df_mentions = df_mentions[df_mentions["response_id"].isin(valid_response_ids)]
        df_mentions = df_mentions.merge(
            df_responses[["id", "stage", "platform"]],
            left_on="response_id",
            right_on="id",
            how="left"
        )

    if not df_citations.empty:
        df_citations = df_citations[df_citations["response_id"].isin(valid_response_ids)]
        df_citations = df_citations.merge(
            df_responses[["id", "stage", "platform"]],
            left_on="response_id",
            right_on="id",
            how="left"
        )

    return df_responses, df_mentions, df_citations


def get_mention_share(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> pd.DataFrame:
    """Computes brand mention share (% of responses containing each brand)."""
    df_responses, df_mentions, _ = get_filtered_data(stage_filter, platform_filter, db_path)

    total_prompts = len(df_responses["prompt_id"].unique()) if not df_responses.empty else 0
    if total_prompts == 0 or df_mentions.empty:
        return pd.DataFrame(columns=["brand_name", "mention_count", "total_prompts", "mention_share_pct"])

    # Load all seed brands to include 0% visibility brands, plus any brands present in mentions
    all_brands = []
    if Path(config.BRANDS_PATH).exists():
        try:
            with open(config.BRANDS_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_brands = json.loads(content)
        except Exception:
            all_brands = []

    mentioned_brands = list(df_mentions["brand_name"].unique()) if not df_mentions.empty else []
    combined_brands = list(dict.fromkeys([b for b in all_brands if isinstance(b, str)] + mentioned_brands))

    if not combined_brands:
        return pd.DataFrame(columns=["brand_name", "mention_count", "total_prompts", "mention_share_pct"])

    # Count distinct responses per brand
    brand_counts = (
        df_mentions.groupby("brand_name")["response_id"]
        .nunique()
        .reset_index()
        .rename(columns={"response_id": "mention_count"})
    )

    # Ensure all seed/mentioned brands exist in final output
    df_all_brands = pd.DataFrame({"brand_name": combined_brands})
    df_share = df_all_brands.merge(brand_counts, on="brand_name", how="left").fillna(0)

    df_share["mention_count"] = df_share["mention_count"].astype(int)
    df_share["total_prompts"] = total_prompts
    df_share["mention_share_pct"] = (df_share["mention_count"] / total_prompts * 100).round(1)

    return df_share.sort_values(by="mention_share_pct", ascending=False).reset_index(drop=True)


def get_average_position(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> pd.DataFrame:
    """Computes mean appearance position rank per brand (lower position = mentioned earlier)."""
    _, df_mentions, _ = get_filtered_data(stage_filter, platform_filter, db_path)

    if df_mentions.empty:
        return pd.DataFrame(columns=["brand_name", "avg_position", "mention_count"])

    avg_pos = (
        df_mentions.groupby("brand_name")
        .agg(
            avg_position=("position", "mean"),
            mention_count=("position", "count")
        )
        .reset_index()
    )

    avg_pos["avg_position"] = avg_pos["avg_position"].round(2)
    return avg_pos.sort_values(by="avg_position", ascending=True).reset_index(drop=True)


def get_sentiment_breakdown(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> pd.DataFrame:
    """Computes recommended, neutral, and negative mention counts per brand."""
    _, df_mentions, _ = get_filtered_data(stage_filter, platform_filter, db_path)

    if df_mentions.empty:
        return pd.DataFrame(columns=["brand_name", "recommended", "neutral", "negative"])

    sentiment_counts = (
        df_mentions.groupby(["brand_name", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["recommended", "neutral", "negative"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0

    return sentiment_counts[["brand_name", "recommended", "neutral", "negative"]]


def get_citation_domains(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> pd.DataFrame:
    """Computes domain citation frequencies."""
    _, _, df_citations = get_filtered_data(stage_filter, platform_filter, db_path)

    if df_citations.empty:
        return pd.DataFrame(columns=["source_domain", "citation_count"])

    domain_counts = (
        df_citations.groupby("source_domain")
        .size()
        .reset_index(name="citation_count")
        .sort_values(by="citation_count", ascending=False)
        .reset_index(drop=True)
    )

    return domain_counts


def get_stage_breakdown(
    platform_filter: Optional[str] = None,
    db_path=None
) -> pd.DataFrame:
    """Computes brand mention share across each funnel stage."""
    df_responses, df_mentions, _ = get_filtered_data(stage_filter=None, platform_filter=platform_filter, db_path=db_path)

    if df_responses.empty or df_mentions.empty:
        return pd.DataFrame()

    with open(config.BRANDS_PATH, "r", encoding="utf-8") as f:
        all_brands = json.load(f)

    stages = df_responses["stage"].unique()
    records = []

    for stage in stages:
        resp_stage = df_responses[df_responses["stage"] == stage]
        total_stage_prompts = len(resp_stage["prompt_id"].unique())
        valid_ids = set(resp_stage["id"])

        mentions_stage = df_mentions[df_mentions["response_id"].isin(valid_ids)]

        for brand in all_brands:
            m_cnt = mentions_stage[mentions_stage["brand_name"] == brand]["response_id"].nunique()
            pct = round(m_cnt / total_stage_prompts * 100, 1) if total_stage_prompts > 0 else 0
            records.append({
                "stage": stage,
                "brand_name": brand,
                "mention_share_pct": pct
            })

    return pd.DataFrame(records)


def generate_insights(
    stage_filter: Optional[str] = None,
    platform_filter: Optional[str] = None,
    db_path=None
) -> List[str]:
    """Generates dynamic analytical insight strings based on stored data."""
    df_share = get_mention_share(stage_filter, platform_filter, db_path)
    df_pos = get_average_position(stage_filter, platform_filter, db_path)
    df_citations = get_citation_domains(stage_filter, platform_filter, db_path)
    df_stage = get_stage_breakdown(platform_filter, db_path)

    insights = []

    if df_share.empty or df_share["mention_count"].sum() == 0:
        return ["No brand mention data available yet. Run the pipeline (`python src/run_pipeline.py`) to generate insights."]

    # 1. Top Brand Mention Leader
    top_brand = df_share.iloc[0]
    insights.append(
        f"**Brand Visibility Leader**: **{top_brand['brand_name']}** leads visibility with a **{top_brand['mention_share_pct']}% mention share** across queried prompts."
    )

    # 2. Average Ranking position
    if not df_pos.empty:
        top_pos = df_pos.iloc[0]
        insights.append(
            f"**Prime Recommendation Rank**: **{top_pos['brand_name']}** earns the highest average placement position (**#{top_pos['avg_position']}**), appearing earliest in generated recommendations."
        )

    # 3. Citation Domain Authority
    if not df_citations.empty:
        top_domain = df_citations.iloc[0]
        insights.append(
            f"**Dominant Citation Source**: Gemini relies heavily on **{top_domain['source_domain']}** ({top_domain['citation_count']} citations) as an authority reference for this niche."
        )
    else:
        insights.append(
            "**Citation Profile**: Gemini responses primarily rely on parametric memory without explicit external markdown URL citations."
        )

    # 4. Stage Gap Analysis
    if not df_stage.empty:
        # Find brand with biggest variance or high recommendation stage presence
        rec_stage = df_stage[df_stage["stage"] == "recommendation"]
        if not rec_stage.empty:
            top_rec = rec_stage.sort_values(by="mention_share_pct", ascending=False).iloc[0]
            insights.append(
                f"**Funnel Stage Dynamics**: In the high-intent *Recommendation* stage, **{top_rec['brand_name']}** captures **{top_rec['mention_share_pct']}% share** of voice."
            )

    return insights
