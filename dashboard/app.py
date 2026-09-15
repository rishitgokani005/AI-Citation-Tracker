import sys
from pathlib import Path

# Add project root and src/ directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.express as px
import pandas as pd

try:
    import config
    import analysis
    import db_manager
    import data_manager
    import run_pipeline
except ImportError:
    from src import config, analysis, db_manager, data_manager, run_pipeline

# Ensure the SQLite schema exists before any query touches it.
# On a fresh deploy (e.g. Streamlit Cloud) the .db file may not exist yet,
# or may exist without tables — db_manager.get_connection() will happily
# create an empty file, but only init_db() actually runs schema.sql.
# Without this, the very first analytics query fails with
# "no such table: responses" (surfaced by Streamlit as a redacted
# pandas.errors.DatabaseError).
try:
    db_manager.init_db()
except Exception as init_err:
    st.error(f"Failed to initialize database schema: {init_err}")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="AI Citation Tracker Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎯 AI Citation Tracker Dashboard")

# Main Dashboard Navigation Tabs
tab_analytics, tab_manage = st.tabs(["📊 Analytics & Insights", "⚙️ Manage & Run Pipeline"])

with tab_analytics:
    # Filter & Controls
    fcol1, fcol2 = st.columns([3, 1])
    with fcol1:
        stages = ["All"] + data_manager.VALID_STAGES
        selected_stage = st.selectbox("Select Funnel Stage", stages, key="analytics_stage_filter")
    with fcol2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", key="refresh_analytics_btn", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Load Data Metrics
    stage_arg = None if selected_stage == "All" else selected_stage
    platform_arg = None

    try:
        df_share = analysis.get_mention_share(stage_filter=stage_arg, platform_filter=platform_arg)
        df_pos = analysis.get_average_position(stage_filter=stage_arg, platform_filter=platform_arg)
        df_sources = analysis.get_citation_domains(stage_filter=stage_arg, platform_filter=platform_arg)
        insights = analysis.generate_insights(stage_filter=stage_arg, platform_filter=platform_arg)
    except Exception as data_err:
        st.error(
            "Could not load analytics data from the database. "
            "This usually means the pipeline hasn't been run yet on this "
            "deployment. Go to the ⚙️ Manage & Run Pipeline tab and click "
            "'Run Pipeline Now' to populate it."
        )
        st.caption(f"Details: {data_err}")
        st.stop()

    # High Level KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_mentions = df_share["mention_count"].sum() if not df_share.empty else 0
        st.metric("Total Mentions", total_mentions)

    with col2:
        top_brand = df_share.iloc[0]["brand_name"] if not df_share.empty and total_mentions > 0 else "N/A"
        st.metric("Top Recommended Brand", top_brand)

    with col3:
        top_share = f"{df_share.iloc[0]['mention_share_pct']}%" if not df_share.empty and total_mentions > 0 else "0%"
        st.metric("Top Brand Share", top_share)

    with col4:
        total_sources = len(df_sources) if not df_sources.empty else 0
        st.metric("Cited Sources Count", total_sources)

    st.markdown("---")

    # Visualizations Row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Brand Mention Share (%)")
        if not df_share.empty and df_share["mention_count"].sum() > 0:
            fig_share = px.bar(
                df_share,
                x="brand_name",
                y="mention_share_pct",
                color="mention_share_pct",
                color_continuous_scale="Blues",
                labels={"brand_name": "Brand", "mention_share_pct": "Mention Share (%)"},
                title="Percentage of Prompts Mentioning Brand"
            )
            fig_share.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Share (%)")
            st.plotly_chart(fig_share, use_container_width=True)
        else:
            st.info("No mention data available for the selected filters. Please run the pipeline first.")

    with col_right:
        st.subheader("🥇 Average Rank / Position (Lower = Better)")
        if not df_pos.empty:
            fig_pos = px.bar(
                df_pos,
                x="brand_name",
                y="avg_position",
                color="avg_position",
                color_continuous_scale="Viridis_r",
                labels={"brand_name": "Brand", "avg_position": "Avg Rank Position"},
                title="Mean Placement Order in LLM Responses"
            )
            fig_pos.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Average Rank Position")
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("No position data available for the selected filters.")

    # Detailed Data Tables
    st.markdown("---")
    col_tbl1, col_tbl2 = st.columns(2)

    with col_tbl1:
        st.subheader("📋 Top Referenced Domains & Sources")
        if not df_sources.empty:
            st.dataframe(df_sources, use_container_width=True)
        else:
            st.info("No domain citations extracted yet.")

    with col_tbl2:
        st.subheader("📊 Sentiment Breakdown per Brand")
        df_sent = analysis.get_sentiment_breakdown(stage_filter=stage_arg, platform_filter=platform_arg)
        if not df_sent.empty:
            st.dataframe(df_sent, use_container_width=True)
        else:
            st.info("No sentiment data available.")

    # Automated Insights Section
    st.markdown("---")
    st.subheader("💡 Key GEO Automated Insights")
    for insight in insights:
        st.markdown(f"- {insight}")

with tab_manage:
    st.header("⚙️ Control Panel: Manage & Run Pipeline")
    st.caption("Add, edit, or delete target prompts and brands, and trigger a live pipeline run directly from your browser.")

    # Section 1: In-App Pipeline Control Panel
    st.subheader("🚀 Live Data Pipeline Runner")

    is_running = st.session_state.get("is_running", False)

    run_clicked = st.button("▶️ Run Pipeline Now", type="primary", disabled=is_running)

    if run_clicked:
        st.session_state["is_running"] = True
        with st.spinner("Querying Google Gemini API across prompts... Please wait."):
            try:
                stats = run_pipeline.run_pipeline(platform="gemini", force=True)
                st.session_state["last_run_stats"] = stats
                st.success("🎉 Pipeline execution completed successfully!")
            except Exception as err:
                st.error(f"Pipeline error: {err}")
            finally:
                st.session_state["is_running"] = False

    # Display Last Run Stats if available
    if "last_run_stats" in st.session_state:
        st_stats = st.session_state["last_run_stats"]
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Total Prompts", st_stats.get("total_prompts", 0))
        mcol2.metric("Processed (New)", st_stats.get("processed", 0))
        mcol3.metric("Skipped (Cached)", st_stats.get("skipped", 0))
        mcol4.metric("Mentions Found", st_stats.get("total_mentions", 0))
        mcol5.metric("Citations Extracted", st_stats.get("total_citations", 0))

    st.markdown("---")

    # Section 2: Full CRUD for Prompts & Brands
    col_mgmt_prompts, col_mgmt_brands = st.columns(2)

    with col_mgmt_prompts:
        p_hdr1, p_hdr2 = st.columns([3, 2])
        with p_hdr1:
            st.subheader("📋 Prompts Management")
        with p_hdr2:
            with st.popover("🗑️ Delete All Prompts"):
                st.warning("⚠️ Are you sure you want to delete ALL prompts? This action cannot be undone!")
                if st.button("Confirm Delete All Prompts", key="confirm_del_all_prompts", type="primary"):
                    try:
                        data_manager.delete_all_prompts()
                        st.success("All prompts deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        # Add Prompt Expander
        with st.expander("➕ Add New Prompt", expanded=False):
            with st.form("add_prompt_form_crud", clear_on_submit=True):
                new_p_text = st.text_input("Prompt Text *", placeholder="e.g. Best CRM software for freelancers")
                new_p_stage = st.selectbox("Funnel Stage *", data_manager.VALID_STAGES)
                sub_p = st.form_submit_button("Submit New Prompt")
                if sub_p:
                    try:
                        added_p = data_manager.add_prompt(new_p_text, new_p_stage)
                        st.success(f"Added prompt ID #{added_p['id']}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        # Prompts List & Actions
        try:
            prompts_list = data_manager.load_prompts()
            st.markdown(f"**Total Prompts ({len(prompts_list)}):**")
            for p in prompts_list:
                p_id = p.get("id")
                p_text = p.get("text", "")
                p_stage = p.get("stage", "unknown")

                row_col1, row_col2, row_col3 = st.columns([5, 1, 1])
                with row_col1:
                    st.write(f"**#{p_id}** `[{p_stage}]` {p_text}")

                with row_col2:
                    with st.popover("✏️ Edit"):
                        with st.form(f"edit_prompt_form_{p_id}"):
                            edit_text = st.text_input("Prompt Text", value=p_text)
                            edit_stage_idx = data_manager.VALID_STAGES.index(p_stage) if p_stage in data_manager.VALID_STAGES else 0
                            edit_stage = st.selectbox("Stage", data_manager.VALID_STAGES, index=edit_stage_idx)
                            if st.form_submit_button("Save Prompt"):
                                try:
                                    data_manager.update_prompt(p_id, edit_text, edit_stage)
                                    st.success(f"Prompt #{p_id} updated!")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))

                with row_col3:
                    with st.popover("🗑️ Delete"):
                        has_db_records = data_manager.prompt_has_responses(p_id)
                        if has_db_records:
                            st.warning(f"⚠️ Historical database responses exist for prompt #{p_id}. Deleting it removes it from future runs, but past records in SQLite will be preserved.")
                        st.write("Are you sure you want to delete this prompt?")
                        if st.button(f"Confirm Delete #{p_id}", key=f"del_p_{p_id}", type="primary"):
                            try:
                                data_manager.delete_prompt(p_id)
                                st.success(f"Prompt #{p_id} deleted!")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                st.divider()
        except Exception as err:
            st.error(f"Error loading prompts: {err}")

    with col_mgmt_brands:
        b_hdr1, b_hdr2 = st.columns([3, 2])
        with b_hdr1:
            st.subheader("🏷️ Target Brands Management")
        with b_hdr2:
            with st.popover("🗑️ Delete All Brands"):
                st.warning("⚠️ Are you sure you want to delete ALL brands? This action cannot be undone!")
                if st.button("Confirm Delete All Brands", key="confirm_del_all_brands", type="primary"):
                    try:
                        data_manager.delete_all_brands()
                        st.success("All brands deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        # Add Brand Expander
        with st.expander("➕ Add New Brand", expanded=False):
            with st.form("add_brand_form_crud", clear_on_submit=True):
                new_b_name = st.text_input("Brand Name *", placeholder="e.g. HubSpot")
                sub_b = st.form_submit_button("Submit New Brand")
                if sub_b:
                    try:
                        added_b = data_manager.add_brand(new_b_name)
                        st.success(f"Added brand '{added_b}'!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        # Brands List & Actions
        try:
            brands_list = data_manager.load_brands()
            st.markdown(f"**Total Brands ({len(brands_list)}):**")
            for b_name in brands_list:
                brow_col1, brow_col2, brow_col3 = st.columns([5, 1, 1])
                with brow_col1:
                    st.write(f"🏷️ **{b_name}**")

                with brow_col2:
                    with st.popover("✏️ Edit"):
                        st.caption("Note: Renaming a brand will NOT rewrite past historical database mention records.")
                        with st.form(f"edit_brand_form_{b_name}"):
                            renamed_brand = st.text_input("Brand Name", value=b_name)
                            if st.form_submit_button("Save Brand"):
                                try:
                                    data_manager.update_brand(b_name, renamed_brand)
                                    st.success(f"Brand renamed to '{renamed_brand}'!")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))

                with brow_col3:
                    with st.popover("🗑️ Delete"):
                        has_brand_history = data_manager.brand_has_mentions(b_name)
                        if has_brand_history:
                            st.warning(f"⚠️ Historical database mentions exist for '{b_name}'. Deleting it removes it from future runs, but past records in SQLite will be preserved.")
                        st.write(f"Delete brand '{b_name}'?")
                        if st.button(f"Confirm Delete", key=f"del_b_{b_name}", type="primary"):
                            try:
                                data_manager.delete_brand(b_name)
                                st.success(f"Brand '{b_name}' deleted!")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                st.divider()
        except Exception as err:
            st.error(f"Error loading brands: {err}")