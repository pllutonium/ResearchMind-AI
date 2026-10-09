"""
Module: app.py
Description: Publication-Grade Minimalist Dashboard for ResearchMind AI Multi-Agent RAG System.
Features: 100% English UI, authentic academic terminal aesthetic, dynamic sequential
stage lighting effect across all 5 specialized agents, and publication-ready PDF export.
"""
import sys
import os
from pathlib import Path
import time
import pandas as pd
import streamlit as st

# Append project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_PDFS_DIR
from src.vectorstore.chroma_store import VectorStore
from src.llm.client import OllamaClient
from src.agents.research_agents import AICoordinator
from src.ingestion.pipeline import ingest_document
from src.db.metadata_store import list_all_papers
from src.reporting.pdf_export import generate_synthesis_pdf
from src.embeddings.embedder import warm_up_embedder

# Streamlit Page Setup
st.set_page_config(
    page_title="ResearchMind AI — MVP Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Editorial Academic Terminal UI & Dynamic Sequential Lighting
custom_css = """
<style>
/* Dark Academic Terminal Background */
body, .stApp {
    background-color: #0e0e0c !important;
    color: #f5f5f0 !important;
    font-family: Georgia, 'Times New Roman', serif !important;
}
header[data-testid="stHeader"] {
    background-color: #0e0e0c !important;
}
.stDeployButton {
    display: none !important;
}
section[data-testid="stSidebar"] {
    background-color: #121210 !important;
    border-right: 1px solid #22221e !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Top Header Typography */
.app-title-serif {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #f6f6f0;
    letter-spacing: -0.01em;
}
.app-sub-mono {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.88rem;
    color: #888880;
}
.app-meta-mono {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.74rem;
    color: #888880;
    text-align: right;
    line-height: 1.5;
}
.intro-paragraph {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 0.95rem;
    color: #c4c4bc;
    line-height: 1.6;
    margin: 14px 0 20px 0;
}

/* 5-Agent Pipeline Horizontal Container */
.pipeline-container {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    border: 1px solid #2a2a24;
    border-radius: 8px;
    background: #11110f;
    overflow: hidden;
    margin: 16px 0 6px 0;
    width: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.pipeline-card {
    padding: 16px 14px;
    border-right: 1px solid #24241f;
    background: #11110f;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 148px;
}
.pipeline-card:last-child {
    border-right: none;
}

/* 1. ACTIVE / ILLUMINATED STATE (MAKES LIGHT UNTIL IT ENDS!) */
@keyframes pulse-card-glow {
    0% {
        box-shadow: inset 0 0 16px rgba(45, 212, 191, 0.2), 0 0 12px rgba(45, 212, 191, 0.3);
        background: #111f19;
    }
    50% {
        box-shadow: inset 0 0 32px rgba(45, 212, 191, 0.45), 0 0 26px rgba(45, 212, 191, 0.65);
        background: #142820;
    }
    100% {
        box-shadow: inset 0 0 16px rgba(45, 212, 191, 0.2), 0 0 12px rgba(45, 212, 191, 0.3);
        background: #111f19;
    }
}
.pipeline-card-active {
    animation: pulse-card-glow 1.4s infinite ease-in-out !important;
    border-top: 3px solid #2dd4bf !important;
    border-bottom: 1px solid #2dd4bf !important;
    z-index: 5;
}
.pipeline-card-active .agent-title {
    color: #ffffff !important;
    text-shadow: 0 0 12px rgba(45, 212, 191, 0.7) !important;
}
.pipeline-card-active .agent-code {
    color: #2dd4bf !important;
    font-weight: 700 !important;
}

/* 2. COMPLETED STATE */
.pipeline-card-done {
    background: #0f1713 !important;
    border-top: 3px solid #10b981 !important;
}
.pipeline-card-done .agent-title {
    color: #f1f5f9 !important;
}
.pipeline-card-done .agent-code {
    color: #6ee7b7 !important;
}

/* 3. IDLE / WAITING STATE */
.pipeline-card-idle {
    background: #11110f !important;
    opacity: 0.65;
    border-top: 3px solid transparent !important;
}
.pipeline-card-idle .agent-title {
    color: #888880 !important;
}
.pipeline-card-idle .agent-code {
    color: #666660 !important;
}

/* Agent Card Typography */
.agent-code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.72rem;
    color: #888880;
    letter-spacing: 0.04em;
}
.agent-title {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 6px 0 6px 0;
    letter-spacing: -0.01em;
}
.agent-desc {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.72rem;
    color: #9c9c94;
    line-height: 1.35;
    min-height: 38px;
    margin: 0;
}

/* Pulsing & Status Dots */
@keyframes dot-ping {
    0% { transform: scale(0.9); opacity: 0.8; box-shadow: 0 0 4px #2dd4bf; }
    50% { transform: scale(1.35); opacity: 1; box-shadow: 0 0 14px #2dd4bf, 0 0 22px #2dd4bf; }
    100% { transform: scale(0.9); opacity: 0.8; box-shadow: 0 0 4px #2dd4bf; }
}
.agent-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.dot-active {
    background-color: #2dd4bf !important;
    animation: dot-ping 1.1s infinite alternate !important;
}
.dot-done {
    background-color: #10b981 !important;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.8) !important;
}
.dot-idle {
    background-color: #383832 !important;
}

.agent-status-tag {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.tag-active { color: #2dd4bf; text-shadow: 0 0 8px rgba(45, 212, 191, 0.6); }
.tag-done { color: #10b981; }
.tag-idle { color: #55554f; }

.stat-text-active {
    font-family: 'Consolas', monospace;
    font-size: 0.72rem;
    color: #2dd4bf;
    font-weight: 600;
}
.stat-text-done {
    font-family: 'Consolas', monospace;
    font-size: 0.72rem;
    color: #10b981;
}
.stat-text-idle {
    font-family: 'Consolas', monospace;
    font-size: 0.72rem;
    color: #55554f;
}

/* Academic Primary Button (Cream/Ivory Style matching reference image) */
button[kind="primary"] {
    background-color: #f5f5ee !important;
    color: #11110f !important;
    border: 1px solid #d4d4c8 !important;
    border-radius: 6px !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    background-color: #ffffff !important;
    color: #000000 !important;
    box-shadow: 0 0 16px rgba(245, 245, 238, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Structured Dimension Cards */
.dimension-card {
    border: 1px solid #24241f;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
    background: #11110f;
}
.dimension-title {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.citation-pill {
    display: inline-block;
    background: #1a1a16;
    color: #93c5fd;
    border: 1px solid #2e2e28;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: 'Consolas', monospace;
    font-weight: 500;
    margin-right: 4px;
}
</style>
"""
st.html(custom_css)

# Session State Initialization (Always defaults to Google Gemini Cloud)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "llm" not in st.session_state:
    st.session_state.llm = OllamaClient(backend="gemini")
if "coordinator" not in st.session_state:
    st.session_state.coordinator = AICoordinator(
        st.session_state.vector_store, st.session_state.llm
    )
if "embedder_warmed" not in st.session_state:
    warm_up_embedder()
    st.session_state.embedder_warmed = True


def ingest_uploaded_pdf(uploaded_file) -> int:
    """Saves uploaded PDF to disk and indexes it through the unified pipeline."""
    dest_path = Path(RAW_PDFS_DIR) / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    summary = ingest_document(dest_path)
    return summary.get("chunk_count", 0)


def render_pipeline_card_container(current_stage: int = 0, stage_status: str = "idle") -> str:
    """
    Renders the 5-Agent Pipeline horizontal container matching the academic MVP simulation layout.
    current_stage: 0 (idle), 1 (search), 2 (reader), 3 (comparison), 4 (gap), 5 (reviewer), 6 (complete)
    stage_status: 'idle', 'running', 'done', 'complete'
    """
    agents = [
        {
            "id": 1,
            "code": "01 · search",
            "title": "Literature search",
            "desc": "Semantic search across corpus embeddings",
            "stat": "evidence retrieved",
            "active_stat": "semantic search active...",
        },
        {
            "id": 2,
            "code": "02 · reader",
            "title": "Paper reader",
            "desc": "Extracts 6 dimensions & grounded citations",
            "stat": "dimensions parsed",
            "active_stat": "extracting dimensions...",
        },
        {
            "id": 3,
            "code": "03 · compare",
            "title": "Comparison",
            "desc": "Contrasts architectures & baseline metrics",
            "stat": "models compared",
            "active_stat": "contrasting baselines...",
        },
        {
            "id": 4,
            "code": "04 · gap",
            "title": "Research gap",
            "desc": "Finds underexplored combinations across corpus",
            "stat": "hypotheses surfaced",
            "active_stat": "surfacing gaps...",
        },
        {
            "id": 5,
            "code": "05 · review",
            "title": "Reviewer",
            "desc": "Evaluates proposal under NeurIPS/ICLR rubric",
            "stat": "scorecard generated",
            "active_stat": "evaluating rubric...",
        },
    ]

    cards_html = []
    for a in agents:
        idx = a["id"]
        is_active = (current_stage == idx and stage_status == "running")
        is_done = (
            stage_status == "complete"
            or current_stage > idx
            or (current_stage == idx and stage_status in ("done", "complete"))
        )

        if is_active:
            # ILLUMINATED / GLOWING STATE (MAKES LIGHT UNTIL IT ENDS!)
            card_class = "pipeline-card pipeline-card-active"
            dot_html = "<span class='agent-dot dot-active'></span>"
            status_tag = "<span class='agent-status-tag tag-active'>• EXECUTING</span>"
            stat_text = f"<span class='stat-text-active'>{a['active_stat']}</span>"
        elif is_done:
            # COMPLETED STATE
            card_class = "pipeline-card pipeline-card-done"
            dot_html = "<span class='agent-dot dot-done'></span>"
            status_tag = "<span class='agent-status-tag tag-done'>✓ COMPLETE</span>"
            stat_text = f"<span class='stat-text-done'>{a['stat']}</span>"
        else:
            # IDLE / WAITING STATE
            card_class = "pipeline-card pipeline-card-idle"
            dot_html = "<span class='agent-dot dot-idle'></span>"
            status_tag = "<span class='agent-status-tag tag-idle'>WAITING</span>"
            stat_text = f"<span class='stat-text-idle'>{a['stat']}</span>"

        card_html = (
            f"<div class='{card_class}'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>"
            f"<span class='agent-code'>{a['code']}</span>"
            f"<div style='display: flex; align-items: center; gap: 5px;'>{status_tag}{dot_html}</div>"
            f"</div>"
            f"<h4 class='agent-title'>{a['title']}</h4>"
            f"<p class='agent-desc'>{a['desc']}</p>"
            f"<div style='margin-top: 14px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06);'>{stat_text}</div>"
            f"</div>"
        )
        cards_html.append(card_html)

    joined_cards = "".join(cards_html)
    subnote = (
        "<div style='font-family: Consolas, monospace; font-size: 0.74rem; color: #71716a; margin-top: 8px; letter-spacing: 0.02em;'>"
        "each stage queries the same vector + metadata store, at a different scope — live execution"
        "</div>"
    )
    return f"<div class='pipeline-container'>{joined_cards}</div>{subnote}"


# ==============================================================================
# SIDEBAR: INFERENCE ENGINE & CORPUS MANAGEMENT
# ==============================================================================
with st.sidebar:
    st.markdown("### ResearchMind AI")
    st.caption("Autonomous Multi-Agent Academic Laboratory")

    docs_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ResearchMind_AI_Codebase_Documentation.pdf"))
    if os.path.exists(docs_pdf_path):
        with open(docs_pdf_path, "rb") as f:
            docs_pdf_bytes = f.read()
        st.download_button(
            label="Download Codebase Manual (PDF)",
            data=docs_pdf_bytes,
            file_name="ResearchMind_AI_Codebase_Documentation.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Inference Engine")

    backend_choice = st.selectbox(
        "Select Engine:",
        ["Google Gemini Cloud", "Local Ollama", "Groq Cloud"],
        index=0 if st.session_state.llm.backend == "gemini" else (1 if st.session_state.llm.backend == "ollama" else 2),
    )

    if backend_choice == "Google Gemini Cloud":
        st.session_state.llm.backend = "gemini"
        gemini_models = [
            "gemini-flash-lite-latest",
            "gemini-3.7-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
        ]
        cur_g_idx = (
            gemini_models.index(st.session_state.llm.gemini_model_name)
            if st.session_state.llm.gemini_model_name in gemini_models
            else 0
        )
        selected_model = st.selectbox("Model", gemini_models, index=cur_g_idx)
        st.session_state.llm.gemini_model_name = selected_model

    elif backend_choice == "Groq Cloud":
        st.session_state.llm.backend = "groq"
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        cur_groq_idx = (
            groq_models.index(st.session_state.llm.groq_model_name)
            if st.session_state.llm.groq_model_name in groq_models
            else 0
        )
        selected_model = st.selectbox("Model", groq_models, index=cur_groq_idx)
        st.session_state.llm.groq_model_name = selected_model

    else:
        st.session_state.llm.backend = "ollama"
        available_models = st.session_state.llm.list_available_models()
        if available_models:
            cur_idx = (
                available_models.index(st.session_state.llm.model_name)
                if st.session_state.llm.model_name in available_models
                else 0
            )
            selected_model = st.selectbox("Model", available_models, index=cur_idx)
            st.session_state.llm.model_name = selected_model

    st.markdown("---")
    st.subheader("Document Corpus Management")

    uploaded_file = st.file_uploader(
        "Upload Scientific PDF",
        type=["pdf"],
        help="PDF will be extracted with OCR density verification and vectorized into ChromaDB.",
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}' into pipeline..."):
                chunk_count = ingest_uploaded_pdf(uploaded_file)
            if chunk_count > 0:
                st.success(f"Indexed {chunk_count} semantic chunks successfully.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("No viable textual content could be extracted from this PDF.")

    st.markdown("---")
    st.subheader("Corpus Registry")

    indexed_paper_ids = st.session_state.vector_store.list_paper_ids()
    st.metric("Indexed Papers", len(indexed_paper_ids))

    if indexed_paper_ids:
        registered_meta = {p["paper_id"]: p for p in list_all_papers()}
        with st.expander("Cataloged Papers", expanded=True):
            for pid in indexed_paper_ids:
                meta = registered_meta.get(pid)
                if meta:
                    st.html(
                        f"<div><b>{pid}</b><br/>"
                        f"<span style='font-size: 0.8rem; color: #a1a1aa;'>Pages: {meta.get('page_count', 'N/A')}</span></div>"
                    )
                else:
                    st.markdown(f"**{pid}**")
    else:
        st.info("No papers indexed yet. Upload a PDF above to begin.")


# ==============================================================================
# HEADER: STYLED TO MATCH ACADEMIC TERMINAL SIMULATION
# ==============================================================================
col_header_left, col_header_right = st.columns([7, 5])
with col_header_left:
    st.html(
        "<div style='display: flex; align-items: baseline; gap: 8px; margin-top: 4px;'>"
        "<span class='app-title-serif'>ResearchMind AI</span>"
        "<span class='app-sub-mono'>/ MVP simulation</span>"
        "</div>"
    )

if st.session_state.llm.backend == "gemini":
    active_backend_label = "Google Gemini Cloud"
    active_model_label = st.session_state.llm.gemini_model_name
    active_chip_color = "#2dd4bf"
elif st.session_state.llm.backend == "groq":
    active_backend_label = "Groq Cloud"
    active_model_label = st.session_state.llm.groq_model_name
    active_chip_color = "#f59e0b"
else:
    active_backend_label = "Ollama Local"
    active_model_label = st.session_state.llm.model_name
    active_chip_color = "#10b981"

with col_header_right:
    st.html(
        f"<div class='app-meta-mono'>"
        f"five-agent pipeline • shared RAG substrate<br/>"
        f"live cloud substrate • <span style='color: {active_chip_color}; font-weight: 600;'>{active_model_label}</span>"
        f"</div>"
    )

st.html(
    "<p class='intro-paragraph'>"
    "This is an autonomous multi-agent research laboratory: select or enter a research target, "
    "and watch the five specialized agents — <b>Search, Reader, Comparison, Gap, Reviewer</b> — "
    "move through the pipeline, lighting up sequentially as each completes its academic deliverable."
    "</p>"
)

if not indexed_paper_ids:
    st.warning("No literature is currently indexed in the vector store. Upload a PDF in the sidebar to begin analysis.")


# ==============================================================================
# MAIN WORKSPACE: 6 TABS (ALL 100% ENGLISH, NO EMOJIS)
# ==============================================================================
tab_coord, tab_reader, tab_search, tab_comp, tab_gap, tab_rev = st.tabs(
    [
        "Autonomous Coordinator",
        "Paper Reader Matrix",
        "Literature Search",
        "Cross-Paper Comparison",
        "Research Gaps & Hypotheses",
        "Peer Review Scorecard",
    ]
)


# ------------------------------------------------------------------------------
# TAB 1: AUTONOMOUS COORDINATOR (SEQUENTIAL LIGHTING PIPELINE)
# ------------------------------------------------------------------------------
with tab_coord:
    col_input, col_btn = st.columns([9, 3])

    with col_input:
        selected_paper = (
            st.selectbox(
                "Select Target Paper:",
                indexed_paper_ids,
                key="coord_target_paper",
                label_visibility="collapsed",
            )
            if indexed_paper_ids
            else None
        )

    with col_btn:
        start_btn = st.button(
            "• Run pipeline",
            type="primary",
            disabled=not selected_paper,
            use_container_width=True,
        )

    st.html(
        "<div style=\"font-family: 'Consolas', monospace; font-size: 0.78rem; color: #2dd4bf; margin: 4px 0 14px 0;\">"
        "workflow selected by planner agent &rarr; Literature Review & Deep Multi-Agent Synthesis"
        "</div>"
    )

    # DYNAMIC PIPELINE BOX (LIGHTS UP AS EACH AGENT RUNS!)
    pipeline_placeholder = st.empty()
    pipeline_placeholder.html(
        render_pipeline_card_container(current_stage=0, stage_status="idle")
    )

    if start_btn:
        status_box = st.status(
            f"Executing 5-Agent Collaborative Pipeline on '{selected_paper}'...",
            expanded=True,
        )

        out_search = st.empty()
        out_reader = st.empty()
        out_comp = st.empty()
        out_gap = st.empty()
        out_rev = st.empty()
        out_download = st.empty()

        final_res = {}
        for event in st.session_state.coordinator.run_collaborative_synthesis_stream(paper_id=selected_paper):
            stage = event.get("stage", 0)
            status = event.get("status")

            if status == "error":
                st.error(event.get("error"))
                break

            elif status == "running":
                agent_name = event.get("agent", "")
                status_box.write(f"Stage {stage}/5 Running: {agent_name}...")
                # LIGHT UP CURRENT AGENT CARD!
                pipeline_placeholder.html(
                    render_pipeline_card_container(current_stage=stage, stage_status="running")
                )

            elif status == "done":
                # CURRENT AGENT CARD FINISHED
                pipeline_placeholder.html(
                    render_pipeline_card_container(current_stage=stage, stage_status="done")
                )
                key = event.get("key")
                val = event.get("result", {})
                final_res[key] = val

                if key == "search":
                    with out_search.container():
                        st.markdown("#### 1. Literature Search & Executive Overview")
                        st.info(val.get("summary", "No summary produced."))
                elif key == "reader":
                    with out_reader.container():
                        st.markdown("#### 2. Paper Reader Structured Schema")
                        table_dict = val.get("table", {})
                        col_dim1, col_dim2 = st.columns(2)
                        keys = list(table_dict.keys())
                        mid = len(keys) // 2
                        with col_dim1:
                            for k in keys[:mid]:
                                st.html(
                                    f"<div class='dimension-card'><div class='dimension-title' style='color: #60a5fa;'>{k}</div><div style='font-size: 0.85rem; line-height: 1.5;'>{table_dict[k]}</div></div>"
                                )
                        with col_dim2:
                            for k in keys[mid:]:
                                color = "#34d399" if "Future" in k else ("#fbbf24" if "Limitation" in k else "#60a5fa")
                                st.html(
                                    f"<div class='dimension-card'><div class='dimension-title' style='color: {color};'>{k}</div><div style='font-size: 0.85rem; line-height: 1.5;'>{table_dict[k]}</div></div>"
                                )
                        if val.get("citations"):
                            st.html(f"<b>Grounded Citations:</b> <span class='citation-pill'>{val.get('citations')}</span>")
                elif key == "comparison":
                    with out_comp.container():
                        st.markdown("#### 3. Cross-Paper & Baseline Comparison")
                        st.markdown(val.get("comparison", "No comparison generated."))
                elif key == "gap":
                    with out_gap.container():
                        st.markdown("#### 4. Discovered Research Gaps & Hypotheses")
                        st.markdown(val.get("analysis", "No gap analysis generated."))
                elif key == "reviewer":
                    with out_rev.container():
                        st.markdown("#### 5. Academic Conference Peer Review")
                        st.markdown(val.get("critique", "No review generated."))

            elif status == "complete":
                # ALL 5 AGENTS FINISHED (HARMONIOUS SUCCESS STATE)
                pipeline_placeholder.html(
                    render_pipeline_card_container(current_stage=6, stage_status="complete")
                )
                status_box.update(
                    label=f"All 5 Agents Completed Successfully for '{selected_paper}'.",
                    state="complete",
                    expanded=False,
                )

                # Render Deliverable Export Buttons
                search_data = final_res.get("search", {})
                reader_data = final_res.get("reader", {})
                table_dict = reader_data.get("table", {})
                comp_data = final_res.get("comparison", {})
                gap_data = final_res.get("gap", {})
                rev_data = final_res.get("reviewer", {})

                active_backend = getattr(st.session_state.llm, "backend", "ollama")
                if active_backend == "gemini":
                    engine_label = f"Google Gemini Cloud ({getattr(st.session_state.llm, 'gemini_model_name', 'gemini-flash-latest')})"
                elif active_backend == "groq":
                    engine_label = f"Groq Cloud ({getattr(st.session_state.llm, 'groq_model_name', 'llama-3.3-70b-versatile')})"
                else:
                    engine_label = f"Local Ollama ({getattr(st.session_state.llm, 'model_name', 'mistral')})"

                report_md = f"""# ResearchMind AI — Collaborative Synthesis Report
Target Document: {selected_paper}
Generated via: {engine_label}
Pipeline: 5-Agent Collaborative Synthesis (NeurIPS / ICLR Rubric)

## 1. Executive Summary
{search_data.get('summary', '')}

## 2. Structured Paper Reader Matrix
| Dimension | Findings |
| :--- | :--- |
""" + "\n".join(f"| **{k}** | {v} |" for k, v in table_dict.items()) + f"""

Citations: {reader_data.get('citations', 'N/A')}


## 3. Comparative Baseline Analysis
{comp_data.get('comparison', '')}

## 4. Discovered Research Gaps & Testable Hypotheses
{gap_data.get('analysis', '')}

## 5. Peer Review Critique & Conference Rubric
{rev_data.get('critique', '')}
"""
                with out_download.container():
                    st.markdown("---")
                    pdf_bytes = generate_synthesis_pdf(
                        paper_id=selected_paper,
                        engine_label=engine_label,
                        search_data=search_data,
                        reader_data=reader_data,
                        comp_data=comp_data,
                        gap_data=gap_data,
                        rev_data=rev_data,
                    )

                    col_dl1, col_dl2 = st.columns([1, 1])
                    with col_dl1:
                        st.download_button(
                            label="Download Synthesis Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"ResearchMind_Synthesis_{selected_paper}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )
                    with col_dl2:
                        st.download_button(
                            label="Download Synthesis Report (Markdown)",
                            data=report_md,
                            file_name=f"ResearchMind_Synthesis_{selected_paper}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )


    st.markdown("---")
    st.markdown("#### Natural Language Query Routing")
    user_query = st.text_input(
        "Enter any inquiry for automatic agent routing:",
        placeholder="e.g., extract the methodology, compare architectures, or identify limitations...",
    )
    if st.button("Route & Execute Query"):
        if user_query.strip():
            with st.spinner("Routing query to the optimal agent..."):
                routed = st.session_state.coordinator.route_query(user_query)
            st.success(f"Routed To: **{routed.get('agent')}**")
            res_val = routed.get("result", {})
            if isinstance(res_val, dict):
                if "table" in res_val:
                    st.table(pd.DataFrame(list(res_val["table"].items()), columns=["Dimension", "Extracted Synthesis"]))
                elif "summary" in res_val:
                    st.info(res_val["summary"])
                elif "comparison" in res_val:
                    st.markdown(res_val["comparison"])
                elif "analysis" in res_val:
                    st.markdown(res_val["analysis"])
                elif "critique" in res_val:
                    st.markdown(res_val["critique"])
                else:
                    st.write(res_val)
            else:
                st.write(res_val)


# ------------------------------------------------------------------------------
# TAB 2: PAPER READER MATRIX
# ------------------------------------------------------------------------------
with tab_reader:
    st.subheader("Paper Reader Agent")
    st.write(
        "Decomposes a specific paper into 6 verified academic dimensions with grounded citations."
    )

    reader_target = (
        st.selectbox("Select Paper to Read:", indexed_paper_ids, key="tab_reader_target")
        if indexed_paper_ids
        else None
    )

    if st.button("Read & Extract Schema", type="primary", disabled=not reader_target):
        with st.spinner(f"Extracting structured schema for '{reader_target}'..."):
            r_res = st.session_state.coordinator.reader_agent.run(paper_id=reader_target)

        table_dict = r_res.get("table", {})
        if table_dict:
            col_r1, col_r2 = st.columns(2)
            keys = list(table_dict.keys())
            mid = len(keys) // 2

            with col_r1:
                for k in keys[:mid]:
                    st.html(
                        f"<div class='dimension-card'>"
                        f"<div class='dimension-title' style='color: #60a5fa;'>{k}</div>"
                        f"<div style='font-size: 0.85rem; line-height: 1.5;'>{table_dict[k]}</div>"
                        f"</div>"
                    )
            with col_r2:
                for k in keys[mid:]:
                    color = "#34d399" if "Future" in k else ("#fbbf24" if "Limitation" in k else "#60a5fa")
                    st.html(
                        f"<div class='dimension-card'>"
                        f"<div class='dimension-title' style='color: {color};'>{k}</div>"
                        f"<div style='font-size: 0.85rem; line-height: 1.5;'>{table_dict[k]}</div>"
                        f"</div>"
                    )

            if r_res.get("citations"):
                st.html(f"<b>Referenced Citations:</b> <span class='citation-pill'>{r_res.get('citations')}</span>")


# ------------------------------------------------------------------------------
# TAB 3: LITERATURE SEARCH
# ------------------------------------------------------------------------------
with tab_search:
    st.subheader("Literature Search Agent")
    st.write("Semantically searches the entire corpus and generates an executive overview with grounded passage references.")

    search_q = st.text_input(
        "Search Query:",
        placeholder="e.g., what evaluation metrics and benchmarks were reported across papers?",
    )
    top_k = st.slider("Passages to retrieve:", min_value=2, max_value=8, value=4)

    if st.button("Execute Search", type="primary"):
        if search_q.strip():
            with st.spinner("Searching vector index and synthesizing overview..."):
                s_res = st.session_state.coordinator.search_agent.run(search_q, n_results=top_k)

            st.markdown("#### Executive Summary")
            st.info(s_res.get("summary", "No summary generated."))

            results = s_res.get("results", [])
            if results:
                st.markdown("#### Retrieved Evidence & Chunks")
                for idx, r in enumerate(results, 1):
                    with st.expander(
                        f"Chunk {idx}: {r.get('paper_id', '?')} — Page {r.get('page_number', 'N/A')} (Distance: {r.get('distance', 0):.4f})"
                    ):
                        st.markdown(f"*{r.get('text', '')}*")


# ------------------------------------------------------------------------------
# TAB 4: CROSS-PAPER COMPARISON
# ------------------------------------------------------------------------------
with tab_comp:
    st.subheader("Comparison Agent")
    st.write("Synthesizes methodological differences, architectural choices, and empirical results across multiple indexed documents.")

    comp_q = st.text_input(
        "Comparative Inquiry:",
        placeholder="e.g., compare the datasets and attention mechanisms used in the indexed papers",
    )

    if st.button("Generate Comparison", type="primary"):
        if comp_q.strip():
            with st.spinner("Synthesizing comparative report across documents..."):
                c_res = st.session_state.coordinator.comparison_agent.run(comp_q)

            st.markdown("#### Cross-Paper Comparative Analysis")
            st.markdown(c_res.get("comparison", "No comparison generated."))


# ------------------------------------------------------------------------------
# TAB 5: RESEARCH GAPS & HYPOTHESES
# ------------------------------------------------------------------------------
with tab_gap:
    st.subheader("Research Gap Agent")
    st.write(
        "Mines explicit limitation statements across the corpus to formulate testable hypotheses and promising research opportunities."
    )

    gap_topic = st.text_input(
        "Topic / Dimension to Investigate:",
        placeholder="e.g., computational bottlenecks, dataset scaling, or hardware constraints",
    )

    if st.button("Discover Gaps & Hypotheses", type="primary"):
        if gap_topic.strip():
            with st.spinner("Analyzing limitations and formulating hypotheses..."):
                g_res = st.session_state.coordinator.gap_agent.run(gap_topic)

            st.markdown("#### Discovered Research Gaps & Proposed Hypotheses")
            st.markdown(g_res.get("analysis", "No gap analysis generated."))


# ------------------------------------------------------------------------------
# TAB 6: PEER REVIEWER SCORECARD
# ------------------------------------------------------------------------------
with tab_rev:
    st.subheader("Proposal Reviewer Agent")
    st.write(
        "Evaluates methodology drafts, pre-prints, or research proposals using academic conference review rubrics (NeurIPS / ICLR style)."
    )

    prop_text = st.text_area(
        "Draft Proposal or Methodology Section:",
        placeholder="Paste your methodology, experiment design, or abstract here for critical review...",
        height=180,
    )

    if st.button("Generate Critical Review", type="primary"):
        if prop_text.strip():
            with st.spinner("Reviewing submission against academic conference standards..."):
                rev_res = st.session_state.coordinator.reviewer_agent.run(prop_text)

            st.markdown("#### Reviewer Evaluation & Conference Rubric")
            st.markdown(rev_res.get("critique", "No critique generated."))
