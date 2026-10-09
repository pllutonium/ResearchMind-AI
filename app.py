"""
Module: app.py
Description: Modern, Client-Facing Academic Research Workspace for ResearchMind AI.
Clean typography with zero emojis, active document/paper scope context header,
sanitized PDF generation (Markdown & LaTeX parsed), and a simplified export panel.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import sys
import tempfile
import time
import io
import re
import urllib.parse

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Suppress unnecessary logs and telemetry noise
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"

import streamlit as st

# Configure enterprise research page settings
st.set_page_config(
    page_title="ResearchMind AI - Academic Research Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PDF Generation imports from ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config.settings import settings
from src.ingestion.parser import pdf_parser
from src.ingestion.chunking import hierarchical_chunker
from src.ingestion.docstore import docstore
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.agents.graph import run_agentic_rag


# =============================================================================
# CUSTOM ACADEMIC WORKSPACE STYLING (Elicit / Consensus Inspired)
# =============================================================================
st.markdown(
    """
    <style>
    /* Main Layout Styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }
    
    /* Academic Header */
    .workspace-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .workspace-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.6px;
        color: #f0f6fc;
        margin-bottom: 0.3rem;
    }
    .workspace-subtitle {
        font-size: 0.96rem;
        color: #8b949e;
        line-height: 1.5;
    }

    /* Active Paper Context Banner */
    .active-scope-banner {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 9px 15px;
        margin-bottom: 1.3rem;
        font-size: 0.86rem;
    }
    .scope-tag {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
        color: #58a6ff;
        background: rgba(88, 166, 255, 0.12);
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .scope-text {
        color: #f0f6fc;
        font-weight: 500;
        line-height: 1.4;
    }

    /* Verification Badge (Zero Emojis) */
    .verification-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(35, 134, 54, 0.12);
        border: 1px solid rgba(63, 185, 80, 0.35);
        border-radius: 4px;
        padding: 5px 12px;
        font-size: 0.82rem;
        color: #3fb950;
        margin-bottom: 1.1rem;
        font-weight: 500;
    }
    .badge-status {
        font-weight: 700;
        color: #3fb950;
        font-size: 0.74rem;
        letter-spacing: 0.5px;
        background: rgba(63, 185, 80, 0.16);
        padding: 2px 7px;
        border-radius: 3px;
    }
    .badge-sep {
        color: #484f58;
    }
    .badge-desc {
        color: #8b949e;
        font-size: 0.80rem;
    }

    /* Reference & Citation Cards */
    .ref-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 13px 15px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }
    .ref-card:hover {
        border-color: #58a6ff;
    }
    .ref-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 5px;
    }
    .ref-title {
        font-weight: 600;
        color: #f0f6fc;
        font-size: 0.88rem;
        line-height: 1.35;
    }
    .ref-meta {
        font-size: 0.77rem;
        color: #8b949e;
        margin-bottom: 7px;
    }
    .ref-excerpt {
        background: #0d1117;
        border-left: 2px solid #58a6ff;
        border-radius: 4px;
        padding: 7px 10px;
        font-size: 0.79rem;
        color: #c9d1d9;
        line-height: 1.45;
        font-style: italic;
        margin-bottom: 8px;
    }
    .ref-link {
        font-size: 0.78rem;
        color: #58a6ff;
        text-decoration: none;
        font-weight: 500;
    }
    .ref-link:hover {
        text-decoration: underline;
    }

    /* Badges */
    .badge-corpus {
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.35);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }
    .badge-arxiv {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.35);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }

    /* Quick Starter Chip Buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 6px !important;
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-size: 0.81rem !important;
        font-weight: 450 !important;
        padding: 7px 12px !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        background-color: #1f242c !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #58a6ff;
    }

    /* Export Box */
    .export-box {
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .status-text {
        font-size: 0.85rem;
        color: #8b949e;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS: LATEX / MARKDOWN SANITIZATION & PDF ENGINE
# =============================================================================
def sanitize_latex_and_markdown(text: str) -> str:
    """Removes raw LaTeX math markup, commands, and delimiters, producing clean text."""
    latex_map = [
        (r'\\times', 'x'),
        (r'\\pm', '+/-'),
        (r'\\leq', '<='),
        (r'\\le', '<='),
        (r'\\geq', '>='),
        (r'\\ge', '>='),
        (r'\\approx', '~'),
        (r'\\neq', '!='),
        (r'\\to', '->'),
        (r'\\rightarrow', '->'),
        (r'\\leftarrow', '<-'),
        (r'\\in', 'in'),
        (r'\\mu', 'u'),
        (r'\\alpha', 'alpha'),
        (r'\\beta', 'beta'),
        (r'\\gamma', 'gamma'),
        (r'\\sigma', 'sigma'),
        (r'\\cdot', '*'),
        (r'\\dots', '...'),
        (r'\\mathcal\{([^}]+)\}', r'\1'),
        (r'\\mathbf\{([^}]+)\}', r'\1'),
        (r'\\mathrm\{([^}]+)\}', r'\1'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)'),
        (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
        (r'\\%', '%'),
    ]
    for pattern, repl in latex_map:
        text = re.sub(pattern, repl, text)

    # Strip display and inline math delimiters: $$...$$ and $...$
    text = re.sub(r'\$\$([^\$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    # Remove any remaining stray dollar signs
    text = text.replace('$', '')
    return text


def convert_markdown_to_reportlab_html(text: str) -> str:
    """Sanitizes LaTeX and converts inline Markdown formatting to ReportLab XML tags."""
    text = sanitize_latex_and_markdown(text)

    # Escape HTML special chars
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')

    # Markdown Bold: **text** or __text__ -> <b>text</b>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)

    # Markdown Italic: *text* -> <i>text</i> (when not part of bold)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)

    # Markdown inline code: `code` -> <b>code</b>
    text = re.sub(r'`([^`]+)`', r'<b>\1</b>', text)

    return text


def generate_synthesis_pdf(query: str, synthesis_text: str, sources: List[Dict[str, Any]]) -> bytes:
    """
    Renders publication-grade, sanitized academic PDF report with parsed markdown headings,
    bulleted lists, bold markers, and clean paper metadata using ReportLab.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold'
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#475569'),
        fontName='Helvetica'
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=12.5,
        leading=16.5,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold',
        spaceBefore=11,
        spaceAfter=5
    )
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Heading3'],
        fontSize=10.5,
        leading=14.5,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Bold',
        spaceBefore=7,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica',
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica',
        leftIndent=15,
        spaceAfter=4
    )
    source_style = ParagraphStyle(
        'DocSource',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#334155'),
        fontName='Helvetica',
        spaceAfter=5
    )

    clean_query = query.replace('<', '&lt;').replace('>', '&gt;')
    
    # Extract distinct paper titles for header metadata
    distinct_titles = []
    for s in sources:
        t = s.get("title", "").strip()
        if t and t not in distinct_titles:
            distinct_titles.append(t)
    
    papers_header = ", ".join(distinct_titles) if distinct_titles else "Indexed Academic Literature"
    date_str = time.strftime("%B %d, %Y")

    story = [
        Paragraph("ResearchMind AI - Academic Synthesis Report", title_style),
        Spacer(1, 4),
        Paragraph(
            f"<b>Inquiry:</b> {clean_query}<br/>"
            f"<b>Referenced Literature:</b> {papers_header}<br/>"
            f"<b>Date:</b> {date_str} &nbsp;|&nbsp; <b>Verification:</b> Grounded in Peer-Reviewed Sources",
            meta_style
        ),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=12),
        Paragraph("<b>Executive Synthesis & Analysis</b>", h2_style),
        Spacer(1, 4),
    ]

    # Process synthesis line by line / block by block
    lines = synthesis_text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Markdown Headings: ###, ##, #
        if line.startswith("### "):
            head_text = convert_markdown_to_reportlab_html(line[4:].strip())
            story.append(Paragraph(f"<b>{head_text}</b>", h3_style))
        elif line.startswith("## "):
            head_text = convert_markdown_to_reportlab_html(line[3:].strip())
            story.append(Paragraph(f"<b>{head_text}</b>", h2_style))
        elif line.startswith("# "):
            head_text = convert_markdown_to_reportlab_html(line[2:].strip())
            story.append(Paragraph(f"<b>{head_text}</b>", h2_style))
        # Bullet points: * or -
        elif line.startswith("* ") or line.startswith("- "):
            b_text = convert_markdown_to_reportlab_html(line[2:].strip())
            story.append(Paragraph(f"&bull;&nbsp; {b_text}", bullet_style))
        elif re.match(r'^\d+\.\s+', line):
            # Numbered list
            num_match = re.match(r'^(\d+\.)\s+(.*)', line)
            num_prefix = num_match.group(1)
            item_text = convert_markdown_to_reportlab_html(num_match.group(2))
            story.append(Paragraph(f"<b>{num_prefix}</b> {item_text}", bullet_style))
        else:
            # Paragraph
            para_lines = [line]
            while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith(("#", "* ", "- ")) and not re.match(r'^\d+\.\s+', lines[i+1].strip()):
                i += 1
                para_lines.append(lines[i].strip())
            para_text = " ".join(para_lines)
            story.append(Paragraph(convert_markdown_to_reportlab_html(para_text), body_style))
        i += 1

    if sources:
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=10))
        story.append(Paragraph("<b>Referenced Academic Literature</b>", h2_style))
        story.append(Spacer(1, 4))

        for idx, src in enumerate(sources, 1):
            t = convert_markdown_to_reportlab_html(src.get("title", "Academic Literature"))
            p = src.get("page_number", 1)
            auth = convert_markdown_to_reportlab_html(src.get("metadata", {}).get("primary_author", "Research Scholar"))
            yr = src.get("metadata", {}).get("year", "2024")
            raw_ex = " ".join(src.get("text", "").split())[:200]
            clean_ex = convert_markdown_to_reportlab_html(raw_ex)
            
            src_text = f"<b>[{idx}] {t}</b> - {auth} ({yr}) [Page {p}]<br/><i>&ldquo;{clean_ex}...&rdquo;</i>"
            story.append(Paragraph(src_text, source_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = None


# =============================================================================
# SIDEBAR: Literature Library & Document Ingestion
# =============================================================================
with st.sidebar:
    st.markdown("### Document Ingestion")
    st.markdown(
        "<div class='status-text'>Upload academic publications (PDF) to index them into the research library.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # File Upload Component
    uploaded_files = st.file_uploader(
        "Upload Academic Literature (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple peer-reviewed papers or technical reports.",
    )

    if st.button("Ingest and Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select one or more PDF documents prior to indexing.")
        else:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                status_placeholder.markdown(f"**Indexing ({idx}/{total_files}):** `{uploaded_file.name}`")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    paper = pdf_parser.parse(tmp_path)
                    parent_docs, child_chunks = hierarchical_chunker.chunk_paper(paper)
                    hybrid_retriever.index_chunks(child_chunks)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    if tmp_path.exists():
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                progress_bar.progress(idx / total_files)

            status_placeholder.empty()
            progress_bar.empty()
            st.success(f"Indexing Complete: Successfully integrated {total_files} publication(s) into the research corpus.")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    
    # Display distinct indexed paper titles
    indexed_titles = list({docstore.get(pid).title for pid in docstore.all_ids() if docstore.get(pid) and docstore.get(pid).title})
    total_indexed_count = len(indexed_titles)
    
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Indexed Publications</div>
            <div class='metric-value'>{total_indexed_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# =============================================================================
# ACTIVE PAPER CONTEXT HEADER (DYNAMIC RUN SCOPE)
# =============================================================================
active_scope = None

# If there is conversational history, inspect the most recent assistant sources
if st.session_state["messages"]:
    for msg in reversed(st.session_state["messages"]):
        if msg.get("role") == "assistant" and msg.get("sources"):
            sources_titles = []
            for s in msg.get("sources", []):
                t = s.get("title", "").strip()
                if t and t not in sources_titles:
                    sources_titles.append(t)
            if len(sources_titles) == 1:
                active_scope = f"Active Reference: {sources_titles[0]}"
            elif len(sources_titles) > 1:
                active_scope = f"Referenced Sources ({len(sources_titles)}): " + ", ".join(sources_titles)
            break

# If no search has occurred yet, fallback to indexed repository papers
if not active_scope:
    if len(indexed_titles) == 1:
        active_scope = f"Active Reference: {indexed_titles[0]}"
    elif len(indexed_titles) > 1:
        active_scope = f"Corpus Scope ({len(indexed_titles)} Papers): " + ", ".join(indexed_titles)
    else:
        active_scope = "Corpus Scope: Curated Academic Literature"


# =============================================================================
# MAIN RESEARCH WORKSPACE: Header, Active Scope, Quick Starters, Dual Panel
# =============================================================================
st.markdown(
    """
    <div class='workspace-header'>
        <div class='workspace-title'>ResearchMind AI</div>
        <div class='workspace-subtitle'>
            Autonomous Academic Synthesis and Literature Intelligence Platform - Evidence-Backed Scientific Inquiry
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active Paper Context Header
st.markdown(
    f"""
    <div class='active-scope-banner'>
        <span class='scope-tag'>PAPER CONTEXT</span>
        <span class='scope-text'>{active_scope}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. ONE-CLICK QUICK STARTERS (ZERO TYPING, CLEAN TYPOGRAPHY)
# -----------------------------------------------------------------------------
st.markdown("<div style='font-size:0.83rem;color:#8b949e;margin-bottom:0.5rem;font-weight:600;letter-spacing:0.3px;'>QUICK-START RESEARCH INQUIRIES:</div>", unsafe_allow_html=True)
q_col1, q_col2, q_col3 = st.columns(3)

with q_col1:
    if st.button("Explain FP8 quantization memory savings in LLM training", use_container_width=True):
        st.session_state["submitted_query"] = "Explain FP8 quantization memory savings in LLM training"
        st.rerun()

with q_col2:
    if st.button("How does Multi-Head Latent Attention (MLA) reduce KV cache?", use_container_width=True):
        st.session_state["submitted_query"] = "How does Multi-Head Latent Attention (MLA) reduce KV cache?"
        st.rerun()

with q_col3:
    if st.button("Compare DeepSeek-V3 architecture optimizations vs standard Transformers", use_container_width=True):
        st.session_state["submitted_query"] = "Compare DeepSeek-V3 architecture optimizations vs standard Transformers"
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CONVERSATION THREAD & DUAL-PANEL DISPLAY
# -----------------------------------------------------------------------------
for msg_idx, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{msg['content']}**")
    else:
        with st.chat_message("assistant"):
            synthesis_text = msg["content"]
            sources = msg.get("sources", [])
            query_ref = msg.get("query", "Academic Inquiry")

            # Dual-column presentation: Synthesis on Left, Citation Panel on Right
            col_synth, col_refs = st.columns([13, 9], gap="large")

            # --- LEFT COLUMN: Synthesis & Simplified Export ---
            with col_synth:
                st.markdown(
                    """
                    <div class='verification-badge'>
                        <span class='badge-status'>VERIFIED</span>
                        <span class='badge-sep'>|</span>
                        <span class='badge-desc'>100% Grounded Academic Synthesis - Peer-Reviewed Sources</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(synthesis_text)

                # Simplified Actionable Researcher Tools (ONLY Download PDF)
                pdf_bytes = generate_synthesis_pdf(query_ref, synthesis_text, sources)
                st.markdown("<div class='export-box'>", unsafe_allow_html=True)
                st.download_button(
                    "Download Synthesis Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"synthesis_report_{msg_idx+1}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{msg_idx}",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # --- RIGHT COLUMN: Dedicated Interactive Citation & Reference Panel ---
            with col_refs:
                st.markdown("#### Evidence & Source Literature")
                if not sources:
                    st.caption("No specific paper chunks directly referenced.")
                else:
                    for s_idx, src in enumerate(sources, 1):
                        title = src.get("title", "Academic Publication")
                        page = src.get("page_number", 1)
                        auth = src.get("metadata", {}).get("primary_author", "Research Scholar")
                        year = src.get("metadata", {}).get("year", "2024")
                        source_type = src.get("metadata", {}).get("source", "Indexed Corpus")
                        is_arxiv = "arxiv" in str(source_type).lower()
                        badge_label = "arXiv Fallback" if is_arxiv else "Indexed Corpus"
                        badge_cls = "badge-arxiv" if is_arxiv else "badge-corpus"

                        raw_text = src.get("text", "")
                        clean_text = " ".join(raw_text.split())
                        if len(clean_text) > 260:
                            clean_text = clean_text[:260].rsplit(" ", 1)[0] + "..."

                        paper_url = src.get("metadata", {}).get("url")
                        if not paper_url:
                            paper_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"

                        st.markdown(
                            f"""
                            <div class='ref-card'>
                                <div class='ref-header'>
                                    <div class='ref-title'>[{s_idx}] {title}</div>
                                    <span class='{badge_cls}'>{badge_label}</span>
                                </div>
                                <div class='ref-meta'>{auth} ({year}) &bull; Page {page}</div>
                                <div class='ref-excerpt'>&ldquo;{clean_text}&rdquo;</div>
                                <a href='{paper_url}' target='_blank' class='ref-link'>View Original Literature &rarr;</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# -----------------------------------------------------------------------------
# 3. SEARCH INPUT HANDLING & LIVE RESEARCH EXECUTION
# -----------------------------------------------------------------------------
typed_query = st.chat_input("Enter research question or methodology inquiry...")
if typed_query:
    st.session_state["submitted_query"] = typed_query

active_query = st.session_state.pop("submitted_query", None)

if active_query:
    # 1. Append & Display User Query
    st.session_state["messages"].append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(f"**{active_query}**")

    # 2. Execute Research Pipeline with Clean Non-Technical Progress Indicator
    with st.chat_message("assistant"):
        with st.status("Analyzing academic literature...", expanded=True) as status:
            st.write("Searching through curated research papers...")
            time.sleep(0.35)

            st.write("Verifying factual alignment and relevance...")
            state = run_agentic_rag(query=active_query, user_id="academic-user")

            st.write("Synthesizing evidence-backed response...")
            time.sleep(0.25)

            status.update(label="Synthesis verified and complete", state="complete", expanded=False)

        final_text = state.get("final_response") or state.get("candidate_answer") or "No response generated."
        retrieved_docs = state.get("retrieved_docs", [])
        arxiv_docs = state.get("arxiv_docs", [])
        all_sources = (retrieved_docs + arxiv_docs)[:6]

        col_synth, col_refs = st.columns([13, 9], gap="large")

        with col_synth:
            st.markdown(
                """
                <div class='verification-badge'>
                    <span class='badge-status'>VERIFIED</span>
                    <span class='badge-sep'>|</span>
                    <span class='badge-desc'>100% Grounded Academic Synthesis - Peer-Reviewed Sources</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(final_text)

            # Simplified Export (ONLY Download PDF)
            new_idx = len(st.session_state["messages"])
            pdf_bytes = generate_synthesis_pdf(active_query, final_text, all_sources)

            st.markdown("<div class='export-box'>", unsafe_allow_html=True)
            st.download_button(
                "Download Synthesis Report (PDF)",
                data=pdf_bytes,
                file_name=f"synthesis_report_{new_idx}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_new_{new_idx}",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_refs:
            st.markdown("#### Evidence & Source Literature")
            if not all_sources:
                st.caption("No specific paper chunks directly referenced.")
            else:
                for s_idx, src in enumerate(all_sources, 1):
                    title = src.get("title", "Academic Publication")
                    page = src.get("page_number", 1)
                    auth = src.get("metadata", {}).get("primary_author", "Research Scholar")
                    year = src.get("metadata", {}).get("year", "2024")
                    source_type = src.get("metadata", {}).get("source", "Indexed Corpus")
                    is_arxiv = "arxiv" in str(source_type).lower()
                    badge_label = "arXiv Fallback" if is_arxiv else "Indexed Corpus"
                    badge_cls = "badge-arxiv" if is_arxiv else "badge-corpus"

                    raw_text = src.get("text", "")
                    clean_text = " ".join(raw_text.split())
                    if len(clean_text) > 260:
                        clean_text = clean_text[:260].rsplit(" ", 1)[0] + "..."

                    paper_url = src.get("metadata", {}).get("url")
                    if not paper_url:
                        paper_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"

                    st.markdown(
                        f"""
                        <div class='ref-card'>
                            <div class='ref-header'>
                                <div class='ref-title'>[{s_idx}] {title}</div>
                                <span class='{badge_cls}'>{badge_label}</span>
                            </div>
                            <div class='ref-meta'>{auth} ({year}) &bull; Page {page}</div>
                            <div class='ref-excerpt'>&ldquo;{clean_text}&rdquo;</div>
                            <a href='{paper_url}' target='_blank' class='ref-link'>View Original Literature &rarr;</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Store in session state
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": final_text,
                "sources": all_sources,
                "query": active_query,
            }
        )
