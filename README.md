# ResearchMind AI — Autonomous Multi-Agent Academic Laboratory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.38+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Google Gemini Cloud](https://img.shields.io/badge/Inference-Google%20Gemini%20Cloud-008080.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ResearchMind AI** is an autonomous multi-agent Retrieval-Augmented Generation (RAG) platform designed for rigorous scientific literature analysis, structured methodological decomposition, cross-paper baseline comparison, hypothesis formulation, and conference-grade peer review.

---

## Key Highlights

- **5 Specialized Autonomous Agents**:
  - `01 · Literature Search`: Semantic corpus search synthesizing core findings and executive overviews.
  - `02 · Paper Reader`: Extracts a structured 6-dimension schema with grounded page citations `(Page X)`.
  - `03 · Comparison`: Contrasts novel architectures against established baselines and computational trade-offs.
  - `04 · Research Gap`: Surfaces unaddressed bottlenecks and formulates formal testable hypotheses ($H_1, H_2$).
  - `05 · Peer Reviewer`: Evaluates proposals under top-tier conference rubrics (NeurIPS / ICLR style).
- **Sub-Second Cloud Inference**: Powered by **Google Gemini Cloud REST API** (`gemini-flash-lite-latest`, `gemini-3.7-flash`), completing the entire 5-agent pipeline in **~14 seconds** with **zero compute load on your laptop**.
- **Dynamic Sequential Stage Lighting**: Editorial academic terminal dashboard (`#0e0e0c`) where each card visibly illuminates with a glowing border and pulsing radar dot while executing in the cloud.
- **Publication-Ready PDF Export**: Generates formal academic PDF synthesis reports with custom two-pass page numbering (`Page X of Y`), tables, and grounded citations via ReportLab.
- **Dual-Path PDF Ingestion**: Native PyMuPDF text parsing with automatic Tesseract OCR fallback for scanned/rasterized pages.
- **Detailed System Documentation**: Includes an in-depth 5-page architectural specification in [`ResearchMind_AI_Codebase_Documentation.pdf`](ResearchMind_AI_Codebase_Documentation.pdf).

---

## System Architecture

```
                                  [ User / Researcher ]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │  Streamlit Academic Dashboard (app.py)        │
                    │  - Editorial Pitch-Black Theme (#0e0e0c)       │
                    │  - Sequential Glowing Stage Lighting          │
                    │  - 6 Functional Research Workspaces           │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      AICoordinator (research_agents.py)       │
                    │      Sequential Pipeline & Intent Routing     │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
        ┌───────────────────┴──────────┐     ┌──────────────┴────────────────┐
        ▼                              ▼     ▼                               ▼
┌───────────────┐              ┌───────────────┐                     ┌───────────────┐
│ Literature    │              │ Paper Reader  │     ...             │ Proposal      │
│ Search Agent  │              │ Agent         │                     │ Reviewer Agent│
└───────┬───────┘              └───────┬───────┘                     └───────┬───────┘
        │                              │                                     │
        └───────────────────────┬──────┴─────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ ChromaDB & SQLite Substrate   │       │ Google Gemini Cloud REST API  │
│ - Dense Embeddings (MiniLM)   │       │ - Sub-second Cloud Latency    │
│ - Scoped 'paper_id' Filtering │       │ - Resilient Auto-Failover     │
│ - Page-Preserving Citations   │       │ - Zero Laptop CPU Compute     │
└───────────────────────────────┘       └───────────────────────────────┘
```

---

## Quickstart Guide

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/researchmind-ai.git
cd researchmind-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate          # On Windows
source venv/bin/activate       # On Linux / macOS
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials

Duplicate `.env.example` to `.env` and insert your free Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-flash-lite-latest
DEFAULT_LLM_BACKEND=gemini
```
>  *Get a free API [aistudio.google.com/apikey](https://aistudio.google.com/apikey).*

### 4. Launch the Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

Open **`http://localhost:8501`** in your browser to start analyzing research papers!

---

## Performance Benchmarks

Measured on the 5-Agent Collaborative Pipeline:

| Stage | Agent Deliverable | Execution Time | Host Laptop Load |
| :--- | :--- | :--- | :--- |
| **01 · Search** | Executive Overview & Citations | **2.49s** | 0% CPU (Cloud) |
| **02 · Reader** | 6-Dimension Structured Schema | **1.45s** | 0% CPU (Cloud) |
| **03 · Compare**| Baseline & Metric Contrasts | **3.08s** | 0% CPU (Cloud) |
| **04 · Gap** | Formulated Hypotheses ($H_1, H_2$) | **2.62s** | 0% CPU (Cloud) |
| **05 · Review** | NeurIPS/ICLR Review Scorecard | **5.21s** | 0% CPU (Cloud) |
| **Total** | **Full 5-Agent Collaborative Pipeline** | **14.86s** | **Zero Compute** |

---

## Repository Structure

```
researchmind-ai/
├── config.py                                   # Centralized configuration & constants
├── requirements.txt                            # Production dependencies
├── .env.example                                # Template for API credentials
├── ResearchMind_AI_Codebase_Documentation.pdf  # Comprehensive 5-page PDF manual
├── dashboard/
│   └── app.py                                  # Minimalist terminal dashboard
├── src/
│   ├── agents/
│   │   └── research_agents.py                  # AICoordinator & 5 specialized sub-agents
│   ├── llm/
│   │   └── client.py                           # Cloud LLM client with auto-failover
│   ├── vectorstore/
│   │   └── chroma_store.py                     # ChromaDB vector index wrapper
│   ├── embeddings/
│   │   └── embedder.py                         # Dense vector encoder & pre-warming
│   ├── ingestion/
│   │   └── pipeline.py                         # PyMuPDF + Tesseract OCR extraction
│   ├── db/
│   │   └── metadata_store.py                   # SQLite metadata catalog
│   └── reporting/
│       └── pdf_export.py                       # Publication-grade PDF generator
└── data/
    ├── raw_pdfs/                               # Target research PDFs
    ├── processed/                              # Intermediate extraction files
    └── chroma_db/                              # Persistent vector storage
```

---

## Documentation

For deep technical details on the architecture, mathematical schemas, and vector indexing heuristics, refer to [`ResearchMind_AI_Codebase_Documentation.pdf`](ResearchMind_AI_Codebase_Documentation.pdf).

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
