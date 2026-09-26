"""
Module: config.py
Description: Centralized configuration management for ResearchMind AI Multi-Agent RAG System.
Features: Environment variable loading, directory path constants, LLM backend defaults,
embedding settings, and automatic Tesseract OCR discovery.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# --- Base Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"
METADATA_DB_PATH = BASE_DIR / "data" / "metadata.sqlite3"

# --- Primary Cloud Engine: Google Gemini Cloud (Recommended: Free, Sub-second, Zero Laptop Load) ---
# Obtain your free API key at: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-lite-latest")

# --- Optional Cloud Engine: Groq Cloud (Sub-second Open Weights) ---
# Obtain your free key at: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# --- Optional Offline Engine: Local Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "2048"))

# --- Default Active Engine ---
DEFAULT_LLM_BACKEND = os.getenv("DEFAULT_LLM_BACKEND", "gemini")

# --- Dense Embeddings Settings ---
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Sliding Window Chunking Parameters ---
CHUNK_SIZE = 800        # Maximum character limit per chunk
CHUNK_OVERLAP = 150     # Overlap window to preserve sentence semantics

# --- Vector Database Collection ---
VECTOR_DB_COLLECTION_NAME = "researchmind_papers"


# --- Automatic Tesseract OCR Discovery ---
def _find_tesseract_cmd():
    import shutil
    env_path = os.getenv("TESSERACT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    system_path = shutil.which("tesseract")
    if system_path:
        return system_path
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\tesseract\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


TESSERACT_CMD = _find_tesseract_cmd()

# Ensure required runtime directories exist
for _dir in (RAW_PDFS_DIR, PROCESSED_DIR, CHROMA_DB_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
