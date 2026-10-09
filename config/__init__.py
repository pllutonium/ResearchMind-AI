"""
Module: config/__init__.py
Description: Unified configuration package. Exports settings singleton as well as
backward-compatible constants for legacy modules and tests.
"""
from config.settings import settings, Settings

# Re-export settings singleton
__all__ = [
    "settings",
    "Settings",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_PDFS_DIR",
    "PROCESSED_DIR",
    "CHROMA_DB_DIR",
    "METADATA_DB_PATH",
    "DOCSTORE_PATH",
    "BM25_INDEX_PATH",
    "SAMPLE_PAPERS_DIR",
    "GEMINI_API_KEY",
    "GEMINI_MODEL_NAME",
    "GROQ_API_KEY",
    "GROQ_MODEL_NAME",
    "OLLAMA_BASE_URL",
    "LLM_MODEL_NAME",
    "LLM_NUM_CTX",
    "DEFAULT_LLM_BACKEND",
    "EMBEDDING_MODEL_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "VECTOR_DB_COLLECTION_NAME",
    "_find_tesseract_cmd",
]

# Backward-compatible module-level aliases
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
RAW_PDFS_DIR = settings.RAW_PDFS_DIR
PROCESSED_DIR = settings.PROCESSED_DIR
CHROMA_DB_DIR = settings.CHROMA_DB_DIR
METADATA_DB_PATH = settings.METADATA_DB_PATH
DOCSTORE_PATH = settings.DOCSTORE_PATH
BM25_INDEX_PATH = settings.BM25_INDEX_PATH
SAMPLE_PAPERS_DIR = settings.SAMPLE_PAPERS_DIR

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL_NAME = settings.GEMINI_MODEL_NAME
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_MODEL_NAME = settings.GROQ_MODEL_NAME
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
LLM_MODEL_NAME = settings.LLM_MODEL_NAME
LLM_NUM_CTX = settings.LLM_NUM_CTX
DEFAULT_LLM_BACKEND = settings.DEFAULT_LLM_BACKEND
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
VECTOR_DB_COLLECTION_NAME = settings.VECTOR_DB_COLLECTION_NAME

_find_tesseract_cmd = settings.find_tesseract_cmd
