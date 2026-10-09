"""
Module: config/settings.py
Description: Production-grade centralized settings using Pydantic BaseSettings.
Loads environment variables from .env and provides strict typing and defaults without Pydantic V1 deprecations.
"""
from pathlib import Path
from typing import Optional
import os
import shutil

# Suppress TensorFlow warnings and force PyTorch backend
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None


class Settings(BaseSettings):
    """Enterprise configuration for ResearchMind AI Agentic RAG system."""

    # Project Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    RAW_PDFS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw_pdfs")
    PROCESSED_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    CHROMA_PERSIST_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "chroma_db")
    CHROMA_DB_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "chroma_db")
    METADATA_DB_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "metadata.sqlite3")
    DOCSTORE_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "docstore.sqlite3")
    BM25_INDEX_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "bm25_index.json")
    SAMPLE_PAPERS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "sample_papers")
    PROMPTS_FILE: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "prompts.yaml")

    # LLM Settings (Google Gemini Cloud)
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, validation_alias="GOOGLE_API_KEY")
    GEMINI_MODEL_NAME: str = Field(default="gemini-flash-lite-latest", validation_alias="GEMINI_MODEL_NAME")
    LLM_TEMPERATURE: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=2048, validation_alias="LLM_MAX_TOKENS")

    # Groq Cloud
    GROQ_API_KEY: str = Field(default="", validation_alias="GROQ_API_KEY")
    GROQ_MODEL_NAME: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL_NAME")

    # Offline Local Ollama
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    LLM_MODEL_NAME: str = Field(default="llama3.1", validation_alias="LLM_MODEL_NAME")
    LLM_NUM_CTX: int = Field(default=2048, validation_alias="LLM_NUM_CTX")
    DEFAULT_LLM_BACKEND: str = Field(default="gemini", validation_alias="DEFAULT_LLM_BACKEND")

    # Embeddings & Reranker
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="RERANKER_MODEL")
    VECTOR_DB_COLLECTION_NAME: str = Field(default="researchmind_papers")

    # Chunking Parameters
    CHUNK_SIZE: int = Field(default=800, validation_alias="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=150, validation_alias="CHUNK_OVERLAP")
    PARENT_CHUNK_SIZE: int = Field(default=800, validation_alias="PARENT_CHUNK_SIZE")
    PARENT_CHUNK_OVERLAP: int = Field(default=100, validation_alias="PARENT_CHUNK_OVERLAP")
    CHILD_CHUNK_SIZE: int = Field(default=200, validation_alias="CHILD_CHUNK_SIZE")
    CHILD_CHUNK_OVERLAP: int = Field(default=30, validation_alias="CHILD_CHUNK_OVERLAP")

    # Retrieval & RRF
    RRF_K: int = Field(default=60, validation_alias="RRF_K")
    INITIAL_CANDIDATE_POOL: int = Field(default=15, validation_alias="INITIAL_CANDIDATE_POOL")
    FINAL_TOP_K: int = Field(default=4, validation_alias="FINAL_TOP_K")

    # arXiv Fallback
    ARXIV_MAX_RESULTS: int = Field(default=3, validation_alias="ARXIV_MAX_RESULTS")

    # Observability (Langfuse)
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")
    LANGFUSE_BASE_URL: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL")

    # Tesseract OCR path
    TESSERACT_PATH: Optional[str] = Field(default=None, validation_alias="TESSERACT_PATH")

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=str(Path(__file__).resolve().parent.parent / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
            populate_by_name=True,
        )
    else:
        class Config:
            env_file = str(Path(__file__).resolve().parent.parent / ".env")
            env_file_encoding = "utf-8"
            extra = "ignore"

    def model_post_init(self, __context):
        # Sync GOOGLE_API_KEY if only GEMINI_API_KEY is supplied
        if self.GEMINI_API_KEY and not self.GOOGLE_API_KEY:
            self.GOOGLE_API_KEY = self.GEMINI_API_KEY
            os.environ["GOOGLE_API_KEY"] = self.GEMINI_API_KEY
        if self.GEMINI_API_KEY:
            os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY

        # Ensure directories exist
        for d in [self.DATA_DIR, self.RAW_PDFS_DIR, self.PROCESSED_DIR, self.CHROMA_PERSIST_DIR, self.SAMPLE_PAPERS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def find_tesseract_cmd(self) -> Optional[str]:
        if self.TESSERACT_PATH and os.path.exists(self.TESSERACT_PATH):
            return self.TESSERACT_PATH
        system_path = shutil.which("tesseract")
        if system_path:
            return system_path
        return None


# Global settings singleton
settings = Settings()
