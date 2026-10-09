import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model_instance: Optional[SentenceTransformer] = None


def warm_up_embedder() -> None:
    """Pre-loads the embedding model singleton into memory to eliminate runtime lag."""
    _get_model()


def _get_model() -> SentenceTransformer:
    """
    Instantiates or retrieves the cached SentenceTransformer model singleton.

    Returns:
        SentenceTransformer: Pre-trained embedding model instance.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of text segments into continuous vector representations.

    Args:
        texts (List[str]): Collection of text chunks.

    Returns:
        List[List[float]]: Corresponding dense embedding vectors.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_single_text(text: str) -> List[float]:
    """
    Encodes a single query string into a vector representation.

    Args:
        text (str): Input query text.

    Returns:
        List[float]: Dense vector representation.
    """
    return embed_texts([text])[0]


if __name__ == "__main__":
    sample_text = ["Deep learning applications in multimodal neurological diagnosis."]
    vector_output = embed_texts(sample_text)
    print(f"[INFO] Vector generation verified. Dimensionality: {len(vector_output[0])}")