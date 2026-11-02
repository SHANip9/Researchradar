"""
core/embedder.py — Embedding Model Management for ResearchRadar.

Loads and caches the SentenceTransformer model as a thread-safe singleton.
Provides batch and single-query embedding functions used by both the
ingestion and retrieval pipelines.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL_NAME
from core.logger import get_logger, timed

log = get_logger("embedder")

# ─── Singleton ────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    """Loads and caches the local SentenceTransformer model.

    Returns:
        The cached ``SentenceTransformer`` instance.
    """
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        log.info("Embedding model loaded successfully")
    return _model


@timed
def embed_texts(texts: list[str]) -> np.ndarray:
    """Encodes a batch of texts into normalised embedding vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        A ``(len(texts), 384)`` numpy array of L2-normalised vectors.
    """
    model = get_embed_model()
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    log.debug("Embedded %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encodes a single query string into a normalised embedding vector.

    Args:
        query: The search query text.

    Returns:
        A 1-D numpy array of shape ``(384,)``.
    """
    model = get_embed_model()
    return model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

