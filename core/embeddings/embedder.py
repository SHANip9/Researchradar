"""
embedder.py — Converts text into numerical vectors (embeddings)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AN EMBEDDING?
    A vector is just a list of numbers. For example:
    "The transformer model is efficient" → [0.23, -0.45, 0.12, ..., 0.67]  (384 numbers)

    Similar sentences have similar vectors. This is how semantic search works —
    we find chunks whose vectors are "close" to the question's vector in space.

WHY all-MiniLM-L6-v2?
    - Free, runs locally (no API cost for embeddings)
    - Produces 384-dimensional vectors (compact = fast)
    - Strong performance on semantic similarity tasks
    - Industry standard for local RAG systems
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# ─── Model Configuration ──────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Singleton: Load model once, reuse across the session ─────
# Loading a transformer model takes ~2-3 seconds.
# If we loaded it every time we embedded a chunk, the app would be unbearably slow.
# Python's global variable trick: _model starts as None, gets populated on first call.
_model = None


def get_model() -> SentenceTransformer:
    """
    Returns the embedding model. Loads it from disk/cache only on first call.
    After that, returns the already-loaded model instantly.
    This is the Singleton design pattern.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading '{MODEL_NAME}' — this happens only once...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[Embedder] Model ready.")
    return _model


def embed_texts(texts: list) -> np.ndarray:
    """
    Converts a list of text strings into embedding vectors (batch processing).

    WHY BATCH PROCESSING?
        Instead of embedding one chunk at a time, we send all chunks to the model
        together. The GPU/CPU processes them in parallel → much faster.

    Args:
        texts (list of str): e.g., ["chunk 1 text", "chunk 2 text", ...]

    Returns:
        numpy array of shape (len(texts), 384)
        Each row is the 384-dimensional vector for one text.

    Example:
        chunks = ["Attention is all you need", "BERT uses bidirectional encoding"]
        vectors = embed_texts(chunks)
        # vectors.shape → (2, 384)
    """
    model = get_model()
    # show_progress_bar=True prints a tqdm progress bar — useful for large papers
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def embed_query(question: str) -> np.ndarray:
    """
    Converts a single question into an embedding vector.

    WHY SEPARATE FROM embed_texts?
        Some embedding models use different internal prompts for
        "query" vs "document" (asymmetric embedding models).
        Keeping them separate future-proofs the code for model swaps.

    Args:
        question (str): The user's question as a plain string.

    Returns:
        numpy array of shape (384,) — a single 384-dimensional vector.
    """
    model = get_model()
    return model.encode(question, convert_to_numpy=True)
