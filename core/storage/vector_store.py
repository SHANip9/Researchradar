"""vector_store.py — Manages the ChromaDB vector database."""

import chromadb
from chromadb.config import Settings
import numpy as np
import os

# ─── Database Configuration ──────────────────────────────────
# File lives at core/storage/vector_store.py → go up 3 levels to project root.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
CLAIMS_DIR = os.path.join(PROJECT_ROOT, "data", "claims")

# Single collection — all papers live here, separated by metadata
COLLECTION_NAME = "researchradar_papers"

# ─── Singleton Connection ────────────────────────────────────
_client = None
_collection = None


def get_collection():
    """
    Returns the ChromaDB collection. Creates the DB connection on first call only.
    """
    global _client, _collection

    if _collection is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[Vector Store] Connected. Collection has {_collection.count()} chunks stored.")

    return _collection


# ─── Write Operations ────────────────────────────────────────

def add_paper_chunks(chunks: list, embeddings: np.ndarray, sentiments: list):
    """
    Stores all chunks (with vectors, sentiment, and enriched metadata) into ChromaDB.

    Args:
        chunks:     List of chunk dicts from pdf_processor (with enriched fields).
        embeddings: Shape (len(chunks), 384) — one vector per chunk.
        sentiments: List of {"label": str, "score": float} per chunk.
    """
    collection = get_collection()

    ids, documents, metadatas, embeddings_list = [], [], [], []

    for i, chunk in enumerate(chunks):
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])

        metadatas.append({
            "source":                chunk["source"],
            "page":                  chunk["page"],
            "char_count":            chunk["char_count"],
            "chunk_index":           chunk.get("chunk_index", i),
            "paper_title":           chunk.get("paper_title", chunk["source"]),
            "author_org":            chunk.get("author_org", ""),
            "category":              chunk.get("category", ""),
            "date":                  chunk.get("date", ""),
            "section":               chunk.get("section", "Unknown"),
            "chunk_sentiment":       sentiments[i]["label"],
            "chunk_sentiment_score": sentiments[i]["score"],
        })

        embeddings_list.append(embeddings[i].tolist())

    # upsert is safe for re-uploads (updates existing, inserts new)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list,
    )

    print(f"[Vector Store] Stored {len(chunks)} chunks for '{chunks[0]['source']}'.")


# ─── Read Operations ─────────────────────────────────────────

def semantic_search(query_embedding: np.ndarray, n_results: int = 20,
                    paper_filter: str = None) -> list:
    """
    Finds chunks most semantically similar to the query vector.

    Returns:
        list of dicts with text, source, page, sentiment, distance,
        and all enriched metadata fields.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    n_results = min(n_results, collection.count())

    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if paper_filter:
        query_params["where"] = {"source": paper_filter}

    results = collection.query(**query_params)

    formatted = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        formatted.append({
            "text":                  results["documents"][0][i],
            "source":                meta.get("source", "unknown"),
            "page":                  meta.get("page", 0),
            "chunk_index":           meta.get("chunk_index", 0),
            "paper_title":           meta.get("paper_title", meta.get("source", "unknown")),
            "author_org":            meta.get("author_org", ""),
            "category":              meta.get("category", ""),
            "date":                  meta.get("date", ""),
            "section":               meta.get("section", "Unknown"),
            "chunk_sentiment":       meta.get("chunk_sentiment", "neutral"),
            "chunk_sentiment_score": meta.get("chunk_sentiment_score", 0.5),
            "distance":              results["distances"][0][i],
        })

    return formatted


def get_all_chunks() -> list:
    """
    Retrieves ALL stored chunks with full metadata.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["documents", "metadatas"])

    return [
        {
            "text":                  results["documents"][i],
            "source":                results["metadatas"][i].get("source", "unknown"),
            "page":                  results["metadatas"][i].get("page", 0),
            "id":                    results["ids"][i],
            "chunk_index":           results["metadatas"][i].get("chunk_index", 0),
            "paper_title":           results["metadatas"][i].get("paper_title", results["metadatas"][i].get("source", "unknown")),
            "section":               results["metadatas"][i].get("section", "Unknown"),
            "author_org":            results["metadatas"][i].get("author_org", ""),
            "category":              results["metadatas"][i].get("category", ""),
            "date":                  results["metadatas"][i].get("date", ""),
            "chunk_sentiment":       results["metadatas"][i].get("chunk_sentiment", "neutral"),
            "chunk_sentiment_score": results["metadatas"][i].get("chunk_sentiment_score", 0.5),
        }
        for i in range(len(results["documents"]))
    ]


def get_chunks_with_embeddings(paper_name: str | None = None) -> list:
    """
    Retrieves chunks with their embeddings from ChromaDB.
    If paper_name is provided, filters by that paper.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    params = {"include": ["documents", "metadatas", "embeddings"]}
    if paper_name:
        params["where"] = {"source": paper_name}

    results = collection.get(**params)

    if not results or not results["documents"]:
        return []

    return [
        {
            "id":                    results["ids"][i],
            "text":                  results["documents"][i],
            "embedding":             results["embeddings"][i],
            "source":                results["metadatas"][i].get("source", "unknown"),
            "page":                  results["metadatas"][i].get("page", 0),
            "chunk_index":           results["metadatas"][i].get("chunk_index", 0),
            "paper_title":           results["metadatas"][i].get("paper_title", results["metadatas"][i].get("source", "unknown")),
            "section":               results["metadatas"][i].get("section", "Unknown"),
            "author_org":            results["metadatas"][i].get("author_org", ""),
            "category":              results["metadatas"][i].get("category", ""),
            "date":                  results["metadatas"][i].get("date", ""),
            "chunk_sentiment":       results["metadatas"][i].get("chunk_sentiment", "neutral"),
            "chunk_sentiment_score": results["metadatas"][i].get("chunk_sentiment_score", 0.5),
        }
        for i in range(len(results["documents"]))
    ]



def get_papers_list() -> list:
    """Returns a sorted list of unique paper names stored in the DB."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    return sorted({m.get("source", "unknown") for m in results["metadatas"]})


def get_paper_sentiment_data(paper_name: str) -> list:
    """
    Returns all chunk sentiment records for a specific paper.
    Used by sentiment_analyzer.py to compute the paper-level aggregate.
    """
    collection = get_collection()
    results = collection.get(
        where={"source": paper_name},
        include=["metadatas"],
    )
    return [
        {
            "label": m.get("chunk_sentiment", "neutral"),
            "score": m.get("chunk_sentiment_score", 0.5),
        }
        for m in results["metadatas"]
    ]


def get_paper_metadata(paper_name: str) -> dict:
    """
    Returns the metadata for the first chunk of a paper (paper-level info).
    """
    collection = get_collection()
    results = collection.get(
        where={"source": paper_name},
        include=["metadatas"],
        limit=1,
    )

    if not results["metadatas"]:
        return {}

    meta = results["metadatas"][0]
    return {
        "source":      meta.get("source", paper_name),
        "paper_title": meta.get("paper_title", meta.get("source", paper_name)),
        "author_org":  meta.get("author_org", ""),
        "category":    meta.get("category", ""),
        "date":        meta.get("date", ""),
    }


def delete_paper(paper_name: str):
    """Removes all chunks for a specific paper from ChromaDB."""
    collection = get_collection()
    collection.delete(where={"source": paper_name})
    print(f"[Vector Store] Deleted all chunks for: {paper_name}")


def clear_all():
    """Deletes ALL data from the database."""
    global _client, _collection
    collection = get_collection()
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    print("[Vector Store] All data cleared.")
