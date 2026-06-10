"""keywords.py — Context-rich semantic keyword extraction using KeyBERT."""

from keybert import KeyBERT
from core.embeddings.embedder import get_model
from core.storage.vector_store import get_papers_list, get_all_chunks

# Singleton: cache KeyBERT instance to prevent reloading
_kw_model = None


def get_keybert_model() -> KeyBERT:
    """
    Returns the KeyBERT model instance.
    Reuses the existing SentenceTransformer model to save memory and loading time.
    """
    global _kw_model
    if _kw_model is None:
        print("[KeyBERT] Initializing KeyBERT model with active SentenceTransformer...")
        # Get our singleton sentence-transformer model
        sentence_model = get_model()
        # Initialize KeyBERT using the shared transformer model
        _kw_model = KeyBERT(model=sentence_model)
        print("[KeyBERT] Model ready.")
    return _kw_model


def extract_keywords_from_text(text: str, top_n: int = 15) -> list[tuple[str, float]]:
    """
    Extracts the top semantic keyphrases from a block of text using KeyBERT.
    
    Args:
        text: The source text.
        top_n: Number of keywords to return.

    Returns:
        List of tuples: [("keyword", score), ...]
    """
    if not text.strip():
        return []

    kw_model = get_keybert_model()
    
    # We use 1-grams and 2-grams, filtering out English stop words.
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )
    return keywords


def extract_keywords_for_all_papers() -> list[dict]:
    """
    Reconstructs each paper's text from its vector database chunks and
    runs KeyBERT to extract the top-15 keyphrases.

    Returns:
        list of dicts: [{"paper_title": str, "keyword": str, "relevance_score": float}]
    """
    papers = get_papers_list()
    if not papers:
        print("[KeyBERT] No papers found in storage to extract keywords from.")
        return []

    # Get all chunks to reconstruct paper texts in-memory
    all_chunks = get_all_chunks()
    
    # Group chunk texts by paper
    paper_texts = {}
    for chunk in all_chunks:
        # Resolve paper title or fallback to source filename
        title = chunk.get("paper_title", chunk.get("source", "unknown"))
        paper_texts.setdefault(title, []).append(chunk)

    # Sort chunks by chunk_index to reconstruct text in reading order
    for title in paper_texts:
        paper_texts[title] = sorted(paper_texts[title], key=lambda x: x.get("chunk_index", 0))

    keyword_data = []

    for title, chunks in paper_texts.items():
        print(f"[KeyBERT] Extracting keywords for: {title}...")
        
        # Combine all chunk texts into one string
        full_text = " ".join([c["text"] for c in chunks])
        
        # Run KeyBERT extraction
        keywords_scores = extract_keywords_from_text(full_text, top_n=15)
        
        for kw, score in keywords_scores:
            keyword_data.append({
                "paper_title":     title,
                "keyword":         kw,
                "relevance_score": round(float(score), 4)
            })

    print(f"[KeyBERT] Extracted keywords for {len(paper_texts)} papers.")
    return keyword_data
