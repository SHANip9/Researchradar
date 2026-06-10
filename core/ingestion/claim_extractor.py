"""claim_extractor.py — Identifies academic claims in paper chunks."""

import json
import os
import re

# ─── Claim Patterns ──────────────────────────────────────────
# Compiled regexes that match typical academic claim language.
CLAIM_PATTERNS = [
    re.compile(
        r"\bwe\s+(show|find|demonstrate|propose|achieve|observe|introduce|present)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bresults\s+(indicate|suggest|show|demonstrate|confirm|reveal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bour\s+(model|approach|method|framework|system)\s+"
        r"(achieves|outperforms|surpasses|improves)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthis\s+(paper|work|study)\s+"
        r"(presents|introduces|proposes|contributes)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(significantly|substantially|consistently)\s+"
        r"(better|worse|higher|lower)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(state-of-the-art|novel|first|new)\s+"
        r"(approach|method|technique|framework)\b",
        re.IGNORECASE,
    ),
]

# ─── Sentence Splitting ──────────────────────────────────────
# Rough sentence boundary: period/question-mark/exclamation followed by space + capital
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_claims_from_text(text: str) -> list[str]:
    """
    Splits *text* into sentences and returns those matching any claim pattern.
    """
    sentences = _SENTENCE_RE.split(text)
    claims: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern in CLAIM_PATTERNS:
            if pattern.search(sentence):
                claims.append(sentence)
                break  # one match is enough per sentence

    return claims


def extract_claims(chunks: list) -> list[dict]:
    """
    Runs claim extraction over every chunk.

    Returns:
        list of claim dicts, each with:
        {claim_text, chunk_id, paper_title, section, page, chunk_index}
    """
    all_claims: list[dict] = []

    for chunk in chunks:
        matched = extract_claims_from_text(chunk["text"])
        for claim_text in matched:
            all_claims.append({
                "claim_text":  claim_text,
                "chunk_id":    chunk.get("chunk_id", ""),
                "paper_title": chunk.get("paper_title", ""),
                "section":     chunk.get("section", "Unknown"),
                "page":        chunk.get("page", 0),
                "chunk_index": chunk.get("chunk_index", 0),
            })

    return all_claims


# ─── Persistence ──────────────────────────────────────────────

def _claims_path(paper_title: str, output_dir: str) -> str:
    """Deterministic filename derived from the paper title."""
    safe_name = re.sub(r"[^\w\-]+", "_", paper_title).strip("_")
    return os.path.join(output_dir, f"{safe_name}_claims.json")


def save_claims(claims: list, paper_title: str, output_dir: str) -> None:
    """Persists extracted claims to a JSON file in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = _claims_path(paper_title, output_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=2, ensure_ascii=False)

    print(f"[Claim Extractor] Saved {len(claims)} claims -> {path}")

def load_claims(paper_title: str, output_dir: str) -> list:
    """Loads previously saved claims. Returns an empty list if file is missing."""
    path = _claims_path(paper_title, output_dir)

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
