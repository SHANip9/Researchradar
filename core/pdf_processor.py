"""
core/pdf_processor.py — PDF Text Extraction & Semantic Chunking.

Handles the first stage of the indexing pipeline:
  1. Page-by-page text extraction via PDFPlumber.
  2. Academic section header detection (Abstract → References).
  3. Paragraph-aware sliding-window chunking with overlap.
"""

import hashlib
import re

import pdfplumber

from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from core.logger import get_logger, timed

log = get_logger("pdf_processor")

# ─── Section Detection ────────────────────────────────────────
KNOWN_SECTIONS: list[str] = [
    "Abstract", "Introduction", "Background", "Related Work",
    "Methodology", "Methods", "System Architecture", "Approach",
    "Experiments", "Evaluation", "Discussion", "Results",
    "Limitations", "Conclusion", "Future Work", "References",
]

_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(" + "|".join(KNOWN_SECTIONS) + r")\b",
    re.IGNORECASE | re.MULTILINE,
)


def detect_section(page_text: str, current_section: str = "Overview") -> str:
    """Detects academic section headers within a page's text.

    Args:
        page_text: Raw text of a single PDF page.
        current_section: The section label carried over from the previous page.

    Returns:
        The detected section name, or *current_section* if none found.
    """
    match = _SECTION_PATTERN.search(page_text)
    return match.group(1).strip().title() if match else current_section


# ─── Text Extraction ─────────────────────────────────────────
@timed
def extract_pdf(pdf_path: str) -> dict[int, str]:
    """Extracts text page-by-page from a PDF file.

    Args:
        pdf_path: Filesystem path to the PDF.

    Returns:
        A dict mapping 1-based page numbers to their extracted text.

    Raises:
        ValueError: If no readable text can be extracted.
    """
    pages: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages[i + 1] = text.strip()
            except Exception as exc:
                log.warning("Failed to extract page %d: %s", i + 1, exc)
                continue
    log.info("Extracted %d pages from PDF", len(pages))
    return pages


# ─── Chunking ─────────────────────────────────────────────────
def _split_into_paragraphs(text: str) -> list[str]:
    """Splits text into paragraphs with sentence-level fallback."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) <= 1 and len(text) > CHUNK_SIZE:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
    return paras


def _make_chunk_id(paper_name: str, page: int, index: int, text: str) -> str:
    """Generates a deterministic, collision-resistant chunk ID."""
    digest = hashlib.sha256(
        f"{paper_name}_{page}_{index}_{text[:40]}".encode()
    ).hexdigest()[:16]
    return f"{paper_name}_{digest}"


@timed
def chunk_document(
    pages: dict[int, str],
    paper_name: str,
    metadata: dict | None = None,
) -> list[dict]:
    """Splits extracted pages into semantically enriched chunks.

    Uses a sliding-window approach that respects paragraph boundaries and
    attaches section labels, page numbers, and unique IDs to every chunk.

    Args:
        pages: Output of :func:`extract_pdf`.
        paper_name: The original filename of the uploaded PDF.
        metadata: Optional extra metadata (``title``, ``author_org``, etc.).

    Returns:
        A list of chunk dicts ready for embedding and storage.
    """
    meta = metadata or {}
    chunks: list[dict] = []
    current_sec = "Introduction"

    for page_num, text in pages.items():
        current_sec = detect_section(text, current_sec)
        paragraphs = _split_into_paragraphs(text)

        running_chunk = ""
        for p in paragraphs:
            # Handle oversized paragraphs by splitting them
            p_parts = [p]
            if len(p) > CHUNK_SIZE:
                p_parts = []
                start = 0
                while start < len(p):
                    end = start + CHUNK_SIZE
                    p_parts.append(p[start:end].strip())
                    if end >= len(p):
                        break
                    start += CHUNK_SIZE - CHUNK_OVERLAP

            for part in p_parts:
                if len(running_chunk) + len(part) <= CHUNK_SIZE:
                    running_chunk += ("\n\n" + part if running_chunk else part)
                else:
                    if running_chunk:
                        chunks.append(_build_chunk(
                            running_chunk, paper_name, meta,
                            page_num, current_sec, len(chunks),
                        ))
                    running_chunk = part

        # Flush remaining content on this page
        if running_chunk:
            chunks.append(_build_chunk(
                running_chunk, paper_name, meta,
                page_num, current_sec, len(chunks),
            ))
            running_chunk = ""

    log.info("Chunked '%s' into %d chunks", paper_name, len(chunks))
    return chunks


def _build_chunk(
    text: str,
    paper_name: str,
    meta: dict,
    page: int,
    section: str,
    index: int,
) -> dict:
    """Assembles a single chunk dict with all required fields."""
    return {
        "id": _make_chunk_id(paper_name, page, index, text),
        "text": text,
        "paper_name": paper_name,
        "paper_title": meta.get("title", paper_name),
        "author_org": meta.get("author_org", "Unknown"),
        "category": meta.get("category", "General"),
        "page": page,
        "section": section,
    }

