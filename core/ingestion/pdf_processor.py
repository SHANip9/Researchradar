"""pdf_processor.py — Extracts text from PDFs, chunks it, and enriches metadata."""

import hashlib
import os

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.ingestion.metadata_tagger import detect_sections, enrich_chunks

# ─── Chunking Configuration ──────────────────────────────────
CHUNK_SIZE = 2000       # characters per chunk  (~512 tokens)
CHUNK_OVERLAP = 200     # characters of overlap  (~50 tokens)


def extract_text_by_page(pdf_path: str) -> dict:
    """
    Reads a PDF and returns its text organised by page number.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        {page_number (1-indexed): page_text}
    """
    pages: dict = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            # Skip image-only pages with no extractable text
            if text and text.strip():
                pages[i + 1] = text.strip()

    return pages


def chunk_document(pages: dict, paper_name: str) -> list:
    """
    Splits all pages into overlapping chunks with metadata.

    Each chunk dict contains:
        text, source, page, chunk_id, char_count, chunk_index

    Args:
        pages:      Output of extract_text_by_page().
        paper_name: Filename, e.g. "attention_is_all_you_need.pdf".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list = []
    chunk_counter = 0

    for page_num, page_text in pages.items():
        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:
            # ─── Deterministic SHA-256 Chunk ID ───────────────────
            hash_input = f"{paper_name}_{page_num}_{chunk_text}".encode()
            chunk_hash = hashlib.sha256(hash_input).hexdigest()[:16]
            chunk_id = f"{paper_name}_p{page_num}_{chunk_hash}"

            all_chunks.append({
                "text":        chunk_text,
                "source":      paper_name,
                "page":        page_num,
                "chunk_id":    chunk_id,
                "char_count":  len(chunk_text),
                "chunk_index": chunk_counter,
            })
            chunk_counter += 1

    return all_chunks


def process_pdf(pdf_path: str, paper_metadata: dict | None = None) -> list:
    """
    Full pipeline: PDF → list of enriched chunks ready for embedding.

    Args:
        pdf_path:       Path to the PDF file.
        paper_metadata: Optional dict with keys paper_title, author_org,
                        category, date.  Falls back to filename/empty strings.

    Returns:
        list of chunk dicts with all metadata fields.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    paper_name = os.path.basename(pdf_path)

    # ─── Build metadata defaults ──────────────────────────────
    if paper_metadata is None:
        paper_metadata = {}
    paper_metadata.setdefault("paper_title", paper_name)
    paper_metadata.setdefault("author_org", "")
    paper_metadata.setdefault("category", "")
    paper_metadata.setdefault("date", "")

    print(f"[PDF Processor] Reading: {paper_name}")
    pages = extract_text_by_page(pdf_path)

    if not pages:
        raise ValueError(
            f"No text found in '{paper_name}'. "
            "It may be a scanned/image-only PDF. Try running OCR on it first."
        )

    print(f"[PDF Processor] {len(pages)} pages extracted. Splitting into chunks...")
    chunks = chunk_document(pages, paper_name)

    # ─── Section detection & metadata enrichment ──────────────
    section_map = detect_sections(pages)
    chunks = enrich_chunks(chunks, paper_metadata, section_map)

    print(f"[PDF Processor] Done — {len(chunks)} chunks created from {len(pages)} pages.")
    return chunks
