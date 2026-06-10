"""metadata_tagger.py — Detects section headings and enriches chunk metadata."""

import re

# ─── Known Academic Section Names ─────────────────────────────
KNOWN_SECTIONS = [
    "Abstract", "Introduction", "Background", "Related Work",
    "Methodology", "Methods", "Approach", "Architecture",
    "Experiments", "Results", "Discussion", "Conclusion",
    "References", "Appendix",
]

# ─── Section Detection Patterns ──────────────────────────────
# Ordered from most specific to least specific so the first match wins.
_SECTION_PATTERNS = [
    # Numbered sections: '1. Introduction', '2.1 Background'
    re.compile(
        r"^\s*\d+(?:\.\d+)*\.?\s+(" + "|".join(KNOWN_SECTIONS) + r")\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Roman-numeral sections: 'III. Methodology'
    re.compile(
        r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X{0,3})\.?\s+("
        + "|".join(KNOWN_SECTIONS)
        + r")\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    # All-caps headings on their own line: 'ABSTRACT', 'INTRODUCTION'
    re.compile(
        r"^\s*(" + "|".join(s.upper() for s in KNOWN_SECTIONS) + r")\s*$",
        re.MULTILINE,
    ),
    # Title-case headings on their own line: 'Related Work', 'Experimental Setup'
    re.compile(
        r"^\s*(" + "|".join(KNOWN_SECTIONS) + r")\s*$",
        re.MULTILINE,
    ),
    # Generic numbered heading (captures any heading text)
    re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+([A-Z][A-Za-z\s]+)$", re.MULTILINE),
]


def detect_sections(pages: dict) -> dict:
    """
    Scans page text for section-like headings and builds a position map.

    Args:
        pages: {page_num: page_text} from the PDF extractor.

    Returns:
        dict mapping (page_num, char_offset) tuples to section name strings.
        Sorted by position so downstream consumers can do bisect lookups.
    """
    section_map: dict = {}

    for page_num, text in pages.items():
        for pattern in _SECTION_PATTERNS:
            for match in pattern.finditer(text):
                # Normalise the heading to title-case for consistency
                heading = match.group(1).strip().title()
                section_map[(page_num, match.start())] = heading

    # Sort by (page, offset) so assign_section_to_chunk can walk in order
    return dict(sorted(section_map.items()))


def assign_section_to_chunk(chunk_text: str, chunk_page: int,
                            section_map: dict) -> str:
    """
    Returns the section name a chunk belongs to.

    Strategy: find the latest section heading that starts on or before
    this chunk's page. If the chunk text explicitly contains a heading,
    that takes priority.
    """
    # Direct match — chunk itself contains a known heading
    for name in KNOWN_SECTIONS:
        if re.search(rf"\b{re.escape(name)}\b", chunk_text, re.IGNORECASE):
            return name.title()

    # Walk the section map backwards and pick the most recent heading
    best_section = "Unknown"
    for (page, _offset), heading in section_map.items():
        if page <= chunk_page:
            best_section = heading
        else:
            break  # map is sorted, so no point continuing

    return best_section


def enrich_chunks(chunks: list, paper_metadata: dict,
                  section_map: dict) -> list:
    """
    Attaches paper-level metadata, section labels, and a sequential index
    to every chunk dict.

    Args:
        chunks:          list of chunk dicts from chunk_document().
        paper_metadata:  {paper_title, author_org, category, date}.
        section_map:     output of detect_sections().

    Returns:
        The same list, mutated in-place with extra keys.
    """
    title    = paper_metadata.get("paper_title", "")
    org      = paper_metadata.get("author_org", "")
    category = paper_metadata.get("category", "")
    date     = paper_metadata.get("date", "")

    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"]  = idx
        chunk["paper_title"]  = title
        chunk["author_org"]   = org
        chunk["category"]     = category
        chunk["date"]         = date
        chunk["section"]      = assign_section_to_chunk(
            chunk["text"], chunk["page"], section_map
        )

    return chunks
