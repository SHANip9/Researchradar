"""
core/config.py — Centralized Configuration Management for ResearchRadar.

Single source of truth for all tunable parameters, paths, and environment
variables. Import from here instead of scattering os.getenv() calls.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ─── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
CLAIMS_DIR = DATA_DIR / "claims"
EXPORTS_DIR = DATA_DIR / "exports"

# ─── Embedding Model ─────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ─── Chunking ─────────────────────────────────────────────────
# 1000 characters ≈ 200–240 WordPiece tokens, fitting strictly within
# all-MiniLM-L6-v2's 256-token context window.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# ─── ChromaDB ─────────────────────────────────────────────────
COLLECTION_NAME: str = "researchradar_v3"

# ─── LLM ──────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2500"))

# ─── Retrieval ────────────────────────────────────────────────
TOP_K_PRECISE: int = int(os.getenv("TOP_K_PRECISE", "8"))
TOP_K_PER_PAPER: int = int(os.getenv("TOP_K_PER_PAPER", "3"))
RRF_K: int = 60  # Reciprocal Rank Fusion constant

# ─── Security ─────────────────────────────────────────────────
MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "500"))
MAX_QUERY_LENGTH: int = 2000
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

# ─── Application ──────────────────────────────────────────────
APP_NAME: str = "ResearchRadar"
APP_VERSION: str = "3.1.0"
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

