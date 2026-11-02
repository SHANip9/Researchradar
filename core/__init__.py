"""
core — ResearchRadar Modular Intelligence Engine.

Exposes clean facades and specialized submodules:
  - config: Application parameters & environment configuration
  - logger: Structured logging & performance tracking
  - security: File validation & input sanitization
  - metrics: Telemetry, latency records & health monitoring
  - embedder: Local SentenceTransformer vector model
  - pdf_processor: PDF text extraction & structured chunking
  - vector_store: ChromaDB vector operations
  - retriever: Hybrid dense + sparse RRF search
  - llm_handler: Claude API & grounded local synthesis
  - query_router: Slash command dispatcher & RAG router
  - plugins: Specialized multi-agent intelligence plugins
  - analytics: Bias inspector, similarity heatmap & rigor scoring
  - ui: Modern obsidian-slate UI design system
"""

from core.engine import (
    ingest_paper,
    get_indexed_papers,
    get_all_chunks,
    hybrid_search,
    generate_llm_response,
    delete_paper,
    clear_database,
)
from core.query_router import parse_and_route_query
from core.security import validate_pdf_upload, sanitize_user_input, sanitize_filename
from core.metrics import metrics
from core.logger import get_logger

__all__ = [
    "ingest_paper",
    "get_indexed_papers",
    "get_all_chunks",
    "hybrid_search",
    "generate_llm_response",
    "delete_paper",
    "clear_database",
    "parse_and_route_query",
    "validate_pdf_upload",
    "sanitize_user_input",
    "sanitize_filename",
    "metrics",
    "get_logger",
]
