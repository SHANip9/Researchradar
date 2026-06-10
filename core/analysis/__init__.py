"""
core/analysis — NLP analysis engines for research paper insights.

Exposes:
- sentiment: Chunk, section, and paper-level sentiment aggregation.
- similarity: Semantic cosine similarity calculations (paper & section levels).
- topics: Unsupervised topic discovery using BERTopic.
- keywords: Contextual keyword extraction using KeyBERT.
"""

from core.analysis.sentiment import aggregate_paper_sentiment, aggregate_section_sentiment
from core.analysis.similarity import compute_paper_similarity_matrix, compute_section_similarity_matrix
from core.analysis.topics import discover_topics
from core.analysis.keywords import extract_keywords_for_all_papers
from core.analysis.powerbi_checker import find_powerbi_port, trigger_live_refresh

