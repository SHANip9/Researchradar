"""
defender.py — Mode B: User attacks, AI defends.

Retrieves relevant chunks from the selected paper and constructs a grounded defense.
"""

import os
import anthropic
from dotenv import load_dotenv
from core.embeddings.embedder import embed_query
from core.storage.vector_store import semantic_search

SYSTEM_PROMPT_TEMPLATE = """You are a staunch academic defender (defense counsel) of the research paper '{paper_title}'.
Your task is to defend the paper's claims and methodologies against the user's objections.

STRICT LAWS OF DEFENSE:
1. Ground every point in the provided CONTEXT chunks. Do not make up arguments.
2. Quote passages directly and cite the page numbers, e.g., (Page 4).
3. If the retrieved context does not contain relevant information to refute the user's objection, you must explicitly admit this limitation: "The paper's text does not contain evidence to counter this objection."
4. Maintain a formal, academic, and firm tone. Focus on text-based evidence, not general assertions."""


def defend_paper_against_objection(
    paper_title: str,
    source_file: str,
    objection: str,
    conversation_history: list | None = None
) -> dict:
    """
    Retrieves chunks relevant to the user's objection from the specific paper
    and calls Claude to defend the paper using only retrieved evidence.

    Returns:
        dict: {"defense": str, "retrieved_chunks": list}
    """
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment.")

    # ─── Step 1: Query ChromaDB for Relevant Chunks in selected paper ───
    query_vector = embed_query(objection)
    # Fetch top 5 chunks specifically for this paper
    chunks = semantic_search(query_vector, n_results=5, paper_filter=source_file)

    if not chunks:
        return {
            "defense": "No relevant text chunks could be retrieved from the paper to formulate a defense.",
            "retrieved_chunks": []
        }

    # ─── Step 2: Format Chunks as Context ──────────────────────────────
    context_parts = []
    for i, c in enumerate(chunks):
        header = f"[Chunk {i+1} | Page: {c['page']} | Section: {c.get('section', 'Unknown')}]"
        context_parts.append(f"{header}\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # ─── Step 3: Format Conversation History ────────────────────────────
    messages = []
    if conversation_history:
        # Map conversation history list of tuples (role, text) to Claude API schema
        for role, text in conversation_history:
            messages.append({"role": role, "content": text})

    # Add the current user objection and context
    user_content = f"""USER'S OBJECTION / ATTACK:
"{objection}"

CONTEXT CHUNKS FOR DEFENSE (use only these to refute the user):
{context_block}

Refute the user's objection based strictly on the context chunks above. Cite direct quotes and page numbers."""
    
    messages.append({"role": "user", "content": user_content})

    # ─── Step 4: Call Claude API ─────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(paper_title=paper_title)

    response = client.messages.create(
        model=claude_model,
        max_tokens=1536,
        system=system_prompt,
        messages=messages
    )

    return {
        "defense":          response.content[0].text,
        "retrieved_chunks": chunks
    }


def defend_papers_against_objection(
    paper_titles: list[str],
    source_files: list[str],
    objection: str,
    conversation_history: list | None = None
) -> dict:
    """
    Retrieves chunks relevant to the user's objection across multiple papers
    and calls Claude to defend the collective consensus or resolve arguments
    grounded only in the retrieved text.
    """
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment.")

    # Query ChromaDB for each paper
    from core.embeddings.embedder import embed_query
    from core.storage.vector_store import semantic_search
    query_vector = embed_query(objection)
    all_chunks = []
    
    # We retrieve 3 chunks per paper to keep context size manageable
    for src in source_files:
        paper_chunks = semantic_search(query_vector, n_results=3, paper_filter=src)
        if paper_chunks:
            all_chunks.extend(paper_chunks)

    if not all_chunks:
        return {
            "defense": "No relevant text chunks could be retrieved from the selected papers to formulate a consensus defense.",
            "retrieved_chunks": []
        }

    # Format Chunks as Context
    context_parts = []
    for i, c in enumerate(all_chunks):
        title = c.get("paper_title", c.get("source", "unknown"))
        header = f"[Chunk {i+1} | Paper: {title} | Page: {c['page']} | Section: {c.get('section', 'Unknown')}]"
        context_parts.append(f"{header}\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    messages = []
    if conversation_history:
        for role, text in conversation_history:
            messages.append({"role": role, "content": text})

    user_content = f"""USER'S OBJECTION / ATTACK:
"{objection}"

CONTEXT CHUNKS FOR DEFENSE (use only these to refute the user):
{context_block}

Refute the user's objection based strictly on the context chunks above. Cite direct quotes, paper names, and page numbers."""
    
    messages.append({"role": "user", "content": user_content})

    client = anthropic.Anthropic(api_key=api_key)
    
    titles_str = ", ".join([f"'{t}'" for t in paper_titles])
    system_prompt = f"""You are a staunch academic defender (defense counsel) of the research papers: {titles_str}.
Your task is to defend the papers' claims and methodologies against the user's objections.

STRICT LAWS OF DEFENSE:
1. Ground every point in the provided CONTEXT chunks. Do not make up arguments.
2. Quote passages directly and cite the paper name and page numbers, e.g., (BERT, Page 4).
3. If the retrieved context does not contain relevant information to refute the user's objection, you must explicitly admit this limitation: "The papers' texts do not contain evidence to counter this objection."
4. Maintain a formal, academic, and firm tone. Focus on text-based evidence, not general assertions."""

    response = client.messages.create(
        model=claude_model,
        max_tokens=1536,
        system=system_prompt,
        messages=messages
    )

    return {
        "defense":          response.content[0].text,
        "retrieved_chunks": all_chunks
    }

