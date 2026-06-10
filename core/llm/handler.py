"""
handler.py — Sends retrieved chunks to Claude and gets a cited answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE MOST IMPORTANT DESIGN DECISION IN THIS FILE:
    The system prompt STRICTLY constrains Claude to answer ONLY from
    the provided chunks. This is what prevents hallucination.

    Without this constraint, Claude would blend its training knowledge
    with your paper content — and you'd never know which was which.
    With it, Claude is a strict reader, not a general knowledge bot.

HOW THE RAG PROMPT IS STRUCTURED:
    System: "You are a research assistant. Answer ONLY from the chunks below.
             If the answer isn't in the chunks, say so explicitly."

    User:   "CONTEXT (retrieved chunks):
             [Paper: paper1.pdf | Page: 4]
             chunk text here...

             [Paper: paper2.pdf | Page: 12]
             chunk text here...

             CONVERSATION HISTORY:
             Q: previous question
             A: previous answer

             QUESTION: user's current question"

WHY FORMAT CHUNKS WITH PAPER NAME + PAGE?
    Claude will repeat this in its answer: "Paper1 (page 4) states that..."
    The format we inject becomes the citation format in the output.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import anthropic
import os
from dotenv import load_dotenv

# ─── Claude Configuration ─────────────────────────────────────
# API key and model are loaded at call time (not import time) so that
# edits to .env are picked up without restarting the app.
MAX_TOKENS = 2048   # Max length of Claude's response

# The strict system prompt — this is the anti-hallucination contract
SYSTEM_PROMPT = """You are ResearchRadar, a precise research paper analysis assistant.

YOUR STRICT RULES:
1. Answer ONLY using information from the CONTEXT chunks provided below.
2. NEVER use your general training knowledge to fill gaps. If something is not in the chunks, say: "This information is not found in the uploaded papers."
3. For EVERY claim you make, cite the source like this: (Paper: filename.pdf | Page: N)
4. When comparing multiple papers, structure your answer clearly — one paper at a time, then a synthesis.
5. Be analytical and precise. Do not pad answers with filler text.
6. If chunks contradict each other, highlight the contradiction explicitly.

Your goal: Be a trusted research instrument, not a chatbot. Every statement must be traceable."""


def _format_chunks_as_context(chunks: list) -> str:
    """
    Formats retrieved chunks into a structured context block for the prompt.

    Why this format?
        We label each chunk with its paper and page number.
        Claude reads these labels and repeats them in citations.
        This is how "Paper X (page 4) states..." appears in the answer.

    Args:
        chunks (list): Retrieved chunk dicts from retriever.py

    Returns:
        str: Formatted context block ready to paste into the prompt.
    """
    if not chunks:
        return "No relevant chunks found in the uploaded papers."

    context_parts = []

    for i, chunk in enumerate(chunks):
        # Group chunks by paper for cleaner context structure
        header = f"[Chunk {i+1} | Paper: {chunk['source']} | Page: {chunk['page']}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, chunks: list, conversation_history: str) -> str:
    """
    Assembles the full user message for Claude's API call.

    The prompt has three sections:
    1. CONTEXT:  The retrieved chunks (what Claude can answer from)
    2. HISTORY:  Previous Q&A turns (for follow-up question support)
    3. QUESTION: The current user question

    Args:
        question (str):             The user's current question.
        chunks (list):              Retrieved chunks from retriever.py
        conversation_history (str): Formatted string of past turns from memory/conversation.py

    Returns:
        str: Complete user message to send to Claude.
    """
    context_block = _format_chunks_as_context(chunks)

    history_block = (
        f"CONVERSATION HISTORY (for context on follow-up questions):\n{conversation_history}"
        if conversation_history.strip()
        else "CONVERSATION HISTORY: This is the first question in this session."
    )

    return f"""CONTEXT — Retrieved from uploaded research papers:

{context_block}

---

{history_block}

---

QUESTION: {question}

Remember: Answer ONLY from the CONTEXT chunks above. Cite every claim with (Paper: name | Page: N)."""


def get_answer(question: str, chunks: list, conversation_history: str = "") -> dict:
    """
    Sends the question + retrieved chunks to Claude and returns a cited answer.

    This is the main function called from app.py.

    Args:
        question (str):             The user's question.
        chunks (list):              Retrieved chunk dicts from retriever.py
        conversation_history (str): Previous Q&A context from conversation.py

    Returns:
        dict: {
            "answer":  str,   ← Claude's full response text
            "sources": list,  ← Unique (source, page) pairs for the citations sidebar
            "model":   str    ← Which Claude model was used
        }

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set in .env
    """
    # ─── Load config at call time (not import time) ──────────────────────
    # This ensures .env edits are picked up without restarting the app.
    load_dotenv(override=True)
    api_key     = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Copy .env.example → .env and add your Claude API key."
        )

    # Build the user message
    user_message = build_prompt(question, chunks, conversation_history)

    # ─── Call Claude API ─────────────────────────────────────────────────
    # anthropic.Anthropic() creates a client using the API key.
    # client.messages.create() sends a single request → response.
    # This is a synchronous (blocking) call — the app waits for the response.
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=claude_model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Extract the text content from Claude's response object
    answer_text = response.content[0].text

    # ─── Build Source Citations List ─────────────────────────────────────
    # Deduplicate: if two chunks came from the same paper + page, show it once
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper": chunk["source"],
                "page":  chunk["page"],
                "sentiment": chunk.get("chunk_sentiment", "neutral"),
            })

    return {
        "answer":  answer_text,
        "sources": sources,
        "model":   claude_model,
    }
