"""
versus.py — Mode C: Paper A vs. Paper B debate versus mode.

Retrieves chunks for both papers and simulates a multi-turn, text-grounded academic debate.
"""

import os
import anthropic
from dotenv import load_dotenv
from core.embeddings.embedder import embed_query
from core.storage.vector_store import semantic_search

SYSTEM_PROMPT_TEMPLATE = """You are representing the authors of the research paper '{current_paper}'.
You are engaging in a rigorous academic debate against the paper '{opposing_paper}' on the topic: "{topic}".

STRICT RULES OF ENGAGEMENT:
1. Ground your arguments ONLY in the provided text chunks of your own paper. Do not invent details.
2. Quote passages directly and cite the page numbers, e.g., (Page 4).
3. If the opposing paper makes a point, directly push back using your paper's methodology or findings.
4. Keep your speech concise and professional (approx. 150-250 words). Focus on technical differences."""


def format_context_block(chunks: list) -> str:
    """Formats list of chunks for the debate prompt context."""
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[Page: {c['page']} | Section: {c.get('section', 'Unknown')}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def run_paper_versus_debate(
    title_a: str, source_a: str,
    title_b: str, source_b: str,
    topic: str
) -> list[dict]:
    """
    Simulates a 4-turn debate between Paper A and Paper B on the chosen topic.

    Returns:
        list of dicts: [{"speaker": str, "speech": str}]
    """
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment.")

    client = anthropic.Anthropic(api_key=api_key)

    # ─── Step 1: Retrieve Chunks for Both Papers ───────────────────────
    query_vector = embed_query(topic)
    chunks_a = semantic_search(query_vector, n_results=4, paper_filter=source_a)
    chunks_b = semantic_search(query_vector, n_results=4, paper_filter=source_b)

    if not chunks_a or not chunks_b:
        return [
            {
                "speaker": "System Error",
                "speech":  "Could not retrieve sufficient chunks from both papers to initiate a debate."
            }
        ]

    context_a = format_context_block(chunks_a)
    context_b = format_context_block(chunks_b)

    debate_transcript = []

    # ─── Turn 1: Paper A Opening Statement ──────────────────────────────
    print("[Versus Debate] Generating Turn 1 (Paper A Opening)...")
    system_a = SYSTEM_PROMPT_TEMPLATE.format(current_paper=title_a, opposing_paper=title_b, topic=topic)
    user_a1 = f"""Here is your paper's context chunks:
{context_a}

Make your opening statement on the debate topic: "{topic}". Focus on explaining your paper's stance using ONLY the chunks above."""

    response_a1 = client.messages.create(
        model=claude_model,
        max_tokens=800,
        system=system_a,
        messages=[{"role": "user", "content": user_a1}]
    )
    speech_a1 = response_a1.content[0].text
    debate_transcript.append({"speaker": title_a, "speech": speech_a1})

    # ─── Turn 2: Paper B Counter-Argument ─────────────────────────────
    print("[Versus Debate] Generating Turn 2 (Paper B Response)...")
    system_b = SYSTEM_PROMPT_TEMPLATE.format(current_paper=title_b, opposing_paper=title_a, topic=topic)
    user_b1 = f"""Here is your paper's context chunks:
{context_b}

The authors of '{title_a}' made the following opening statement:
"{speech_a1}"

Refute their statement and explain why your paper's approach on "{topic}" is superior or different, using ONLY your context chunks."""

    response_b1 = client.messages.create(
        model=claude_model,
        max_tokens=800,
        system=system_b,
        messages=[{"role": "user", "content": user_b1}]
    )
    speech_b1 = response_b1.content[0].text
    debate_transcript.append({"speaker": title_b, "speech": speech_b1})

    # ─── Turn 3: Paper A Rebuttal ──────────────────────────────────────
    print("[Versus Debate] Generating Turn 3 (Paper A Rebuttal)...")
    user_a2 = f"""Here is your paper's context chunks:
{context_a}

We recap the debate on: "{topic}"

Your Opening: "{speech_a1}"
Their Counter-Argument: "{speech_b1}"

Rebut their counter-argument and defend your stance, citing quotes and evidence ONLY from your context chunks."""

    # Maintain session context by passing past turns
    messages_a2 = [
        {"role": "user", "content": user_a1},
        {"role": "assistant", "content": speech_a1},
        {"role": "user", "content": f"The authors of '{title_b}' responded: \"{speech_b1}\"\n\nRebut their argument."}
    ]

    response_a2 = client.messages.create(
        model=claude_model,
        max_tokens=800,
        system=system_a,
        messages=messages_a2
    )
    speech_a2 = response_a2.content[0].text
    debate_transcript.append({"speaker": title_a, "speech": speech_a2})

    # ─── Turn 4: Paper B Closing Statement ─────────────────────────────
    print("[Versus Debate] Generating Turn 4 (Paper B Closing)...")
    messages_b2 = [
        {"role": "user", "content": user_b1},
        {"role": "assistant", "content": speech_b1},
        {"role": "user", "content": f"The authors of '{title_a}' rebutted: \"{speech_a2}\"\n\nProvide your closing statements."}
    ]

    response_b2 = client.messages.create(
        model=claude_model,
        max_tokens=800,
        system=system_b,
        messages=messages_b2
    )
    speech_b2 = response_b2.content[0].text
    debate_transcript.append({"speaker": title_b, "speech": speech_b2})

    print("[Versus Debate] Debate completed successfully.")
    return debate_transcript
