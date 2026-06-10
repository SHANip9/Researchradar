"""
challenger.py — Mode A: AI challenges pre-extracted paper claims.

Acts as a rigorous peer reviewer, identifying assumptions and weaknesses in claims.
"""

import os
import anthropic
from dotenv import load_dotenv

# Resolve CLAIMS_DIR relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAIMS_DIR = os.path.join(PROJECT_ROOT, "data", "claims")

# Peer reviewer system instructions
SYSTEM_PROMPT = """You are a world-class, skeptical academic peer reviewer.
Your goal is to identify underlying assumptions, technical gaps, edge cases, and limitations in research paper claims.
Do not be polite or agreeable. Be analytical, precise, critical, and constructive.
Your critiques should be technical and directly address the logic of the claim."""


def challenge_claim(paper_title: str, claim_text: str, section: str) -> str:
    """
    Calls Claude to critique a single research claim.
    """
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment. Please add it to your .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"""Document: {paper_title}
Section: {section}
Claim: {claim_text}

Objection prompt: Critically analyze this claim. What are the unstated assumptions, potential execution limitations, missing control groups, or alternative explanations for these findings?"""

    response = client.messages.create(
        model=claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text


def challenge_paper_claims(paper_title: str) -> list[dict]:
    """
    Loads all claims for a paper and runs the challenger critique pipeline.

    Returns:
        list of dicts: [{"claim_text": str, "section": str, "critique": str}]
    """
    from core.ingestion.claim_extractor import load_claims
    
    claims = load_claims(paper_title, CLAIMS_DIR)
    if not claims:
        print(f"[Challenger] No claims found in storage for: {paper_title}")
        return []

    results = []
    print(f"[Challenger] Challenging {len(claims)} claims for '{paper_title}'...")

    # For safety/pacing, we will limit to the top 6 claims if there are too many
    for claim in claims[:6]:
        claim_text = claim["claim_text"]
        section = claim.get("section", "Unknown")
        
        try:
            critique = challenge_claim(paper_title, claim_text, section)
            results.append({
                "claim_text": claim_text,
                "section":    section,
                "critique":   critique
            })
        except Exception as e:
            print(f"[Challenger] Failed to challenge claim: {str(e)}")
            results.append({
                "claim_text": claim_text,
                "section":    section,
                "critique":   f"Error generating challenge: {str(e)}"
            })

    return results
