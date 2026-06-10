"""
core/debate — AI-powered research debate engine.

Exposes:
- Mode A (Challenger): AI peer reviews paper claims.
- Mode B (Defender): AI defends paper against user attacks using only quotes.
- Mode C (Versus): Automated academic debate between two papers.
"""

from core.debate.challenger import challenge_paper_claims
from core.debate.defender import defend_paper_against_objection, defend_papers_against_objection
from core.debate.versus import run_paper_versus_debate
from core.debate.crossexam import run_cross_examination


