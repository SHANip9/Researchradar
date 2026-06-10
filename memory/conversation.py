"""
conversation.py — Stores and retrieves conversation history for follow-up questions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CONVERSATION MEMORY?
    Without memory, every question is isolated.
    "Which paper disagreed with that?" — the LLM has no idea what "that" refers to.

    With memory, we pass the last N turns to Claude in every prompt,
    so it can resolve references like "that", "the previous paper", "their method", etc.

DESIGN CHOICE — WINDOW MEMORY (last N turns):
    We don't keep the entire conversation history (that would grow the prompt
    indefinitely and eventually hit token limits).
    Instead, we keep a sliding window of the last MAX_TURNS turns.
    For a research session, 5 turns of context is almost always enough.

HOW IT FITS INTO THE PIPELINE:
    User asks question
    → memory.get_history() → appended to prompt
    → Claude answers with context of past Q&A
    → memory.add_turn(question, answer) → stored for next question
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import deque

# ─── Configuration ────────────────────────────────────────────
MAX_TURNS = 5   # Keep only the last 5 Q&A pairs in the prompt context
                # Increase if you want more follow-up context (increases token usage)


class ConversationMemory:
    """
    A sliding-window conversation memory for one research session.

    Each "turn" is one (question, answer) pair.
    We keep the last MAX_TURNS turns and format them as a readable string
    for injection into the Claude prompt.

    Uses collections.deque(maxlen=MAX_TURNS) for O(1) automatic eviction
    of the oldest turn when capacity is exceeded (vs O(n) for list.pop(0)).

    Usage in app.py:
        memory = ConversationMemory()                # Create at session start
        memory.add_turn(question, answer)            # After each answer
        history = memory.get_history()               # Pass to llm_handler
        memory.clear()                               # On "New Session" button
    """

    def __init__(self):
        # Internal storage: deque of {"question": str, "answer": str}
        # maxlen=MAX_TURNS → oldest turn is automatically dropped when full
        self._turns = deque(maxlen=MAX_TURNS)

    def add_turn(self, question: str, answer: str):
        """
        Records a completed Q&A turn.

        Args:
            question (str): What the user asked.
            answer (str):   What Claude responded with.
        """
        # deque with maxlen handles eviction automatically —
        # when full, appending a new item drops the oldest one in O(1).
        self._turns.append({"question": question, "answer": answer})

    def get_history(self) -> str:
        """
        Returns conversation history as a formatted string for the LLM prompt.

        Format:
            Q1: What is the training method in paper 1?
            A1: Paper 1 uses ... (Paper: attention.pdf | Page: 3)

            Q2: Which paper disagreed with that?
            A2: ...

        Returns:
            str: Formatted history string, or empty string if no history yet.
        """
        if not self._turns:
            return ""

        history_lines = []
        for i, turn in enumerate(self._turns):
            history_lines.append(f"Q{i+1}: {turn['question']}")
            # Truncate very long answers in the history to save tokens
            # (The full answer was already seen by the user — we just need context)
            answer_preview = turn["answer"][:500] + "..." if len(turn["answer"]) > 500 else turn["answer"]
            history_lines.append(f"A{i+1}: {answer_preview}")
            history_lines.append("")  # Blank line between turns for readability

        return "\n".join(history_lines)

    def clear(self):
        """Resets the conversation. Called when user starts a new session."""
        self._turns = deque(maxlen=MAX_TURNS)
        print("[Memory] Conversation history cleared.")

    def get_turn_count(self) -> int:
        """Returns the number of turns stored (useful for UI display)."""
        return len(self._turns)

    def get_last_question(self) -> str:
        """Returns the most recent question, or empty string if no history."""
        if self._turns:
            return self._turns[-1]["question"]
        return ""
