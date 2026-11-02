"""
memory/conversation.py — Bounded Sliding-Window Conversation Memory.

Stores the most recent Q&A turns in a bounded window to provide coherent context
for follow-up questions while preventing prompt bloat.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turn:
    """Represents a single query-answer turn."""
    user_query: str
    response_content: str
    agent: str = "qa"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    """Bounded sliding-window memory manager."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self.turns: list[Turn] = []

    def add_turn(self, user_query: str, response_content: str, agent: str = "qa") -> None:
        """Adds a turn and enforces the sliding window limit."""
        self.turns.append(Turn(user_query=user_query, response_content=response_content, agent=agent))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history_prompt(self, max_chars_per_turn: int = 500) -> str:
        """Formats the historical turns as a bounded conversational context string."""
        if not self.turns:
            return ""

        formatted_turns = []
        for idx, turn in enumerate(self.turns, 1):
            truncated_res = turn.response_content[:max_chars_per_turn]
            if len(turn.response_content) > max_chars_per_turn:
                truncated_res += "..."
            formatted_turns.append(f"Turn {idx} [{turn.agent.upper()}]:\nUser: {turn.user_query}\nAssistant: {truncated_res}")

        return "\n\n".join(formatted_turns)

    def clear(self) -> None:
        """Clears memory history."""
        self.turns.clear()
