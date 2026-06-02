"""
Memory Manager — In-memory memory with simple persistence.
Follows the MemoryManager pattern from agent/memory_manager.py in the real Hermes.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Simple in-memory memory manager.
    In the real Hermes, this would support pluggable providers (MemoryProvider ABC).
    """

    def __init__(self, memory_file: Optional[Path] = None):
        self.memory_file = memory_file or Path("~/.mini_hermes/memory.json")
        self.memory_file = self.memory_file.expanduser()
        self.memories: List[str] = []
        self._load()

    def _load(self):
        """Load memories from disk."""
        if self.memory_file.exists():
            try:
                data = json.loads(self.memory_file.read_text())
                self.memories = data.get("memories", [])
            except (json.JSONDecodeError, IOError):
                self.memories = []

    def _save(self):
        """Save memories to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps({"memories": self.memories}, indent=2))

    def add(self, content: str):
        """Add a memory."""
        self.memories.append(content)
        self._save()

    def search(self, query: str, top_k: int = 5) -> str:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        scored = []
        for mem in self.memories:
            mem_words = set(mem.lower().split())
            score = len(query_words & mem_words)
            if score > 0:
                scored.append((score, mem))

        scored.sort(reverse=True)
        results = [mem for _, mem in scored[:top_k]]
        return "\n".join(f"- {r}" for r in results) if results else "No relevant memories."

    def build_context_block(self, query: str = "") -> str:
        """Build the memory context block for system prompt."""
        if not self.memories:
            return ""

        if query:
            relevant = self.search(query)
            return f"<memory-context>\n{relevant}\n</memory-context>"

        # Return recent memories
        recent = self.memories[-5:]
        return "<memory-context>\n" + "\n".join(f"- {m}" for m in recent) + "\n</memory-context>"

    def clear(self):
        """Clear all memories."""
        self.memories = []
        self._save()
