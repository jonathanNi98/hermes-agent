"""
Prompt Builder — System prompt construction with three-tier architecture.
Follows the three-tier pattern from agent/system_prompt.py in the real Hermes.

Three tiers:
  - stable: identity (SOUL.md), tool guidance, skills — rarely changes
  - context: system_message, context files (AGENTS.md, .cursorrules) — per session
  - volatile: memory, timestamp, session info — per turn
"""

from typing import List
from datetime import datetime


DEFAULT_SOUL = """You are Mini Hermes, a helpful AI assistant.

You have access to tools. When you need to perform an action, use the available tools.

Guidelines:
- Think step by step
- When using tools, extract the necessary parameters from the conversation
- After getting tool results, respond with the answer
- If a tool fails, explain the error and suggest alternatives
"""


def build_system_prompt(
    system_message: str = "",
    soul_md: str = "",
    memory_context: str = "",
    enabled_tools: List[dict] = None,
) -> str:
    """Build the complete system prompt with three tiers."""

    # Tier 1: Stable — identity and tool guidance
    stable = soul_md or DEFAULT_SOUL

    # Add tool guidance
    if enabled_tools:
        tool_names = [t["name"] for t in enabled_tools]
        stable += f"\n\nAvailable tools: {', '.join(tool_names)}"

    # Tier 2: Context — user-supplied context
    context = ""
    if system_message:
        context += system_message + "\n\n"

    # Tier 3: Volatile — memory and session info
    volatile = ""
    if memory_context:
        volatile += f"[Memory context]\n{memory_context}\n\n"
    volatile += f"[Session timestamp: {datetime.now().isoformat()}]\n"

    return stable.strip() + "\n\n" + context.strip() + "\n\n" + volatile.strip()
