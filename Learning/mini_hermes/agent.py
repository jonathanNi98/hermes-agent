"""
Agent — The main agent class.
Follows the AIAgent pattern from run_agent.py in the real Hermes.
"""

import uuid
from pathlib import Path
from typing import List, Optional

from mini_hermes.provider import ProviderProfile, create_adapter, BaseAdapter
from mini_hermes.memory_manager import MemoryManager
from mini_hermes.tool_registry import registry
from mini_hermes import conversation_loop


class AIAgent:
    """
    Mini Hermes Agent.

    Usage:
        agent = AIAgent()
        response = agent.run_conversation("Hello!")
    """

    def __init__(
        self,
        model: str = "gpt-4",
        provider: str = "openai",
        api_key: str = "",
        base_url: str = "",
        system_message: str = "",
        max_iterations: int = 20,
    ):
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.provider_name = provider
        self.system_message = system_message
        self.max_iterations = max_iterations

        # Provider setup
        self.provider = ProviderProfile(
            name=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.adapter: BaseAdapter = create_adapter(self.provider)

        # Memory
        self.memory = MemoryManager()

        # Tools — discover and register
        self.tool_schemas: List[dict] = []
        self._register_tools()

        # Message history
        self.messages: List[dict] = []

    def _register_tools(self):
        """Register built-in tools and discover tools."""
        # Import built-in tools to trigger registration
        from mini_hermes.tools import calculator, read_file, write_file

        # Get schemas for all registered tools
        self.tool_schemas = registry.get_definitions()

    def run_conversation(self, user_message: str) -> str:
        """
        Run a conversation turn.
        Thin forwarder to conversation_loop.run_conversation.
        """
        return conversation_loop.run_conversation(self, user_message)

    def reset(self):
        """Reset the agent's message history."""
        self.messages = []

    def get_session_id(self) -> str:
        return self.session_id
