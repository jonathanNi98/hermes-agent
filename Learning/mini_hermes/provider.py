"""
Provider — Model calling abstraction.
Follows the ProviderProfile pattern from providers/base.py in the real Hermes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ProviderProfile:
    """Declarative provider configuration."""
    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    default_aux_model: str = ""


class BaseAdapter:
    """Base class for model adapters."""

    def __init__(self, profile: ProviderProfile):
        self.profile = profile

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """Call the model. Returns a dict with content, tool_calls, usage."""
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    """OpenAI-compatible API adapter."""

    def __init__(self, profile: ProviderProfile):
        super().__init__(profile)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=profile.api_key or None,
                base_url=profile.base_url or None,
            )
        except ImportError:
            self.client = None

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """Call OpenAI-compatible API."""
        if self.client is None:
            # Fallback: return a mock response for testing
            return self._mock_response(messages)

        params = {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": self.profile.max_tokens,
            "temperature": self.profile.temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)

        # Normalize to Hermes format
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "usage": {
                "input_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                "output_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
            }
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in choice.message.tool_calls
            ]

        return result

    def _mock_response(self, messages: List[dict]) -> dict:
        """Mock response when OpenAI client is unavailable."""
        last_msg = messages[-1]["content"] if messages else ""
        return {
            "content": f"[Mock] I received: {last_msg[:50]}",
            "tool_calls": [],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }


def create_adapter(profile: ProviderProfile) -> BaseAdapter:
    """Factory to create the right adapter based on provider name."""
    adapters = {
        "openai": OpenAIAdapter,
        "anthropic": OpenAIAdapter,  # Reuse for simplicity
        "custom": OpenAIAdapter,
    }
    adapter_cls = adapters.get(profile.name, OpenAIAdapter)
    return adapter_cls(profile)
