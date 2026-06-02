"""
Conversation Loop — The main agent loop.
Follows the pattern from agent/conversation_loop.py in the real Hermes.
"""

import json
import logging
import time
import uuid
from typing import List, Optional

from mini_hermes.provider import ProviderProfile, create_adapter
from mini_hermes.prompt_builder import build_system_prompt
from mini_hermes.memory_manager import MemoryManager
from mini_hermes.tool_executor import execute_tool_calls_sequential
from mini_hermes.trace_logger import trace_logger

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 20


def run_conversation(
    agent,
    user_message: str,
) -> str:
    """
    The main conversation loop.

    1. Build messages (system + history + memory + tools)
    2. Call the model
    3. If tool calls: execute them and loop
    4. If no tool calls: return the response
    """
    agent.messages.append({"role": "user", "content": user_message})
    trace_logger.set_session(agent.session_id)

    for iteration in range(MAX_ITERATIONS):
        # Step 1: Build messages
        messages = _build_messages(agent, user_message)

        # Step 2: Call the model
        trace_logger.log_api_request(
            model=agent.provider.profile.model,
            message_count=len(messages),
            tool_count=len(agent.tool_schemas),
        )

        response = agent.adapter.chat(messages, tools=agent.tool_schemas)

        trace_logger.log_api_response(
            model=agent.provider.profile.model,
            content_preview=response.get("content", ""),
            tool_call_count=len(response.get("tool_calls", [])),
        )

        # Step 3: Check for tool calls
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            # No tools — this is the final response
            final_content = response.get("content", "")
            agent.messages.append({"role": "assistant", "content": final_content})
            return final_content

        # Step 4: Execute tool calls
        tool_results = execute_tool_calls_sequential(tool_calls)

        for i, tc in enumerate(tool_calls):
            result = tool_results[i]
            agent.messages.append({
                "role": "tool",
                "content": result.get("result", ""),
                "tool_call_id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
            })

            trace_logger.log_tool_call(
                tool_name=tc.get("function", {}).get("name", ""),
                args=json.loads(tc.get("function", {}).get("arguments", "{}")),
            )

        # Step 5: Add assistant message with tool_calls
        agent.messages.append({
            "role": "assistant",
            "content": response.get("content", ""),
            "tool_calls": tool_calls,
        })

        logger.info(f"Iteration {iteration + 1}: executed {len(tool_calls)} tool(s)")

    return "Max iterations reached"


def _build_messages(agent, user_message: str) -> List[dict]:
    """Build the messages list for the API call."""
    messages = []

    # System prompt
    memory_context = agent.memory.build_context_block(user_message)
    system_prompt = build_system_prompt(
        system_message=agent.system_message,
        memory_context=memory_context,
        enabled_tools=agent.tool_schemas,
    )
    messages.append({"role": "system", "content": system_prompt})

    # History (excluding the latest user message we just added)
    history = agent.messages[:-1]
    messages.extend(history)

    # Latest user message
    messages.append({"role": "user", "content": user_message})

    return messages
