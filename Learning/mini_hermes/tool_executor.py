"""
Tool Executor — Execute tool calls from the model.
Follows the pattern from agent/tool_executor.py in the real Hermes.
"""

import json
import logging
import time
from typing import List, Any

from mini_hermes.tool_registry import registry

logger = logging.getLogger(__name__)

_MAX_TOOL_WORKERS = 8


def execute_tool_calls_sequential(tool_calls: List[dict]) -> List[dict]:
    """Execute tool calls one by one, in order."""
    results = []
    for tc in tool_calls:
        result = execute_single_tool(tc)
        results.append(result)
    return results


def execute_single_tool(tool_call: dict) -> dict:
    """Execute a single tool call."""
    name = tool_call.get("function", {}).get("name", "")
    raw_args = tool_call.get("function", {}).get("arguments", "{}")

    # Parse arguments
    try:
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        else:
            args = raw_args
    except json.JSONDecodeError:
        return {
            "tool_call_id": tool_call.get("id", ""),
            "result": json.dumps({"error": "Invalid JSON arguments"}),
        }

    start_time = time.time()
    logger.info(f"Executing tool: {name} with args: {args}")

    result_str = registry.dispatch(name, args)

    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Tool {name} completed in {duration_ms}ms")

    try:
        result_obj = json.loads(result_str)
    except json.JSONDecodeError:
        result_obj = {"raw": result_str}

    return {
        "tool_call_id": tool_call.get("id", ""),
        "result": result_str,
        "success": "error" not in result_obj,
        "duration_ms": duration_ms,
    }
