"""
Tool Registry — Tool registration and dispatch.
Follows the registry pattern from tools/registry.py in the real Hermes.
"""

import ast
import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ToolEntry:
    """Metadata for a single registered tool."""
    __slots__ = ("name", "schema", "handler", "toolset", "check_fn")

    def __init__(
        self,
        name: str,
        schema: dict,
        handler: Callable,
        toolset: str = "default",
        check_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.schema = schema
        self.handler = handler
        self.toolset = toolset
        self.check_fn = check_fn


class Registry:
    """Central tool registry — singleton."""

    def __init__(self):
        self.entries: Dict[str, ToolEntry] = {}
        self._generation = 0

    def register(
        self,
        name: str,
        schema: dict,
        handler: Callable,
        toolset: str = "default",
        check_fn: Optional[Callable] = None,
    ):
        """Register a tool."""
        self.entries[name] = ToolEntry(name, schema, handler, toolset, check_fn)
        self._generation += 1
        logger.debug(f"Registered tool: {name}")

    def get_definitions(self, tool_names: Optional[List[str]] = None) -> List[dict]:
        """Get tool schemas for API call."""
        result = []
        for name, entry in self.entries.items():
            if tool_names and name not in tool_names:
                continue
            result.append(entry.schema)
        return result

    def dispatch(self, name: str, args: dict) -> str:
        """Execute a tool and return JSON string result."""
        entry = self.entries.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = entry.handler(args)
            return json.dumps(result)
        except Exception as e:
            logger.exception(f"Tool {name} raised: {e}")
            return json.dumps({"error": str(e)})

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        return self.entries.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.entries.keys())


# Global registry instance
registry = Registry()


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Discover tools in a directory by AST-scanning for registry.register() calls."""
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).parent / "tools"
    discovered = []

    for path in sorted(tools_path.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue

        # Check if module has top-level registry.register() call
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr):
                call = stmt.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "register"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "registry"
                ):
                    try:
                        importlib.import_module(f"mini_hermes.tools.{path.stem}")
                        discovered.append(path.stem)
                    except Exception as e:
                        logger.warning(f"Could not import {path.stem}: {e}")
                    break

    return discovered
