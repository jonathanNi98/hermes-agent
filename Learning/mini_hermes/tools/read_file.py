"""
Read File Tool — Safely read a file's contents.
"""

from mini_hermes.tool_registry import registry
from mini_hermes.safety import read_file_safe


def read_file_handler(args: dict) -> dict:
    """Handle read_file tool calls."""
    path = args.get("path", "")

    if not path:
        return {"success": False, "error": "No path provided"}

    success, content = read_file_safe(path)

    if success:
        # Truncate very long files
        max_preview = 5000
        if len(content) > max_preview:
            content = content[:max_preview] + f"\n... [truncated, {len(content)} total bytes]"

        return {
            "success": True,
            "path": path,
            "content": content,
            "bytes": len(content.encode("utf-8")),
        }
    else:
        return {"success": False, "path": path, "error": content}


registry.register(
    name="read_file",
    schema={
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. File must be within allowed directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    handler=read_file_handler,
    toolset="file",
)
