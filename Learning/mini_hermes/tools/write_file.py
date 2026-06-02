"""
Write File Tool — Safely write content to a file.
"""

from mini_hermes.tool_registry import registry
from mini_hermes.safety import write_file_safe


def write_file_handler(args: dict) -> dict:
    """Handle write_file tool calls."""
    path = args.get("path", "")
    content = args.get("content", "")

    if not path:
        return {"success": False, "error": "No path provided"}

    success, message = write_file_safe(path, content)

    if success:
        return {
            "success": True,
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
        }
    else:
        return {"success": False, "path": path, "error": message}


registry.register(
    name="write_file",
    schema={
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. File must be within allowed directories. Use read_file first if you need to check existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    handler=write_file_handler,
    toolset="file",
)
