"""
Trace Logger — JSONL-based structured logging for debugging.
Inspired by the logging patterns in hermes_logging.py in the real Hermes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JSONLTraceLogger:
    """
    JSONL format trace logger.

    Each line is a JSON object with:
      - timestamp: ISO format
      - event_type: "api_request" | "tool_call" | "tool_result" | etc.
      - session_id: conversation session ID
      - data: event-specific payload
    """

    def __init__(self, trace_dir: str = "~/.mini_hermes/traces"):
        self.trace_dir = Path(trace_dir).expanduser()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: Optional[str] = None

    def set_session(self, session_id: str):
        self.session_id = session_id

    def log(self, event_type: str, data: Dict[str, Any]):
        """Write a single trace record."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_id": self.session_id or "unknown",
            "data": data,
        }
        trace_file = self.trace_dir / f"{datetime.now():%Y%m%d}.jsonl"
        try:
            with open(trace_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError:
            logger.warning(f"Could not write to {trace_file}")

    def log_api_request(self, model: str, message_count: int, tool_count: int):
        self.log("api_request", {
            "model": model,
            "message_count": message_count,
            "tool_count": tool_count,
        })

    def log_api_response(self, model: str, content_preview: str, tool_call_count: int):
        self.log("api_response", {
            "model": model,
            "content_preview": content_preview[:200],
            "tool_call_count": tool_call_count,
        })

    def log_tool_call(self, tool_name: str, args: Dict[str, Any]):
        self.log("tool_call", {
            "tool": tool_name,
            "args": args,
        })

    def log_tool_result(self, tool_name: str, success: bool, duration_ms: int):
        self.log("tool_result", {
            "tool": tool_name,
            "success": success,
            "duration_ms": duration_ms,
        })


# Global logger instance
trace_logger = JSONLTraceLogger()
