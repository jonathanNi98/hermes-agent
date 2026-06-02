"""
Safety — Path restriction and basic security checks.
Follows the path_security pattern from tools/path_security.py in the real Hermes.
"""

from pathlib import Path
from typing import Optional


# Allowed roots for file operations
ALLOWED_ROOTS = [
    Path("/tmp"),
    Path("."),
]

# Blocked paths (absolute or pattern)
BLOCKED_PATHS = [
    "/etc",
    "/root",
    "/home",
    ".ssh",
    ".hermes",
]


def validate_within_dir(path: str, root: Path) -> Optional[str]:
    """
    Ensure *path* resolves to a location within *root*.

    Returns error message string if validation fails, or None if safe.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def is_path_safe(path: str) -> tuple[bool, str]:
    """
    Check if a path is safe to access.

    Returns (is_safe, error_message).
    """
    p = Path(path).expanduser()

    # Quick traversal check
    if ".." in p.parts:
        return False, "Path traversal not allowed"

    resolved = p.resolve()

    # Check against blocked paths
    for blocked in BLOCKED_PATHS:
        blocked_resolved = Path(blocked).resolve()
        try:
            resolved.relative_to(blocked_resolved)
            return False, f"Path is in blocked area: {blocked}"
        except ValueError:
            pass

    # Check allowed roots
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True, ""
        except ValueError:
            pass

    return False, f"Path not in allowed directories: {[str(r) for r in ALLOWED_ROOTS]}"


def read_file_safe(path: str, max_size: int = 1024 * 1024) -> tuple[bool, str]:
    """Safely read a file, respecting path restrictions."""
    is_safe, error = is_path_safe(path)
    if not is_safe:
        return False, error

    try:
        p = Path(path)
        if not p.exists():
            return False, f"File not found: {path}"

        if p.stat().st_size > max_size:
            return False, f"File too large: {p.stat().st_size} bytes (max {max_size})"

        content = p.read_text(encoding="utf-8")
        return True, content
    except Exception as e:
        return False, str(e)


def write_file_safe(path: str, content: str) -> tuple[bool, str]:
    """Safely write a file, respecting path restrictions."""
    is_safe, error = is_path_safe(path)
    if not is_safe:
        return False, error

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True, f"Written to {path}"
    except Exception as e:
        return False, str(e)
