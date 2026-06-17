#!/usr/bin/env python3
"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main instructions)
and optional supporting files like references, templates, and examples.

Inspired by Anthropic's Claude Skills system with progressive disclosure architecture:
- Metadata (name ≤64 chars, description ≤1024 chars) - shown in skills_list
- Full Instructions - loaded via skill_view when needed
- Linked Files (references, templates) - loaded on demand

Directory Structure:
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # Main instructions (required)
    │   ├── references/        # Supporting documentation
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # Templates for output
    │   │   └── template.md
    │   └── assets/            # Supplementary files (agentskills.io standard)
    └── category/              # Category folder for organization
        └── another-skill/
            └── SKILL.md

SKILL.md Format (YAML Frontmatter, agentskills.io compatible):
    ---
    name: skill-name              # Required, max 64 chars
    description: Brief description # Required, max 1024 chars
    version: 1.0.0                # Optional
    license: MIT                  # Optional (agentskills.io)
    platforms: [macos]            # Optional — restrict to specific OS platforms
                                  #   Valid: macos, linux, windows
                                  #   Omit to load on all platforms (default)
    prerequisites:                # Optional — legacy runtime requirements
      env_vars: [API_KEY]         #   Legacy env var names are normalized into
                                  #   required_environment_variables on load.
      commands: [curl, jq]        #   Command checks remain advisory only.
    compatibility: Requires X     # Optional (agentskills.io)
    metadata:                     # Optional, arbitrary key-value (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---

    # Skill Title

    Full instructions and content here...

Available tools:
- skills_list: List skills with metadata (progressive disclosure tier 1)
- skill_view: Load full skill content (progressive disclosure tier 2-3)

Usage:
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements

    # List all skills (returns metadata only - token efficient)
    result = skills_list()

    # View a skill's main content (loads full instructions)
    content = skill_view("axolotl")

    # View a reference file within a skill (loads linked file)
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import json
import logging

from hermes_constants import get_hermes_home, display_hermes_home
import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from utils import env_var_enabled
from agent.skill_utils import EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS

logger = logging.getLogger(__name__)


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
# This is the single source of truth -- agent edits, hub installs, and bundled
# skills all coexist here without polluting the git repo.
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"

# Anthropic-recommended limits for progressive disclosure efficiency
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# Platform identifiers for the 'platforms' frontmatter field.
# Maps user-friendly names to sys.platform prefixes.
_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REMOTE_ENV_BACKENDS = frozenset(
    {"docker", "singularity", "modal", "ssh", "daytona"}
)
_secret_capture_callback = None


def load_env() -> Dict[str, str]:
    """Load profile-scoped environment variables from HERMES_HOME/.env."""
    env_path = get_hermes_home() / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


# Prompt injection detection — shared by local-skill and plugin-skill paths.
_INJECTION_PATTERNS: list = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "<system>",
    "]]>",
]


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Check if a skill is compatible with the current OS platform.

    Delegates to ``agent.skill_utils.skill_matches_platform`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


def _normalize_prerequisite_values(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}

    help_text = setup.get("help")
    normalized_help = (
        str(help_text).strip()
        if isinstance(help_text, str) and help_text.strip()
        else None
    )

    collect_secrets_raw = setup.get("collect_secrets")
    if isinstance(collect_secrets_raw, dict):
        collect_secrets_raw = [collect_secrets_raw]
    if not isinstance(collect_secrets_raw, list):
        collect_secrets_raw = []

    collect_secrets: List[Dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: Dict[str, Any] = {
            "env_var": env_var,
            "prompt": prompt,
            "secret": bool(item.get("secret", True)),
        }
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)

    return {
        "help": normalized_help,
        "collect_secrets": collect_secrets,
    }


def _get_required_environment_variables(
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }

        help_text = (
            entry.get("help")
            or entry.get("provider_url")
            or entry.get("url")
            or setup.get("help")
        )
        if isinstance(help_text, str) and help_text.strip():
            normalized["help"] = help_text.strip()

        required_for = entry.get("required_for")
        if isinstance(required_for, str) and required_for.strip():
            normalized["required_for"] = required_for.strip()

        if entry.get("optional"):
            normalized["optional"] = True

        seen.add(env_name)
        required.append(normalized)

    for item in required_raw:
        if isinstance(item, str):
            _append_required({"name": item})
            continue
        if isinstance(item, dict):
            _append_required(item)

    for item in setup["collect_secrets"]:
        _append_required(
            {
                "name": item.get("env_var"),
                "prompt": item.get("prompt"),
                "help": item.get("provider_url") or setup.get("help"),
            }
        )

    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})

    return required


def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not missing_entries:
        return {
            "missing_names": [],
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    missing_names = [entry["name"] for entry in missing_entries]
    if _is_gateway_surface():
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": _gateway_setup_hint(),
        }

    if _secret_capture_callback is None:
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    setup_skipped = False
    remaining_names: List[str] = []

    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        if entry.get("help"):
            metadata["help"] = entry["help"]
        if entry.get("required_for"):
            metadata["required_for"] = entry["required_for"]

        try:
            callback_result = _secret_capture_callback(
                entry["name"],
                entry["prompt"],
                metadata,
            )
        except Exception:
            logger.warning(
                f"Secret capture callback failed for {entry['name']}", exc_info=True
            )
            callback_result = {
                "success": False,
                "stored_as": entry["name"],
                "validated": False,
                "skipped": True,
            }

        success = isinstance(callback_result, dict) and bool(
            callback_result.get("success")
        )
        skipped = isinstance(callback_result, dict) and bool(
            callback_result.get("skipped")
        )
        if success and not skipped:
            continue

        setup_skipped = True
        remaining_names.append(entry["name"])

    return {
        "missing_names": remaining_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": None,
    }


def _is_gateway_surface() -> bool:
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_SESSION_PLATFORM"))


def _get_terminal_backend_name() -> str:
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _is_env_var_persisted(
    var_name: str, env_snapshot: Dict[str, str] | None = None
) -> bool:
    if env_snapshot is None:
        env_snapshot = load_env()
    if var_name in env_snapshot:
        return bool(env_snapshot.get(var_name))
    return bool(os.getenv(var_name))


def _remaining_required_environment_names(
    required_env_vars: List[Dict[str, Any]],
    capture_result: Dict[str, Any],
    *,
    env_snapshot: Dict[str, str] | None = None,
) -> List[str]:
    missing_names = set(capture_result["missing_names"])

    if env_snapshot is None:
        env_snapshot = load_env()
    remaining = []
    for entry in required_env_vars:
        name = entry["name"]
        if entry.get("optional"):
            continue
        if name in missing_names or not _is_env_var_persisted(name, env_snapshot):
            remaining.append(name)
    return remaining


def _gateway_setup_hint() -> str:
    try:
        from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE

        return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
    except Exception:
        return f"Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to {display_hermes_home()}/.env manually."


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Delegates to ``agent.skill_utils.parse_frontmatter`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """
    Extract category from skill path based on directory structure.

    For paths like: ~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    Also works for external skill dirs configured via skills.external_dirs.
    """
    # Try the module-level SKILLS_DIR first (respects monkeypatching in tests),
    # then fall back to external dirs from config.
    dirs_to_check = [SKILLS_DIR]
    try:
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    except Exception:
        pass
    for skills_dir in dirs_to_check:
        try:
            rel_path = skill_path.relative_to(skills_dir)
            parts = rel_path.parts
            if len(parts) >= 3:
                return parts[0]
        except ValueError:
            continue
    return None


def _parse_tags(tags_value) -> List[str]:
    """
    Parse tags from frontmatter value.

    Handles:
    - Already-parsed list (from yaml.safe_load): [tag1, tag2]
    - String with brackets: "[tag1, tag2]"
    - Comma-separated string: "tag1, tag2"

    Args:
        tags_value: Raw tags value — may be a list or string

    Returns:
        List of tag strings
    """
    if not tags_value:
        return []

    # yaml.safe_load already returns a list for [tag1, tag2]
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]

    # String fallback — handle bracket-wrapped or comma-separated
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]

    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]



def _get_disabled_skill_names() -> Set[str]:
    """Load disabled skill names from config.

    Delegates to ``agent.skill_utils.get_disabled_skill_names`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import get_disabled_skill_names
    return get_disabled_skill_names()


def _get_session_platform() -> str:
    """Resolve the current platform from gateway session context.

    Mirrors the platform-resolution logic in
    ``agent.skill_utils.get_disabled_skill_names`` so that
    ``_is_skill_disabled`` respects ``HERMES_SESSION_PLATFORM``.
    """
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_PLATFORM") or ""
    except Exception:
        return ""


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Check if a skill is disabled in config.

    Resolves the active platform from (in order of precedence):
    1. Explicit ``platform`` argument
    2. ``HERMES_PLATFORM`` environment variable
    3. ``HERMES_SESSION_PLATFORM`` from gateway session context
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM") or _get_session_platform()
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
            if platform_disabled is not None:
                return name in platform_disabled
        return name in skills_cfg.get("disabled", [])
    except Exception:
        return False


def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """Recursively find all skills in ~/.hermes/skills/ and external dirs.

    Args:
        skip_disabled: If True, return ALL skills regardless of disabled
            state (used by ``hermes skills`` config UI). Default False
            filters out disabled skills.

    Returns:
        List of skill metadata dicts (name, description, category).
    """
    from agent.skill_utils import get_external_skills_dirs, iter_skill_index_files

    skills = []
    seen_names: set = set()

    # Load disabled set once (not per-skill)
    disabled = set() if skip_disabled else _get_disabled_skill_names()

    # Scan local dir first, then external dirs (local takes precedence)
    dirs_to_scan = []
    if SKILLS_DIR.exists():
        dirs_to_scan.append(SKILLS_DIR)
    dirs_to_scan.extend(get_external_skills_dirs())

    for scan_dir in dirs_to_scan:
        for skill_md in iter_skill_index_files(scan_dir, "SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue

            skill_dir = skill_md.parent

            try:
                content = skill_md.read_text(encoding="utf-8")[:4000]
                frontmatter, body = _parse_frontmatter(content)

                if not skill_matches_platform(frontmatter):
                    continue

                name = frontmatter.get("name", skill_dir.name)[:MAX_NAME_LENGTH]
                if name in seen_names:
                    continue
                if name in disabled:
                    continue

                description = frontmatter.get("description", "")
                if not description:
                    for line in body.strip().split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            description = line
                            break

                if len(description) > MAX_DESCRIPTION_LENGTH:
                    description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."

                category = _get_category_from_path(skill_md)

                seen_names.add(name)
                skills.append({
                    "name": name,
                    "description": description,
                    "category": category,
                })

            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
                continue
            except Exception as e:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True
                )
                continue

    return skills


def _sort_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


# ────────────────────────────────────────────────────────────
# 1.1 skills_list — Tier 1 入口(只返 name + description)
# ────────────────────────────────────────────────────────────
#
# 角色:Skills 系统的"目录页"。
#      LLM 调它一次性看到所有可用 skill 的名字和简介。
#      **不**返完整正文 — 那是 skill_view() 的活。
#
# 设计:Progressive Disclosure Tier 1
#      一次列 50 个 skill,只返 (name, description, category),
#      每个 ~30-50 tokens。完整内容按需调 skill_view。
#      避免一次塞 50 个 SKILL.md 把 context 撑爆。
#
# 返回:JSON 字符串(不是 dict) — 走 LLM tool_result 的标准格式
def skills_list(category: str = None, task_id: str = None) -> str:
    """
    List all available skills (progressive disclosure tier 1 - minimal metadata).

    Returns only name + description to minimize token usage. Use skill_view() to
    load full content, tags, related files, etc.

    Args:
        category: Optional category filter (e.g., "mlops")
        task_id: Optional task identifier used to probe the active backend

    Returns:
        JSON string with minimal skill info: name, description, category
    """
    try:
        # 1.2 目录存在性检查 + 首次创建
        # 用户可能刚装 hermes 还没建 ~/.hermes/skills/
        # 这里顺手建好(不报错),返一个空 list + 友好提示
        if not SKILLS_DIR.exists():
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": f"No skills found. Skills directory created at {display_hermes_home()}/skills/",
                },
                ensure_ascii=False,
            )

        # 1.3 扫所有 skill(本地 + 外部 + 插件)
        # _find_all_skills() 内部做:
        #   1. 递归 os.walk SKILLS_DIR 和 external_dirs
        #   2. parse_frontmatter 取 name/description
        #   3. 平台过滤(current_platform 不在 platforms 里 → 跳过)
        #   4. disabled 过滤
        #   5. 去重(seen_names)
        #   6. 异常吞掉(一个坏 skill 不能让整个 list 失败)
        # 注意:每个 skill **只读前 4000 字符**(frontmatter 足够)
        all_skills = _find_all_skills()

        # 1.4 空集友好处理
        # 目录存在但里面啥也没有(用户没装任何 skill)
        # 不报错,返空 + 提示
        if not all_skills:
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found in skills/ directory.",
                },
                ensure_ascii=False,
            )

        # 1.5 可选 category 过滤
        # LLM 可能传 "github" 想只看 GitHub 相关 skill
        # 不传 → 返所有
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]

        # 1.6 稳定排序
        # 按 (category, name) 排,保证两次调用的顺序一致
        # 重要:LLM 多次调 skills_list 看到的顺序不能变
        all_skills = _sort_skills(all_skills)

        # 1.7 提取所有 category(去重 + 排序)
        # 给 LLM 一个"所有分类"的概览
        # 用 set 去重,再 sorted 排序
        # 过滤掉 None(category 为空的 skill 不进分类列表)
        categories = sorted(
            {s.get("category") for s in all_skills if s.get("category")}
        )

        # 1.8 返最终结果
        # hint 字段告诉 LLM:想看完整内容请调 skill_view()
        # 引导 LLM 进入 Tier 2(progressive disclosure 的下一步)
        return json.dumps(
            {
                "success": True,
                "skills": all_skills,
                "categories": categories,
                "count": len(all_skills),
                "hint": "Use skill_view(name) to see full content, tags, and linked files",
            },
            ensure_ascii=False,
        )

    # 1.9 顶层兜底
    # 任何意外都转成 tool_result 的 error 格式
    # 不抛 — LLM 拿到的应该是可解析的 JSON,不是 Python 异常
    except Exception as e:
        return tool_error(str(e), success=False)


# ── Plugin skill serving ──────────────────────────────────────────────────


def _serve_plugin_skill(
    skill_md: Path,
    namespace: str,
    bare: str,
    *,
    preprocess: bool = True,
    session_id: str | None = None,
) -> str:
    """Read a plugin-provided skill, apply guards, return JSON."""
    from hermes_cli.plugins import _get_disabled_plugins, get_plugin_manager

    if namespace in _get_disabled_plugins():
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Plugin '{namespace}' is disabled. "
                    f"Re-enable with: hermes plugins enable {namespace}"
                ),
            },
            ensure_ascii=False,
        )

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Failed to read skill '{namespace}:{bare}': {e}"},
            ensure_ascii=False,
        )

    parsed_frontmatter: Dict[str, Any] = {}
    try:
        parsed_frontmatter, _ = _parse_frontmatter(content)
    except Exception:
        pass

    if not skill_matches_platform(parsed_frontmatter):
        return json.dumps(
            {
                "success": False,
                "error": f"Skill '{namespace}:{bare}' is not supported on this platform.",
                "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
            },
            ensure_ascii=False,
        )

    # Injection scan — log but still serve (matches local-skill behaviour)
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        logger.warning(
            "Plugin skill '%s:%s' contains patterns that may indicate prompt injection",
            namespace, bare,
        )

    description = str(parsed_frontmatter.get("description", ""))
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

    # Bundle context banner — tells the agent about sibling skills
    try:
        siblings = [
            s for s in get_plugin_manager().list_plugin_skills(namespace)
            if s != bare
        ]
        if siblings:
            sib_list = ", ".join(siblings)
            banner = (
                f"[Bundle context: This skill is part of the '{namespace}' plugin.\n"
                f"Sibling skills: {sib_list}.\n"
                f"Use qualified form to invoke siblings (e.g. {namespace}:{siblings[0]}).]\n\n"
            )
        else:
            banner = f"[Bundle context: This skill is part of the '{namespace}' plugin.]\n\n"
    except Exception:
        banner = ""

    rendered_content = content
    if preprocess:
        try:
            from agent.skill_preprocessing import preprocess_skill_content

            rendered_content = preprocess_skill_content(
                content,
                skill_md.parent,
                session_id=session_id,
            )
        except Exception:
            logger.debug(
                "Could not preprocess plugin skill %s:%s", namespace, bare, exc_info=True
            )

    return json.dumps(
        {
            "success": True,
            "name": f"{namespace}:{bare}",
            "content": f"{banner}{rendered_content}" if banner else rendered_content,
            "description": description,
            "linked_files": None,
            "readiness_status": SkillReadinessStatus.AVAILABLE.value,
        },
        ensure_ascii=False,
    )


# ────────────────────────────────────────────────────────────
# 2.1 skill_view — Tier 2 入口(返 SKILL.md 完整内容 / 链接文件)
# ────────────────────────────────────────────────────────────
#
# 角色:Skills 系统的"正文章节"。
#      LLM 看到 skills_list 里有想用的 skill,调这个拿完整内容。
#      可以加 file_path 参数拿 references/ templates/ assets/ 里的子文件。
#
# 设计:Progressive Disclosure Tier 2/3
#      一次只返 1 个 skill 的内容(可能是几万 tokens 的 SKILL.md,
#      或者一个 references 子文件)。按需加载,绝不一次塞所有 skill。
#
# 入口形态:2 种 name 格式
#   "github-code-review"      → 本地 SKILLS_DIR 扫描
#   "myplugin:some-skill"     → 插件命名空间分发
#
# 返回:JSON 字符串(走 LLM tool_result)
def skill_view(
    name: str,
    file_path: str = None,
    task_id: str = None,
    preprocess: bool = True,
) -> str:
    """
    View the content of a skill or a specific file within a skill directory.

    Args:
        name: Name or path of the skill (e.g., "axolotl" or "03-fine-tuning/axolotl").
            Qualified names like "plugin:skill" resolve to plugin-provided skills.
        file_path: Optional path to a specific file within the skill (e.g., "references/api.md")
        task_id: Optional task identifier used to probe the active backend
        preprocess: Apply configured SKILL.md template and inline shell rendering
            to main skill content. Internal slash/preload callers disable this
            because they render the skill message themselves.

    Returns:
        JSON string with skill content or error message
    """
    try:
        local_category_name: str | None = None
        # ── Qualified name dispatch (plugin skills) ──────────────────
        # Names containing ':' are routed to the plugin skill registry.
        # Bare names fall through to the existing flat-tree scan below.
        # 2.2 Plugin 分支:qualified name dispatch
        if ":" in name:
            from agent.skill_utils import is_valid_namespace, parse_qualified_name
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            namespace, bare = parse_qualified_name(name)
            # 2.3 namespace 白名单校验([a-zA-Z0-9_-]+),防注入
            if not is_valid_namespace(namespace):
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"Invalid namespace '{namespace}' in '{name}'. "
                            f"Namespaces must match [a-zA-Z0-9_-]+."
                        ),
                    },
                    ensure_ascii=False,
                )

            discover_plugins()  # idempotent — 2.4 插件查找
            pm = get_plugin_manager()
            plugin_skill_md = pm.find_plugin_skill(name)

            if plugin_skill_md is not None:
                if not plugin_skill_md.exists():
                    # 2.4a Stale 懒清理 — 文件没了就清注册表
                    pm.remove_plugin_skill(name)
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Skill '{name}' file no longer exists at "
                                f"{plugin_skill_md}. The registry entry has "
                                f"been cleaned up — try again after the "
                                f"plugin is reloaded."
                            ),
                        },
                        ensure_ascii=False,
                    )
                return _serve_plugin_skill(  # 2.4b 渲染插件 skill
                    plugin_skill_md,
                    namespace,
                    bare,
                    preprocess=preprocess,
                    session_id=task_id,
                )

            # 2.4c 插件存在但具体 skill 缺,列其他 skill 给 LLM
            available = pm.list_plugin_skills(namespace)
            if available:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Skill '{bare}' not found in plugin '{namespace}'.",
                        "available_skills": [f"{namespace}:{s}" for s in available],
                        "hint": f"The '{namespace}' plugin provides {len(available)} skill(s).",
                    },
                    ensure_ascii=False,
                )
            # Plugin itself not found — fall through to flat-tree scan.
            # Categorized local skills also use `category:skill` in config and
            # gateway prompts, so preserve that form and translate it to the
            # on-disk `category/skill` path during the local scan below.
            if bare:
                local_category_name = f"{namespace}/{bare}"  # 2.4d plugin 缺注册,试本地分类路径

        from agent.skill_utils import get_external_skills_dirs

        # 2.5 构建搜索目录列表(本地 + 外部,本地优先)
        all_dirs = []
        if SKILLS_DIR.exists():
            all_dirs.append(SKILLS_DIR)
        all_dirs.extend(get_external_skills_dirs())

        if not all_dirs:
            return json.dumps(
                {
                    "success": False,
                    "error": "Skills directory does not exist yet. It will be created on first install.",
                },
                ensure_ascii=False,
            )

        skill_dir = None
        skill_md = None

        # 2.6 冲突检测 — 收集所有候选,**不静默 shadow**
        # 设计:多候选碰撞时拒绝猜测,返错让用户显式选
        # silent shadowing 是真实 bug 类:`/skills` 显示一个,实际加载另一个
        from agent.skill_utils import iter_skill_index_files

        candidates: List[Tuple[Optional[Path], Path]] = []  # (skill_dir, skill_md)
        seen_md: set = set()  # 用 resolve() 后的真实路径做 key

        def _record(sd: Optional[Path], smd: Path) -> None:
            try:
                key = smd.resolve()
            except Exception:
                key = smd
            if key in seen_md:
                return
            seen_md.add(key)
            candidates.append((sd, smd))

        # 2.7 三种查找 strategy(每种都 try,收集所有命中)
        for search_dir in all_dirs:
            # 2.7.1 Strategy 1: 直接路径(顶层 "axolotl" 或 "mlops/axolotl")
            direct_path = search_dir / name
            if direct_path.is_dir() and (direct_path / "SKILL.md").exists():
                _record(direct_path, direct_path / "SKILL.md")
            elif direct_path.with_suffix(".md").exists():
                _record(None, direct_path.with_suffix(".md"))

            # 2.7.2 Strategy 1b: 分类路径(plugin namespace fall-through)
            # 例:"myplugin:explore" 无插件注册,试本地 "myplugin/explore"
            if local_category_name:
                categorized_path = search_dir / local_category_name
                if categorized_path.is_dir() and (categorized_path / "SKILL.md").exists():
                    _record(categorized_path, categorized_path / "SKILL.md")
                elif categorized_path.with_suffix(".md").exists():
                    _record(None, categorized_path.with_suffix(".md"))

            # 2.7.3 Strategy 2: 递归按目录名找
            # 适用:深层嵌套的 skill,如 "foundations/runtime/explore-codebase"
            for found_skill_md in iter_skill_index_files(search_dir, "SKILL.md"):
                if found_skill_md.parent.name == name:
                    _record(found_skill_md.parent, found_skill_md)

            # 2.7.4 Strategy 3: 老 flat 格式 <name>.md(老式单文件 skill)
            for found_md in search_dir.rglob(f"{name}.md"):
                if found_md.name != "SKILL.md":  # 排除 SKILL.md 自身
                    _record(None, found_md)

        # 2.8 多候选 → 拒绝猜测,返错(不静默选第一个)
        if len(candidates) > 1:
            paths = [str(smd) for _, smd in candidates]
            logging.getLogger(__name__).warning(
                "Skill name collision for '%s': %d candidates — %s",
                name, len(candidates), "; ".join(paths),
            )
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Ambiguous skill name '{name}': {len(candidates)} skills "
                        "match across your local skills dir and external_dirs. "
                        "Refusing to guess — load one explicitly by its categorized path."
                    ),
                    "matches": paths,
                    "hint": (
                        "Pass the full relative path instead of the bare name "
                        "(e.g., 'category/skill-name'), or rename one of the "
                        "colliding skills so each name is unique."
                    ),
                },
                ensure_ascii=False,
            )

        # 2.9 单候选 → 采纳;空 → 返 not found 错误(附可用 skill)
        if candidates:
            skill_dir, skill_md = candidates[0]

        if not skill_md or not skill_md.exists():
            available = [s["name"] for s in _sort_skills(_find_all_skills())[:20]]
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' not found.",
                    "available_skills": available,
                    "hint": "Use skills_list to see all available skills",
                },
                ensure_ascii=False,
            )

        # 2.10 读全文(只读 1 次,platform 检查 + 主内容复用)
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to read skill '{name}': {e}",
                },
                ensure_ascii=False,
            )

        # 2.11 安全检查 1:位置验证 — 警告(非阻断)
        # skill 不在 trusted 目录里 → 可能是 symlink 攻击 / 误装
        # trusted = SKILLS_DIR + 配置的 external_dirs
        _outside_skills_dir = True
        _trusted_dirs = [SKILLS_DIR.resolve()]
        try:
            _trusted_dirs.extend(d.resolve() for d in all_dirs[1:])
        except Exception:
            pass
        for _td in _trusted_dirs:
            try:
                skill_md.resolve().relative_to(_td)
                _outside_skills_dir = False
                break
            except ValueError:
                continue

        # 2.12 安全检查 2:prompt 注入检测(子串匹配 _INJECTION_PATTERNS)
        # 误报可接受,漏报危险 — false positive 比 false negative 安全
        _content_lower = content.lower()
        _injection_detected = any(p in _content_lower for p in _INJECTION_PATTERNS)

        if _outside_skills_dir or _injection_detected:
            _warnings = []
            if _outside_skills_dir:
                _warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
            if _injection_detected:
                _warnings.append("skill content contains patterns that may indicate prompt injection")
            logging.getLogger(__name__).warning("Skill security warning for '%s': %s", name, "; ".join(_warnings))

        # 2.13 解析 frontmatter(只 1 次,后面复用;失败兜底空 dict)
        parsed_frontmatter: Dict[str, Any] = {}
        try:
            parsed_frontmatter, _ = _parse_frontmatter(content)
        except Exception:
            parsed_frontmatter = {}

        # 2.14 平台过滤(再次 — 防绕过 list 直接 view)
        if not skill_matches_platform(parsed_frontmatter):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' is not supported on this platform.",
                    "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
                },
                ensure_ascii=False,
            )

        # 2.15 disabled 检查 — 用户配过禁用的不加载
        resolved_name = parsed_frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{resolved_name}' is disabled. "
                        "Enable it with `hermes skills` or inspect the files directly on disk."
                    ),
                },
                ensure_ascii=False,
            )

        # 2.16 file_path 分支 — 加载 linked file(不是主 SKILL.md)
        if file_path and skill_dir:
            from tools.path_security import validate_within_dir, has_traversal_component

            # 2.16.1 路径遍历防护 — 防 LLM 越权
            if has_traversal_component(file_path):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path traversal ('..') is not allowed.",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )

            target_file = skill_dir / file_path

            # 2.16.2 软链攻击防御 — .resolve() 真实路径再校验
            traversal_error = validate_within_dir(target_file, skill_dir)
            if traversal_error:
                return json.dumps(
                    {
                        "success": False,
                        "error": traversal_error,
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            if not target_file.exists():
                # 2.16.3 文件不存在 → 列所有可用文件(按类型分组,引导 LLM)
                available_files = {
                    "references": [],
                    "templates": [],
                    "assets": [],
                    "scripts": [],
                    "other": [],
                }

                for f in skill_dir.rglob("*"):
                    if f.is_file() and f.name != "SKILL.md":
                        rel = str(f.relative_to(skill_dir))
                        if rel.startswith("references/"):
                            available_files["references"].append(rel)
                        elif rel.startswith("templates/"):
                            available_files["templates"].append(rel)
                        elif rel.startswith("assets/"):
                            available_files["assets"].append(rel)
                        elif rel.startswith("scripts/"):
                            available_files["scripts"].append(rel)
                        elif f.suffix in {
                            ".md",
                            ".py",
                            ".yaml",
                            ".yml",
                            ".json",
                            ".tex",
                            ".sh",
                        }:
                            available_files["other"].append(rel)

                # Remove empty categories
                available_files = {k: v for k, v in available_files.items() if v}

                return json.dumps(
                    {
                        "success": False,
                        "error": f"File '{file_path}' not found in skill '{name}'.",
                        "available_files": available_files,
                        "hint": "Use one of the available file paths listed above",
                    },
                    ensure_ascii=False,
                )

            # 2.16.4 读 linked file 内容
            try:
                content = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # 2.16.5 二进制文件 → 返元数据,不返内容
                return json.dumps(
                    {
                        "success": True,
                        "name": name,
                        "file": file_path,
                        "content": f"[Binary file: {target_file.name}, size: {target_file.stat().st_size} bytes]",
                        "is_binary": True,
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "success": True,
                    "name": name,
                    "file": file_path,
                    "content": content,
                    "file_type": target_file.suffix,
                },
                ensure_ascii=False,
            )

        # 2.17 主 SKILL.md 路径开始 — 复用 2.13 已 parse 的 frontmatter
        frontmatter = parsed_frontmatter

        # 2.18 扫 4 类 linked 目录(agentskills.io 标准)
        # references/ templates/ assets/ scripts/ 都是约定俗成的子目录
        reference_files = []
        template_files = []
        asset_files = []
        script_files = []

        if skill_dir:
            # 2.18.1 references/ — 支持文档(只 .md)
            references_dir = skill_dir / "references"
            if references_dir.exists():
                reference_files = [
                    str(f.relative_to(skill_dir)) for f in references_dir.glob("*.md")
                ]

            # 2.18.2 templates/ — 模板(多种扩展名)
            templates_dir = skill_dir / "templates"
            if templates_dir.exists():
                for ext in [
                    "*.md",
                    "*.py",
                    "*.yaml",
                    "*.yml",
                    "*.json",
                    "*.tex",
                    "*.sh",
                ]:
                    template_files.extend(
                        [
                            str(f.relative_to(skill_dir))
                            for f in templates_dir.rglob(ext)
                        ]
                    )

            # 2.18.3 assets/ — 资源文件(任意类型,agentskills.io 标准)
            assets_dir = skill_dir / "assets"
            if assets_dir.exists():
                for f in assets_dir.rglob("*"):
                    if f.is_file():
                        asset_files.append(str(f.relative_to(skill_dir)))

            # 2.18.4 scripts/ — 可执行脚本
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                for ext in ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"]:
                    script_files.extend(
                        [str(f.relative_to(skill_dir)) for f in scripts_dir.glob(ext)]
                    )

        # 2.19 tags / related_skills(backward compat)
        # 优先 metadata.hermes.* (agentskills.io 约定),兜底 top-level(老格式)
        hermes_meta = {}
        metadata = frontmatter.get("metadata")
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {}) or {}

        tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
        related_skills = _parse_tags(
            hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
        )

        # 2.20 拼装 linked_files dict — 给 LLM 一眼看到"这 skill 有这些资源"
        linked_files = {}
        if reference_files:
            linked_files["references"] = reference_files
        if template_files:
            linked_files["templates"] = template_files
        if asset_files:
            linked_files["assets"] = asset_files
        if script_files:
            linked_files["scripts"] = script_files

        # 2.21 算 relative path(展示用,fallback 用父级)
        try:
            rel_path = str(skill_md.relative_to(SKILLS_DIR))
        except ValueError:
            # External skill — 用自己的父级做相对
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get(
            "name", skill_md.stem if not skill_dir else skill_dir.name
        )
        # 2.22 环境变量 / 凭证捕获
        # legacy_env_vars: prerequisites.env_vars 老格式
        # required_env_vars: 规范化后的"必需"列表
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
        required_env_vars = _get_required_environment_variables(
            frontmatter, legacy_env_vars
        )
        backend = _get_terminal_backend_name()
        env_snapshot = load_env()
        # 2.22.1 找"缺"的 env var(optional 跳过 + 已持久化跳过)
        missing_required_env_vars = [
            e
            for e in required_env_vars
            if not e.get("optional")
            and not _is_env_var_persisted(e["name"], env_snapshot)
        ]
        # 2.22.2 捕获 — mark "需要 setup"(实际弹窗输入由调用方处理)
        capture_result = _capture_required_environment_variables(
            skill_name,
            missing_required_env_vars,
        )
        if missing_required_env_vars:
            env_snapshot = load_env()
        remaining_missing_required_envs = _remaining_required_environment_names(
            required_env_vars,
            capture_result,
            env_snapshot=env_snapshot,
        )
        setup_needed = bool(remaining_missing_required_envs)

        # 2.23 注册"已存在"的 env var → 沙箱透传(execute_code / terminal)
        # 只注册"已经设置"的,缺的不注册(在 setup_needed 里高亮)
        available_env_names = [
            e["name"]
            for e in required_env_vars
            if e["name"] not in remaining_missing_required_envs
        ]
        if available_env_names:
            try:
                from tools.env_passthrough import register_env_passthrough

                register_env_passthrough(available_env_names)
            except Exception:
                logger.debug(
                    "Could not register env passthrough for skill %s",
                    skill_name,
                    exc_info=True,
                )

        # 2.24 注册凭证文件 → 远程沙箱挂载(Modal / Docker)
        # 存在的注册,缺的高亮在 setup_needed
        required_cred_files_raw = frontmatter.get("required_credential_files", [])
        if not isinstance(required_cred_files_raw, list):
            required_cred_files_raw = []
        missing_cred_files: list = []
        if required_cred_files_raw:
            try:
                from tools.credential_files import register_credential_files

                missing_cred_files = register_credential_files(required_cred_files_raw)
                if missing_cred_files:
                    setup_needed = True
            except Exception:
                logger.debug(
                    "Could not register credential files for skill %s",
                    skill_name,
                    exc_info=True,
                )

        # 2.25 预处理(模板变量 + 内联 shell,默认 shell 关)
        rendered_content = content
        if preprocess:
            try:
                from agent.skill_preprocessing import preprocess_skill_content

                rendered_content = preprocess_skill_content(
                    content,
                    skill_dir,
                    session_id=task_id,
                )
            except Exception:
                logger.debug(
                    "Could not preprocess skill content for %s", skill_name, exc_info=True
                )

        # 2.26 拼装最终结果(LLM 看到的就是这个 dict)
        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": tags,
            "related_skills": related_skills,
            "content": rendered_content,
            "path": rel_path,
            "skill_dir": str(skill_dir) if skill_dir else None,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'"
            if linked_files
            else None,
            "required_environment_variables": required_env_vars,
            "required_commands": [],
            "missing_required_environment_variables": remaining_missing_required_envs,
            "missing_credential_files": missing_cred_files,
            "missing_required_commands": [],
            "setup_needed": setup_needed,
            "setup_skipped": capture_result["setup_skipped"],
            "readiness_status": SkillReadinessStatus.SETUP_NEEDED.value
            if setup_needed
            else SkillReadinessStatus.AVAILABLE.value,
        }

        setup_help = next((e["help"] for e in required_env_vars if e.get("help")), None)
        if setup_help:
            result["setup_help"] = setup_help

        if capture_result["gateway_setup_hint"]:
            result["gateway_setup_hint"] = capture_result["gateway_setup_hint"]

        if setup_needed:
            missing_items = [
                f"env ${env_name}" for env_name in remaining_missing_required_envs
            ] + [
                f"file {path}" for path in missing_cred_files
            ]
            setup_note = _build_setup_note(
                SkillReadinessStatus.SETUP_NEEDED,
                missing_items,
                setup_help,
            )
            if backend in _REMOTE_ENV_BACKENDS and setup_note:
                setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
            if setup_note:
                result["setup_note"] = setup_note

        # Surface agentskills.io optional fields when present
        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return tool_error(str(e), success=False)




if __name__ == "__main__":
    """Test the skills tool"""
    print("🎯 Skills Tool Test")
    print("=" * 60)

    # Test listing skills
    print("\n📋 Listing all skills:")
    result = json.loads(skills_list())
    if result["success"]:
        print(
            f"Found {result['count']} skills in {len(result.get('categories', []))} categories"
        )
        print(f"Categories: {result.get('categories', [])}")
        print("\nFirst 10 skills:")
        for skill in result["skills"][:10]:
            cat = f"[{skill['category']}] " if skill.get("category") else ""
            print(f"  • {cat}{skill['name']}: {skill['description'][:60]}...")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a skill
    print("\n📖 Viewing skill 'axolotl':")
    result = json.loads(skill_view("axolotl"))
    if result["success"]:
        print(f"Name: {result['name']}")
        print(f"Description: {result.get('description', 'N/A')[:100]}...")
        print(f"Content length: {len(result['content'])} chars")
        if result.get("linked_files"):
            print(f"Linked files: {result['linked_files']}")
    else:
        print(f"Error: {result['error']}")

    # Test viewing a reference file
    print("\n📄 Viewing reference file 'axolotl/references/dataset-formats.md':")
    result = json.loads(skill_view("axolotl", "references/dataset-formats.md"))
    if result["success"]:
        print(f"File: {result['file']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"Preview: {result['content'][:150]}...")
    else:
        print(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(
        category=args.get("category"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count on success. Best-effort: a
    telemetry failure never breaks the tool call."""
    name = args.get("name", "")
    result = skill_view(
        name, file_path=args.get("file_path"), task_id=kw.get("task_id")
    )
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            # Use the resolved skill name from the payload when present —
            # qualified forms ("plugin:skill") return with the canonical name.
            resolved = parsed.get("name") or name
            if resolved:
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # A skill_view tool call is the agent actively loading the skill
                # to act on it — that counts as use, not just a browse/view.
                # Curator's stale timer keys off last_used_at (see agent/curator.py).
                bump_use(str(resolved))
    except Exception:
        pass
    return result


registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=_skill_view_with_bump,
    check_fn=check_skills_requirements,
    emoji="📚",
)
