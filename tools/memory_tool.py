#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

# 1.1 这是干什么的
# "内置 memory"——agent 自己的"长期记忆",落盘到文件
# 两个 markdown 文件:
#   * MEMORY.md — agent 的个人笔记(环境事实 / 项目约定 / 工具怪癖)
#   * USER.md   — 对用户的认知(偏好 / 沟通风格 / 期望)
#
# 1.2 关键设计:Frozen Snapshot
# session 启动时一次性把两个 md 读出来,**冻结**成 system prompt 的一部分
# session 中途写入只更新磁盘,**不**刷新 system prompt
# 为什么?—— 保护 prefix cache(LLM 看到 system prompt 没变,缓存就还在)
# 下次 session 启动时才读新内容
#
# 1.3 文件结构
#   2.x Imports + helpers
#   3.x get_memory_dir / ENTRY_DELIMITER
#   4.x _scan_memory_content / _drift_error
#   5.x MemoryStore 类(490 行,核心)
#   6.x memory_tool 入口函数(LLM 调的工具)
#   7.x check_memory_requirements

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

# 2.1 Imports 分组
# 标准库:json/logging/os/tempfile/time/pathlib
# cross-platform:fcntl (Unix) / msvcrt (Windows) 二选一
# 项目内:atomic_replace(原子写文件)、threat_patterns(安全扫描)
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
# 2.2 跨平台文件锁
# fcntl 只能在 Unix,msvcrt 只能在 Windows
# 任一能 import 就行——Hermes 主要跑 Unix(Mac/Linux)
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# 3.1 get_memory_dir — 拿 memory 文件的目录
# **动态获取**——profile 切换时 HERMES_HOME 会变,这里每次重算
# (老版本是模块级常量,profile 切换后 stale)
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

# 3.2 ENTRY_DELIMITER — 内存条目之间的分隔符
# § 是 section sign,unicode 字符,很少出现在普通文本里
# 多行条目里也用它,加上前后的 \n 让序列化/反序列化不依赖行数
ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# 4.1 Memory content scanning — 轻量注入/泄露检测
# ---------------------------------------------------------------------------
# Patterns 在 tools/threat_patterns.py(单一来源,所有 scanner 共享)
# Memory 用 "strict" scope(最严):
#   * memory 是 user-curated,被 flag 后可以人工改
#   * memory 进 system prompt 是 frozen 的,中毒条目持续整个 session
#   * 跨 session 也还在(直到显式 remove)
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


# 4.2 _scan_memory_content — 扫一段内容看有没有威胁
# 返 None = 通过;返 str = 错误消息
def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


# 4.3 _drift_error — "磁盘和 parser 状态不一致"时的错误响应
# 场景:
#   * patch tool 改了 MEMORY.md
#   * shell >> MEMORY.md 追加了内容
#   * 另一个 session 同时写入
# 这种情况下,直接写会**覆盖**外部修改 → 静默数据丢失
# 防御:拒绝写入,把磁盘当前内容备份到 .bak.<ts>,告诉用户怎么修
# (issue #26045)
def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


# 5.1 MemoryStore 类 — 490 行的核心
# 每个 AIAgent 持一个实例
#
# === 双状态设计 ===
#   * _system_prompt_snapshot: 冻结 snapshot(只在 load_from_disk 时设)
#                            注入到 system prompt,整个 session 不变
#                            → 保护 LLM prefix cache
#   * memory_entries / user_entries: 活状态
#                                  工具调用 mutate,持久化到磁盘
#                                  工具响应总是反映这个"最新"状态
class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # 5.2 __init__ — 初始状态(还没 load_from_disk,snapshot 是空)
    # memory_char_limit: 2200 字符(不是 token!char 数 model-independent)
    # user_char_limit:   1375 字符
    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    # 5.3 load_from_disk — session 启动时调一次
    # 流程:
    #   1. 读 MEMORY.md / USER.md
    #   2. 去重
    #   3. **sanitize**(每条都过威胁扫描)
    #   4. 渲染成 markdown block
    #   5. 存到 _system_prompt_snapshot
    #
    # **安全设计**:
    #   中毒条目不进 snapshot(被 [BLOCKED: ...] 替换)
    #   但**保留在** memory_entries / user_entries(用户能 read 看到)
    #   → 防止"静默吃掉"用户可能正在 debug 的内容
    #
    # **prefix cache 保证**:
    #   扫描是确定性的(从磁盘字节起算),所以 snapshot 在整个 session 内稳定
    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        ``memory(action=read)`` and remove them — silently dropping them
        would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text so the user
        # can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    # 5.4 _sanitize_entries_for_snapshot — 静态方法
    # 每条 entry 跑 threat scan;命中就用 [BLOCKED: ...] 替换
    # 已经 blocked 标记的 / 空 entry 透传
    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        # === 日志:函数入口 — 记下要扫多少条、来自哪个文件 ===
        # 配 debug 级别:正常情况不打印,只排查时被 grep
        # 用 logger.debug 而不是 .info:这是 hot path(每个 session 启动都跑),
        # info 级别会污染日常日志
        logger.debug(
            "Memory sanitization starting: %d entries from %s",
            len(entries), filename,
        )

        sanitized: List[str] = []
        blocked_count = 0   # 累计:被 blocked 的 entry 数
        passed_count = 0    # 累计:clean 透传的 entry 数
        for idx, entry in enumerate(entries):
            # === 日志:进入扫描前 — 记下当前是第几条 + entry 长度 ===
            # entry 长度比 entry 内容更重要:打印内容可能 PII / prompt injection 残留,
            # 打印长度足以判断"是不是超长 entry"导致的性能问题
            logger.debug(
                "Memory sanitization scanning entry %d/%d from %s (len=%d)",
                idx + 1, len(entries), filename, len(entry),
            )
            if not entry or entry.startswith("[BLOCKED:"):
                # === 日志:空 entry / 已 blocked 的 entry 透传 ===
                # 这条分支不需要做扫描,但要记录"我们看到了,选择跳过"
                # 否则排查"为什么这条 entry 进了 snapshot 却没看到扫描日志"会困惑
                logger.debug(
                    "Memory sanitization entry %d passed through "
                    "(empty or already [BLOCKED: marker])",
                    idx + 1,
                )
                sanitized.append(entry)
                passed_count += 1
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                # === 日志:threat 命中 — 已有 warning,补个 debug 看 entry 预览 ===
                # warning 已经是"必须看到的告警",debug 这里补充:
                #   * entry 前 100 字符(方便排查是哪种内容触发)
                #   * findings 具体 ID(从 warning 也知道,但放一起方便 grep)
                # %r 用 repr 形式打印,空格 / 换行会显式标出
                logger.debug(
                    "Memory sanitization THREAT found: entry %d from %s, "
                    "findings=%s, entry_preview=%r",
                    idx + 1, filename, findings, entry[:100],
                )
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=read) to inspect and memory(action=remove) "
                    f"to delete the original.]"
                )
                blocked_count += 1
            else:
                # === 日志:clean entry — 显式记下"扫过了,没找到" ===
                # 不打的话扫过 1000 条 clean entry 不会留下任何痕迹
                # 排查"是不是某些 entry 被漏扫了"时这条日志能救命
                logger.debug(
                    "Memory sanitization entry %d clean (no threats)",
                    idx + 1,
                )
                sanitized.append(entry)
                passed_count += 1

        # === 日志:函数出口汇总 — 1 行看清结果 ===
        # 比 per-entry 日志更友好:大文件扫完时直接看汇总,
        # 需要细节再 grep 单条
        logger.debug(
            "Memory sanitization done: %d passed, %d blocked, %d total from %s",
            passed_count, blocked_count, len(entries), filename,
        )
        return sanitized

    # 5.5 _file_lock — context manager 形式的文件锁
    # 关键:**用单独的 .lock 文件**锁,而不是锁 memory 文件本身
    # 因为 atomic_replace 是 os.replace() 替换 inode,锁文件 inode 没用
    # 两层 session 写同一个文件不会相互覆盖
    # Unix: fcntl.flock; Windows: msvcrt.locking
    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    # 5.6 _path_for — target 名 → 磁盘文件
    # "user" → USER.md;其它(默认 "memory") → MEMORY.md
    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    # 5.7 _reload_target — 在文件锁下"重新读最新"再 mutate
    # 为什么要 reload?——两个 session 可能同时打开同一个文件
    # 我们 mutate 之前要看到对方的最新修改,避免覆盖
    # 返 backup 路径 = 检测到外部 drift(对方写的格式我们不认)
    #   此时**不**flush,留给上层报错
    # 返 None = 干净 reload,继续 mutate
    def _reload_target(self, target: str) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.
        """
        path = self._path_for(target)
        bak = self._detect_external_drift(target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

    # 5.8 save_to_disk — 每次 mutate 后持久化
    # 走 atomic_replace 写(瞬间切换 inode,不会留半截文件)
    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    # 5.9 _entries_for / _set_entries — target 字符串分派
    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    # 5.10 add — 追加一条 entry
    # 流程:
    #   1. trim content(去前后空白)
    #   2. 空检查
    #   3. **威胁扫描** (strict scope)
    #   4. 文件锁 + reload
    #   5. drift 检查
    #   6. 去重检查(完全相同的不加)
    #   7. 预算检查(加完会不会超字符上限)
    #   8. 真 append + 写盘
    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # If external drift was detected, the file was backed up to .bak.<ts>
            # — refuse the mutation so we don't clobber the un-roundtrippable
            # content the patch tool / shell append / sister session wrote.
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    # 5.11 replace — 替换一条 entry
    # 用**子串匹配**(不是 ID,不是整段)
    # 原因:LLM 经常记不准完整文本
    # 安全设计:多匹配时拒绝(除非全相同)
    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    # 5.12 remove — 删一条 entry(子串匹配)
    # 多匹配拒绝,除非全相同
    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    # 5.13 format_for_system_prompt — 拿 frozen snapshot
    # 关键:**不**返 live state(活状态会 mid-session 变,污染 prefix cache)
    # 返 load_from_disk 时的 snapshot
    # 返 None = 加载时这个 target 没有 entry
    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    # 5.14 _success_response — 标准成功响应(给 LLM 看)
    # 包含:target / entries / usage 百分比 / entry_count / message
    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    # 5.15 _render_block — 渲染 system prompt 块
    # 格式:<HEADER>\n<entries joined>\n
    # HEADER 含 "X% used" 提示(让 LLM 知道剩余空间)
    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _detect_external_drift(self, target: str) -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    # 5.16 _write_file — 写文件的"正确姿势"
    # 关键:不用 open("w")+flock(那会先 truncate 再锁,有竞态)
    # 改用:**temp file + atomic rename**
    #   1. mkstemp 创临时文件
    #   2. 写完 + fsync(确保落盘)
    #   3. atomic_replace = os.replace(瞬间切 inode)
    # 读者永远看到完整的旧文件或完整的新文件
    # 同一个 directory 才保证 atomic rename(不同 fs 上 os.replace 可能 fail)
    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


# 6.1 memory_tool — LLM 调的工具入口
# 单一函数分发 4 个 action:add / replace / remove
# (read 不在这——read 通过 prompt 里的 snapshot + memory tool 的 response 看)
# 参数校验:store 必须有 / target 必须是 memory/user / action 必须有对应必填参数
def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


# 6.2 check_memory_requirements — 兼容性钩子
# 别的工具(比如 web_search)有 check_fn 查"环境是否支持"
# memory 总可用,永远返 True
def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# 7.1 MEMORY_SCHEMA — OpenAI function-calling 格式的工具 schema
# 7.1 关键:description 写得**非常长**——这是 LLM 决定"要不要调"的唯一依据
# 7.1 包含:
#   * WHEN TO SAVE(主动调用的场景)
#   * PRIORITY(用户偏好 > 环境事实 > 流程)
#   * DO NOT save(任务进度、session 结果)
#   * TWO TARGETS 区分
#   * 3 ACTIONS
#   * SKIP 列表
# 7.1 注意:这种"长 description"是 Hermes 工具的普遍风格——
# 7.1 LLM 看到 description 才会知道"什么时候该调 / 不该调"
# 7.1 不像普通 API doc,这是给 LLM 看的"行为指南"
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




