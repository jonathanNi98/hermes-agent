"""MemoryManager — orchestrates memory providers for the agent.

# 1.1 这是干什么的
# MemoryManager = 整个 memory 子系统的"协调器"
# 持有 N 个 MemoryProvider,统一调它们的接口
# 取代了早期"散在各处 backend 代码"——所有 memory 逻辑集中这一个文件
#
# 1.2 关键约束
# **只能有 1 个外部 plugin provider**——尝试注册第二个会被拒绝
# 原因:(1) schema 膨胀 (2) 数据双写不一致 (3) 同步死循环
#
# 1.3 在 run_agent.py 里的用法
#   self._memory_manager = MemoryManager()
#   self._memory_manager.add_provider(plugin_provider)  # 至多 1 个
#   # System prompt
#   prompt_parts.append(self._memory_manager.build_system_prompt())
#   # Pre-turn
#   context = self._memory_manager.prefetch_all(user_message)
#   # Post-turn
#   self._memory_manager.sync_all(user_msg, assistant_response)
#   self._memory_manager.queue_prefetch_all(user_msg)
#
# 1.4 文件结构(按编号)
#   2.x Context fencing helpers(sanitize)
#   3.x StreamingContextScrubber 类
#   4.x build_memory_context_block(纯函数)
#   5.x MemoryManager 类(本文件核心)

Single integration point in run_agent.py. Replaces scattered per-backend
code with one manager that delegates to registered providers.

Only ONE external plugin provider is allowed at a time — attempting to
register a second external provider is rejected with a warning.  This
prevents tool schema bloat and conflicting memory backends.

Usage in run_agent.py:
    self._memory_manager = MemoryManager()
    # Only ONE of these:
    self._memory_manager.add_provider(plugin_provider)

    # System prompt
    prompt_parts.append(self._memory_manager.build_system_prompt())

    # Pre-turn
    context = self._memory_manager.prefetch_all(user_message)

    # Post-turn
    self._memory_manager.sync_all(user_msg, assistant_response)
    self._memory_manager.queue_prefetch_all(user_msg)
"""

from __future__ import annotations

# 2.1 Imports 分组说明
# logging   - 标准日志
# re        - 正则(sanitize 用)
# inspect   - 反射(查 provider 方法签名)
# typing    - 类型提示
# MemoryProvider - 接口 ABC
# tool_error - 标准错误返回 helper
import logging
import re
import inspect
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# 2.2 Context fencing 防御
# ---------------------------------------------------------------------------
# 这些正则用来"擦掉"provider 输出里**不应该让 LLM 再引用**的标记
# 防止 prompt injection(LLM 看到旧的 <memory-context> 块,误以为是新指令)
# ---------------------------------------------------------------------------

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>',
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)


# 2.3 sanitize_context(纯函数)
# 用途:provider 返回的"被注入到 prompt 的文本"先过一次这个,
#       把 fence 标签 / system note 都擦掉(防止 LLM 引用 / 注入)
# 三步:
#   1. 删整段 <memory-context>...</memory-context>
#   2. 删 [System note: ... recalled memory context ...]
#   3. 删残留的 <memory-context> 标签
def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
    return text


class StreamingContextScrubber:
    """Stateful scrubber for streaming text that may contain split memory-context spans.

    The one-shot ``sanitize_context`` regex cannot survive chunk boundaries:
    a ``<memory-context>`` opened in one delta and closed in a later delta
    leaks its payload to the UI because the non-greedy block regex needs
    both tags in one string.  This scrubber runs a small state machine
    across deltas, holding back partial-tag tails and discarding
    everything inside a span (including the system-note line).

    Usage::

        scrubber = StreamingContextScrubber()
        for delta in stream:
            visible = scrubber.feed(delta)
            if visible:
                emit(visible)
        trailing = scrubber.flush()  # at end of stream
        if trailing:
            emit(trailing)

    The scrubber is re-entrant per agent instance.  Callers building new
    top-level responses (new turn) should create a fresh scrubber or call
    ``reset()``.
    """

    # 3.1 类常量 — 标签字符串
    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    # 3.2 __init__ — 状态机初始状态
    # _in_span: 当前是不是在 <memory-context>...</> 里
    # _buf:    暂时"扣下"的尾部(可能是标签的开头,等下个 chunk 决定)
    # _at_block_boundary: 上一段可见文本是不是停在"行首空白"(决定下一个 < 是不是 block 起手)
    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    # 3.3 reset — 给新一轮用
    def reset(self) -> None:
        self._in_span = False
        self._buf = ""
        self._at_block_boundary = True

    # 3.4 feed — 处理一个新 chunk
    # 这是状态机的核心:
    #   * 如果在 span 内 → 找 </memory-context>,找到就出 span,找不到就把"可能是 close 标签开头"的尾巴扣下
    #   * 如果不在 span 内 → 找 <memory-context>,找到就入 span,找不到就发可见文本 + 扣 open 标签尾巴
    def feed(self, text: str) -> str:
        """Return the visible portion of ``text`` after scrubbing.

        Any trailing fragment that could be the start of an open/close tag
        is held back in the internal buffer and surfaced on the next
        ``feed()`` call or discarded/emitted by ``flush()``.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close — skip span content + tag, continue
                buf = buf[idx + len(self._CLOSE_TAG):]
                self._in_span = False
            else:
                idx = self._find_boundary_open_tag(buf)
                if idx == -1:
                    # No open tag — hold back a potential partial open tag
                    held = (
                        self._max_pending_open_suffix(buf)
                        or self._max_partial_suffix(buf, self._OPEN_TAG)
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                if idx > 0:
                    self._append_visible(out, buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG):]
                self._in_span = True

        return "".join(out)

    # 3.5 flush — 流结束时
    # 安全策略:如果还在 span 内没关闭,直接丢弃
    # (泄漏部分 memory context 比截断答案更糟)
    # 否则把扣下的尾巴原样吐出来(发现不是真标签)
    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag).
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix.

        Case-insensitive.  Returns 0 if no suffix could start the tag.
        """
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    # 3.6 _max_partial_suffix — "扣下"标签的开头
    # 比如 buf 末尾是 "<memo" → 可能是 "<memory-context>" 的开头
    # 返回这个"可能是前缀"的长度(让 feed 把这段扣下,等下个 chunk 再决定)
    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix.

        Case-insensitive.  Returns 0 if no suffix could start the tag.
        """
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    # 3.7 _find_boundary_open_tag — 找"block 边界"上的 open 标签
    # 区别于"任意位置的 <memory-context>":这里只在**行首**(block 起手)才算
    # 避免 LLM 在中间位置提到 "<memory-context>" 这个字符串被误擦
    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span."""
        buf_lower = buf.lower()
        search_start = 0
        while True:
            idx = buf_lower.find(self._OPEN_TAG, search_start)
            if idx == -1:
                return -1
            if self._is_block_boundary(buf, idx) and self._has_block_opener_suffix(buf, idx):
                return idx
            search_start = idx + 1

    # 3.8 _max_pending_open_suffix — "整标签已到,等下一个字符确认"
    # buf 完整以 <memory-context> 结尾,但需要看下个字符是不是 \r\n
    # (确保它真的是 block 起手,而不是行内文字)
    def _max_pending_open_suffix(self, buf: str) -> int:
        """Hold a complete boundary tag until the following char confirms it."""
        if not buf.lower().endswith(self._OPEN_TAG):
            return 0
        idx = len(buf) - len(self._OPEN_TAG)
        if not self._is_block_boundary(buf, idx):
            return 0
        return len(self._OPEN_TAG)

    # 3.9 _has_block_opener_suffix — open 标签后面必须是 \r 或 \n
    # (这才能说明它真的是 block 起手)
    def _has_block_opener_suffix(self, buf: str, idx: int) -> bool:
        after_idx = idx + len(self._OPEN_TAG)
        if after_idx >= len(buf):
            return False
        return buf[after_idx] in "\r\n"

    # 3.10 _is_block_boundary — "这个位置之前是不是行首(空白)"
    # idx=0 时看 _at_block_boundary(从上一段继承)
    # 否则看 buf[:idx] 的最后一个 \n 后面是不是空白
    def _is_block_boundary(self, buf: str, idx: int) -> bool:
        if idx == 0:
            return self._at_block_boundary
        preceding = buf[:idx]
        last_newline = preceding.rfind("\n")
        if last_newline == -1:
            return self._at_block_boundary and preceding.strip() == ""
        return preceding[last_newline + 1:].strip() == ""

    # 3.11 _append_visible — 加可见文本 + 顺手更新 boundary 状态
    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    # 3.12 _update_block_boundary — 看完新一段文本,刷新"我是不是在行首"
    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            self._at_block_boundary = text[last_newline + 1:].strip() == ""
        else:
            self._at_block_boundary = self._at_block_boundary and text.strip() == ""


# 4.1 build_memory_context_block(纯函数)
# 用途:把 provider 返回的 raw context 包成标准 fence 块
# 格式:
#   <memory-context>
#   [System note: ... not new user input ...]
#   <provider 返回的内容>
#   </memory-context>
#
# **安全检查**:如果 provider 自己已经包过 fence,sanitize 后会变,
# logger.warning 提示"provider 在重复包装"——bug 信号
def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in a fenced block with system note."""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )


# 5.1 MemoryManager — 协调器
# 设计要点:
#   1. 内置 builtin 永远在第一个
#   2. 外部 plugin 至多 1 个(被 add_provider 强制)
#   3. 一个 provider 失败不能 block 其它(try/except 隔离)
#   4. 提供统一的"prefetch / sync / build_system_prompt"接口给 run_agent
class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider.

    The builtin provider is always first. Only one non-builtin (external)
    provider is allowed.  Failures in one provider never block the other.
    """

    # 5.2 __init__ — 初始状态
    # _providers:        全部已注册的 provider(builtin 永远在 index 0)
    # _tool_to_provider: 工具名 → provider 的路由表(LLM 调工具时用)
    # _has_external:     是否已有非 builtin 的 provider(用来强制"只能 1 个")
    def __init__(self) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False  # True once a non-builtin provider is added

    # -- Registration --------------------------------------------------------

    # 5.3 add_provider — 注册一个 provider
    # 规则:
    #   * builtin 永远接受
    #   * 非 builtin 只能加 1 个(第二个被 warning 拒掉)
    #   * 顺便建工具名 → provider 路由表
    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a memory provider.

        Built-in provider (name ``"builtin"``) is always accepted.
        Only **one** external (non-builtin) provider is allowed — a second
        attempt is rejected with a warning.
        """
        is_builtin = provider.name == "builtin"

        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if p.name != "builtin"), "unknown"
                )
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider is "
                    "allowed at a time. Configure which one via memory.provider "
                    "in config.yaml.",
                    provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # Index tool names → provider for routing
        for schema in provider.get_tool_schemas():
            tool_name = schema.get("name", "")
            if tool_name and tool_name not in self._tool_to_provider:
                self._tool_to_provider[tool_name] = provider
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s, "
                    "ignoring from %s",
                    tool_name,
                    self._tool_to_provider[tool_name].name,
                    provider.name,
                )

        logger.info(
            "Memory provider '%s' registered (%d tools)",
            provider.name,
            len(provider.get_tool_schemas()),
        )

    # 5.4 providers — 公开 provider 列表(返回 copy,避免外部 mutate)
    @property
    def providers(self) -> List[MemoryProvider]:
        """All registered providers in order."""
        return list(self._providers)

    # 5.5 get_provider — 按名字查找
    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """Get a provider by name, or None if not registered."""
        for p in self._providers:
            if p.name == name:
                return p
        return None

    # -- System prompt -------------------------------------------------------

    # 5.6 build_system_prompt — 收集所有 provider 的"静态"system prompt 块
    # 错误隔离:一个 provider 抛了不影响其它
    def build_system_prompt(self) -> str:
        """Collect system prompt blocks from all providers.

        Returns combined text, or empty string if no providers contribute.
        Each non-empty block is labeled with the provider name.
        """
        blocks = []
        for provider in self._providers:
            try:
                block = provider.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' system_prompt_block() failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(blocks)

    # -- Prefetch / recall ---------------------------------------------------

    # 5.7 prefetch_all — turn 前预取所有 provider 的记忆
    # 返拼接好的文本(给 LLM 当 background 上下文)
    # 用 debug 不是 warning:这是非关键路径(LLM 没记忆也照样能聊)
    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers.

        Returns merged context text labeled by provider. Empty providers
        are skipped. Failures in one provider don't block others.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.prefetch(query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    # 5.8 queue_prefetch_all — turn 后排队"下一轮"的预取
    # 大部分 provider 默认 no-op,只有自己起后台线程的才需要实现
    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch on all providers for the next turn."""
        for provider in self._providers:
            try:
                provider.queue_prefetch(query, session_id=session_id)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' queue_prefetch failed (non-fatal): %s",
                    provider.name, e,
                )

    # -- Sync ----------------------------------------------------------------

    # 5.9 _provider_sync_accepts_messages — 反射检查
    # 为什么要反射?——不同 provider 的 sync_turn 签名可能不一样:
    #   * 老 provider 没 messages 参数
    #   * 新 provider 加了 messages: list[dict]
    #   * 有些用 **kwargs(全收)
    # 这里用 inspect 看签名,决定要不要传 messages
    # 这是个**兼容性** shim——保证新 manager 不会把老 provider 调崩
    @staticmethod
    def _provider_sync_accepts_messages(provider: MemoryProvider) -> bool:
        """Return whether sync_turn accepts a messages keyword."""
        try:
            signature = inspect.signature(provider.sync_turn)
        except (TypeError, ValueError):
            return True
        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return True
        return "messages" in signature.parameters

    # 5.10 sync_all — turn 后把这一轮写入所有 provider
    # **警告级**:sync 失败比 prefetch 严重(可能丢数据)
    # 所以这里用 warning 不是 debug
    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Sync a completed turn to all providers."""
        for provider in self._providers:
            try:
                if messages is not None and self._provider_sync_accepts_messages(provider):
                    provider.sync_turn(
                        user_content,
                        assistant_content,
                        session_id=session_id,
                        messages=messages,
                    )
                else:
                    provider.sync_turn(
                        user_content,
                        assistant_content,
                        session_id=session_id,
                    )
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' sync_turn failed: %s",
                    provider.name, e,
                )

    # -- Tools ---------------------------------------------------------------

    # 5.11 get_all_tool_schemas — 收集所有 provider 的工具 schema
    # seen 集合去重(理论上 add_provider 已经处理冲突了,这里双保险)
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect tool schemas from all providers."""
        schemas = []
        seen = set()
        for provider in self._providers:
            try:
                for schema in provider.get_tool_schemas():
                    name = schema.get("name", "")
                    if name and name not in seen:
                        schemas.append(schema)
                        seen.add(name)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' get_tool_schemas() failed: %s",
                    provider.name, e,
                )
        return schemas

    # 5.12 get_all_tool_names — 给 LLM 看的"工具名集合"
    # 用 set:LLM toolset 判断 O(1) 查
    def get_all_tool_names(self) -> set:
        """Return set of all tool names across all providers."""
        return set(self._tool_to_provider.keys())

    # 5.13 has_tool — O(1) 查"这个工具是不是 memory provider 的"
    def has_tool(self, tool_name: str) -> bool:
        """Check if any provider handles this tool."""
        return tool_name in self._tool_to_provider

    # 5.14 handle_tool_call — 路由:哪个 provider 负责这个工具
    # 调 tool_executor.py 那个 9 elif 路由表之前先来这查
    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """Route a tool call to the correct provider.

        Returns JSON string result. Raises ValueError if no provider
        handles the tool.
        """
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error(
                "Memory provider '%s' handle_tool_call(%s) failed: %s",
                provider.name, tool_name, e,
            )
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    # -- Lifecycle hooks -----------------------------------------------------

    # 5.15 on_turn_start — 通知所有 provider "新 turn 开始"
    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Notify all providers of a new turn.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        """
        for provider in self._providers:
            try:
                provider.on_turn_start(turn_number, message, **kwargs)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_turn_start failed: %s",
                    provider.name, e,
                )

    # 5.16 on_session_end — session 真正结束时(CLI 退出 / /reset)
    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Notify all providers of session end."""
        for provider in self._providers:
            try:
                provider.on_session_end(messages)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                )

    # ─────────────────────────────────────────────────────────────────────
    # 5.17 on_session_switch — session_id 轮转通知
    # ─────────────────────────────────────────────────────────────────────
    #
    # === 这是干什么的? ===
    # 通知所有已注册的 provider:"agent 的 session_id 变了",
    # 让他们**在不重启**的情况下刷新内部缓存,确保后续 write 落到正确的 session。
    #
    # === 5 个触发场景(都是"session_id 变了"或"等价于变了") ===
    #   1. /resume    恢复历史对话(同一个 session,继续聊)
    #   2. /branch    从某个节点 fork 出去(新 session_id,有 parent lineage)
    #   3. /reset     /new  全新对话(新 session_id,无 parent)
    #   4. context compression  旧消息被压成 summary,session_id 跟着变(continuation lineage)
    #   5. /undo      撤回若干 turn(session_id 不变,但 transcript 缩短 → rewound=True)
    #
    # === 为什么不让 provider shutdown + initialize? ===
    # 5 个场景里 provider 跟 LLM API 的连接、加载的模型、起的 worker 线程**都能复用**。
    # 如果每次都 shutdown + initialize:
    #   * 慢(honcho / mem0 远端要重新握手)
    #   * 数据丢失(本地 vector index 要重新建)
    #   * 资源浪费(起新线程)
    # 所以折中:**保留 provider 实例**,只让它"清掉 per-session 缓存"。
    #
    # === 2 个 boolean 的语义 ===
    #   reset=True    全新对话,provider 应该 flush 累积 buffer(_session_turns / _turn_counter)
    #   rewound=True  session_id 不变但 transcript 缩短(/undo),
    #                 provider 应该 invalidate per-turn document state
    # 这俩**不互斥**——理论上可以 reset+rewound,但实践中只会有一个=True。
    #
    # === 关键设计:rewound 条件转发 ===
    # 普通的 /resume /branch /new / compression 都**不**传 rewound。
    # 只有 /undo 显式 set True。
    # 如果无脑把 rewound=False 塞进 **kwargs,会污染那些"用 **kwargs 捕获额外参数"的 provider
    # —— 它们的断言(`assert kwargs == {}`)会爆。
    # 所以这里**只**在 rewound 真为 True 时才塞进 kwargs。
    #
    # === 错误隔离 ===
    # 单个 provider 抛异常不影响其它(同 prefetch / sync_turn)。
    # 用 debug 不是 warning:session 切换不是关键路径,失败了就让 provider 自己
    # 用 stale state 继续(下次 sync_turn 还是会调,有机会修正)。
    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Notify all providers that the agent's session_id has rotated.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new``, and
        context compression — any path that reassigns
        ``AIAgent.session_id`` without tearing the provider down.

        Providers keep running; they only need to refresh cached
        per-session state so subsequent writes land in the correct
        session's record. See ``MemoryProvider.on_session_switch`` for
        the full contract.

        ``rewound=True`` signals that session_id is unchanged but the
        transcript was truncated; providers caching per-turn document
        state should invalidate.
        """
        # 1) 空字符串保护:有些 caller 可能在状态异常时传 ""(比如 CLI 启动前),
        #    此时不该让 provider 跑去刷一个"空 session"的状态。
        #    直接 no-op 静默返回,不报错。
        if not new_session_id:
            return
        # 2) rewound 条件转发——见上方"关键设计"注释。
        #    只有 /undo 路径会传 True,其它场景都跳过,保持 kwargs 干净。
        if rewound:
            kwargs["rewound"] = True
        # 3) 遍历所有 provider,挨个通知。
        #    reset / parent_session_id 总是显式传(它们每个调用方都会 set,
        #    不会污染 **kwargs);rewound 由上面 if 决定是否塞进 kwargs。
        for provider in self._providers:
            try:
                provider.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:
                # debug 而非 warning:不是关键路径,失败可恢复
                logger.debug(
                    "Memory provider '%s' on_session_switch failed: %s",
                    provider.name, e,
                )

    # 5.18 on_pre_compress — 压缩前让 provider 提取"关心的事实"
    # 返的文本塞进压缩 prompt,compressor 知道"这些要保留"
    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Notify all providers before context compression.

        Returns combined text from providers to include in the compression
        summary prompt. Empty string if no provider contributes.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.on_pre_compress(messages)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_pre_compress failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────
    # 5.21 _provider_memory_write_metadata_mode — 反射分派兼容器
    # ─────────────────────────────────────────────────────────────────────
    #
    # === 这是干什么的? ===
    # 不同 provider 的 `on_memory_write` 签名不一样,这个方法**用反射**判断
    # 该怎么传 metadata 进去。返 3 种 mode 之一:`"keyword"` / `"positional"` / `"legacy"`。
    #
    # === 为什么需要这种兼容层? ===
    # 历史上 `on_memory_write` 签名演化过:
    #   阶段 1(老):def on_memory_write(action, target, content)         → 3 个参数,无 metadata
    #   阶段 2(中):def on_memory_write(action, target, content, metadata) → 4 个参数,metadata 位置参数
    #   阶段 3(新):def on_memory_write(action, target, content, *, metadata=None)
    #                                                              → metadata 关键字参数
    # 还有些 provider 用 **kwargs("全收")。
    # 新 manager 调老 provider 必须 100% 兼容——否则升级即崩。
    # 跟 5.9 `_provider_sync_accepts_messages` 是**同一个 shim 模式**(都靠 inspect 反射)。
    #
    # === 3 种 mode 的判别优先级 ===
    #   1. **keyword**(优先):有 **kwargs 或显式 metadata= 关键字参数 → 用 keyword 调
    #   2. **positional**:有 ≥4 个位置参数(老位置式 metadata) → 用位置调
    #   3. **legacy**:3 个参数的老 API → 不传 metadata
    #
    # === 异常处理 ===
    # inspect.signature 偶尔会抛 (TypeError, ValueError)——某些用 C 写的
    # 内置方法 / decorated 过的方法拿不到签名。
    # 这种情况下保守地当 "keyword" 处理——传 metadata= 进去,**最坏情况是
    # provider 收到意外 kwarg 报错**,但 try/except 在 caller 那层兜了。
    @staticmethod
    def _provider_memory_write_metadata_mode(provider: MemoryProvider) -> str:
        """Return how to pass metadata to a provider's memory-write hook."""
        # 1) 拿签名。失败 → 保守用 keyword(给 caller try/except 兜底的机会)
        try:
            signature = inspect.signature(provider.on_memory_write)
        except (TypeError, ValueError):
            return "keyword"

        params = list(signature.parameters.values())
        # 2) 优先判断 "keyword"——只要有 **kwargs 兜底,keyword 调一定不会爆
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return "keyword"
        # 3) 显式 metadata 关键字参数(如 stage 3 的 `*, metadata=None`)
        if "metadata" in signature.parameters:
            return "keyword"

        # 4) 没有 metadata 也没有 **kwargs → 只能位置调
        #    筛掉 *args(那不算"接受的固定参数"),只看 named params
        accepted = [
            p for p in params
            if p.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        # 5) ≥4 个 → 位置式 metadata 存在(stage 2:`def foo(a, b, c, metadata)`)
        if len(accepted) >= 4:
            return "positional"
        # 6) <4 个 → 老 API,3 个参数,无 metadata
        return "legacy"

    # ─────────────────────────────────────────────────────────────────────
    # 5.22 on_memory_write — 内置 memory 工具写入时,通知外部 provider
    # ─────────────────────────────────────────────────────────────────────
    #
    # === 这是干什么的? ===
    # LLM 调内置 `memory` 工具(add/replace/remove)时,manager **同时**把
    # 这次写动作转发给所有**非 builtin** 的 provider,让它们各自决定要不要
    # **双写**一份到自己的后端。
    #
    # === 为什么 builtin 自己要 skip? ===
    # builtin 是"source of truth"——它自己就是写 MEMORY.md 的那个,
    # 再 notify 它会形成**自循环**(builtin 写 → manager 通知 → builtin 再写)。
    # 用 `provider.name == "builtin"` 判断(跟 add_provider 用的同一标志)。
    #
    # === 三路分派 + 错误隔离 ===
    # 3 种 metadata_mode 决定怎么调,try/except 包住保证**单 provider 失败不影响其它**。
    # debug 级别:不是关键路径,跟 on_session_switch 同样哲学。
    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify external providers when the built-in memory tool writes.

        Skips the builtin provider itself (it's the source of the write).
        """
        # 1) 遍历所有 provider,builtin 跳过
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                # 2) 反射判断这个 provider 吃哪种 metadata 签名
                metadata_mode = self._provider_memory_write_metadata_mode(provider)
                # 3) 三路分派:
                #   - keyword   → metadata=dict(metadata or {}) 关键字传
                #   - positional→ metadata 位置传(老 API)
                #   - legacy    → 不传 metadata(老 API)
                # dict(metadata or {}) 防御性 copy:防止 caller 之后改原 dict
                if metadata_mode == "keyword":
                    provider.on_memory_write(
                        action, target, content, metadata=dict(metadata or {})
                    )
                elif metadata_mode == "positional":
                    provider.on_memory_write(action, target, content, dict(metadata or {}))
                else:
                    provider.on_memory_write(action, target, content)
            except Exception as e:
                # debug 级别:不是关键路径,失败可恢复
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s",
                    provider.name, e,
                )

    # ─────────────────────────────────────────────────────────────────────
    # 5.23 on_delegation — 父 agent 通知 "我刚派了子任务,结果是这样"
    # ─────────────────────────────────────────────────────────────────────
    #
    # === 这是干什么的? ===
    # 当父 agent **delegate** 一个子 agent(跑完了)时,把 task + result
    # 转发给**父 agent 自己**的所有 provider。让父的 memory 记一笔
    # "我干过这件事,结果是这样"。
    #
    # === 跟 on_memory_write 的关键区别 ===
    # 1. **没有 skip builtin**——builtin 也想记这次 delegation
    #    (用户可能关心"我之前都派过什么子任务")
    # 2. **没有反射分派**——`on_delegation` 签名在 8 个 provider 里一致
    #    (task, result, *, child_session_id, **kwargs),不需要兼容层
    # 3. **子 agent 自己不会收到这个事件**——子 agent 起新 session 时
    #    `skip_memory=True`,不在它的 _providers 里
    #
    # === 错误隔离 ===
    # 同上,debug 级别 + try/except 单点。
    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Notify all providers that a subagent completed."""
        for provider in self._providers:
            try:
                provider.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_delegation failed: %s",
                    provider.name, e,
                )

    # 5.19 shutdown_all — 退出时清理所有 provider
    # **逆序**清理:后注册先关(栈式资源释放)
    def shutdown_all(self) -> None:
        """Shut down all providers (reverse order for clean teardown)."""
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' shutdown failed: %s",
                    provider.name, e,
                )

    # 5.20 initialize_all — 启动时初始化所有 provider
    # 自动注入 hermes_home(每个 provider 都需要知道 home 在哪)
    # 用 lazy import 避免循环依赖
    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers.

        Automatically injects ``hermes_home`` into *kwargs* so that every
        provider can resolve profile-scoped storage paths without importing
        ``get_hermes_home()`` themselves.
        """
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            try:
                provider.initialize(session_id=session_id, **kwargs)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' initialize failed: %s",
                    provider.name, e,
                )
