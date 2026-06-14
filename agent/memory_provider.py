"""Abstract base class for pluggable memory providers.

# === 这个文件是干什么的? ===
# 定义"memory provider"的**接口契约**——所有外部 memory 后端
# (Honcho、Hindsight、Mem0、Supermemory 等)都必须实现这个 ABC。
#
# === 什么是 Memory Provider? ===
# LLM 本身"无状态"——关掉对话,下次开就忘了所有事。
# Memory Provider 给 agent 装上"长期记忆":
#   1. **prefetch**: turn 开始前把"可能相关"的旧记忆捞出来,塞进 system prompt
#   2. **sync_turn**: turn 结束后把这一轮对话写回去(异步,用户无感知)
#   3. **get_tool_schemas**: 暴露给 LLM 的工具(如 hindsight_retain / hindsight_recall)
#
# === 和 MemoryManager 的关系 ===
# MemoryProvider = "单个后端" (ABC + 实现)
# MemoryManager = "协调器"   (持有 N 个 provider,统一调它们的接口)
# 关系:Manager 编排,Provider 工作
# 限制:Manager 强制"只能 1 个外部 provider"——防 schema 膨胀
#
# === 8 个外部 provider (在 plugins/memory/ 下) ===
#   honcho / hindsight / mem0 / supermemory / openviking
#   retaindb / holographic / byterover
# 内置 provider (代码里写死,通常叫 'builtin')
#
# === Prefetch/Sync 分离的设计哲学 ===
# turn 前 prefetch: 延迟优化,LLM 拿到的"记忆"已经准备好
# turn 后 sync_turn: 写不阻塞响应(用户不感知延迟)
# 读写分离 → 优化感知延迟

Memory providers give the agent persistent recall across sessions.
The MemoryManager enforces a one-external-provider limit to prevent
tool schema bloat and conflicting memory backends.

External providers (Honcho, Hindsight, Mem0, etc.) are registered
and managed via MemoryManager. Only one external provider runs at a
time.

Registration:
  Plugins ship in plugins/memory/<name>/ and are activated via
  the memory.provider config key.

Lifecycle (called by MemoryManager, wired in run_agent.py):
  initialize()          — connect, create resources, warm up
  system_prompt_block()  — static text for the system prompt
  prefetch(query)        — background recall before each turn
  sync_turn(user, asst)  — async write after each turn
  get_tool_schemas()     — tool schemas to expose to the model
  handle_tool_call()     — dispatch a tool call
  shutdown()             — clean exit

Optional hooks (override to opt in):
  on_turn_start(turn, message, **kwargs) — per-turn tick with runtime context
  on_session_end(messages)               — end-of-session extraction
  on_session_switch(new_session_id, **kwargs) — mid-process session_id rotation
  on_pre_compress(messages) -> str       — extract before context compression
  on_memory_write(action, target, content, metadata=None) — mirror built-in memory writes
  on_delegation(task, result, **kwargs)  — parent-side observation of subagent work
"""

from __future__ import annotations

# === 依赖说明 ===
# ABC: Python 内置的 abstract base class 支持
#   * @abstractmethod 装饰的方法必须被子类实现
#   * 实例化 ABC 子类时,如果有未实现的 abstract 方法,会抛 TypeError
# typing: 类型提示(List/Dict/Optional/Any)
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# MemoryProvider — 单一抽象基类
# =============================================================================
#
# === 8 + 6 + 2 + 4 接口分 4 档 ===
# 这个 ABC 总共定义了 8 个 abstract method(必须实现)
# + 6 个带默认实现的方法(可选重写)
# + 2 个 helper(配置/保存)
# + 4 个 optional hook(纯通知)
# 学习时按下面分组看:
#   * 8 个 abstract = 必修课
#   * 6 个带默认 = 选修课
#   * 4 个 hook = 边角通知
#   * 2 个 helper = 配置文件相关
class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    # === 1/8 必备:provider 短名 ===
    # 用 property 装饰(而不是普通方法),调用方可以 `provider.name` 直接拿
    # 注意:这是唯一用 @property + @abstractmethod 的方法,
    #      因为 name 是"无入参 + 返字符串"的纯属性
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # =================================================================
    # Core lifecycle(8 个 abstract method 的"灵魂 8 件套")
    # =================================================================
    # 这 8 个方法是每个 provider **必须实现** 的核心生命周期
    # MemoryManager 按固定顺序调它们:
    #   init 时:is_available → initialize
    #   每个 turn:prefetch → (LLM) → sync_turn
    #   系统 prompt 拼装:system_prompt_block + prefetch
    #   LLM 调工具:handle_tool_call
    #   退出:shutdown
    # -- Core lifecycle (implement these) ------------------------------------

    # === 2/8 必备:健康检查 ===
    # 在 agent 启动时调一次,决定"激活"还是"跳过"
    # 必须**快速且不联网**——只查 config 和本地 deps
    # 比如:hindsight 查 ~/.hindsight/ 是否存在;honcho 查 HONCHO_API_KEY 环境变量
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured, has credentials, and is ready.

        Called during agent init to decide whether to activate the provider.
        Should not make network calls — just check config and installed deps.
        """

    # === 3/8 必备:初始化 ===
    # 拿到"启动凭证"才能工作:
    #   * 建表 / 建 bank(Hindsight 本地需要建 vector store)
    #   * 登录 / 拿 token(Honcho 等远程)
    #   * 起后台线程(异步 prefetch 的 worker)
    # kwargs 详解:
    #   * hermes_home  ← 多 profile 隔离(每个 profile 有独立 ~/.hermes/<profile>/)
    #   * platform     ← "cli" / "telegram" / "discord" / "cron"
    #   * agent_context← "primary" / "subagent" / "cron" / "flush"
    #                   ← subagent 和 cron 应该 skip write(否则污染 user 表示)
    #   * agent_identity ← profile name,用于 per-profile 数据隔离
    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session.

        Called once at agent startup. May create resources (banks, tables),
        establish connections, start background threads, etc.

        kwargs always include:
          - hermes_home (str): The active HERMES_HOME directory path. Use this
            for profile-scoped storage instead of hardcoding ``~/.hermes``.
          - platform (str): "cli", "telegram", "discord", "cron", etc.

        kwargs may also include:
          - agent_context (str): "primary", "subagent", "cron", or "flush".
            Providers should skip writes for non-primary contexts (cron system
            prompts would corrupt user representations).
          - agent_identity (str): Profile name (e.g. "coder"). Use for
            per-profile provider identity scoping.
          - agent_workspace (str): Shared workspace name (e.g. "hermes").
          - parent_session_id (str): For subagents, the parent's session_id.
          - user_id (str): Platform user identifier (gateway sessions).
          - user_id_alt (str): Optional alternate stable platform user identifier.
        """

    # 3.4 system_prompt_block — 静态文本块(可选重写)
    # 区别于 prefetch:这里是"固定信息"(provider 状态、操作说明)
    # prefetch 是"动态召回"(每次 turn 重新查)
    def system_prompt_block(self) -> str:
        """Return text to include in the system prompt.

        Called during system prompt assembly. Return empty string to skip.
        This is for STATIC provider info (instructions, status). Prefetched
        recall context is injected separately via prefetch().
        """
        return ""

    # 3.5 prefetch — 每次 turn 前预取记忆(可重写)
    # 关键:必须**快**——主循环会等这个结果
    # 慢 IO 的 provider 应该用 queue_prefetch 在后台预跑,这里只返缓存
    # session_id 是给"gateway 群聊"等多 session 场景用的,简单场景忽略
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn.

        Called before each API call. Return formatted text to inject as
        context, or empty string if nothing relevant. Implementations
        should be fast — use background threads for the actual recall
        and return cached results here.

        session_id is provided for providers serving concurrent sessions
        (gateway group chats, cached agents). Providers that don't need
        per-session scoping can ignore it.
        """
        return ""

    # 3.6 queue_prefetch — 排队"下一轮"的预取(可重写)
    # 默认 no-op,需要"提前 1 轮预取"的 provider 才会实现
    # 比如:hindsight 当前 turn 返结果时,同时发起"下个 turn"的 query
    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn.

        Called after each turn completes. The result will be consumed
        by prefetch() on the next turn. Default is no-op — providers
        that do background prefetching should override this.
        """

    # 3.7 sync_turn — 每次 turn 后写记忆(可重写)
    # **不能阻塞**——慢的 provider 必须 queue 起来后台写
    # messages 包含完整的 OpenAI 风格对话历史(assistant tool_call + tool result)
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to the backend.

        Called after each turn. Should be non-blocking — queue for
        background processing if the backend has latency.

        ``messages`` is the OpenAI-style conversation message list as of the
        completed turn, including any assistant tool calls and tool results.
        Providers that do not need raw turn context can ignore it.
        """

    # 3.8 get_tool_schemas — 暴露给 LLM 的工具(4/8 abstract)
    # 返 OpenAI 格式 schema:[{"name": "hindsight_retain", "description": ..., "parameters": ...}, ...]
    # 返 [] 表示这个 provider 是"context-only"——没有工具,只往 system prompt 塞文本
    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this provider exposes.

        Each schema follows the OpenAI function calling format:
        {"name": "...", "description": "...", "parameters": {...}}

        Return empty list if this provider has no tools (context-only).
        """

    # 3.9 handle_tool_call — 实际执行 LLM 调的工具(可重写)
    # 返 JSON 字符串(标准 tool_result 格式)
    # MemoryManager 只对"出现在 get_tool_schemas() 里的"名字调它
    # 默认实现是 raise NotImplementedError——没暴露工具的 provider 不需要重写
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools.

        Must return a JSON string (the tool result).
        Only called for tool names returned by get_tool_schemas().
        """
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    # 3.10 shutdown — 清理退出(可重写,5/8 abstract?其实 shutdown 是默认 no-op)
    # 默认 no-op,需要"关连接/flush 队列"的 provider 才需要重写
    # 比如:hindsight 可能要 flush 它的后台 prefetch 线程
    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # =================================================================
    # Optional hooks(8 个钩子,全部默认 no-op)
    # =================================================================
    # 区别于 8 件套:钩子是"通知",不是"必须"
    # 实现了就能在特定事件上做额外事;不实现什么都不发生
    # -- Optional hooks (override to opt in) ---------------------------------

    # 4.1 on_turn_start — turn 开始通知
    # 用途:turn 计数、定期维护、scope 切换
    # kwargs 里有 remaining_tokens / model / platform / tool_count(provider 自取)
    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn with the user message.

        Use for turn-counting, scope management, periodic maintenance.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        Providers use what they need; extras are ignored.
        """

    # 4.2 on_session_end — session 结束通知
    # 用途:对话结束时的"总结提取"——把所有 turn 压缩成几条核心事实
    # 注意:不是每个 turn 调,是**真结束**才调(CLI exit / /reset / gateway 过期)
    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends (explicit exit or timeout).

        Use for end-of-session fact extraction, summarization, etc.
        messages is the full conversation history.

        NOT called after every turn — only at actual session boundaries
        (CLI exit, /reset, gateway session expiry).
        """

    # 4.3 on_session_switch — session 切换通知(最复杂的钩子)
    # 触发场景:`/resume` `/branch` `/reset` `/new`  +  context 压缩
    # 用途:provider 内部缓存了 per-session 状态时,告诉它"换 session 了,该清理"
    # 比如:honcho 内部有 _document_id,切换时必须重新初始化
    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Called when the agent switches session_id mid-process.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new`` (CLI), the
        gateway equivalents, and context compression — any path that
        reassigns ``AIAgent.session_id`` without tearing the provider down.

        Providers that cache per-session state in ``initialize()``
        (``_session_id``, ``_document_id``, accumulated turn buffers,
        counters) should update or reset that state here so subsequent
        writes land in the correct session's record.

        Parameters
        ----------
        new_session_id:
            The session_id the agent just switched to.
        parent_session_id:
            The previous session_id, if meaningful — set for ``/branch``
            (fork lineage), context compression (continuation lineage),
            and ``/resume`` (the session we're leaving). Empty string
            when no lineage applies.
        reset:
            ``True`` when this is a genuinely new conversation, not a
            resumption of an existing one. Fired by ``/reset`` / ``/new``.
            Providers should flush accumulated per-session buffers
            (``_session_turns``, ``_turn_counter``, etc.) when this is
            set. ``False`` for ``/resume`` / ``/branch`` / compression
            where the logical conversation continues under the new id.
        rewound:
            ``True`` if session_id is unchanged but the transcript was
            truncated; providers caching per-turn document state should
            invalidate.

        Default is no-op for backward compatibility.
        """

    # 4.4 on_pre_compress — 上下文压缩前通知
    # 用途:在旧消息被压成 summary **之前**,把它们"有价值的事实"提取出来
    # 返的字符串会塞进压缩 prompt,让 compressor 知道"这些是 provider 关心的"
    # 默认返 ""(no contribution),保持向后兼容
    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compression discards old messages.

        Use to extract insights from messages about to be compressed.
        messages is the list that will be summarized/discarded.

        Return text to include in the compression summary prompt so the
        compressor preserves provider-extracted insights. Return empty
        string for no contribution (backwards-compatible default).
        """
        return ""

    # 4.5 on_delegation — 父 agent 看到子 agent 完成时
    # 用途:父 agent 的 provider 拿到"我刚派了 X,拿到 Y 结果"——可以记下来
    # 注意:**只** 父 agent 收到,子 agent 自己**没有** provider session
    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Called on the PARENT agent when a subagent completes.

        The parent's memory provider gets the task+result pair as an
        observation of what was delegated and what came back. The subagent
        itself has no provider session (skip_memory=True).

        task: the delegation prompt
        result: the subagent's final response
        child_session_id: the subagent's session_id
        """

    # =================================================================
    # Config helpers(2 个,CLI 配 provider 用)
    # =================================================================

    # 5.1 get_config_schema — 告诉 CLI "我需要哪些配置项"
    # 用途:`hermes memory setup` 命令会用这个 schema 走一遍配置流程
    # 字段含义:见 docstring
    # 返 [] 表示"无需配置"(纯本地 provider,如 builtin)
    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields this provider needs for setup.

        Used by 'hermes memory setup' to walk the user through configuration.
        Each field is a dict with:
          key:         config key name (e.g. 'api_key', 'mode')
          description: human-readable description
          secret:      True if this should go to .env (default: False)
          required:    True if required (default: False)
          default:     default value (optional)
          choices:     list of valid values (optional)
          url:         URL where user can get this credential (optional)
          env_var:     explicit env var name for secrets (default: auto-generated)

        Return empty list if no config needed (e.g. local-only providers).
        """
        return []

    # 5.2 save_config — 把"非 secret"配置写到自己家的位置
    # 比如:hindsight 可能写 ~/.hindsight/config.yaml
    #      honcho 可能写 ~/.honcho/config.json
    # "secret" 类(api_key/token)走 .env,不走这里
    # **新 plugin 必须**实现 save_config 或全用 env var(并在 get_config_schema 里标 env_var)
    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to the provider's native location.

        Called by 'hermes memory setup' after collecting user inputs.
        ``values`` contains only non-secret fields (secrets go to .env).
        ``hermes_home`` is the active HERMES_HOME directory path.

        Providers with native config files (JSON, YAML) should override
        this to write to their expected location. Providers that use only
        env vars can leave the default (no-op).

        All new memory provider plugins MUST implement either:
        - save_config() for native config file formats, OR
        - use only env vars (in which case get_config_schema() fields
          should all have ``env_var`` set and this method stays no-op).
        """

    # 5.3 on_memory_write — 桥接内置 memory 工具的写动作
    # 用途:LLM 调内置 memory_tool("add" / "replace" / "remove")时,
    #      通知外部 provider 也写一份
    # 这样"内置 memory"和"外部 provider"是**双写**的,数据一致
    # metadata 字段约定:write_origin / execution_context / session_id 等
    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when the built-in memory tool writes an entry.

        action: 'add', 'replace', or 'remove'
        target: 'memory' or 'user'
        content: the entry content
        metadata: structured provenance for the write, when available. Common
          keys include ``write_origin``, ``execution_context``, ``session_id``,
          ``parent_session_id``, ``platform``, and ``tool_name``.

        Use to mirror built-in memory writes to your backend.
        """
