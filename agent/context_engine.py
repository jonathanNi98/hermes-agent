"""Abstract base class for pluggable context engines.

A context engine controls how conversation context is managed when
approaching the model's token limit. The built-in ContextCompressor
is the default implementation. Third-party engines (e.g. LCM) can
replace it via the plugin system or by being placed in the
``plugins/context_engine/<name>/`` directory.

Selection is config-driven: ``context.engine`` in config.yaml.
Default is ``"compressor"`` (the built-in). Only one engine is active.

The engine is responsible for:
  - Deciding when compaction should fire
  - Performing compaction (summarization, DAG construction, etc.)
  - Optionally exposing tools the agent can call (e.g. lcm_grep)
  - Tracking token usage from API responses

Lifecycle:
  1. Engine is instantiated and registered (plugin register() or default)
  2. on_session_start() called when a conversation begins
  3. update_from_response() called after each API response with usage data
  4. should_compress() checked after each turn
  5. compress() called when should_compress() returns True
  6. on_session_end() called at real session boundaries (CLI exit, /reset,
     gateway session expiry) — NOT per-turn
"""
# ═══════════════════════════════════════════════════════════════
# 【学习要点】context_engine.py 的角色 —— "可插拔的接口"
#
# 跟 system_prompt.py / prompt_builder.py 完全不同:
#   - 这两个是"具体实现"
#   - context_engine.py 是"抽象基类 (ABC)"
#   - 它**只定义接口**,不实现任何逻辑
#
# 解决什么问题:
#   "如何管理超出模型 token 限制的对话上下文"
#   - 默认实现: ContextCompressor (传统"压缩老消息"方式)
#   - 第三方实现: LCM (Logical Context Management, DAG 形式)
#   - 自定义实现: 用户/插件写的别的
#
# 关键设计: **可插拔**
#   - 用户在 config.yaml 选 engine: "compressor" / "lcm" / ...
#   - 主循环永远调相同的接口方法
#   - 不同实现可以无差别替换
#   - 这就是 OOP 里 "abstract base class + multiple implementations" 的标准模式
#
# 跟 system_prompt 链路的关系:
#   - system_prompt.py 负责 system 消息的构造
#   - context_engine.py 负责 conversation 历史的压缩/管理
#   - 两者都跟"对话历史怎么发"有关,但职责分开
#
# ABC 模式在 Hermes 里的应用:
#   - ContextEngine (本文件)   ← 上下文引擎
#   - 后续可能还有: ToolProvider / MemoryProvider 等
# ═══════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """Base class all context engines must implement."""

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 1】class ContextEngine(ABC) —— 抽象基类
    # ABC = Abstract Base Class (Python abc 模块)
    # 不能直接实例化,必须由子类实现所有 @abstractmethod 才能用
    # 这就是 "接口" 的 Python 实现方式
    # ═══════════════════════════════════════════════════════════════

    # -- Identity ----------------------------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 2】name 属性(抽象方法)
    # 用途: 给引擎一个短标识符("compressor" / "lcm" / ...)
    # 调用方: 启动时调一次,用来选 engine / 显示 / 日志
    # 必须实现: 所有子类都得有 name
    # ═══════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # -- Token state (read by run_agent.py for display/logging) ------------
    #
    # Engines MUST maintain these. run_agent.py reads them directly.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 3】6 个 token 状态字段
    # 作用: run_agent.py 直接读这些字段做显示/日志
    # 引擎必须维护(子类赋值)
    #
    # 字段含义:
    #   last_prompt_tokens   - 上次 API 调用的 prompt token 数
    #   last_completion_tokens - 上次 API 调用的 completion token 数
    #   last_total_tokens    - 上次 API 调用的总 token 数
    #   threshold_tokens     - 压缩阈值(达到这个值就压缩)
    #   context_length       - 模型的最大 context window
    #   compression_count    - 本 session 已压缩次数
    # ═══════════════════════════════════════════════════════════════

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- Compaction parameters (read by run_agent.py for preflight) --------
    #
    # These control the preflight compression check.  Subclasses may
    # override via __init__ or property; defaults are sensible for most
    # engines.
    #
    # protect_first_n semantics (since PR #13754): count of non-system head
    # messages always preserved verbatim, IN ADDITION to the system prompt
    # which is always implicitly protected.  Default 3 keeps the
    # historical "system + first 3 non-system messages" head shape.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4】3 个压缩参数
    # 作用: 决定"什么时候压缩"和"压缩时保留多少头/尾"
    # 默认值对大多数引擎都合理
    #
    # threshold_percent: 0.75 意味着用到 75% context 就压缩
    #   留 25% 余量,防止新消息一发就超
    #
    # protect_first_n: 3 保留头部 3 条非 system 消息(原样不动)
    #   → 早期对话不会被压缩掉
    #
    # protect_last_n: 6 保留尾部 6 条消息(原样不动)
    #   → 最新对话永远完整
    #
    # 压缩时只压"中间"部分 (头 3 + 尾 6 之间的)
    # ═══════════════════════════════════════════════════════════════

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- Core interface ----------------------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 5】核心接口(3 个 @abstractmethod)
    # 这 3 个是任何引擎**必须实现**的:
    #   - update_from_response(usage)
    #   - should_compress(prompt_tokens)
    #   - compress(messages, ...)
    # 缺一个 → 子类无法实例化(ABC 强制)
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call with a normalized usage dict. The legacy
        keys ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``
        are always present. Newer hosts also include canonical buckets:
        ``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens``, and ``reasoning_tokens``. Engines should
        treat those fields as optional for compatibility with older hosts.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 5.1】update_from_response(usage) —— 更新 token 用量
        # 调用时机: 每个 API 响应后
        # 输入: 标准化的 usage dict(包含 prompt/completion/total + 5 个新字段)
        # 职责: 更新自己的 last_*_tokens 字段
        # 兼容: 5 个新字段(input/output/cache_read/cache_write/reasoning)是可选的
        #       老 provider 不提供时,引擎应能正常工作
        # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 5.2】should_compress(prompt_tokens) —— 要不要压缩
        # 调用时机: 每个 turn 后
        # 输入: 当前 prompt 的 token 数(可选)
        # 输出: True = 应该压缩 / False = 继续累积
        # 决策依据: 通常是 last_prompt_tokens > threshold_tokens
        #          或是 prompt_tokens 超过 context window 的 75%
        # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Compact the message list and return the new message list.

        This is the main entry point. The engine receives the full message
        list and returns a (possibly shorter) list that fits within the
        context budget. The implementation is free to summarize, build a
        DAG, or do anything else — as long as the returned list is a valid
        OpenAI-format message sequence.

        Args:
            focus_topic: Optional topic string from manual ``/compress <focus>``.
                Engines that support guided compression should prioritise
                preserving information related to this topic.  Engines that
                don't support it may simply ignore this argument.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 5.3】compress(messages, ...) —— 真正执行压缩
        # 输入: 完整 messages 列表
        # 输出: (可能更短)的消息列表(仍是 OpenAI 格式)
        #
        # 自由度高: 可以:
        #   - 老消息总结成 summary
        #   - 构建 DAG(LCM 风格)
        #   - 砍掉冗余 tool_calls
        #   - 任何其他策略
        # 唯一约束: 输出必须仍是 OpenAI 格式消息列表
        #
        # focus_topic: 用户手动 /compress <focus> 的可选主题
        #              支持有向压缩的引擎应该优先保留跟 focus 相关的信息
        # ═══════════════════════════════════════════════════════════════

    # -- Optional: pre-flight check ----------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 6】pre-flight 检查(2 个有默认实现的方法)
    # 可选: 子类可以不 override(用默认 False / 不重算)
    # 用途: API 调用前的"廉价"检查,避免无谓压缩
    # ═══════════════════════════════════════════════════════════════

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call (no real token count yet).

        Default returns False (skip pre-flight). Override if your engine
        can do a cheap estimate.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 6.1】should_compress_preflight(messages) —— 起飞前检查
        # 用途: 还没拿到真实 API 响应时,用粗估判断要不要压缩
        # 默认: False (跳过 preflight)
        # 子类 override: 能廉价估算的话可以提前压缩(避免真的超)
        # ═══════════════════════════════════════════════════════════════
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True when preflight should trust recent real usage instead.

        Built-in compression uses this to avoid re-compacting from known-noisy
        rough estimates after a compressed request has already fit. Third-party
        engines can ignore it safely.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 6.2】should_defer_preflight_to_real_usage —— 决定用粗估 vs 真实数据
        # 用途: 防止"基于已知 noisy 粗估再次压缩"导致的死循环
        # 场景: 上次压缩后真实数据 OK,但粗估还是说"超了" → 别再压
        # 默认: False (第三方引擎可以忽略)
        # ═══════════════════════════════════════════════════════════════
        return False

    # -- Optional: manual /compress preflight ------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 7】has_content_to_compress —— gateway /compress 预检
    # 用途: 用户手动调 /compress 命令时的预检
    # 默认: True (总是尝试)
    # ═══════════════════════════════════════════════════════════════

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check: is there anything in ``messages`` that can be compacted?

        Used by the gateway ``/compress`` command as a preflight guard —
        returning False lets the gateway report "nothing to compress yet"
        without making an LLM call.

        Default returns True (always attempt).  Engines with a cheap way
        to introspect their own head/tail boundaries should override this
        to return False when the transcript is still entirely protected.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 7.1】has_content_to_compress —— 是否有可压缩内容
        # 用途: 防止 gateway 在没东西可压时还调 LLM 浪费钱
        # 返回 False: 没东西可压(gateway 报告 "nothing to compress yet")
        # 返回 True: 有内容可压,继续
        # 默认: True (总是尝试,让 compressor 决定)
        # 子类 override: 能便宜地看 head/tail 边界时可以优化
        # ═══════════════════════════════════════════════════════════════
        return True

    # -- Optional: session lifecycle ---------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 8】session 生命周期 3 个钩子
    # on_session_start: 新 session 启动(加载持久化状态)
    # on_session_end: session 真正结束(刷状态/关连接)
    # on_session_reset: /new 或 /reset 命令
    # 注意: NOT 每个 turn 调,只在 session 真正切换时
    # ═══════════════════════════════════════════════════════════════

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Use this to load persisted state (DAG, store) for the session.
        kwargs may include hermes_home, platform, model, etc.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 8.1】on_session_start —— session 开始钩子
        # 触发: 新 session 启动(gateway 新请求 / CLI 启动)
        # 用途: 加载持久化状态(LCM 加载 DAG 等)
        # kwargs: 可能有 hermes_home / platform / model 等
        # 默认: 空(什么都不做)
        # ═══════════════════════════════════════════════════════════════

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry).

        Use this to flush state, close DB connections, etc.
        NOT called per-turn — only when the session truly ends.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 8.2】on_session_end —— session 结束钩子
        # 触发: 真实 session 边界(CLI 退出 / /reset / gateway 过期)
        # 用途: 刷状态、关 DB 连接、保存 DAG
        # 关键: NOT 每个 turn 调,只在 session 真正结束时
        # 默认: 空
        # ═══════════════════════════════════════════════════════════════

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset per-session state.

        Default resets compression_count and token tracking.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 8.3】on_session_reset —— /new 或 /reset 触发的重置
        # 默认实现: 把 token 计数和 compression_count 清零
        # 子类可以 override 加自己的清理逻辑
        # ═══════════════════════════════════════════════════════════════
        # 8.3.1 重置 4 个 token 字段
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        # 8.3.2 重置压缩计数
        self.compression_count = 0

    # -- Optional: tools ---------------------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 9】工具暴露(2 个方法)
    # 用途: 一些引擎(LCM)给 agent 暴露额外工具
    # 例: LCM 暴露 lcm_grep / lcm_describe / lcm_expand
    # 默认: 不暴露任何工具
    # ═══════════════════════════════════════════════════════════════

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides to the agent.

        Default returns empty list (no tools). LCM would return schemas
        for lcm_grep, lcm_describe, lcm_expand here.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 9.1】get_tool_schemas —— 列出引擎要暴露的工具
        # 用途: 引擎(如 LCM)可以给 agent 提供额外工具
        # 默认: 空列表(传统 Compressor 不需要工具)
        # LCM 实现: 返回 [lcm_grep_schema, lcm_describe_schema, lcm_expand_schema]
        # ═══════════════════════════════════════════════════════════════
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent.

        Only called for tool names returned by get_tool_schemas().
        Must return a JSON string.

        kwargs may include:
          messages: the current in-memory message list (for live ingestion)
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 9.2】handle_tool_call —— 处理 agent 调的工具
        # 输入: 工具名 + 参数
        # 输出: JSON 字符串(OpenAI tool 响应格式)
        # 调用: 只在 get_tool_schemas() 返回的工具名被调
        # 默认实现: 报错(因为默认 get_tool_schemas() 返回 [])
        # 真的有用工具的引擎(LCM)在这里 dispatch
        # ═══════════════════════════════════════════════════════════════
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 10】get_status —— 状态显示/日志
    # 用途: CLI / /insights 命令读这个 dict 来显示 token 用量
    # 返回: 标准字段(供上层消费)
    # ═══════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging.

        Default returns the standard fields run_agent.py expects.
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 10.1】get_status —— 状态显示
        # 字段:
        #   last_prompt_tokens  - 上次 prompt token 数
        #   threshold_tokens     - 压缩阈值
        #   context_length       - 模型最大 context
        #   usage_percent        - 使用百分比
        #   compression_count    - 压缩次数
        # 默认实现: 标准字段
        # ═══════════════════════════════════════════════════════════════
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Optional: model switch support ------------------------------------
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 11】update_model —— 模型切换支持
    # 触发: 用户切换 model / fallback 激活
    # 默认实现: 更新 context_length + 重新算 threshold
    # 子类 override: 可能还要重算 DAG budget / 切换 summary model
    # ═══════════════════════════════════════════════════════════════

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Called when the user switches models or on fallback activation.

        Default updates context_length and recalculates threshold_tokens
        from threshold_percent. Override if your engine needs more
        (e.g. recalculate DAG budgets, switch summary models).
        """
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 11.1】update_model —— 模型切换
        # 触发: 用户切 model / fallback 激活
        # 输入: 新 model + 新 context_length + 可选 base_url/api_key/provider/api_mode
        # 默认实现: 只更新 context_length + 重算 threshold
        # 子类 override: 可能重算 DAG budget / 切换 summary model
        # ═══════════════════════════════════════════════════════════════
        # 11.1.1 更新 context_length
        self.context_length = context_length
        # 11.1.2 重算 threshold = context_length * 75%
        self.threshold_tokens = int(context_length * self.threshold_percent)
