"""The agent conversation loop — extracted from ``run_agent.AIAgent``.

This is the biggest single chunk pulled out of ``run_agent.py``: the
roughly 3,900-line :func:`run_conversation` body that drives one user
turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks, background memory/skill review nudges).

The function takes the parent ``AIAgent`` instance as its first
argument (``agent``) and accesses its state via attribute lookup.
``_ra().AIAgent.run_conversation`` is now a thin forwarder.

Symbols that production code or tests patch on ``run_agent`` directly
(``handle_function_call``, ``_set_interrupt``, ``OpenAI``, ...) are
resolved through :func:`_ra` so those patches keep working.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_context_length_from_provider_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import jittered_backoff
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from hermes_logging import set_session_context
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """Return a user-facing error when Ollama is loaded with too little context."""
    if not getattr(agent, "tools", None):
        return None

    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if not isinstance(runtime_ctx, int) or runtime_ctx <= 0:
        return None
    if runtime_ctx >= MINIMUM_CONTEXT_LENGTH:
        return None

    model = getattr(agent, "model", "") or "the selected model"
    base_url = getattr(agent, "base_url", "") or "unknown base URL"
    provider = getattr(agent, "provider", "") or "unknown"
    tool_count = len(getattr(agent, "tools", None) or [])

    logger.warning(
        "Ollama runtime context too small for Hermes tool use: "
        "model=%s provider=%s base_url=%s runtime_context=%d "
        "minimum_context=%d estimated_request_tokens=%d tool_count=%d "
        "session=%s",
        model,
        provider,
        base_url,
        runtime_ctx,
        MINIMUM_CONTEXT_LENGTH,
        request_tokens,
        tool_count,
        getattr(agent, "session_id", None) or "none",
    )

    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime "
        f"context, but Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens "
        "for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the "
        "model before trying again. A known-good starting point is 65,536 "
        "tokens. In Hermes config, set `model.ollama_num_ctx: 65536` "
        "(and `model.context_length: 65536` if you also override the displayed "
        "model context). If you manage the model through an Ollama Modelfile, "
        "set `PARAMETER num_ctx 65536` there instead."
    )


def _ra():
    """Lazy reference to ``run_agent`` so callers can patch
    ``run_agent.handle_function_call`` / ``run_agent._set_interrupt`` /
    ``run_agent.OpenAI`` and have those patches reach this code path.
    """
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        return message or ""
    except Exception:
        return ""


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    message = _nous_entitlement_message(capability)
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    provider = (provider or "").strip().lower()
    if provider == "nous":
        return True
    base = str(base_url or "")
    return (
        base_url_host_matches(base, "inference-api.nousresearch.com")
        or base_url_host_matches(base, "inference.nousresearch.com")
    )


def _billing_or_entitlement_message(
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> str:
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"
    lines = [
        (
            f"{provider_label} reported that billing, credits, or account "
            f"entitlement is exhausted for {model_label}."
        ),
        "Add credits or update billing with that provider, then retry.",
    ]
    if base_url_host_matches(str(base_url or ""), "openrouter.ai"):
        lines.append("OpenRouter credits: https://openrouter.ai/settings/credits")
    lines.append("You can switch providers temporarily with /model <model> --provider <provider>.")
    return "\n".join(lines)


def _print_billing_or_entitlement_guidance(
    agent,
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> bool:
    message = _billing_or_entitlement_message(
        capability=capability,
        provider=provider,
        base_url=base_url,
        model=model,
    )
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """Refresh Nous runtime credentials after a fresh paid-entitlement check."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        if account_info.paid_service_access is not True:
            return False
        return agent._try_refresh_nous_client_credentials(
            force=True,
        )
    except Exception:
        return False


def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """Restore the cached system prompt from the session DB or build it fresh.

    Mutates ``agent._cached_system_prompt`` and persists a freshly-built
    prompt back to the session DB on first build.  Extracted from
    ``run_conversation`` so the prefix-cache restore path can be tested in
    isolation.

    Three-way state distinction for the stored row, surfaced via logs so
    silent prefix-cache misses are visible in ``agent.log``:

      * ``missing`` — no session row yet (legitimate first turn).
      * ``null``   — row exists, ``system_prompt`` column is NULL.
        Legacy session predating system-prompt persistence, or a migration
        leftover.  Warns when ``conversation_history`` is non-empty.
      * ``empty``  — row exists, ``system_prompt`` column is the empty
        string.  Indicates a previous-turn write that ran but stored
        nothing (silent persistence bug).  Always warns.
      * ``present`` — row exists with a usable prompt → reused verbatim.

    Read or write failures against the session DB log at WARNING (not
    DEBUG) so persistent issues (disk full, schema drift, lock contention)
    surface without needing verbose mode.  This used to be a debug-level
    log that silently broke prefix-cache reuse on the gateway path
    (which constructs a fresh ``AIAgent`` per turn and depends on this
    DB roundtrip).
    """
    stored_prompt = None
    stored_state = "missing"
    if conversation_history and agent._session_db:
        try:
            session_row = agent._session_db.get_session(agent.session_id)
            if session_row is not None:
                raw_prompt = session_row.get("system_prompt")
                if raw_prompt is None:
                    stored_state = "null"
                elif raw_prompt == "":
                    stored_state = "empty"
                else:
                    stored_prompt = raw_prompt
                    stored_state = "present"
        except Exception as exc:
            logger.warning(
                "Session DB get_session failed for system-prompt restore "
                "(session=%s): %s. Falling back to fresh build — prefix "
                "cache will miss for this turn.",
                agent.session_id, exc,
            )

    if stored_prompt:
        # Continuing session — reuse the exact system prompt from the
        # previous turn so the Anthropic cache prefix matches.
        agent._cached_system_prompt = stored_prompt
        return

    if conversation_history and stored_state in ("null", "empty"):
        # Continuing session whose stored prompt is unusable.  The
        # previous turn's write either never happened or wrote an empty
        # string — either way every turn now rebuilds and the prefix
        # cache misses every time.
        logger.warning(
            "Stored system prompt for session %s is %s; rebuilding "
            "from scratch this turn. Prefix cache will miss until "
            "the rebuild persists. Investigate the previous turn's "
            "update_system_prompt write path.",
            agent.session_id, stored_state,
        )

    # First turn of a new session (or recovering from a broken stored
    # prompt) — build from scratch.
    agent._cached_system_prompt = agent._build_system_prompt(system_message)

    # Plugin hook: on_session_start — fired once when a brand-new
    # session is created (not on continuation).  Plugins can use this
    # to initialise session-scoped state (e.g. warm a memory cache).
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_start hook failed: %s", exc)

    # Persist the system prompt snapshot in SQLite.  Failure here used
    # to log at DEBUG, which silently broke prefix-cache reuse on the
    # gateway path (fresh AIAgent per turn → reads from this row every
    # subsequent turn).
    if agent._session_db:
        try:
            agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
        except Exception as exc:
            logger.warning(
                "Session DB update_system_prompt failed for session %s: "
                "%s. Subsequent turns will rebuild the system prompt and "
                "miss the prefix cache.",
                agent.session_id, exc,
            )


def _get_continuation_prompt(is_partial_stub: bool, dropped_tools: Optional[List[str]] = None) -> str:
    if is_partial_stub and dropped_tools:
        tool_list = ", ".join(dropped_tools[:3])
        return (
            "[System: Your previous tool call "
            f"({tool_list}) was too large and "
            "the stream timed out before it "
            "could be delivered. Do NOT retry "
            "the same tool call with the same "
            "large content. Instead, break the "
            "content into multiple smaller tool "
            "calls (e.g. use multiple patch calls "
            "or write smaller files). Each tool "
            "call's arguments must be under ~8K "
            "tokens to avoid stream timeouts.]"
        )
    elif is_partial_stub:
        return (
            "[System: The previous response was cut off by a "
            "network error mid-stream. Continue exactly where "
            "you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]"
        )
    else:
        return (
            "[System: Your previous response was truncated by the output "
            "length limit. Continue exactly where you left off. Do not "
            "restart or repeat prior text. Finish the answer directly.]"
        )


def run_conversation(
    agent,
    user_message: str,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行完整的对话循环，包含工具调用，直到完成。

    这是 Hermes Agent 的核心主循环。用户输入 → 构建消息 → 调用模型 →
    执行工具 → 循环直到无工具调用 → 返回最终响应。

    Args:
        user_message (str): 用户消息/问题
        system_message (str): 自定义 system message（可选）
        conversation_history (List[Dict]): 之前的对话历史（可选）
        task_id (str): 任务唯一标识符，用于隔离并发任务的 VM（可选，自动生成）
        stream_callback: 流式回调，每个 text delta 都会调用。
            用于 TTS 管道在完整响应前开始音频生成。
            为 None 时，使用标准非流式调用。
        persist_user_message: 可选的干净用户消息，用于存储到 transcripts/history。
            当 user_message 包含 API 专用的合成前缀时使用。

    Returns:
        Dict: 完整对话结果，包含最终响应和消息历史
    """
    # 保护 stdio 免受 OSError（管道断裂）影响。
    # 在 systemd/headless/daemon 环境下防止 write 崩溃。
    # 只安装一次，流正常时透明，不影响健康场景。
    _install_safe_stdio()

    # 确保数据库会话已创建（首次使用时创建 session DB 行）
    agent._ensure_db_session()

    # 通知 auxiliary_client 当前的主 provider/model。
    #
    # 用途：某些工具的行为依赖当前模型（如 vision_analyze 的快速路径），
    # 需要看到 CLI/gateway 的覆盖配置，而非 config.yaml 的陈旧默认值。
    # 幂等操作，每轮调用都安全。
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    except Exception:
        pass

    # 给当前线程的所有日志打上 session ID 标签。
    # 这样 `hermes logs --session <id>` 可以过滤出单个对话的所有日志。
    set_session_context(agent.session_id)

    # 绑定 skill 写入来源的 ContextVar。
    #
    # 用途：让工具处理器（如 skill_manage create）能够区分：
    #   - 在后台 agent-improvement review fork 中运行
    #   - 在前台用户-directed turn 中运行
    #
    # 设置在每次调用顶部；review fork 在自己的线程上运行，
    # 所以前台的值不会泄漏到其中。
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # 如果上一个 turn 激活了 fallback，恢复主 runtime。
    #
    # 场景：某个 turn 触发降级（provider 不可用等），下一个 turn 应该
    # 用回首选模型重试。此调用在 _fallback_activated 为 False 时是 no-op。
    agent._restore_primary_runtime()

    # 清理用户输入中的 surrogate 字符。
    #
    # 问题来源：从富文本编辑器（Google Docs、Word 等）粘贴时，
    # 可能注入不合法的 UTF-8 lone surrogates，导致 OpenAI SDK 的 JSON 序列化崩溃。
    if isinstance(user_message, str):
        user_message = _sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = _sanitize_surrogates(persist_user_message)

    # 保存流式回调，供 _interruptible_api_call 使用
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message

    # 生成唯一的 task_id，用于隔离并发任务之间的 VM
    effective_task_id = task_id or str(uuid.uuid4())

    # 将 task_id 暴露给中途运行的工具（如 delegate_tool）。
    # 在任何工具分发之前设置，确保子代理启动时看到的是真实的父 task_id，而非 None。
    agent._current_task_id = effective_task_id

    # 在每个 turn 开头重置重试计数器和迭代预算。
    # 确保上一个 turn 的子代理使用量不会占用下一个 turn 的预算。
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None

    # Vision 支持标志：当服务器拒绝 image_url 内容时设为 False。
    # 防止向只支持 text 的端点重复发送图片。
    agent._vision_supported = True

    # Pre-turn 连接健康检查：检测并清理死掉的 TCP 连接。
    # 防止下一个 API 调用挂在僵尸 socket 上。
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 检测到之前 provider 问题遗留的死连接 — 已自动清理。继续使用新连接。"
                )
        except Exception:
            pass

    # 通过 status_callback 重放压缩警告（gateway 平台）。
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # 只发送一次

    # 注意：_turns_since_memory 和 _iters_since_skill 不在这里重置。
    # 它们在 __init__ 初始化，必须跨 run_conversation 调用持久化，
    # 这样 nudge 逻辑才能在 CLI 模式下正确累积。
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # 记录对话 turn 启动日志（用于调试和可观测性）
    _preview_text = _summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # 初始化 messages 列表（复制，避免修改调用者的列表）
    messages = list(conversation_history) if conversation_history else []

    # 从对话历史中恢复 todo store。
    # Gateway 每个消息创建一个新 AIAgent，内存中的 store 是空的，
    # 需要从历史中最新的 todo tool 响应中恢复状态。
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # 从持久化的历史中恢复 per-session nudge 计数器。
    # Gateway 每个消息创建新 AIAgent，所以 _turns_since_memory 和 _user_turn_count
    # 每个 turn 都从 0 开始。需要从 conversation_history 中重建有效计数。
    # 幂等操作：已累积计数器的缓存 agent 保持不变。
    # See issue #22357.
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                # % 保持原始的 1-N 周期，避免 resume 时立即触发（会让用户惊讶）。
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval


    # Prefill 消息（few-shot priming）只在 API 调用时注入，不存储在 messages 列表中。
    # 这使它们是临时的：不会保存到 session DB、session logs 或 batch trajectories，
    # 但会在每次 API 调用时自动重新应用。
    
    # 追踪用户 turn 数，用于 memory flush 和周期性 nudge 逻辑
    agent._user_turn_count += 1

    # 重置流式上下文清理器（StreamingContextScrubber）
    #
    # 背景：模型流式输出时，<memory-context> 标签可能拆分到多个 chunk：
    #   chunk1: "以下是记忆 <memory"
    #   chunk2: "-context> 重要信息...</memory-context>"
    #
    # 问题：正则表达式需要同一字符串中有起止标签才能匹配，
    #       流式输出的 chunk 边界会破坏匹配，导致标签内容泄露到 UI。
    #
    # 解决：StreamingContextScrubber 是状态机，逐 chunk 处理：
    #   - 遇到开标签 → 进入 span 模式，丢弃内容
    #   - 遇到闭标签 → 退出 span 模式
    #   - 每个新 turn 开始时 reset()，防止上一个 turn 的残留状态污染
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()

    # 重置 think 清理器（StreamingThinkScrubber）
    #
    # 原因同上：如果上一个流被中断，可能停留在未终止的 <thinking> 块内，
    # 新 turn 需要重置状态。
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # 保存原始用户消息（不注入 nudge）。
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # 追踪 memory nudge 触发条件（基于 turn）。
    # Skill 触发器在 agent 循环完成后检查，基于本 turn 使用的工具迭代次数。
    _should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            _should_review_memory = True
            agent._turns_since_memory = 0

    # 添加用户消息
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx
    
    if not agent.quiet_mode:
        _print_preview = _summarize_user_message_for_log(user_message)
        agent._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")
    
    # ── System prompt（每个 session 缓存，用于 prefix caching）────
    # 首次调用时构建，之后所有调用复用。
    # 只在上下文压缩事件后重建（这会使缓存失效并从磁盘重新加载 memory）。
    #
    # 对于继续的 session（gateway 每个消息创建新 AIAgent），
    # 从 session DB 加载存储的 system prompt，而不是重建。
    # 重建会拾取磁盘上的 memory 变化（模型已经知道，因为它写的！），
    # 产生不同的 system prompt 并破坏 Anthropic prefix cache。
    if agent._cached_system_prompt is None:
        _restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # ── Preflight 上下文压缩 ──
    # 在进入主循环之前，检查加载的对话历史是否已超过模型的上下文阈值。
    # 处理用户切换到更小上下文窗口的模型时仍有大 session 的情况——
    # 主动压缩，而不是等待 API 错误（可能被捕获为不可重试的 4xx 并完全中止请求）。
    if (
        agent.compression_enabled
        and len(messages) > agent.context_compressor.protect_first_n
                            + agent.context_compressor.protect_last_n + 1
    ):
        # 包含工具 schema tokens——使用大量工具时，这些可能增加 20-30K+ tokens，
        # 旧的 sys+msg 估计完全忽略了这些。
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)

        if not _preflight_deferred:
            # 保持 CLI/ACP 上下文显示与 preflight 实际测量同步。
            # 状态栏读取 compressor.last_prompt_tokens，否则只在*成功的*API 响应后更新。
            # 当对话自上次成功调用后增长了——或压缩失败了（如辅助摘要模型超时）——
            # 状态栏卡在旧的较小值，而 preflight 报告更大数字，看起来不同步。
            # 用新的估计值播种（只向上修订；真正的 update_from_response 会在下次 API 调用后纠正）。
            # 延迟时跳过——延迟的估计已知会过度计数，
            # 所以信任它会重新引入我们正在避免的不同步。
            if _preflight_tokens > (_compressor.last_prompt_tokens or 0):
                _compressor.last_prompt_tokens = _preflight_tokens

        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compressor.should_compress(_preflight_tokens):
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            # 对于很大的 session 和很小的上下文窗口，可能需要多次压缩传递
            # （每次传递总结中间 N 条消息）。
            for _pass in range(3):
                _orig_len = len(messages)
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if len(messages) >= _orig_len:
                    break  # Cannot compress further
                # 压缩创建了新 session——清除历史引用，
                # 这样 _flush_messages_to_session_db 会写入所有压缩消息到新 session 的 SQLite，
                # 而不是因为 conversation_history 仍是压缩前的长度而跳过它们。
                conversation_history = None
                # 修复：压缩后重置重试计数器，让模型在压缩后的上下文上获得新预算。
                # 没有这个，上一次压缩前的重试会延续，
                # 模型在压缩引起的上下文丢失后立即遇到"(empty)"。
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                # Re-estimate after compression
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if not _compressor.should_compress(_preflight_tokens):
                    break  # Under threshold or anti-thrash guard stopped it

    # 插件钩子：pre_llm_call
    # 在工具调用循环之前每个 turn 触发一次。插件可以返回包含 context key 的字典
    # （或纯字符串），其值会追加到当前 turn 的用户消息。
    #
    # Context 总是注入到用户消息，绝不是 system prompt。
    # 这保持了 prompt 缓存前缀——system prompt 在各 turn 之间保持相同，
    # 所以缓存的 tokens 可以重用。system prompt 是 Hermes 的领地；
    # 插件在用户输入旁边贡献 context。
    #
    # 所有注入的 context 都是临时的（不持久化到 session DB）。
    _plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)
        if _ctx_parts:
            _plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # 主对话循环
    api_call_count = 0
    final_response = None
    interrupted = False
    failed = False
    codex_ack_continuations = 0
    length_continue_retries = 0
    truncated_tool_call_retries = 0
    truncated_response_parts: List[str] = []
    compression_attempts = 0
    _turn_exit_reason = "unknown"  # 诊断：循环为什么结束

    # 每个 turn 的文件变更验证器状态。按解析后的路径索引；
    # 每次失败的 write_file/patch 调用记录错误预览。
    # 之后对同一路径的成功写入会移除条目（模型恢复了）。
    # Turn 结束时，仍存在的条目会在 advisory footer 中显示，
    # 这样模型不能在实际文件未改变的情况下过度声称成功。
    agent._turn_failed_file_mutations: Dict[str, Dict[str, Any]] = {}
    
    # 记录执行线程，以便 interrupt()/clear_interrupt() 可以
    # 将工具级中断信号限定到当前 agent 线程。
    # 必须在任何线程级中断同步之前设置。
    agent._execution_thread_id = threading.current_thread().ident

    # 始终清除前一个 turn 的过时线程状态。
    # 如果中断在启动完成前到达，保留它并绑定到此执行线程，而不是丢弃。
    _ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        _ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # 通知内存提供者新 turn 开始了，以便节拍追踪工作。
    # 必须在 prefetch_all() 之前发生，这样提供者知道这是第几个 turn，
    # 并可以通过 contextCadence/dialecticCadence 控制 context/dialectic 刷新。
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # 外部内存提供者：在工具循环前 prefetch 一次。
    # 每次迭代复用缓存结果，避免每次工具调用都调用 prefetch_all()
    # （10次工具调用 = 10倍延迟 + 成本）。
    # 使用 original_user_message（干净的输入）—— user_message 可能包含
    # 注入的 skill 内容，会膨胀或破坏 provider 查询。
    _ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            _ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    # 可选的 opt-in runtime：如果 api_mode == codex_app_server，
    # 将 turn 交给 codex app-server 子进程
    #（terminal/file ops/patching 都在 Codex 内部运行）。
    # 默认 Hermes 路径完全绕过。
    # See agent/transports/codex_app_server_session.py for the adapter
    # and references/codex-app-server-runtime.md for the rationale.
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=_should_review_memory,
        )

    while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
        # 主循环：条件是 (api_call_count < max_iterations AND iteration_budget.remaining > 0)
        # 或者 _budget_grace_call（预算耗尽后给模型最后一次机会）
        # 重置每个 turn 的检查点去重，以便每次迭代可以拍摄一个快照
        agent._checkpoint_mgr.new_turn()

        # 检查中断请求（如用户发送了新消息）
        if agent._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not agent.quiet_mode:
                agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break
        
        api_call_count += 1
        agent._api_call_count = api_call_count
        agent._touch_activity(f"starting API call #{api_call_count}")

        # Grace call：预算耗尽了，但我们给模型多一次机会。
        # 消费 grace 标志，以便无论结果如何，循环都会在此次迭代后退出。
        if agent._budget_grace_call:
            agent._budget_grace_call = False
        elif not agent.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not agent.quiet_mode:
                agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)")
            break

        # 为 gateway hooks 触发 step_callback（agent:step 事件）
        if agent.step_callback is not None:
            try:
                prev_tools = []
                for _idx, _m in enumerate(reversed(messages)):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        _fwd_start = len(messages) - _idx
                        _results_by_id = {}
                        for _tm in messages[_fwd_start:]:
                            if _tm.get("role") != "tool":
                                break
                            _tcid = _tm.get("tool_call_id")
                            if _tcid:
                                _results_by_id[_tcid] = _tm.get("content", "")
                        prev_tools = [
                            {
                                "name": tc["function"]["name"],
                                "result": _results_by_id.get(tc.get("id")),
                                "arguments": tc["function"].get("arguments"),
                            }
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                agent.step_callback(api_call_count, prev_tools)
            except Exception as _step_err:
                logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

        # 追踪工具调用迭代次数，用于 skill nudge。
        # 计数器在 skill_manage 实际使用时重置。
        if (agent._skill_nudge_interval > 0
                and "skill_manage" in agent.valid_tool_names):
            agent._iters_since_skill += 1
        
        # ── Pre-API-call /steer drain ──────────────────────────────────
        # If a /steer arrived during the previous API call (while the model
        # was thinking), drain it now — before we build api_messages — so
        # the model sees the steer text on THIS iteration.  Without this,
        # steers sent during an API call only land after the NEXT tool batch,
        # which may never come if the model returns a final response.
        #
        # 从后向前扫描 messages 列表中最后一条 tool-role 消息。
        # 如果找到，将 steer 追加到那里。如果没有（第一次迭代，还没有工具），
        # steer 保持待处理状态，等待下一个工具批次——
        # 注入到 user 消息会破坏 role 交替，而且没有工具输出可以搭便车。
        _pre_api_steer = agent._drain_pending_steer()
        if _pre_api_steer:
            _injected = False
            for _si in range(len(messages) - 1, -1, -1):
                _sm = messages[_si]
                if isinstance(_sm, dict) and _sm.get("role") == "tool":
                    marker = f"\n\nUser guidance: {_pre_api_steer}"
                    existing = _sm.get("content", "")
                    if isinstance(existing, str):
                        _sm["content"] = existing + marker
                    else:
                        # Multimodal content blocks — append text block
                        try:
                            blocks = list(existing) if existing else []
                            blocks.append({"type": "text", "text": marker})
                            _sm["content"] = blocks
                        except Exception:
                            pass
                    _injected = True
                    logger.debug(
                        "Pre-API-call steer drain: injected into tool msg at index %d",
                        _si,
                    )
                    break
            if not _injected:
                # 没有工具消息可以注入——放回去，以便 post-tool-execution drain 稍后获取。
                _lock = getattr(agent, "_pending_steer_lock", None)
                if _lock is not None:
                    with _lock:
                        if agent._pending_steer:
                            agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                        else:
                            agent._pending_steer = _pre_api_steer
                else:
                    existing = getattr(agent, "_pending_steer", None)
                    agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

        # 格式化修复tool calls的参数，防止它们在 API 调用时被 provider 拒绝。
        # 准备 API 调用消息
        # 如果有临时 system prompt，将其添加到消息前面
        # 注意：推理通过 <think> 标签嵌入 content 中以存储轨迹。
        # 但像 Moonshot AI 这样的 provider 需要 assistant 消息上
        # 单独的 'reasoning_content' 字段（带 tool_calls）。这里处理这两种情况。
        request_logger = getattr(agent, "logger", None) or logging.getLogger(__name__)
        repaired_tool_calls = agent._sanitize_tool_call_arguments(
            messages,
            logger=request_logger,
            session_id=agent.session_id,
        )
        if repaired_tool_calls > 0:
            request_logger.info(
                "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                repaired_tool_calls,
                agent.session_id or "-",
            )

        # 防御性：在 API 调用前修复格式错误的 role 交替。
        # 捕获 history 陷入 ``tool → user`` 或 ``user → user`` 尾部的情况
        #（如空响应脚手架被剥离后，孤儿工具结果后到达新用户消息）。
        # 大多数 provider 在格式错误的序列上返回空内容，
        # 否则会无限重新触发空重试循环。
        repaired_seq = agent._repair_message_sequence(messages)
        if repaired_seq > 0:
            request_logger.info(
                "Repaired %s message-alternation violations before request (session=%s)",
                repaired_seq,
                agent.session_id or "-",
            )

        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # 将临时 context 注入当前 turn 的用户消息。
            # 来源：memory manager prefetch + plugin pre_llm_call hooks
            # 目标="user_message"（默认）。两者都是
            # API 调用时专用——messages 中的原始消息从不变异，所以不会泄漏到 session 持久化。
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # 对于所有 assistant 消息，将 reasoning 传回 API
            # 这确保多轮推理上下文被保留
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # 移除 'reasoning' 字段——它只用于轨迹存储
            # 我们已经将它复制到上面的 'reasoning_content' 供 API 使用
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # 移除 finish_reason——严格的 API 不接受（如 Mistral）
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # 剥离内部 thinking-prefill 标记
            api_msg.pop("_thinking_prefill", None)
            # 为严格的 provider（如 Mistral、Fireworks 等）剥离 Codex Responses API 字段
            # 它们拒绝未知字段。使用新字典，以便内部 messages 列表保留字段
            # 以保持 Codex Responses 兼容性。
            if agent._should_sanitize_tool_calls():
                agent._sanitize_tool_calls_for_strict_api(api_msg)
            # 保留 'reasoning_details'——OpenRouter 用它进行多轮推理上下文
            # signature 字段帮助维持推理连续性
            api_messages.append(api_msg)

        # 构建最终 system message：缓存的 prompt + 临时 system prompt。
        # 临时 additions 是 API 调用时专用的（不持久化到 session DB）。
        # 外部 recall context 注入到用户消息，而非 system prompt，
        # 所以稳定的缓存前缀保持不变。
        #
        # 注意：plugin pre_llm_call hooks 的 context 注入到用户消息
        #（见上面的注入块），不是 system prompt。
        # 这是故意的——system prompt 修改会破坏 prompt 缓存前缀。
        # system prompt 保留给 Hermes 内部使用。
        #
        # Hermes 不变量：system prompt 每个 session 构建一次
        #（缓存在 _cached_system_prompt 上），每个 turn 逐字重放。
        # 我们将它作为单个 content 字符串发送，所以字节在 turn 之间是字节稳定的，
        # 上游 prompt 缓存保持热。
        effective_system = active_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # 在 system prompt 之后、对话历史之前注入临时 prefill 消息。
        # 相同的 API 调用时专用模式。
        if agent.prefill_messages:
            sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # 为原生 Anthropic、OpenRouter 和第三方 Anthropic 兼容
        # gateway 上的 Claude 模型应用 Anthropic prompt caching。
        # 自动检测：如果 _use_prompt_caching 设置了，
        # 注入 cache_control 断点（system + 最后3条消息），
        # 在多轮对话中将输入 token 成本降低约 75%。
        if agent._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(
                api_messages,
                cache_ttl=agent._cache_ttl,
                native_anthropic=agent._use_native_cache_layout,
            )

        # Safety net: strip orphaned tool results / add stubs for missing
        # results before sending to the API.  Runs unconditionally — not
        # gated on context_compressor — so orphans from session loading or
        # manual message manipulation are always caught.
        api_messages = agent._sanitize_api_messages(api_messages)

        # 删除仅 thinking 的 assistant turns（推理但无可见输出和 tool_calls），
        # 并合并遗留的相邻用户消息。
        # 防止 Anthropic 400s（"assistant 消息中的最后一个块不能是 `thinking`"）
        # 和第三方 Anthropic 兼容 gateway 的等效错误。
        # 只在每次调用副本上运行——存储的对话历史保留 reasoning 块
        # 用于 UI 记录和 session 持久化。
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        # 规范化消息空白和 tool-call JSON 以获得一致的
        # 前缀匹配。确保跨 turn 的位精确前缀，
        # 这使得本地推理服务器（llama.cpp、vLLM、Ollama）上的 KV 缓存重用成为可能，
        # 并提高云 provider 的缓存命中率。
        # 在 api_messages（API 副本）上操作，所以 messages 中的
        # 原始对话历史不受影响。
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        tc["function"]["arguments"] = _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # 在 API 调用前主动剥离任何 surrogate 字符。
        # 通过 Ollama 服务的模型（Kimi K2.5、GLM-5、Qwen）可能返回
        # 导致 OpenAI SDK 内部 json.dumps() 崩溃的 lone surrogates (U+D800-U+DFFF)。
        # 在这里清理可防止 3 次重试循环。
        _sanitize_messages_surrogates(api_messages)

        # 计算近似的请求大小用于日志
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        approx_request_tokens = estimate_request_tokens_rough(
            api_messages, tools=agent.tools or None
        )

        _runtime_context_error = _ollama_context_limit_error(
            agent, approx_request_tokens
        )
        if _runtime_context_error:
            final_response = _runtime_context_error
            failed = True
            _turn_exit_reason = "ollama_runtime_context_too_small"
            messages.append({"role": "assistant", "content": final_response})
            agent._emit_status("❌ Ollama runtime context is too small for Hermes tool use")
            api_call_count -= 1
            agent._api_call_count = api_call_count
            try:
                agent.iteration_budget.refund()
            except Exception:
                pass
            break
        
        # 安静模式的 thinking spinner（API 调用期间动画）
        thinking_spinner = None
        
        if not agent.quiet_mode:
            agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
            agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
        else:
            # Animated thinking spinner in quiet mode
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if agent.thinking_callback:
                # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                # (works in both streaming and non-streaming modes)
                agent.thinking_callback(f"{face} {verb}...")
            elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
                # Raw KawaiiSpinner only when no streaming consumers and the
                # spinner output has a safe sink.
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
                thinking_spinner.start()
        
        # 如果 verbose，记录请求详情
        if agent.verbose_logging:
            logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 1】初始化内层重试循环的状态变量
        # ═══════════════════════════════════════════════════════════════
        # 1.1 记录 API 调用的起始时间戳
        #     → 后面用来计算 api_duration(给 verbose 日志和 metrics 用)
        api_start_time = time.time()
        # 1.2 初始化内层重试计数器为 0(每次外层 tool-call 迭代都会重置)
        retry_count = 0
        # 1.3 从 agent 配置里拿 max_retries 上限(默认通常是 5)
        max_retries = agent._api_max_retries
        # ═══════════════════════════════════════════════════════════════
        # 【学习要点】以下是 14 个 *_attempted 一次性布尔标志
        # 设计模式：Finite State Machine(有限状态机)
        # 每个标志代表一种"只重试一次的特殊恢复路径"
        # 为什么用标志而不是计数器？—— 因为每种错误恢复都是幂等性操作
        # (refresh token / reformat payload / shrink image),重复执行反而有害
        # 比如 refresh_token 已经成功了,再调一次可能让用户被踢出登录
        # 标志位确保每种恢复路径"在主循环期间最多走一次"
        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 2】声明 14 个一次性状态标志(每个代表一种只走一次的恢复路径)
        # 设计原则: 标志位而非计数器,因为每种恢复都是幂等性操作
        # 重复执行反而有害(如重复 refresh_token 会让用户被踢出登录)
        # ═══════════════════════════════════════════════════════════════
        primary_recovery_attempted = False          # 主 provider 失败后是否已切到 fallback
        max_compression_attempts = 3                # 压缩重试上限(防无限压缩循环)
        codex_auth_retry_attempted=False            # Codex 401 → 尝试重新拿 token
        anthropic_auth_retry_attempted=False        # Anthropic 401 → 重新拿 token
        nous_auth_retry_attempted=False             # Nous 401 → 重新拿 token
        nous_paid_entitlement_refresh_attempted=False  # Nous 付费额度过期 → 主动刷新
        copilot_auth_retry_attempted=False          # GitHub Copilot 401 → 重新拿 token
        thinking_sig_retry_attempted = False        # Anthropic 推理签名错 → 剥 thinking 重试
        invalid_encrypted_content_retry_attempted = False  # 加密 content 字段不合法 → 剥密重试
        image_shrink_retry_attempted = False        # 图片超 provider 限制 → 缩小重试
        multimodal_tool_content_retry_attempted = False  # 工具返回多模态内容炸 SDK → 文本化重试
        oauth_1m_beta_retry_attempted = False       # 1M context beta header 错 → 关掉重试
        llama_cpp_grammar_retry_attempted = False   # llama.cpp grammar 错 → 关闭 grammar 重试
        has_retried_429 = False                     # 429 速率限制 → 退避重试(可能跟其他标志并存)
        restart_with_compressed_messages = False    # 压缩后需要"重启"while 循环而不是 continue
        restart_with_length_continuation = False    # finish_reason=length 时需要"续写"而不是重试

        # ═══════════════════════════════════════════════════════════════
        # 【步骤 3】初始化跨循环共享变量
        # ═══════════════════════════════════════════════════════════════
        # 3.1 默认 finish_reason 为 "stop",如果所有重试都失败也不会触发 None
        finish_reason = "stop"
        # 3.2 response 默认为 None,防止所有重试失败后 UnboundLocalError
        response = None
        # 3.3 api_kwargs 默认为 None,except handler 里要引用它
        api_kwargs = None

        # ═══════════════════════════════════════════════════════════════
        # 【学习要点】while retry_count < max_retries 是「内层重试循环」
        # 嵌套在外层「主对话循环」里面
        # 外层 while(api_call_count < max_iterations) 跑 tool-call 迭代
        # 内层 while(retry_count < max_retries) 跑 API 错误恢复
        # 关系:1 次外层迭代可能触发 30+ 次内层重试
        # 设计原因:tool_call 迭代是"业务层",API 重试是"传输层"
        # 分开两层循环,职责清晰,也方便在重试时保留 messages 状态
        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4】进入内层 API 重试循环(while retry_count < max_retries)
        # 每次进入此循环 = 1 次 API 调用尝试
        # 循环退出条件: retry_count >= max_retries(被 fallback 截走则重置)
        # ═══════════════════════════════════════════════════════════════
        while retry_count < max_retries:
            # API 调用重试循环
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.1】Nous Portal 跨 session 速率限制守卫
            # 设计: 速率限制是「进程级 / 跨 session」共享的
            # 你这个 session 没被限流,但其他 session 被限流了 → 你也跳过
            # 原因: Nous 按 RPH 计费,限流期每个请求都会让限流时间延长
            # ═══════════════════════════════════════════════════════════════
            # ── Nous Portal 速率限制守卫 ──────────────────────
            # 如果另一个 session 已经记录了 Nous 被速率限制，
            # 完全跳过 API 调用。每次尝试（包括 SDK 级重试）
            # 都会计入 RPH，加深速率限制孔。
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.1.1】仅当 provider 是 "nous" 时才做这个检查
            # 其他 provider 没跨 session 限流共享,直接跳过
            # ═══════════════════════════════════════════════════════════════
            if agent.provider == "nous":
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.1.2】动态 import nous_rate_guard 模块
                # 用 try/except 包裹是因为这个模块可能不存在(没装 Nous 用户)
                # ImportError 时静默跳过,继续正常 API 调用
                # ═══════════════════════════════════════════════════════════════
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    # 4.1.3 调用 nous_rate_limit_remaining() 获取剩余限流时间(秒)
                    #      返回 None 表示没有限流记录
                    _nous_remaining = nous_rate_limit_remaining()
                    # 4.1.4 如果有限流记录(>0 秒剩余)
                    if _nous_remaining is not None and _nous_remaining > 0:
                        # 4.1.5 构造给用户看的限流提示信息
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        # 4.1.6 把提示 buffer 到 verbose 输出(等会一起打印)
                        agent._buffer_vprint(
                            f"⏳ {_nous_msg} Trying fallback..."
                        )
                        # 4.1.7 把状态推给 UI(给 spinner 或 status bar)
                        agent._buffer_status(f"⏳ {_nous_msg}")
                        # 4.1.8 尝试切换到 fallback provider
                        #      成功 → 重置 retry_count=0,compression_attempts=0
                        #             让新 provider 重新跑所有重试
                        #      失败 → 进入下面的 return 分支
                        if agent._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        # No fallback available — surface buffered context
                        # so user sees the rate-limit message that led here.
                        # 4.1.9 没有 fallback 可用 → 刷出 buffer 给用户看
                        agent._flush_status_buffer()
                        # 4.1.10 先 persist session(即便失败也要保存)
                        agent._persist_session(messages, conversation_history)
                        # 4.1.11 返回失败结果给上层
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                # 4.1.12 限流模块没装 → 静默跳过(不影响正常流程)
                except ImportError:
                    pass
                # 4.1.13 其他异常也吞掉,绝不让限流守卫破坏 agent loop
                except Exception:
                    pass  # Never let rate guard break the agent loop

            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.2】开始 try 块 — 实际 API 调用包装在 try/except 里
            # 所有 retry logic 都在这个 try 块的 except 分支里
            # ═══════════════════════════════════════════════════════════════
            try:
                # 4.2.1 重置流式 token 跟踪器(每次重试都要清零)
                agent._reset_stream_delivery_tracking()
                # 4.2.2 为当前 provider 重新应用 reasoning echo-back
                #      (当 fallback 切换到需要 reasoning_content 的 provider 时必要)
                # api_messages is built once, before this retry loop, while the
                # primary provider is active.  A mid-conversation fallback can
                # switch to a require-side provider (DeepSeek / Kimi / MiMo) that
                # rejects assistant turns lacking reasoning_content.  Re-apply the
                # echo-back pad for the *current* provider here (idempotent no-op
                # unless the active provider needs it) so the fallback request
                # isn't sent with stale, primary-shaped reasoning fields.
                agent._reapply_reasoning_echo_for_provider(api_messages)
                # 4.2.3 把 api_messages 包装成 provider 需要的 api_kwargs 字典
                #      (这个函数内部会处理 tools / temperature / model 等参数)
                api_kwargs = agent._build_api_kwargs(api_messages)
                # 4.2.4 如果 agent 配置要求强制 ASCII,把非 ASCII 字符转义
                #      (极少见,某些严格 provider 需要)
                if agent._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                # 4.2.5 Codex Responses API 需要额外的 preflight 调整
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.3】触发 pre_api_request 插件钩子(可观察性/审计用)
                # 注意: 这跟前面的 pre_llm_call 是不同的钩子
                #   pre_llm_call: 注入 context 到用户消息
                #   pre_api_request: 纯观察,不影响请求内容(用于 logging/tracing)
                # ═══════════════════════════════════════════════════════════════
                try:
                    # 4.3.1 动态 import plugin 系统
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    # 4.3.2 从 api_kwargs 里拿消息数组(OpenAI 用 "messages", Codex 用 "input")
                    request_messages = api_kwargs.get("messages")
                    if not isinstance(request_messages, list):
                        request_messages = api_kwargs.get("input")
                    # 4.3.3 如果都没拿到,直接用原始的 api_messages
                    if not isinstance(request_messages, list):
                        request_messages = api_messages
                    # 4.3.4 触发钩子,传所有元数据(15 个参数)
                    #      list(request_messages) 是浅拷贝(只复制外层 list)
                    #      内部 dict 不变(agent loop 不会改它们)
                    #      用浅拷贝避免插件做异步快照时看到后来的 mutation
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        session_id=agent.session_id or "",
                        user_message=original_user_message,
                        conversation_history=list(messages),
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        request_messages=list(request_messages) if isinstance(request_messages, list) else [],
                        message_count=len(api_messages),
                        tool_count=len(agent.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=agent.max_tokens,
                    )
                # 4.3.5 任何插件异常都吞掉,绝不能让插件破坏主循环
                except Exception:
                    pass

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.4】(可选) dump 请求用于调试
                # HERMES_DUMP_REQUESTS=1 → 写 api_kwargs 到磁盘给开发者看
                # ═══════════════════════════════════════════════════════════════
                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.5】决策:本次调用用流式还是非流式
                # 原则: 总是优先流式(健康检查),特殊情况下才用非流式
                # Always prefer the streaming path — even without stream
                # consumers.  Streaming gives us fine-grained health
                # checking (90s stale-stream detection, 60s read timeout)
                # that the non-streaming path lacks.  Without this,
                # subagents and other quiet-mode callers can hang
                # indefinitely when the provider keeps the connection
                # alive with SSE pings but never delivers a response.
                # The streaming path is a no-op for callbacks when no
                # consumers are registered, and falls back to non-
                # streaming automatically if the provider doesn't
                # support it.
                # ═══════════════════════════════════════════════════════════════
                # 4.5.1 定义 _stop_spinner 闭包 — 流式收到第一块内容时调用
                #      用来关闭 thinking spinner(开始有输出了)
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                # 4.5.2 默认 _use_streaming = True(优先流式)
                _use_streaming = True
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.5.3】特殊情况 1: 之前尝试过,provider 明确不支持流式
                # → 标记 _disable_streaming=True,后续都走非流式
                #   Provider signaled "stream not supported" on a previous
                #   attempt — switch to non-streaming for the rest of this
                #   session instead of re-failing every retry.
                # ═══════════════════════════════════════════════════════════════
                if getattr(agent, "_disable_streaming", False):
                    _use_streaming = False
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.5.4】特殊情况 2: CopilotACP 用 subprocess stdio 通信
                # → 返回的是 SimpleNamespace,不是可迭代的 stream
                #   CopilotACPClient communicates via subprocess stdio and
                #   returns a plain SimpleNamespace — not an iterable
                #   stream.  Mirror the ACP exclusion used for Responses
                #   API upgrade (lines ~1083-1085).
                # ═══════════════════════════════════════════════════════════════
                elif (
                    agent.provider == "copilot-acp"
                    or str(agent.base_url or "").lower().startswith("acp://copilot")
                    or str(agent.base_url or "").lower().startswith("acp+tcp://")
                ):
                    _use_streaming = False
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.5.5】特殊情况 3: 没消费者,且是 Mock 测试客户端
                #   No display/TTS consumer. Still prefer streaming for
                #   health checking, but skip for Mock clients in tests
                #   (mocks return SimpleNamespace, not stream iterators).
                # ═══════════════════════════════════════════════════════════════
                elif not agent._has_stream_consumers():
                    from unittest.mock import Mock
                    if isinstance(getattr(agent, "client", None), Mock):
                        _use_streaming = False

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.6】真正发起 API 调用(流式 vs 非流式)
                # _interruptible_* 前缀含义: 内部用 thread+标志实现可中断
                # 用户在 API 等待时发新消息会立即返回,不会卡死
                # ═══════════════════════════════════════════════════════════════
                # 4.6.1 流式分支: 带 on_first_delta 回调(收到第一块时关 spinner)
                if _use_streaming:
                    response = agent._interruptible_streaming_api_call(
                        api_kwargs, on_first_delta=_stop_spinner
                    )
                # 4.6.2 非流式分支: 一次性拿完整响应
                else:
                    response = agent._interruptible_api_call(api_kwargs)

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.7】API 调用成功,做收尾
                # ═══════════════════════════════════════════════════════════════
                # 4.7.1 计算 API 调用耗时(秒)
                api_duration = time.time() - api_start_time

                # 4.7.2 静默停止 thinking spinner(response box 或 tool 输出更信息密集)
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # 4.7.3 非安静模式:打印 API 调用耗时
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

                # 4.7.4 Verbose 模式:记录响应模型和 usage 到 debug log
                if agent.verbose_logging:
                    # Log response with provider info if available
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                
                # ═══════════════════════════════════════════════════════════════
                # 【学习要点】Response 形状校验 — 在解析 finish_reason 之前
                # 关键问题: 不同 provider 失败模式不一样
                #   - OpenAI: 抛异常(被外层 try/except 捕获)
                #   - Anthropic: 抛异常
                #   - Codex Responses: 返回 response.status="failed"
                #   - 部分 OpenRouter 中转: 返回 response 但 error 字段非空
                # 所以这里统一做形状检查,失败就进 fallback 链
                # 触发条件:
                #   - response is None
                #   - validate_response() 返回 False(transport 层的格式校验)
                #   - Codex status="failed" 或 status="incomplete" 但 reason 不对
                # 不在这里 throw,而是标记 response_invalid + continue 重试
                # → 让 14 个 *_attempted 标志有机会逐个尝试
                # ═══════════════════════════════════════════════════════════════
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.8】Response 形状校验(在解析 finish_reason 之前)
                # 关键问题: 不同 provider 失败模式不一样
                #   - OpenAI: 抛异常(被外层 try/except 捕获)
                #   - Anthropic: 抛异常
                #   - Codex Responses: 返回 response.status="failed"
                #   - 部分 OpenRouter 中转: 返回 response 但 error 字段非空
                # 不在这里 throw,而是标记 response_invalid + continue 重试
                # → 让 14 个 *_attempted 标志有机会逐个尝试
                # ═══════════════════════════════════════════════════════════════
                # 4.8.1 初始化形状校验状态
                # Validate response shape before proceeding
                response_invalid = False
                error_details = []
                # 4.8.2 Codex Responses API 特殊处理: 用 transport.validate_response
                if agent.api_mode == "codex_responses":
                    _ct_v = agent._get_transport()
                    # 4.8.3 校验失败
                    if not _ct_v.validate_response(response):
                        # 4.8.4 response 是 None(SDK 都没返回对象)
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        else:
                            # Provider returned a terminal failure (e.g. quota exhaustion).
                            # Treat as invalid so the fallback chain is triggered instead of
                            # letting the error bubble up outside the retry/fallback loop.
                            _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                            if _codex_resp_status in {"failed", "cancelled"}:
                                _codex_error_obj = getattr(response, "error", None)
                                _codex_error_msg = (
                                    _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                    else str(_codex_error_obj) if _codex_error_obj
                                    else f"Responses API returned status '{_codex_resp_status}'"
                                )
                                logger.warning(
                                    "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                    _codex_resp_status, _codex_error_msg,
                                    agent._client_log_context(),
                                )
                                response_invalid = True
                                error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                            else:
                                # output_text fallback: stream backfill may have failed
                                # but normalize can still recover from output_text
                                _out_text = getattr(response, "output_text", None)
                                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                if _out_text_stripped:
                                    logger.debug(
                                        "Codex response.output is empty but output_text is present "
                                        "(%d chars); deferring to normalization.",
                                        len(_out_text_stripped),
                                    )
                                else:
                                    _resp_status = getattr(response, "status", None)
                                    _resp_incomplete = getattr(response, "incomplete_details", None)
                                    logger.warning(
                                        "Codex response.output is empty after stream backfill "
                                        "(status=%s, incomplete_details=%s, model=%s). %s",
                                        _resp_status, _resp_incomplete,
                                        getattr(response, "model", None),
                                        f"api_mode={agent.api_mode} provider={agent.provider}",
                                    )
                                    response_invalid = True
                                    error_details.append("response.output is empty")
                elif agent.api_mode == "anthropic_messages":
                    _tv = agent._get_transport()
                    if not _tv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("response.content invalid (not a non-empty list)")
                elif agent.api_mode == "bedrock_converse":
                    _btv = agent._get_transport()
                    if not _btv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("Bedrock response invalid (no output or choices)")
                else:
                    _ctv = agent._get_transport()
                    if not _ctv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")

                if response_invalid:
                    # Stop spinner silently — retry status is now buffered
                    # and only surfaced if every retry+fallback exhausts.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    
                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    retry_count += 1
                    
                    # Eager fallback: empty/malformed responses are a common
                    # rate-limit symptom.  Switch to fallback immediately
                    # rather than retrying with extended backoff.
                    if agent._fallback_index < len(agent._fallback_chain):
                        agent._buffer_status("⚠️ Empty/malformed response — switching to fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue

                    # Check for error field in response (some providers include this)
                    error_msg = "Unknown"
                    provider_name = "Unknown"
                    if response and hasattr(response, 'error') and response.error:
                        error_msg = str(response.error)
                        # Try to extract provider from error metadata
                        if hasattr(response.error, 'metadata') and response.error.metadata:
                            provider_name = response.error.metadata.get('provider_name', 'Unknown')
                    elif response and hasattr(response, 'message') and response.message:
                        error_msg = str(response.message)
                    
                    # Try to get provider from model field (OpenRouter often returns actual model used)
                    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                        provider_name = f"model={response.model}"
                    
                    # Check for x-openrouter-provider or similar metadata
                    if provider_name == "Unknown" and response:
                        # Log all response attributes for debugging
                        resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                        if agent.verbose_logging:
                            logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.9】构造给用户看的诊断信息(provider + 错误码 + 耗时)
                    # ═══════════════════════════════════════════════════════════════
                    # 4.9.1 初始化错误码变量
                    # Extract error code from response for contextual diagnostics
                    _resp_error_code = None
                    # 4.9.2 从 response.error.code 提取错误码(可能是 attr 或 dict key)
                    if response and hasattr(response, 'error') and response.error:
                        _code_raw = getattr(response.error, 'code', None)
                        if _code_raw is None and isinstance(response.error, dict):
                            _code_raw = response.error.get('code')
                        # 4.9.3 转成 int(可能失败 → 跳过)
                        if _code_raw is not None:
                            try:
                                _resp_error_code = int(_code_raw)
                            except (TypeError, ValueError):
                                pass

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.10】根据错误码 + 响应时间推断"失败原因提示"
                    # 不总是假设 rate limit,而是更精确的诊断
                    # Build a human-readable failure hint from the error code
                    # and response time, instead of always assuming rate limiting.
                    # ═══════════════════════════════════════════════════════════════
                    # 4.10.1 Cloudflare 524 → 上游超时
                    if _resp_error_code == 524:
                        _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                    # 4.10.2 网关 504 → gateway timeout
                    elif _resp_error_code == 504:
                        _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                    # 4.10.3 速率限制 429
                    elif _resp_error_code == 429:
                        _failure_hint = f"rate limited by upstream provider (429)"
                    # 4.10.4 服务器错误 500/502
                    elif _resp_error_code in {500, 502}:
                        _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                    # 4.10.5 上游过载 503/529
                    elif _resp_error_code in {503, 529}:
                        _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                    # 4.10.6 其他错误码 → 直接显示
                    elif _resp_error_code is not None:
                        _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                    # 4.10.7 响应快 + 没错误码 → 推断为速率限制
                    elif api_duration < 10:
                        _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                    # 4.10.8 响应慢 → 推断为上游超时
                    elif api_duration > 60:
                        _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                    # 4.10.9 其他情况 → 显示原始响应时间
                    else:
                        _failure_hint = f"response time {api_duration:.1f}s"

                    # 4.11 把诊断信息 buffer 到 verbose 输出
                    agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                    agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
                    # 4.12 清理 provider 错误信息(去敏感信息/标准化格式)
                    cleaned_provider_error = agent._clean_error_message(error_msg)
                    agent._buffer_vprint(f"   📝 Provider message: {cleaned_provider_error}")
                    agent._buffer_vprint(f"   ⏱️  {_failure_hint}")

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.13】Max retries 耗尽后的 Fallback 决策
                    # 关键设计: 永不轻易"放弃" — 先试 fallback,再退出
                    # 决策树:
                    #   1. retry_count >= max_retries?
                    #      ├─ 否 → continue 重试
                    #      └─ 是 → 检查 _has_pending_fallback() 是否还有备份
                    #   2. 有 backup?
                    #      ├─ 是 → _try_activate_fallback() 切换 provider
                    #      │       重置 retry_count=0,compression_attempts=0
                    #      │       让新的 provider 重新跑所有重试
                    #      └─ 否 → emit error + persist + return 失败
                    # 为什么要重置 retry_count?
                    #   → 切到 fallback 是新起点,不应继承主 provider 的重试计数
                    # 为什么要 persist_session?
                    #   → 即便失败也要把当前状态写盘,下次 /continue 能接上
                    # ═══════════════════════════════════════════════════════════════
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.13.1】如果重试次数已达上限
                    # 决策: 试 fallback,失败就真的放弃
                    if retry_count >= max_retries:
                        # 4.13.2 Try fallback before giving up
                        # 4.13.3 有 pending fallback → 通知 UI
                        if agent._has_pending_fallback():
                            agent._buffer_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        # 4.13.4 切换到 fallback provider
                        if agent._try_activate_fallback():
                            # 4.13.5 重置所有状态:让新 provider 重新跑所有重试
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            # 4.13.6 continue 到 while 顶部,重新尝试
                            continue
                        # ═══════════════════════════════════════════════════════════════
                        # 【步骤 4.13.7】没有 fallback 可用 → 真的放弃了
                        # Terminal — flush buffered retry trace so user sees what happened.
                        # ═══════════════════════════════════════════════════════════════
                        # 4.13.8 刷出 buffered 状态让用户看到
                        agent._flush_status_buffer()
                        # 4.13.9 发最终错误状态
                        agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        # 4.13.10 写 error log
                        logger.error(f"{agent.log_prefix}Invalid API response after {max_retries} retries.")
                        # 4.13.11 即便失败也 persist session
                        agent._persist_session(messages, conversation_history)
                        # 4.13.12 返回失败结果
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                            "failed": True  # Mark as failure for filtering
                        }

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.14】还没到 max_retries → 退避后重试
                    # 设计: jittered exponential backoff(抖动指数退避)
                    #   基础延迟 5s,上限 120s
                    #   jitter(随机抖动)防止多个重试同步撞车
                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    # ═══════════════════════════════════════════════════════════════
                    # 4.14.1 计算退避时间
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    # 4.14.2 通知用户:将在 N 秒后重试 + 失败原因
                    agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
                    # 4.14.3 写 warning log
                    logger.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.15】分小段 sleep,保持对中断的响应
                    # Sleep in small increments to stay responsive to interrupts
                    # ═══════════════════════════════════════════════════════════════
                    # 4.15.1 记录 sleep 结束时间
                    sleep_end = time.time() + wait_time
                    # 4.15.2 活动触碰计数器(gateway 用,30s 一次)
                    _backoff_touch_counter = 0
                    # 4.15.3 分小段 sleep 的循环(0.2s 一段)
                    while time.time() < sleep_end:
                        # 4.15.4 检查用户是否中断
                        if agent._interrupt_requested:
                            # 4.15.5 中断发生:打印 + persist + 清除中断标志 + 返回
                            agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            agent._persist_session(messages, conversation_history)
                            agent.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        # 4.15.6 小睡 0.2s(响应中断)
                        time.sleep(0.2)
                        # Touch activity every ~30s so the gateway's inactivity
                        # monitor knows we're alive during backoff waits.
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            agent._touch_activity(
                                f"retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
                    continue  # Retry the API call

                # ═══════════════════════════════════════════════════════════════
                # 【学习要点】finish_reason 归一化:4 种 API 协议 → 1 个统一语义
                # 4 种 api_mode 在这里分支:
                #   1. codex_responses    → 看 response.status / incomplete_details
                #   2. anthropic_messages → 看 response.stop_reason
                #   3. bedrock_converse   → 已由 transport 归一化,直接读
                #   4. chat_completions   → 调 transport.normalize_response()
                # 为什么需要归一化? OpenAI 用 "stop"/"length"/"tool_calls"
                #                       Anthropic 用 "end_turn"/"max_tokens"/"tool_use"
                #                       Bedrock 用 "end_turn"/"max_tokens"/"tool_use"
                #                       名称都不一样,但语义对得上
                # 归一化后所有路径都用 finish_reason ∈ {stop, length, tool_calls}
                # 后续 length 触发自动续写,stop 触发退出,tool_calls 触发工具执行
                # ═══════════════════════════════════════════════════════════════
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.16】归一化 finish_reason(4 种 API 协议 → 1 个统一语义)
                # 4 种 api_mode 在这里分支:
                #   1. codex_responses    → 看 response.status / incomplete_details
                #   2. anthropic_messages → 看 response.stop_reason
                #   3. bedrock_converse   → 已由 transport 归一化,直接读
                #   4. chat_completions   → 调 transport.normalize_response()
                # 归一化后所有路径都用 finish_reason ∈ {stop, length, tool_calls}
                # ═══════════════════════════════════════════════════════════════
                # Check finish_reason before proceeding
                if agent.api_mode == "codex_responses":
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.16.1】Codex Responses API 特殊处理
                    # 看 status 字段和 incomplete_details.reason
                    # ═══════════════════════════════════════════════════════════════
                    # 4.16.1.1 拿 response.status(可能是 "completed"/"incomplete"/"failed")
                    status = getattr(response, "status", None)
                    # 4.16.1.2 拿 incomplete_details(只在 status=="incomplete" 时有意义)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    # 4.16.1.3 incomplete_details 可能是 dict 或 SimpleNamespace
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    # 4.16.1.4 映射到统一语义: max_output_tokens/length → "length"
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.16.2】Anthropic Messages API: 用 transport 映射 stop_reason
                # Anthropic: end_turn→stop, max_tokens→length, tool_use→tool_calls
                # ═══════════════════════════════════════════════════════════════
                elif agent.api_mode == "anthropic_messages":
                    _tfr = agent._get_transport()
                    finish_reason = _tfr.map_finish_reason(response.stop_reason)
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.16.3】Bedrock Converse: transport 已归一化
                # ═══════════════════════════════════════════════════════════════
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock response already normalized at dispatch — use transport
                    _bt_fr = agent._get_transport()
                    _bedrock_result = _bt_fr.normalize_response(response)
                    finish_reason = _bedrock_result.finish_reason
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.16.4】OpenAI Chat Completions: 调 transport.normalize_response
                # 这个方法会构造标准的 assistant_message(包含 content/tool_calls/reasoning)
                # ═══════════════════════════════════════════════════════════════
                else:
                    _cc_fr = agent._get_transport()
                    _finish_result = _cc_fr.normalize_response(response)
                    finish_reason = _finish_result.finish_reason
                    assistant_message = _finish_result
                    if agent._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.17】检测到 length(模型输出被截断)
                # 处理: 给用户提示 + 准备自动续写
                # ═══════════════════════════════════════════════════════════════
                if finish_reason == "length":
                    # 4.17.1 检查是否是网络中断造成的 partial stub
                    if getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID:
                        # 4.17.2 网络中断的特殊提示
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Stream interrupted by network error "
                            f"(finish_reason='length' on partial-stream-stub)",
                            force=True,
                        )
                    # 4.17.3 正常 max_tokens 截断
                    else:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Response truncated "
                            f"(finish_reason='length') - model hit max output tokens",
                            force=True,
                        )

                    # Normalize the truncated response to a single OpenAI-style
                    # message shape so text-continuation and tool-call retry
                    # work uniformly across chat_completions, bedrock_converse,
                    # and anthropic_messages.  For Anthropic we use the same
                    # adapter the agent loop already relies on so the rebuilt
                    # interim assistant message is byte-identical to what
                    # would have been appended in the non-truncated path.
                    _trunc_msg = None
                    _trunc_transport = agent._get_transport()
                    if agent.api_mode == "anthropic_messages":
                        _trunc_result = _trunc_transport.normalize_response(
                            response, strip_tool_prefix=agent._is_anthropic_oauth
                        )
                    else:
                        _trunc_result = _trunc_transport.normalize_response(response)
                    _trunc_msg = _trunc_result

                    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                    # ── Detect thinking-budget exhaustion ──────────────
                    # When the model spends ALL output tokens on reasoning
                    # and has none left for the response, continuation
                    # retries are pointless.  Detect this early and give a
                    # targeted error instead of wasting 3 API calls.
                    # A response is "thinking exhausted" only when the model
                    # actually produced reasoning blocks but no visible text after
                    # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                    # NVIDIA Build, minimax) may return content=None or an empty
                    # string for unrelated reasons — treat those as normal
                    # truncations that deserve continuation retries, not as
                    # thinking-budget exhaustion.
                    _has_think_tags = bool(
                        _trunc_content and re.search(
                            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                            _trunc_content,
                            re.IGNORECASE,
                        )
                    )
                    _thinking_exhausted = (
                        not _trunc_has_tool_calls
                        and _has_think_tags
                        and (
                            (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
                            or _trunc_content is None
                        )
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.18】检测 thinking budget 耗尽
                    # 场景: 模型把全部 output tokens 用来思考了,没剩给响应
                    # 这种情况续写没用,直接返回用户友好提示
                    # ═══════════════════════════════════════════════════════════════
                    if _thinking_exhausted:
                        # 4.18.1 构造错误描述
                        _exhaust_error = (
                            "Model used all output tokens on reasoning with none left "
                            "for the response. Try lowering reasoning effort or "
                            "increasing max_tokens."
                        )
                        agent._vprint(
                            f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
                            f"no visible response was produced.",
                            force=True,
                        )
                        # Return a user-friendly message as the response so
                        # CLI (response box) and gateway (chat message) both
                        # display it naturally instead of a suppressed error.
                        _exhaust_response = (
                            "⚠️ **Thinking Budget Exhausted**\n\n"
                            "The model used all its output tokens on reasoning "
                            "and had none left for the actual response.\n\n"
                            "To fix this:\n"
                            "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                            "→ Or switch to a larger/non-reasoning model with `/model`"
                        )
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": _exhaust_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": _exhaust_error,
                        }

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.19】Length 续写逻辑(自动续接被截断的响应)
                    # 仅适用于 chat_completions / bedrock_converse / anthropic_messages
                    # 续写次数限制: 最多 3 次
                    # ═══════════════════════════════════════════════════════════════
                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        # 4.19.1 把 transport 归一化后的 message 用作 assistant_message
                        assistant_message = _trunc_msg
                        # 4.19.2 只对"没有 tool_calls"的截断处理(有 tool_calls 的走另一条路)
                        if assistant_message is not None and not _trunc_has_tool_calls:
                            # 4.19.3 续写计数器 +1
                            length_continue_retries += 1
                            # 4.19.4 把截断的部分作为 assistant message 入 history
                            interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                            messages.append(interim_msg)
                            # 4.19.5 累加 partial response 部分(用于最终拼接)
                            if assistant_message.content:
                                truncated_response_parts.append(assistant_message.content)

                            if length_continue_retries < 3:
                                _is_partial_stream_stub = (
                                    getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                                )
                                _dropped_tools = getattr(
                                    response, "_dropped_tool_names", None
                                )

                                if _is_partial_stream_stub and _dropped_tools:
                                    _tool_list = ", ".join(_dropped_tools[:3])
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted mid "
                                        f"tool-call ({_tool_list}) — requesting "
                                        f"chunked retry "
                                        f"({length_continue_retries}/3)..."
                                    )
                                elif _is_partial_stream_stub:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted — "
                                        f"requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                else:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )

                                _continue_content = _get_continuation_prompt(
                                    _is_partial_stream_stub, _dropped_tools
                                )
                                continue_msg = {
                                    "role": "user",
                                    "content": _continue_content,
                                }
                                messages.append(continue_msg)
                                agent._session_messages = messages
                                restart_with_length_continuation = True
                                break

                            partial_response = agent._strip_think_blocks("".join(truncated_response_parts)).strip()
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": partial_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response remained truncated after 3 continuation attempts",
                            }

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.20】截断的是 tool_call 而不是文本 → 不同的续写策略
                    # 区别: 文本截断 → 续写(让模型继续)
                    #       工具截断 → 重试(让模型重新生成完整 tool_call)
                    # 同样最多 3 次
                    # ═══════════════════════════════════════════════════════════════
                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        # 4.20.1 用归一化后的 message
                        assistant_message = _trunc_msg
                        # 4.20.2 有 tool_calls 的情况(工具 JSON 截断)
                        if assistant_message is not None and _trunc_has_tool_calls:
                            # 4.20.3 检查是否是网络中断造成的 stub stall
                            _is_stub_stall = (
                                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                            )
                            # 4.20.4 还没超过 3 次重试
                            if truncated_tool_call_retries < 3:
                                # 4.20.5 重试计数器 +1
                                truncated_tool_call_retries += 1
                                # 4.20.6 区分两种情况给用户看
                                if _is_stub_stall:
                                    # 4.20.7 网络中断:说"stream interrupted"
                                    agent._buffer_vprint(
                                        f"⚠️  Stream interrupted mid tool-call — "
                                        f"retrying ({truncated_tool_call_retries}/3)..."
                                    )
                                # 4.20.8 正常截断:说"truncated tool call"
                                else:
                                    agent._buffer_vprint(
                                        f"⚠️  Truncated tool call detected — "
                                        f"retrying API call "
                                        f"({truncated_tool_call_retries}/3)..."
                                    )
                                # Boost max_tokens on each retry so the model has
                                # more room to complete the tool-call JSON. A
                                # network stall doesn't need a bigger budget, but
                                # a genuine output-cap truncation does, and the
                                # boost is harmless for the stall case.
                                _tc_boost_base = agent.max_tokens if agent.max_tokens else 4096
                                _tc_boost = _tc_boost_base * (truncated_tool_call_retries + 1)
                                _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
                                if _tc_requested_cap is not None:
                                    _tc_boost = max(_tc_boost, _tc_requested_cap)
                                _tc_boost_cap = max(32768, _tc_requested_cap or 0)
                                agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
                                # Don't append the broken response to messages;
                                # just re-run the same API call from the current
                                # message state, giving the model another chance.
                                continue
                            agent._flush_status_buffer()
                            if _is_stub_stall:
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after 3 retries — the action was not executed.",
                                    force=True,
                                )
                            else:
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                    force=True,
                                )
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": (
                                    "Stream repeatedly dropped mid tool-call (network); "
                                    "the tool was not executed"
                                    if _is_stub_stall
                                    else "Response truncated due to output length limit"
                                ),
                            }

                    # If we have prior messages, roll back to last complete state
                    if len(messages) > 1:
                        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)

                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit"
                        }
                    else:
                        # First message was truncated - mark as failed
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": "First response truncated due to output length limit"
                        }
                
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.21】成功路径! 记录 token 用量给 context 管理
                # 这里处理 usage 字段(每种 API 协议格式不同,需要归一化)
                # ═══════════════════════════════════════════════════════════════
                # Track actual token usage from response for context management
                if hasattr(response, 'usage') and response.usage:
                    # 4.21.1 归一化 usage(OpenAI / Anthropic / Codex 三种格式)
                    canonical_usage = normalize_usage(
                        response.usage,
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.2】提取标准 token 数
                    # ═══════════════════════════════════════════════════════════════
                    # 4.21.2.1 prompt_tokens = 输入 token 数
                    prompt_tokens = canonical_usage.prompt_tokens
                    # 4.21.2.2 completion_tokens = 输出 token 数
                    completion_tokens = canonical_usage.output_tokens
                    # 4.21.2.3 total_tokens = 合计
                    total_tokens = canonical_usage.total_tokens
                    # Forward canonical token + cache buckets so context engines
                    # can make decisions on cache hit ratios / reasoning costs,
                    # not just legacy aggregate tokens. Legacy keys stay for
                    # back-compat with engines that only read prompt/completion/total.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.3】构造完整 usage_dict(给 context compressor 用)
                    # 包含 8 个字段: 3 个老字段 + 5 个新字段
                    # ═══════════════════════════════════════════════════════════════
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "input_tokens": canonical_usage.input_tokens,
                        "output_tokens": canonical_usage.output_tokens,
                        "cache_read_tokens": canonical_usage.cache_read_tokens,
                        "cache_write_tokens": canonical_usage.cache_write_tokens,
                        "reasoning_tokens": canonical_usage.reasoning_tokens,
                    }
                    # 4.21.4 通知 context compressor 更新内部状态
                    # (会更新它的滑动窗口 / 触发下次预压缩等)
                    agent.context_compressor.update_from_response(usage_dict)

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.5】缓存探测到的 context_length
                    # 触发条件: 之前因 context_length 错误探测过
                    # Cache discovered context length after successful call.
                    # Only persist limits confirmed by the provider (parsed
                    # from the error message), not guessed probe tiers.
                    # ═══════════════════════════════════════════════════════════════
                    if getattr(agent.context_compressor, "_context_probed", False):
                        ctx = agent.context_compressor.context_length
                        # 4.21.5.1 如果是可持久化的探测(从错误信息解析的,不是猜的)
                        if getattr(agent.context_compressor, "_context_probe_persistable", False):
                            # 4.21.5.2 持久化到磁盘
                            save_context_length(agent.model, agent.base_url, ctx)
                            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
                        # 4.21.5.3 重置探测标志
                        agent.context_compressor._context_probed = False
                        agent.context_compressor._context_probe_persistable = False

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.6】累加 session 级 token(给 /insights 用)
                    # ═══════════════════════════════════════════════════════════════
                    agent.session_prompt_tokens += prompt_tokens
                    agent.session_completion_tokens += completion_tokens
                    agent.session_total_tokens += total_tokens
                    # ═══════════════════════════════════════════════════════════════
                    # 【学习要点】Session 级 token 用量统计 — 9 个累加器
                    # 用途:
                    #   - 成本审计(按 prompt/completion/cache_read 分开计费)
                    #   - 上下文压缩触发条件(总 token 接近 model 上限)
                    #   - 调试时定位"哪个 turn 把 context 撑爆的"
                    # 关键设计: cache_read 和 cache_write 分开统计
                    #   cache_read = $0.30/M (Anthropic)  cache_write = $3.75/M
                    #   命中率 75% 意味着节省 90% 成本
                    #   → 这两个数字是评估 prompt 缓存效果的核心指标
                    # session_reasoning_tokens 单独统计: 推理 token 走 OpenAI o1
                    #   定价($60/M)远高于普通 completion($15/M)
                    # ═══════════════════════════════════════════════════════════════
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.7】继续累加 5 个细分 token 累加器
                    #   cache_read / cache_write / reasoning 各自分开
                    # ═══════════════════════════════════════════════════════════════
                    # 4.21.7.1 API 调用次数 +1
                    agent.session_api_calls += 1
                    # 4.21.7.2 累加输入(精确)
                    agent.session_input_tokens += canonical_usage.input_tokens
                    # 4.21.7.3 累加输出
                    agent.session_output_tokens += canonical_usage.output_tokens
                    # 4.21.7.4 缓存命中 token(便宜)
                    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
                    # 4.21.7.5 缓存写入 token(贵)
                    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
                    # 4.21.7.6 推理 token(最贵,OpenAI o1 类)
                    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.8】记 API call info log(给 grep 用)
                    # Log API call details for debugging/observability
                    # ═══════════════════════════════════════════════════════════════
                    # 4.21.8.1 构造 cache 命中率字符串
                    _cache_pct = ""
                    if canonical_usage.cache_read_tokens and prompt_tokens:
                        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                    # 4.21.8.2 写 info log(单行,易 grep)
                    logger.info(
                        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                        agent.session_api_calls, agent.model, agent.provider or "unknown",
                        prompt_tokens, completion_tokens, total_tokens,
                        api_duration, _cache_pct,
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.9】估算本次 API 调用的成本(美元)
                    # estimate_usage_cost 会按 model + provider 查定价表
                    # ═══════════════════════════════════════════════════════════════
                    cost_result = estimate_usage_cost(
                        agent.model,
                        canonical_usage,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_key=getattr(agent, "api_key", ""),
                    )
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.10】如果能算出来成本,累加到 session
                    # ═══════════════════════════════════════════════════════════════
                    if cost_result.amount_usd is not None:
                        # 4.21.10.1 累加美元成本
                        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
                    # 4.21.10.2 记 cost status(可能 "estimated"/"unknown")
                    agent.session_cost_status = cost_result.status
                    # 4.21.10.3 记 cost source(从哪个 pricing 表来的)
                    agent.session_cost_source = cost_result.source

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.11】持久化 token 计数到 session DB(/insights 用)
                    # Persist token counts to session DB for /insights.
                    # Do this for every platform with a session_id so non-CLI
                    # sessions (gateway, cron, delegated runs) cannot lose
                    # token/accounting data if a higher-level persistence path
                    # is skipped or fails. Gateway/session-store writes use
                    # absolute totals, so they safely overwrite these per-call
                    # deltas instead of double-counting them.
                    # ═══════════════════════════════════════════════════════════════
                    if agent._session_db and agent.session_id:
                        try:
                            # 4.21.11.1 确保 session row 存在
                            # Ensure the session row exists before attempting UPDATE.
                            # Under concurrent load (cron/kanban), the initial
                            # _ensure_db_session() may have failed due to SQLite
                            # locking.  Retry here so per-call token deltas are
                            # not silently lost (UPDATE on a non-existent row
                            # affects 0 rows without error).
                            if not agent._session_db_created:
                                agent._ensure_db_session()
                            # 4.21.11.2 写入 6 个细分字段
                            agent._session_db.update_token_counts(
                                agent.session_id,
                                input_tokens=canonical_usage.input_tokens,
                                output_tokens=canonical_usage.output_tokens,
                                cache_read_tokens=canonical_usage.cache_read_tokens,
                                cache_write_tokens=canonical_usage.cache_write_tokens,
                                reasoning_tokens=canonical_usage.reasoning_tokens,
                                estimated_cost_usd=float(cost_result.amount_usd)
                                if cost_result.amount_usd is not None else None,
                                cost_status=cost_result.status,
                                cost_source=cost_result.source,
                                billing_provider=agent.provider,
                                billing_base_url=agent.base_url,
                                billing_mode="subscription_included"
                                if cost_result.status == "included" else None,
                                model=agent.model,
                                api_call_count=1,
                            )
                        # 4.21.11.3 持久化失败只 debug log,不打断主流程
                        except Exception as e:
                            # Log token persistence failures so they're
                            # visible in agent.log — silent loss here is
                            # the root cause of undercounted analytics.
                            logger.debug(
                                "Token persistence failed (session=%s, tokens=%d): %s",
                                agent.session_id, total_tokens, e,
                            )

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.12】Verbose 模式: 打印 token 详情
                    # ═══════════════════════════════════════════════════════════════
                    if agent.verbose_logging:
                        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.21.13】给用户看 cache 命中率
                    # Surface cache hit stats for any provider that reports
                    # them — not just those where we inject cache_control
                    # markers.  OpenAI/Kimi/DeepSeek/Qwen all do automatic
                    # server-side prefix caching and return
                    # ``prompt_tokens_details.cached_tokens``; users
                    # previously could not see their cache % because this
                    # line was gated on ``_use_prompt_caching``, which is
                    # only True for Anthropic-style marker injection.
                    # ``canonical_usage`` is already normalised from all
                    # three API shapes (Anthropic / Codex / OpenAI-chat)
                    # so we can rely on its values directly.
                    # ═══════════════════════════════════════════════════════════════
                    # 4.21.13.1 拿到 cache_read / cache_write
                    cached = canonical_usage.cache_read_tokens
                    written = canonical_usage.cache_write_tokens
                    prompt = usage_dict["prompt_tokens"]
                    # 4.21.13.2 有 cache 数据 + 非安静模式 → 显示
                    if (cached or written) and not agent.quiet_mode:
                        # 4.21.13.3 计算命中率
                        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                        # 4.21.13.4 打印给用户看
                        agent._vprint(
                            f"{agent.log_prefix}   💾 Cache: "
                            f"{cached:,}/{prompt:,} tokens "
                            f"({hit_pct:.0f}% hit, {written:,} written)"
                        )
                
                # ═══════════════════════════════════════════════════════════════
                # 【学习要点】成功 ≠ 可用 — 紧跟的空响应检查
                # 关键概念: HTTP 200 拿到 bytes 回来 ≠ 模型给了有用输出
                # 失败模式:
                #   1. model 返回空 content (空字符串 / null)
                #   2. model 只返回 thinking 没有正文
                #   3. model 一直 tool_calls 不停调同一个工具
                #   4. router 把 finish_reason 改成 tool_calls 但实际是 length
                # has_retried_429 = False  ← 重置 429 重试标志(成功后才重置)
                # 为什么不立即清 retry buffer?
                #   → "拿到 bytes" 还不够,需要确认 content/tool_calls 都有
                #   → 空响应会在下面 3500-3700 的空响应防护段处理
                #   → buffer 留到确认真的成功才清,避免漏掉用户应该看到的状态
                # ═══════════════════════════════════════════════════════════════
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.22】成功响应处理 + 退出内层重试循环
                # 关键概念: HTTP 200 拿到 bytes 回来 ≠ 模型给了有用输出
                # 失败模式:
                #   1. model 返回空 content (空字符串 / null)
                #   2. model 只返回 thinking 没有正文
                #   3. model 一直 tool_calls 不停调同一个工具
                #   4. router 把 finish_reason 改成 tool_calls 但实际是 length
                # has_retried_429 = False  ← 重置 429 重试标志(成功后才重置)
                # 为什么不立即清 retry buffer?
                #   → "拿到 bytes" 还不够,需要确认 content/tool_calls 都有
                #   → 空响应会在下面 3500-3700 的空响应防护段处理
                #   → buffer 留到确认真的成功才清,避免漏掉用户应该看到的状态
                # ═══════════════════════════════════════════════════════════════
                # 4.22.1 重置 429 退避标志(成功后才重置,失败会保留)
                has_retried_429 = False
                # Note: don't clear the retry buffer here — an "API call
                # success" only means we got bytes back, not that we got
                # usable content. Empty responses still loop through the
                # empty-retry path below; the buffer is cleared when
                # genuinely successful content is detected later (~L4127).
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.22.2】Nous 跨 session 限流状态清理
                # 成功 = 限流已恢复,清掉本地限流标记让其他 session 也能用
                # Clear Nous rate limit state on successful request —
                # proves the limit has reset and other sessions can
                # resume hitting Nous.
                # ═══════════════════════════════════════════════════════════════
                if agent.provider == "nous":
                    # 4.22.2.1 动态 import + 清限流(失败不报错)
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                # 4.22.3 触摸活动计时器(gateway 用,30s 不活动会断)
                agent._touch_activity(f"API call #{api_call_count} completed")
                # 4.22.4 退出内层 while retry 循环(成功)
                break

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                api_elapsed = time.time() - api_start_time
                agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
                agent._persist_session(messages, conversation_history)
                interrupted = True
                final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                break

            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.23】异常处理分支(API 调用抛出任何异常)
            # 设计: 抓所有 Exception(而不是只抓特定类型)
            #       然后在内部用 isinstance() 分类处理
            # ═══════════════════════════════════════════════════════════════
            except Exception as api_error:
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.23.1】停止 spinner
                # 原因: 重试状态会 buffered 起来,只在所有重试+fallback 用完时才 flush
                # Stop spinner silently — retry status is buffered and
                # only flushed when every retry+fallback is exhausted.
                # ═══════════════════════════════════════════════════════════════
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # -----------------------------------------------------------
                # UnicodeEncodeError recovery.  Two common causes:
                #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                #      (Google Docs, rich-text editors) — sanitize and retry.
                #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                #      (e.g. Chromebooks) — any non-ASCII character fails.
                #      Detect via the error message mentioning 'ascii' codec.
                # We sanitize messages in-place and may retry twice:
                # first to strip surrogates, then once more for pure
                # ASCII-only locale sanitization if needed.
                # -----------------------------------------------------------
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.23.2】UnicodeEncodeError 专项处理
                # 两种常见原因:
                #   1. Lone surrogates (U+D800..U+DFFF) 来自剪贴板粘贴
                #      (Google Docs、富文本编辑器) — 清洗后重试
                #   2. ASCII codec 错误(LANG=C 或非 UTF-8 locale 系统,
                #      如 Chromebook)— 任何非 ASCII 字符都失败
                #      检测: 错误信息里包含 'ascii' codec
                # 可能重试 2 次: 第一次剥 surrogate,第二次 ASCII-only 清洗
                # 计数器 _unicode_sanitization_passes 防止无限循环
                # ═══════════════════════════════════════════════════════════════
                if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
                    # 4.23.2.1 把错误信息转小写做检测
                    _err_str = str(api_error).lower()
                    # 4.23.2.2 检测是否是 ASCII codec 错误
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    # 4.23.2.3 检测是否是 surrogate 错误(utf-8 拒绝 U+D800..U+DFFF)
                    # Detect surrogate errors — utf-8 codec refusing to
                    # encode U+D800..U+DFFF.  The error text is:
                    #   "'utf-8' codec can't encode characters in position
                    #    N-M: surrogates not allowed"
                    _is_surrogate_error = (
                        "surrogate" in _err_str
                        or ("'utf-8'" in _err_str and not _is_ascii_codec)
                    )
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.23.3】清洗 messages / api_messages / api_kwargs
                    # 三处都要洗,因为每处都可能带 surrogate
                    #   - messages: 标准 list
                    #   - api_messages: API 副本,可能带 reasoning_content/reasoning_details
                    #   - api_kwargs: 已包装的参数字典
                    #   - prefill_messages: few-shot priming 消息
                    # ═══════════════════════════════════════════════════════════════
                    # 4.23.3.1 清洗标准 messages 列表
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    # 4.23.3.2 清洗 API 副本
                    if isinstance(api_messages, list):
                        if _sanitize_messages_surrogates(api_messages):
                            _surrogates_found = True
                    # 4.23.3.3 清洗 api_kwargs
                    if isinstance(api_kwargs, dict):
                        if _sanitize_structure_surrogates(api_kwargs):
                            _surrogates_found = True
                    # 4.23.3.4 清洗 prefill_messages
                    if isinstance(getattr(agent, "prefill_messages", None), list):
                        if _sanitize_messages_surrogates(agent.prefill_messages):
                            _surrogates_found = True
                    # Gate the retry on the error type, not on whether we
                    # found anything — _force_ascii_payload / the extended
                    # surrogate walker above cover all known paths, but a
                    # new transformed field could still slip through.  If
                    # the error was a surrogate encode failure, always let
                    # the retry run; the proactive sanitizer at line ~8781
                    # runs again on the next iteration.  Bounded by
                    # _unicode_sanitization_passes < 2 (outer guard).
                    if _surrogates_found or _is_surrogate_error:
                        agent._unicode_sanitization_passes += 1
                        if _surrogates_found:
                            agent._buffer_vprint(
                                f"⚠️  Stripped invalid surrogate characters from messages. Retrying..."
                            )
                        else:
                            agent._buffer_vprint(
                                f"⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
                            )
                        continue
                    if _is_ascii_codec:
                        agent._force_ascii_payload = True
                        # ═══════════════════════════════════════════════════════════════
                        # 【步骤 4.24】ASCII codec 错误处理
                        # 触发: 系统 locale 不是 UTF-8,任何非 ASCII 字符都失败
                        # 解决: 全面剥非 ASCII 字符(更激进,保留 surrogate 处理之外的)
                        # Sanitize both the canonical `messages` list and
                        # `api_messages` (the API-copy built before the retry
                        # loop, which may contain extra fields like
                        # reasoning_content that are not in `messages`).
                        # ═══════════════════════════════════════════════════════════════
                        # 4.24.1 清洗标准 messages
                        _messages_sanitized = _sanitize_messages_non_ascii(messages)
                        # 4.24.2 清洗 API 副本
                        if isinstance(api_messages, list):
                            _sanitize_messages_non_ascii(api_messages)
                        # 4.24.3 清洗 api_kwargs(防止 _build_api_kwargs 缓存带非 ASCII)
                        # Also sanitize the last api_kwargs if already built,
                        # so a leftover non-ASCII value in a transformed field
                        # (e.g. extra_body, reasoning_content) doesn't survive
                        # into the next attempt via _build_api_kwargs cache paths.
                        if isinstance(api_kwargs, dict):
                            _sanitize_structure_non_ascii(api_kwargs)
                        # 4.24.4 清洗 prefill_messages(few-shot priming)
                        _prefill_sanitized = False
                        if isinstance(getattr(agent, "prefill_messages", None), list):
                            _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)
                        # 4.24.5 清洗 tool schema(工具定义)
                        _tools_sanitized = False
                        if isinstance(getattr(agent, "tools", None), list):
                            _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)
                        # 4.24.6 清洗 system prompt(包括缓存版本)
                        _system_sanitized = False
                        if isinstance(active_system_prompt, str):
                            _sanitized_system = _strip_non_ascii(active_system_prompt)
                            if _sanitized_system != active_system_prompt:
                                active_system_prompt = _sanitized_system
                                agent._cached_system_prompt = _sanitized_system
                                _system_sanitized = True
                        # 4.24.7 清洗 ephemeral system prompt
                        if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                            _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                            if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                                agent.ephemeral_system_prompt = _sanitized_ephemeral
                                _system_sanitized = True

                        _headers_sanitized = False
                        _default_headers = (
                            agent._client_kwargs.get("default_headers")
                            if isinstance(getattr(agent, "_client_kwargs", None), dict)
                            else None
                        )
                        if isinstance(_default_headers, dict):
                            _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                        # Sanitize the API key — non-ASCII characters in
                        # credentials (e.g. ʋ instead of v from a bad
                        # copy-paste) cause httpx to fail when encoding
                        # the Authorization header as ASCII.  This is the
                        # most common cause of persistent UnicodeEncodeError
                        # that survives message/tool sanitization (#6843).
                        _credential_sanitized = False
                        _raw_key = getattr(agent, "api_key", None) or ""
                        # Entra ID bearer providers are callables — their
                        # minted JWTs are always ASCII, so no sanitization
                        # is needed (and ``_strip_non_ascii`` would crash
                        # on a callable input).
                        if _raw_key and isinstance(_raw_key, str):
                            _clean_key = _strip_non_ascii(_raw_key)
                            if _clean_key != _raw_key:
                                agent.api_key = _clean_key
                                if isinstance(getattr(agent, "_client_kwargs", None), dict):
                                    agent._client_kwargs["api_key"] = _clean_key
                                # Also update the live client — it holds its
                                # own copy of api_key which auth_headers reads
                                # dynamically on every request.
                                if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                                    agent.client.api_key = _clean_key
                                _credential_sanitized = True
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                                    f"(bad copy-paste?) — stripped them. If auth fails, "
                                    f"re-copy the key from your provider's dashboard.",
                                    force=True,
                                )

                        # Always retry on ASCII codec detection —
                        # _force_ascii_payload guarantees the full
                        # api_kwargs payload is sanitized on the
                        # next iteration (line ~8475).  Even when
                        # per-component checks above find nothing
                        # (e.g. non-ASCII only in api_messages'
                        # reasoning_content), the flag catches it.
                        # Bounded by _unicode_sanitization_passes < 2.
                        # ═══════════════════════════════════════════════════════════════
                        # 【步骤 4.25】统计 ASCII 清洗 + 增加清洗次数
                        # 设计: _force_ascii_payload 标志位保证下一轮彻底清洗整个 api_kwargs
                        # 即便 per-component 检查没发现(比如非 ASCII 只在
                        # api_messages 的 reasoning_content),这个标志也能捕获
                        # 受 _unicode_sanitization_passes < 2 约束
                        # ═══════════════════════════════════════════════════════════════
                        agent._unicode_sanitization_passes += 1
                        # 4.25.1 统计"是否真清到了东西"(6 个来源的 OR)
                        _any_sanitized = (
                            _messages_sanitized
                            or _prefill_sanitized
                            or _tools_sanitized
                            or _system_sanitized
                            or _headers_sanitized
                            or _credential_sanitized
                        )
                        if _any_sanitized:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                force=True,
                            )
                        continue

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.26】图片拒绝恢复
                # 触发: provider 不支持多模态,4xx 错误 "Only 'text' content type..."
                # 例子: mlx-lm、text-only endpoints、多模态模型的纯文本 fallback
                # 恢复: 第一次触发就剥所有图片 + 标记 session 为 vision-unsupported
                #       之后整个 session 都用纯文本
                # 检测: best-effort 英文短语匹配(本地化/重写过的错误会绕过)
                # ── Image-rejection recovery ──────────────────────────────
                # ═══════════════════════════════════════════════════════════════
                # 4.26.1 提取错误信息文本(用于模式匹配)
                _err_body = ""
                try:
                    _err_body = str(getattr(api_error, "body", None) or
                                    getattr(api_error, "message", None) or
                                    str(api_error))
                except Exception:
                    pass
                _err_status = getattr(api_error, "status_code", None)
                _IMAGE_REJECTION_PHRASES = (
                    "only 'text' content type is supported",
                    "only text content type is supported",
                    "image_url is not supported",
                    "image content is not supported",
                    "multimodal is not supported",
                    "multimodal content is not supported",
                    "multimodal input is not supported",
                    "vision is not supported",
                    "vision input is not supported",
                    "does not support images",
                    "does not support image input",
                    "does not support multimodal",
                    "does not support vision",
                    "model does not support image",
                    # ChatGPT-account Codex backend
                    # (https://chatgpt.com/backend-api/codex) rejects
                    # data:image/...base64 URLs in input_image fields
                    # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
                    # Expected a valid URL, but got a value with an
                    # invalid format." The OpenAI Responses API on the
                    # public endpoint accepts data URLs, but the
                    # ChatGPT-account variant does not. Without this
                    # phrase the agent cascaded into compression /
                    # context-too-large recovery instead of just
                    # stripping the images. Match is narrow on
                    # purpose — keyed on the field-path apostrophe so
                    # we don't false-trip on other URL validation
                    # errors. (issue #23570)
                    "image_url'. expected",
                    # DeepSeek's OpenAI-compatible API reports text-only
                    # request-body variants as:
                    # "unknown variant `image_url`, expected `text`".
                    "unknown variant `image_url`, expected `text`",
                    "unknown variant image_url, expected text",
                )
                # 4.26.2 把错误信息转小写做短语匹配
                _err_lower = _err_body.lower()
                # 4.26.3 检查是否匹配任一图片拒绝短语
                _looks_like_image_rejection = any(
                    p in _err_lower for p in _IMAGE_REJECTION_PHRASES
                )
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.26.4】4xx-only 门控
                # 永不把 5xx/timeout 当作"服务器拒绝图片"
                # 这些是瞬时错误,必须走普通重试路径
                # 4xx-only gate: never interpret 5xx/timeout as "server
                # said no to images" — those are transient and must
                # route to the normal retry path.
                # ═══════════════════════════════════════════════════════════════
                _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
                if (
                    getattr(agent, "_vision_supported", True)
                    and _looks_like_image_rejection
                    and _status_ok
                ):
                    agent._vision_supported = False
                    _imgs_removed = _strip_images_from_messages(messages)
                    if isinstance(api_messages, list):
                        _strip_images_from_messages(api_messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Server rejected image content — "
                        f"switching to text-only mode for this session"
                        + (". Stripped images from history and retrying." if _imgs_removed else "."),
                        force=True,
                    )
                    continue

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.27】提取 HTTP 状态码和错误上下文
                # 通用错误处理前的最后准备
                # ═══════════════════════════════════════════════════════════════
                # 4.27.1 拿 status_code(可能为 None,如网络错误)
                status_code = getattr(api_error, "status_code", None)
                # 4.27.2 提取错误详情(给诊断日志用)
                error_context = agent._extract_api_error_context(api_error)

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.28】Error Classifier — 错误的结构化分类
                # classify_api_error() 返回 FailoverReason 枚举 + 决策建议
                # 关键设计: 不是所有错误都"重试" — 有些必须切 provider
                # FailoverReason 主要取值:
                #   RATE_LIMIT        → 退避重试 (同 provider)
                #   CONTEXT_OVERFLOW  → 压缩 + 重试
                #   TIMEOUT           → 重试,可能换 provider
                #   MODEL_UNAVAILABLE → 立即切 fallback
                #   long_context_tier → 切 200k 模式 + 压缩(Anthropic Max 订阅)
                #   AUTH_FAILED       → 触发 *_auth_retry_attempted 标志
                # 入参 context_length 是必要的:
                #   → 同样的 40000 token 错误,在 200k 上下文模型是"溢出"
                #   → 在 8k 上下文模型是"正常但太多"
                # 分类结果驱动后面 2700+ 行的所有 if/elif 分支
                # ═══════════════════════════════════════════════════════════════
                # ── Classify the error for structured recovery decisions ──
                # 4.28.1 拿 context_compressor(可能没有,默认 200k)
                _compressor = getattr(agent, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                # 4.28.2 调 classify_api_error 返回结构化分类结果
                classified = classify_api_error(
                    api_error,
                    provider=getattr(agent, "provider", "") or "",
                    model=getattr(agent, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.29】Nous 付费 entitlement 刷新(billing 错误)
                # 触发: 错误是 billing 类型 + provider 是 Nous + 没尝试过刷新
                # 解决: 主动刷 entitlement 凭证(可能用户付费了,需要重新同步)
                # ═══════════════════════════════════════════════════════════════
                if (
                    classified.reason == FailoverReason.billing
                    and _is_nous_inference_route(
                        getattr(agent, "provider", "") or "",
                        getattr(agent, "base_url", "") or "",
                    )
                    and not nous_paid_entitlement_refresh_attempted
                ):
                    # 4.29.1 标志位置 True(防止重试时再次尝试)
                    nous_paid_entitlement_refresh_attempted = True
                    # 4.29.2 调 refresh 函数(返回是否成功)
                    if _try_refresh_nous_paid_entitlement_credentials(agent):
                        # 4.29.3 成功 → 通知用户 + continue 重试
                        agent._vprint(
                            f"{agent.log_prefix}🔐 Nous paid access verified — "
                            "refreshed runtime credentials and retrying request...",
                            force=True,
                        )
                        continue

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.30】凭证池恢复(429/401/402 等)
                # 触发: 多个 provider 共享一个凭证池,池里可能还有别的 key
                # 行为: 换池里的其他 key 重试(就像换 provider 一样)
                # ═══════════════════════════════════════════════════════════════
                recovered_with_pool, has_retried_429 = agent._recover_with_credential_pool(
                    status_code=status_code,
                    has_retried_429=has_retried_429,
                    classified_reason=classified.reason,
                    error_context=error_context,
                )
                # 4.30.1 池恢复成功 → continue 重试
                if recovered_with_pool:
                    continue

                # Image-too-large recovery: shrink oversized native image
                # parts in-place and retry once.  Triggered by Anthropic's
                # per-image 5 MB ceiling (400 with "image exceeds 5 MB
                # maximum") or any other provider that complains about
                # image size.  If shrink fails or a second attempt still
                # fails, fall through to normal error handling.
                if (
                    classified.reason == FailoverReason.image_too_large
                    # 4.30.2 标志位置 True(防止重复尝试)
                    and not image_shrink_retry_attempted
                ):
                    image_shrink_retry_attempted = True
                    # 4.30.3 尝试缩小图片(只试一次)
                    if agent._try_shrink_image_parts_in_messages(api_messages):
                        # 4.30.4 缩小成功 → 通知用户 + continue
                        agent._vprint(
                            f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                            f"shrank and retrying...",
                            force=True,
                        )
                        continue
                    # 4.30.5 缩小失败 → 记录到 log,继续走通用错误处理
                    else:
                        logger.info(
                            "image-shrink recovery: no data-URL image parts found "
                            "or shrink didn't reduce size; surfacing original error."
                        )

                # Multimodal-tool-content recovery: providers that follow
                # the OpenAI spec strictly (tool message content must be a
                # string) reject our list-type content with a 400.  Strip
                # image parts from any list-type tool messages, mark the
                # (provider, model) as no-list-tool-content for the rest
                # of this session so future tool results preemptively
                # downgrade, and retry once.  See issue #27344.
                if (
                    classified.reason == FailoverReason.multimodal_tool_content_unsupported
                    and not multimodal_tool_content_retry_attempted
                ):
                    multimodal_tool_content_retry_attempted = True
                    if agent._try_strip_image_parts_from_tool_messages(api_messages):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Provider rejected list-type tool content — "
                            f"downgraded screenshots to text and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "multimodal-tool-content recovery: no list-type tool "
                            "messages with image parts found; surfacing original error."
                        )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.31】Anthropic OAuth 1M context beta 拒绝恢复
                # 触发: 用户的 OAuth 订阅不支持 1M context beta
                # 错误: "long context beta is not yet available for this subscription"
                # 解决: 禁用 beta header + 重建 client(整个 session)
                # 优势: 1M-capable 订阅不受影响(反应式 vs 无条件省)
                # Anthropic OAuth subscription rejected the 1M-context beta
                # header ("long context beta is not yet available for this
                # subscription"). Disable the beta for the rest of this
                # session, rebuild the client, and retry once.  1M-capable
                # subscriptions never hit this branch — they accept the
                # beta and keep full 1M context.
                # ═══════════════════════════════════════════════════════════════
                if (
                    classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                    and agent.api_mode == "anthropic_messages"
                    and agent._is_anthropic_oauth
                    and not oauth_1m_beta_retry_attempted
                ):
                    # 4.31.1 标志位置 True
                    oauth_1m_beta_retry_attempted = True
                    # 4.31.2 检查是否已经禁用过(防止重复)
                    if not getattr(agent, "_oauth_1m_beta_disabled", False):
                        # 4.31.3 标记禁用了 beta
                        agent._oauth_1m_beta_disabled = True
                        # 4.31.4 关掉旧 client
                        try:
                            agent._anthropic_client.close()
                        except Exception:
                            pass
                        # 4.31.5 重建 client(新 header 配置)
                        agent._rebuild_anthropic_client()
                        agent._vprint(
                            f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                            f"the 1M-context beta — disabled for this session and retrying...",
                            force=True,
                        )
                        continue

                if (
                    agent.api_mode == "codex_responses"
                    and agent.provider in {"openai-codex", "xai-oauth"}
                    and status_code == 401
                    and not codex_auth_retry_attempted
                ):
                    codex_auth_retry_attempted = True
                    if agent._try_refresh_codex_client_credentials(force=True):
                        _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
                        agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
                        continue
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.32】Nous 401 凭证刷新
                # 触发: Nous provider 返回 401(认证失败)
                # 解决: 主动刷新 agent key,再试一次
                # 关键: 只在第一次失败时尝试(标志位 nous_auth_retry_attempted)
                # ═══════════════════════════════════════════════════════════════
                if (
                    agent.api_mode == "chat_completions"
                    and agent.provider == "nous"
                    and status_code == 401
                    and not nous_auth_retry_attempted
                ):
                    # 4.32.1 标志位置 True(防止重复尝试)
                    nous_auth_retry_attempted = True
                    # 4.32.2 强制刷新 client 凭证
                    if agent._try_refresh_nous_client_credentials(force=True):
                        # 4.32.3 刷新成功 → 通知 + continue
                        print(f"{agent.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info.
                    # Most common causes: Portal OAuth expired/revoked,
                    # account out of credits, or agent key blocked.
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    _body_text = ""
                    try:
                        _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                        if _body is not None:
                            _body_text = str(_body)[:200]
                    except Exception:
                        pass
                    print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                    if _body_text:
                        print(f"{agent.log_prefix}   Response: {_body_text}")
                    if not _print_nous_entitlement_guidance(agent, "Nous model access"):
                        print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    print(f"{agent.log_prefix}     • Re-authenticate: hermes auth add nous")
                    print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                    print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                    print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                if (
                    agent.provider == "copilot"
                    and status_code == 401
                    and not copilot_auth_retry_attempted
                ):
                    copilot_auth_retry_attempted = True
                    if agent._try_refresh_copilot_client_credentials():
                        agent._buffer_vprint(f"🔐 Copilot credentials refreshed after 401. Retrying request...")
                        continue
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.33】Anthropic 401 凭证刷新
                # 触发: Anthropic Messages API 返回 401
                # 解决: 刷新凭证(可能是 API key 过期或 OAuth token 失效)
                # 关键: Azure Foundry Entra ID 走 httpx event hook 模式
                # ═══════════════════════════════════════════════════════════════
                if (
                    agent.api_mode == "anthropic_messages"
                    and status_code == 401
                    and hasattr(agent, '_anthropic_api_key')
                    and not anthropic_auth_retry_attempted
                ):
                    # 4.33.1 标志位 + 必要的 import
                    anthropic_auth_retry_attempted = True
                    from agent.anthropic_adapter import _is_oauth_token
                    from agent.azure_identity_adapter import is_token_provider
                    # 4.33.2 尝试刷新凭证
                    if agent._try_refresh_anthropic_client_credentials():
                        # 4.33.3 刷新成功 → continue
                        print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info
                    key = agent._anthropic_api_key
                    print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
                    if is_token_provider(key):
                        # Azure Foundry Entra ID — the bearer token is
                        # minted per-request by an httpx event hook on a
                        # custom http_client passed to the SDK. The 401
                        # means Azure rejected the JWT (RBAC role missing,
                        # az login expired, IMDS unreachable, etc.).
                        print(f"{agent.log_prefix}   Auth method: Microsoft Entra ID (httpx event hook)")
                        print(f"{agent.log_prefix}   Run `hermes doctor` for credential-chain diagnostics, or")
                        print(f"{agent.log_prefix}   `az login` if your developer session expired.")
                    else:
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{agent.log_prefix}   Auth method: {auth_method}")
                        print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                    print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                    print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                    print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                    print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.34】Thinking block 签名恢复(Anthropic 特有)
                # 背景: Anthropic 给 thinking blocks 签名(基于整个 turn content)
                #       任何上游修改(压缩/截断/消息合并)都会让签名失效
                #       → HTTP 400
                # 恢复: 剥所有 reasoning_details,下次重试不带 thinking blocks
                # 注意: 一次性 — 不能再失败的话就走通用错误处理
                # ── Thinking block signature recovery ─────────────────
                # ═══════════════════════════════════════════════════════════════
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not thinking_sig_retry_attempted
                ):
                    # 4.34.1 标志位 +1
                    thinking_sig_retry_attempted = True
                    # 4.34.2 遍历 messages,剥掉 reasoning_details
                    for _m in messages:
                        if isinstance(_m, dict):
                            _m.pop("reasoning_details", None)
                    # 4.34.3 通知用户
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Thinking block signature invalid — "
                        f"stripped all thinking blocks, retrying...",
                        force=True,
                    )
                    # 4.34.4 写 warning log
                    logger.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d messages",
                        agent.log_prefix, len(messages),
                    )
                    # 4.34.5 continue 重试
                    continue

                # ── Invalid encrypted reasoning replay recovery ───────
                # OpenAI Responses API surfaces (and some compatible relays)
                # return HTTP 400 ``invalid_encrypted_content`` when a
                # replayed ``codex_reasoning_items`` blob from a previous
                # turn fails verification (provider rotated the encryption
                # key, the route doesn't actually persist reasoning state,
                # etc.).  Recovery: disable replay for the rest of the
                # session, strip cached items from history, retry once.
                # One-shot — if a second 400 fires we fall through to the
                # normal retry/backoff path.  Only fires for codex_responses
                # mode with at least one assistant message that has cached
                # ``codex_reasoning_items``; without replay state, the
                # error is unrelated to our cache so the normal retry path
                # handles it (the provider is rejecting something else).
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.35】加密 reasoning replay 失败恢复(Codex 特有)
                # Codex Responses API 走加密 reasoning 回放
                # 如果上游拒绝加密 reasoning 块 → HTTP 400
                # 解决: 禁用 replay 模式 + 剥 codex_reasoning_items
                # 一次性(标志位控制)
                # ═══════════════════════════════════════════════════════════════
                if (
                    classified.reason == FailoverReason.invalid_encrypted_content
                    and not invalid_encrypted_content_retry_attempted
                    and agent.api_mode == "codex_responses"
                    and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
                    and any(
                        isinstance(_m, dict)
                        and _m.get("role") == "assistant"
                        and isinstance(_m.get("codex_reasoning_items"), list)
                        and _m.get("codex_reasoning_items")
                        for _m in messages
                    )
                ):
                    # 4.35.1 标志位 +1
                    invalid_encrypted_content_retry_attempted = True
                    # 4.35.2 禁用 replay + 剥 items(返回剥了多少的 stats)
                    replay_stats = agent._disable_codex_reasoning_replay(messages)
                    # 4.35.3 通知用户
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Encrypted reasoning replay was rejected by the provider — "
                        f"disabled replay and stripped {replay_stats['items']} item(s) from "
                        f"{replay_stats['messages']} message(s), retrying...",
                        force=True,
                    )
                    # 4.35.4 写 log
                    logger.warning(
                        "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
                        agent.log_prefix,
                        replay_stats["items"],
                        replay_stats["messages"],
                    )
                    # 4.35.5 continue
                    continue

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.36】llama.cpp grammar-parse 恢复
                # 背景: llama.cpp 的 json-schema-to-grammar 转换器拒绝
                #       regex 转义类(\d, \w, \s) 和大部分 format 值
                #       MCP server 经常为 date/phone/email 参数发这些
                # 恢复: 剥 agent.tools 里的 pattern/format 后重试
                # 注意: 默认保留 pattern/format(给云端 provider 完整提示)
                #       这个分支只对本地 llama.cpp OAI server 用户触发
                # ── llama.cpp grammar-parse recovery ──────────────────
                # ═══════════════════════════════════════════════════════════════
                if (
                    classified.reason == FailoverReason.llama_cpp_grammar_pattern
                    and not llama_cpp_grammar_retry_attempted
                ):
                    # 4.36.1 标志位 +1
                    llama_cpp_grammar_retry_attempted = True
                    # 4.36.2 调 strip helper 剥 pattern/format
                    try:
                        from tools.schema_sanitizer import strip_pattern_and_format
                        _, _stripped = strip_pattern_and_format(agent.tools)
                    # 4.36.3 strip 失败只 log,继续
                    except Exception as _strip_exc:  # pragma: no cover — defensive
                        logger.warning(
                            "%sllama.cpp grammar recovery: strip helper failed: %s",
                            agent.log_prefix, _strip_exc,
                        )
                        _stripped = 0
                    # 4.36.4 真剥到了 → 通知用户
                    if _stripped:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                            f"stripped {_stripped} pattern/format keyword(s), retrying...",
                            force=True,
                        )
                        logger.warning(
                            "%sllama.cpp grammar recovery: stripped %d "
                            "pattern/format keyword(s) from tool schemas",
                            agent.log_prefix, _stripped,
                        )
                        continue
                    # No keywords found to strip — fall through to normal
                    # retry path rather than loop forever on the same error.
                    logger.warning(
                        "%sllama.cpp grammar error but no pattern/format "
                        "keywords to strip — falling through to normal retry",
                        agent.log_prefix,
                    )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.37】通用错误恢复路径(所有特定恢复都失败后的回退)
                # 走到这里说明: 没切 fallback,没成功刷新凭证,没剥 surrogate...
                # 通用做法: retry_count += 1 → 走 backoff → 重试
                # ═══════════════════════════════════════════════════════════════
                # 4.37.1 重试计数器 +1(让 while 条件有机会终止)
                retry_count += 1
                # 4.37.2 记录累计耗时
                elapsed_time = time.time() - api_start_time
                # 4.37.3 触摸活动计时器
                agent._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.38】错误摘要 + 详细 log
                # ═══════════════════════════════════════════════════════════════
                # 4.38.1 异常类名(如 APIConnectionError / RateLimitError)
                error_type = type(api_error).__name__
                # 4.38.2 错误信息(小写,方便后续模式匹配)
                error_msg = str(api_error).lower()
                # 4.38.3 友好摘要(去掉敏感信息)
                _error_summary = agent._summarize_api_error(api_error)
                # 4.38.4 写 warning log(单行,易 grep)
                logger.warning(
                    "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                    retry_count,
                    max_retries,
                    error_type,
                    agent._client_log_context(),
                    _error_summary,
                )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.39】给用户看诊断信息
                # 4 行: 错误 + provider/model + endpoint + 错误摘要
                # ═══════════════════════════════════════════════════════════════
                _provider = getattr(agent, "provider", "unknown")
                _base = getattr(agent, "base_url", "unknown")
                _model = getattr(agent, "model", "unknown")
                _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                agent._buffer_vprint(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}")
                agent._buffer_vprint(f"   🔌 Provider: {_provider}  Model: {_model}")
                agent._buffer_vprint(f"   🌐 Endpoint: {_base}")
                agent._buffer_vprint(f"   📝 Error: {_error_summary}")
                # 4.39.1 4xx 错误时显示 body 详情(5xx 不显示,可能是瞬时)
                if status_code and status_code < 500:
                    _err_body = getattr(api_error, "body", None)
                    _err_body_str = str(_err_body)[:300] if _err_body else None
                    if _err_body_str:
                        agent._buffer_vprint(f"   📋 Details: {_err_body_str}")
                # 4.39.2 耗时 + 上下文大小(帮助判断是不是 context 太大)
                agent._buffer_vprint(f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.40】OpenRouter "no tool endpoints" 提示
                # 场景: OpenRouter 说该 model 没有支持 tool calling 的 provider
                #       原因可能是 provider_routing.only 限制
                # 触发: 错误信息包含 "support tool use"
                # 跟其他 retry trace 一样 buffered,只在所有重试用完才显示
                # 避免在 fallback 自动恢复时刷屏
                # Actionable hint for OpenRouter "no tool endpoints" error.
                # Buffered like the rest of the retry trace — surfaced only
                # if every retry+fallback exhausts.  Avoids spamming users
                # who recover automatically via fallback.
                # ═══════════════════════════════════════════════════════════════
                if (
                    agent._is_openrouter_url()
                    and "support tool use" in error_msg
                ):
                    agent._buffer_vprint(
                        f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings."
                    )
                    if agent.providers_allowed:
                        agent._buffer_vprint(
                            f"      Your provider_routing.only restriction is filtering out tool-capable providers."
                        )
                        agent._buffer_vprint(
                            f"      Try removing the restriction or adding providers that support tools for this model."
                        )
                    agent._buffer_vprint(
                        f"      Check which providers support tools: https://openrouter.ai/models/{_model}"
                    )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.41】决定重试前先检查用户中断
                # 设计: 错误恢复可能耗时长(backoff),用户在等的时候可能发新消息
                #       优先响应中断,别让用户等
                # Check for interrupt before deciding to retry
                # ═══════════════════════════════════════════════════════════════
                if agent._interrupt_requested:
                    # 4.41.1 通知用户
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    # 4.41.2 persist + 清中断标志
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    # 4.41.3 返回中断状态
                    return {
                        "final_response": f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                # ═══════════════════════════════════════════════════════════════
                # 【学习要点】HTTP 413 Payload Too Large — 压缩重试
                # 为什么 413 必须先于通用 4xx 处理?
                #   → 通用 4xx 处理器会"放弃"或"切 fallback"
                #   → 但 413 是"我的请求太大了",切 provider 也解决不了
                #   → 正确反应: 压缩历史 + 减小 payload + 同一个 provider 重试
                # 触发场景:
                #   - 用户的 history 累计到 >1MB JSON
                #   - 一次性发送大量 image blocks
                #   - tools schema 巨大(自定义了 100+ 工具)
                # 恢复路径:
                #   1. compression_attempts += 1
                #   2. 调 _compress_context() 压缩 messages
                #   3. restart_with_compressed_messages = True (跳出内层 while)
                #   4. 重新构建 api_messages (用压缩后的 messages)
                #   5. 重试
                # max_compression_attempts=3 防无限压缩循环
                # ═══════════════════════════════════════════════════════════════
                # Check for 413 payload-too-large BEFORE generic 4xx handler.
                # A 413 is a payload-size error — the correct response is to
                # compress history and retry, not abort immediately.
                status_code = getattr(api_error, "status_code", None)

                # ═══════════════════════════════════════════════════════════════
                # 【学习要点】Anthropic 1M Context Tier Gate
                # 特殊场景: Claude Max 订阅用户没有购买 1M-context tier
                #   → 发送 >200k token 请求时,Anthropic 返回 HTTP 429
                #   → 错误信息是 "Extra usage is required for long context requests"
                # 为什么不直接重试或切 provider?
                #   → 切其他 provider 模型不一样,响应会变
                #   → 重试同一个 provider 还是 429
                #   → 唯一办法: 把 context 压到 200k 以下(标准 tier 上限)
                # 恢复路径:
                #   1. compressor.context_length = 200000
                #   2. 触发压缩让历史 < 200k
                #   3. 重试
                # 这个 gate 是 Hermes 专门为 Anthropic 订阅档位做的适配
                # → 不影响 OpenAI/Codex/Bedrock(它们没有 tier 概念)
                # ═══════════════════════════════════════════════════════════════
                # ── Anthropic Sonnet long-context tier gate ───────────
                # Anthropic returns HTTP 429 "Extra usage is required for
                # long context requests" when a Claude Max (or similar)
                # subscription doesn't include the 1M-context tier.  This
                # is NOT a transient rate limit — retrying or switching
                # credentials won't help.  Reduce context to 200k (the
                # standard tier) and compress.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.42】Anthropic 1M Context Tier Gate(200k 降级)
                # 触发: 用户的 Anthropic 订阅(Claude Max)不包括 1M-context tier
                # 错误: HTTP 429 "Extra usage is required for long context requests"
                # 关键: 这不是瞬时限流 — 重试/切凭证都无效
                # 解决: 把 context_length 压到 200k(标准 tier 上限)
                #       触发压缩让历史 < 200k
                # 适用范围: 只影响 Anthropic 订阅档位
                # ═══════════════════════════════════════════════════════════════
                if classified.reason == FailoverReason.long_context_tier:
                    # 4.42.1 标准 tier 上限
                    _reduced_ctx = 200000
                    # 4.42.2 拿 compressor
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length
                    # 4.42.3 只在原本 > 200k 时才需要降级
                    if old_ctx > _reduced_ctx:
                        # 4.42.4 更新 compressor 的 context_length
                        compressor.update_model(
                            model=agent.model,
                            context_length=_reduced_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Don't persist — this is a subscription-tier
                            # limitation, not a model capability.  If the
                            # user later enables extra usage the 1M limit
                            # should come back automatically.
                            compressor._context_probe_persistable = False
                        agent._buffer_vprint(
                            f"⚠️  Anthropic long-context tier "
                            f"requires extra usage — reducing context: "
                            f"{old_ctx:,} → {_reduced_ctx:,} tokens"
                        )

                    # 4.42.5 标记不持久化(tier 限制,不是模型能力)
                    # Don't persist — this is a subscription-tier
                    # limitation, not a model capability.  If the
                    # user later enables extra usage the 1M limit
                    # should come back automatically.
                    compressor._context_probe_persistable = False
                    # 4.42.6 通知用户降级
                    agent._buffer_vprint(
                        f"⚠️  Anthropic long-context tier "
                        f"requires extra usage — reducing context: "
                        f"{old_ctx:,} → {_reduced_ctx:,} tokens"
                    )

                    # 4.42.7 压缩计数器 +1
                    compression_attempts += 1
                    # 4.42.8 还没超过压缩上限
                    if compression_attempts <= max_compression_attempts:
                        # 4.42.9 记录原始长度(用于判断是否真压缩了)
                        original_len = len(messages)
                        # 4.42.10 调 _compress_context 压缩历史
                        messages, active_system_prompt = agent._compress_context(
                            messages, system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # 4.42.11 清空 conversation_history(让新 session 写入压缩后的)
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None
                        # 4.42.12 真压缩了 或 降级了 → 标记重启
                        if len(messages) < original_len or old_ctx > _reduced_ctx:
                            # 4.42.13 通知用户
                            agent._buffer_status(
                                f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                f"(was {old_ctx:,}), retrying..."
                            )
                            # 4.42.14 等 2s(给 provider 喘息)
                            time.sleep(2)
                            # 4.42.15 标志位:跳出内层 while 重启
                            restart_with_compressed_messages = True
                            break
                    # Fall through to normal error handling if compression
                    # is exhausted or didn't help.

                # Eager fallback for rate-limit errors (429 or quota exhaustion).
                # When a fallback model is configured, switch immediately instead
                # of burning through retries with exponential backoff -- the
                # primary provider won't recover within the retry window.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.43】限流/账单错误时切换 fallback
                # 触发: classified.reason 是 RATE_LIMIT 或 BILLING
                #       且还有 fallback chain 可用
                # 关键: 不要在凭证池可能自愈时急切切 fallback
                #       (参考 _pool_may_recover_from_rate_limit)
                # ═══════════════════════════════════════════════════════════════
                # 4.43.1 判断是否限流/账单类错误
                is_rate_limited = classified.reason in {
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                }
                # 4.43.2 限流 + 有 fallback chain
                if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
                    # 4.43.3 检查凭证池是否可能自愈
                    # Don't eagerly fallback if credential pool rotation may
                    # still recover.  See _pool_may_recover_from_rate_limit
                    # for the single-credential-pool and CloudCode-quota
                    # exceptions.  Fixes #11314 and #13636.
                    pool_may_recover = _ra()._pool_may_recover_from_rate_limit(
                        agent._credential_pool,
                        provider=agent.provider,
                        base_url=getattr(agent, "base_url", None),
                    )
                    # 4.43.4 池子不能自愈才切
                    if not pool_may_recover:
                        # 4.43.5 区分 billing 和 rate limit 给不同消息
                        if classified.reason == FailoverReason.billing:
                            agent._buffer_status(
                                "⚠️ Billing or credits exhausted — switching to fallback provider..."
                            )
                        else:
                            agent._buffer_status("⚠️ Rate limited — switching to fallback provider...")
                        # 4.43.6 切 fallback(传 reason 给 cooldown 用)
                        if agent._try_activate_fallback(reason=classified.reason):
                            # 4.43.7 重置状态 + continue
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                # ── Nous Portal: record rate limit & skip retries ─────
                # When Nous returns a 429 that is a genuine account-
                # level rate limit, record the reset time to a shared
                # file so ALL sessions (cron, gateway, auxiliary) know
                # not to pile on, then skip further retries -- each
                # one burns another RPH request and deepens the hole.
                # The retry loop's top-of-iteration guard will catch
                # this on the next pass and try fallback or bail.
                #
                # IMPORTANT: Nous Portal multiplexes multiple upstream
                # providers (DeepSeek, Kimi, MiMo, Hermes).  A 429 can
                # also mean an UPSTREAM provider is out of capacity
                # for one specific model -- transient, clears in
                # seconds, nothing to do with the caller's quota.
                # Tripping the cross-session breaker on that would
                # block every Nous model for minutes.  We use
                # ``is_genuine_nous_rate_limit`` to tell the two
                # apart via the 429's own x-ratelimit-* headers and
                # the last-known-good state captured on the previous
                # successful response.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.44】Nous 跨 session 限流记录
                # 设计: Nous 是 portal,多 upstream 共享(DeepSeek/Kimi/MiMo/Hermes)
                #       429 可能含义:
                #         a) 真正账户级限流(用户 quota 用完)
                #         b) 某个 upstream provider 临时满载(几秒就好)
                #       区分: 用 x-ratelimit-* headers + last-known-good 状态
                #       误触发会阻塞所有 Nous model 数分钟,所以要小心
                # ═══════════════════════════════════════════════════════════════
                if (
                    is_rate_limited
                    and agent.provider == "nous"
                    and classified.reason == FailoverReason.rate_limit
                    and not recovered_with_pool
                ):
                    # 4.44.1 默认不是 genuine(默认不信)
                    _genuine_nous_rate_limit = False
                    # 4.44.2 调 is_genuine_nous_rate_limit 区分两种 429
                    try:
                        from agent.nous_rate_guard import (
                            is_genuine_nous_rate_limit,
                            record_nous_rate_limit,
                        )
                        # 4.44.3 拿错误响应的 headers(用于判断 genuine)
                        _err_resp = getattr(api_error, "response", None)
                        _err_hdrs = (
                            getattr(_err_resp, "headers", None)
                            if _err_resp else None
                        )
                        # 4.44.4 用 headers + last_known_state 判断
                        _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                            headers=_err_hdrs,
                            last_known_state=agent._rate_limit_state,
                        )
                        # 4.44.5 真正账户级限流 → 记录到共享状态
                        if _genuine_nous_rate_limit:
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        # 4.44.6 upstream 容量 429 → 不触发跨 session breaker
                        else:
                            logger.info(
                                "Nous 429 looks like upstream capacity "
                                "(no exhausted bucket in headers or "
                                "last-known state) -- not tripping "
                                "cross-session breaker."
                            )
                    # 4.44.7 检测失败不报错
                    except Exception:
                        pass
                    # 4.44.8 真正账户级限流 → 直接跳到 max_retries
                    if _genuine_nous_rate_limit:
                        # Skip straight to max_retries -- the
                        # top-of-loop guard will handle fallback or
                        # bail cleanly.
                        retry_count = max_retries
                        continue
                    # 4.44.9 upstream 容量 429 → 走普通重试(别的 model 可能成功)
                    # Upstream capacity 429: fall through to normal
                    # retry logic.  A different model (or the same
                    # model a moment later) will typically succeed.

                is_payload_too_large = (
                    classified.reason == FailoverReason.payload_too_large
                )

                # Actionable hint for GitHub Models (Azure) 413 errors.
                # The free tier enforces a hard 8K token cap per request,
                # which Hermes' system prompt + tool schemas alone exceed.
                # Compression can't help — the floor is the system prompt
                # itself, not the conversation — so surface a clear "not
                # compatible" message instead of looping into three futile
                # compression attempts.
                if (
                    status_code == 413
                    and isinstance(agent.base_url, str)
                    and "models.inference.ai.azure.com" in agent.base_url
                ):
                    agent._vprint(
                        f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
                        force=True,
                    )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.45】HTTP 413 Payload Too Large — 压缩重试
                # 为什么 413 必须先于通用 4xx 处理?
                #   → 通用 4xx 处理器会"放弃"或"切 fallback"
                #   → 但 413 是"我的请求太大了",切 provider 也解决不了
                #   → 正确反应: 压缩历史 + 减小 payload + 同一个 provider 重试
                # ═══════════════════════════════════════════════════════════════
                if is_payload_too_large:
                    # 4.45.1 压缩计数器 +1
                    compression_attempts += 1
                    # 4.45.2 超过压缩上限 → 真的放弃了
                    if compression_attempts > max_compression_attempts:
                        # Terminal — surface the buffered retry trace.
                        # 4.45.2.1 刷出 buffered 状态
                        agent._flush_status_buffer()
                        # 4.45.2.2 通知用户达到上限
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                        # 4.45.2.3 给建议
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        # 4.45.2.4 写 error log
                        logger.error(f"{agent.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                        # 4.45.2.5 persist session
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    # 4.45.3 通知用户正在尝试压缩
                    agent._buffer_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                    # 4.45.4 记录压缩前长度(对比是否真压缩了)
                    original_len = len(messages)
                    # 4.45.5 调 _compress_context 压缩
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # 4.45.6 清空 conversation_history(让新 session 写入压缩后的)
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    conversation_history = None

                    # 4.45.7 真压缩了 → 重启
                    if len(messages) < original_len:
                        # 4.45.7.1 通知用户
                        agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        # 4.45.7.2 等 2s(给 provider 喘息)
                        time.sleep(2)  # Brief pause between compression retries
                        # 4.45.7.3 标志位 + break 跳出内层 while
                        restart_with_compressed_messages = True
                        break
                    # 4.45.8 没压缩成 → 真的放弃
                    else:
                        # Terminal — surface buffered context so the user
                        # sees what compression attempts were made.
                        # 4.45.8.1 刷出 buffered 状态
                        agent._flush_status_buffer()
                        # 4.45.8.2 通知用户
                        agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                        # 4.45.8.3 给建议
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        # 4.45.8.4 写 error log
                        logger.error(f"{agent.log_prefix}413 payload too large. Cannot compress further.")
                        # 4.45.8.5 persist
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": "Request payload too large (413). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.46】Context-Length 错误处理(必须在通用 4xx 之前)
                # 检测: 显式错误信息 / 通用 400 + 大 session 启发 / 服务器断连 + 大 session
                # 关键: 区分两种完全不同的错误:
                #   1. "Prompt too long" — 输入超 context 窗口
                #      解决: 减小 context_length + 压缩历史
                #   2. "max_tokens too large" — 输入没问题,但 input + requested_max > window
                #      解决: 减小 max_tokens(输出上限),不动 context_length
                # 注意:
                #   max_tokens       = 单次响应的输出 token 上限
                #   context_length   = 总窗口(input + output 合计)
                # ═══════════════════════════════════════════════════════════════
                # Check for context-length errors BEFORE generic 4xx handler.
                # The classifier detects context overflow from: explicit error
                # messages, generic 400 + large session heuristic (#1630), and
                # server disconnect + large session pattern (#2153).
                # 4.46.1 判断是否是 context overflow 错误
                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )

                if is_context_length_error:
                    # 4.46.2 拿 compressor 和当前 context 长度
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length

                    # ── Distinguish two very different errors ───────────
                    # 1. "Prompt too long": the INPUT exceeds the context window.
                    #    Fix: reduce context_length + compress history.
                    # 2. "max_tokens too large": input is fine, but
                    #    input_tokens + requested max_tokens > context_window.
                    #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                    #    Do NOT shrink context_length — the window is unchanged.
                    #
                    # Note: max_tokens = output token cap (one response).
                    #       context_length = total window (input + output combined).
                    # 4.46.3 从错误信息解析可用输出空间
                    available_out = parse_available_output_tokens_from_error(error_msg)
                    if available_out is not None:
                        # Error is purely about the output cap being too large.
                        # Cap output to the available space and retry without
                        # touching context_length or triggering compression.
                        safe_out = max(1, available_out - 64)  # small safety margin
                        agent._ephemeral_max_output_tokens = safe_out
                        agent._buffer_vprint(
                            f"⚠️  Output cap too large for current prompt — "
                            f"retrying with max_tokens={safe_out:,} "
                            f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})"
                        )
                        # 4.46.4 压缩计数(防止无限循环)
                        # Still count against compression_attempts so we don't
                        # loop forever if the error keeps recurring.
                        compression_attempts += 1
                        # 4.46.5 压缩超限 → 真的放弃
                        if compression_attempts > max_compression_attempts:
                            agent._flush_status_buffer()
                            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            agent._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        # 4.46.6 标记重启 + break
                        restart_with_compressed_messages = True
                        break

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.47】第二种错误:输入超 context
                    # 关键: 只在 provider 显式报告更小限制时才减小 context_length
                    #       不能猜探测 tier(可能把 1M 窗口错猜成 256K/128K/64K)
                    # Error is about the INPUT being too large.  Only reduce
                    # context_length when the provider explicitly reports the
                    # real lower limit.  If the provider only says "input
                    # exceeds the context window", keep the configured window
                    # and try compression; guessing probe tiers can incorrectly
                    # turn a user-configured 1M window into 256K/128K/64K.
                    # ═══════════════════════════════════════════════════════════════
                    # 4.47.1 从错误信息解析 provider 报告的实际限制
                    new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
                    _provider_lower = (getattr(agent, "provider", "") or "").lower()
                    _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.48】特殊 provider 适配(MiniMax)
                    # MiniMax 的错误信息是特殊的 "context window exceeds limit"
                    # 需要用 minimax_delta_only_overflow 标志单独处理
                    # ═══════════════════════════════════════════════════════════════
                    # 4.48.1 判断是否是 MiniMax provider
                    is_minimax_provider = (
                        _provider_lower in {"minimax", "minimax-cn"}
                        or _base_lower.startswith((
                            "https://api.minimax.io/anthropic",
                            "https://api.minimaxi.com/anthropic",
                        ))
                    )
                    # 4.48.2 MiniMax 特殊错误模式
                    minimax_delta_only_overflow = (
                        is_minimax_provider
                        and new_ctx is None
                        and "context window exceeds limit (" in error_msg
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.49】Provider 报告了实际限制 → 更新 compressor
                    # ═══════════════════════════════════════════════════════════════
                    # 4.49.1 有限制可更新
                    if new_ctx is not None:
                        # 4.49.1.1 通知用户检测到限制
                        agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
                        # 4.49.1.2 更新 compressor
                        compressor.update_model(
                            model=agent.model,
                            context_length=new_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).  This
                        # value came from the provider, so it is safe to cache.
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            compressor._context_probe_persistable = True
                        agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
                    elif minimax_delta_only_overflow:
                        agent._buffer_vprint(
                            f"Provider reported overflow amount only; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )
                    else:
                        agent._buffer_vprint(
                            f"⚠️  Context length exceeded, but provider did not report a max context length; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.50】压缩控制(防止无限压缩循环)
                    # ═══════════════════════════════════════════════════════════════
                    # 4.50.1 压缩计数 +1
                    compression_attempts += 1
                    # 4.50.2 超过 max_compression_attempts → 放弃
                    if compression_attempts > max_compression_attempts:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    # 4.50.3 通知用户正在压缩
                    agent._buffer_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                    # 4.50.4 记录原始长度(对比压缩效果)
                    original_len = len(messages)
                    # 4.50.5 调 _compress_context
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    conversation_history = None

                    if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                        if len(messages) < original_len:
                            agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries
                        restart_with_compressed_messages = True
                        break
                    else:
                        # Can't compress further and already at minimum tier
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                        logger.error(f"{agent.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.51】检查不可重试的客户端错误
                # 分类器已经处理了: 413, 429, 529 (瞬时), context overflow
                # 通用 400 启发
                # 本地校验错误 (ValueError, TypeError) 是编程 bug
                # 例外:
                #   - UnicodeEncodeError → 上面 surrogate 路径处理
                #   - json.JSONDecodeError → 瞬时 provider/网络故障
                #                            (响应体畸形、流截断、路由损坏)
                #                            应该重试 (#14782)
                # Check for non-retryable client errors.  The classifier
                # already accounts for 413, 429, 529 (transient), context
                # overflow, and generic-400 heuristics.  Local validation
                # errors (ValueError, TypeError) are programming bugs.
                # Exclude UnicodeEncodeError — it's a ValueError subclass
                # but is handled separately by the surrogate sanitization
                # path above.  Exclude json.JSONDecodeError — also a
                # ValueError subclass, but it indicates a transient
                # provider/network failure (malformed response body,
                # truncated stream, routing layer corruption), not a
                # local programming bug, and should be retried (#14782).
                # ═══════════════════════════════════════════════════════════════
                # 4.51.1 判断是否是本地校验错误
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(
                        api_error, (UnicodeEncodeError, json.JSONDecodeError)
                    )
                    # ssl.SSLError (and its subclass SSLCertVerificationError)
                    # inherits from OSError *and* ValueError via Python MRO,
                    # so the isinstance(ValueError) check above would
                    # misclassify a TLS transport failure as a local
                    # programming bug and abort without retrying.  Exclude
                    # ssl.SSLError explicitly so the error classifier's
                    # retryable=True mapping takes effect instead.
                    and not isinstance(api_error, ssl.SSLError)
                    # Provider/SDK "NoneType is not iterable" failures are
                    # shape mismatches from upstream (e.g. chatgpt.com Codex
                    # backend response.completed.output=null) — not local
                    # programming bugs.  Even after #33042 made our own
                    # consumer immune, third-party shims and mocked clients
                    # can still surface this shape via TypeError.  Treat
                    # them as retryable so the error classifier's normal
                    # retry/fallback path runs instead of killing the turn
                    # as non-retryable (which left Telegram users staring
                    # at a bare "Non-retryable error" with no recovery).
                    and not (
                        isinstance(api_error, TypeError)
                        and "nonetype" in str(api_error).lower()
                        and "not iterable" in str(api_error).lower()
                    )
                )
                # ``FailoverReason.billing`` (HTTP 402) is NOT in this
                # exclusion set.  By the time we reach this block:
                #   • credential-pool rotation (line ~2031) has already
                #     fired for billing and either ``continue``d or
                #     returned (False, ...) — pool is exhausted or absent.
                #   • the eager-fallback branch above (line ~2422) also
                #     fires on billing and ``continue``s if a fallback
                #     provider is configured.
                # Falling through to here means BOTH recovery paths
                # gave up.  Treating 402 as retryable from this point
                # just burns more paid requests against a depleted
                # balance with no recovery mechanism left — see #31273
                # (real-world: ~$40 in 48h on a 24/7 gateway).  Aborting
                # mirrors how 401/403 (also ``should_fallback=True``)
                # already behave once their recovery paths have failed.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.52】判断是否是 fatal 客户端错误(不能再重试)
                # 走到这里说明: 池恢复没了 + eager fallback 没了 + 压缩没了
                # 唯一的合理选择: 终止 + 提示用户
                # FailoverReason.billing (HTTP 402) 不在 retryable 集合
                # 因为到这里时所有恢复路径都用完了(参见 #31273)
                # ═══════════════════════════════════════════════════════════════
                # 4.52.1 判断客户端错误条件
                is_client_error = (
                    is_local_validation_error
                    or (
                        # 不可重试 + 不能压缩 + 不是已知瞬时错误
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in {
                            FailoverReason.rate_limit,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        }
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # Try fallback before aborting — a different provider may
                    # not have the same issue (rate limit, auth, etc.). Only
                    # announce the attempt when a fallback chain actually
                    # exists; otherwise "trying fallback..." is a lie and the
                    # session looks like it's recovering when it's about to
                    # abort silently (#35314, #17446).
                    if agent._has_pending_fallback():
                        if classified.reason == FailoverReason.content_policy_blocked:
                            agent._buffer_status("⚠️ Provider safety filter blocked this request — trying fallback...")
                    # 4.52.2 还有 fallback → 最后试一次
                    if is_client_error and agent._has_pending_fallback():
                        if classified.reason == FailoverReason.content_policy_blocked:
                            agent._buffer_status(f"⚠️ Provider safety filter — trying fallback...")
                        else:
                            agent._buffer_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    # 4.52.3 激活 fallback
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    # 4.52.4 dump 请求(给开发者调试)
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                    # 4.52.5 终止 - 刷出 buffered 状态
                    # Terminal — flush buffered context so the user sees
                    # what was tried before the abort.
                    agent._flush_status_buffer()
                    # 4.52.6 区分内容安全过滤 vs 其他
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._emit_status(
                            f"❌ Provider safety filter blocked this request: "
                            f"{agent._summarize_api_error(api_error)}"
                        )
                    else:
                        agent._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{agent._summarize_api_error(api_error)}"
                        )
                    agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                    agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    # Actionable guidance for common auth errors
                    if classified.is_auth or classified.reason == FailoverReason.billing:
                        if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        ):
                            pass
                        elif _provider == "nous" and _print_nous_entitlement_guidance(
                            agent,
                            "Nous model access",
                        ):
                            pass
                        elif _provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
                            if _provider == "openai-codex":
                                agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            elif _provider == "xai-oauth":
                                agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.", force=True)
                            else:  # nous
                                agent._vprint(f"{agent.log_prefix}   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be", force=True)
                                agent._vprint(f"{agent.log_prefix}      expired, revoked, or your account may be out of credits. To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Re-authenticate: hermes auth add nous --type oauth", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Check your portal account: https://portal.nousresearch.com", force=True)
                                # ``:free`` is OpenRouter slug syntax; Nous Portal will reject
                                # the model name even after a successful re-auth.
                                if isinstance(_model, str) and _model.endswith(":free"):
                                    agent._vprint(f"{agent.log_prefix}      ⚠️  Note: `{_model}` looks like an OpenRouter slug (`:free` suffix).", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous Portal won't recognize that model name. Either switch to a", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous catalog model, or run `/model openrouter:{_model}` to use OpenRouter.", force=True)
                        else:
                            agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Does your account have access to {_model}?", force=True)
                            if base_url_host_matches(str(_base), "openrouter.ai"):
                                agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                    # Content-policy blocks deserve their own actionable
                    # guidance — neither "fix your API key" nor "retry won't
                    # help" tells the user what to actually do. The provider
                    # has refused this specific prompt, so the recovery is
                    # either a rephrase or routing to a different model.
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's safety filter rejected this specific prompt.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Try rephrasing the request, narrowing the context, or splitting into smaller steps.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Configure a fallback provider so future blocks route automatically:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        hermes fallback add   (interactive picker — same as `hermes model`)",
                            force=True,
                        )
                    # 4.53.4 写 error log
                    logger.error(f"{agent.log_prefix}Non-retryable client error: {api_error}")
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.54】决定是否 persist session
                    # 特殊情况: 400 + 大 session → 不 persist
                    #   (写盘会让 session 更大,下次还会失败 #1630)
                    # Skip session persistence when the error is likely
                    # context-overflow related (status 400 + large session).
                    # Persisting the failed user message would make the
                    # session even larger, causing the same failure on the
                    # next attempt. (#1630)
                    # ═══════════════════════════════════════════════════════════════
                    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                        # 4.54.1 通知不 persist
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Skipping session persistence "
                            f"for large failed session to prevent growth loop.",
                            force=True,
                        )
                    # 4.54.2 其他情况正常 persist
                    else:
                        agent._persist_session(messages, conversation_history)
                    if classified.reason == FailoverReason.content_policy_blocked:
                        _summary = agent._summarize_api_error(api_error)
                        _policy_response = (
                            f"⚠️  The model provider's safety filter blocked this request "
                            f"(not a Hermes/gateway failure).\n\n"
                            f"Provider message: {_summary}\n\n"
                            f"Try rephrasing the request, narrowing the context, or "
                            f"adding a fallback provider with `hermes fallback add`."
                        )
                        return {
                            "final_response": _policy_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": f"content_policy_blocked: {_summary}",
                        }
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": str(api_error),
                    }

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.55】重试次数用尽后的最终决策
                # 决策树:
                #   1. 试 primary 恢复(重建 client,处理 stale connection/TCP reset)
                #   2. 试 fallback
                #   3. 都没用 → 终止
                # ═══════════════════════════════════════════════════════════════
                if retry_count >= max_retries:
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.55.1】尝试 primary 恢复
                    # 设计: 瞬时传输错误(stale 连接池/TCP reset)可以通过重建 client 修复
                    # 每个 API call 块只尝试一次(标志位控制)
                    # Before falling back, try rebuilding the primary
                    # client once for transient transport errors (stale
                    # connection pool, TCP reset).  Only attempted once
                    # per API call block.
                    # ═══════════════════════════════════════════════════════════════
                    if not primary_recovery_attempted and agent._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        # 4.55.1.1 标志位置 True
                        primary_recovery_attempted = True
                        # 4.55.1.2 重置 retry + continue
                        retry_count = 0
                        continue
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.55.2】尝试 fallback
                    # Try fallback before giving up entirely
                    # ═══════════════════════════════════════════════════════════════
                    # 4.55.2.1 有 pending fallback → 通知用户
                    if agent._has_pending_fallback():
                        agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    # 4.55.2.2 激活 fallback
                    if agent._try_activate_fallback():
                        # 4.55.2.3 重置 + continue
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.55.3】所有路径用尽 → 终止
                    # Terminal — flush buffered retry/fallback trace.
                    # ═══════════════════════════════════════════════════════════════
                    agent._flush_status_buffer()
                    _final_summary = agent._summarize_api_error(api_error)
                    _billing_guidance = ""
                    if classified.reason == FailoverReason.billing:
                        agent._emit_status(f"❌ Billing or credits exhausted — {_final_summary}")
                        _billing_guidance = _billing_or_entitlement_message(
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                        _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                    elif is_rate_limited:
                        agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                    else:
                        agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                    agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.56】检测 SSE 流中断模式 + 给可操作建议
                    # 场景: 模型生成超大 tool_call(write_file 大内容)→ proxy/CDN 断流
                    # 错误: "Network connection lost" 等(无 status_code)
                    # 建议: 用 execute_code 配合 Python open() 写大文件,或分批写
                    # Detect SSE stream-drop pattern (e.g. "Network
                    # connection lost") and surface actionable guidance.
                    # This typically happens when the model generates a
                    # very large tool call (write_file with huge content)
                    # and the proxy/CDN drops the stream mid-response.
                    # ═══════════════════════════════════════════════════════════════
                    # 4.56.1 检测流中断(无 status_code + 6 个错误短语)
                    _is_stream_drop = (
                        not getattr(api_error, "status_code", None)
                        and any(p in error_msg for p in (
                            "connection lost", "connection reset",
                            "connection closed", "network connection",
                            "network error", "terminated",
                        ))
                    )
                    # 4.56.2 流中断 → 给具体建议
                    if _is_stream_drop:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's stream "
                            f"connection keeps dropping. This often happens "
                            f"when the model tries to write a very large "
                            f"file in a single tool call.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Try asking the model "
                            f"to use execute_code with Python's open() for "
                            f"large files, or to write the file in smaller "
                            f"sections.",
                            force=True,
                        )

                    logger.error(
                        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                        agent.log_prefix, max_retries, _final_summary,
                        _provider, _model, len(api_messages), f"{approx_tokens:,}",
                    )
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                    # 4.56.3 persist session(失败也存,让用户能 continue)
                    agent._persist_session(messages, conversation_history)
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.57】构造最终失败响应
                    # 区分 billing / 普通失败 / 流中断 给不同消息
                    # ═══════════════════════════════════════════════════════════════
                    if classified.reason == FailoverReason.billing:
                        # 4.57.1 billing 错误 → 加 entitlement 提示
                        _final_response = f"Billing or credits exhausted: {_final_summary}"
                        if _billing_guidance:
                            _final_response += f"\n\n{_billing_guidance}"
                    else:
                        # 4.57.2 普通失败
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                    # 4.57.3 流中断 → 加流中断建议
                    if _is_stream_drop:
                        _final_response += (
                            "\n\nThe provider's stream connection keeps "
                            "dropping — this often happens when generating "
                            "very large tool call responses (e.g. write_file "
                            "with long content). Try asking me to use "
                            "execute_code with Python's open() for large "
                            "files, or to write in smaller sections."
                        )
                    # 4.57.4 返回失败
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _final_summary,
                    }

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.58】还没达到 max_retries → 退避
                # 限流时优先尊重 Retry-After header(标准 HTTP 行为)
                # For rate limits, respect the Retry-After header if present
                # ═══════════════════════════════════════════════════════════════
                # 4.58.1 初始化 retry_after(用于限流的精确退避)
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                _retry_after = min(float(_ra_raw), 120)  # Cap at 2 minutes
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                if is_rate_limited:
                    agent._buffer_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                else:
                    agent._buffer_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    agent._client_log_context(),
                    api_error,
                )
                # Sleep in small increments so we can respond to interrupts quickly
                # instead of blocking the entire wait_time in one sleep() call
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if agent._interrupt_requested:
                        agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        agent._persist_session(messages, conversation_history)
                        agent.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # Check interrupt every 200ms
                    # Touch activity every ~30s so the gateway's inactivity
                    # monitor knows we're alive during backoff waits.
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                        agent._touch_activity(
                            f"error retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
        
        # If the API call was interrupted, skip response processing
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.59】处理"压缩后重试"标记(节省预算)
        # 触发场景: 上一次 API 报 413(请求体过大)→ 压缩了 messages
        #           这一次重新发 API,成功了
        # 关键设计:
        #   1. api_call_count -= 1  → 把这次 API 调用"不算数"
        #      (压缩本身不算 turn,用户感受不到)
        #   2. iteration_budget.refund() → 退回 budget
        #   3. retry_count += 1 → 防止无限压缩循环
        #      (如果压缩后还是 413,继续重试只是浪费)
        # ═══════════════════════════════════════════════════════════════
        if restart_with_compressed_messages:
            # 4.59.1 API 调用计数 -1(这次不算)
            api_call_count -= 1
            # 4.59.2 退回 budget(本来快用完了,压缩救了回来)
            agent.iteration_budget.refund()
            # Count compression restarts toward the retry limit to prevent
            # infinite loops when compression reduces messages but not enough
            # to fit the context window.
            # 4.59.3 重试计数 +1(计入 retry_count,避免无限压缩循环)
            retry_count += 1
            # 4.59.4 清除压缩标记
            restart_with_compressed_messages = False
            continue

        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.60】处理"长度续写"标记(放大 max_tokens 重试)
        # 触发场景: 上一次 API finish_reason="length"(输出被截断)
        #           这一轮要续写,所以要把 max_tokens 调大
        # 关键设计:
        #   _boost = base × (重试次数 + 1)
        #     retry 1: 2× base
        #     retry 2: 3× base
        #   _boost_cap = max(32768, _requested_cap)
        #     至少 32k 兜底,如果 provider 原本请求更大就保留
        # 写入 _ephemeral_max_output_tokens(临时值,不影响 agent.max_tokens)
        # ═══════════════════════════════════════════════════════════════
        if restart_with_length_continuation:
            # 4.60.1 计算基础值: agent.max_tokens 或默认 4096
            _boost_base = agent.max_tokens if agent.max_tokens else 4096
            # 4.60.2 计算 boost 倍数: 1,2,3,...
            _boost = _boost_base * (length_continue_retries + 1)
            # Progressively boost the output token budget on each retry.
            # Retry 1 → 2× base, retry 2 → 3× base, capped at 32 768.
            # Applies to all providers via _ephemeral_max_output_tokens.
            # If the original request already used a larger provider/model
            # default budget, keep that floor so continuation retries do
            # not accidentally downshift to a much smaller cap.
            # 4.60.3 取出请求里原本的 max_tokens 上限(如果有)
            _requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
            # 4.60.4 如果有请求上限,确保 boost 不小于它
            if _requested_cap is not None:
                _boost = max(_boost, _requested_cap)
            # 4.60.5 计算最终上限: 至少 32k 或请求上限
            _boost_cap = max(32768, _requested_cap or 0)
            # 4.60.6 写入临时 max_tokens(只影响这一次重试)
            agent._ephemeral_max_output_tokens = min(_boost, _boost_cap)
            continue

        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.61】安全网: response 是 None(理论不该到这里)
        # 触发条件: 重试次数用尽 + 一直没成功响应
        #          (例如一直报 context length 错误,直到 retry_count 用完)
        # 正常情况下 retry 循环退出时 response 一定有值
        # 这一段是给异常路径的最后兜底
        # ═══════════════════════════════════════════════════════════════
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
            agent._persist_session(messages, conversation_history)
            break

        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.62】归一化 response(transport 层抽象)
        # 关键设计: 不同 provider 返回的 response 字段名不一样
        #   - OpenAI:    response.choices[0].message.tool_calls
        #   - Anthropic: response.content[0].input
        #   - Codex:     response.output[*]
        # transport.normalize_response 把它统一成统一格式
        # strip_tool_prefix: Anthropic OAuth 需要去掉 "functions." 前缀
        # ═══════════════════════════════════════════════════════════════
        try:
            # 4.62.1 取出 transport(每个 provider 一个 transport 实现)
            _transport = agent._get_transport()
            # 4.62.2 准备归一化参数
            _normalize_kwargs = {}
            # 4.62.3 Anthropic OAuth 需要剥 tool prefix
            if agent.api_mode == "anthropic_messages":
                _normalize_kwargs["strip_tool_prefix"] = agent._is_anthropic_oauth
            # 4.62.4 调用 transport 的归一化方法
            normalized = _transport.normalize_response(response, **_normalize_kwargs)
            # 4.62.5 assistant_message 现在是统一格式
            assistant_message = normalized
            # 4.62.6 取出 finish_reason(stop / length / tool_calls 等)
            finish_reason = normalized.finish_reason

            
            # Normalize content to string — some OpenAI-compatible servers
            # (llama-server, etc.) return content as a dict or list instead
            # of a plain string, which crashes downstream .strip() calls.
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.63】content 强制转 str(防御 OpenAI 兼容服务器的怪格式)
            # 触发场景: llama-server 等非标准 OpenAI 兼容服务
            #           可能返回 content 是 dict / list(不是 str)
            # 如果不转,后续 .strip() 会 AttributeError
            # 兼容 3 种格式:
            #   1. dict:  抽 .text 字段 / .content 字段 / 整个 json.dumps
            #   2. list:  多模态,逐项抽 text(Multimodal content)
            #   3. 其他:  str() 强转
            # ═══════════════════════════════════════════════════════════════
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                # 4.63.1 取出原始 content
                raw = assistant_message.content
                # 4.63.2 如果是 dict → 取 text / content / 全 dump
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                # 4.63.3 如果是 list → 多模态,逐项抽 text
                elif isinstance(raw, list):
                    # Multimodal content list — extract text parts
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                # 4.63.4 其他类型 → str() 强转
                else:
                    assistant_message.content = str(raw)

            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.64】post_api_request 插件钩子(给插件看响应)
            # 触发时机: API 响应成功归一化之后
            # 用途: 插件可以审计 / 记录 / 修改响应
            # 失败处理: 任何异常都吞掉,不打扰主流程
            # ═══════════════════════════════════════════════════════════════
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                # 4.64.1 取出 tool_calls 和文本长度
                _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                _assistant_text = assistant_message.content or ""
                # 4.64.2 触发钩子(传一堆上下文)
                _invoke_hook(
                    "post_api_request",
                    task_id=effective_task_id,
                    session_id=agent.session_id or "",
                    platform=agent.platform or "",
                    model=agent.model,
                    provider=agent.provider,
                    base_url=agent.base_url,
                    api_mode=agent.api_mode,
                    api_call_count=api_call_count,
                    api_duration=api_duration,
                    finish_reason=finish_reason,
                    message_count=len(api_messages),
                    response_model=getattr(response, "model", None),
                    response=response,
                    usage=agent._usage_summary_for_api_request_hook(response),
                    assistant_message=assistant_message,
                    assistant_content_chars=len(_assistant_text),
                    assistant_tool_call_count=len(_assistant_tool_calls),
                )
            # 4.64.3 钩子失败不打断
            except Exception:
                pass


            # Handle assistant response
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.65】打印 assistant 响应(非 quiet 模式)
            # verbose 模式:打印完整内容
            # 普通模式:截断到 100 字符(避免刷屏)
            # quiet 模式:完全不打印
            # ═══════════════════════════════════════════════════════════════
            if assistant_message.content and not agent.quiet_mode:
                # 4.65.1 verbose 模式 → 完整内容
                if agent.verbose_logging:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")
                # 4.65.2 普通模式 → 截断 100 字符
                else:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Notify progress callback of model's thinking (used by subagent
            # delegation to relay the child's reasoning to the parent display).
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.66】把 model 的推理文本转发给 progress callback
            # 主要场景: subagent 委派(子代理要推 reasoning 给父代理显示)
            # 关键步骤:
            #   1. 去掉 REASONING_SCRATCHPAD/think/reasoning 等 XML 标签
            #   2. 提取首行(80 字符)用于 subagent 转发
            #   3. 提取前 500 字符用于 reasoning.available 事件
            # ═══════════════════════════════════════════════════════════════
            if (assistant_message.content and agent.tool_progress_callback):
                # 4.66.1 取 content 并 strip
                _think_text = assistant_message.content.strip()
                # Strip reasoning XML tags that shouldn't leak to parent display
                # 4.66.2 剥掉 XML 标签(REASONING_SCRATCHPAD/think/reasoning)
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # For subagents: relay first line to parent display (existing behaviour).
                # For all agents with a structured callback: emit reasoning.available event.
                # 4.66.3 提取首行(80 字符)
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                # 4.66.4 subagent → 推首行给父代理("_thinking" 事件)
                if first_line and getattr(agent, '_delegate_depth', 0) > 0:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                # 4.66.5 非 subagent 但有 callback → reasoning.available 事件
                elif _think_text:
                    try:
                        agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass

            # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
            # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.67】检测未闭合的 REASONING_SCRATCHPAD(模型推理到一半没钱了)
            # 触发场景: model 写了 <REASONING_SCRATCHPAD> 但没写 </REASONING_SCRATCHPAD>
            #          → 推理写到一半,max_tokens 用完被截断
            # 防御:
            #   - 最多重试 2 次
            #   - 重试时不写 messages(只重发 API)
            #   - 重试用完 → 标 partial,回滚到上一个 assistant 消息
            # ═══════════════════════════════════════════════════════════════
            if has_incomplete_scratchpad(assistant_message.content or ""):
                # 4.67.1 不完整 scratchpad 计数 +1
                agent._incomplete_scratchpad_retries += 1

                agent._buffer_vprint(f"⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")

                # 4.67.2 没超 2 次 → 重试
                if agent._incomplete_scratchpad_retries <= 2:
                    agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
                    # Don't add the broken message, just retry
                    continue
                # 4.67.3 超过 2 次 → 终止 turn
                else:
                    # Max retries - discard this turn and save as partial
                    # 4.67.4 刷出状态 buffer(让用户看到重试历史)
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    # 4.67.5 重置计数器
                    agent._incomplete_scratchpad_retries = 0

                    # 4.67.6 回滚到上一个 assistant 消息(丢弃这次坏响应)
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                    # 4.67.7 清理 task 资源
                    agent._cleanup_task_resources(effective_task_id)
                    # 4.67.8 persist session(部分结果也保存)
                    agent._persist_session(messages, conversation_history)

                    return {
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }

            
            # Reset incomplete scratchpad counter on clean response
            agent._incomplete_scratchpad_retries = 0

            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.68】Codex Responses API 的 "incomplete" 处理
            # 触发条件: agent.api_mode == "codex_responses" 且 finish_reason == "incomplete"
            # 这是 OpenAI Codex Responses API 特有的中间状态
            # 含义: model 输出了一部分,需要"继续"
            # 防御: 最多继续 2 次(防止无限续写)
            # ═══════════════════════════════════════════════════════════════
            if agent.api_mode == "codex_responses" and finish_reason == "incomplete":

                agent._codex_incomplete_retries += 1

                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                interim_has_content = bool((interim_msg.get("content") or "").strip())
                interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
                interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

                if (
                    interim_has_content
                    or interim_has_reasoning
                    or interim_has_codex_reasoning
                    or interim_has_codex_message_items
                ):
                    last_msg = messages[-1] if messages else None
                    # Duplicate detection: two consecutive incomplete assistant
                    # messages with identical content AND reasoning are collapsed.
                    # For provider-state-only changes (encrypted reasoning
                    # items or replayable message ids/phases/statuses differ
                    # while visible content/reasoning are unchanged), compare
                    # those opaque payloads too so we don't silently drop the
                    # newer continuation state.
                    last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                    interim_codex_items = interim_msg.get("codex_reasoning_items")
                    last_codex_message_items = last_msg.get("codex_message_items") if isinstance(last_msg, dict) else None
                    interim_codex_message_items = interim_msg.get("codex_message_items")
                    duplicate_interim = (
                        isinstance(last_msg, dict)
                        and last_msg.get("role") == "assistant"
                        and last_msg.get("finish_reason") == "incomplete"
                        and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                        and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        and last_codex_items == interim_codex_items
                        and last_codex_message_items == interim_codex_message_items
                    )
                    if not duplicate_interim:
                        messages.append(interim_msg)
                        agent._emit_interim_assistant_message(interim_msg)

                if agent._codex_incomplete_retries < 3:
                    if not agent.quiet_mode:
                        agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({agent._codex_incomplete_retries}/3)")
                    agent._session_messages = messages
                    continue

                agent._codex_incomplete_retries = 0
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Codex response remained incomplete after 3 continuation attempts",
                }
            elif hasattr(agent, "_codex_incomplete_retries"):
                agent._codex_incomplete_retries = 0

            # Check for tool calls
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.70】分支: model 返回了 tool_calls
            # 主循环在这里进入"工具执行"分支,后续流程:
            #   1. 打印工具调用信息(quiet 模式跳过)
            #   2. verbose 模式打印每个工具的 args
            #   3. 工具名修复(拼错→正确,如 "wirte_file" → "write_file")
            #   4. 工具名验证(不存在→拒收,最多 3 次)
            #   5. 工具参数 JSON 验证
            #   6. post-call guardrails (cap delegate, dedupe)
            #   7. 实际执行工具
            #   8. 检查压缩 + 继续下一轮
            # ═══════════════════════════════════════════════════════════════
            if assistant_message.tool_calls:
                # 4.70.1 打印"处理 N 个工具调用"提示
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

                # 4.70.2 verbose 模式:打印每个工具的入参
                if agent.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")

                # Validate tool call names - detect model hallucinations
                # Repair mismatched tool names before validating
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.71】工具名拼写错误自动修复
                # 触发场景: model 把 "write_file" 写成 "wirte_file" / "writefile"
                # 修复方式: _repair_tool_call 内部有别名/拼写映射表
                # 修复成功率: 大约 80%(常见错字)
                # 修复失败: 进入下一步"invalid_tool_calls"处理
                # ═══════════════════════════════════════════════════════════════
                for tc in assistant_message.tool_calls:
                    # 4.71.1 工具名不在 valid_tool_names 中 → 尝试修复
                    if tc.function.name not in agent.valid_tool_names:
                        # 4.71.2 调修复函数(可能返回 None)
                        repaired = agent._repair_tool_call(tc.function.name)
                        # 4.71.3 修复成功 → 替换
                        if repaired:
                            print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                            tc.function.name = repaired
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.72】工具名验证(修复后还不对就当幻觉)
                # invalid_tool_calls 收集修复后还不在白名单的工具名
                # 处理策略:
                #   - 最多重试 3 次(给 model 自己纠正机会)
                #   - 把"工具不存在"作为 tool_result 回喂给 model
                #   - 超过 3 次 → return partial
                # ═══════════════════════════════════════════════════════════════
                invalid_tool_calls = [

                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in agent.valid_tool_names
                ]
                if invalid_tool_calls:
                    # Track retries for invalid tool calls
                    # 4.72.1 幻觉工具计数 +1
                    agent._invalid_tool_retries += 1

                    # Return helpful error to model — model can agent-correct next turn
                    # 4.72.2 列出所有合法工具名(给 model 提示)
                    available = ", ".join(sorted(agent.valid_tool_names))
                    # 4.72.3 取第一个幻觉工具名做提示
                    invalid_name = invalid_tool_calls[0]
                    # 4.72.4 长名字截断到 80 字符
                    invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                    agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.73】幻觉工具超过 3 次 → 终止 turn
                    # 给 model 3 次自我纠正机会
                    # 超过 3 次:认为 model 系统性问题,放弃
                    # ═══════════════════════════════════════════════════════════════
                    if agent._invalid_tool_retries >= 3:
                        # 4.73.1 刷出状态 buffer
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                        # 4.73.2 重置计数
                        agent._invalid_tool_retries = 0
                        # 4.73.3 persist session
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": f"Model generated invalid tool call: {invalid_preview}"
                        }

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.74】构造 tool_result 回喂给 model
                    # 思路: 把"工具不存在"作为工具执行结果塞回 messages
                    # 这样 model 下次有机会"看见"自己的错误并纠正
                    # 关键: 用 role="tool" 而不是 role="user"
                    #   (role 交替约束: assistant(tool_calls) → tool → assistant → ...)
                    # ═══════════════════════════════════════════════════════════════
                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    # 4.74.1 把 assistant 消息(含坏 tool_calls)加入 messages
                    messages.append(assistant_msg)
                    # 4.74.2 为每个 tool_call 构造 tool_result
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in agent.valid_tool_names:
                            # 4.74.2a 真正幻觉的工具 → "工具不存在"
                            content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                        else:
                            # 4.74.2b 同一 turn 里有其他工具名合法 → "被跳过"
                            content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                        messages.append({
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                    # 4.74.3 continue → 回到主循环顶部,model 看到错误后重试
                    continue
                # Reset retry counter on successful tool call validation
                # 4.74.4 工具名全部合法 → 重置计数器
                agent._invalid_tool_retries = 0

                # Validate tool call arguments are valid JSON
                # Handle empty strings as empty objects (common model quirk)
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.75】工具参数 JSON 验证(4 层防御)
                # 关键问题: 模型有时输出 "function.arguments: null" 或空字符串
                #          有时又是被截断的半截 JSON (router 改写 finish_reason 时常见)
                # 防御层:
                #   1. dict/list  → 重新 json.dumps 规范化
                #   2. 非 str 类型 → str() 强转(罕见但有些 model 会传错类型)
                #   3. 空字符串/空白 → 当成 "{}"(空对象),跳过验证
                #   4. 正常 JSON   → json.loads() 验一遍,失败丢进 invalid_json_args
                # 失败处理(下一段):
                #   - 如果检测到「截断」(以 } 或 ] 结尾的判断),整个 response 拒绝执行
                #   - 否则把坏 JSON 的 tool_call 标记为 skipped,其他 tool 继续执行
                # 为什么不全拒? → 一个 tool_call 坏掉不应浪费整个 turn
                # ═══════════════════════════════════════════════════════════════
                invalid_json_args = []
                for tc in assistant_message.tool_calls:
                    # 4.75.1 取出 tool_call 的 arguments 字段
                    args = tc.function.arguments
                    # 4.75.2 防御层 1: dict/list → 重新 dumps 规范化
                    if isinstance(args, (dict, list)):
                        tc.function.arguments = json.dumps(args)
                        continue
                    # 4.75.3 防御层 2: 非 str 类型 → 强转
                    if args is not None and not isinstance(args, str):
                        tc.function.arguments = str(args)
                        args = tc.function.arguments
                    # Treat empty/whitespace strings as empty object
                    # 4.75.4 防御层 3: 空字符串/空白 → 当成 "{}"
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    # 4.75.5 防御层 4: 正常 JSON → json.loads 验一遍
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        # 4.75.6 JSON 解析失败 → 记录 (name, error)
                        invalid_json_args.append((tc.function.name, str(e)))

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.76】处理 invalid_json_args(分类:截断 vs 格式错)
                # 关键: 截断 (truncation) 和 模型错(garbage) 处理方式不同
                #   - 截断:router 改写 finish_reason="length" 为 "tool_calls"
                #          (隐藏了截断事实)
                #   - 模型错:真的输出坏了 JSON
                # 区分方法: args 末尾是否以 } 或 ] 结尾
                #   - 截断: 末尾不是 } 或 ]
                #   - 错格式: 末尾是 } 或 ] (但中间坏了)
                # ═══════════════════════════════════════════════════════════════
                if invalid_json_args:

                    # Check if the invalid JSON is due to truncation rather
                    # than a model formatting mistake.  Routers sometimes
                    # rewrite finish_reason from "length" to "tool_calls",
                    # hiding the truncation from the length handler above.
                    # Detect truncation: args that don't end with } or ]
                    # (after stripping whitespace) are cut off mid-stream.
                    # 4.76.1 检测: 是否有 tool_call 的 args 末尾不是 } 或 ]
                    _truncated = any(
                        not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                        for tc in assistant_message.tool_calls
                        if tc.function.name in {n for n, _ in invalid_json_args}
                    )
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.77】截断分支:整个 response 拒绝执行
                    # 原因: 截断说明 model 输出 tokens 不够,继续执行会半成品
                    # 处理: 终止 turn,标 partial,清理资源
                    # 注意: 重置 _invalid_json_retries(因为这是不同问题)
                    # ═══════════════════════════════════════════════════════════════
                    if _truncated:
                        # 4.77.1 打印警告
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                            f"(finish_reason={finish_reason!r}) — refusing to execute.",
                            force=True,
                        )
                        # 4.77.2 重置 JSON 错误计数(这是截断,不是 JSON 错)
                        agent._invalid_json_retries = 0
                        # 4.77.3 清理 task 资源
                        agent._cleanup_task_resources(effective_task_id)
                        # 4.77.4 persist
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit",
                        }

                    # Track retries for invalid JSON arguments
                    # 4.77.5 JSON 错误计数 +1
                    agent._invalid_json_retries += 1

                    # 4.77.6 取第一个错误的 tool_call 名字和错误消息
                    tool_name, error_msg = invalid_json_args[0]
                    agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.78】JSON 错误重试分支(最多 3 次)
                    # 策略 1 (1-3 次内): 啥都不加,直接重发 API
                    # 策略 2 (3 次后): 给 model 注入 recovery 提示
                    #   把错误作为 tool_result 塞回去
                    #   让 model "看到"自己错了并纠正
                    # ═══════════════════════════════════════════════════════════════
                    if agent._invalid_json_retries < 3:
                        # 4.78.1 重试:不写 messages,只重发 API
                        agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
                        # Don't add anything to messages, just retry the API call
                        continue
                    else:
                        # ═══════════════════════════════════════════════════════════════
                        # 【步骤 4.79】JSON 错误重试 3 次后:注入 recovery 工具结果
                        # 关键: 不 return partial,而是把"JSON 错误"作为 tool_result
                        #       让 model 自己看到错误并自我修复
                        # 关键: 用 role="tool" 不用 role="user"(role 交替约束)
                        # ═══════════════════════════════════════════════════════════════
                        # Instead of returning partial, inject tool error results so the model can recover.
                        # Using tool results (not user messages) preserves role alternation.
                        agent._buffer_vprint(f"⚠️  Injecting recovery tool results for invalid JSON...")
                        # 4.79.1 重置计数
                        agent._invalid_json_retries = 0  # Reset for next attempt

                        # Append the assistant message with its (broken) tool_calls
                        # 4.79.2 把坏 assistant 消息加入 messages
                        recovery_assistant = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(recovery_assistant)

                        # Respond with tool error results for each tool call
                        # 4.79.3 收集坏 JSON 的工具名
                        invalid_names = {name for name, _ in invalid_json_args}
                        # 4.79.4 为每个 tool_call 构造 tool_result
                        for tc in assistant_message.tool_calls:
                            if tc.function.name in invalid_names:
                                # 4.79.4a 真坏 JSON 的工具 → 给详细错误提示
                                err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                tool_result = (
                                    f"Error: Invalid JSON arguments. {err}. "
                                    f"For tools with no required parameters, use an empty object: {{}}. "
                                    f"Please retry with valid JSON."
                                )
                            else:
                                # 4.79.4b 同一 turn 里的其他工具 → "被跳过"
                                tool_result = "Skipped: other tool call in this response had invalid JSON."
                            messages.append({
                                "role": "tool",
                                "name": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        # 4.79.5 continue → 回到主循环顶部,model 看到错误后重试
                        continue

                # Reset retry counter on successful JSON validation
                # 4.79.6 全部 JSON 合法 → 重置计数器
                agent._invalid_json_retries = 0


                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.80】Post-call Guardrails — 工具调用后置防护
                # 三个动作:
                #   1. _cap_delegate_task_calls
                #      → 限制一次响应中 delegate_task 的数量(默认 8)
                #      → 防止 model 一次性开 20 个子代理把 token 烧光
                #   2. _deduplicate_tool_calls
                #      → 去重完全相同的 tool_call(name+args hash 相同)
                #      → 防止 model 重复调同一个工具(推理 bug)
                #   3. _build_assistant_message
                #      → 把 transport 层的 response 转换成 messages 数组的格式
                #      → 携带 finish_reason,reasoning_content,tool_calls
                # 然后是 "content + tool_calls" 的特殊处理:
                #   → model 在同一 turn 既给文字答案又调工具
                #   → 把文字部分存到 _last_content_with_tools
                #   → 如果下个 turn model 返回空,用这段作为兜底
                #   → 常见于: "答案是 X" + 调 memory 工具记录
                # ═══════════════════════════════════════════════════════════════
                # ── Post-call guardrails ──────────────────────────
                # 4.80.1 限制 delegate_task 数量(防止 token 风暴)
                assistant_message.tool_calls = agent._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                # 4.80.2 去重完全相同的 tool_call(防推理 bug)
                assistant_message.tool_calls = agent._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )
                # 4.80.3 构造标准的 messages 数组元素
                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

                
                # If this turn has both content AND tool_calls, capture the content
                # as a fallback final response. Common pattern: model delivers its
                # answer and calls memory/skill tools as a side-effect in the same
                # turn. If the follow-up turn after tools is empty, we use this.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.81】content+tool_calls 兜底:把"答案文本"存为备用 final
                # 触发场景: model 同 turn 既给答案("答案是 X")又调 memory 工具
                #          下个 turn model 返回空 → 用这段作为兜底
                # 关键: _HOUSEKEEPING_TOOLS 是白名单(memory/todo/skill_manage 等)
                #       只有"全部工具都是 housekeeping"才静默
                #       有 read_file / terminal 等"实在"工具 → 保留输出
                # ═══════════════════════════════════════════════════════════════
                turn_content = assistant_message.content or ""
                # 4.81.1 有内容且 think block 之后有真东西
                if turn_content and agent._has_content_after_think_block(turn_content):
                    # 4.81.2 存为兜底 final(给下个 turn 空响应时用)
                    agent._last_content_with_tools = turn_content
                    # Only mute subsequent output when EVERY tool call in
                    # this turn is post-response housekeeping (memory, todo,
                    # skill_manage, etc.).  If any substantive tool is present
                    # (search_files, read_file, write_file, terminal, ...),
                    # keep output visible so the user sees progress.
                    # 4.81.3 定义 housekeeping 工具白名单
                    _HOUSEKEEPING_TOOLS = frozenset({
                        "memory", "todo", "skill_manage", "session_search",
                    })
                    # 4.81.4 检查:所有 tool_call 是否都在白名单
                    _all_housekeeping = all(
                        tc.function.name in _HOUSEKEEPING_TOOLS
                        for tc in assistant_message.tool_calls
                    )
                    # 4.81.5 记录"是否全是 housekeeping"
                    agent._last_content_tools_all_housekeeping = _all_housekeeping
                    # 4.81.6 全是 housekeeping + 有 stream consumer → 静默
                    if _all_housekeeping and agent._has_stream_consumers():
                        agent._mute_post_response = True
                    # 4.81.7 否则 → 显示一行"摘要消息"
                    elif agent._should_emit_quiet_tool_messages():
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._vprint(f"  ┊ 💬 {clean}")

                
                # Pop thinking-only prefill message(s) before appending
                # (tool-call path — same rationale as the final-response path).
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.82】清理上一次留下的 thinking_prefill 消息
                # 为什么: 之前 _thinking_prefill 标记的 assistant 消息是临时的
                #         这次要"用真工具"了,要把假的 prefill 弹掉
                #         否则会污染 messages 历史
                # ═══════════════════════════════════════════════════════════════
                _had_prefill = False
                # 4.82.1 循环弹掉尾部所有 _thinking_prefill 消息
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # Reset prefill counter when tool calls follow a prefill
                # recovery.  Without this, the counter accumulates across
                # the whole conversation — a model that intermittently
                # empties (empty → prefill → tools → empty → prefill →
                # tools) burns both prefill attempts and the third empty
                # gets zero recovery.  Resetting here treats each tool-
                # call success as a fresh start.
                # 4.82.2 成功工具调用后,重置 prefill / empty 计数器
                if _had_prefill:
                    agent._thinking_prefill_retries = 0
                    agent._empty_content_retries = 0
                # Successful tool execution — reset the post-tool nudge
                # flag so it can fire again if the model goes empty on
                # a LATER tool round.
                # 4.82.3 重置 post-tool nudge flag(下次还能再 nudge)
                agent._post_tool_empty_retried = False

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.83】把 assistant 消息正式加入 messages
                # 顺序很重要:必须先加 assistant 消息,再加 tool_result
                # (OpenAI/Anthropic 的消息格式要求)
                # ═══════════════════════════════════════════════════════════════
                # 4.83.1 加入 assistant 消息
                messages.append(assistant_msg)
                # 4.83.2 推送中间消息给 UI(subagent 转发用)
                agent._emit_interim_assistant_message(assistant_msg)

                # Close any open streaming display (response box, reasoning
                # box) before tool execution begins.  Intermediate turns may
                # have streamed early content that opened the response box;
                # flushing here prevents it from wrapping tool feed lines.
                # Only signal the display callback — TTS (_stream_callback)
                # should NOT receive None (it uses None as end-of-stream).
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.84】关闭流式 display box(给 tool 输出腾位置)
                # 关键: 不能发 None 给 TTS callback(None 是它的"结束"信号)
                #      只发 None 给 display callback
                # ═══════════════════════════════════════════════════════════════
                # 4.84.1 有 stream callback → 发 None 关闭 box
                if agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(None)
                    except Exception:
                        pass

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.85】工具执行 — 主循环委托给 agent 自身方法
                # 关键设计: 这一行是 conversation_loop.py 唯一一次直接调工具的入口
                # 实际执行在 agent/_execute_tool_calls → tool_executor.py
                # 执行模式:
                #   - 单个 tool_call: _execute_tool_calls_sequential
                #   - 多个独立 tool_call: _execute_tool_calls_concurrent
                #     (ThreadPoolExecutor max_workers=8)
                # effective_task_id: 隔离并发任务的 VM 边界
                # api_call_count: 传递给工具,用于嵌套 turn 计数
                # 工具结果回填:
                #   - 成功 → messages.append({"role": "tool", "content": result})
                #   - 失败 → messages.append({"role": "tool", "content": error_msg})
                #   - 审批拒绝 → 同样回填,model 看到拒绝理由
                # 执行后: 检查 _tool_guardrail_halt_decision
                #   - 工具 guardrail 触发(危险命令/越权访问)→ 立刻结束 turn
                #   - 由 _toolguard_controlled_halt_response 构造给用户看的解释
                # ═══════════════════════════════════════════════════════════════
                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)


                if agent._tool_guardrail_halt_decision is not None:
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.86】工具 guardrail 触发 → 立即结束 turn
                    # 触发场景: 工具被安全系统拦下(危险命令/越权访问/违反沙箱)
                    # 处理:
                    #   1. 标记 turn_exit_reason = "guardrail_halt"
                    #   2. 构造给用户看的解释
                    #   3. 推送到 stream callback(让 SSE/TUI 客户端看到)
                    #   4. break 退出主循环
                    # ═══════════════════════════════════════════════════════════════
                    decision = agent._tool_guardrail_halt_decision
                    # 4.86.1 记录退出原因
                    _turn_exit_reason = "guardrail_halt"
                    # 4.86.2 构造给用户看的解释
                    final_response = agent._toolguard_controlled_halt_response(decision)
                    agent._emit_status(
                        f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                    )
                    # 4.86.3 把解释加进 messages
                    messages.append({"role": "assistant", "content": final_response})
                    # Emit the halt message to the client so it's not
                    # indistinguishable from a crash.  The stream display
                    # was flushed (callback(None)) before tool execution,
                    # but the callback is still alive — fire the text
                    # through it so SSE/TUI clients see the explanation.
                    # 4.86.4 推送 halt 消息给 stream
                    if final_response:
                        agent._safe_print(f"\n{final_response}\n")
                        if agent.stream_delta_callback:
                            try:
                                agent.stream_delta_callback(final_response)
                                agent.stream_delta_callback(None)
                            except Exception:
                                pass
                    # 4.86.5 break 退出主循环
                    break

                # Reset per-turn retry counters after successful tool
                # execution so a single truncation doesn't poison the
                # entire conversation.
                # 4.86.6 重置 truncated_tool_call 计数
                truncated_tool_call_retries = 0

                # Signal that a paragraph break is needed before the next
                # streamed text.  We don't emit it immediately because
                # multiple consecutive tool iterations would stack up
                # redundant blank lines.  Instead, _fire_stream_delta()
                # will prepend a single "\n\n" the next time real text
                # arrives.
                # 4.86.7 标记流式需要段落分隔
                agent._stream_needs_break = True

                # Refund the iteration if the ONLY tool(s) called were
                # execute_code (programmatic tool calling).  These are
                # cheap RPC-style calls that shouldn't eat the budget.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.87】execute_code 单独优惠:退还 budget
                # 原因: execute_code 是程序化调用,不是 LLM 推理
                #       开销小,不应该算 turn budget
                # ═══════════════════════════════════════════════════════════════
                # 4.87.1 取出本次调用的所有工具名
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                # 4.87.2 如果只有 execute_code → 退 budget
                if _tc_names == {"execute_code"}:
                    agent.iteration_budget.refund()

                
                # Use real token counts from the API response to decide
                # compression.  prompt_tokens + completion_tokens is the
                # actual context size the provider reported plus the
                # assistant turn — a tight lower bound for the next prompt.
                # Tool results appended above aren't counted yet, but the
                # threshold (default 50%) leaves ample headroom; if tool
                # results push past it, the next API call will report the
                # real total and trigger compression then.
                #
                # If last_prompt_tokens is 0 (stale after API disconnect
                # or provider returned no usage data), fall back to rough
                # estimate to avoid missing compression.  Without this,
                # a session can grow unbounded after disconnects because
                # should_compress(0) never fires.  (#2153)
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.88】决定压缩的真实 token 数(3 种路径)
                # 优先级: prompt_tokens(API 报告) > 0(刚拿到,首选)
                #          == -1(刚压过,跳过)
                #          == 0(失联,粗估兜底)
                # 关键: 只用 prompt_tokens,不用 completion_tokens
                #       因为 thinking model(GLM/QwQ/DeepSeek R1)completion 很大
                #       包含 reasoning 不占 context window
                # ═══════════════════════════════════════════════════════════════
                _compressor = agent.context_compressor
                # 4.88.1 路径 1: API 报告了 prompt_tokens(>0)
                if _compressor.last_prompt_tokens > 0:
                    # Only use prompt_tokens — completion/reasoning
                    # tokens don't consume context window space.
                    # Thinking models (GLM-5.1, QwQ, DeepSeek R1)
                    # inflate completion_tokens with reasoning,
                    # causing premature compression.  (#12026)
                    _real_tokens = _compressor.last_prompt_tokens
                # 4.88.2 路径 2: 刚压缩过(=-1,避免重复压缩)
                elif _compressor.last_prompt_tokens == -1:
                    # Compression just ran and no API-reported prompt count
                    # has arrived yet. Avoid treating a schema-heavy rough
                    # post-compression estimate as real context pressure.
                    _real_tokens = 0
                # 4.88.3 路径 3: API 失联/无 usage(==0,粗估兜底)
                else:
                    # Include tool schemas — with 50+ tools enabled
                    # these add 20-30K tokens the messages-only
                    # estimate misses, which can skip compression
                    # past the configured threshold (#14695).
                    _real_tokens = estimate_request_tokens_rough(
                        messages, tools=agent.tools or None
                    )

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.89】判断并执行压缩
                # 触发条件: compression_enabled + 实际 token 超过阈值
                # 默认阈值: context window 的 50%
                # 压缩后: 新的 session_id,需要清空 conversation_history
                # ═══════════════════════════════════════════════════════════════
                if agent.compression_enabled and _compressor.should_compress(_real_tokens):
                    # 4.89.1 打印"压缩中"提示
                    agent._safe_print("  ⟳ compacting context…")
                    # 4.89.2 调压缩函数(返回新的 messages + system_prompt)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=agent.context_compressor.last_prompt_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history so
                    # _flush_messages_to_session_db writes compressed messages
                    # to the new session (see preflight compression comment).
                    # 4.89.3 清空 history(压缩创建了新 session)
                    conversation_history = None

                # Save session log incrementally (so progress is visible even if interrupted)
                # 4.89.4 增量保存 session(中断也能看到进度)
                agent._session_messages = messages

                # Continue loop for next response
                # 4.89.5 continue → 回到主循环顶部,处理下一轮
                continue

            
            else:
                # No tool calls - this is the final response
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.90】分支: model 没有 tool_calls = 这是最终文本回复
                # 后续: 5 种 final 路径选择(按优先级)
                #   1. partial stream recovery → 用已经流出去的内容
                #   2. fallback prior turn content → 用上 turn 存的 housekeeping 内容
                #   3. post-tool nudge → 注入 user 提示让 model 继续
                #   4. thinking prefill → model 在推理中,补一句让 model 接着写
                #   5. truly empty + retry → 重试
                #   6. fallback provider → 切到下一个 provider
                #   7. "(empty)" 终止哨兵
                # ═══════════════════════════════════════════════════════════════
                final_response = assistant_message.content or ""

                # Fix: unmute output when entering the no-tool-call branch
                # so the user can see empty-response warnings and recovery
                # status messages.  _mute_post_response was set during a
                # prior housekeeping tool turn and should not silence the
                # final response path.
                # 4.90.1 解除 mute(让用户看到空响应警告)
                agent._mute_post_response = False

                # Check if response only has think block with no actual content after it
                # 4.90.2 检测: 是不是只有 think block,没有真内容
                if not agent._has_content_after_think_block(final_response):
                    # ── Partial stream recovery ─────────────────────
                    # If content was already streamed to the user before
                    # the connection died, use it as the final response
                    # instead of falling through to prior-turn fallback
                    # or wasting API calls on retries.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.91】Partial stream recovery(用已流出的部分作为 final)
                    # 触发场景: SSE 连接中断,但已经有部分内容流出去给用户看了
                    #          API 调用可能因为网络断失败
                    # 关键: 不重试,直接把 _current_streamed_assistant_text 用上
                    #       (用户已经看到了,别浪费 API)
                    # ═══════════════════════════════════════════════════════════════
                    # 4.91.1 取当前已流出去的文本
                    _partial_streamed = (
                        getattr(agent, "_current_streamed_assistant_text", "") or ""
                    )
                    # 4.91.2 部分流出有真内容 → 用它
                    if agent._has_content_after_think_block(_partial_streamed):
                        _turn_exit_reason = "partial_stream_recovery"
                        # 4.91.3 剥 think blocks
                        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
                        logger.info(
                            "Partial stream content delivered (%d chars) "
                            "— using as final response",
                            len(_recovered),
                        )
                        agent._emit_status(
                            "↻ Stream interrupted — using delivered content "
                            "as final response"
                        )
                        # 4.91.4 标记 + 用 recovery 内容 + break
                        final_response = _recovered
                        agent._response_was_previewed = True
                        break

                    # If the previous turn already delivered real content alongside
                    # HOUSEKEEPING tool calls (e.g. "You're welcome!" + memory save),
                    # the model has nothing more to say. Use the earlier content
                    # immediately instead of wasting API calls on retries.
                    # NOTE: Only use this shortcut when ALL tools in that turn were
                    # housekeeping (memory, todo, etc.).  When substantive tools
                    # were called (terminal, search_files, etc.), the content was
                    # likely mid-task narration ("I'll scan the directory...") and
                    # the empty follow-up means the model choked — let the
                    # post-tool nudge below handle that instead of exiting early.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.92】Fallback prior-turn content(用上 turn 的 housekeeping 内容)
                    # 触发场景: 上 turn 给了"答案是 X" + 调了 memory 工具
                    #          这 turn model 没说话(empty)
                    # 关键: 只在 _last_content_tools_all_housekeeping=True 时用
                    #       (避免把"我去查文件"这种中途话当成 final 答案)
                    # ═══════════════════════════════════════════════════════════════
                    # 4.92.1 取上 turn 存的兜底内容
                    fallback = getattr(agent, '_last_content_with_tools', None)
                    # 4.92.2 有兜底 + 全是 housekeeping → 用它
                    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
                        _turn_exit_reason = "fallback_prior_turn_content"
                        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                        agent._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                        # 4.92.3 清掉兜底(下次别再用)
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        agent._empty_content_retries = 0
                        # Do NOT modify the assistant message content — the
                        # old code injected "Calling the X tools..." which
                        # poisoned the conversation history.  Just use the
                        # fallback text as the final response and break.
                        # 4.92.4 用剥 think 后的兜底文本
                        final_response = agent._strip_think_blocks(fallback).strip()
                        agent._response_was_previewed = True
                        # 4.92.5 break 退出主循环
                        break

                    # ── Post-tool-call empty response nudge ───────────
                    # The model returned empty after executing tool calls.
                    # This covers two cases:
                    #  (a) No prior-turn content at all — model went silent
                    #  (b) Prior turn had content + SUBSTANTIVE tools (the
                    #      fallback above was skipped because the content
                    #      was mid-task narration, not a final answer)
                    # Instead of giving up, nudge the model to continue by
                    # appending a user-level hint.  This is the #9400 case:
                    # weaker models (mimo-v2-pro, GLM-5, etc.) sometimes
                    # return empty after tool results instead of continuing
                    # to the next step.  One retry with a nudge usually
                    # fixes it.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.93】Post-tool nudge(给 model 注入 user 提示继续)
                    # 触发场景: model 调了工具 → 看到结果 → 沉默了(empty)
                    # 解决: 假装是 user 提醒 model "请处理 tool 结果"
                    # 关键: role 交替约束(tool → user → assistant 是合法的)
                    # 限制: 只 nudge 1 次,避免无限 nudge 循环
                    # ═══════════════════════════════════════════════════════════════
                    # 4.93.1 检测前 5 条消息里有没有 tool 结果
                    _prior_was_tool = any(
                        m.get("role") == "tool"
                        for m in messages[-5:]  # check recent messages
                    )
                    # Detect Qwen3/Ollama-style in-content thinking blocks.
                    # Ollama puts <think> in the content field (not in
                    # reasoning_content), so _has_structured below would
                    # miss it.  We check here so thinking-only responses
                    # after tool calls route to prefill instead of nudge.
                    # 4.93.2 检测是不是 Qwen3/Ollama 内联 think 块
                    _has_inline_thinking = bool(
                        re.search(
                            r'<think>|<thinking>|<reasoning>',
                            final_response or "",
                            re.IGNORECASE,
                        )
                    )
                    # 4.93.3 触发 nudge 的条件: 之前是 tool + 没 nudge 过 + 不是 thinking
                    if (
                        _prior_was_tool
                        and not getattr(agent, "_post_tool_empty_retried", False)
                        and not _has_inline_thinking  # thinking model still working — let prefill handle
                    ):
                        # 4.93.4 标记已 nudge 过(防止无限 nudge)
                        # 4.93.4 标记已 nudge 过(防止无限 nudge)
                        agent._post_tool_empty_retried = True
                        # Clear stale narration so it doesn't resurface
                        # on a later empty response after the nudge.
                        # ═══════════════════════════════════════════════════════════════
                        # 【步骤 4.94】构造 nudge 消息并继续
                        # 关键约束: tool → user 直接跳是 role 交替违法的
                        #          必须先加一个 assistant("(empty)") 充数
                        #          再加 user 提示
                        # ═══════════════════════════════════════════════════════════════

                        # 4.94.1 清掉上 turn 的兜底
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        logger.info(
                            "Empty response after tool calls — nudging model "
                            "to continue processing"
                        )
                        agent._buffer_status(
                            "⚠️ Model returned empty after tool calls — "
                            "nudging to continue"
                        )
                        # Append the empty assistant message first so the
                        # message sequence stays valid:
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # Without this, we'd have tool → user which most
                        # APIs reject as an invalid sequence.
                        # 4.94.2 构造假的 assistant("(empty)") 充数消息
                        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        _nudge_msg["content"] = "(empty)"
                        _nudge_msg["_empty_recovery_synthetic"] = True
                        messages.append(_nudge_msg)
                        # 4.94.3 加 user 提示让 model 继续
                        messages.append({
                            "role": "user",
                            "content": (
                                "You just executed tool calls but returned an "
                                "empty response. Please process the tool "
                                "results above and continue with the task."
                            ),
                            "_empty_recovery_synthetic": True,
                        })
                        # 4.94.4 continue → 回到主循环顶部重发
                        continue


                    # ── Thinking-only prefill continuation ──────────
                    # The model produced structured reasoning (via API
                    # fields) but no visible text content.  Rather than
                    # giving up, append the assistant message as-is and
                    # continue — the model will see its own reasoning
                    # on the next turn and produce the text portion.
                    # Inspired by clawdbot's "incomplete-text" recovery.
                    # Also covers Qwen3/Ollama in-content <think> blocks
                    # (detected above as _has_inline_thinking).
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.95】Thinking-only prefill(让 model 接着写)
                    # 触发场景: model 在 thinking 中(reasoning 字段有内容)
                    #          但 visible content 是空
                    # 解决: 把这次响应原样加入 messages(标记 incomplete)
                    #       下 turn model 会看到自己的推理并接着输出
                    # 限制: 最多 2 次
                    # ═══════════════════════════════════════════════════════════════
                    # 4.95.1 检测 4 种结构化 reasoning
                    _has_structured = bool(
                        getattr(assistant_message, "reasoning", None)
                        or getattr(assistant_message, "reasoning_content", None)
                        or getattr(assistant_message, "reasoning_details", None)
                        or _has_inline_thinking
                    )
                    # 4.95.2 有结构化 reasoning + 没超 2 次 → prefill
                    if _has_structured and agent._thinking_prefill_retries < 2:
                        # 4.95.3 prefill 计数 +1
                        agent._thinking_prefill_retries += 1
                        logger.info(
                            "Thinking-only response (no visible content) — "
                            "prefilling to continue (%d/2)",
                            agent._thinking_prefill_retries,
                        )
                        agent._buffer_status(
                            f"↻ Thinking-only response — prefilling to continue "
                            f"({agent._thinking_prefill_retries}/2)"
                        )
                        # 4.95.4 构造 interim 消息,标记 incomplete
                        interim_msg = agent._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        interim_msg["_thinking_prefill"] = True
                        messages.append(interim_msg)
                        # 4.95.5 保存 session + continue
                        agent._session_messages = messages
                        continue

                    # ── Empty response retry ──────────────────────
                    # Model returned nothing usable.  Retry up to 3
                    # times before attempting fallback.  This covers
                    # both truly empty responses (no content, no
                    # reasoning) AND reasoning-only responses after
                    # prefill exhaustion — models like mimo-v2-pro
                    # always populate reasoning fields via OpenRouter,
                    # so the old `not _has_structured` guard blocked
                    # retries for every reasoning model after prefill.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.96】Empty response retry(最多 3 次)
                    # 触发条件: 真空(没有 content 也没有 reasoning) + prefill 用完
                    # 限制: 3 次后转 fallback
                    # ═══════════════════════════════════════════════════════════════
                    # 4.96.1 是不是真没东西(剥 think block 后也空)
                    _truly_empty = not agent._strip_think_blocks(
                        final_response
                    ).strip()
                    # 4.96.2 是不是 prefill 用完的 reasoning-only
                    _prefill_exhausted = (
                        _has_structured
                        and agent._thinking_prefill_retries >= 2
                    )
                    # 4.96.3 真空 + 没超 3 次 → 重试
                    if _truly_empty and (not _has_structured or _prefill_exhausted) and agent._empty_content_retries < 3:
                        # 4.96.4 empty 计数 +1
                        agent._empty_content_retries += 1
                        logger.warning(
                            "Empty response (no content or reasoning) — "
                            "retry %d/3 (model=%s)",
                            agent._empty_content_retries, agent.model,
                        )
                        agent._buffer_status(
                            f"⚠️ Empty response from model — retrying "
                            f"({agent._empty_content_retries}/3)"
                        )
                        # 4.96.5 continue → 重新发 API
                        continue

                    # ── Exhausted retries — try fallback provider ──
                    # Before giving up with "(empty)", attempt to
                    # switch to the next provider in the fallback
                    # chain.  This covers the case where a model
                    # (e.g. GLM-4.5-Air) consistently returns empty
                    # due to context degradation or provider issues.
                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.97】尝试 fallback provider(切到下一个)
                    # 触发条件: 重试用尽 + 真空 + 有 fallback_chain
                    # 设计: 切到 _fallback_chain 的下一个 provider
                    #       如果有 → 重置 empty 计数 + continue
                    #       如果没 → 终止
                    # ═══════════════════════════════════════════════════════════════
                    if _truly_empty and agent._fallback_chain:
                        logger.warning(
                            "Empty response after %d retries — "
                            "attempting fallback (model=%s, provider=%s)",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._buffer_status(

                            "⚠️ Model returning empty responses — "
                            "switching to fallback provider..."
                        )
                        # 4.97.1 尝试激活 fallback(切到下一个 provider)
                        if agent._try_activate_fallback():
                            # 4.97.2 fallback 成功 → 重置 empty 计数
                            agent._empty_content_retries = 0
                            agent._buffer_status(
                                f"↻ Switched to fallback: {agent.model} "
                                f"({agent.provider})"
                            )
                            logger.info(
                                "Fallback activated after empty responses: "
                                "now using %s on %s",
                                agent.model, agent.provider,
                            )
                            # 4.97.3 continue → 用新 provider 重发
                            continue

                    # ═══════════════════════════════════════════════════════════════
                    # 【步骤 4.98】所有路径用尽: "(empty)" 终止哨兵
                    # 触发条件: 重试 3 次 + fallback 也用完(或没有 fallback)
                    # 关键: 标 _empty_terminal_sentinel = True
                    #       持久化时不存(防止下次 /continue 回放)
                    # ═══════════════════════════════════════════════════════════════
                    # Exhausted retries and fallback chain (or no
                    # fallback configured).  Fall through to the
                    # "(empty)" terminal.
                    # Surface the buffered retry/fallback trace so the
                    # user can see what was attempted before "(empty)".
                    # 4.98.1 刷出状态 buffer(让用户看到 retry 痕迹)
                    agent._flush_status_buffer()
                    _turn_exit_reason = "empty_response_exhausted"
                    # 4.98.2 抽出 reasoning 文本(可能只是 reasoning-only)
                    reasoning_text = agent._extract_reasoning(assistant_message)
                    # 4.98.3 弹掉尾部临时脚手架(空的 nudge/prefill 等)
                    agent._drop_trailing_empty_response_scaffolding(messages)
                    # 4.98.4 构造 final 消息,内容 "(empty)"
                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    assistant_msg["content"] = "(empty)"
                    # 4.98.5 标 sentinel(防止 persist 时污染历史)
                    assistant_msg["_empty_terminal_sentinel"] = True
                    messages.append(assistant_msg)

                    # 4.98.6 reasoning-only 警告(只警告不阻断)
                    if reasoning_text:
                        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        logger.warning(
                            "Reasoning-only response (no visible content) "
                            "after exhausting retries and fallback. "
                            "Reasoning: %s", reasoning_preview,
                        )
                        agent._emit_status(
                            "⚠️ Model produced reasoning but no visible "
                            "response after all retries. Returning empty."
                        )
                    else:
                        # 4.98.7 真空警告
                        logger.warning(
                            "Empty response (no content or reasoning) "
                            "after %d retries. No fallback available. "
                            "model=%s provider=%s",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._emit_status(
                            "❌ Model returned no content after all retries"
                            + (" and fallback attempts." if agent._fallback_chain else
                               ". No fallback providers configured.")
                        )

                    # 4.98.8 final_response = "(empty)" + break
                    final_response = "(empty)"
                    break

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.99】成功路径:有真内容时清理所有 retry 计数器
                # 关键: 一旦拿到真内容,所有空响应/prefill 计数都重置
                #       清掉状态 buffer(用户不需要看到 retry 痕迹)
                # ═══════════════════════════════════════════════════════════════
                # Reset retry counter/signature on successful content
                # 4.99.1 重置所有空响应 / prefill 计数
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                # Successful content reached — drop any buffered retry
                # status from earlier failed attempts in this turn.
                # 4.99.2 清状态 buffer
                agent._clear_status_buffer()

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.100】Codex Responses API 的中间确认 (ack)
                # 触发条件: api_mode == "codex_responses" + model 给的是"我来干活了"型回复
                #          (例如 "Sure, I'll search for that")
                # 解决: 注入 user 提示让 model 真开始干(调工具)
                # 限制: 最多 2 次
                # ═══════════════════════════════════════════════════════════════
                if (
                    agent.api_mode == "codex_responses"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    # 4.100.1 ack 计数 +1
                    codex_ack_continuations += 1
                    # 4.100.2 加 incomplete 标记
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    agent._emit_interim_assistant_message(interim_msg)

                    # 4.100.3 加 user 提示让 model 真开始
                    continue_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(continue_msg)
                    agent._session_messages = messages
                    # 4.100.4 continue → 重发
                    continue

                # 4.100.5 走了 ack 路径之外 → 重置 ack 计数
                codex_ack_continuations = 0

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.101】合并前 length 截断续写留下的 parts
                # 触发条件: 之前 finish_reason="length" 续写过
                #          truncated_response_parts 有内容
                # 处理: 把 parts 拼到 final_response 前面(时序)
                #       清空 parts + 重置 length_continue_retries
                # ═══════════════════════════════════════════════════════════════
                if truncated_response_parts:
                    # 4.101.1 拼上之前的 parts
                    final_response = "".join(truncated_response_parts) + final_response
                    # 4.101.2 清空
                    truncated_response_parts = []
                    # 4.101.3 重置续写计数
                    length_continue_retries = 0

                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.102】最终处理:剥 think block,构造 final 消息
                # 4.102.1 剥 think blocks(让用户看到纯文本)
                final_response = agent._strip_think_blocks(final_response).strip()
                # 4.102.2 构造最终 assistant 消息
                final_msg = agent._build_assistant_message(assistant_message, finish_reason)

                # Pop thinking-only prefill and empty-response retry
                # scaffolding before appending the final response.  These
                # internal turns are only for the next API retry and should
                # not become durable transcript context.
                # ═══════════════════════════════════════════════════════════════
                # 【步骤 4.103】弹掉所有 retry 时的临时脚手架消息
                # 弹掉 3 种标记的消息:
                #   - _thinking_prefill
                #   - _empty_recovery_synthetic
                #   - _empty_terminal_sentinel
                # 原因: 这些是重试时的临时辅助,不该成为持久化历史
                # ═══════════════════════════════════════════════════════════════
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and (
                        messages[-1].get("_thinking_prefill")
                        or messages[-1].get("_empty_recovery_synthetic")
                        or messages[-1].get("_empty_terminal_sentinel")
                    )
                ):
                    messages.pop()


                messages.append(final_msg)
                
                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not agent.quiet_mode:
                    agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            try:
                print(f"❌ {error_msg}")
            except (OSError, ValueError):
                logger.error(error_msg)

            # Emit the full traceback at ERROR level so it lands in both
            # agent.log AND errors.log.  Previously this was logged at DEBUG,
            # which meant intermittent outer-loop failures were unreproducible
            # — users would see a one-line summary on screen with no way to
            # recover the call site.  logger.exception() includes the
            # traceback automatically and emits at ERROR.
            # 4.104.1 记录完整 traceback 到 agent.log + errors.log
            logger.exception("Outer loop error in API call #%d", api_call_count)

            # If an assistant message with tool_calls was already appended,
            # the API expects a role="tool" result for every tool_call_id.
            # Fill in error results for any that weren't answered yet.
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.105】补齐未应答的 tool_calls(防 API 报错)
            # 触发场景: 外层 except 捕获到异常
            #          此时 messages 里可能已经有 assistant(tool_calls)
            #          但没有对应的 tool_result
            #          下次 API 会因为"tool_call_id 没有 result"而报错
            # 处理: 倒着扫 messages,找到有 tool_calls 的 assistant
            #       给所有未应答的 tool_call_id 塞个"Error" tool_result
            # ═══════════════════════════════════════════════════════════════
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # 4.105.1 收集已经应答的 tool_call_id
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    # 4.105.2 给没应答的塞错误结果
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            messages.append(err_msg)
                break

            # Non-tool errors don't need a synthetic message injected.
            # The error is already printed to the user (line above), and
            # the retry loop continues.  Injecting a fake user/assistant
            # message pollutes history, burns tokens, and risks violating
            # role-alternation invariants.

            # If we're near the limit, break to avoid infinite loops
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.106】接近 max_iterations → 终止 turn
            # 触发条件: 异常时已经用了 max_iterations-1 次
            # 关键: 给个"apology"消息,标记退出
            # ═══════════════════════════════════════════════════════════════
            if api_call_count >= agent.max_iterations - 1:
                _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # Append as assistant so the history stays valid for
                # session resume (avoids consecutive user messages).
                messages.append({"role": "assistant", "content": final_response})
                break

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.107】外层 while 退出后:处理 max_iterations 耗尽
    # 触发条件: api_call_count >= max_iterations 或 iteration_budget 用完
    # 处理: 调 _handle_max_iterations(再发一次 API,无工具,只请求总结)
    # ═══════════════════════════════════════════════════════════════
    if final_response is None and (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    ):
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        # 4.107.1 状态提示
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        # 4.107.2 非 quiet 模式 → 打印
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        # 4.107.3 调 _handle_max_iterations 总结
        final_response = agent._handle_max_iterations(messages, api_call_count)

        # If running as a kanban worker, signal the dispatcher that the
        # worker could not complete (rather than treating it as a
        # protocol violation).  The agent loop strips tools before calling
        # _handle_max_iterations, so the model cannot call kanban_block
        # itself — we must do it on its behalf.
        #
        # We route through ``_record_task_failure(outcome="timed_out")``
        # rather than ``kanban_block`` so this counts toward the
        # ``consecutive_failures`` counter and the dispatcher's
        # ``failure_limit`` circuit breaker (#29747 gap 2).  Without this,
        # a task whose worker keeps exhausting its budget would block
        # silently each run, get auto-promoted by the operator (or never
        # surface), and re-block in an endless loop with no signal.
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.108】Kanban worker 超时 → 记 failure
        # 触发条件: HERMES_KANBAN_TASK 环境变量存在(说明是 kanban worker)
        # 设计: 调 _record_task_failure 而不是 kanban_block
        #       让 consecutive_failures 计数 +1
        #       触发 dispatcher's failure_limit 熔断器
        # ═══════════════════════════════════════════════════════════════
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                from hermes_cli import kanban_db as _kb
                # 4.108.1 开 DB 连接
                _conn = _kb.connect()
                try:
                    # 4.108.2 记 failure(outcome="timed_out")
                    _kb._record_task_failure(
                        _conn,
                        _kanban_task,
                        error=(
                            f"Iteration budget exhausted "
                            f"({api_call_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                        outcome="timed_out",
                        release_claim=True,
                        end_run=True,
                        event_payload_extra={
                            "budget_used": api_call_count,
                            "budget_max": agent.max_iterations,
                        },
                    )
                    logger.info(
                        "recorded budget-exhausted failure for task %s (%d/%d)",
                        _kanban_task, api_call_count, agent.max_iterations,
                    )
                finally:
                    # 4.108.3 关连接(无论成败)
                    try:
                        _conn.close()
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "Failed to record budget-exhausted failure for task %s",
                    _kanban_task,
                    exc_info=True,
                )


    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.109】判断 conversation 是否成功完成
    # 3 个条件同时满足才 completed=True:
    #   1. final_response 不是 None(有真东西返回)
    #   2. api_call_count < max_iterations(没跑满)
    #   3. not failed(没出 fatal error)
    # ═══════════════════════════════════════════════════════════════
    # Determine if conversation completed successfully
    completed = (
        final_response is not None
        and api_call_count < agent.max_iterations
        and not failed
    )

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.109.1】保存 trajectory(轨迹)
    # 用途: 记录这次 turn 的完整路径,用于调试和回放
    # user_message 可能是多模态(列表),先归一化成字符串
    # ═══════════════════════════════════════════════════════════════
    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)


    # Clean up VM and browser for this task after conversation completes
    # ═══════════════════════════════════════════════════════════════
    # 【学习要点】run_conversation 收尾 — 4 个收尾动作
    # 顺序很重要,不能乱:
    #   1. _cleanup_task_resources(effective_task_id)
    #      → 释放子代理 task 资源(file handles / sandbox / 线程)
    #      → 不清的话子代理的 VM 会泄漏
    #   2. _drop_trailing_empty_response_scaffolding(messages)
    #      → 删掉内部的 assistant("(empty)") / 恢复 nudge 等临时脚手架
    #      → 防止下次 /continue 时回放这些噪音消息
    #      → 必须先于 persist,否则会写到 session DB 里
    #   3. _persist_session(messages, conversation_history)
    #      → 写两份:
    #         a) JSON log (用于 /resume 恢复)
    #         b) SQLite session DB (用于 /insights 查询)
    #      → 即使这次 turn 失败也 persist,保证不丢数据
    #   4. (下面) Turn-exit diagnostic log
    #      → 解释为什么这个 turn 结束(给运维/调试用)
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.110】清理 task 资源
    # 4.110.1 释放子代理 task 资源(file handles / sandbox / 线程)
    agent._cleanup_task_resources(effective_task_id)

    # Persist session to both JSON log and SQLite only after private retry
    # scaffolding has been removed. Otherwise a later user "continue" turn
    # can replay assistant("(empty)") / recovery nudges and fall into the
    # same empty-response loop again.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.111】弹掉临时脚手架 + 持久化
    # 顺序很重要:
    #   1. 先 drop 临时脚手架(防 /continue 回放)
    #   2. 再 persist
    # ═══════════════════════════════════════════════════════════════
    # 4.111.1 弹尾部临时脚手架
    agent._drop_trailing_empty_response_scaffolding(messages)
    # 4.111.2 持久化(JSON log + SQLite)
    agent._persist_session(messages, conversation_history)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.112】Turn-exit 诊断日志 — 解释"为什么这个 turn 结束了"
    # 收集 turn 快照的 8 个字段:
    #   - _last_msg_role: 最后一条消息的 role
    #   - _last_tool_name: 如果是 tool,哪个工具(回溯找)
    #   - _turn_tool_count: 本 turn 调了几个工具
    #   - _resp_len: final_response 长度
    #   - _budget_used / _budget_max: budget 用量
    # 写入 agent.log,分两个等级:
    #   - 正常 → INFO
    #   - 异常(mid-work "just stops")→ WARNING
    # ═══════════════════════════════════════════════════════════════
    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    # 4.112.1 取最后一条消息的 role
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        # 4.112.2 倒着扫,找到最后那个调 tool 的 assistant
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    # 4.112.3 统计本 turn 调过几个工具
    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    # 4.112.4 final_response 长度
    _resp_len = len(final_response) if final_response else 0
    # 4.112.5 budget 用量
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.113】诊断日志 — 12 字段 turn 快照
    # 字段:
    #   reason    : _turn_exit_reason (12 种退出原因之一)
    #   model     : 当前 model 名
    #   api_calls : 本 turn 调了几次 API
    #   budget    : iteration_budget 用量
    #   tool_turns: 有 tool_call 的 assistant 消息数
    #   last_msg  : 最后一条消息的 role
    #   resp_len  : final_response 长度(0 = 没产生输出)
    # ═══════════════════════════════════════════════════════════════
    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    # 4.113.1 "just stops" 场景 → WARNING(让人 grep 到)
    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        # 4.113.2 正常结束 → INFO
        logger.info(_diag_msg, *_diag_args)


    # File-mutation verifier footer.
    # ═══════════════════════════════════════════════════════════════
    # 【学习要点】File Mutation Verifier — 防止 model "撒谎" 说文件改了
    # 问题场景(PR #15524-adjacent):
    #   → model 一批并行 patch_file 调用
    #   → 一半失败 "Could not find old_string"
    #   → model 在总结时说"所有文件都改好了"
    #   → 用户必须手动 git status 才能发现被骗
    # 解决: Hermes 在 turn 结尾检查 _turn_failed_file_mutations
    #   - 还在的失败记录 → model 没说"恢复了" → 加 advisory footer
    #   - 已被后续成功写入覆盖 → 不加 footer
    # 设计哲学: 让 model "结构性不可能"在文件没改时声称改了
    # 启用条件: _file_mutation_verifier_enabled() (用户配置)
    # 位置: 在 final_response 已经有内容且没被中断时,附加到末尾
    # ═══════════════════════════════════════════════════════════════
    # If one or more ``write_file`` / ``patch`` calls failed during this
    # turn and were never superseded by a successful write to the same
    # path, append an advisory footer to the assistant response.  This
    # catches the specific case — reported by Ben Eng (#15524-adjacent)
    # — where a model issues a batch of parallel patches, half of them
    # fail with "Could not find old_string", and the model summarises
    # the turn claiming every file was edited.  The user then has to
    # manually run ``git status`` to catch the lie.  With this footer
    # the truth is surfaced on every turn, so over-claiming is
    # structurally impossible past the model.
    #
    # Gate: only applied when a real text response exists for this
    # turn and the user didn't interrupt.  Empty/interrupted turns
    # already have other surface text that shouldn't be augmented.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.114】File-mutation verifier footer(防 model 撒谎)
    # 问题场景(PR #15524-adjacent):
    #   → model 一批并行 patch_file 调用
    #   → 一半失败 "Could not find old_string"
    #   → model 在总结时说"所有文件都改好了"
    #   → 用户必须手动 git status 才能发现被骗
    # 解决: Hermes 在 turn 结尾检查 _turn_failed_file_mutations
    #   - 还在的失败记录 → model 没说"恢复了" → 加 advisory footer
    #   - 已被后续成功写入覆盖 → 不加 footer
    # 设计哲学: 让 model "结构性不可能"在文件没改时声称改了
    # 启用条件: _file_mutation_verifier_enabled() (用户配置)
    # 位置: 在 final_response 已经有内容且没被中断时,附加到末尾
    # ═══════════════════════════════════════════════════════════════
    if final_response and not interrupted:
        try:
            # 4.114.1 取本 turn 失败的文件 mutation 记录
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            # 4.114.2 有失败 + 用户启用 verifier → 加 footer
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    # 4.114.3 把 footer 拼到 final_response 末尾
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.115】Turn-completion Explainer — 兜底解释 "为什么没输出"
    # 用户痛点(issue #34452): turn 结束后 response box 是空的或残缺
    # 触发场景:
    #   - 多次重试后放弃
    #   - 流式被截断 (partial/truncated)
    #   - 还有 tool result 在排队 (still-pending)
    #   - 跑满 max_iterations / iteration_budget
    # 设计原则 (gate carefully):
    #   - text_response(...) 路径正常退出 → 不解释 (避免噪音)
    #   - 真的没有可用回复时:
    #     * 空响应
    #     * "(empty)" 终止哨兵
    #     * 短到不像完整答案 (无终止标点)
    #   - 否则保留 model 原文
    # 来源: _turn_exit_reason (12 种退出原因之一)
    # 与 file-mutation footer 同一模式: 兜底显示真相,不依赖 model 自报
    # ═══════════════════════════════════════════════════════════════
    if not interrupted:
        try:
            # 4.115.1 用户启用 explainer 才执行
            if agent._turn_completion_explainer_enabled():
                # 4.115.2 剥空白
                _stripped = (final_response or "").strip()
                # 4.115.3 是否"空终止" (空 / "(empty)")
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                # 4.115.4 是否"短残片" (无终止标点 ≤24字符)
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:] not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                # 4.115.5 真空 OR 短残片 → 加解释
                if _is_empty_terminal or _is_partial_fragment:
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason
                    )
                    if _explanation:
                        if _is_empty_terminal:
                            # 4.115.5a 真空 → 用解释替换
                            final_response = _explanation
                        else:
                            # 4.115.5b 短残片 → 保留原文 + 追加解释
                            final_response = (
                                _stripped + "\n\n" + _explanation
                            )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)


    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.116】response_transformed 标记初始化
    # 用途: 给 transform 钩子用,标记 final_response 是否被插件改过
    # ═══════════════════════════════════════════════════════════════
    _response_transformed = False

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.117】transform_llm_output 钩子(插件可改写响应)
    # 触发时机: tool-calling 循环跑完后
    # 用途: 插件可改写 model 输出(比如翻译、加签名、过滤敏感词)
    # 规则: 第一个返回非空字符串的钩子胜出
    #       None/空字符串 → 不变
    # ═══════════════════════════════════════════════════════════════
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            # 4.117.1 触发钩子(可能多个插件)
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            # 4.117.2 第一个非空字符串胜出
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.118】post_llm_call 钩子(插件可观察整个 turn)
    # 触发时机: tool-calling 循环跑完后
    # 用途: 插件可同步对话到外部系统(如 Notion / Slack / 自家 DB)
    # 关键: 不修改响应,只观察
    # ═══════════════════════════════════════════════════════════════
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            # 4.118.1 触发 post_llm_call 钩子
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.119】提取本 turn 的 reasoning(给 UI 显示)
    # 关键约束:
    #   1. 只在本 turn 内找(到 user 消息就停)
    #   2. 取最近的非空 reasoning
    #   3. 不能跨 turn (防止显示上 turn 的 reasoning)
    # 为什么取最近: 同一 turn 内可能多次调用 tool
    #   tool-call 步骤的 reasoning 是最新的
    #   final-answer 步骤的 reasoning 可能是 None
    # ═══════════════════════════════════════════════════════════════
    last_reasoning = None
    # 4.119.1 倒着扫(最近的优先)
    for msg in reversed(messages):
        # 4.119.2 遇到 user 消息 → 停(turn 边界)
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        # 4.119.3 找最近的有 reasoning 的 assistant
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.120】构造 return 的 result 字典(33+ 字段)
    # 这个 dict 会被上层(cli.py / gateway)消费
    # 字段分类:
    #   - 响应相关: final_response / last_reasoning / response_transformed
    #   - 状态相关: completed / partial / interrupted / failed / turn_exit_reason
    #   - 计数相关: api_calls / 8 个 token 累加器 / cost
    #   - 元数据: model / provider / base_url / session_id
    # ═══════════════════════════════════════════════════════════════
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        # 4.120.1 5 个新 token 累加器
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        # 4.120.2 3 个老 token 累加器(向后兼容)
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        # 4.120.3 压缩相关
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        # 4.120.4 成本相关
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        "session_id": agent.session_id,
    }
    # 4.120.5 工具 guardrail 触发 → 记录 metadata
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.121】处理 /steer 残留(交给 caller 当下 turn)
    # 触发场景: 用户在 agent 跑完后发 /steer(新指令)
    #          但已经没有 tool 批次可以让它注入
    # 处理: 把 steer 排空,放 result 里让 caller 处理
    # ═══════════════════════════════════════════════════════════════
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    # 4.121.1 重置 previewed 标记
    agent._response_was_previewed = False

    # Include interrupt message if one triggered the interrupt
    # 4.121.2 中断时附上中断消息
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    # 4.121.3 清中断状态
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    # 4.121.4 清 stream callback(防泄漏)
    agent._stream_callback = None

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.122】检查 skill nudge 触发条件
    # 触发条件: _skill_nudge_interval > 0 + iters 累积到阈值 + 有 skill_manage
    # 触发后: 重置 _iters_since_skill,后台生成 skill review
    # ═══════════════════════════════════════════════════════════════
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        # 4.122.1 触发 → 重置计数
        _should_review_skills = True
        agent._iters_since_skill = 0

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.123】外部 memory provider 同步
    # 把本 turn 的 final_response 同步到外部存储
    # 同时 queue 下次 prefetch
    # ═══════════════════════════════════════════════════════════════
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.124】后台 memory / skill review
    # 触发条件: 有 final_response + 没中断 + (memory review OR skill review 需要)
    # 关键: 在响应**送出之后**才跑(不抢用户任务的 model 注意力)
    # 失败: best-effort,失败不报错
    # ═══════════════════════════════════════════════════════════════
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort


    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.125】on_session_end 钩子(整个 run_conversation 最后一步)
    # 触发时机: 每次 run_conversation 调用结束
    # 用途: 插件可清理资源、刷 buffer、记录结束事件
    # 注意: 这里**不**调 on_session_end() / shutdown_all()
    #       因为 run_conversation 在多 turn session 里调一次
    #       不能在中间 turn 杀掉 provider
    # ═══════════════════════════════════════════════════════════════
    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        # 4.125.1 触发 on_session_end 钩子
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            completed=completed,
            interrupted=interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.126】return result(run_conversation 结束)
    # result 含 33+ 字段:
    #   - final_response(用户要看的)
    #   - messages(历史)
    #   - 5 + 3 个 token 累加器
    #   - 状态(completed / partial / interrupted)
    #   - 元数据(model / provider / session_id)
    # 上层(cli.py / gateway)消费这个 dict 决定下一步动作
    # ═══════════════════════════════════════════════════════════════
    return result



# ═══════════════════════════════════════════════════════════════
# 【模块导出】__all__ = ["run_conversation"]
# 含义: from conversation_loop import * 只能拿到 run_conversation
# 隐藏内部辅助函数(避免污染命名空间)
# ═══════════════════════════════════════════════════════════════
__all__ = ["run_conversation"]

