#!/usr/bin/env python3
"""
Model Tools Module

# === 这个文件是干什么的? ===
# 是 tools/registry.py 上面的一层"对外门面",也是 agent 主循环调用工具的入口。
# 每个工具文件在 tools/ 目录下,import 时调 registry.register() 自注册;
# 这个模块在模块加载时(import 阶段)就触发 discover_builtin_tools() 把
# 80+ 个工具文件都 import 一遍,让它们把 schema/handler 塞进全局 registry。
#
# === 跟 registry.py 的关系 ===
# tools/registry.py    底层:数据存储 + 增删改查
# model_tools.py      上层:对外门面(主循环、CLI、RL 环境都 import 这个)
# 关系:本文件 import registry,反过来 registry 通过 lazy import
#      调本文件的 _run_async(避免循环 import)
#
# === 公共 API(签名保持兼容老 2400 行版本) ===
#     get_tool_definitions(enabled, disabled, quiet)         # 给 LLM 拼 tool schema
#     handle_function_call(name, args, task_id, user_task)   # 执行工具
#     TOOL_TO_TOOLSET_MAP / TOOLSET_REQUIREMENTS             # 兼容老代码的模块级常量
#     get_all_tool_names() / get_toolset_for_tool(name) ...  # 查询 helper
#
# === 三大设计要点(看完整个文件回头看) ===
# 1. Async 桥接:_run_async 把 async handler 塞进同步主循环
# 2. 多层缓存:tool defs 缓存 + check_fn TTL 缓存 + generation counter
# 3. 容错:参数类型纠正、错误清洗、未知工具降级
"""

import os
import json
import re
import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import discover_builtin_tools, registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================
#
# === 为什么需要这一层? ===
# 1. agent 主循环(run_agent.py)是同步的,但有些工具(handler)是 async def
# 2. 主循环直接 await 会卡死(没有运行中的 loop)
# 3. asyncio.run() 又会"建 loop → 跑 → 关 loop",导致缓存的 httpx/AsyncOpenAI
#    客户端在 GC 时尝试 close 它们的 transport,transport 绑在已死的 loop 上
#    → 抛 "Event loop is closed"
#
# === 解决方案:持久化的 loop ===
# 主线程一个 loop(全程不关),工作线程(并行工具执行)每个线程一个 loop
# 这样异步客户端的 transport 一直绑在"活的 loop"上,GC 清理时不会炸。

# 主线程的持久 loop(被多个工具调用共享)
_tool_loop = None
_tool_loop_lock = threading.Lock()

# 工作线程的持久 loop(threading.local 让每个线程有自己的一份)
_worker_thread_local = threading.local()


def _get_tool_loop():
    """Return a long-lived event loop for running async tool handlers.

    Using a persistent loop (instead of asyncio.run() which creates and
    *closes* a fresh loop every time) prevents "Event loop is closed"
    errors that occur when cached httpx/AsyncOpenAI clients attempt to
    close their transport on a dead loop during garbage collection.

    # === 双重检查锁 ===
    # 先无锁判断 → 锁内再判断 → 创建/复用
    # 锁粒度:只保护"创建 loop"这一句,后续调用 99% 走快路径(无锁)
    """
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Return a persistent event loop for the current worker thread.

    Each worker thread (e.g., delegate_task's ThreadPoolExecutor threads)
    gets its own long-lived loop stored in thread-local storage.  This
    prevents the "Event loop is closed" errors that occurred when
    asyncio.run() was used per-call: asyncio.run() creates a loop, runs
    the coroutine, then *closes* the loop — but cached httpx/AsyncOpenAI
    clients remain bound to that now-dead loop and raise RuntimeError
    during garbage collection or subsequent use.

    By keeping the loop alive for the thread's lifetime, cached clients
    stay valid and their cleanup runs on a live loop.

    # === threading.local 的妙用 ===
    # 每个 worker 线程第一次调这个函数时,threading.local 给它分配一个属性"loop"。
    # 后续调用直接从 thread-local 拿,O(1) 不需要任何全局锁。
    # 线程结束时 loop 不主动关(让它跟线程一起死)。
    #
    # === asyncio.set_event_loop 的作用 ===
    # 让 `asyncio.get_event_loop()` 在这个线程里也返这个 loop,
    # 兼容"老代码隐式假设线程有默认 loop"的场景。
    """
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.

    For the common CLI path (no running loop), we use a persistent event
    loop so that cached async clients (httpx / AsyncOpenAI) remain bound
    to a live loop and don't trigger "Event loop is closed" on GC.

    When called from a worker thread (parallel tool execution), we use a
    per-thread persistent loop to avoid both contention with the main
    thread's shared loop AND the "Event loop is closed" errors caused by
    asyncio.run()'s create-and-destroy lifecycle.

    This is the single source of truth for sync->async bridging in tool
    handlers. Each handler is self-protecting via this function.

    # === 三分支:按"当前线程的环境"决定怎么跑 coroutine ===
    # 分支 1:当前线程已经有 running loop(gateway / RL env)
    #         → 不能 reuse(冲突),扔到一次性 thread,自建 loop,跑完销毁
    # 分支 2:在 worker 线程(非主线程)上
    #         → 用线程本地的持久 loop(避免和主线程 loop 抢,也避免 asyncio.run 销毁)
    # 分支 3:在主线程,没有 running loop(最常见的 CLI 路径)
    #         → 用主线程的持久 loop
    """
    # === 探测"当前线程有没有正在运行的 loop" ===
    # get_running_loop() 在没有 loop 的线程里会抛 RuntimeError
    # 用 try/except 兜底,跟 Rust 的 unwrap_or 一样
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # === 分支 1:在 async 上下文里(gateway / RL env) ===
        # 不能再用 caller 的 loop(会冲突 / 死锁)
        # 也不能用主线程的持久 loop(同一个进程 = 同一个 loop)
        # 解决:开一个一次性 ThreadPoolExecutor,里面 new_event_loop
        # 优点:这个 worker loop 由我们持有引用,timeout 时能 cancel 它
        #      (以前用 ThreadPoolExecutor.cancel() 只能取消未启动的 future,
        #      对 running 任务无效,导致 300s 超时每次都泄漏一个线程)
        import concurrent.futures

        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        # Event 用于同步"worker loop 准备好了",避免 timeout cancel 时 race condition
        loop_ready = threading.Event()

        def _run_in_worker():
            nonlocal worker_loop
            worker_loop = asyncio.new_event_loop()
            loop_ready.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(coro)
            finally:
                try:
                    # 把所有还在 pending 的 task 都 cancel,然后跑一次 gather 等它们真死
                    # 这样 worker_loop 关闭时不会有"半挂"任务
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run_in_worker)
        try:
            # 5 分钟硬超时(300s)——给长时间运行的工具足够的预算
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # === 超时处理 ===
            # 等 worker 起来(wait 是 blocking 但最多 1 秒),拿到它的 loop 引用
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    # call_soon_threadsafe 从别的线程安全地调度 cancel
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    # Loop 已经关了,啥都不用做
                    pass
            raise
        finally:
            # wait=False:不阻塞 caller 等待线程结束。我们已经发起 cancel 了,
            # 线程会在 coroutine 下次 await 时退出。
            pool.shutdown(wait=False)

    # === 分支 2:在 worker 线程(非主线程)上 ===
    # 比如 delegate_task 在 ThreadPoolExecutor 里跑并行工具。
    # 用线程本地的持久 loop——既不和主线程抢,也避免 asyncio.run 的销毁问题
    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    # === 分支 3:在主线程,无 running loop(常见 CLI 路径) ===
    # 走持久 loop,这样 async 客户端的 transport 一直绑在活 loop 上
    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP tool discovery (external MCP servers from config) used to run here as
# a module-level side effect.  It was removed because discover_mcp_tools()
# internally uses a blocking future.result(timeout=120) wait, and the
# gateway lazy-imports this module from inside the asyncio event loop on
# the first user message — freezing Discord/Telegram heartbeats for up to
# 120s whenever any configured MCP server was slow or unreachable (#16856).
#
# Each entry point now runs discovery explicitly at its own startup:
#   - gateway/run.py            -> start_gateway() uses run_in_executor
#   - cli.py, hermes_cli/*      -> inline on startup (no event loop)
#   - tui_gateway/server.py     -> inline on startup (no event loop)
#   - acp_adapter/server.py     -> asyncio.to_thread on session init

# Plugin tool discovery (user/project/pip plugins)
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================
#
# === 这两个模块级常量是给老代码吃的"形状兼容层" ===
# 老 API 期望 `model_tools.TOOL_TO_TOOLSET_MAP` / `model_tools.TOOLSET_REQUIREMENTS`
# 这两个常量在模块加载时一次性算出来。
#
# ⚠️ 静态:这两个变量在 import 时一次性算。如果用户中途 register 新工具,
# 这个常量不会自动更新。所以新代码应该直接调 registry.get_*() 方法。
# 留着这两个只是为了不破坏老 API。

# {tool_name: toolset_name} 的扁平映射——给 batch_runner.py 用
TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

# {toolset_name: {name, env_vars, check_fn, setup_url, tools}}——给 cli.py / doctor.py 用
TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# === "上一次"解析出的工具名 ===
# _last_resolved_tool_names 在 get_tool_definitions() 每次被调时更新。
# 谁在看?code_execution_tool——它需要在沙箱里"列举当前 session 可用工具",
# 跟主进程保持一致。模块级变量是"最近一次"的快照,够用。
_last_resolved_tool_names: List[str] = []


# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================
#
# === 老 API 的 toolset 名 → 新 API 的工具名列表 ===
# 早期(2400 行版本)用"功能区"作为分组单位,名字带 _tools 后缀:
#   "web_tools" 包含 ["web_search", "web_extract"]
#   "browser_tools" 包含 ["browser_navigate", "browser_click", ...]
# 新 API 直接用 toolset 名(单数,不带 _tools),toolsets/ 模块里定义。
#
# 这张表是给老配置文件 / 老用户命令用的"翻译器":
#   用户说"启用 web_tools" → 查表 → 真正启用 ["web_search", "web_extract"]
_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "moa_tools": ["mixture_of_agents"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================
#
# === 这个模块的"主入口" ===
# 整个 agent 的工具系统对外就这一个函数,主循环每 turn 都会调它:
#     tools = get_tool_definitions(enabled, disabled, quiet=True)
#     messages = build_messages(..., tools=tools, ...)
# 它决定"这一轮对话,LLM 看到哪些工具可选"。

# === 缓存 ===
# 缓存 key: (frozenset(enabled), frozenset(disabled), registry._generation)
# 缓存值: List[Dict]  # 完整的 OpenAI-format 工具 schema 列表
#
# 为什么需要缓存?
#   - 每次调都要走 registry 遍历 + check_fn 探测 + schema 过滤 ≈ 7ms
#   - gateway / AIAgent 每个 turn 都调一次(每秒可能几十次)
#   - 没缓存的话,光工具发现就吃光 CPU
#
# 何时失效?(三重保险)
#   1. registry._generation 变化(register / deregister 触发)
#   2. enabled/disabled 变化(用户改配置)
#   3. 配置文件 mtime 变化(dynamic schema 依赖配置)
#
# 为什么只在 quiet_mode=True 时缓存?
#   quiet_mode=False 时,函数内部会 print 工具选择过程(给用户看"我用了哪些工具")。
#   缓存后这部分打印就跳过——破坏 UX。所以非 quiet 模式不缓存。
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}


def _clear_tool_defs_cache() -> None:
    """Drop memoized get_tool_definitions() results. Called when dynamic
    schema dependencies change (e.g. discord capability cache reset,
    execute_code sandbox reconfigured).

    # === 什么时候外部要主动失效? ===
    # 当 dynamic_schema_overrides 依赖的状态变了,但配置文件没变时:
    #   - discord 的 capability 缓存被重置
    #   - execute_code 的 sandbox 模式从 local 切到 docker
    #   - 任何不在配置文件 mtime 范围内的"运行时状态"变更
    """
    _tool_defs_cache.clear()


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with toolset-based filtering.

    All tools must be part of a toolset to be accessible.

    Args:
        enabled_toolsets: Only include tools from these toolsets.
        disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
        quiet_mode: Suppress status prints.
        skip_tool_search_assembly: When True, return the pre-assembly tool list
            (raw schemas for every enabled tool). Used internally by the
            tool_search / tool_describe bridge handlers so they can read the
            real catalog, not the already-collapsed one. Public callers should
            leave this False.

    Returns:
        Filtered list of OpenAI-format tool definitions.

    # === 这个函数的核心职责 ===
    # 1. 按 enabled/disabled toolset 过滤可用工具
    # 2. 应用 tool_search 组装(当工具太多,只暴露"搜索"入口)
    # 3. 应用 LCM 标签筛选(本地模型减负)
    # 4. 返回 OpenAI 格式 schema 列表
    #
    # === 缓存策略 ===
    # quiet_mode=True 时走缓存(性能),quiet_mode=False 时不走(要 print)。
    # 缓存 key 包含 6 个分量,任一变化都让缓存自动失效。
    """
    # Fast path: memoized result when the caller doesn't need stdout prints.
    # The cache key captures every argument-level input; the registry
    # generation captures registry mutations (MCP refresh, plugin load).
    # check_fn results are TTL-cached one level down, inside
    # registry.get_definitions. The config-mtime fingerprint below captures
    # user-visible config edits that affect dynamic schemas (execute_code
    # mode, discord action allowlist, etc.) without needing an explicit
    # invalidate hook on every config-writer.
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path
            cfg_path = get_config_path()
            cfg_stat = cfg_path.stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
            bool(os.environ.get("HERMES_KANBAN_TASK")),
            bool(skip_tool_search_assembly),
        )
        cached = _tool_defs_cache.get(cache_key)
        if cached is not None:
            # === 缓存命中 ===
            # 同样要更新 _last_resolved_tool_names,让下游 caller
            # (比如 code_execution_tool)看到一致状态
            global _last_resolved_tool_names
            _last_resolved_tool_names = [t["function"]["name"] for t in cached]
            # Return a shallow copy of the list but share the dict references —
            # schemas are treated as read-only by all known callers.
            # 为什么 shallow copy?caller 可能 append / extend 这个 list,
            # 我们不想污染缓存。
            return list(cached)

    result = _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode,
                                       skip_tool_search_assembly=skip_tool_search_assembly)
    if quiet_mode:
        # === 缓存未命中 → 算 → 写缓存 → 返 shallow copy ===
        # 同样要 shallow copy:issue #17335 修过这个 bug。
        # 老代码会 `self.tools = get_tool_definitions(...)` 然后 `self.tools.append(memory_tool)`。
        # 如果不 copy,append 的结果会污染缓存,
        # 下一个 turn 拿到的是"已经 append 过 memory_tool 的版本",
        # 再 append 又多一个,Gateway 长期跑下来,工具名重复累积,
        # DeepSeek / MiMo / Kimi 这种"严格去重"的 provider 就会 400 报错。
        # Cache the freshly-computed list, but hand callers a shallow copy so
        # downstream mutations (e.g. run_agent appending memory/LCM tool
        # schemas to self.tools) don't poison the cache. Without this, a
        # long-lived Gateway process accumulates duplicate tool names across
        # agent inits and providers that enforce unique tool names
        # (DeepSeek, Xiaomi MiMo, Moonshot Kimi) reject the request with
        # HTTP 400. Mirrors the cache-hit path above. (issue #17335)
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`.

    # === 这个函数分 7 步 ===
    # 1. 解析 enabled_toolsets → 候选工具集合(用 set 防重)
    # 2. 应用 disabled_toolsets 做差集(后减,避免 disabled 被 enabled 覆盖)
    # 3. 调 registry.get_definitions(自动跑 check_fn 过滤)
    # 4. 动态调整:execute_code / discord / browser 等 schema(随运行时配置)
    # 5. print 一行总览
    # 6. 全局 sanitize(给 llama.cpp 兼容)
    # 7. Tool Search 组装(工具太多 → 用 3 个桥接工具代理)
    #
    # === Kanban worker 强制注入 ===
    # 看到环境变量 HERMES_KANBAN_TASK 存在,就强制把 "kanban" 工具集
    # 加进 enabled——dispatcher spawn 的 worker 必须能 complete / block / heartbeat,
    # 即使 assignee profile 限制了其他 toolset。
    """
    # ════════════════════════════════════════════════════════════════
    # Step 1: 解析 enabled_toolsets → 候选工具集合
    # ════════════════════════════════════════════════════════════════
    # 用 set 防重;老 API 名(_tools 后缀)走 _LEGACY_TOOLSET_MAP 翻译;
    # 未知 toolset 仅 print warning,不抛异常(容错)
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        # === Kanban worker 强制注入(见 docstring) ===
        effective_enabled_toolsets = list(enabled_toolsets)
        if os.environ.get("HERMES_KANBAN_TASK") and "kanban" not in effective_enabled_toolsets:
            # Dispatcher-spawned workers are scoped by HERMES_KANBAN_TASK and
            # must always receive the lifecycle handoff tools. Assignee
            # profiles may intentionally restrict their normal chat toolsets
            # (for token/cost reasons), but that should not strip the kanban
            # worker's completion/block/heartbeat surface.
            effective_enabled_toolsets.append("kanban")
        for toolset_name in effective_enabled_toolsets:
            # 优先用新 API 验证 + 解析 toolset 名
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            # 兜底:老 API 名字(_tools 后缀)查 _LEGACY_TOOLSET_MAP
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        # === 默认:全部工具都开 ===
        # 比如用户没指定 enabled,就把所有已注册 toolset 的工具都拉进来
        # Default: start with everything
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # ════════════════════════════════════════════════════════════════
    # Step 2: 应用 disabled_toolsets 做差集
    # ════════════════════════════════════════════════════════════════
    # 关键:disabled 是后减,不是"先减"
    # 顺序是 enabled ∪ ... \ disabled
    # 这样如果用户 enabled 了一个大 toolset(包含很多子工具),但同时
    # disabled 了其中某些具体 toolset,disabled 的会"穿透"出来。
    # issue #17309 修过这个 bug:之前 disabled 在 enabled 之前算,
    # 导致 enabled 里的"覆盖了"disabled 的工具集。
    # Always apply disabled toolsets as a subtraction step at the end.
    # This ensures that even if a composite toolset (like hermes-cli)
    # is enabled, any tools belonging to a disabled toolset are strictly
    # stripped out. See issue #17309.
    if disabled_toolsets:
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")

    # ════════════════════════════════════════════════════════════════
    # Step 3: 调 registry.get_definitions(自动跑 check_fn 过滤)
    # ════════════════════════════════════════════════════════════════
    # registry 会做:
    #   1. 从 set 里取每个 tool name
    #   2. 跑 entry.check_fn() → 失败的工具被过滤(无 check_fn 的默认通过)
    #   3. 拼 OpenAI 格式 `{"type": "function", "function": {...}}`
    #   4. 应用 dynamic_schema_overrides(运行时改 schema)
    # 返回 filtered_tools: 只包含 check_fn 通过的工具 schema 列表
    #
    # Plugin-registered tools are now resolved through the normal toolset
    # path — validate_toolset() / resolve_toolset() / get_all_toolsets()
    # all check the tool registry for plugin-provided toolsets.  No bypass
    # needed; plugins respect enabled_toolsets / disabled_toolsets like any
    # other toolset.

    # Ask the registry for schemas (only returns tools whose check_fn passes)
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    # The set of tool names that actually passed check_fn filtering.
    # Use this (not tools_to_include) for any downstream schema that references
    # other tools by name — otherwise the model sees tools mentioned in
    # descriptions that don't actually exist, and hallucinates calls to them.
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # ════════════════════════════════════════════════════════════════
    # Step 4: 动态 schema 调整(随运行时配置 / 环境探测)
    # ════════════════════════════════════════════════════════════════
    # 这一步要解决的核心问题:
    #   工具的 description / parameters 里有"提到其他工具名"的地方
    #   (比如 execute_code 说"可以调 web_search / web_extract")。
    #   如果提到的工具实际上不可用(check_fn 不过、没启用),模型会
    #   调一个"幻觉工具"。所以要按 available_tool_names 动态改写。
    #
    # 涉及 3 个工具:execute_code / discord / discord_admin / browser_navigate

    # ════════════════════════════════════════════════════════════════
    # Step 4a: 重构 execute_code 的 schema
    # ════════════════════════════════════════════════════════════════
    # 静态 schema 列出"沙箱内可调的工具"(默认全开),但实际可能
    # 有些工具没装(API key 缺)/ 被 disabled。按 available_tool_names
    # 重写,只暴露真正可用的。否则模型会在沙箱里调一个空工具。
    # issue:#560-discord
    # Rebuild execute_code schema to only list sandbox tools that are actually
    # available.  Without this, the model sees "web_search is available in
    # execute_code" even when the API key isn't configured or the toolset is
    # disabled (#560-discord).
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled, mode=_get_execution_mode())
        for i, td in enumerate(filtered_tools):
            if td.get("function", {}).get("name") == "execute_code":
                filtered_tools[i] = {"type": "function", "function": dynamic_schema}
                break

    # ════════════════════════════════════════════════════════════════
    # Step 4b: 重构 discord / discord_admin 的 schema
    # ════════════════════════════════════════════════════════════════
    # 静态 schema 列出所有 discord action,但实际:
    #   - 某些 action 需要 bot 有特定 intent(去 GET /applications/@me 探测)
    #   - 用户在 config 里可能设了"允许的 action 白名单"
    # 动态 schema 只暴露 bot 真正能用、用户允许用的 action,
    # 否则模型会调一个 Discord API 拒绝的 action。
    # Rebuild discord / discord_admin schemas based on the bot's privileged
    # intents (detected from GET /applications/@me) and the user's action
    # allowlist in config.  Hides actions the bot's intents don't support so
    # the model never attempts them, and annotates fetch_messages when the
    # MESSAGE_CONTENT intent is missing.
    _discord_schema_fns = {
        "discord": "get_dynamic_schema_core",
        "discord_admin": "get_dynamic_schema_admin",
    }
    for discord_tool_name in _discord_schema_fns:
        if discord_tool_name in available_tool_names:
            try:
                from tools import discord_tool as _dt
                schema_fn = getattr(_dt, _discord_schema_fns[discord_tool_name])
                dynamic = schema_fn()
            except Exception:
                dynamic = None
            if dynamic is None:
                filtered_tools = [
                    t for t in filtered_tools
                    if t.get("function", {}).get("name") != discord_tool_name
                ]
                available_tool_names.discard(discord_tool_name)
            else:
                for i, td in enumerate(filtered_tools):
                    if td.get("function", {}).get("name") == discord_tool_name:
                        filtered_tools[i] = {"type": "function", "function": dynamic}
                        break

    # ════════════════════════════════════════════════════════════════
    # Step 4c: 清理 browser_navigate 的 description
    # ════════════════════════════════════════════════════════════════
    # 静态 description 说"推荐用 web_search / web_extract(更快更便宜)"
    # 但如果这两个工具没开,模型会幻觉去调它们。
    # 修法:从 description 里把这一句删掉。
    # Strip web tool cross-references from browser_navigate description when
    # web_search / web_extract are not available.  The static schema says
    # "prefer web_search or web_extract" which causes the model to hallucinate
    # those tools when they're missing.
    if "browser_navigate" in available_tool_names:
        web_tools_available = {"web_search", "web_extract"} & available_tool_names
        if not web_tools_available:
            for i, td in enumerate(filtered_tools):
                if td.get("function", {}).get("name") == "browser_navigate":
                    desc = td["function"].get("description", "")
                    desc = desc.replace(
                        " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                        "",
                    )
                    filtered_tools[i] = {
                        "type": "function",
                        "function": {**td["function"], "description": desc},
                    }
                    break

    # ════════════════════════════════════════════════════════════════
    # Step 5: print 一行总览(给用户看"这一轮用了哪些工具")
    # ════════════════════════════════════════════════════════════════
    # 非 quiet 模式下,在终端 print 一行 🛠️ emoji 开头的总结
    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    # 把"这一轮解析出的工具名"塞进模块级变量,给 code_execution_tool 看
    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    # ════════════════════════════════════════════════════════════════
    # Step 6: 全局 sanitize(给 llama.cpp 兼容)
    # ════════════════════════════════════════════════════════════════
    # 不同后端对 JSON schema 的容忍度不一样:
    #   - 云厂商(OpenAI/Anthropic)宽容,基本啥都接
    #   - llama.cpp 严格:用 schema 编译 GBNF 语法,某些 shape 直接拒
    #     例如 bare `"type": "object"` 没 properties 会被拒
    #     字符串值的 schema 节点(从某些有 bug 的 MCP server 来)也会被拒
    #
    # sanitize_tool_schemas 把这些 corner case 规范化掉。
    # 对已合规的 schema 是 no-op。
    # Sanitize schemas for broad backend compatibility. llama.cpp's
    # json-schema-to-grammar converter (used by its OAI server to build
    # GBNF tool-call parsers) rejects some shapes that cloud providers
    # silently accept — bare "type": "object" with no properties,
    # string-valued schema nodes from malformed MCP servers, etc. This
    # is a no-op for schemas that are already well-formed.
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    # ════════════════════════════════════════════════════════════════
    # Step 7: Tool Search 组装(工具太多 → 用 3 个桥接工具代理)
    # ════════════════════════════════════════════════════════════════
    # 背景:用户开了 50+ 工具(MCP 拉一堆 server + 各种 plugin),
    # 把所有 schema 灌进 prompt 一次,光工具列表就吃掉 10%+ context。
    # 解法:渐进式披露(progressive disclosure)
    #   - 核心工具(toolsets._HERMES_CORE_TOOLS)永远直接给 LLM
    #   - 周边工具(MCP / plugin)替换成 3 个桥接工具:
    #       tool_search(query) → 找匹配的工具
    #       tool_describe(name) → 看某个工具的 schema
    #       tool_call(name, args) → 间接调用那个工具
    #   - 阈值:deferrable tools 的总 token > 10% context 时才启用
    #
    # 故意放在最后一步:前面 sanitize 已经规范化了 schema,组装 idempotent
    # (调两次结果一样)。
    # ── Tool Search (progressive disclosure) ────────────────────────────
    # Conditionally replace MCP + plugin (non-core) tools with three bridge
    # tools (tool_search / tool_describe / tool_call) when the deferrable
    # surface exceeds the configured threshold (default 10% of context
    # window). Core Hermes tools (toolsets._HERMES_CORE_TOOLS) are NEVER
    # deferred. See tools/tool_search.py for full design notes.
    #
    # This is deliberately the last step before returning — sanitization
    # has already normalized schemas, and the assembly is idempotent in
    # case some caller invokes get_tool_definitions twice.
    try:
        from tools.tool_search import assemble_tool_defs, load_config as _load_ts_config
        ts_cfg = _load_ts_config()
        if not skip_tool_search_assembly and ts_cfg.enabled != "off":
            context_length = _resolve_active_context_length()
            assembly = assemble_tool_defs(
                filtered_tools,
                context_length=context_length,
                config=ts_cfg,
            )
            if assembly.activated and not quiet_mode:
                print(
                    f"🔎 Tool Search: {assembly.deferred_count} MCP/plugin tools deferred "
                    f"(~{assembly.deferred_tokens} tokens) behind tool_search/describe/call. "
                    f"Threshold ~{assembly.threshold_tokens} tokens."
                )
            filtered_tools = assembly.tool_defs
    except Exception as e:  # pragma: no cover — never break tool loading
        logger.warning("Tool search assembly skipped: %s", e)

    return filtered_tools


def _resolve_active_context_length() -> int:
    """Look up the active model's context length for the tool-search gate.

    Returns 0 when the model can't be resolved — ``should_activate`` falls
    back to a fixed token cutoff in that case.

    # === 作用:算出"当前模型 context 窗口有多大" ===
    # 用来给 Tool Search 决策:
    #   deferrable tools 的总 token > 10% context  → 激活 tool search
    #
    # 流程:
    #   1. 从 config.yaml 拿 model.model 字段(或 model.default 兜底)
    #   2. 用 model_id 查 model_metadata 表
    #   3. 拿不到 → 返回 0(让 tool_search 走固定 token 兜底)
    #
    # 异常全 catch,debug 级别 log:
    #   任何一步炸了都不影响主流程(没有 context length 就用兜底)
    """
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        model_id = (model_cfg.get("model") or model_cfg.get("default") or "").strip()
        if not model_id:
            return 0
        from agent.model_metadata import get_model_context_length
        return int(get_model_context_length(model_id) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools whose execution is intercepted by the agent loop (run_agent.py)
# because they need agent-level state (TodoStore, MemoryStore, etc.).
# The registry still holds their schemas; dispatch just returns a stub error
# so if something slips through, the LLM sees a sensible message.
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# Tool error sanitization
# =========================================================================
#
# === 为什么需要? ===
# 工具抛异常时,异常消息(str(e) 可能含任何东西:用户输入、文件内容、
# 网络响应)会作为 `tool` 消息的 content 灌进 model context。
#
# 风险 1:json.dumps 已经做了 quote/backslash 转义,所以原始
#         `</tool_call>` 这种 token 不会破坏 message 框架(框架层安全)。
# 风险 2:但 model 会"读"这些 token——它可能误以为"这是结构化标记",
#         在解析时出错,或者被 adversarial 攻击诱导到 role-confusion
#         (假装自己是 user/system 角色)。
#
# 防御:把 XML role tag、markdown fence、CDATA 这种"结构化 token"
#      全 strip 掉,并加 [TOOL_ERROR] 前缀让 model 知道这是错误段。
# 是 defense-in-depth,便宜划算。
# Ported from ironclaw#1639.

# 4 类需要清洗的 framing token:
# 1. role tag: <tool_call> / </function_call> / <system> ...
_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>',
    re.IGNORECASE,
)
# 2. 开头的代码 fence: ```json / ```xml / ```html / ```markdown / ```
_TOOL_ERROR_FENCE_OPEN_RE = re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE)
# 3. 结尾的代码 fence
_TOOL_ERROR_FENCE_CLOSE_RE = re.compile(r'\s*```\s*$', re.MULTILINE)
# 4. CDATA 块(可能含隐藏的 XML/HTML 注入)
_TOOL_ERROR_CDATA_RE = re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL)
# 长度上限 2000——避免 1 个错误吃掉大量 context
_TOOL_ERROR_MAX_LEN = 2000


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before showing it to the model.

    See _TOOL_ERROR_ROLE_TAG_RE docstring above for rationale.

    # === 清洗流程(5 步) ===
    # 1. 空字符串 → 直接返 "[TOOL_ERROR] "(占位)
    # 2. strip role tag
    # 3. strip 开 / 闭 fence
    # 4. strip CDATA
    # 5. 超过 2000 字符 → 截断 + "..."
    # 6. 加 [TOOL_ERROR] 前缀
    #
    # 注:前缀 [TOOL_ERROR] 是给 model 的"语义标签",
    # 告诉它"下面这段是工具报错,不是用户输入或你的输出"
    """
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_msg)
    sanitized = _TOOL_ERROR_FENCE_OPEN_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_FENCE_CLOSE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


# =========================================================================
# Tool argument type coercion
# =========================================================================
#
# === 为什么需要 coerce? ===
# 现实:LLM 经常传错类型
#   - 数字当字符串:  "42" 而不是 42
#   - 布尔当字符串: "true" 而不是 true
#   - 数组当裸标量: {"urls": "https://a.com"} 但 schema 要求 array
#     这种情况在 DeepSeek / Qwen / GLM 这些 open-weight 模型上特别常见
#
# 这一段:对照工具的 JSON Schema,安全地把字符串纠正成 schema 期望的类型。
# 纠正失败 → 保留原值(不抛异常,让 tool handler 自己判断)
#
# 涉及:coerce_tool_args / _coerce_value / _schema_allows_null /
#      _coerce_json / _coerce_number / _coerce_boolean

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce tool call arguments to match their JSON Schema types.

    LLMs frequently return numbers as strings (``"42"`` instead of ``42``)
    and booleans as strings (``"true"`` instead of ``true``).  This compares
    each argument value against the tool's registered JSON Schema and attempts
    safe coercion when the value is a string but the schema expects a different
    type.  Original values are preserved when coercion fails.

    Handles ``"type": "integer"``, ``"type": "number"``, ``"type": "boolean"``,
    and union types (``"type": ["integer", "string"]``).

    Also wraps bare scalar values in a single-element list when the schema
    declares ``"type": "array"``.  Open-weight models (DeepSeek, Qwen, GLM)
    sometimes emit ``{"urls": "https://a.com"}`` when the tool expects
    ``{"urls": ["https://a.com"]}``; wrapping here avoids a confusing tool
    failure on what is otherwise a well-formed call.

    # === 流程(对 args 的每个 key) ===
    # 1. 从 schema.parameters.properties 拿这个 key 的 prop_schema
    # 2. 如果 schema 期望 array,值不是 list/tuple/None → 包成单元素 list
    # 3. 如果 schema 期望 integer/number/boolean,值是字符串 → _coerce_value
    # 4. 纠正成功(返回值不等于原值)→ 替换
    # 5. 纠正失败(返回原值)→ 保留不动
    #
    # === 为什么不"不纠正就抛异常"? ===
    # 工具 handler 可能有兜底逻辑(比如 read_file 的 normalize_read_pagination),
    # 传错类型让 handler 自己处理,比直接 400 报错更友好。
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # ════════════════════════════════════════════════════════════════
        # 子分支 A:schema 期望 array,值不是 list → 包成单元素 list
        # ════════════════════════════════════════════════════════════════
        # 字符串值先过 _coerce_value:
        #   - "[\"a\", \"b\"]" 这种 JSON 编码的数组会被 json.loads 解析
        #   - "null" 在 nullable schema 下会被转成 None
        # 否则单字符串就包成 ["字符串"]。
        #
        # None 自身保留(模型可能想"省略"或"空列表",工具 handler 有默认值兜底)
        # Wrap bare non-list values when the schema declares ``array``.
        # Strings still go through _coerce_value first so JSON-encoded
        # arrays (``'["a","b"]'``) get parsed and nullable ``"null"``
        # becomes ``None`` rather than ``["null"]``.
        # ``None`` itself is preserved — we don't know whether the model
        # meant "omit" or "empty list", and tools with sensible defaults
        # (e.g. read_file's normalize_read_pagination) already handle it.
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    # _coerce_value handled it (JSON-parsed list or
                    # nullable "null" → None).
                    args[key] = coerced
                    continue
                # If the string looks like a JSON array but _coerce_value
                # failed to parse it, warn clearly instead of silently wrapping.
                if value.strip().startswith("["):
                    logger.warning(
                        "coerce_tool_args: %s.%s looks like a JSON array string "
                        "but could not be parsed — model may have emitted a "
                        "JSON-encoded string instead of a native array. "
                        "Falling back to single-element list.",
                        tool_name, key,
                    )
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        # ════════════════════════════════════════════════════════════════
        # 子分支 B:非 array 的标量类型纠正
        # ════════════════════════════════════════════════════════════════
        # 只对字符串值纠正(数字 / 布尔已经是正确类型就跳过)
        # 没有 expected_type 且 schema 不是 nullable → 跳过(不知道往哪转)
        if not isinstance(value, str):
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced

    return args


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.

    # === 路由表(按 expected_type 分发) ===
    # "null"             → "null" 字符串 → None(nullable 才生效)
    # "integer"/"number" → _coerce_number(后者允许小数)
    # "boolean"          → _coerce_boolean("true"/"false")
    # "array"            → _coerce_json(str, list)  [JSON 解析]
    # "object"           → _coerce_json(str, dict)  [JSON 解析]
    # list(union)        → 逐个尝试,第一个成功的类型
    # 其他/失败          → 返回原值
    """
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in {"integer", "number"}:
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """Return True when a JSON Schema fragment explicitly permits null.

    # === JSON Schema 表达"可空"有 4 种写法 ===
    # 1. type: "null"(单独 null 类型)
    # 2. type: ["string", "null"] (联合类型,显式列 null)
    # 3. nullable: true (OpenAPI 风格,不是标准 JSON Schema)
    # 4. anyOf/oneOf 包含 {"type": "null"} (标准联合类型)
    # 4 种全部要识别,缺一个就漏判。
    """
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    if schema.get("nullable") is True:
        return True

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") == "null":
                return True

    return False


def _coerce_json(value: str, expected_python_type: type):
    """Parse *value* as JSON when the schema expects an array or object.

    Handles model output drift where a complex oneOf/discriminated-union schema
    causes the LLM to emit the array/object as a JSON string instead of a native
    structure.  Returns the original string if parsing fails or yields the wrong
    Python type.

    # === 什么场景会触发 ===
    # 复杂 schema(oneOf / discriminated union)经常让 LLM "偷懒"——把
    # 整个 array/object 当字符串输出("["a", "b"]")而不是原生结构。
    # 这种情况 _coerce_value 分流到 _coerce_json 处理:
    #   1. json.loads 尝试解析
    #   2. 解析成功 + 类型对 → 返解析后的对象
    #   3. 解析失败 或 类型不对 → 返原值(handler 看到字符串再处理)
    """
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "coerce_tool_args: failed to parse string as JSON for expected type %s: %s",
            expected_python_type.__name__,
            exc,
        )
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    logger.warning(
        "coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
        type(parsed).__name__,
        expected_python_type.__name__,
    )
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number.  Returns original string on failure.

    # === 三道防御 ===
    # 1. float() 解析失败(非数字字符串)→ 返原值
    # 2. inf / -inf / NaN 不可 JSON 序列化 → 返原值(NaN 的 f != f 是个老 trick)
    # 3. integer_only=True 但有小数部分("3.14")→ 返原值
    #    (削掉小数会"无声地"改变语义,不如让 handler 自己处理)
    #
    # === 整数优先 ===
    # "42.0" 也返 42(int),因为 f == int(f) 判定为整数无小数
    """
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan — not JSON-serializable, keep original string
    if f != f or f == float("inf") or f == float("-inf"):
        return value
    # If it looks like an integer (no fractional part), return int
    if f == int(f):
        return int(f)
    if integer_only:
        # Schema wants an integer but value has decimals — keep as string
        return value
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean.  Returns original string on failure.

    # 严格只认 "true" / "false"(case-insensitive、strip 空白)
    # "yes" / "1" / "on" 不算——不引入模糊语义。
    # 失败就返原字符串(handler 可能另有判断逻辑)
    """
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    """
    Main function call dispatcher that routes calls to the tool registry.

    # === 整个 agent 调工具的"主入口" ===
    # 主循环(run_agent.py)拿到模型返回的 tool_call 就调这个函数。
    # 它要做 5 件事:
    #   1. 参数类型纠正("42" → 42)
    #   2. Tool Search bridge 处理(tool_search / tool_describe / tool_call)
    #   3. Agent-loop 工具拦截(todo / memory / session_search / delegate_task)
    #   4. pre/post hook(确认/审计)
    #   5. read_file 入口走 read_file_tool(read_file / search_files 合并)
    #   6. registry.dispatch()(实际执行 handler)
    #
    # === 11 个参数 ===
    # function_name / function_args   模型说的工具名 + 参数
    # task_id                         terminal/browser session 隔离 ID
    # tool_call_id                    OpenAI tool_call.id(用于 trace / 取消)
    # session_id                      agent session(让工具看 session 状态)
    # user_task                       原始用户任务(给 browser_snapshot 用上下文)
    # enabled_tools                   当前 session 启用的工具名(给 execute_code)
    # skip_pre_tool_call_hook         True 跳过 pre-hook(给内部桥接用)
    # enabled_toolsets / disabled_toolsets  给 Tool Search bridge 限定可见工具

    Args:
        function_name: Name of the function to call.
        function_args: Arguments for the function.
        task_id: Unique identifier for terminal/browser session isolation.
        user_task: The user's original task (for browser_snapshot context).
        enabled_tools: Tool names enabled for this session.  When provided,
                       execute_code uses this list to determine which sandbox
                       tools to generate.  Falls back to the process-global
                       ``_last_resolved_tool_names`` for backward compat.
        enabled_toolsets: The session's enabled toolsets.  Used to scope the
                       Tool Search bridge catalog so ``tool_search`` /
                       ``tool_describe`` / ``tool_call`` only see and invoke
                       tools the session was actually granted.  ``None`` means
                       "no restriction" (the caller scopes to every toolset),
                       matching ``get_tool_definitions`` semantics.
        disabled_toolsets: The session's disabled toolsets, applied as a
                       subtraction when scoping the bridge catalog.

    Returns:
        Function result as a JSON string.
    """
    # ════════════════════════════════════════════════════════════════
    # Step 1: 参数类型纠正
    # ════════════════════════════════════════════════════════════════
    # "42" → 42, "true" → True, ["42"] → [42] 等
    # 纠正失败保留原值(handler 兜底)
    # Coerce string arguments to their schema-declared types (e.g. "42"→42)
    function_args = coerce_tool_args(function_name, function_args)

    # ════════════════════════════════════════════════════════════════
    # Step 2: Tool Search bridge 调度
    # ════════════════════════════════════════════════════════════════
    # 背景:_compute_tool_definitions 在工具太多时,把周边工具"折叠"成
    #       3 个桥接工具(tool_search / tool_describe / tool_call)。
    # 调度策略:
    #   tool_search  → 直接读 catalog(纯查,inline 处理)
    #   tool_describe → 直接读 catalog(纯查,inline 处理)
    #   tool_call    → 拆包到底层工具,递归 dispatch(让 hook 看到真名)
    # ── Tool Search bridge dispatch ──────────────────────────────────
    # tool_search and tool_describe are pure catalog reads — handle them
    # inline. tool_call is unwrapped to the underlying tool so that every
    # downstream hook (pre/post, edit approval, guardrails) sees the real
    # tool name, not the bridge.
    _ts_mod = None
    try:
        from tools import tool_search as _ts_mod  # noqa: F401
    except Exception:
        _ts_mod = None

    if _ts_mod is not None and _ts_mod.is_bridge_tool(function_name):
        try:
            # Use skip_tool_search_assembly=True so we see the real catalog,
            # not the already-collapsed bridge-only list (the bridge would
            # otherwise be searching only itself).
            #
            # Scope the catalog to the session's toolsets so the bridge can
            # only surface and invoke tools the session was actually granted.
            # Without this, a restricted-toolset session (subagent, kanban
            # worker, curated gateway session) would see and be able to call
            # the entire process registry via the bridge. Passing the same
            # enabled/disabled toolsets the session was assembled with keeps
            # the deferred catalog identical to the deferrable subset of the
            # session's own tool list, and avoids polluting the process-global
            # _last_resolved_tool_names with out-of-scope tools.
            current_defs = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True,
            ) or []
        except Exception:
            current_defs = []
        if function_name == _ts_mod.TOOL_SEARCH_NAME:
            # 纯查:模型想找"web_*"开头的工具 → 返匹配列表
            return _ts_mod.dispatch_tool_search(function_args or {},
                                                current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_DESCRIBE_NAME:
            # 纯查:模型想看某个工具的完整 schema
            return _ts_mod.dispatch_tool_describe(function_args or {},
                                                  current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_CALL_NAME:
            # 拆包:把 tool_call 还原成"真工具名 + 真参数",递归 dispatch
            # 好处:所有 hook(pre/post、确认、guardrail)都看到真名
            underlying_name, underlying_args, err = _ts_mod.resolve_underlying_call(function_args or {})
            if err or not underlying_name:
                return json.dumps({"error": err or "tool_call could not be resolved"},
                                  ensure_ascii=False)
            # Defense in depth: the underlying tool MUST be in the session's
            # scoped deferrable catalog. resolve_underlying_call() only checks
            # that the name is deferrable in the global registry; this gate
            # additionally rejects any tool the session was not granted, so a
            # restricted session can never invoke an out-of-scope tool through
            # the bridge even if the catalog scoping above regressed.
            _scoped_deferrable = _ts_mod.scoped_deferrable_names(current_defs)
            if underlying_name not in _scoped_deferrable:
                return json.dumps({
                    "error": (
                        f"'{underlying_name}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    ),
                }, ensure_ascii=False)
            # Recurse with the underlying tool. All hooks fire against the
            # real tool name. The bridge is invisible to hooks by design.
            return handle_function_call(
                function_name=underlying_name,
                function_args=underlying_args,
                task_id=task_id,
                tool_call_id=tool_call_id,
                session_id=session_id,
                user_task=user_task,
                enabled_tools=enabled_tools,
                skip_pre_tool_call_hook=skip_pre_tool_call_hook,
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )

    try:
        # ════════════════════════════════════════════════════════════════
        # Step 3: Agent-loop 工具拦截
        # ════════════════════════════════════════════════════════════════
        # 一些工具(todo / memory / session_search / delegate_task)需要
        # agent 级别的状态(TodoStore / MemoryStore),registry 里只放了
        # schema,handler 是个空壳。run_agent.py 会在主循环里拦截它们自己
        # 处理。如果这里意外走到 → 返错误让模型重试或换工具。
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        # ════════════════════════════════════════════════════════════════
        # Step 4: pre-tool-call hook(确认/审计)
        # ════════════════════════════════════════════════════════════════
        # plugin 在执行前可以检查这次调用:
        #   - 安全检查(黑名单命令、敏感文件路径)
        #   - 用户确认(危险操作)
        #   - 审计日志
        # get_pre_tool_call_block_message() 返回 None 表示放行,
        # 返回 str 表示拦截,把 str 当错误消息返给模型。
        #
        # skip_pre_tool_call_hook=True 时跳过(避免重复触发,run_agent
        # 自己已经检查过)
        #
        # Single-fire contract: pre_tool_call fires exactly once per tool
        # execution. get_pre_tool_call_block_message() internally calls
        # invoke_hook("pre_tool_call", ...) and returns the first block
        # directive (if any), so observer plugins see the hook on that same
        # pass. When skip=True, the caller already fired it — do nothing
        # here.
        if not skip_pre_tool_call_hook:
            block_message: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name,
                    function_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                )
            except Exception as _hook_err:
                logger.debug("pre_tool_call hook error: %s", _hook_err)

            if block_message is not None:
                return json.dumps({"error": block_message}, ensure_ascii=False)

        # ════════════════════════════════════════════════════════════════
        # Step 5: ACP / Zed 编辑审批(write_file / patch 之前)
        # ════════════════════════════════════════════════════════════════
        # 在 ACP session 里(用 Zed IDE 接进来),写文件需要用户确认。
        # CLI / gateway 路径 ContextVar 没绑,直接跳过。
        # 如果审批 guard 自身炸了,且这次调的是写工具 → 兜底拒绝
        # (fail-closed:宁可多拦截,不要放行未知状态)
        # ACP/Zed edit approval runs before any file mutation.  The requester
        # is bound via ContextVar only for ACP sessions, so CLI/gateway paths
        # are unaffected when it is unset.
        try:
            from acp_adapter.edit_approval import maybe_require_edit_approval

            edit_block_message = maybe_require_edit_approval(function_name, function_args)
            if edit_block_message is not None:
                return edit_block_message
        except Exception as _edit_approval_err:
            logger.debug("ACP edit approval guard error: %s", _edit_approval_err)
            if function_name in {"write_file", "patch"}:
                return json.dumps({"error": "Edit approval denied: approval guard failed"}, ensure_ascii=False)

        # ════════════════════════════════════════════════════════════════
        # Step 6a: 读循环检测(non-read 工具重置连续 read 计数)
        # ════════════════════════════════════════════════════════════════
        # 模型陷入"连读 N 次文件"时,read_file_tool 内部会强制打断,
        # 提示模型换策略。notify_other_tool_call 就是告诉 tracker:
        # "这次不是 read,清零连续计数"。
        # Notify the read-loop tracker when a non-read/search tool runs,
        # so the *consecutive* counter resets (reads after other work are fine).
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # ════════════════════════════════════════════════════════════════
        # Step 6b: 实际派发(registry.dispatch)
        # ════════════════════════════════════════════════════════════════
        # 走到这里说明前面所有 guard 都放行。
        # 计时:用 monotonic()(墙钟跳变不影响),给 post-hook 一个 duration_ms。
        # 模仿 Claude Code 2.1.119 的 PostToolUse hook 输入。
        #
        # 特殊:execute_code 多传 enabled_tools(沙箱内能调的工具列表),
        #       防止 subagent 偷换 process-global _last_resolved_tool_names。
        # Measure tool dispatch latency so post_tool_call and
        # transform_tool_result hooks can observe per-tool duration.
        # Inspired by Claude Code 2.1.119, which added ``duration_ms`` to
        # PostToolUse hook inputs so plugin authors can build latency
        # dashboards, budget alerts, and regression canaries without having
        # to wrap every tool manually.  We use monotonic() so the value is
        # unaffected by wall-clock adjustments during the call.
        _dispatch_start = time.monotonic()
        if function_name == "execute_code":
            # Prefer the caller-provided list so subagents can't overwrite
            # the parent's tool set via the process-global.
            sandbox_enabled = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                enabled_tools=sandbox_enabled,
            )
        else:
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                user_task=user_task,
            )
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)

        # ════════════════════════════════════════════════════════════════
        # Step 7: post-tool-call hook(观察)+ transform_tool_result(改写)
        # ════════════════════════════════════════════════════════════════
        # post_tool_call  是观察型 hook,通知 plugin"工具跑完了"
        #   (plugin 自己用,不影响返回值)
        # transform_tool_result  是改写型 hook:
        #   - 多个 plugin 都跑,第一个返 str 的覆盖 result
        #   - 返 None / 非 str 忽略
        #   - fail-open:hook 炸了不传播
        # 顺序:先 post_tool_call(看),再 transform(改写)——保证 transform
        #       拿到的是真实结果
        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook(
                "post_tool_call",
                tool_name=function_name,
                args=function_args,
                result=result,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                duration_ms=duration_ms,
            )
        except Exception as _hook_err:
            logger.debug("post_tool_call hook error: %s", _hook_err)

        # Generic tool-result canonicalization seam: plugins receive the
        # final result string (JSON, usually) and may replace it by
        # returning a string from transform_tool_result. Runs after
        # post_tool_call (which stays observational) and before the result
        # is appended back into conversation context. Fail-open; the first
        # valid string return wins; non-string returns are ignored.
        try:
            from hermes_cli.plugins import invoke_hook
            hook_results = invoke_hook(
                "transform_tool_result",
                tool_name=function_name,
                args=function_args,
                result=result,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                duration_ms=duration_ms,
            )
            for hook_result in hook_results:
                if isinstance(hook_result, str):
                    result = hook_result
                    break
        except Exception as _hook_err:
            logger.debug("transform_tool_result hook error: %s", _hook_err)

        return result

    except Exception as e:
        # ════════════════════════════════════════════════════════════════
        # 异常兜底:任何一步炸了都不能污染主循环
        # ════════════════════════════════════════════════════════════════
        # logger.exception 记 traceback
        # 走 _sanitize_tool_error 把 framing token 洗掉
        # 返 JSON 错误给模型(它会决定重试 or 换工具)
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": _sanitize_tool_error(error_msg)}, ensure_ascii=False)


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================
#
# === 5 个 thin wrapper ===
# 这些函数存在的唯一原因:老代码(plugins、外部脚本、CLI 子命令)
# 直接 import `model_tools.xxx`,而不是 `model_tools.registry.xxx`。
# 现在保留这些 1-行 wrapper,既不破坏老 API,又统一了"对外门面"。

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
