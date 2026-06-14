"""Tool-call execution — sequential and concurrent dispatch.

# === 这个文件是干什么的? ===
# agent 主循环(run_agent.py)拿到 LLM 吐回来的 assistant_message(含 tool_calls 列表)
# 之后,**实际执行这些工具调用**就是由本文件完成的。
# 文件提供两种执行模式:
#   * execute_tool_calls_concurrent  → 多线程并发跑(默认;LLM 一次吐多个独立工具时)
#   * execute_tool_calls_sequential  → 单线程顺序跑(交互式工具、调试模式)
#
# === 跟 run_agent.py 的关系 ===
# 历史上这两个函数是 AIAgent 的方法(_execute_tool_calls_*)。
# 抽出来成 module-level 函数,主要是为了:
#   1. 让本文件可以独立 import、独立测试(不依赖 AIAgent 类)
#   2. 让文件长度可控(原 run_agent.py 太长,接近 4000 行)
# 但为了不破坏现有调用点和测试,run_agent.py 里保留同名 thin wrapper,
# 直接转调这里的 module 函数,看起来像方法。
#
# === monkey-patch 兼容 ===
# 关键设计:本文件**不直接 import run_agent**,而是用 _ra() 延迟获取
# run_agent 模块的引用。这样:
#   * 测试可以 patch run_agent._set_interrupt(老路径)
#   * 而本文件通过 _ra()._set_interrupt 拿到被 patch 后的版本
# 这是把"方法拆出去"的标准 Python 技巧——抽函数不抽 import 链。

# === 跟 model_tools.py 的关系 ===
# model_tools.py:负责"准备工具 schema"和"分发 1 个工具调用"
# tool_executor.py:负责"批量执行 + 并发控制 + 中断处理 + 结果回填 messages"
# 也就是说 model_tools 关心"单个工具",本文件关心"一批工具怎么一起跑"。

Both AIAgent methods (``_execute_tool_calls_sequential`` and
``_execute_tool_calls_concurrent``) live here as module-level
functions that take the parent ``AIAgent`` as their first argument.

``run_agent`` keeps thin wrappers so existing call sites work; tests
that patch ``run_agent._set_interrupt`` are honored because the
extracted functions reach back through the ``run_agent`` module via
``_ra()`` for that symbol.
"""

from __future__ import annotations

# === 标准库 import 分组(按用途) ===
import concurrent.futures   # 线程池:concurrent.futures.ThreadPoolExecutor,max_workers=_MAX_TOOL_WORKERS
                            # 核心数据结构 Future:可以 submit() 后 future.add_done_callback / .result()
import json                 # 解析 LLM 吐的工具参数(tool_call.function.arguments 是 JSON 字符串)
import logging              # 日志
import os                   # 环境变量(读 HERMES_KANBAN_TASK、HERMES_PARALLEL_MAX_WORKERS 等)
import random               # 用于"staggered start"——给 worker 一点随机 jitter,避免尖峰
                            # (细节在并发核心段会说)
import threading            # 线程锁、线程本地存储(threading.local);并发收结果时用锁
import time                 # monotonic() 计时,看每个工具跑了多久
from typing import Optional # Optional[str] = str | None,3.10 之前的兼容写法

# === 第三方 / 项目内 import(按子系统分组) ===
# Display 子系统(agent/display.py):UI 层的工具——kawaii 动画、emoji、cute message、检测工具失败
# 这些都是**纯展示用**,不影响执行逻辑,只是让输出好看
from agent.display import (
    KawaiiSpinner,                            # 可爱的 spinner 动画字符集
    build_tool_preview as _build_tool_preview,  # 把工具调用格式化成"🔍 read_file(/path/to/x)"这样的预览
    get_cute_tool_message as _get_cute_tool_message_impl,  # 跑完工具后给"✅ Done!~"这种 cute 反馈
    get_tool_emoji as _get_tool_emoji,        # 给工具名匹配 emoji(🐱 for read_file,🐶 for write_file,...)
    _detect_tool_failure,                     # 看 tool 输出内容判断"是失败了吗?"(启发式,不是抛异常)
)

# Guardrail 子系统:工具调用前的安全检查(类似 firewall)
# ToolGuardrailDecision 是 Enum:ALLOW / BLOCK_WITH_REASON / NEEDS_APPROVAL
from agent.tool_guardrails import ToolGuardrailDecision

# Dispatch helpers:分散的"执行小工具"——分两类
#   1. _is_destructive_command / _is_multimodal_tool_result / _multimodal_text_summary / _append_subdir_hint_to_multimodal
#      → 各种"判定/格式化"辅助,纯函数
#   2. make_tool_result_message
#      → 把工具结果包装成 API 要求的 {"role": "tool", "tool_call_id": ..., "content": ...} 格式
from agent.tool_dispatch_helpers import (
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    make_tool_result_message,
)

# 终端工具的环境变量传递(worker 线程需要拿到主线程的 env)
from tools.terminal_tool import (
    get_active_env,
)

# 线程上下文传递:把主线程的 contextvars 复制到 worker 线程
# (Python 默认 contextvars 不会自动跨线程传播,需要手动 propagate)
from tools.thread_context import propagate_context_to_thread

# 工具结果存储:大结果 → 落盘,小结果 → 留在 messages;还有 turn 预算检查
from tools.tool_result_storage import (
    maybe_persist_tool_result,    # 如果结果太大,写到 ~/.hermes/tool_results/<id>.json,只返摘要
    enforce_turn_budget,          # 检查本 turn 累计 token 没超上限
)

logger = logging.getLogger(__name__)


# === 并发上限 ===
# 8 是怎么来的?
# - 太少(如 2):LLM 一次吐 5 个 read_file 就要排队
# - 太多(如 64):每个 worker 都有自己的 httpx/AsyncOpenAI 客户端缓存,内存吃紧
# - 8 是经验值:够 LLM 一次吐的最大工具数用,又不至于爆内存
#
# 这个常量在 run_agent.py 也有一个同名版本(MAX_PARALLEL_TOOL_WORKERS),
# 这里 mirror 一下是为了让"只 import 本文件的测试"也能拿到。
_MAX_TOOL_WORKERS = 8


# === 工具 1:_ra() — 延迟拿到 run_agent 模块的引用 ===
# 这是把"AIAgent 方法拆成 module 函数"必须做的 hack。
#
# 背景:
#   老代码里 _execute_tool_calls_sequential 是 AIAgent 的方法,
#   它直接 self._set_interrupt(...) 就拿到"中断信号处理函数"。
#   现在拆成 module 函数,没有 self 了,需要去拿 run_agent 模块里
#   的 _set_interrupt。
#
# 关键:**不能**在文件顶部 `import run_agent`,否则:
#   * run_agent.py 内部会 `from agent.tool_executor import ...`
#   * 形成循环 import,Python 启动时就崩
#
# 所以用"延迟到函数被调用时才 import"的 lazy 模式,
# 这时候 run_agent 已经加载完了,import 是安全的。
#
# 还要保留 `_ra` 这个"间接层",是为了兼容 monkey-patch 测试:
#   ```python
#   # 测试代码
#   import run_agent
#   run_agent._set_interrupt = lambda: True
#   ```
#   如果本文件直接 import run_agent 并缓存 ._set_interrupt,
#   那 patch 在 import 之后就不生效了。
#   用 _ra() 每次重新拿模块,patch 总是拿到最新版。
def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    return run_agent


# === 工具 2:_tool_search_scoped_names() — Tool Search 桥接的安全闸 ===
#
# === 为什么需要这个函数? ===
# 回忆 model_tools.py 里的 Tool Search 设计:
#   1. LLM 调 tool_call 桥接(类似 "查一下,这个我要调 read_file")
#   2. model_tools.handle_function_call 在 Step 2 拆包,把 tool_call 还原成 read_file
#   3. 原本这一步会**再过一次 scope 检查**(看 read_file 在不在 session 授权范围)
#
# 关键问题:**本文件在执行 tool 之前也拆包** (L148-200 段),
# 拆完之后**直接 dispatch**,**绕过了** model_tools 那个 scope check。
# 也就是说如果 LLM 调 tool_call("read_file"),
# 正常路径下 model_tools 会检查"这个 read_file 你能调吗?",
# 但 tool_executor 在更早的地方就拆包了,model_tools 看到的是"已经拆好的 read_file",
# 不会再做 scope 检查。
#
# 解决方案:**在 tool_executor 拆包的地方自己做一次 scope 检查**。
# 这个检查需要知道"当前 session 能调哪些 deferrable 工具"——就是本函数的活。
#
# === 什么算"授权范围"? ===
# 不是 registry 里所有工具,而是"本 session 显式 enabled / disabled 后剩下的
# deferrable 工具"。"deferrable"指可以被 Tool Search 桥接(即可折叠)的工具,
# 核心工具(read_file 之类)不在折叠范围,所以也不受 tool_call 桥接影响。
#
# === 缓存策略 ===
# 这个函数**每个工具调用都会被查一次**(L582 那种),如果每次都重算会很贵。
# 所以缓存到 agent._tool_search_scope_cache,key 含:
#   * registry._generation  → MCP server 重连时自动失效
#   * frozenset(enabled)    → 改 toolset 配置时失效
#   * frozenset(disabled)   → 改 toolset 配置时失效
#
# === 容错 ===
# 所有 import 和计算都包了 try/except,任何一步失败都返 frozenset()
# (空集)——这意味着"安全失败":如果校验失败,就**什么都不允许调**。
# 总比"校验失败就放行"安全。
def _tool_search_scoped_names(agent) -> frozenset:
    """Return the deferrable tool names the session may invoke via tool_call.

    The Tool Search unwrap dispatches the underlying tool directly, bypassing
    the bridge branch (and its scope check) in
    ``model_tools.handle_function_call``. To keep a restricted-toolset session
    (subagent, kanban worker, curated gateway session) from reaching tools it
    was never granted, the unwrap validates the underlying name against this
    set: the deferrable subset of the session's own enabled/disabled toolset
    scope.

    Result is cached on the agent and refreshed when the tool registry's
    generation changes (e.g. an MCP server reconnects), so the common case is
    a dict lookup, not a full tool-defs rebuild on every tool call.
    """
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
        return frozenset()

    # === Step 2:从 agent 拿配置,算 cache key ===
    # 用 getattr 而不是直接 .属性 访问:agent 可能没有这些属性(老版本/测试桩)
    # None 表示"未设置"——这是 model_tools 的语义:None = 全部开
    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)

    # === cache key 设计(3 维) ===
    # 1. registry._generation → MCP server 重连时 +1,key 失效,重新算
    # 2. frozenset(enabled)  → 配置变就失效
    # 3. frozenset(disabled) → 配置变就失效
    # 没把 agent 自己放 key 里:一个 agent 内配置不会变(除了 generation)
    cache_key = (
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )

    # === Step 3:查缓存(无锁,无所谓) ===
    # 这里不锁是有意的:即便两个线程同时算,谁先 set 谁的值,
    # 反正两次算的结果应该是相同的(frozenset)。
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]   # ← 命中:直接返,0 成本

    # === Step 4:缓存未命中 → 重新算 ===
    # 算的过程:
    #   model_tools.get_tool_definitions(quiet=True) 拿 session 实际的工具 schema
    #   skip_tool_search_assembly=True 拿完整 catalog(否则会拿到折叠后的 bridge-only 列表)
    #   _ts.scoped_deferrable_names()  筛出"deferrable"那部分
    try:
        scoped_defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,                # 不 print 工具列表
            skip_tool_search_assembly=True, # 拿"未折叠"的完整 catalog
        ) or []
        names = _ts.scoped_deferrable_names(scoped_defs)
    except Exception:
        # 算的过程中任何一步炸了 → 返空集("安全失败")
        names = frozenset()

    # === Step 5:写缓存(也用 try 包) ===
    # setattr 可能失败(agent 是 Mock / frozen dataclass 之类),
    # 失败也不抛——下次再算就行。
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:
        pass
    return names


# === execute_tool_calls_concurrent — 核心并发执行器 ===
#
# === 这个函数做什么? ===
# 主循环(LLM 这一轮吐了 N 个 tool_calls)调本函数,
# 本函数:
#   1. 预处理(中断检查、参数解析、Tool Search 拆包、guardrail、checkpoint)
#   2. 起一个 ThreadPoolExecutor(max_workers=8) 并发跑
#   3. 按原顺序收集结果(LLM API 要求 tool result 按 tool_call_id 顺序回填)
#   4. 写回 messages,触发 post-call hook
#
# === 这个函数的 6 大步骤 ===
#   Step A 中断检查(用户按 Ctrl-C 的话啥也别做)
#   Step B 解析 + bookkeeping(nudge counter 重置 + JSON parse + 兜底空 dict)
#   Step C Tool Search 拆包 + 安全闸(本文件拆包会绕过 model_tools 的 scope check,所以这里必须自己看)
#   Step D 决定 block / allow(plugin hook → guardrail → 结果写进 parsed_calls)
#   Step E Checkpoint 预演(只对要跑的工具做 file snapshot)
#   Step F 并发跑 + 收集结果(L200+ 段)
#
# === 关键设计点 ===
# 1. **顺序保持**:results = [None] * num_tools,然后按 i 索引填,
#    即便 worker 完成的顺序乱,最终 messages 也是按 LLM 原始吐的顺序。
# 2. **提前 block**:Step D 在 submit 到线程池之前就决定要不要跑,
#    block 的工具不浪费线程资源。
# 3. **失败不传染**:一个 tool 抛异常不影响其他 tool 继续跑。
def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Execute multiple tool calls concurrently using a thread pool.

    Results are collected in the original tool-call order and appended to
    messages so the API sees them in the expected sequence.
    """
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)

    # ════════════════════════════════════════════════════════════════
    # Step A: 中断检查(用户按 Ctrl-C / SIGINT 的话啥也别做)
    # ════════════════════════════════════════════════════════════════
    # 顺序版(concurrent + sequential)都有这一步,目的:用户中断时
    # 不要再起任何 worker 线程(虽然线程本身不能被 SIGINT 直接取消,
    # 但**不发起**就不会浪费 work)。
    #
    # 注意:**这一步是开工前的检查**。如果用户在工具已经跑起来之后按 Ctrl-C,
    # 那个 worker 不会自动停——只能等它跑完。
    # 顺序版(L548)会**每个工具开工前**再查一次,提供更细粒度的取消。
    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        for tc in tool_calls:
            messages.append(make_tool_result_message(
                tc.function.name,
                f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                tc.id,
            ))
        return

    # ════════════════════════════════════════════════════════════════
    # Step B: 解析 + pre-execution bookkeeping
    # ════════════════════════════════════════════════════════════════
    # parsed_calls 里每个元素是 5-tuple:
    #   (tool_call, function_name, function_args, block_result, blocked_by_guardrail)
    # 这样后面 Step F 收结果时元组解包直接对得上。
    parsed_calls = []  # list of (tool_call, function_name, function_args, block_result, blocked_by_guardrail)
    for tool_call in tool_calls:
        function_name = tool_call.function.name

        # === B-1: 重置 nudge counter(劝学机制) ===
        # Hermes 里有"提示学习"的机制:如果 LLM 很久没调 memory / skill_manage,
        # 主循环会**自动注入 system message 提醒**。
        # 一旦 LLM 真的调了,counter 重置,提醒就停。
        # 这条规则:
        #   * LLM 调 memory → _turns_since_memory = 0
        #   * LLM 调 skill_manage → _iters_since_skill = 0
        # Reset nudge counters
        if function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        # === B-2: 解析 LLM 吐的 JSON 字符串参数 ===
        # 三个兜底:
        #   1. JSON 解析失败(LLM 偶尔吐坏 JSON)→ 空 dict
        #   2. 解析成功但不是 dict(LLM 偶尔吐 "5" 这种 string)→ 空 dict
        #   3. logger.warning 已经记下了,这里静默兜底
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            function_args = {}
        if not isinstance(function_args, dict):
            function_args = {}

        # ════════════════════════════════════════════════════════════
        # Step C: Tool Search 拆包 + 安全闸
        # ════════════════════════════════════════════════════════════
        # === 为什么这里要拆? ===
        # 正常路径下,LLM 调 tool_call("read_file", {"path": "x.txt"})
        # 会先到 model_tools.handle_function_call → 拆成 read_file → 检查 scope → dispatch
        #
        # 但本文件想"让 hook 看到真名",所以在**这里**就拆了。
        # 拆完之后直接 dispatch,绕过了 model_tools 的 scope check。
        # 所以**必须**在这里自己做一次 scope check(下面 _ts_scope_block 那一段)。
        #
        # === OpenClaw 教训 ===
        # 注释里说:"hooks must observe the real tool name"。
        # 老版本 Hermes 让 hook 看到 tool_call("read_file"),hook 不知道
        # 这是哪个真工具,做不了统计 / 安全策略 / 限流。
        # 新版本强制拆开,hook 看到的是真 read_file。
        #
        # === tool_call_id 必须保留 ===
        # 拆包后,function_name 变成 read_file,**但 tool_call.id 不变**——
        # API 用 tool_call_id 匹配 tool_call 和 tool_result。
        # 这个 id 写在 tool_call.function.name 之外的 tool_call.id 上(L371 append),
        # 拆包不动它。
        _ts_scope_block = None
        try:
            from tools import tool_search as _ts
            if function_name == _ts.TOOL_CALL_NAME:
                _underlying, _underlying_args, _err = _ts.resolve_underlying_call(function_args)
                if not _err and _underlying:
                    if _underlying in _tool_search_scoped_names(agent):
                        # === 拆包成功 + 在授权范围 → 换成真工具名 + 真参数 ===
                        function_name = _underlying
                        function_args = _underlying_args
                    else:
                        # === 拆包成功但**不在授权范围** → 标记 block,后面 Step D 短路 ===
                        # 不直接抛异常,而是构造一个"假成功 + 错误消息"的结果。
                        # 这样 LLM 收到错误消息后会调 tool_search 自己找能用的工具。
                        _ts_scope_block = json.dumps({
                            "error": (
                                f"'{_underlying}' is not available in this session. "
                                "Use tool_search to find tools you can call."
                            ),
                        }, ensure_ascii=False)
        except Exception:
            # 任何异常都吞掉,继续走——tool_call 拆包失败就当普通 tool_call 处理
            # (虽然不太可能成功)
            pass

        # ════════════════════════════════════════════════════════════
        # Step D: 决定 block / allow(顺序:tool_search scope → plugin hook → guardrail)
        # ════════════════════════════════════════════════════════════
        # 三道关卡,**任一返回 block 就短路**:
        #   1. _ts_scope_block(Step C 设的)→ 越权 tool_call
        #   2. plugin hook:get_pre_tool_call_block_message → 用户/插件主动禁止
        #   3. guardrail:agent._tool_guardrails.before_call → 系统安全规则
        #
        # === 为什么"在 checkpoint 之前"做这个判断? ===
        # 注释里写得很清楚:"We must know whether the tool will execute
        # before touching checkpoint state"。
        # 如果先做 checkpoint 再发现要 block,就白白创建了 file snapshot,
        # 浪费磁盘 + 让 undo 历史里多了个不需要的条目。
        block_result = None
        blocked_by_guardrail = False
        if _ts_scope_block is not None:
            # Out-of-scope tool_call: reject before hooks/guardrails/dispatch.
            block_result = _ts_scope_block
        else:
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                # plugin hook:外部扩展可以注册"禁止某工具 + 某参数"的规则
                # 比如企业版可以加"禁止 read_file 读 ~/.ssh/"之类的策略
                block_message = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                # hook 自身出错 → 当作"没意见",继续走 guardrail
                block_message = None

            if block_message is not None:
                block_result = json.dumps({"error": block_message}, ensure_ascii=False)
            else:
                # === guardrail:系统级安全检查(类似 firewall) ===
                # 决定可能是 ALLOW / BLOCK_WITH_REASON / NEEDS_APPROVAL
                # 后者需要用户交互式确认(下面会处理)
                guardrail_decision = agent._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    block_result = agent._guardrail_block_result(guardrail_decision)
                    blocked_by_guardrail = True

        # ════════════════════════════════════════════════════════════
        # Step E: Checkpoint 预演(只为"会跑"的工具做 file snapshot)
        # ════════════════════════════════════════════════════════════
        # Checkpoint 是 Hermes 的"安全网"——改文件前先 git 备份,
        # 用户后悔了可以一键 undo。
        # 只在 block_result is None(工具会真跑)时才做,否则白浪费磁盘。
        if block_result is None:
            # === E-1: 文件修改类工具的 checkpoint ===
            # write_file / patch 是显式改文件 → 必须 checkpoint
            if function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        # get_working_dir_for_path 决定这个 file 属于哪个 worktree
                        # (subagent / kanban worker 可能有自己的 worktree)
                        work_dir = agent._checkpoint_mgr.get_working_dir_for_path(file_path)
                        agent._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    # checkpoint 失败不能 block 工具跑——只是失去 undo 能力
                    pass

            # === E-2: 危险 terminal 命令的 checkpoint ===
            # "危险"指:rm -rf /、dd、chmod -R 777、git push --force 之类
            # _is_destructive_command 是个黑名单/启发式判断
            if function_name == "terminal" and agent._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        agent._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

        # === Step B/C/D/E 都做完,append 到 parsed_calls 等 Step F 用 ===
        # 注意 5-tuple 的最后两个元素:block_result 和 blocked_by_guardrail
        # block_result 可能是 None(没 block)或一个 JSON 字符串(block 消息)
        # blocked_by_guardrail 是个 bool——区分 block 的"原因"
        parsed_calls.append((tool_call, function_name, function_args, block_result, blocked_by_guardrail))

    # ════════════════════════════════════════════════════════════════
    # 准备阶段的最后一段:logging / callbacks
    # ════════════════════════════════════════════════════════════════
    # 这部分**不依赖** worker 线程,可以和实际执行并行,
    # 所以放在 ThreadPoolExecutor 之前。
    #
    # === 三类通知 ===
    # 1. 控制台 print(quiet_mode 跳过)
    # 2. tool_progress_callback(给 TUI / dashboard 用,事件名 "tool.started")
    # 3. tool_start_callback(给 plugin 用,带 tool_call.id)
    # ── Logging / callbacks ──────────────────────────────────────────
    tool_names_str = ", ".join(name for _, name, _, _, _ in parsed_calls)

    # === 1. 控制台 print:一总览 + 每工具一行 ===
    # 总览:⚡ Concurrent: 3 tool calls — read_file, web_search, terminal
    # 单项:📞 Tool 1: read_file(['path']) - {"path": "x.txt"}
    if not agent.quiet_mode:
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
        for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls, 1):
            args_str = json.dumps(args, ensure_ascii=False)
            if agent.verbose_logging:
                # 详细模式:打完整 args(可能很大)
                print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                print(agent._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
            else:
                # 简洁模式:只截前 N 个字符(agent.log_prefix_chars 配置)
                args_preview = args_str[:agent.log_prefix_chars] + "..." if len(args_str) > agent.log_prefix_chars else args_str
                print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

    # === 2. tool_progress_callback:"tool.started" 事件 ===
    # 这个 callback 是 TUI / dashboard / IDE 插件订阅的,
    # 用来画"工具开始跑"的高亮、动画、日志条目
    # 跳过 block 掉的(block 的工具根本没跑)
    for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
        if block_result is not None:
            continue
        if agent.tool_progress_callback:
            try:
                preview = _build_tool_preview(name, args)
                agent.tool_progress_callback("tool.started", name, preview, args)
            except Exception as cb_err:
                # callback 失败不能影响主流程,只 debug 记一下
                logging.debug(f"Tool progress callback error: {cb_err}")

    # === 3. tool_start_callback:带 tool_call.id 的开始事件 ===
    # plugin 用来关联"哪个 API tool_call 对应哪个工具跑"
    for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
        if block_result is not None:
            continue
        if agent.tool_start_callback:
            try:
                agent.tool_start_callback(tc.id, name, args)
            except Exception as cb_err:
                logging.debug(f"Tool start callback error: {cb_err}")

    # ════════════════════════════════════════════════════════════════
    # Step F1: 准备 results 数组(顺序保持的关键)
    # ════════════════════════════════════════════════════════════════
    # 关键设计:
    #   results = [None] * num_tools
    #   每个 worker 完成后写 results[index] = (...)
    # 这样:
    #   1. 即便 worker 完成顺序乱(因为 IO 时延不一样),
    #   2. results[i] 也对应 LLM 第 i 个 tool_call
    #   3. 下游 messages.append 按顺序遍历,API 看到的 tool_result 顺序就对
    #
    # 每个 slot 是 6-tuple:
    #   (function_name, function_args, function_result, duration, is_error, blocked)
    # ── Concurrent execution ─────────────────────────────────────────
    # Each slot holds (function_name, function_args, function_result, duration, error_flag, blocked_flag)
    results = [None] * num_tools

    # === 先填好"已 block"的 slot(它们不跑 worker,直接有结果) ===
    # blocked slot 的特殊值:blocked=True, duration=0.0
    # 这样下面 post-execution 阶段一看 blocked=True 就知道是 guardrail 拦的
    for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
        if block_result is not None:
            results[i] = (name, args, block_result, 0.0, True, True)

    # === activity 心跳:告诉 gateway "我正在忙,别 kill 我" ===
    # gateway 监控 inactivity(没动静就视为 hang 杀掉),
    # 这里 _touch_activity 把"executing 3 tools concurrently: read_file, ..."记成 last activity
    # Touch activity before launching workers so the gateway knows
    # we're executing tools (not stuck).
    agent._current_tool = tool_names_str
    agent._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

    # ════════════════════════════════════════════════════════════════
    # Step F2: worker 函数(每个工具实际跑这一段)
    # ════════════════════════════════════════════════════════════════
    # 这个 def 在 execute_tool_calls_concurrent 内部,是个 closure:
    #   * 捕获 agent, tool_call, function_name, function_args
    #   * 捕获外层的 results 列表(用 index 索引)
    #   * 捕获外层的 messages(可能要 push 中间结果)
    #
    # === 4 段执行流程 ===
    # 1. 注册 worker tid(让 AIAgent.interrupt() 能定向 SIGINT)
    # 2. 处理"注册竞态"(避免 interrupt 在注册前触发)
    # 3. 设置 thread-local activity callback(让 _wait_for_process 心跳)
    # 4. try/finally 包住 _invoke_tool 实际跑 + 错误兜底
    def _run_tool(index, tool_call, function_name, function_args):
        """Worker function executed in a thread."""
        # === 1. 注册 worker tid(给 interrupt 用) ===
        # AIAgent.interrupt() 收到 Ctrl-C 后:
        #   * 看 agent._tool_worker_threads(所有正在跑的 worker)
        #   * 给每个 tid 调 _set_interrupt(True, tid)
        # 工具内部调 is_interrupted() 时检查自己的 tid,
        # 发现被设了中断位就主动退出(比如 terminal 调 _wait_for_process)
        #
        # 必须第一件事就做,否则从注册到第一次检查之间有 race window
        # Register this worker tid so the agent can fan out an interrupt
        # to it — see AIAgent.interrupt().  Must happen first thing, and
        # must be paired with discard + clear in the finally block.
        _worker_tid = threading.current_thread().ident
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(_worker_tid)

        # === 2. 处理"注册竞态" ===
        # 场景:agent 已经在 agent._tool_worker_threads 为空时 fan out interrupt,
        # 然后 worker 才开始注册。worker 注册时 _interrupt_requested=True,
        # 但 set 还没收到它。
        # 这里补救:如果 interrupt 已请求,主动给自己设中断位。
        # Race: if the agent was interrupted between fan-out (which
        # snapshotted an empty/earlier set) and our registration, apply
        # the interrupt to our own tid now so is_interrupted() inside
        # the tool returns True on the next poll.
        if agent._interrupt_requested:
            try:
                _ra()._set_interrupt(True, _worker_tid)
            except Exception:
                pass

        # === 3. thread-local activity callback ===
        # _wait_for_process(terminal 工具的 subprocess wait)在长命令期间
        # 调 activity callback 推 heartbeat 给 gateway。
        # 这个 callback 是 thread-local 的——主线程设的 worker 看不到。
        # 所以每个 worker 要重新设。
        # Set the activity callback on THIS worker thread so
        # _wait_for_process (terminal commands) can fire heartbeats.
        # The callback is thread-local; the main thread's callback
        # is invisible to worker threads.
        try:
            from tools.environments.base import set_activity_callback
            set_activity_callback(agent._touch_activity)
        except Exception:
            pass

        # === 4. 实际跑工具 ===
        # Approval/sudo callbacks (thread-local) and the agent turn's
        # ContextVars are propagated by propagate_context_to_thread() at the
        # submit site below (GHSA-qg5c-hvr5-hjgr, #13617).
        start = time.time()
        try:
            try:
                # === 4a. 调 agent._invoke_tool(真工具调用入口) ===
                # pre_tool_block_checked=True 告诉 _invoke_tool:
                #   "Step D 已经检查过 block 了,你别再查一次"
                # (避免双重检查 + 可能的"绕过 block"漏洞)
                result = agent._invoke_tool(
                    function_name,
                    function_args,
                    effective_task_id,
                    tool_call.id,
                    messages=messages,
                    pre_tool_block_checked=True,
                )
            except Exception as tool_error:
                # === 4b. 工具内部异常 → 不传染,自己吞掉,结果标 error ===
                # 关键:**不能 throw 出去**,否则 ThreadPoolExecutor 看不到结果
                # 而且一个工具的失败不该让其他工具的 messages 也写不进去
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start

            # === 4c. 启发式判断"工具返回的是不是失败" ===
            # 很多工具不抛异常但返回内容里有 "Error" / "Traceback" / "failed"
            # _detect_tool_failure 用关键字 + 模式匹配判断
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))

            # === 4d. 写 results[index](顺序保持的关键) ===
            # 即便多个 worker 并发完成,这里用 index 写,不会冲突
            results[index] = (function_name, function_args, result, duration, is_error, False)
        finally:
            # === 清理:tid 反注册 + 中断位清零 ===
            # Tear down worker-tid tracking.  Clear any interrupt bit we may
            # have set so the next task scheduled onto this recycled tid
            # starts with a clean slate.  This MUST be in a finally block
            # because BaseException subclasses (CancelledError, KeyboardInterrupt)
            # bypass ``except Exception`` and would otherwise leak the tid
            # into _interrupted_threads, poisoning the recycled thread.
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(_worker_tid)
            try:
                _ra()._set_interrupt(False, _worker_tid)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════════
    # Step F3: 起 spinner + 跑 ThreadPoolExecutor
    # ════════════════════════════════════════════════════════════════
    # === spinner ===
    # CLI 模式才显示,TUI 模式跳过(TUI 自己有进度条)
    # 用 KawaiiSpinner 画个"⚡ running N tools concurrently"
    # Start spinner for CLI mode (skip when TUI handles tool progress)
    spinner = None
    if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
        face = random.choice(KawaiiSpinner.get_waiting_faces())  # ← random 用在这(随机挑一个 cute 表情)
        spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=agent._print_fn)
        spinner.start()

    try:
        # === 筛出"真要跑"的工具(剔除 block 掉的) ===
        # runnable_calls 是 (i, tc, name, args) 四元组列表
        runnable_calls = [
            (i, tc, name, args)
            for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls)
            if block_result is None
        ]
        futures = []
        if runnable_calls:
            # === max_workers 自适应 ===
            # 太少工具就少开线程:5 个工具开 8 线程浪费,2 个工具开 8 线程更浪费
            max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # === submit 每个工具 ===
                for i, tc, name, args in runnable_calls:
                    # propagate_context_to_thread 的作用:
                    #   * 把主线程的 ContextVars(如 _approval_session_key)复制到 worker
                    #   * 设置 thread-local approval/sudo callbacks
                    #   * worker 退出时自动清理
                    # 不传这个会出 GHSA-qg5c-hvr5-hjgr(approval key 丢失,需要 sudo 的工具炸了)
                    # Propagate the agent turn's ContextVars (e.g.
                    # _approval_session_key) AND thread-local approval/sudo
                    # callbacks into the worker thread; clears callbacks on exit.
                    f = executor.submit(
                        propagate_context_to_thread(_run_tool), i, tc, name, args
                    )
                    futures.append(f)

                # ════════════════════════════════════════════════════
                # 等待所有 worker 完成(带心跳 + 中断检测)
                # ════════════════════════════════════════════════════
                # 普通 concurrent.futures.wait 阻塞等所有完成,
                # 但这里**每 5 秒 poll 一次**,原因:
                #   1. 让 gateway 看到"我还活着"的心跳(否则 kill)
                #   2. 让用户按 Ctrl-C 时能及时响应 cancel
                # Wait for all to complete with periodic heartbeats so the
                # gateway's inactivity monitor doesn't kill us during long
                # concurrent tool batches. Also check for user interrupts
                # so we don't block indefinitely when the user sends /stop
                # or a new message during concurrent tool execution.
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    done, not_done = concurrent.futures.wait(
                        futures, timeout=5.0,    # ← 每 5s 醒一次
                    )
                    if not not_done:
                        break   # ← 全部完成,退出循环

                    # === 中断检测 ===
                    # 已经在跑的 tool 内部的 per-thread interrupt signal 会让它退出
                    # (terminal / execute_code 这种支持中断的工具)
                    # 但 read_file / web_search 这种没中断检查的工具会跑到底
                    # 解决:cancel 还没开始的 future(让它们不跑)
                    # Check for interrupt — the per-thread interrupt signal
                    # already causes individual tools (terminal, execute_code)
                    # to abort, but tools without interrupt checks (web_search,
                    # read_file) will run to completion. Cancel any futures
                    # that haven't started yet so we don't block on them.
                    if agent._interrupt_requested:
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            agent._vprint(
                                f"{agent.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # 给已经跑起来的工具 3s 让它们响应中断
                        # (不是 kill -9 那种硬杀,是 graceful exit)
                        # Give already-running tools a moment to notice the
                        # per-thread interrupt signal and exit gracefully.
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    # === 心跳:每 30s 推一次 activity 给 gateway ===
                    # 用 _conc_elapsed % 30 < 6 是因为 5s 轮询周期,
                    # 30s / 5s = 6,这样"每 6 次轮询里有 1 次"满足 < 6
                    _conc_elapsed = int(time.time() - _conc_start)
                    # Heartbeat every ~30s (6 × 5s poll intervals)
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[futures.index(f)][1]
                            for f in not_done
                            if f in futures
                        ]
                        agent._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
    finally:
        # === spinner 收尾 ===
        # Build a summary message for the spinner stop
        if spinner:
            completed = sum(1 for r in results if r is not None)
            total_dur = sum(r[3] for r in results if r is not None)
            spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

    # ════════════════════════════════════════════════════════════════
    # Step G: Post-execution — 按顺序处理每个工具的结果
    # ════════════════════════════════════════════════════════════════
    # 这一段遍历 parsed_calls,**按 i 索引读 results[i]**,
    # 保证 messages 里的 tool_result 顺序 = LLM 原始 tool_call 顺序。
    #
    # 对每个工具依次做:
    #   1. 读 results[i](可能 None = 被 interrupt 取消)
    #   2. 加 guardrail observation(给事后审计看)
    #   3. 记 file-mutation(给 turn-end verifier)
    #   4. 发 tool_progress_callback "tool.completed"
    #   5. 打 ✅ cute message / 详细 log
    #   6. 大结果 → 落盘(maybe_persist_tool_result)
    #   7. 加 subdir hint(如果有)
    #   8. 适配当前模型的 content format
    #   9. append make_tool_result_message 到 messages
    #  10. drain /steer(用户中途新消息)
    # ── Post-execution: display per-tool results ─────────────────────
    for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
        r = results[i]
        blocked = False
        if r is None:
            # === 异常路径:results 没填(被 cancel 了) ===
            # 两种可能:
            #   1. 中断:用户按 Ctrl-C,worker 被 cancel
            #   2. 神秘丢失:worker 应该写 results[i] 但没写
            # 第 1 种最常见,第 2 种几乎不可能
            # Tool was cancelled (interrupt) or thread didn't return
            if agent._interrupt_requested:
                function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
            else:
                function_result = f"Error executing tool '{name}': thread did not return a result"
            tool_duration = 0.0
        else:
            # === 正常路径:从 6-tuple 解包 ===
            function_name, function_args, function_result, tool_duration, is_error, blocked = r

            if not blocked:
                # === 1. guardrail observation ===
                # guardrail 不仅能"前置 block",还能"事后添加观察信息"
                # 比如"PII 检测:输出含 3 个 SSN 号码,已脱敏"
                # _append_guardrail_observation 把这些 observation 拼到 result 末尾
                function_result = agent._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=is_error,
                )

            if is_error:
                # === 2. 错误 log(multimodal 结果要转纯文本摘要) ===
                _err_text = _multimodal_text_summary(function_result)
                result_preview = _err_text[:200] if len(_err_text) > 200 else _err_text
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

            # === 3. file-mutation 追踪 ===
            # turn 结束后会有 verifier 检查"LLM 说改了文件,实际改没改"
            # 这里记一下"成功 / 失败 / 时长"供 verifier 用
            # blocked 的工具没真跑,**不能**算 mutation
            # Track file-mutation outcome for the turn-end verifier.
            # `blocked` calls never actually ran — don't let a guardrail
            # block count as either a failure or a success.
            if not blocked:
                try:
                    agent._record_file_mutation_result(
                        function_name, function_args, function_result, is_error,
                    )
                except Exception as _ver_err:
                    logging.debug("file-mutation verifier record failed: %s", _ver_err)

            # === 4. tool_progress_callback "tool.completed" 事件 ===
            # 通知 TUI / dashboard 这个工具跑完了
            if not blocked and agent.tool_progress_callback:
                try:
                    agent.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=is_error,
                        result=function_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            # === 5. 详细 log(只有 verbose 模式) ===
            if agent.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

        # === 6. cute message / 详细 log(只针对"跑完"的) ===
        # Print cute message per tool
        if agent._should_emit_quiet_tool_messages():
            # cute 模式:打 "🐱 All done! 用了 1.2 秒~" 这种
            cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
            agent._safe_print(f"  {cute_msg}")
        elif not agent.quiet_mode:
            # 普通模式:打 "✅ Tool 1 completed in 1.23s - 摘要"
            _preview_str = _multimodal_text_summary(function_result)
            if agent.verbose_logging:
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", _preview_str))
            else:
                response_preview = _preview_str[:agent.log_prefix_chars] + "..." if len(_preview_str) > agent.log_prefix_chars else _preview_str
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

        # === 7. activity heartbeat ===
        # 让 gateway 知道"我又完成了一步,别杀我"
        agent._current_tool = None
        agent._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

        # === 8. tool_complete_callback(给 plugin 用) ===
        if not blocked and agent.tool_complete_callback:
            try:
                agent.tool_complete_callback(tc.id, name, args, function_result)
            except Exception as cb_err:
                logging.debug(f"Tool complete callback error: {cb_err}")

        # === 9. 大结果 → 落盘 ===
        # 比如 read_file 拿到 1GB 日志,不能直接喂给 LLM
        # maybe_persist_tool_result 会写到 ~/.hermes/tool_results/<id>.json,
        # 返回的 content 替换成"完整路径 + 摘要",LLM 自己读
        # multimodal(图片)结果不落盘——base64 图片需要直接送
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=name,
            tool_use_id=tc.id,
            env=get_active_env(effective_task_id),
        ) if not _is_multimodal_tool_result(function_result) else function_result

        # === 10. subdirectory hint(辅助 LLM 找文件) ===
        # 比如 LLM 在 /home/u/proj/ 下工作,但 read_file 了 /home/u/other/x.py
        # 这里追加"~提醒:你也可以在 proj/ 下找"的提示
        subdir_hints = agent._subdirectory_hints.check_tool_call(name, args)
        if subdir_hints:
            if _is_multimodal_tool_result(function_result):
                # multimodal:把 hint 加到 text summary 部分,不动图片 block
                # Append the hint to the text summary part so the model
                # still sees it; don't touch the image blocks.
                _append_subdir_hint_to_multimodal(function_result, subdir_hints)
            else:
                function_result += subdir_hints

        # === 11. 转成当前模型能吃的内容格式 ===
        # Unwrap _multimodal dicts to an OpenAI-style content list so any
        # vision-capable provider receives [{type:text},{type:image_url}]
        # rather than a raw Python dict.  The Anthropic adapter already
        # accepts content lists; vision-capable OpenAI-compatible servers
        # (mlx-vlm, GPT-4o, …) accept image_url in tool messages natively.
        # Text-only servers get a string-safe fallback here so a rejected
        # image tool result never poisons canonical session history.
        # String results pass through unchanged.
        _tool_content = agent._tool_result_content_for_active_model(name, function_result)
        messages.append(make_tool_result_message(name, _tool_content, tc.id))

        # === 12. drain 用户的 /steer 新消息 ===
        # /steer 是"用户中途插入的新指令",在工具完成后立刻喂给 LLM
        # 不用等所有工具跑完(那样延迟太高)
        # ── Per-tool /steer drain ───────────────────────────────────
        # Same as the sequential path: drain between each collected
        # result so the steer lands as early as possible.
        agent._apply_pending_steer_to_tool_results(messages, 1)

    # ════════════════════════════════════════════════════════════════
    # Step H: 整 turn 预算检查 + /steer 最终注入
    # ════════════════════════════════════════════════════════════════
    # ── Per-turn aggregate budget enforcement ─────────────────────────
    # 检查"这一 turn 所有 tool_result 加起来"的 token 数
    # 超过 budget 的话 → 截断 / 摘要 / 报错
    # num_tools 可能被改(虽然这段没改),先 reassign 一遍防止 stale
    num_tools = len(parsed_calls)
    if num_tools > 0:
        turn_tool_msgs = messages[-num_tools:]   # 取刚 append 的 num_tools 条
        enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

    # ── /steer injection ──────────────────────────────────────────────
    # Append any pending user steer text to the last tool result so the
    # agent sees it on its next iteration. Runs AFTER budget enforcement
    # so the steer marker is never truncated. See steer() for details.
    #
    # 顺序关键:budget 之后再注入 steer
    # 否则 steer marker 可能被 budget 截断,LLM 看不到用户的最新指令
    if num_tools > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools)



# === execute_tool_calls_sequential — 顺序执行器 ===
#
# === 这个函数做什么? ===
# 和 concurrent 版**几乎一样的逻辑**,但**不并发**——
# 工具一个个跑(1 → 2 → 3 ...),跑完一个再跑下一个。
#
# === 什么时候用顺序? ===
# 主循环(LLM 这一轮)决定"用 concurrent 还是 sequential"的依据:
#   * 工具数 == 1                              → 顺序(没必要起线程池)
#   * LLM 用 tool_search 折叠的 bridge 工具      → 顺序(避免 BM25 抢资源)
#   * 交互式工具(terminal、ask_user、clarify)   → 顺序(用户要看到中间输出)
#   * tool_delay > 0(测试用,模拟人类节奏)        → 顺序(否则 sleep 没意义)
#   * 其它                                      → concurrent
#
# === 跟 concurrent 版的核心差异 ===
# 1. **每步 interrupt 检查**:concurrent 版只在开头查一次,
#    顺序版**每个工具开工前**查一次,粒度更细
# 2. **dispatch 路由表**:顺序版有个 200+ 行的 if/elif 链,
#    concurrent 版统一调 agent._invoke_tool(由 agent 内部路由)
# 3. **更精细的 spinner 控制**:每个分支独立控制 cute message
#
# === 9 大步骤(对每个 tool_call 重复) ===
#   1. interrupt 检查(每步)
#   2. 解析 args
#   3. Tool Search 拆包
#   4. block 评估(plugin + guardrail)
#   5. nudge counter 重置
#   6. log + activity + callbacks
#   7. Checkpoint(只对"会跑"的)
#   8. dispatch(大 if/elif 链)
#   9. 后处理:guardrail obs / file mutation / 落盘 / messages.append / steer
def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        # ════════════════════════════════════════════════════════════
        # Step 1: 每步 interrupt 检查(比 concurrent 版更细)
        # ════════════════════════════════════════════════════════════
        # 为什么"每步查"?
        # concurrent 版所有工具**同时跑**,用户在工具 1 跑的时候按 Ctrl-C,
        # 工具 1 在它的"内部轮询点"自然发现中断(比如 terminal 调 _wait_for_process),
        # 工具 2/3/4 还没跑就被 cancel 掉。
        #
        # 顺序版前一个工具可能跑 10 秒,这期间用户随时能按 Ctrl-C。
        # 每个工具开工前再查一次,可以避免**完全不需要跑的工具有个开头**。
        # SAFETY: check interrupt BEFORE starting each tool.
        # If the user sent "stop" during a previous tool's execution,
        # do NOT start any more tools -- skip them all immediately.
        if agent._interrupt_requested:
            remaining_calls = assistant_message.tool_calls[i-1:]
            if remaining_calls:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
            for skipped_tc in remaining_calls:
                skipped_name = skipped_tc.function.name
                # 注意:即便被 skip,也要 append 一个 tool_result_message
                # 因为 API 要求每个 tool_call 都有对应 tool_result
                # (否则下一轮 API 调用会被 400 拒)
                skip_msg = {
                    "role": "tool",
                    "name": skipped_name,
                    "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                    "tool_call_id": skipped_tc.id,
                }
                messages.append(skip_msg)
            break   # ← 跳出整个 for 循环,后面的工具全部不跑

        function_name = tool_call.function.name

        # ════════════════════════════════════════════════════════════
        # Step 2: 解析 args(和 concurrent 版一样的兜底)
        # ════════════════════════════════════════════════════════════
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            # 这里比 concurrent 版多了 logger.warning(因为 sequential 一次跑一个,
            # 可以有更细的诊断信息)
            logger.warning(f"Unexpected JSON error after validation: {e}")
            function_args = {}
        if not isinstance(function_args, dict):
            function_args = {}

        # ════════════════════════════════════════════════════════════
        # Step 3: Tool Search 拆包(和 concurrent 版逻辑完全一样)
        # ════════════════════════════════════════════════════════════
        # Tool Search unwrap — see execute_tool_calls_concurrent for full
        # rationale, including the scope gate (the unwrap dispatches the
        # underlying tool directly, so session toolset scope is enforced here).
        _ts_scope_block: Optional[str] = None
        try:
            from tools import tool_search as _ts
            if function_name == _ts.TOOL_CALL_NAME:
                _underlying, _underlying_args, _err = _ts.resolve_underlying_call(function_args)
                if not _err and _underlying:
                    if _underlying in _tool_search_scoped_names(agent):
                        # 拆包成功 + 在授权范围
                        function_name = _underlying
                        function_args = _underlying_args
                    else:
                        # 拆包成功但越权 → 标记 block
                        _ts_scope_block = (
                            f"'{_underlying}' is not available in this session. "
                            "Use tool_search to find tools you can call."
                        )
        except Exception:
            pass

        # ════════════════════════════════════════════════════════════
        # Step 4: block 评估(plugin hook → guardrail)
        # ════════════════════════════════════════════════════════════
        # Check plugin hooks for a block directive before executing.
        _block_msg: Optional[str] = None
        if _ts_scope_block is not None:
            _block_msg = _ts_scope_block
        else:
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass

        # guardrail 单独留一个变量,因为后面 dispatch 时要区分
        # "plugin block" vs "guardrail block"(给的提示消息不同)
        _guardrail_block_decision: ToolGuardrailDecision | None = None
        if _block_msg is None:
            guardrail_decision = agent._tool_guardrails.before_call(function_name, function_args)
            if not guardrail_decision.allows_execution:
                _guardrail_block_decision = guardrail_decision

        # === 合并 block 状态 ===
        # 后面所有 if not _execution_blocked 都是"真要跑"的检查
        _execution_blocked = _block_msg is not None or _guardrail_block_decision is not None

        # ════════════════════════════════════════════════════════════
        # Step 5: nudge counter 重置(只有"真要跑"才重置)
        # ════════════════════════════════════════════════════════════
        # 这点和 concurrent 版**不一样**:
        #   concurrent 版在 Step B(解析 args 时)就重置,无论后面 block 不 block
        #   sequential 版要等确认"不会被 block"才重置
        # 为什么?因为"被 block 的工具"不算"LLM 真的用 memory"——
        # LLM 只是**试图**调 memory 但被拦了,nudge 应该继续。
        if _execution_blocked:
            # Tool blocked by plugin or guardrail policy — skip counters,
            # callbacks, checkpointing, activity mutation, and real execution.
            pass
        # Reset nudge counters when the relevant tool is actually used
        elif function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        # ════════════════════════════════════════════════════════════
        # Step 6: log + activity + callbacks
        # ════════════════════════════════════════════════════════════
        if not agent.quiet_mode:
            args_str = json.dumps(function_args, ensure_ascii=False)
            if agent.verbose_logging:
                print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                print(agent._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
            else:
                args_preview = args_str[:agent.log_prefix_chars] + "..." if len(args_str) > agent.log_prefix_chars else args_str
                print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

        # === activity 心跳(主线程,所以不用 thread-local) ===
        if not _execution_blocked:
            agent._current_tool = function_name
            agent._touch_activity(f"executing tool: {function_name}")

        # === 顺序版不用 thread-local callback(主线程就是 worker) ===
        # 但还是要 set_activity_callback,因为 _wait_for_process 等长命令
        # 会通过这个 callback 推 heartbeat
        # Set activity callback for long-running tool execution (terminal
        # commands, etc.) so the gateway's inactivity monitor doesn't kill
        # the agent while a command is running.
        if not _execution_blocked:
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(agent._touch_activity)
            except Exception:
                pass

        # === tool_progress_callback / tool_start_callback(同 concurrent 版) ===
        if not _execution_blocked and agent.tool_progress_callback:
            try:
                preview = _build_tool_preview(function_name, function_args)
                agent.tool_progress_callback("tool.started", function_name, preview, function_args)
            except Exception as cb_err:
                logging.debug(f"Tool progress callback error: {cb_err}")

        if not _execution_blocked and agent.tool_start_callback:
            try:
                agent.tool_start_callback(tool_call.id, function_name, function_args)
            except Exception as cb_err:
                logging.debug(f"Tool start callback error: {cb_err}")

        # ════════════════════════════════════════════════════════════
        # Step 7: Checkpoint(同 concurrent 版)
        # ════════════════════════════════════════════════════════════
        # Checkpoint: snapshot working dir before file-mutating tools
        if not _execution_blocked and function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
            try:
                file_path = function_args.get("path", "")
                if file_path:
                    work_dir = agent._checkpoint_mgr.get_working_dir_for_path(file_path)
                    agent._checkpoint_mgr.ensure_checkpoint(
                        work_dir, f"before {function_name}"
                    )
            except Exception:
                pass  # never block tool execution

        # Checkpoint before destructive terminal commands
        if not _execution_blocked and function_name == "terminal" and agent._checkpoint_mgr.enabled:
            try:
                cmd = function_args.get("command", "")
                if _is_destructive_command(cmd):
                    cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                    agent._checkpoint_mgr.ensure_checkpoint(
                        cwd, f"before terminal: {cmd[:60]}"
                    )
            except Exception:
                pass  # never block tool execution

        # === 记录 dispatch 开始时间(给下面 elapsed 用) ===
        tool_start_time = time.time()

        # ════════════════════════════════════════════════════════════
        # Step 8: dispatch(大 if/elif 链) ⭐ 顺序版独有
        # ════════════════════════════════════════════════════════════
        # 这条 if/elif 链是顺序版**和并发版最大的区别**:
        # 并发版统一调 agent._invoke_tool(由 agent 内部统一路由),
        # 顺序版在这里手写路由。
        #
        # 为什么手写?因为顺序版要对**特定工具**做特殊处理:
        #   * todo: 单独走 todo_tool(直接读 agent._todo_store)
        #   * session_search: 走 session_db(不是普通 registry 工具)
        #   * memory: 走 memory_tool + memory_manager.on_memory_write 桥接
        #   * clarify: 调 agent.clarify_callback(可能阻塞等用户回复)
        #   * delegate_task: 调 agent._dispatch_delegate_task(可能起子 agent)
        #   * context engine 工具: 调 agent.context_compressor
        #   * memory provider 工具: 调 agent._memory_manager
        #   * 其它: 调 model_tools.handle_function_call(走统一 dispatcher)
        #
        # 每个分支都独立处理:
        #   1. 自己的 spinner(每个工具不同的 emoji / label)
        #   2. 自己的 cute message
        #   3. 自己的异常兜底
        #   4. 自己的 timing
        if _block_msg is not None:
            # Tool blocked by plugin policy — return error without executing.
            function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
            tool_duration = 0.0
        elif _guardrail_block_decision is not None:
            # Tool blocked by tool-loop guardrail — synthesize exactly one
            # tool result for the original tool_call_id without executing.
            function_result = agent._guardrail_block_result(_guardrail_block_decision)
            tool_duration = 0.0
        elif function_name == "todo":
            # === 分支 1:todo(任务列表管理) ===
            # 不走 registry,因为 todo 直接读 agent._todo_store(per-agent 内存状态)
            from tools.todo_tool import todo_tool as _todo_tool
            function_result = _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=agent._todo_store,
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
        elif function_name == "session_search":
            # === 分支 2:session_search(查历史 session) ===
            # 不走 registry,直接读 session_db(可能在 subagent 里没有)
            session_db = agent._get_session_db_for_recall()
            if not session_db:
                from hermes_state import format_session_db_unavailable
                function_result = json.dumps({"success": False, "error": format_session_db_unavailable()})
            else:
                from tools.session_search_tool import session_search as _session_search
                function_result = _session_search(
                    query=function_args.get("query", ""),
                    role_filter=function_args.get("role_filter"),
                    limit=function_args.get("limit", 3),
                    session_id=function_args.get("session_id"),
                    around_message_id=function_args.get("around_message_id"),
                    window=function_args.get("window", 5),
                    sort=function_args.get("sort"),
                    db=session_db,
                    current_session_id=agent.session_id,
                )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
        elif function_name == "memory":
            # === 分支 3:memory(写入 agent 长期记忆) ===
            # 内置 memory 工具 + **额外的** memory_manager 桥接
            # 桥接的作用:如果有外部 memory provider(hindsight、honcho 等),
            # 同一个写入动作要同步到外部,避免双份数据不一致
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            function_result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=agent._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if agent._memory_manager and function_args.get("action") in {"add", "replace"}:
                try:
                    agent._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                        metadata=agent._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=getattr(tool_call, "id", None),
                        ),
                    )
                except Exception:
                    pass
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
        elif function_name == "clarify":
            # === 分支 4:clarify(向用户提问,可能阻塞) ===
            # callback 是 agent.clarify_callback,通常是 CLI/TUI 的"输入框"
            # 这个工具可能**阻塞等用户输入**——所以必须顺序跑
            from tools.clarify_tool import clarify_tool as _clarify_tool
            function_result = _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=agent.clarify_callback,
            )
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
        elif function_name == "delegate_task":
            # === 分支 5:delegate_task(派生子 agent) ===
            # 走 agent._dispatch_delegate_task,可能起 subagent(多 agent 协作)
            # spinner 标记"🔀 delegating N tasks · (/agents to monitor)"
            # 告诉用户"正在派活,你可以用 /agents 监控"
            tasks_arg = function_args.get("tasks")
            if tasks_arg and isinstance(tasks_arg, list):
                spinner_label = f"🔀 delegating {len(tasks_arg)} tasks · (/agents to monitor)"
            else:
                goal_preview = (function_args.get("goal") or "")[:30]
                spinner_label = (
                    f"🔀 {goal_preview} · (/agents to monitor)"
                    if goal_preview
                    else "🔀 delegating · (/agents to monitor)"
                )
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            agent._delegate_spinner = spinner
            _delegate_result = None
            try:
                function_result = agent._dispatch_delegate_task(function_args)
                _delegate_result = function_result
            finally:
                # 不管成功失败都要清掉 spinner
                agent._delegate_spinner = None
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._context_engine_tool_names and function_name in agent._context_engine_tool_names:
            # === 分支 6:context engine 工具(lcm_grep / lcm_describe / lcm_expand) ===
            # 走 context_compressor(我们在 Day 3 详细读过)
            # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
            spinner = None
            if agent._should_emit_quiet_tool_messages():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _ce_result = None
            try:
                function_result = agent.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                _ce_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
            # === 分支 7:外部 memory provider 工具(hindsight_retain / honcho_search / ...) ===
            # 这些工具**不在 registry 里**,走 MemoryManager 统一路由
            # Memory provider tools (hindsight_retain, honcho_search, etc.)
            # These are not in the tool registry — route through MemoryManager.
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _mem_result = None
            try:
                function_result = agent._memory_manager.handle_tool_call(function_name, function_args)
                _mem_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent.quiet_mode:
            # === 分支 8:quiet 模式(显式 spinner,no print 噪声) ===
            # 和 default 区别只在 spinner 控制
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                preview = _build_tool_preview(function_name, function_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _spinner_result = None
            try:
                function_result = _ra().handle_function_call(
                    function_name, function_args, effective_task_id,
                    tool_call_id=tool_call.id,
                    session_id=agent.session_id or "",
                    enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                    skip_pre_tool_call_hook=True,        # ← 顺序版跳过 pre-hook(concurrent 已设)
                    enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                    disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                )
                _spinner_result = function_result
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        else:
            # === 分支 9:default 路径(走 model_tools.handle_function_call) ===
            # 这是**最常见**的路径——80% 的工具走这里
            # 所有特殊处理都走完了,剩下的都是"普通 registry 工具"
            try:
                function_result = _ra().handle_function_call(
                    function_name, function_args, effective_task_id,
                    tool_call_id=tool_call.id,
                    session_id=agent.session_id or "",
                    enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                    skip_pre_tool_call_hook=True,
                    enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                    disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                )
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
            tool_duration = time.time() - tool_start_time

        # ════════════════════════════════════════════════════════════
        # Step 9: 后处理(和 concurrent 版几乎一样)
        # ════════════════════════════════════════════════════════════
        # 9 步:
        #   a. 决定 preview(字符串截断 / multimodal 留 dict)
        #   b. 错误检测 + guardrail observation
        #   c. 错误/成功 log
        #   d. file-mutation 追踪
        #   e. tool_progress_callback "tool.completed"
        #   f. activity heartbeat + 清 _current_tool
        #   g. verbose log
        #   h. tool_complete_callback
        #   i. 大结果落盘 + subdir hint + content 适配
        #   j. messages.append + /steer drain
        #   k. 整 turn 的 print + interrupt re-check + tool_delay

        # === a. preview 准备 ===
        # 字符串结果:截前 200 字符(避免 log 爆炸)
        # multimodal(dict)结果:不能 slice,留 dict 自己处理
        if isinstance(function_result, str):
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
            _result_len = len(function_result)
        else:
            # Multimodal dict result (_multimodal=True) — not sliceable as string
            result_preview = function_result
            _result_len = len(str(function_result))

        # === b. 错误检测 + guardrail observation ===
        # Log tool errors to the persistent error log so [error] tags
        # in the UI always have a corresponding detailed entry on disk.
        _is_error_result, _ = _detect_tool_failure(function_name, function_result)
        if not _execution_blocked:
            # guardrail 不仅能 block,还能事后"补充观察信息"
            # 比如"输出含 3 个邮箱,已脱敏"之类
            function_result = agent._append_guardrail_observation(
                function_name,
                function_args,
                function_result,
                failed=_is_error_result,
            )
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
        # === c. 错误/成功 log ===
        if _is_error_result:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

        # === d. file-mutation 追踪(给 turn-end verifier) ===
        # Track file-mutation outcome for the turn-end verifier.  See
        # the concurrent path for the rationale; both paths must feed
        # the same state so the footer reflects every tool call in the
        # turn, not just the parallel ones.
        if not _execution_blocked:
            try:
                agent._record_file_mutation_result(
                    function_name, function_args, function_result, _is_error_result,
                )
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)

        # === e. tool_progress_callback "tool.completed" ===
        if not _execution_blocked and agent.tool_progress_callback:
            try:
                agent.tool_progress_callback(
                    "tool.completed", function_name, None, None,
                    duration=tool_duration, is_error=_is_error_result,
                    result=function_result,
                )
            except Exception as cb_err:
                logging.debug(f"Tool progress callback error: {cb_err}")

        # === f. activity heartbeat + 清 _current_tool ===
        agent._current_tool = None
        agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

        # === g. verbose log(只 verbose 模式) ===
        if agent.verbose_logging:
            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
            _log_result = _multimodal_text_summary(function_result)
            logging.debug(f"Tool result ({len(_log_result)} chars): {_log_result}")

        # === h. tool_complete_callback(给 plugin) ===
        if not _execution_blocked and agent.tool_complete_callback:
            try:
                agent.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
            except Exception as cb_err:
                logging.debug(f"Tool complete callback error: {cb_err}")

        # === i. 大结果落盘 + subdir hint + content 适配 ===
        # 这 3 步是**通用的后处理**,在 messages.append 之前必做
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=function_name,
            tool_use_id=tool_call.id,
            env=get_active_env(effective_task_id),
        ) if not _is_multimodal_tool_result(function_result) else function_result

        # Discover subdirectory context files from tool arguments
        subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
        if subdir_hints:
            if _is_multimodal_tool_result(function_result):
                _append_subdir_hint_to_multimodal(function_result, subdir_hints)
            else:
                function_result += subdir_hints

        # Unwrap _multimodal dicts to an OpenAI-style content list
        # (see parallel path for rationale). String results pass through.
        _tool_content = agent._tool_result_content_for_active_model(function_name, function_result)
        messages.append(make_tool_result_message(function_name, _tool_content, tool_call.id))

        # === j. /steer drain(每个工具之间) ===
        # ── Per-tool /steer drain ───────────────────────────────────
        # Drain pending steer BETWEEN individual tool calls so the
        # injection lands as soon as a tool finishes — not after the
        # entire batch.  The model sees it on the next API iteration.
        agent._apply_pending_steer_to_tool_results(messages, 1)

        # === k. 整 turn 的 print(只在非 quiet 模式) ===
        if not agent.quiet_mode:
            if agent.verbose_logging:
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", function_result))
            else:
                _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                response_preview = _fr_str[:agent.log_prefix_chars] + "..." if len(_fr_str) > agent.log_prefix_chars else _fr_str
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

        # === l. 工具完成后**再查一次** interrupt ===
        # 和 Step 1 的 interrupt 检查互为补充:
        #   Step 1 在"开工前"查
        #   这里在"完工后"查(查完准备跑下一个)
        # 这个 race window 解决:用户在前一个工具跑完、append 之后按 Ctrl-C
        if agent._interrupt_requested and i < len(assistant_message.tool_calls):
            remaining = len(assistant_message.tool_calls) - i
            agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
            for skipped_tc in assistant_message.tool_calls[i:]:
                skipped_name = skipped_tc.function.name
                # 注意这里和 Step 1 的 skip_msg 文案不同:
                # Step 1 说"was skipped due to user interrupt"
                # 这里说"was not started. User sent a new message"
                # 因为"新消息"是更具体的"已经发了 /steer 之类"的中断原因
                messages.append(make_tool_result_message(
                    skipped_name,
                    f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                    skipped_tc.id,
                ))
            break

        # === m. tool_delay(测试 / 限流用) ===
        # 比如 unit test 想模拟"工具跑得慢",可以设 tool_delay = 0.5
        # 每个工具之间 sleep 一下
        if agent.tool_delay > 0 and i < len(assistant_message.tool_calls):
            time.sleep(agent.tool_delay)

    # ════════════════════════════════════════════════════════════════
    # 整 turn 收尾(同 concurrent 版)
    # ════════════════════════════════════════════════════════════════
    # ── Per-turn aggregate budget enforcement ─────────────────────────
    # 和 concurrent 版一样的 turn-level budget 检查
    # 变量名 num_tools_seq 是为了**不**和外层循环的 num_tools 冲突(虽然这是局部)
    num_tools_seq = len(assistant_message.tool_calls)
    if num_tools_seq > 0:
        enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

    # ── /steer injection ──────────────────────────────────────────────
    # See _execute_tool_calls_parallel for the rationale. Same hook,
    # applied to sequential execution as well.
    #
    # 关键:这一段在 budget 之后跑
    # 如果先 inject steer 再 budget,steer marker 可能被截断
    if num_tools_seq > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools_seq)


# === __all__ 显式导出 ===
# 只导出两个公开函数,_ra() / _tool_search_scoped_names() 是私有
# 测试代码应该只 import 公开 API,不要碰私有 helper
__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
]




__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
]
