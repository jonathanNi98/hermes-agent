#!/usr/bin/env python3
"""
# ============================================================
# Delegate Tool —— 子智能体(Spawn Subagent)架构
# ============================================================
# 1.1 本文件做什么
# ------------------------------------------------------------
#   1) 在 LLM 看来,本文件只暴露一个 tool:`delegate_task`
#   2) 这个 tool 让 LLM 能"召唤"一个或多个 child AIAgent
#   3) 召唤出的 child:
#       - 有自己全新的 conversation(看不到父的 history)
#       - 有自己的 task_id(独立 terminal session + file_state 缓存)
#       - 只能用被裁剪过的 toolset(白名单 - 黑名单)
#       - 有自己独立的 system prompt(goal + context 拼出)
#   4) 父 LLM 只能看到最终 summary,看不到子 agent 的中间 tool call
#
# 1.2 文件组织(15 大块)
# ------------------------------------------------------------
#   1.x 模块头 + 导入 + 黑名单 + 审批回调
#   2.x Toolset 白名单
#   3.x 运行时常量 + TUI 注册表
#   4.x TUI 观测 API
#   5.x Role / Depth / Config getters
#   6.x DelegateEvent 事件枚举
#   7.x 子 agent 的 system prompt 构造
#   8.x 进度回调构造
#   9.x 核心:_build_child_agent 构造 child AIAgent
#  10.x Timeout diagnostic dump
#  11.x _run_single_child 跑单个 child
#  12.x delegate_task 顶层入口 + 凭证解析
#  13.x Schema + registry 注册
# ============================================================
"""

import enum
import json
import logging

logger = logging.getLogger(__name__)
import os
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from typing import Any, Dict, List, Optional

from toolsets import TOOLSETS

# Sentinel value used by the runtime provider system for providers that are
# not natively known (named custom providers, third-party aggregators, etc.).
# Must match hermes_cli.runtime_provider.RUNTIME_PROVIDER_TYPE_CUSTOM.
_RUNTIME_PROVIDER_CUSTOM = "custom"
from tools import file_state
from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb
from utils import base_url_hostname, is_truthy_value


# 1.3 DELEGATE_BLOCKED_TOOLS —— 子 agent 永久不能用的 5 个 tool
# ------------------------------------------------------------
#   - delegate_task:防止递归无限 fork(可由 role='orchestrator' 局部放开)
#   - clarify:        子不能跟用户交互(parent 在同步等待)
#   - memory:         防止子污染父的 MEMORY.md(全局共享文件)
#   - send_message:   防止子产生跨平台副作用(Telegram/Discord 发消息)
#   - execute_code:   鼓励子"一步一步推理"而不是写脚本一把梭
# 用 frozenset:在 _strip_blocked_tools 里 set 比较 O(1)
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "execute_code",  # children should reason step-by-step, not write scripts
    ]
)


# ---------------------------------------------------------------------------
# 1.4 子 agent 的"危险命令"审批回调(两个 + 一个选择器)
# ---------------------------------------------------------------------------
# 设计动机:
#   子 agent 跑在 ThreadPoolExecutor 的 worker thread 里。
#   父 TUI 用的 input()/prompt_toolkit 在主线程占了 stdin。
#   如果子跑 shell 时遇到 dangerous command,terminal_tool 会去
#   tools/terminal_tool.py 的 threading.local() 找审批回调 —
#   worker thread 没有,会 fallback 到 input(),结果就死锁
#   (子卡在 input() 上,父 TUI 也在等 input())。
#
# 解决:用 ThreadPoolExecutor(initializer=..., initargs=(cb,)) 给
#      每个 worker thread 注入一个**非交互式**回调,要么 auto-deny
#      要么 auto-approve。
#
# Gateway 模式不受影响 — gateway 走 tools/approval.py 的 per-session queue,
#                      不会用这里的 TLS 回调。
#
# 配置入口:delegation.subagent_auto_approve
#   false(默认) → _subagent_auto_deny    (安全,与 leaf tool 黑名单配套)
#   true         → _subagent_auto_approve(YOLO,cron / batch 场景自己开)
def _subagent_auto_deny(command: str, description: str, **kwargs) -> str:
    """# 1.4.1 默认安全姿态:遇到危险命令一律拒。

    返 'deny' → 子 agent 收到拒绝 → 可以选择**换做法**重试(不会崩)。
    关键:永远不调 input(),所以不会卡死父 TUI。
    """
    logger.warning(
        "Subagent auto-denied dangerous command: %s (%s). "
        "Set delegation.subagent_auto_approve: true to allow.",
        command, description,
    )
    return "deny"


def _subagent_auto_approve(command: str, description: str, **kwargs) -> str:
    """# 1.4.2 YOLO 姿态:由用户显式开启时注入,返 'once' 跳过审批。

    'once' 是 terminal_tool 的协议:本次放行,但下次 dangerous 命令还会再问
    (因为这是 worker thread 上的回调,父 TUI 还是看不到)。
    配 YOLO 场景:cronjob / batch 没人盯着。
    """
    logger.warning(
        "Subagent auto-approved dangerous command: %s (%s)",
        command, description,
    )
    return "once"


def _get_subagent_approval_callback():
    """# 1.4.3 选择器:读 config,决定 worker thread 上装哪个回调。

    只配 config.yaml,没有 env var 覆盖(故意的 — 改这个值要改文件)。
    返回的是**函数对象**,会在 _run_single_child 构造 ThreadPoolExecutor 时
    通过 initializer= 装到 worker thread 上。
    """
    cfg = _load_config()
    val = cfg.get("subagent_auto_approve", False)
    if is_truthy_value(val):
        return _subagent_auto_approve
    return _subagent_auto_deny

# ---------------------------------------------------------------------------
# 2.1 子 agent 可用的 toolset 白名单(用于 LLM 看到的 schema 描述)
# ---------------------------------------------------------------------------
# 这段只决定**展示给 LLM 的可选 toolset 列表**。具体的 toolset 实际可用性
# 还要再过一遍 _strip_blocked_tools / _expand_parent_toolsets。

# 排除三类 toolset:
#   1) 名字属于排除集:debugging / safe / moa / rl 这些场景化工具集
#   2) composite/platform:以 "hermes-" 开头是平台复合集
#   3) 工具全部进了黑名单的:空 toolset 给了也没用
#
# 关键:刻意把 "delegation" 也排除掉 ——
#   子 agent 想要"再 spawn 子 agent" 不应该通过 toolset 申请,
#   而是通过 role='orchestrator' 参数(_build_child_agent 会按 role
#   把 delegation toolset 重新加回去,见 9.x)。
_EXCLUDED_TOOLSET_NAMES = frozenset({"debugging", "safe", "delegation", "moa", "rl"})
_SUBAGENT_TOOLSETS = sorted(
    name
    for name, defn in TOOLSETS.items()
    if name not in _EXCLUDED_TOOLSET_NAMES
    and not name.startswith("hermes-")
    and not all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
)
# LLM 在 schema 描述里看到的就是这个串:"'terminal', 'file', 'web', ..."
_TOOLSET_LIST_STR = ", ".join(f"'{n}'" for n in _SUBAGENT_TOOLSETS)


# 2.2 运行时常量
# ---------------------------------------------------------------------------
# - _DEFAULT_MAX_CONCURRENT_CHILDREN:用户没配 delegation.max_concurrent_children 时的默认
# - MAX_DEPTH:                    _get_max_spawn_depth 的默认 fallback(测试也 import 这个符号)
# - _MIN_SPAWN_DEPTH / _MAX_SPAWN_DEPTH_CAP:合法范围 [1, 3]
#   - 1:只有父能 spawn(parent=0 → child=1),孙子直接拒
#   - 2:父 + orchestrator 子各能 spawn(支持两层嵌套)
#   - 3:最多三层(parent → orchestrator → orchestrator → leaf)
_DEFAULT_MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 1  # flat by default: parent (0) -> child (1); grandchild rejected unless max_spawn_depth raised.
# Configurable depth cap consulted by _get_max_spawn_depth; MAX_DEPTH
# stays as the default fallback and is still the symbol tests import.
_MIN_SPAWN_DEPTH = 1
_MAX_SPAWN_DEPTH_CAP = 3


# ---------------------------------------------------------------------------
# 3.1 模块级运行态(进程级单例,跨 delegate_task 调用)
# ---------------------------------------------------------------------------
# 被谁用:
#   - TUI 观测层(overlay / control surface)显示活跃子 agent
#   - gateway RPC:delegation.pause / delegation.status / subagent.interrupt
#   - 子 agent 的回调 / interrupt 传播
# 为什么模块级:跨整个 process 的 delegate_task 调用 + 嵌套 orchestrator
#            链路要共享同一个状态。
_spawn_pause_lock = threading.Lock()
_spawn_paused: bool = False

_active_subagents_lock = threading.Lock()
# subagent_id -> mutable record tracking the live child agent.  Stays only
# for the lifetime of the run; _run_single_child is the owner.
_active_subagents: Dict[str, Dict[str, Any]] = {}


# 3.2 TUI 观测 API(给 /agents overlay + gateway 用)
# ---------------------------------------------------------------------------

def set_spawn_paused(paused: bool) -> bool:
    """# 3.2.1 全局暂停新 spawn。

    - 已经在跑的子 agent **不被打断**(继续跑完)
    - 只阻止 delegate_task 接受新任务,直接 fail-fast 返 "spawning paused" 错误
    - 返新状态(给调用方确认)

    用途:runaway tree 检测 / 用户在 TUI 按 'p' 暂停 fan-out。
    """
    global _spawn_paused
    with _spawn_pause_lock:
        _spawn_paused = bool(paused)
        return _spawn_paused


def is_spawn_paused() -> bool:
    """# 3.2.2 读 _spawn_paused(用同一把锁,保证原子读)。"""
    with _spawn_pause_lock:
        return _spawn_paused


def _register_subagent(record: Dict[str, Any]) -> None:
    """# 3.2.3 把活着的子 agent 塞进 _active_subagents 字典。

    key 是 subagent_id(由 _build_child_agent 生成,见 9.x)。
    若 record 里没 subagent_id 直接 return(不抛异常,只静默放弃)。
    """
    sid = record.get("subagent_id")
    if not sid:
        return
    with _active_subagents_lock:
        _active_subagents[sid] = record


def _unregister_subagent(subagent_id: str) -> None:
    """# 3.2.4 子 agent 跑完(成功/失败/超时)时从字典里删除。pop 默认 None,不会 KeyError。"""
    with _active_subagents_lock:
        _active_subagents.pop(subagent_id, None)


def interrupt_subagent(subagent_id: str) -> bool:
    """# 3.2.5 软中断指定子 agent。

    不能硬杀 worker thread(Python 不支持),只能:
      - 设 child 的 interrupt flag
      - 该 flag 会传播到正在跑的 tool
      - 递归传给孙子(AIAgent.interrupt 内部处理)
    返 True = 找到了对应的子并发了中断;False = 没找到 / 中断失败。
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return False
    agent = record.get("agent")
    if agent is None:
        return False
    try:
        agent.interrupt(f"Interrupted via TUI ({subagent_id})")
    except Exception as exc:
        logger.debug("interrupt_subagent(%s) failed: %s", subagent_id, exc)
        return False
    return True


def list_active_subagents() -> List[Dict[str, Any]]:
    """# 3.2.6 快照当前所有活着的子 agent(给 TUI 树状显示用)。

    每条 record:subagent_id / parent_id / depth / goal / model /
                 started_at / tool_count / status
    返回的是**深拷贝**(剥掉 'agent' 字段 ——
    那是 AIAgent 引用,不能直接暴露给 TUI 序列化层)。
    任何线程调都安全 ——
    """
    with _active_subagents_lock:
        return [
            {k: v for k, v in r.items() if k != "agent"}
            for r in _active_subagents.values()
        ]


# 4.1 _extract_output_tail —— 从子 agent 的 conversation 里捞最后 N 个 tool 结果
# ---------------------------------------------------------------------------
# 用途:TUI overlay 的 "Output" 折叠面板(cc-swarm-parity 功能)。
# 两遍扫:
#   第一遍(正向):建 tool_call_id → tool_name 的索引
#                (因为 tool result 在消息流里只带 tool_call_id,
#                 不知道原本的 tool 名,得回头查 assistant 消息)
#   第二遍(反向):从消息末尾开始找 tool result,凑够 max_entries 就停
# 最后再 reverse 一下 → 按时间顺序给 TUI 展示。
# 每条:{tool, preview, is_error}
def _extract_output_tail(
    result: Dict[str, Any],
    *,
    max_entries: int = 12,
    max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    """Pull the last N tool-call results from a child's conversation.

    Powers the overlay's "Output" section — the cc-swarm-parity feature.
    We reuse the same messages list the trajectory saver walks, taking
    only the tail to keep event payloads small.  Each entry is
    ``{tool, preview, is_error}``.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return []

    # Walk in reverse to build a tail; stop when we have enough.
    tail: List[Dict[str, Any]] = []
    pending_call_by_id: Dict[str, str] = {}

    # First pass (forward): build tool_call_id -> tool_name map
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                if tc_id:
                    pending_call_by_id[tc_id] = str(fn.get("name") or "tool")

    # Second pass (reverse): pick tool results, newest first
    for msg in reversed(messages):
        if len(tail) >= max_entries:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        is_error = _looks_like_error_output(content)
        tool_name = pending_call_by_id.get(msg.get("tool_call_id") or "", "tool")
        # Preserve line structure so the overlay's wrapped scroll region can
        # show real output rather than a whitespace-collapsed blob. We still
        # cap the payload size to keep events bounded.
        preview = content[:max_chars]
        tail.append({"tool": tool_name, "preview": preview, "is_error": is_error})

    tail.reverse()  # restore chronological order for display
    return tail


# 4.2 _looks_like_error_output —— "这段 tool 输出是不是 error" 的保守判定
# ---------------------------------------------------------------------------
# 关键:从老启发式("看到 error 子串就标红")改成了**多证据**判定:
#   1) JSON 里有 "error" 键(显式)
#   2) JSON 里有 status:error/failed/failure/timeout
#   3) 第一行以经典错误标记开头:error: / failed: / Traceback / exception:
# 老启发式的 bug:正常的 terminal / JSON 输出里也常有 "error" 字样,
# 全标红太吵,严重影响 TUI 可读性。
def _looks_like_error_output(content: str) -> bool:
    """Conservative stderr/error detector for tool-result previews.

    The old heuristic flagged any preview containing the substring "error",
    which painted perfectly normal terminal/json output red.  We now only
    mark output as an error when there is stronger evidence:
      - structured JSON with an ``error`` key
      - structured JSON with ``status`` of error/failed
      - first line starts with a classic error marker
    """
    if not content:
        return False

    head = content.lstrip()
    if head.startswith("{") or head.startswith("["):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return True
                status = str(parsed.get("status") or "").strip().lower()
                if status in {"error", "failed", "failure", "timeout"}:
                    return True
        except Exception:
            pass

    first = content.splitlines()[0].strip().lower() if content.splitlines() else ""
    return (
        first.startswith("error:")
        or first.startswith("failed:")
        or first.startswith("traceback ")
        or first.startswith("exception:")
    )


# 5.1 _normalize_role —— 把用户传的 role 字符串规整成 'leaf' 或 'orchestrator'
# ---------------------------------------------------------------------------
# 规整规则(注意**默默降级** pattern —— 不抛异常):
#   None / 空       → 'leaf'
#   'leaf'          → 'leaf'
#   'orchestrator'  → 'orchestrator'
#   其他乱七八糟    → warning log + 'leaf'
# 这样上层(_build_child_agent)只需要处理两个 case。
def _normalize_role(r: Optional[str]) -> str:
    """Normalise a caller-provided role to 'leaf' or 'orchestrator'.

    None/empty -> 'leaf'.  Unknown strings coerce to 'leaf' with a
    warning log (matches the silent-degrade pattern of
    _get_orchestrator_enabled).  _build_child_agent adds a second
    degrade layer for depth/kill-switch bounds.
    """
    if r is None or not r:
        return "leaf"
    r_norm = str(r).strip().lower()
    if r_norm in {"leaf", "orchestrator"}:
        return r_norm
    logger.warning("Unknown delegate_task role=%r, coercing to 'leaf'", r)
    return "leaf"


# 5.2 _get_max_concurrent_children —— 读并发上限
# ---------------------------------------------------------------------------
# 优先级(与 delegate_task 全局保持一致):config.yaml > env > 默认
#   config:delegation.max_concurrent_children
#   env  :DELEGATION_MAX_CONCURRENT_CHILDREN
#   默认 :3
# - 下限 1(不能 0,否则 batch 永远不跑)
# - 上限 10 触发 warning(每个 child 独立消费 token,放大成本)
def _get_max_concurrent_children() -> int:
    """Read delegation.max_concurrent_children from config, falling back to
    DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (3).

    Users can raise this as high as they want; only the floor (1) is enforced.

    Uses the same ``_load_config()`` path that the rest of ``delegate_task``
    uses, keeping config priority consistent (config.yaml > env > default).
    """
    cfg = _load_config()
    val = cfg.get("max_concurrent_children")
    if val is not None:
        try:
            result = max(1, int(val))
            if result > 10:
                logger.warning(
                    "delegation.max_concurrent_children=%d: each child consumes API tokens "
                    "independently. High values multiply cost linearly.",
                    result,
                )
            return result
        except (TypeError, ValueError):
            logger.warning(
                "delegation.max_concurrent_children=%r is not a valid integer; "
                "using default %d",
                val,
                _DEFAULT_MAX_CONCURRENT_CHILDREN,
            )
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
    return _DEFAULT_MAX_CONCURRENT_CHILDREN


# 5.3 _get_child_timeout —— 单个子 agent 的硬超时(秒)
# ---------------------------------------------------------------------------
# 优先级(同样 config > env > 默认):
#   config:delegation.child_timeout_seconds
#   env  :DELEGATION_CHILD_TIMEOUT_SECONDS
#   默认 :600s(10 分钟)
# 下限 30s(再短连一次工具调用都跑不完),由 max(30.0, ...) 兜底。
# 触发后果:_run_single_child 收到 FuturesTimeoutError →
#           发中断信号 + 写 diagnostic dump(见 11.x)。
def _get_child_timeout() -> float:
    """Read delegation.child_timeout_seconds from config.

    Returns the number of seconds a single child agent is allowed to run
    before being considered stuck.  Default: 600 s (10 minutes).
    """
    cfg = _load_config()
    val = cfg.get("child_timeout_seconds")
    if val is not None:
        try:
            return max(30.0, float(val))
        except (TypeError, ValueError):
            logger.warning(
                "delegation.child_timeout_seconds=%r is not a valid number; "
                "using default %d",
                val,
                DEFAULT_CHILD_TIMEOUT,
            )
    env_val = os.getenv("DELEGATION_CHILD_TIMEOUT_SECONDS")
    if env_val:
        try:
            return max(30.0, float(env_val))
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_CHILD_TIMEOUT)


# 5.4 _get_max_spawn_depth —— 读嵌套深度上限(夹到 [1, 3])
# ---------------------------------------------------------------------------
# depth 含义:
#   0 = parent agent
#   N = 嵌套到第 N 层
# max_spawn_depth = N 表示 0..N-1 层都能 spawn,N 层就是 leaf floor。
#
# 默认 1:扁平(parent → child 就到顶,child 不能再 spawn)。
# 调高到 2/3 解锁嵌套 orchestrator。
#
# 注意 ——
#   这里是"depth ceiling",**不等于** role 开关。
#   即使 depth ≥ 2,orchestrator_enabled=false 仍然把 role 强制降级成 leaf
#   (见 _get_orchestrator_enabled)。
# 夹到 [1, 3]:超出范围会打 warning 并 clamp,而不是抛异常。
def _get_max_spawn_depth() -> int:
    """Read delegation.max_spawn_depth from config, clamped to [1, 3].

    depth 0 = parent agent.  max_spawn_depth = N means agents at depths
    0..N-1 can spawn; depth N is the leaf floor.  Default 1 is flat:
    parent spawns children (depth 1), depth-1 children cannot spawn
    (blocked by this guard AND, for leaf children, by the delegation
    toolset strip in _strip_blocked_tools).

    Raise to 2 or 3 to unlock nested orchestration. role="orchestrator"
    removes the toolset strip for depth-1 children when
    max_spawn_depth >= 2, enabling them to spawn their own workers.
    """
    cfg = _load_config()
    val = cfg.get("max_spawn_depth")
    if val is None:
        return MAX_DEPTH
    try:
        ival = int(val)
    except (TypeError, ValueError):
        logger.warning(
            "delegation.max_spawn_depth=%r is not a valid integer; " "using default %d",
            val,
            MAX_DEPTH,
        )
        return MAX_DEPTH
    clamped = max(_MIN_SPAWN_DEPTH, min(_MAX_SPAWN_DEPTH_CAP, ival))
    if clamped != ival:
        logger.warning(
            "delegation.max_spawn_depth=%d out of range [%d, %d]; " "clamping to %d",
            ival,
            _MIN_SPAWN_DEPTH,
            _MAX_SPAWN_DEPTH_CAP,
            clamped,
        )
    return clamped


# 5.5 _get_orchestrator_enabled —— orchestrator 角色的总开关
# ---------------------------------------------------------------------------
# config:delegation.orchestrator_enabled(默认 True)
# 作用:即使 depth ≥ 2,这里返 False 也会把 role='orchestrator' 强制降级成 leaf
#      并剥掉 delegation toolset。
# 用法:运营想在不删代码的情况下关掉嵌套 fan-out 时用
#      (例如某个事故后要止血)。
def _get_orchestrator_enabled() -> bool:
    """Global kill switch for the orchestrator role.

    When False, role="orchestrator" is silently forced to "leaf" in
    _build_child_agent and the delegation toolset is stripped as before.
    Lets an operator disable the feature without a code revert.
    """
    cfg = _load_config()
    val = cfg.get("orchestrator_enabled", True)
    if isinstance(val, bool):
        return val
    # Accept "true"/"false" strings from YAML that doesn't auto-coerce.
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "on"}
    return True


# 5.6 _get_inherit_mcp_toolsets —— 缩窄 toolset 时是否保留父的 MCP toolset
# ---------------------------------------------------------------------------
# config:delegation.inherit_mcp_toolsets(默认 True)
# 场景:父启用了 mcp-github,mcp-slack,delegate_task 给子指定
#       toolsets=['terminal']。
#       True:子也带 mcp-github / mcp-slack(因为父有)
#       False:子只有 terminal,不带 MCP
# 默认 True 是为了"子能继承父的资源/连接"。
def _get_inherit_mcp_toolsets() -> bool:
    """Whether narrowed child toolsets should keep the parent's MCP toolsets."""
    cfg = _load_config()
    return is_truthy_value(cfg.get("inherit_mcp_toolsets"), default=True)


# 5.7 _is_mcp_toolset_name —— 判定某个 toolset 名字是否属于 MCP
# ---------------------------------------------------------------------------
# 两层判定:
#   1) 名字以 "mcp-" 开头 → 是
#   2) 通过 tools.registry 查到别名指向的目标,以 "mcp-" 开头 → 是
# 后者用来处理"MCP toolset 给起了个别名"的情况。
# 任何异常都吞掉,返 False(默认保守)。
def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry

        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))


# 5.8 _expand_parent_toolsets —— 展开 composite toolset,让"具名 toolset"能匹配上
# ---------------------------------------------------------------------------
# 背景:父用 "hermes-cli"(composite,包含所有核心 tool)。
#       delegate_task 给子指定 toolsets=['web','terminal']。
#       简单按名字集合相交 → "web" ∉ {"hermes-cli"} → 子拿不到 web。
#       显然不对 —— 父的 hermes-cli 里明明包含 web。
#
# 解法:
#   1) 收集父所有 toolset 的 tool 名字 → 父可用 tool 集合
#   2) 遍历 TOOLSETS,凡是"全部 tool 都在父集合里"的"具名 toolset"加进来
#   3) 原父 toolset 名(hermes-cli)也保留(可能子某些逻辑要查)
# 这样'web'/'terminal'就能通过"subset 检查"被加进 expanded_parent,
# _build_child_agent 里的 set 交集就放行了。
def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Expand composite toolsets so individual toolset names are recognized.

    When a parent uses a composite toolset like ``hermes-cli`` (which bundles
    all core tools), the child may request individual toolsets such as ``web``
    or ``terminal``.  A simple name-based intersection would reject them
    because ``"web" != "hermes-cli"``.

    This helper collects the tool names from each parent toolset, then adds
    the names of any individual toolsets whose tools are a *subset* of the
    parent's available tools.  The original parent toolset names are preserved.
    """
    parent_tool_names: set = set()
    for ts_name in parent_toolsets:
        ts_def = TOOLSETS.get(ts_name)
        if ts_def:
            parent_tool_names.update(ts_def.get("tools", []))

    if not parent_tool_names:
        return set(parent_toolsets)

    expanded = set(parent_toolsets)
    for ts_name, ts_def in TOOLSETS.items():
        if ts_name in expanded:
            continue
        ts_tools = ts_def.get("tools", [])
        if ts_tools and set(ts_tools).issubset(parent_tool_names):
            expanded.add(ts_name)
    return expanded


# 5.9 _preserve_parent_mcp_toolsets —— 补上"子要的但漏写"的父 MCP toolset
# ---------------------------------------------------------------------------
# 场景:父有 {mcp-github, terminal},子要 ['terminal'] 且 inherit_mcp_toolsets=True。
#      简单交集 → ['terminal'],但 mcp-github 丢了。
#      我们希望子"该有的都有",所以把父的 MCP toolset 补回去。
# 用 sorted() 保证顺序稳定(测试断言友好)。
def _preserve_parent_mcp_toolsets(
    child_toolsets: List[str], parent_toolsets: set[str]
) -> List[str]:
    """Append any parent MCP toolsets that are missing from a narrowed child."""
    preserved = list(child_toolsets)
    for toolset_name in sorted(parent_toolsets):
        if _is_mcp_toolset_name(toolset_name) and toolset_name not in preserved:
            preserved.append(toolset_name)
    return preserved


# 5.10 默认值 + 心跳相关常量
# ---------------------------------------------------------------------------
# - DEFAULT_MAX_ITERATIONS = 50:子 agent 自己的最大循环次数(独立于父的预算)
#                                配置项:delegation.max_iterations
# - DEFAULT_CHILD_TIMEOUT   = 600s:子 agent 硬超时(见 _get_child_timeout)
# - _HEARTBEAT_INTERVAL     = 30s:心跳线程多久 touch 一次父的 _last_activity_ts
#                             (目的:防止 gateway inactivity timeout 把
#                              "正在跑 delegate_task 的父"误杀)
# - _HEARTBEAT_STALE_CYCLES_IDLE   = 15  * 30s = 450s
#     父处于"turn 之间(idle,没在跑 tool)"时,450s 没推进就停心跳
# - _HEARTBEAT_STALE_CYCLES_IN_TOOL = 40  * 30s = 1200s
#     父在跑长 tool(terminal / web fetch / file read)时,1200s 才停
#     区分两者的意义:合法长 tool 不应该被误判为"卡死"。
#     (child_timeout_seconds 600s 仍是最终硬上限)
# - DEFAULT_TOOLSETS = ["terminal", "file", "web"]:
#     父完全没指定 toolset 时给子的兜底
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_CHILD_TIMEOUT = 600  # seconds before a child agent is considered stuck
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
# Stale-heartbeat thresholds. A child with no API-call progress is either:
#   - idle between turns (no current_tool) — probably stuck on a slow API call
#   - inside a tool (current_tool set) — probably running a legitimately long
#     operation (terminal command, web fetch, large file read)
# The idle ceiling stays tight so genuinely stuck children don't mask the gateway
# timeout. The in-tool ceiling is much higher so legit long-running tools get
# time to finish; child_timeout_seconds (default 600s) is still the hard cap.
_HEARTBEAT_STALE_CYCLES_IDLE = 15  # 15 * 30s = 450s idle between turns → stale
_HEARTBEAT_STALE_CYCLES_IN_TOOL = 40  # 40 * 30s = 1200s stuck on same tool → stale
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


# ---------------------------------------------------------------------------
# Delegation progress event types
# ---------------------------------------------------------------------------


# 6.1 DelegateEvent —— 委派过程中要发出的事件枚举
# ---------------------------------------------------------------------------
# 继承 str,enum:既能用 event == DelegateEvent.TASK_xxx 比较,
#               又能直接当字符串序列化(给 gateway SSE / ACP 适配器)。
#
# 现状(去重窗口期):
#   - 旧的字符串事件名("_thinking" / "tool.started" / ...)来自老版 child agent
#   - 新的 enum 名字("delegate.task_progress" / ...)是未来方向
#   - 转换表 _LEGACY_EVENT_MAP 把老名字映射到 enum
#   - 外部消费者在 deprecation 窗口期仍会收到老字符串
#
# 保留字段(目前**不**发,留给未来):
#   TASK_SPAWNED / TASK_COMPLETED / TASK_FAILED —— orchestrator 生命周期事件
class DelegateEvent(str, enum.Enum):
    """Formal event types emitted during delegation progress.

    _build_child_progress_callback normalises incoming legacy strings
    (``tool.started``, ``_thinking``, …) to these enum values via
    ``_LEGACY_EVENT_MAP``.  External consumers (gateway SSE, ACP adapter,
    CLI) still receive the legacy strings during the deprecation window.

    TASK_SPAWNED / TASK_COMPLETED / TASK_FAILED are reserved for
    future orchestrator lifecycle events and are not currently emitted.
    """

    TASK_SPAWNED = "delegate.task_spawned"
    TASK_PROGRESS = "delegate.task_progress"
    TASK_COMPLETED = "delegate.task_completed"
    TASK_FAILED = "delegate.task_failed"
    TASK_THINKING = "delegate.task_thinking"
    TASK_TOOL_STARTED = "delegate.tool_started"
    TASK_TOOL_COMPLETED = "delegate.tool_completed"


# 6.2 _LEGACY_EVENT_MAP —— 老事件字符串 → DelegateEvent 的转换表
# ---------------------------------------------------------------------------
# 进入 _callback 的 event_type 可能是:
#   - 老字符串:"_thinking" / "tool.started" / ...
#   - 已经是 DelegateEvent 枚举(新代码会这样传)
#   - "delegate.*" 形式(新风格字符串)
# 三种都要能正常处理:已在 8.x _callback 内部按序尝试。
_LEGACY_EVENT_MAP: Dict[str, DelegateEvent] = {
    "_thinking": DelegateEvent.TASK_THINKING,
    "reasoning.available": DelegateEvent.TASK_THINKING,
    "tool.started": DelegateEvent.TASK_TOOL_STARTED,
    "tool.completed": DelegateEvent.TASK_TOOL_COMPLETED,
    "subagent_progress": DelegateEvent.TASK_PROGRESS,
}


# 6.3 check_delegate_requirements —— 委派无外部依赖,永远可用
# ---------------------------------------------------------------------------
# 给 tool registry 用,跟其他 tool 的接口保持一致(检查是否可用)。
# delegate_task 不需要额外安装 / 凭证,所以永远返 True。
def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


# 7.1 _build_child_system_prompt —— 给子 agent 拼一段"专注的任务"system prompt
# ---------------------------------------------------------------------------
# 设计要点:
#   1) 只给任务相关的信息 —— 父的 conversation 历史、记忆、其他 context
#      一律**不带**进来(子是从零开始的)
#   2) 末尾强制"汇报三件事"(做了什么 / 找到什么 / 改了哪些文件 /
#      遇到什么问题) —— 这是给父用的 summary 模板
#   3) workspace_path 必须是**绝对真实路径**才注入,否则空着不写
#      避免教会子 agent 一个假的 /workspace/... 路径
#   4) role='orchestrator' 时**追加**委派能力段(WHEN/WHEN NOT to delegate),
#      模版参考 OpenClaw buildSubagentSystemPrompt(canSpawn 分支)
#   5) depth 数字是**真实值**(由 config 算出来),不是 LLM 自己脑补
#      —— 防止它自信地说"我还能再 spawn 5 层"
def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    role: str = "leaf",
    max_spawn_depth: int = 2,
    child_depth: int = 1,
) -> str:
    """Build a focused system prompt for a child agent.

    When role='orchestrator', appends a delegation-capability block
    modeled on OpenClaw's buildSubagentSystemPrompt (canSpawn branch at
    inspiration/openclaw/src/agents/subagent-system-prompt.ts:63-95).
    The depth note is literal truth (grounded in the passed config) so
    the LLM doesn't confabulate nesting capabilities that don't exist.
    """
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    if role == "orchestrator":
        child_note = (
            "Your own children MUST be leaves (cannot delegate further) "
            "because they would be at the depth floor — you cannot pass "
            "role='orchestrator' to your own delegate_task calls."
            if child_depth + 1 >= max_spawn_depth
            else "Your own children can themselves be orchestrators or leaves, "
            "depending on the `role` you pass to delegate_task. Default is "
            "'leaf'; pass role='orchestrator' explicitly when a child "
            "needs to further decompose its work."
        )
        parts.append(
            "\n## Subagent Spawning (Orchestrator Role)\n"
            "You have access to the `delegate_task` tool and CAN spawn "
            "your own subagents to parallelize independent work.\n\n"
            "WHEN to delegate:\n"
            "- The goal decomposes into 2+ independent subtasks that can "
            "run in parallel (e.g. research A and B simultaneously).\n"
            "- A subtask is reasoning-heavy and would flood your context "
            "with intermediate data.\n\n"
            "WHEN NOT to delegate:\n"
            "- Single-step mechanical work — do it directly.\n"
            "- Trivial tasks you can execute in one or two tool calls.\n"
            "- Re-delegating your entire assigned goal to one worker "
            "(that's just pass-through with no value added).\n\n"
            "Coordinate your workers' results and synthesize them before "
            "reporting back to your parent. You are responsible for the "
            "final summary, not your workers.\n\n"
            f"NOTE: You are at depth {child_depth}. The delegation tree "
            f"is capped at max_spawn_depth={max_spawn_depth}. {child_note}"
        )
    return "\n".join(parts)


# 7.2 _resolve_workspace_hint —— best-effort 探测子 agent 该用的本地工作目录
# ---------------------------------------------------------------------------
# 探查顺序(谁有就用谁):
#   1) TERMINAL_CWD 环境变量
#   2) parent_agent._subdirectory_hints.working_dir
#   3) parent_agent.terminal_cwd
#   4) parent_agent.cwd
# 每个候选都要:
#   - 不是空
#   - expanduser + abspath 后**是绝对路径**
#   - abspath 后**真存在**这个目录
# 任何一个不行就跳下一个,都不行 → 返 None,system prompt 里就不写路径。
# 关键:宁可不给路径,也不教子一个错的(例如 /workspace/... 在容器里)。
def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts.

    We only inject a path when we have a concrete absolute directory. This avoids
    teaching subagents a fake container path while still helping them avoid
    guessing `/workspace/...` for local repo tasks.
    """
    candidates = [
        os.getenv("TERMINAL_CWD"),
        getattr(
            getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None
        ),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            continue
        if os.path.isabs(text) and os.path.isdir(text):
            return text
    return None


# 7.3 _strip_blocked_tools —— 从 toolset 列表里剥掉"不该给子"的
# ---------------------------------------------------------------------------
# 这里和 DELEGATE_BLOCKED_TOOLS(1.3)的区别:
#   1.3 是按**单个 tool 名字**黑名单 → 子拿不到 delegate_task / clarify 等
#   7.3 是按**整个 toolset 名字**剔除 → 子不能整个用 delegation / clarify
#      / memory / code_execution toolset
# 注意:
#   - 对 orchestrator 来说,"delegation" 会在 _build_child_agent 里
#     **重新加回去**(因为它的"再 spawn"能力由 role 决定,不是继承的)
#   - 单纯的 list comprehension,O(n) 非常快
def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation",
        "clarify",
        "memory",
        "code_execution",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


# 8.1 _build_child_progress_callback —— 构造一个把子 agent 事件转给父显示的回调
# ---------------------------------------------------------------------------
# 两条显示路径:
#   CLI    :通过 spinner.print_above() 在父的 delegation 转圈上方画树形
#   Gateway:批量包成 summary,定时刷给父的 progress callback
#
# identity kwargs(subagent_id / parent_id / depth / model / toolsets)
# 嵌进每一条事件,TUI 用这些**重建活的 spawn 树** + 路由
# per-branch 控制(kill / pause)回具体的 subagent_id。
# 都是可选 —— 老调用方不传也能跑(只是 TUI 上看是平铺列表)。
#
# 返 None:父根本没装 display 机制(没 spinner 也没 parent_cb),
#         子 agent 不挂回调,行为零变化(等于老版本)。
def _build_child_progress_callback(
    task_index: int,
    goal: str,
    parent_agent,
    task_count: int = 1,
    *,
    subagent_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    depth: Optional[int] = None,
    model: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    The identity kwargs (``subagent_id``, ``parent_id``, ``depth``, ``model``,
    ``toolsets``) are threaded into every relayed event so the TUI can
    reconstruct the live spawn tree and route per-branch controls (kill,
    pause) back by ``subagent_id``.  All are optional for backward compat —
    older callers that ignore them still produce a flat list on the TUI.

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    # 8.1.1 拿父的两个显示钩子(都可能为 None —— 老调用方没装)
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # 8.1.2 batch 模式才在前面加 [1] / [2] / [3] 编号
    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""
    goal_label = (goal or "").strip()

    # 8.1.3 Gateway 批处理缓冲
    # 攒 5 个 tool 名 → 合并成一条 "🔀 [1] read_file, terminal, ..."
    # 避免每条 tool 都打一次事件爆 gateway SSE
    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []
    # 用 list 包裹是为了让内层闭包能 mutate(普通 int 不可变)
    _tool_count = [0]  # per-subagent running counter (list for closure mutation)

    # 8.1.4 _identity_kwargs —— 给每条事件附上"我是谁"元数据
    def _identity_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "task_index": task_index,
            "task_count": task_count,
            "goal": goal_label,
        }
        if subagent_id is not None:
            kw["subagent_id"] = subagent_id
        if parent_id is not None:
            kw["parent_id"] = parent_id
        if depth is not None:
            kw["depth"] = depth
        if model is not None:
            kw["model"] = model
        if toolsets is not None:
            kw["toolsets"] = list(toolsets)
        kw["tool_count"] = _tool_count[0]
        return kw

    # 8.1.5 _relay —— 把事件扔给父的 parent_cb
    # 关键:kwargs 让 caller override(比如 caller 想塞 status=duration_seconds)
    # 父 callback 失败不阻断 —— 只 log debug
    def _relay(
        event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        if not parent_cb:
            return
        payload = _identity_kwargs()
        payload.update(kwargs)  # caller overrides (e.g. status, duration_seconds)
        try:
            parent_cb(event_type, tool_name, preview, args, **payload)
        except Exception as e:
            logger.debug("Parent callback failed: %s", e)

    # 8.1.6 _callback —— 子 agent 真正调用的回调
    # 它要做的事:
    #   a) 处理 lifecycle 事件(subagent.start / subagent.complete)
    #   b) 把老字符串 / 新字符串 / enum 三种 event_type 归一化
    #   c) 按事件类型分派到具体渲染 + 中继逻辑
    def _callback(
        event_type, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        # Lifecycle events emitted by the orchestrator itself — handled
        # before enum normalisation since they are not part of DelegateEvent.
        if event_type == "subagent.start":
            if spinner and goal_label:
                short = (
                    (goal_label[:55] + "...") if len(goal_label) > 55 else goal_label
                )
                try:
                    spinner.print_above(f" {prefix}├─ 🔀 {short}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.start", preview=preview or goal_label or "", **kwargs)
            return

        if event_type == "subagent.complete":
            _relay("subagent.complete", preview=preview, **kwargs)
            return

        # Normalise legacy strings, new-style "delegate.*" strings, and
        # DelegateEvent enum values all to a single DelegateEvent.  The
        # original implementation only accepted the five legacy strings;
        # enum-typed callers were silently dropped.
        if isinstance(event_type, DelegateEvent):
            event = event_type
        else:
            event = _LEGACY_EVENT_MAP.get(event_type)
            if event is None:
                try:
                    event = DelegateEvent(event_type)
                except (ValueError, TypeError):
                    return  # Unknown event — ignore

        if event == DelegateEvent.TASK_THINKING:
            text = preview or tool_name or ""
            if spinner:
                short = (text[:55] + "...") if len(text) > 55 else text
                try:
                    spinner.print_above(f' {prefix}├─ 💭 "{short}"')
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.thinking", preview=text)
            return

        if event == DelegateEvent.TASK_TOOL_COMPLETED:
            return

        if event == DelegateEvent.TASK_PROGRESS:
            # Pre-batched progress summary relayed from a nested
            # orchestrator's grandchild (upstream emits as
            # parent_cb("subagent_progress", summary_string) where the
            # summary lands in the tool_name positional slot).  Treat as
            # a pass-through: render distinctly (not via the tool-start
            # emoji lookup, which would mistake the summary string for a
            # tool name) and relay upward without re-batching.
            summary_text = tool_name or preview or ""
            if spinner and summary_text:
                try:
                    spinner.print_above(f" {prefix}├─ 🔀 {summary_text}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            if parent_cb:
                try:
                    parent_cb("subagent_progress", f"{prefix}{summary_text}")
                except Exception as e:
                    logger.debug("Parent callback relay failed: %s", e)
            return

        # TASK_TOOL_STARTED — display and batch for parent relay
        _tool_count[0] += 1
        if subagent_id is not None:
            with _active_subagents_lock:
                rec = _active_subagents.get(subagent_id)
                if rec is not None:
                    rec["tool_count"] = _tool_count[0]
                    rec["last_tool"] = tool_name or ""
        if spinner:
            short = (
                (preview[:35] + "...")
                if preview and len(preview) > 35
                else (preview or "")
            )
            from agent.display import get_tool_emoji

            emoji = get_tool_emoji(tool_name or "")
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f'  "{short}"'
            try:
                spinner.print_above(line)
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _relay("subagent.tool", tool_name, preview, args)
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
            _batch.clear()

    _callback._flush = _flush
    return _callback


# ===========================================================================
# 9. _build_child_agent —— 构造子 AIAgent(本文件最核心的函数)
# ===========================================================================
# 9.0 契约 / 在主线程上构造
# ---------------------------------------------------------------------------
# 9.0.1 什么时候调
#   delegate_task → 在主线程上对每个 task 调一次 → 拿到一个 child AIAgent
#   → 然后丢进 _run_single_child 跑(可能在 worker thread 里)
#
# 9.0.2 关键设计
#   - **在主线程构造**(thread-safe):AIAgent.__init__ 调 get_tool_definitions()
#     会改 module-global 状态(model_tools._last_resolved_tool_names)。
#     必须在主线程串行做,不能并发。所以 delegate_task 自己的 for 循环
#     也是单线程构造,然后才丢进 ThreadPoolExecutor 跑。
#   - 不跑,只构造。返回构造好的 child,实际 run 在 _run_single_child。
#
# 9.0.3 参数分组
#   - 任务元数据:task_index / goal / context / toolsets / model /
#                 max_iterations / task_count
#   - 凭证覆盖(来自 delegation config):override_provider / override_base_url /
#     override_api_key / override_api_mode
#     → 让子跑在和父不同的 provider:model 上(例如父 Nous Portal,子 OpenRouter 便宜模型)
#   - ACP 覆盖:override_acp_command / override_acp_args
#     → 让非 ACP 的父 spawn 出 ACP 子(copilot --acp --stdio)
#   - 角色:role ∈ {'leaf', 'orchestrator'} —— 决定子能不能再 spawn
#
# 9.0.4 函数体结构(后续 9.1~9.11 注释)
#   9.1  Role 解析(尊重 caller,但 kill switch / depth 会强制降级)
#   9.2  Subagent 身份(subagent_id / parent_id / tui_depth)
#   9.3  父 toolset 推导(enabled_toolsets=None 时从 valid_tool_names 反推)
#   9.4  子 toolset 计算(相交 + 补 MCP + strip 黑名单)
#   9.5  写 system prompt(goal + context + workspace + role 段)
#   9.6  凭证解析(override > 父继承,ACP 路径特殊处理)
#   9.7  Reasoning effort(可配覆盖父的)
#   9.8  Fallback chain 继承
#   9.9  Provider filters(override 时清空,否则继承)
#   9.10 构造 AIAgent(把上面所有配置塞进构造器)
#   9.11 后置绑定(深度、role 记录、active_children 注册、宣告 spawn)
def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    task_count: int,
    parent_agent,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    # ACP transport overrides — lets a non-ACP parent spawn ACP child agents
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Per-call role controlling whether the child can further delegate.
    # 'leaf' (default) cannot; 'orchestrator' retains the delegation
    # toolset subject to depth/kill-switch bounds applied below.
    role: str = "leaf",
):
    """
    Build a child AIAgent on the main thread (thread-safe construction).
    Returns the constructed child agent without running it.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent
    import uuid as _uuid

    # 9.1 Role 解析 —— 唯一一处把 role 强制降级成 'leaf' 的地方
    # ---------------------------------------------------------------------------
    # 双重护栏:
    #   a) orchestrator_enabled = False(总开关)   → 降级
    #   b) child_depth ≥ max_spawn(已经到深度上限) → 降级
    # 即 caller 传了 'orchestrator' 也要过这两道闸门。
    # 设计意图:把降级规则集中到一处,避免散在多处不一致。
    # 调用方传进来的 role 已经在 delegate_task 那里 _normalize_role 过了,
    # 所以这里只可能收到 'leaf' 或 'orchestrator'。
    # ── Role resolution ─────────────────────────────────────────────────
    # Honor the caller's role only when BOTH the kill switch and the
    # child's depth allow it.  This is the single point where role
    # degrades to 'leaf' — keeps the rule predictable.  Callers pass
    # the normalised role (_normalize_role ran in delegate_task) so
    # we only deal with 'leaf' or 'orchestrator' here.
    child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    max_spawn = _get_max_spawn_depth()
    orchestrator_ok = _get_orchestrator_enabled() and child_depth < max_spawn
    effective_role = role if (role == "orchestrator" and orchestrator_ok) else "leaf"

    # 9.2 Subagent 身份 —— 跨多个数据结构共用的 key
    # ---------------------------------------------------------------------------
    # subagent_id 在这里生成,被以下三处共享:
    #   - 进度回调(每条事件都带这个 id)
    #   - spawn_requested 事件
    #   - _active_subagents 注册表(TUI 用来定位和 kill)
    # 格式 "sa-<task_index>-<8位hex>" —— 人类可读 + 唯一
    # parent_subagent_id 在嵌套(orchestrator → worker)时非空
    # tui_depth 0 = 第一层子(给 UI 用),内部 child_depth 1
    # ── Subagent identity (stable across events, 0-indexed for TUI) ─────
    # subagent_id is generated here so the progress callback, the
    # spawn_requested event, and the _active_subagents registry all share
    # one key.  parent_id is non-None when THIS parent is itself a subagent
    # (nested orchestrator -> worker chain).
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
    tui_depth = max(0, child_depth - 1)  # 0 = first-level child for the UI

    # 提前读一次 config(后面 9.4 算 toolset 还要用 max_spawn 之类的)
    delegation_cfg = _load_config()

    # 9.3 父 toolset 推导(三种情况,取一种)
    # ---------------------------------------------------------------------------
    # a) parent.enabled_toolsets 明确指定 → 直接用
    # b) parent.enabled_toolsets is None(默认"全开")+ parent 有 valid_tool_names
    #    → 反推:每个 tool 名 → 它的 toolset → 去重得集合
    # c) 啥都没有 → 兜底 DEFAULT_TOOLSETS
    # 关键:enabled_toolsets=None 不等于"没开任何 toolset",
    #      它代表"全开",所以不能直接当空集用。
    # When no explicit toolsets given, inherit from parent's enabled toolsets
    # so disabled tools (e.g. web) don't leak to subagents.
    # Note: enabled_toolsets=None means "all tools enabled" (the default),
    # so we must derive effective toolsets from the parent's loaded tools.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        # enabled_toolsets is None (all tools) — derive from loaded tool names
        import model_tools

        parent_toolsets = {
            ts
            for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    # 9.4 子 toolset 计算(三种入口,最终都要过 _strip_blocked_tools)
    # ---------------------------------------------------------------------------
    # 路径 1:caller 显式给了 toolsets(最常见)
    #   → 跟父的 expanded_parent(见 5.8)求交集
    #   → 决定能否保留父的 MCP toolset(见 5.9)
    #   → 剥掉黑名单
    # 路径 2:caller 没给 + 父明确有 enabled_toolsets
    #   → 直接继承父的(剥黑名单)
    # 路径 3:caller 没给 + 父没明确
    #   → 用反推的 parent_toolsets(剥黑名单)
    # 路径 4:啥都没 → 兜底 DEFAULT_TOOLSETS(剥黑名单)
    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks.
        # Expand composite toolsets (e.g. hermes-cli) so that individual
        # toolset names (e.g. web, terminal) are recognised during intersection.
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        child_toolsets = [t for t in toolsets if t in expanded_parent]
        if _get_inherit_mcp_toolsets():
            child_toolsets = _preserve_parent_mcp_toolsets(
                child_toolsets, parent_toolsets
            )
        child_toolsets = _strip_blocked_tools(child_toolsets)
    elif parent_agent and parent_enabled is not None:
        child_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        child_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    # 9.4.1 Orchestrator 专属补救:把 "delegation" toolset 加回来
    # ---------------------------------------------------------------------------
    # _strip_blocked_tools 之前一刀切把 delegation 删了(见 1.3 + 7.3),
    # 但 orchestrator 必须有它才能再 spawn。
    # 不依赖父 toolset 是否包含 delegation —— orchestrator 的"再 spawn 能力"
    # 来自 role,不是继承,所以无条件加。
    # Orchestrators retain the 'delegation' toolset that _strip_blocked_tools
    # removed.  The re-add is unconditional on parent-toolset membership because
    # orchestrator capability is granted by role, not inherited — see the
    # test_intersection_preserves_delegation_bound test for the design rationale.
    if effective_role == "orchestrator" and "delegation" not in child_toolsets:
        child_toolsets.append("delegation")

    # 9.5 写子 agent 的 system prompt
    # ---------------------------------------------------------------------------
    # workspace_hint 必须是真实存在的绝对路径,见 7.2。
    # role=orchestrator 才会追加"再 spawn"段,见 7.1。
    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        workspace_path=workspace_hint,
        role=effective_role,
        max_spawn_depth=max_spawn,
        child_depth=child_depth,
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Resolve the child's effective model early so it can ride on every event.
    effective_model_for_cb = model or getattr(parent_agent, "model", None)

    # 9.6 构造进度回调 —— 把子 agent 的事件转给父的 TUI / gateway
    # ---------------------------------------------------------------------------
    # identity kwargs(subagent_id / parent_id / depth / model / toolsets)
    # 每条事件都带,TUI 用来重建活的 spawn 树 + 路由 per-branch 控制
    # Build progress callback to relay tool calls to parent display.
    # Identity kwargs thread the subagent_id through every emitted event so the
    # TUI can reconstruct the spawn tree and route per-branch controls.
    child_progress_cb = _build_child_progress_callback(
        task_index,
        goal,
        parent_agent,
        task_count,
        subagent_id=subagent_id,
        parent_id=parent_subagent_id,
        depth=tui_depth,
        model=effective_model_for_cb,
        toolsets=child_toolsets,
    )

    # 9.7 子 agent 的循环预算
    # ---------------------------------------------------------------------------
    # 每个子有自己独立的 max_iterations(默认 50,配置 delegation.max_iterations)。
    # 意味着 parent + N children 的总迭代数可以**超过** 父的 max_iterations。
    # 由用户在 config.yaml 控制。
    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    # 9.8 思考回调 —— 包装 child_progress_cb,标准化成 "_thinking" 老事件名
    # ---------------------------------------------------------------------------
    # 子 agent 的 AIAgent 在产生 thinking 时会调 thinking_callback(text),
    # 我们直接转发成 "_thinking" 给进度回调(由 8.1.6 那边归一化处理)。
    child_thinking_cb = None
    if child_progress_cb:

        def _child_thinking(text: str) -> None:
            if not text:
                return
            try:
                child_progress_cb("_thinking", text)
            except Exception as e:
                logger.debug("Child thinking callback relay failed: %s", e)

        child_thinking_cb = _child_thinking

    # 9.9 凭证解析 —— override > 父继承(每项都遵循这个优先级)
    # ---------------------------------------------------------------------------
    # Resolve effective credentials: config override > parent inherit
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    effective_api_key = override_api_key or parent_api_key
    # 9.9.1 api_mode 特殊处理(关键 bug 修复:#20558)
    # ---------------------------------------------------------------------------
    # api_mode **不能**在 provider 改变时继承父的:
    #   父用 anthropic_messages(MiniMax 走 Anthropic 协议)
    #   子 override_provider=deepseek → 应该用 chat_completions
    # 继承父的 mode → 子打 404(子用 anthropic 协议打 DeepSeek 的 endpoint)
    # 修复:provider 变 → api_mode 设 None,让 run_agent.py 按目标 provider 重新推导
    # Bug #20558 / PR #20563: api_mode must NOT be inherited when the child uses a
    # different provider than the parent — each provider has its own API surface
    # (e.g. MiniMax uses anthropic_messages, DeepSeek uses chat_completions).
    # Inheriting the parent's mode causes 404 errors when the child routes to the
    # wrong endpoint.  Derive the mode from the target provider when it differs.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # 9.9.2 ACP 覆盖(transport 切换)
    # ---------------------------------------------------------------------------
    effective_acp_command = override_acp_command or getattr(
        parent_agent, "acp_command", None
    )
    effective_acp_args = list(
        override_acp_args
        if override_acp_args is not None
        else (getattr(parent_agent, "acp_args", []) or [])
    )

    # 9.9.3 一旦 override_provider,就不能继承父的 ACP transport
    # ---------------------------------------------------------------------------
    # 父走 ACP(copilot --acp --stdio)时,acp_command 非空。
    # 子 override_provider=minimax(直连 OpenAI 兼容)时,不应该还走 ACP,
    # 否则 run_agent.py 会初始化 CopilotACPClient,完全忽略 override 凭证
    # (issue #16816)。
    # 当 override_provider is set (e.g. delegation.provider: minimax-cn),
    # the subagent must use direct API calls — not the parent's ACP transport.
    # Inheriting acp_command unconditionally causes run_agent.py to initialize
    # CopilotACPClient, bypassing override credentials entirely (issue #16816).
    if override_provider and not override_acp_command:
        effective_acp_command = None
        effective_acp_args = []

    # 9.9.4 反向:override_acp_command 强制 provider = copilot-acp
    # ---------------------------------------------------------------------------
    # 如果 caller 显式指定了 ACP transport,provider 必须是 copilot-acp
    # 才能让 run_agent.py 走 CopilotACPClient 路径。
    if override_acp_command:
        # If explicitly forcing an ACP transport override, the provider MUST be copilot-acp
        # so run_agent.py initializes the CopilotACPClient.
        effective_provider = "copilot-acp"
        effective_api_mode = "chat_completions"

    # 9.10 Reasoning effort —— delegate_task 自己的覆盖 > 父继承
    # ---------------------------------------------------------------------------
    # 配置:delegation.reasoning_effort
    # 解析失败的合法字符串 → warning + 继承父的(不动)
    # 整个 try 块挂了 → debug log + 继承父的(不阻断 spawn)
    # Resolve reasoning config: delegation override > parent inherit
    parent_reasoning = getattr(parent_agent, "reasoning_config", None)
    child_reasoning = parent_reasoning
    try:
        delegation_effort = str(delegation_cfg.get("reasoning_effort") or "").strip()
        if delegation_effort:
            from hermes_constants import parse_reasoning_effort

            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is not None:
                child_reasoning = parsed
            else:
                logger.warning(
                    "Unknown delegation.reasoning_effort '%s', inheriting parent level",
                    delegation_effort,
                )
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

    # 9.11 Fallback chain 继承
    # ---------------------------------------------------------------------------
    # 子要能像顶层 agent 那样在 rate-limit / 凭证耗尽时切到 fallback model,
    # 所以 _fallback_chain 列表要原样传过去(支持 list 和 dict 两种形式,
    # 由 AIAgent 内部处理)。
    # Inherit the parent's fallback provider chain so subagents can recover
    # from rate-limits and credential exhaustion exactly like the top-level
    # agent does.  _fallback_chain is a list accepted by AIAgent's
    # fallback_model parameter (which handles both list and dict forms).
    parent_fallback = getattr(parent_agent, "_fallback_chain", None) or None

    # 9.12 OpenRouter provider-preference filters
    # ---------------------------------------------------------------------------
    # 默认:继承父的(子走同一个 provider 时,路由约束保持一致)
    # 一旦 override_provider(走不同 provider)→ 清空 filters,
    # 否则父的 only=["Anthropic"] 会强行把子拉回父的 provider
    # (用户明明要切到 minimax 的就废了)。
    # 特殊:openrouter_min_coding_score 是 model-gated(只在 pareto-code 上发)
    # 所以即使 provider 变了也保留,其他 model 上是 no-op。
    # Inherit the parent's OpenRouter provider-preference filters by default
    # (so subagents routed to the same provider honour the same routing
    # constraints).  BUT: when `delegation.provider` is set the user is
    # explicitly asking the child to run on a different provider, and
    # parent-level OpenRouter filters (e.g. `only=["Anthropic"]`) would
    # silently force the child back onto the parent's provider. Clear the
    # filters in that case so the delegated provider is honoured.
    child_providers_allowed = getattr(parent_agent, "providers_allowed", None)
    child_providers_ignored = getattr(parent_agent, "providers_ignored", None)
    child_providers_order = getattr(parent_agent, "providers_order", None)
    child_provider_sort = getattr(parent_agent, "provider_sort", None)
    child_openrouter_min_coding_score = getattr(parent_agent, "openrouter_min_coding_score", None)
    if override_provider:
        child_providers_allowed = None
        child_providers_ignored = None
        child_providers_order = None
        child_provider_sort = None
        # Note: openrouter_min_coding_score is model-gated (only emitted on
        # openrouter/pareto-code), so we keep it inherited even when the
        # provider is overridden — it's a no-op on any other model.

    # 9.13 真正构造 AIAgent(把上面 9.1~9.12 的所有解析结果塞进构造器)
    # ---------------------------------------------------------------------------
    # 关键参数说明(为什么这样传):
    #   - quiet_mode=True:        子不直接打印,所有输出走 progress_cb → 父的 TUI
    #   - ephemeral_system_prompt:子用任务专属 prompt,**不**走父的 prompt 文件
    #   - skip_context_files=True:不读 AGENTS.md / CLAUDE.md 等(它们是父的视角)
    #   - skip_memory=True:        不写 MEMORY.md(防止子污染父的长期记忆)
    #   - clarify_callback=None:   子**不能**问用户
    #   - parent_session_id:       子记到父的 session DB 下,方便轨迹查询
    #   - iteration_budget=None:   每个子用全新的预算(独立计费)
    child = AIAgent(
        base_url=effective_base_url,
        api_key=effective_api_key,
        model=effective_model,
        provider=effective_provider,
        api_mode=effective_api_mode,
        acp_command=effective_acp_command,
        acp_args=effective_acp_args,
        max_iterations=max_iterations,
        max_tokens=getattr(parent_agent, "max_tokens", None),
        reasoning_config=child_reasoning,
        prefill_messages=getattr(parent_agent, "prefill_messages", None),
        fallback_model=parent_fallback,
        enabled_toolsets=child_toolsets,
        quiet_mode=True,
        ephemeral_system_prompt=child_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform=parent_agent.platform,
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        thinking_callback=child_thinking_cb,
        session_db=getattr(parent_agent, "_session_db", None),
        parent_session_id=getattr(parent_agent, "session_id", None),
        providers_allowed=child_providers_allowed,
        providers_ignored=child_providers_ignored,
        providers_order=child_providers_order,
        provider_sort=child_provider_sort,
        openrouter_min_coding_score=child_openrouter_min_coding_score,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
    )
    # 9.14 后置绑定(不能用构造器传的元数据,挂在 child 实例上)
    # ---------------------------------------------------------------------------
    # - _print_fn:           让子的 print 走父的(保持输出风格)
    # - _delegate_depth:     给下一层判断 depth 用
    # - _delegate_role:      记录"实际生效"的 role(kill switch 降级过的)
    # - _subagent_id:        嵌套时给孙子用 + interrupt_subagent 查找用
    # - _parent_subagent_id: 嵌套时给孙子用,定位它的"祖父"
    # - _subagent_goal:      TUI 显示用
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = child_depth
    # Stash the post-degrade role for introspection (leaf if the
    # kill switch or depth bounded the caller's requested role).
    child._delegate_role = effective_role
    # Stash subagent identity for nested-delegation event propagation and
    # for _run_single_child / interrupt_subagent to look up by id.
    child._subagent_id = subagent_id
    child._parent_subagent_id = parent_subagent_id
    child._subagent_goal = goal

    # 9.15 凭证池共享(让子能 rotate 凭证,不被钉在一个 key 上)
    # ---------------------------------------------------------------------------
    # 规则(见 _resolve_child_credential_pool):
    #   1) provider 同父 → 共享父的池(cooldown / rotation 状态同步)
    #   2) provider 不同 → 尝试加载目标 provider 自己的池
    #   3) 没有池 → 子用继承来的固定凭证(老行为)
    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(effective_provider, parent_agent)
    if child_pool is not None:
        child._credential_pool = child_pool

    # 9.16 注册到父的 _active_children(用于 interrupt 传播)
    # ---------------------------------------------------------------------------
    # 父 agent 中断时,会遍历 _active_children 给每个 child.interrupt()
    # 用 lock 保护(如果父有),没有就 best-effort 追加
    # Register child for interrupt propagation
    if hasattr(parent_agent, "_active_children"):
        lock = getattr(parent_agent, "_active_children_lock", None)
        if lock:
            with lock:
                parent_agent._active_children.append(child)
        else:
            parent_agent._active_children.append(child)

    # 9.17 立刻宣告 spawn(不等 run 开始)
    # ---------------------------------------------------------------------------
    # 子可能在 max_concurrent_children 队列里等几秒才真正开始跑,
    # TUI 要在 run 启动**之前**就有节点(用户能看到 "正在排队...")
    # Announce the spawn immediately — the child may sit in a queue
    # for seconds if max_concurrent_children is saturated, so the TUI
    # wants a node in the tree before run starts.
    if child_progress_cb:
        try:
            child_progress_cb("subagent.spawn_requested", preview=goal)
        except Exception as exc:
            logger.debug("spawn_requested relay failed: %s", exc)

    return child


# ===========================================================================
# 10. _dump_subagent_timeout_diagnostic —— 子 agent 0-API-call 超时诊断
# ===========================================================================
# 触发条件(由 11.x _run_single_child 调用):
#   - 子 agent 整体超时
#   - 在此期间**一次 API 调用都没发**(api_call_count == 0)
#
# 为什么需要这个:
#   issue #14726 —— 用户报"子 agent 超时 300s 无响应",
#   但一次 API 都没发,根本看不到子卡在哪里。
#   写一个诊断日志到 ~/.hermes/logs/subagent-<sid>-<ts>.log,
#   包含:子配置 / prompt 大小 / tool schema 大小 / 活动摘要 / worker stack。
#
# 返:写出的文件绝对路径(失败返 None)
def _dump_subagent_timeout_diagnostic(
    *,
    child: Any,
    task_index: int,
    timeout_seconds: float,
    duration_seconds: float,
    worker_thread: Optional[threading.Thread],
    goal: str,
) -> Optional[str]:
    """Write a structured diagnostic dump for a subagent that timed out
    before making any API call.

    See issue #14726: users hit "subagent timed out after 300s with no response"
    with zero API calls and no way to inspect what happened. This helper
    writes a dedicated log under ``~/.hermes/logs/subagent-<sid>-<ts>.log``
    capturing the child's config, system-prompt / tool-schema sizes, activity
    tracker snapshot, and the worker thread's Python stack at timeout.

    Returns the absolute path to the diagnostic file, or None on failure.
    """
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        import sys as _sys
        import traceback as _traceback

        hermes_home = get_hermes_home()
        logs_dir = hermes_home / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{ts}.log"

        lines: List[str] = []
        def _w(line: str = "") -> None:
            lines.append(line)

        _w(f"# Subagent timeout diagnostic — issue #14726")
        _w(f"# Generated: {_dt.datetime.now().isoformat()}")
        _w("")
        _w("## Timeout")
        _w(f"  task_index:        {task_index}")
        _w(f"  subagent_id:       {subagent_id}")
        _w(f"  configured_timeout: {timeout_seconds}s")
        _w(f"  actual_duration:   {duration_seconds:.2f}s")
        _w("")

        _w("## Goal")
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        _w(_goal_preview or "(empty)")
        _w("")

        _w("## Child config")
        for attr in (
            "model", "provider", "api_mode", "base_url", "max_iterations",
            "quiet_mode", "skip_memory", "skip_context_files", "platform",
            "_delegate_role", "_delegate_depth",
        ):
            try:
                val = getattr(child, attr, None)
                # Redact api_key-shaped values defensively
                if isinstance(val, str) and attr == "base_url":
                    pass
                _w(f"  {attr}: {val!r}")
            except Exception:
                _w(f"  {attr}: <unreadable>")
        _w("")

        _w("## Toolsets")
        enabled = getattr(child, "enabled_toolsets", None)
        _w(f"  enabled_toolsets:  {enabled!r}")
        tool_names = getattr(child, "valid_tool_names", None)
        if tool_names:
            _w(f"  loaded tool count: {len(tool_names)}")
            try:
                _w(f"  loaded tools:      {sorted(tool_names)}")
            except Exception:
                pass
        _w("")

        _w("## Prompt / schema sizes")
        try:
            sys_prompt = getattr(child, "ephemeral_system_prompt", None) \
                or getattr(child, "system_prompt", None) \
                or ""
            _w(f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if isinstance(sys_prompt, str) else 'n/a'}")
            _w(f"  system_prompt_chars: {len(sys_prompt) if isinstance(sys_prompt, str) else 'n/a'}")
        except Exception as exc:
            _w(f"  system_prompt: <error: {exc}>")
        try:
            tools_schema = getattr(child, "tools", None)
            if tools_schema is not None:
                _schema_json = json.dumps(tools_schema, default=str)
                _w(f"  tool_schema_count: {len(tools_schema)}")
                _w(f"  tool_schema_bytes: {len(_schema_json.encode('utf-8'))}")
        except Exception as exc:
            _w(f"  tool_schema: <error: {exc}>")
        _w("")

        _w("## Activity summary")
        try:
            summary = child.get_activity_summary()
            for k, v in summary.items():
                _w(f"  {k}: {v!r}")
        except Exception as exc:
            _w(f"  <get_activity_summary failed: {exc}>")
        _w("")

        _w("## Worker thread stack at timeout")
        if worker_thread is not None and worker_thread.is_alive():
            frames = _sys._current_frames()
            worker_frame = frames.get(worker_thread.ident)
            if worker_frame is not None:
                stack = _traceback.format_stack(worker_frame)
                for frame_line in stack:
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"  {sub}")
            else:
                _w("  <worker frame not available>")
        elif worker_thread is None:
            _w("  <no worker thread handle>")
        else:
            _w("  <worker thread already exited>")
        _w("")

        _w("## Notes")
        _w("  This file is written ONLY when a subagent times out with 0 API calls.")
        _w("  0-API-call timeouts mean the child never reached its first LLM request.")
        _w("  Common causes: oversized prompt rejected by provider, transport hang,")
        _w("  credential resolution stuck. See issue #14726 for context.")

        dump_path.write_text("\n".join(lines), encoding="utf-8")
        return str(dump_path)
    except Exception as exc:
        logger.warning("Subagent timeout diagnostic dump failed: %s", exc)
        return None


# ===========================================================================
# 11. _run_single_child —— 跑一个构造好的子 agent(在 worker thread 里)
# ===========================================================================
# 输入:由 _build_child_agent 构造好的 child AIAgent
# 输出:结构化 dict(task_index / status / summary / error / api_calls /
#                  duration_seconds / _child_role / _child_cost_usd /
#                  tool_trace / diagnostic_path / [stale_paths])
#
# 11.1 函数体结构(后续小节):
#   11.2 设置(进度回调 / 凭证租约 / 心跳线程)
#   11.3 心跳循环逻辑 + stale 判定
#   11.4 注册到 _active_subagents(给 TUI / interrupt 用)
#   11.5 注册任务 ID + 父读文件快照(file_state 协调)
#   11.6 跑 child.run_conversation(带硬超时)
#   11.7 超时 / 异常分支(写 diagnostic + 返回 timeout dict)
#   11.8 成功路径(flush 批处理 + 解析 status)
#   11.9 构造 tool_trace(从 messages 反推)
#   11.10 文件状态协调(子改了父读过的文件 → 提示父重读)
#   11.11 构造 complete 事件 payload
#   11.12 外层 except(任何意外都接住,返 error dict)
#   11.13 finally(停心跳 / unregister / 释放凭证租约 / 还原 global / close child)
def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built child agent. Called from within a thread.
    Returns a structured result dict.
    """
    child_start = time.monotonic()

    # 11.2 启动前的设置
    # ---------------------------------------------------------------------------
    # Get the progress callback from the child agent
    child_progress_cb = getattr(child, "tool_progress_callback", None)

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools

    _saved_tool_names = getattr(
        child, "_delegate_saved_tool_names", list(model_tools._last_resolved_tool_names)
    )

    # 11.2.1 凭证租约(避免多个子钉到同一个 key 上撞 rate-limit)
    # ---------------------------------------------------------------------------
    child_pool = getattr(child, "_credential_pool", None)
    leased_cred_id = None
    if child_pool is not None:
        # acquire_lease() 从池里租一个凭证(挑 cooldown 最久的)
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            try:
                leased_entry = child_pool.current()
                if leased_entry is not None and hasattr(child, "_swap_credential"):
                    child._swap_credential(leased_entry)
            except Exception as exc:
                logger.debug("Failed to bind child to leased credential: %s", exc)

    # 11.3 心跳线程 —— 防 gateway inactivity timeout 误杀
    # ---------------------------------------------------------------------------
    # 关键问题:
    #   delegate_task 启动后,父的 _last_activity_ts **冻结**,
    #   gateway 的 inactivity timer 在跑 → 一段时间后认为父"无活动" → 杀
    #   即使子 agent 正在忙,父的"无活动"计数也不会动。
    # 解决:每 30s 摸一下父的 _touch_activity()(让父看起来在活动)
    #
    # stale 判定 —— 跟踪子 agent 的 (iter, current_tool) 对:
    #   - 两个都没变 → 算一次 stale
    #   - 任一变了 → 重置计数
    # idle 阈值(15*30=450s)/ in-tool 阈值(40*30=1200s) 见 5.10
    # 停心跳(break)而不是继续:让 gateway 超时真正生效
    # Heartbeat: periodically propagate child activity to the parent so the
    # gateway inactivity timeout doesn't fire while the subagent is working.
    # Without this, the parent's _last_activity_ts freezes when delegate_task
    # starts and the gateway eventually kills the agent for "no activity".
    _heartbeat_stop = threading.Event()
    # Stale detection: track the child's (tool, iteration) pair across
    # heartbeat cycles. If neither advances, count the cycle as stale.
    # Different thresholds for idle vs in-tool (see _HEARTBEAT_STALE_CYCLES_*).
    _last_seen_iter = [0]
    _last_seen_tool = [None]  # type: list
    _stale_count = [0]

    # 11.3.1 心跳循环逻辑(每 _HEARTBEAT_INTERVAL 跑一次)
    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, "_touch_activity", None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)

                # Stale detection: count cycles where neither the iteration
                # count nor the current_tool advances. A child running a
                # legitimately long-running tool (terminal command, web
                # fetch) keeps current_tool set but doesn't advance
                # api_call_count — we don't want that to look stale at the
                # idle threshold.
                iter_advanced = child_iter > _last_seen_iter[0]
                tool_changed = child_tool != _last_seen_tool[0]
                if iter_advanced or tool_changed:
                    _last_seen_iter[0] = child_iter
                    _last_seen_tool[0] = child_tool
                    _stale_count[0] = 0
                else:
                    _stale_count[0] += 1

                # Pick threshold based on whether the child is currently
                # inside a tool call. In-tool threshold is high enough to
                # cover legitimately slow tools; idle threshold stays
                # tight so the gateway timeout can fire on a truly wedged
                # child.
                stale_limit = (
                    _HEARTBEAT_STALE_CYCLES_IN_TOOL
                    if child_tool
                    else _HEARTBEAT_STALE_CYCLES_IDLE
                )
                if _stale_count[0] >= stale_limit:
                    logger.warning(
                        "Subagent %d appears stale (no progress for %d "
                        "heartbeat cycles, tool=%s) — stopping heartbeat",
                        task_index,
                        _stale_count[0],
                        child_tool or "<none>",
                    )
                    break  # stop touching parent, let gateway timeout fire

                if child_tool:
                    desc = (
                        f"delegate_task: subagent running {child_tool} "
                        f"(iteration {child_iter}/{child_max})"
                    )
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (
                            f"delegate_task: subagent {child_desc} "
                            f"(iteration {child_iter}/{child_max})"
                        )
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    # daemon=True:子跑完心跳线程还没退出时,不让它阻止进程退出
    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)

    # 11.4 注册到模块级 _active_subagents(给 TUI 找子用)
    # ---------------------------------------------------------------------------
    # 注册字段:
    #   - subagent_id:  找这个子的 key
    #   - parent_id:    嵌套时定位它的"祖父"
    #   - depth:        TUI 上第一层子从 0 开始算(内部 child_depth 1)
    #   - goal / model:显示用
    #   - started_at:   unix 时间戳,算耗时用
    #   - status:       "running" → 跑完由调用方改
    #   - tool_count:   progress_cb 那边会更新
    #   - agent:        AIAgent 引用(给 interrupt_subagent 用,不暴露给 list API)
    # Test doubles(MagicMock)可能没稳定的 _subagent_id 字符串 → 跳过注册
    # Register the live agent in the module-level registry so the TUI can
    # target it by subagent_id (kill, pause, status queries).  Unregistered
    # in the finally block, even when the child raises.  Test doubles that
    # hand us a MagicMock don't carry stable ids; skip registration then.
    _raw_sid = getattr(child, "_subagent_id", None)
    _subagent_id = _raw_sid if isinstance(_raw_sid, str) else None
    if _subagent_id:
        _raw_depth = getattr(child, "_delegate_depth", 1)
        _tui_depth = max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0
        _parent_sid = getattr(child, "_parent_subagent_id", None)
        _register_subagent(
            {
                "subagent_id": _subagent_id,
                "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
                "depth": _tui_depth,
                "goal": goal,
                "model": (
                    getattr(child, "model", None)
                    if isinstance(getattr(child, "model", None), str)
                    else None
                ),
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": child,
            }
        )

    try:
        # 启动心跳线程 + 发 "subagent.start" 事件给 TUI
        _heartbeat_thread.start()
        if child_progress_cb:
            try:
                child_progress_cb("subagent.start", preview=goal)
            except Exception as e:
                logger.debug("Progress callback start failed: %s", e)

        # 11.5 子任务的 task_id + 父读文件快照
        # ---------------------------------------------------------------------------
        # 关键设计:child_task_id **复用** _subagent_id(不再生成新的 uuid)。
        # 这样三处都共享同一个 key:
        #   - file_state 写入记录
        #   - _active_subagents 注册表
        #   - TUI 事件
        # 缺失时兜底:生成 f"subagent-{idx}-{8hex}"。
        #
        # parent_reads_snapshot:跑子之前,父已经读过的文件列表。
        # 跑完用来检测"子改了父读过的文件",见 11.10。
        # File-state coordination: reuse the stable subagent_id as the child's
        # task_id so file_state writes, active-subagents registry, and TUI
        # events all share one key.  Falls back to a fresh uuid only if the
        # pre-built id is somehow missing.
        import uuid as _uuid

        child_task_id = _subagent_id or f"subagent-{task_index}-{_uuid.uuid4().hex[:8]}"
        parent_task_id = getattr(parent_agent, "_current_task_id", None)
        wall_start = time.time()
        parent_reads_snapshot = (
            list(file_state.known_reads(parent_task_id)) if parent_task_id else []
        )

        # 11.6 跑子 agent(带硬超时,防止子 hang 死)
        # ---------------------------------------------------------------------------
        # 超时机制:
        #   - 用一个**单线程** ThreadPoolExecutor 装 child.run_conversation()
        #   - initializer 把非交互审批回调装到 worker thread(防 TUI 死锁,见 1.4)
        #   - .result(timeout=child_timeout) 触发 FuturesTimeoutError
        #
        # _worker_thread_holder:timeout 时我们要 dump worker 的 Python stack
        # (issue #14726),所以要抓住 worker thread 的引用
        # Run child with a hard timeout to prevent indefinite blocking
        # when the child's API call or tool-level HTTP request hangs.
        child_timeout = _get_child_timeout()
        _timeout_executor = ThreadPoolExecutor(
            max_workers=1,
            # Install a non-interactive approval callback in the worker thread
            # so dangerous-command prompts from the subagent don't fall back to
            # input() and deadlock the parent's prompt_toolkit TUI.
            # Callback (deny vs approve) is governed by delegation.subagent_auto_approve.
            initializer=_set_subagent_approval_cb,
            initargs=(_get_subagent_approval_callback(),),
        )
        # Capture the worker thread so the timeout diagnostic can dump its
        # Python stack (see #14726 — 0-API-call hangs are opaque without it).
        _worker_thread_holder: Dict[str, Optional[threading.Thread]] = {"t": None}

        # 11.6.1 _run_with_thread_capture —— 抓 worker thread 引用 + 真正开跑
        def _run_with_thread_capture():
            _worker_thread_holder["t"] = threading.current_thread()
            return child.run_conversation(
                user_message=goal,
                task_id=child_task_id,
            )

        _child_future = _timeout_executor.submit(_run_with_thread_capture)
        try:
            result = _child_future.result(timeout=child_timeout)
        # 11.7 超时 / 异常分支
        # ---------------------------------------------------------------------------
        # 任何异常都走这里 —— 包含真超时(FuturesTimeoutError)、
        # 子抛出的普通 Exception、API 错误等。
        # 处理流程:
        #   1) 发 interrupt 给子,让它从下一个 iteration boundary 退出
        #   2) 算 duration
        #   3) 如果是 0-API-call 超时 → 写 diagnostic dump
        #   4) 发 "subagent.complete" 事件给 TUI
        #   5) 返 timeout / error dict(不再抛)
        except Exception as _timeout_exc:
            # Signal the child to stop so its thread can exit cleanly.
            try:
                if hasattr(child, "interrupt"):
                    child.interrupt()
                elif hasattr(child, "_interrupt_requested"):
                    child._interrupt_requested = True
            except Exception:
                pass

            # 11.7.1 判定:是超时还是其他异常
            is_timeout = isinstance(_timeout_exc, (FuturesTimeoutError, TimeoutError))
            duration = round(time.monotonic() - child_start, 2)
            logger.warning(
                "Subagent %d %s after %.1fs",
                task_index,
                "timed out" if is_timeout else f"raised {type(_timeout_exc).__name__}",
                duration,
            )

            # 11.7.2 0-API-call 超时 → 写 diagnostic dump(见 10.x)
            # ---------------------------------------------------------------------------
            # "超时且 0 API call" 是最 opaque 的失败:
            #   子根本没发出过请求,看不到子卡在哪
            # 见 issue #14726 —— 拿到的 diagnostic 含 prompt 大小 / 凭证状态 / worker stack
            # When a subagent times out BEFORE making any API call, dump a
            # diagnostic to help users (and us) see what the child was doing.
            # See #14726 — without this, 0-API-call hangs are black boxes.
            diagnostic_path: Optional[str] = None
            child_api_calls = 0
            try:
                _summary = child.get_activity_summary()
                child_api_calls = int(_summary.get("api_call_count", 0) or 0)
            except Exception:
                pass
            if is_timeout and child_api_calls == 0:
                diagnostic_path = _dump_subagent_timeout_diagnostic(
                    child=child,
                    task_index=task_index,
                    timeout_seconds=float(child_timeout),
                    duration_seconds=float(duration),
                    worker_thread=_worker_thread_holder.get("t"),
                    goal=goal,
                )
                if diagnostic_path:
                    logger.warning(
                        "Subagent %d 0-API-call timeout — diagnostic written to %s",
                        task_index,
                        diagnostic_path,
                    )

            if child_progress_cb:
                try:
                    child_progress_cb(
                        "subagent.complete",
                        preview=(
                            f"Timed out after {duration}s"
                            if is_timeout
                            else str(_timeout_exc)
                        ),
                        status="timeout" if is_timeout else "error",
                        duration_seconds=duration,
                        summary="",
                    )
                except Exception:
                    pass

            if is_timeout:
                if child_api_calls == 0:
                    _err = (
                        f"Subagent timed out after {child_timeout}s without "
                        f"making any API call — the child never reached its "
                        f"first LLM request (prompt construction, credential "
                        f"resolution, or transport may be stuck)."
                    )
                    if diagnostic_path:
                        _err += f" Diagnostic: {diagnostic_path}"
                else:
                    _err = (
                        f"Subagent timed out after {child_timeout}s with "
                        f"{child_api_calls} API call(s) completed — likely "
                        f"stuck on a slow API call or unresponsive network request."
                    )
            else:
                _err = str(_timeout_exc)

            return {
                "task_index": task_index,
                "status": "timeout" if is_timeout else "error",
                "summary": None,
                "error": _err,
                "exit_reason": "timeout" if is_timeout else "error",
                "api_calls": child_api_calls,
                "duration_seconds": duration,
                "_child_role": getattr(child, "_delegate_role", None),
                "diagnostic_path": diagnostic_path,
            }
        # 11.7.3 收尾(在 except 分支里,无论成功/失败都执行)
        # ---------------------------------------------------------------------------
        # wait=False:如果 worker 卡在阻塞 I/O,wait=True 会**永久卡住**
        # 关掉 executor 但不阻塞,worker 会在后台自然结束
        finally:
            # Shut down executor without waiting — if the child thread
            # is stuck on blocking I/O, wait=True would hang forever.
            _timeout_executor.shutdown(wait=False)

        # 11.8 成功路径
        # ---------------------------------------------------------------------------
        # 1) flush 批处理(把最后 < 5 个 tool 名推给 gateway)
        # 2) 解析 status:
        #    - interrupted=True → "interrupted"
        #    - 有 summary       → "completed"(看 exit_reason 知道是 done 还是 max_iter)
        #    - 都没            → "failed"
        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = result.get("api_calls", 0)

        if interrupted:
            status = "interrupted"
        elif summary:
            # A summary means the subagent produced usable output.
            # exit_reason ("completed" vs "max_iter") already
            # tells the parent *how* the task ended.
            status = "completed"
        else:
            status = "failed"

        # 11.9 从 messages 反推 tool_trace
        # ---------------------------------------------------------------------------
        # 不用 child 单独保存的 trace(那个可能不准),
        # 直接 walk 子 agent 的 messages 自己重建。
        # 两遍扫:
        #   第一遍:assistant 消息 → 建 tool_call_id → tool_name 索引
        #   第二遍:tool 消息 → 通过 tool_call_id 配对(支持并行 tool call)
        # 配不上的话 fallback 到"最后一条"(老消息没 tool_call_id 也能跑)
        # Build tool trace from conversation messages (already in memory).
        # Uses tool_call_id to correctly pair parallel tool calls with results.
        tool_trace: list[Dict[str, Any]] = []
        trace_by_id: Dict[str, Dict[str, Any]] = {}
        messages = result.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(fn.get("arguments", "")),
                        }
                        tool_trace.append(entry_t)
                        tc_id = tc.get("id")
                        if tc_id:
                            trace_by_id[tc_id] = entry_t
                elif msg.get("role") == "tool":
                    content = msg.get("content", "")
                    is_error = _looks_like_error_output(content)
                    result_meta = {
                        "result_bytes": len(content),
                        "status": "error" if is_error else "ok",
                    }
                    # Match by tool_call_id for parallel calls
                    tc_id = msg.get("tool_call_id")
                    target = trace_by_id.get(tc_id) if tc_id else None
                    if target is not None:
                        target.update(result_meta)
                    elif tool_trace:
                        # Fallback for messages without tool_call_id
                        tool_trace[-1].update(result_meta)

        # 11.9.1 解析 exit_reason(成功路径的"为什么结束")
        # ---------------------------------------------------------------------------
        # Determine exit reason
        if interrupted:
            exit_reason = "interrupted"
        elif completed:
            exit_reason = "completed"
        else:
            exit_reason = "max_iterations"

        # 11.9.2 抽 token / model(对 MagicMock 安全)
        # ---------------------------------------------------------------------------
        # Extract token counts (safe for mock objects)
        _input_tokens = getattr(child, "session_prompt_tokens", 0)
        _output_tokens = getattr(child, "session_completion_tokens", 0)
        _model = getattr(child, "model", None)

        # 11.9.3 构造最终 entry dict(返给 delegate_task)
        # ---------------------------------------------------------------------------
        # 关键字段:
        #   - tokens.{input,output}  : 来自 child(可能为 None / MagicMock → 兜 0)
        #   - tool_trace             : 见 11.9
        #   - _child_role            : 给 12.x 的 subagent_stop hook 用
        #     ⚠️ 必须在 child.close() **之前**拿,close 后属性可能被清
        #   - _child_cost_usd        : 给 12.x 的 cost rollup 用
        #     ⚠️ 同上
        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            "tokens": {
                "input": (
                    _input_tokens if isinstance(_input_tokens, (int, float)) else 0
                ),
                "output": (
                    _output_tokens if isinstance(_output_tokens, (int, float)) else 0
                ),
            },
            "tool_trace": tool_trace,
            # Captured before the finally block calls child.close() so the
            # parent thread can fire subagent_stop with the correct role.
            # Stripped before the dict is serialised back to the model.
            "_child_role": getattr(child, "_delegate_role", None),
            # Captured before child.close() so the parent aggregator can fold
            # the child's total spend into the parent's session cost.  Port of
            # Kilo-Org/kilocode#9448 — previously the footer only reflected the
            # parent's direct API calls and under-counted subagent-heavy runs.
            # Stripped before the dict is serialised back to the model.
            "_child_cost_usd": (
                float(getattr(child, "session_estimated_cost_usd", 0.0) or 0.0)
                if isinstance(
                    getattr(child, "session_estimated_cost_usd", 0.0),
                    (int, float),
                )
                else 0.0
            ),
        }
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")

        # 11.10 跨 agent 文件状态协调(关键的"父读过的文件被子改了"提示)
        # ---------------------------------------------------------------------------
        # 场景:
        #   1) 父 LLM 用 read_file 读 /foo.py(file_state 记录 parent 读了)
        #   2) delegate_task 跑子 agent
        #   3) 子用 write_file 改 /foo.py
        #   4) 父想再 edit /foo.py → 它的 view 已经过时
        # 解决:跑子前快照父的 reads;跑完查"任何非父 task_id 写过的文件",
        #      如果有父读过的 → 把这些路径附加到 entry.summary 末尾,
        #      或者写到 entry.stale_paths(给上层 UI 渲染用)。
        # 检查 ANY 非父 task_id → 也覆盖嵌套 orchestrator→worker 链。
        # Cross-agent file-state reminder.  If this subagent wrote any
        # files the parent had already read, surface it so the parent
        # knows to re-read before editing — the scenario that motivated
        # the registry.  We check writes by ANY non-parent task_id (not
        # just this child's), which also covers transitive writes from
        # nested orchestrator→worker chains.
        try:
            if parent_task_id and parent_reads_snapshot:
                sibling_writes = file_state.writes_since(
                    parent_task_id, wall_start, parent_reads_snapshot
                )
                if sibling_writes:
                    mod_paths = sorted(
                        {p for paths in sibling_writes.values() for p in paths}
                    )
                    if mod_paths:
                        reminder = (
                            "\n\n[NOTE: subagent modified files the parent "
                            "previously read — re-read before editing: "
                            + ", ".join(mod_paths[:8])
                            + (
                                f" (+{len(mod_paths) - 8} more)"
                                if len(mod_paths) > 8
                                else ""
                            )
                            + "]"
                        )
                        if entry.get("summary"):
                            entry["summary"] = entry["summary"] + reminder
                        else:
                            entry["stale_paths"] = mod_paths
        except Exception:
            logger.debug("file_state sibling-write check failed", exc_info=True)

        # 11.11 构造 subagent.complete 事件的 payload(给 TUI overlay 用)
        # ---------------------------------------------------------------------------
        # 字段内容:
        #   - preview:           截 160 字,显示在 TUI 主线上
        #   - status:            "completed" / "failed" / "interrupted"
        #   - summary:           截 500 字,detail 面板用
        #   - tokens.{in,out,reasoning}: 来自 child
        #   - files_read/written:从 file_state 抽,各 cap 40
        #   - output_tail:       见 4.1,8 条 / 各 600 字
        #   - cost_usd:          可选(MagicMock / 老 fixture 拿不到)
        # Per-branch observability payload: tokens, cost, files touched, and
        # a tail of tool-call results.  Fed into the TUI's overlay detail
        # pane + accordion rollups (features 1, 2, 4).  All fields are
        # optional — missing data degrades gracefully on the client.
        _cost_usd = getattr(child, "session_estimated_cost_usd", None)
        _reasoning_tokens = getattr(child, "session_reasoning_tokens", 0)
        try:
            _files_read = list(file_state.known_reads(child_task_id))[:40]
        except Exception:
            _files_read = []
        try:
            _files_written_map = file_state.writes_since(
                "", wall_start, []
            )  # all writes since wall_start
        except Exception:
            _files_written_map = {}
        _files_written = sorted(
            {
                p
                for tid, paths in _files_written_map.items()
                if tid == child_task_id
                for p in paths
            }
        )[:40]

        _output_tail = _extract_output_tail(result, max_entries=8, max_chars=600)

        complete_kwargs: Dict[str, Any] = {
            "preview": summary[:160] if summary else entry.get("error", ""),
            "status": status,
            "duration_seconds": duration,
            "summary": summary[:500] if summary else entry.get("error", ""),
            "input_tokens": (
                int(_input_tokens) if isinstance(_input_tokens, (int, float)) else 0
            ),
            "output_tokens": (
                int(_output_tokens) if isinstance(_output_tokens, (int, float)) else 0
            ),
            "reasoning_tokens": (
                int(_reasoning_tokens)
                if isinstance(_reasoning_tokens, (int, float))
                else 0
            ),
            "api_calls": int(api_calls) if isinstance(api_calls, (int, float)) else 0,
            "files_read": _files_read,
            "files_written": _files_written,
            "output_tail": _output_tail,
        }
        if _cost_usd is not None:
            try:
                complete_kwargs["cost_usd"] = float(_cost_usd)
            except (TypeError, ValueError):
                pass

        if child_progress_cb:
            try:
                child_progress_cb("subagent.complete", **complete_kwargs)
            except Exception as e:
                logger.debug("Progress callback completion failed: %s", e)

        return entry

    # 11.12 外层 except —— 任何上面 try 块里没被内部 except 抓住的异常
    # ---------------------------------------------------------------------------
    # (e.g. setup 阶段就崩 / heartbeat start 失败 / 极端情况)
    # 统一兜底:返 error dict,绝不抛(否则会污染父的 conversation)
    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        if child_progress_cb:
            try:
                child_progress_cb(
                    "subagent.complete",
                    preview=str(exc),
                    status="failed",
                    duration_seconds=duration,
                    summary=str(exc),
                )
            except Exception as e:
                logger.debug("Progress callback failure relay failed: %s", e)
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
            "_child_role": getattr(child, "_delegate_role", None),
        }

    # 11.13 finally —— 无论成功/失败/超时/异常都要跑的清理
    # ---------------------------------------------------------------------------
    # 清理顺序(每步都防漏,失败只 log debug 不阻断):
    #   1) 停心跳(用 Event 让 _heartbeat_loop 自然退出;join 设 5s 上限)
    #   2) 从 _active_subagents 注销
    #   3) 释放凭证租约(让其他子能拿到这个 key)
    #   4) 还原 model_tools._last_resolved_tool_names(全局状态)
    #   5) 从父的 _active_children 移除(用锁保护)
    #   6) close child(关掉 terminal sandbox / browser / httpx client)
    finally:
        # 11.13.1 停心跳 + join(ident 守护:.start() 没跑 ident 仍是 None)
        # ---------------------------------------------------------------------------
        # Stop the heartbeat thread so it doesn't keep touching parent activity
        # after the child has finished (or failed).  Guard the join: .start()
        # now lives inside the try block, so if it raised (OS thread
        # exhaustion) the thread was never started and Thread.join() would
        # raise RuntimeError.  ident is None until start() succeeds.
        _heartbeat_stop.set()
        if _heartbeat_thread.ident is not None:
            _heartbeat_thread.join(timeout=5)

        # 11.13.2 从 TUI 注册表里注销
        # ---------------------------------------------------------------------------
        # Drop the TUI-facing registry entry.  Safe to call even if the
        # child was never registered (e.g. ID missing on test doubles).
        if _subagent_id:
            _unregister_subagent(_subagent_id)

        # 11.13.3 释放凭证租约(让其他子 / 下一轮能用这个 key)
        # ---------------------------------------------------------------------------
        if child_pool is not None and leased_cred_id is not None:
            try:
                child_pool.release_lease(leased_cred_id)
            except Exception as exc:
                logger.debug("Failed to release credential lease: %s", exc)

        # 11.13.4 还原全局 _last_resolved_tool_names(给后续 execute_code 等用)
        # ---------------------------------------------------------------------------
        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Remove child from active tracking

        # 11.13.5 从父的 _active_children 移除(给 interrupt 传播用)
        # ---------------------------------------------------------------------------
        # Unregister child from interrupt propagation
        if hasattr(parent_agent, "_active_children"):
            try:
                lock = getattr(parent_agent, "_active_children_lock", None)
                if lock:
                    with lock:
                        parent_agent._active_children.remove(child)
                else:
                    parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)

        # 11.13.6 关掉子 agent 持有的资源
        # ---------------------------------------------------------------------------
        # 包括:terminal sandbox / browser daemon / 后台 process / httpx client
        # 必须关,否则子进程会 outlive 委派,浪费 fd / 端口 / 内存
        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) so subagent subprocesses
        # don't outlive the delegation.
        try:
            if hasattr(child, "close"):
                child.close()
        except Exception:
            logger.debug("Failed to close child agent after delegation")


# ===========================================================================
# 12. delegate_task —— 顶层入口(给 LLM 看到的 tool handler)
# ===========================================================================
# 12.0 契约
# ---------------------------------------------------------------------------
# 12.0.1 输入模式
#   模式 1 单任务:  goal + context + toolsets + role
#   模式 2 批任务:  tasks=[{goal, context, ...}, ...]
#                  一起并行跑,返 N 个结果
#
# 12.0.2 输出
#   永远返 JSON 字符串:
#     {"results": [entry, entry, ...], "total_duration_seconds": ...}
#   任何错误也是 tool_error() 返的 JSON,绝不抛异常
#
# 12.0.3 函数体结构(后续 12.1~12.11)
#   12.1  _recover_tasks_from_json_string —— 容错 JSON 字符串
#   12.2  守卫:parent_agent 必须有 / 暂停态 / depth 上限
#   12.3  加载 config(并发上限 / max_iterations / 凭证)
#   12.4  规范化 role / 解析 task list / 验证
#   12.5  保存父 tool names(防子构造污染 global)+ 构建所有子
#   12.6  单任务:直接同步跑;多任务:ThreadPoolExecutor 并行
#   12.7  通知父的 memory provider(可选)
#   12.8  触发 subagent_stop hook + 累加 child cost
#   12.9  把子成本合并到父 session
#   12.10 返最终 JSON
# ===========================================================================

# 12.1 _recover_tasks_from_json_string —— 把 "tasks" 字符串容错地解析
# ---------------------------------------------------------------------------
# 场景:某些 LLM 不知怎么把 tasks 整个序列化成 JSON 字符串再传进来。
# 这里尝试反序列化,失败给个清晰的错误消息,而不是让上游崩。
# 返 (parsed_list | None, error_msg | None)
#  - 成功:(list, None)
#  - 不是字符串:(None, None)—— 让上层走"没传 tasks"的分支
#  - 解析失败:(None, "清晰的错误消息")
#  - 解析成功但不是 list:(None, "类型不对的错误消息")
def _recover_tasks_from_json_string(
    tasks: Any,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not isinstance(tasks, str):
        return None, None
    raw = tasks.strip()
    if not raw:
        return None, "Provide either 'goal' (single task) or 'tasks' (batch)."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (
            "tasks must be a JSON array of task objects; received a string "
            f"that could not be parsed as JSON ({exc.msg})."
        )
    if not isinstance(parsed, list):
        return None, (
            f"tasks must be a JSON array of task objects; parsed "
            f"{type(parsed).__name__} instead."
        )
    return parsed, None


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets, role)
      - Batch:  provide tasks array [{goal, context, toolsets, role}, ...]

    The 'role' parameter controls whether a child can further delegate:
    'leaf' (default) cannot; 'orchestrator' retains the delegation
    toolset and can spawn its own workers, bounded by
    delegation.max_spawn_depth.  Per-task role beats the top-level one.

    Returns JSON with results array, one entry per task.
    """
    # 12.2 入口守卫
    # ---------------------------------------------------------------------------
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # 12.2.1 全局暂停闸门(TUI / RPC 触发,见 3.2.1)
    # ---------------------------------------------------------------------------
    # Operator-controlled kill switch — lets the TUI freeze new fan-out
    # when a runaway tree is detected, without interrupting already-running
    # children.  Cleared via the matching `delegation.pause` RPC.
    if is_spawn_paused():
        return tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI "
            "(`p` in /agents) or the `delegation.pause` RPC before retrying."
        )

    # 12.3 加载 config + 规范化 role
    # ---------------------------------------------------------------------------
    # Normalise the top-level role once; per-task overrides re-normalise.
    top_role = _normalize_role(role)

    # 12.3.1 depth 闸门
    # ---------------------------------------------------------------------------
    # depth = parent._delegate_depth(parent 是 0,第一层子 1,...)
    # ≥ max_spawn 直接拒(防止无限嵌套)
    # 错误消息里给用户指出可调的 config
    # Depth limit — configurable via delegation.max_spawn_depth,
    # default 2 for parity with the original MAX_DEPTH constant.
    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return json.dumps(
            {
                "error": (
                    f"Delegation depth limit reached (depth={depth}, "
                    f"max_spawn_depth={max_spawn}). Raise "
                    f"delegation.max_spawn_depth in config.yaml if deeper "
                    f"nesting is required (cap: {_MAX_SPAWN_DEPTH_CAP})."
                )
            }
        )

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    # 12.3.2 关键安全设计:忽略 caller 传的 max_iterations
    # ---------------------------------------------------------------------------
    # config 是**权威**值。理由:
    #   - 用户配的预算是可预测的(父 50 步 + 子 50 步 = 100 步可控)
    #   - LLM 自作主张传更小值 → 中途截断让用户懵
    #   - 万一从老 schema / stale provider 漏过来,debug log 一下直接丢弃
    # Model-supplied max_iterations is ignored — the config value is authoritative
    # so users get predictable budgets. The kwarg is retained for internal callers
    # and tests; a model-emitted value here would only shrink the budget and
    # surprise the user mid-run. Log and drop it if one slips through from a
    # cached tool schema or a stale provider.
    if max_iterations is not None and max_iterations != default_max_iter:
        logger.debug(
            "delegate_task: ignoring caller-supplied max_iterations=%s; "
            "using delegation.max_iterations=%s from config",
            max_iterations, default_max_iter,
        )
    effective_max_iter = default_max_iter

    # 12.3.3 解析凭证覆盖(delegation.provider / delegation.base_url)
    # ---------------------------------------------------------------------------
    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    # 12.4 规范成 task list(单任务包成单元素列表)
    # ---------------------------------------------------------------------------
    # 优先级:
    #   1) tasks 是 list → 直接用(也可能是 _recover_tasks_from_json_string 解析回来的)
    #   2) tasks 是字符串 → 走 JSON 反序列化(见 12.1)
    #   3) 没 tasks 但有 goal → 包成 [{goal, context, toolsets, role}]
    #   4) 都没有 → 报错
    # Normalize to task list
    max_children = _get_max_concurrent_children()
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return tool_error(tasks_error)
    if recovered_tasks is not None:
        tasks = recovered_tasks

    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            return tool_error(
                f"Too many tasks: {len(tasks)} provided, but "
                f"max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple "
                f"delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [
            {"goal": goal, "context": context, "toolsets": toolsets, "role": top_role}
        ]
    else:
        return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")

    if not task_list:
        return tool_error("No tasks provided.")

    # 12.4.1 验证每个 task 都合法
    # ---------------------------------------------------------------------------
    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            return tool_error(
                f"Task {i} must be an object, got {type(task).__name__}."
            )
        if not task.get("goal", "").strip():
            return tool_error(f"Task {i} is missing a 'goal'.")

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # 12.5 保存父的 tool names(防子构造污染 global)
    # ---------------------------------------------------------------------------
    # _build_child_agent() → AIAgent() → get_tool_definitions() 会覆盖
    # model_tools._last_resolved_tool_names(写的是子的 toolset)。
    # 必须在**第一个子构造之前**把父的 toolset 快照下来。
    # 后续每个子构造后,把子构造写到 child._delegate_saved_tool_names,
    # _run_single_child 的 finally 会再把 global 还原回来(见 11.13.4)。
    # Save parent tool names BEFORE any child construction mutates the global.
    # _build_child_agent() calls AIAgent() which calls get_tool_definitions(),
    # which overwrites model_tools._last_resolved_tool_names with child's toolset.
    import model_tools as _model_tools

    _parent_tool_names = list(_model_tools._last_resolved_tool_names)

    # 12.5.1 构造所有子 agent(主线程串行,见 9.0)
    # ---------------------------------------------------------------------------
    # try/finally 保证:
    #   - 任一子构造抛异常 → global 也能还原(否则污染后续)
    #   - 所有子都构造完 → 一次性还原(而不是每个子之后还原一次)
    # Build all child agents on the main thread (thread-safe construction)
    # Wrapped in try/finally so the global is always restored even if a
    # child build raises (otherwise _last_resolved_tool_names stays corrupted).
    children = []
    try:
        for i, t in enumerate(task_list):
            task_acp_args = t.get("acp_args") if "acp_args" in t else None
            # Per-task role beats top-level; normalise again so unknown
            # per-task values warn and degrade to leaf uniformly.
            effective_role = _normalize_role(t.get("role") or top_role)
            child = _build_child_agent(
                task_index=i,
                goal=t["goal"],
                context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets,
                model=creds["model"],
                max_iterations=effective_max_iter,
                task_count=n_tasks,
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_acp_command=t.get("acp_command")
                or acp_command
                or creds.get("command"),
                override_acp_args=(
                    task_acp_args
                    if task_acp_args is not None
                    else (acp_args if acp_args is not None else creds.get("args"))
                ),
                role=effective_role,
            )
            # 关键:把"父的 toolset"挂到 child 上,给 11.13.4 还原 global 用
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    # 12.6 单任务 vs 批任务
    # ---------------------------------------------------------------------------
    # 单任务:直接同步跑(省一个 ThreadPoolExecutor 的开销)
    # 多任务:max_workers=max_children 的 ThreadPoolExecutor 并行
    #       用 wait(timeout=0.5) 轮询而不是 as_completed() —
    #         防止父被 interrupt 时还卡在 as_completed 上
    if n_tasks == 1:
        # 12.6.1 单任务路径:直接调,不开线程池
        # ---------------------------------------------------------------------------
        # Single task -- run directly (no thread pool overhead)
        _i, _t, child = children[0]
        result = _run_single_child(0, _t["goal"], child, parent_agent)
        results.append(result)
    else:
        # 12.6.2 批任务路径:ThreadPoolExecutor 并行 + 轮询中断
        # ---------------------------------------------------------------------------
        # Batch -- run in parallel with per-task progress lines
        completed_count = 0
        spinner_ref = getattr(parent_agent, "_delegate_spinner", None)

        with ThreadPoolExecutor(max_workers=max_children) as executor:
            futures = {}
            for i, t, child in children:
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    child=child,
                    parent_agent=parent_agent,
                )
                futures[future] = i

            # Poll futures with interrupt checking.  as_completed() blocks
            # until ALL futures finish — if a child agent gets stuck,
            # the parent blocks forever even after interrupt propagation.
            # Instead, use wait() with a short timeout so we can bail
            # when the parent is interrupted.
            # Map task_index -> child agent, so fabricated entries for
            # still-pending futures can carry the correct _delegate_role.
            _child_by_index = {i: child for (i, _, child) in children}

            pending = set(futures.keys())
            # 12.6.3 轮询循环:每 0.5s 检查一次,同时看父是否被 interrupt
            # ---------------------------------------------------------------------------
            # 不能用 as_completed():会阻塞到所有 future 跑完
            # 用 wait(timeout=0.5) + FIRST_COMPLETED → 每 0.5s 醒一次
            # 父被 interrupt → 收已经完成的 + 给剩下的标 "interrupted"
            while pending:
                if getattr(parent_agent, "_interrupt_requested", False) is True:
                    # Parent interrupted — collect whatever finished and
                    # abandon the rest.  Children already received the
                    # interrupt signal; we just can't wait forever.
                    for f in pending:
                        idx = futures[f]
                        if f.done():
                            try:
                                entry = f.result()
                            except Exception as exc:
                                entry = {
                                    "task_index": idx,
                                    "status": "error",
                                    "summary": None,
                                    "error": str(exc),
                                    "api_calls": 0,
                                    "duration_seconds": 0,
                                    "_child_role": getattr(
                                        _child_by_index.get(idx), "_delegate_role", None
                                    ),
                                }
                        else:
                            entry = {
                                "task_index": idx,
                                "status": "interrupted",
                                "summary": None,
                                "error": "Parent agent interrupted — child did not finish in time",
                                "api_calls": 0,
                                "duration_seconds": 0,
                                "_child_role": getattr(
                                    _child_by_index.get(idx), "_delegate_role", None
                                ),
                            }
                        results.append(entry)
                        completed_count += 1
                    break

                from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED

                done, pending = _cf_wait(
                    pending, timeout=0.5, return_when=FIRST_COMPLETED
                )
                for future in done:
                    try:
                        entry = future.result()
                    except Exception as exc:
                        idx = futures[future]
                        entry = {
                            "task_index": idx,
                            "status": "error",
                            "summary": None,
                            "error": str(exc),
                            "api_calls": 0,
                            "duration_seconds": 0,
                            "_child_role": getattr(
                                _child_by_index.get(idx), "_delegate_role", None
                            ),
                        }
                    results.append(entry)
                    completed_count += 1

                    # 12.6.4 每个 task 跑完,在 spinner 上方打一行 "✓ [1/3] ..."
                    # ---------------------------------------------------------------------------
                    # Print per-task completion line above the spinner
                    idx = entry["task_index"]
                    label = (
                        task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                    )
                    dur = entry.get("duration_seconds", 0)
                    status = entry.get("status", "?")
                    icon = "✓" if status == "completed" else "✗"
                    remaining = n_tasks - completed_count
                    completion_line = f"{icon} [{idx+1}/{n_tasks}] {label}  ({dur}s)"
                    if spinner_ref:
                        try:
                            spinner_ref.print_above(completion_line)
                        except Exception:
                            print(f"  {completion_line}")
                    else:
                        print(f"  {completion_line}")

                    # Update spinner text to show remaining count
                    if spinner_ref and remaining > 0:
                        try:
                            spinner_ref.update_text(
                                f"🔀 {remaining} task{'s' if remaining != 1 else ''} remaining"
                            )
                        except Exception as e:
                            logger.debug("Spinner update_text failed: %s", e)

        # 12.6.5 按 task_index 排,保证 results 顺序 = 输入顺序
        # ---------------------------------------------------------------------------
        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    # 12.7 通知父的 memory provider(可选,带 _memory_manager 才调)
    # ---------------------------------------------------------------------------
    # 把每个 task 的 goal + summary 喂给 memory manager,让它可以记"我委派过这些事"
    # Notify parent's memory provider of delegation outcomes
    if (
        parent_agent
        and hasattr(parent_agent, "_memory_manager")
        and parent_agent._memory_manager
    ):
        for entry in results:
            try:
                _task_goal = (
                    task_list[entry["task_index"]]["goal"]
                    if entry["task_index"] < len(task_list)
                    else ""
                )
                parent_agent._memory_manager.on_delegation(
                    task=_task_goal,
                    result=entry.get("summary", "") or "",
                    child_session_id=(
                        getattr(children[entry["task_index"]][2], "session_id", "")
                        if entry["task_index"] < len(children)
                        else ""
                    ),
                )
            except Exception:
                pass

    # 12.8 触发 subagent_stop hook + 累加 child cost
    # ---------------------------------------------------------------------------
    # 重要:在**父线程**串行触发,不是 worker thread 里触发。
    # 原因:Python plugin / shell hook 作者不用考虑并发
    #      (hook 内部有非线程安全状态的话也不会爆)
    # 顺便累加 _children_cost_total(给 12.9 用),不重复 walk results。
    # Fire subagent_stop hooks once per child, serialised on the parent thread.
    # This keeps Python-plugin and shell-hook callbacks off of the worker threads
    # that ran the children, so hook authors don't need to reason about
    # concurrent invocation.  Role was captured into the entry dict in
    # _run_single_child (or the fabricated-entry branches above) before the
    # child was closed.
    _parent_session_id = getattr(parent_agent, "session_id", None)
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
    except Exception:
        _invoke_hook = None
    # Aggregate child spend here so the parent's footer/UI reflect the true
    # cost of a subagent-heavy turn.  Port of Kilo-Org/kilocode#9448.  Each
    # child's cost was captured in _run_single_child before its AIAgent was
    # closed; we fold them into the parent in one pass alongside the
    # subagent_stop hook loop so we don't walk `results` twice.
    _children_cost_total = 0.0
    # 12.8.1 一个循环做两件事:消费临时字段 + 触发 hook + 累加 cost
    # ---------------------------------------------------------------------------
    for entry in results:
        # _child_role / _child_cost_usd 是 _run_single_child 加的临时字段
        # 这里 pop 出来用,顺手从 entry 里删掉(返回给 LLM 前清除)
        child_role = entry.pop("_child_role", None)
        child_cost = entry.pop("_child_cost_usd", 0.0)
        try:
            if child_cost:
                _children_cost_total += float(child_cost)
        except (TypeError, ValueError):
            pass
        if _invoke_hook is None:
            continue
        try:
            _invoke_hook(
                "subagent_stop",
                parent_session_id=_parent_session_id,
                child_role=child_role,
                child_summary=entry.get("summary"),
                child_status=entry.get("status"),
                duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
            )
        except Exception:
            logger.debug("subagent_stop hook invocation failed", exc_info=True)

    # 12.9 把子成本合并到父的 session(递归 rollup)
    # ---------------------------------------------------------------------------
    # 关键性质 —— **可加性**:
    #   每个 delegate_task 调用都把自己的"直接子"成本加到父上。
    #   嵌套时(orchestrator 自己也调 delegate_task):
    #     - orchestrator 层把自己直接子的 cost 加到自己上
    #     - orchestrator 跑完,父把 orchestrator 整个(已含子的 cost)再加到自己上
    #   最终父能看到"完整树"的总成本。
    # cost_source / cost_status 字段升级:
    #   父自己没花过钱(只 delegate_task)→ UI 会显示 "none",
    #   现在应该显示 "subagent"(数据其实是子贡献的)
    # Fold the aggregated child cost into the parent's session total.  This is
    # additive — each delegate_task call contributes its own children — so
    # nested orchestrator→worker trees roll up naturally: each layer's own
    # delegate_task() folds its direct children in, and when the orchestrator
    # itself finishes, its parent folds the orchestrator's now-inflated total
    # on top.  Degrades silently if the parent lacks the counter (older test
    # fixtures, etc.).
    if _children_cost_total > 0.0:
        try:
            current = float(getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0)
            parent_agent.session_estimated_cost_usd = current + _children_cost_total
            # Upgrade the cost_source so the UI doesn't label a partially-real
            # total as "none" when the parent itself hadn't billed any calls
            # yet (rare but possible when the parent's only action this turn
            # was delegate_task).
            if getattr(parent_agent, "session_cost_source", "none") in {None, "", "none"}:
                parent_agent.session_cost_source = "subagent"
            if getattr(parent_agent, "session_cost_status", "unknown") in {None, "", "unknown"}:
                parent_agent.session_cost_status = "estimated"
        except Exception:
            logger.debug("Subagent cost rollup failed", exc_info=True)

    # 12.10 拼最终 JSON
    # ---------------------------------------------------------------------------
    # 永远返 JSON 字符串(给 LLM 看的 tool result)
    total_duration = round(time.monotonic() - overall_start, 2)

    return json.dumps(
        {
            "results": results,
            "total_duration_seconds": total_duration,
        },
        ensure_ascii=False,
    )


# 12.11.1 _resolve_child_credential_pool —— 给子找凭证池
# ---------------------------------------------------------------------------
# 规则(按顺序尝试):
#   1) 子没指定 provider → 共享父的池
#   2) 子 provider 同父 → 共享父的池(cooldown / rotation 状态同步)
#   3) 子 provider 不同 → 加载目标 provider 自己的池
#   4) 没有 → 返 None,子用继承来的固定凭证
def _resolve_child_credential_pool(effective_provider: Optional[str], parent_agent):
    """Resolve a credential pool for the child agent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the child keep the inherited
       fixed credential behavior.
    """
    if not effective_provider:
        return getattr(parent_agent, "_credential_pool", None)

    parent_provider = getattr(parent_agent, "provider", None) or ""
    parent_pool = getattr(parent_agent, "_credential_pool", None)
    if parent_pool is not None and effective_provider == parent_provider:
        return parent_pool

    try:
        from agent.credential_pool import load_pool

        pool = load_pool(effective_provider)
        if pool is not None and pool.has_credentials():
            return pool
    except Exception as exc:
        logger.debug(
            "Could not load credential pool for child provider '%s': %s",
            effective_provider,
            exc,
        )
    return None


# 12.11.2 _resolve_delegation_credentials —— 解析 delegation.{provider,base_url,...}
# ---------------------------------------------------------------------------
# 两条独立路径:
#   路径 A:配置了 delegation.base_url
#     → 用直连 OpenAI 兼容端点
#     → api_key 没配 → 返 None(让子继承父的)
#       (provider 把 key 存到非 OPENAI_API_KEY 的环境变量时,
#        不需要用户在 delegation.api_key 重复配)
#     → 自动按 base_url 推断 api_mode
#       (Azure AI Foundry / MiniMax / Zhipu GLM / LiteLLM 代理等
#        Anthropic 兼容端点 → anthropic_messages,
#        不靠用户手工配 api_mode,见 #10213)
#   路径 B:配置了 delegation.provider(没 base_url)
#     → 走完整的 runtime provider 解析(和 CLI / gateway 启动一样的路径)
#     → 拿完整的 (base_url, api_key, api_mode, provider) 凭证包
#   路径 C:啥都没配
#     → 返 None 全字段 → 子继承父
#
# 任何路径上的 ValueError 都向上抛 → 12.3.3 翻译成 tool_error 返给 LLM
def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.base_url`` is configured, subagents use that direct
    OpenAI-compatible endpoint. ``delegation.api_key`` overrides the key; when
    omitted, ``api_key`` is returned as ``None`` so ``_build_child_agent``
    inherits the parent agent's key (``effective_api_key = override_api_key or
    parent_api_key``). This lets providers that store their key outside
    ``OPENAI_API_KEY`` (e.g. ``MINIMAX_API_KEY``, ``DASHSCOPE_API_KEY``) work
    without a duplicate config entry.

    Otherwise, if ``delegation.provider`` is configured, the full credential
    bundle (base_url, api_key, api_mode, provider) is resolved via the runtime
    provider system — the same path used by CLI/gateway startup. This lets
    subagents run on a completely different provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None
    configured_api_mode = str(cfg.get("api_mode") or "").strip().lower() or None

    if configured_base_url:
        # When delegation.api_key is not set, return None so _build_child_agent
        # falls back to the parent agent's API key via the credential inheritance
        # path (effective_api_key = override_api_key or parent_api_key). This
        # lets providers that store their key in a non-OPENAI_API_KEY env var
        # (e.g. MINIMAX_API_KEY, DASHSCOPE_API_KEY) work without requiring
        # callers to duplicate the key under delegation.api_key.
        api_key = configured_api_key  # None → inherited from parent in _build_child_agent

        # Use the shared URL-based api_mode detector (same path the main agent's
        # runtime resolver uses) so Anthropic-compatible direct endpoints with a
        # /anthropic suffix — Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM
        # proxies — pick the right transport automatically. Without this,
        # subagents would default to chat_completions and hit 404s on endpoints
        # that only speak the Anthropic Messages protocol. Fixes #10213.
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = _detect_api_mode_for_url(configured_base_url) or "chat_completions"
        if (
            base_url_hostname(configured_base_url) == "chatgpt.com"
            and "/backend-api/codex" in base_lower
        ):
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif base_url_hostname(configured_base_url) == "api.anthropic.com":
            provider = "anthropic"
            api_mode = "anthropic_messages"
        elif "api.kimi.com/coding" in base_lower:
            provider = "custom"
            api_mode = "anthropic_messages"

        # Explicit delegation.api_mode in config always wins. Lets users force
        # a transport for non-standard endpoints the URL heuristic can't detect.
        if configured_api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            api_mode = configured_api_mode

        return {
            "model": configured_model,
            "provider": provider,
            "base_url": configured_base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=configured_provider, target_model=configured_model)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes auth'."
        )

    return {
        "model": configured_model or runtime.get("model") or None,
        "provider": configured_provider if runtime.get("provider") == _RUNTIME_PROVIDER_CUSTOM else runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


# 12.11.3 _load_config —— 读 delegation 这块配置
# ---------------------------------------------------------------------------
# 优先级:
#   1) cli.CLI_CONFIG.get("delegation")   ← CLI 进程内 config
#   2) hermes_cli.config.load_config().get("delegation")   ← 持久 config.yaml
#   3) {}   ← 都没有
# 任意异常都吞 → 返 {} → 走默认值
# 这么设计是确保 CLI / gateway / cron 各种入口都能拿到 delegation.*
# Load delegation config from CLI_CONFIG or persistent config.
def _load_config() -> dict:
    """Load delegation config from CLI_CONFIG or persistent config.

    Checks the runtime config (cli.py CLI_CONFIG) first, then falls back
    to the persistent config (hermes_cli/config.py load_config()) so that
    ``delegation.model`` / ``delegation.provider`` are picked up regardless
    of the entry point (CLI, gateway, cron).
    """
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation") or {}
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        full = load_config()
        return full.get("delegation") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------


# ===========================================================================
# 13. Schema 构造 + registry 注册
# ===========================================================================
# 13.0 为什么这块很关键
# ---------------------------------------------------------------------------
# LLM 看到的 tool 描述不是写死的 —— 它要随 config 变(用户的并发上限、
# nesting 是否开启)动态生成。否则 LLM 会"自我封顶在 default 3 / default 2"上,
# 即使用户已经调高了 delegation.max_concurrent_children。
#
# 模式:DELEGATE_TASK_SCHEMA 是**占位**的 description,
#       真正的 description 每次 get_definitions() 调
#       _build_dynamic_schema_overrides() 现场拼。
# ===========================================================================

# 13.1 _build_top_level_description —— 顶层 description(给 LLM 的 tool 描述)
# ---------------------------------------------------------------------------
# 三个核心事实必须随 config 变:
#   - max_concurrent_children(影响 batch 上限)
#   - max_spawn_depth / orchestrator_enabled(影响 nesting 段)
# 错误路径全包 try → 失败用默认值(描述生成失败不应该让 tool 装不上)
def _build_top_level_description() -> str:
    """Compose the delegate_task tool description with current runtime limits.

    The model needs to know its actual ceilings (not the framework defaults),
    otherwise it self-caps at "default 3" / "default 2" even when the user has
    raised delegation.max_concurrent_children / max_spawn_depth. Called both
    at module import (to seed DELEGATE_TASK_SCHEMA) and on every
    get_definitions() call via dynamic_schema_overrides.
    """
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    try:
        orchestrator_on = _get_orchestrator_enabled()
    except Exception:
        orchestrator_on = True

    if max_depth >= 2 and orchestrator_on:
        nesting_clause = (
            f"Nested delegation IS enabled for this user "
            f"(max_spawn_depth={max_depth}): pass role='orchestrator' on a "
            f"child to let it spawn its own workers, up to {max_depth - 1} "
            f"additional level(s) deep."
        )
    elif max_depth >= 2 and not orchestrator_on:
        nesting_clause = (
            f"Nested delegation is DISABLED on this install "
            f"(delegation.orchestrator_enabled=false), even though "
            f"max_spawn_depth={max_depth}. role='orchestrator' is silently "
            f"forced to 'leaf'."
        )
    else:
        nesting_clause = (
            f"Nested delegation is OFF for this user "
            f"(max_spawn_depth={max_depth}): every child is a leaf and "
            f"cannot delegate further. Raise delegation.max_spawn_depth in "
            f"config.yaml to enable nesting."
        )

    return (
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        f"2. Batch (parallel): provide 'tasks' array with up to {max_children} "
        f"items concurrently for this user (configured via "
        f"delegation.max_concurrent_children in config.yaml). "
        f"All run in parallel and results are returned together. {nesting_clause}\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n"
        "- Durable long-running work that must outlive the current turn -> "
        "use cronjob (action='create') or terminal(background=True, "
        "notify_on_complete=True) instead. delegate_task runs SYNCHRONOUSLY "
        "inside the parent turn: if the parent is interrupted (user sends a "
        "new message, /stop, /new) the child is cancelled with status="
        "'interrupted' and its work is discarded. Children cannot continue "
        "in the background.\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- If the user is writing in a non-English language, or asked for "
        "output in a specific language / tone / style, say so in 'context' "
        "(e.g. \"respond in Chinese\", \"return output in Japanese\"). "
        "Otherwise subagents default to English and their summaries will "
        "contaminate your final reply with the wrong language.\n"
        "- Subagent summaries are SELF-REPORTS, not verified facts. A subagent "
        "that claims \"uploaded successfully\" or \"file written\" may be wrong. "
        "For operations with external side-effects (HTTP POST/PUT, remote "
        "writes, file creation at shared paths, publishing), require the "
        "subagent to return a verifiable handle (URL, ID, absolute path, HTTP "
        "status) and verify it yourself — fetch the URL, stat the file, read "
        "back the content — before telling the user the operation succeeded.\n"
        "- Leaf subagents (role='leaf', the default) CANNOT call: "
        "delegate_task, clarify, memory, send_message, execute_code.\n"
        "- Orchestrator subagents (role='orchestrator') retain "
        "delegate_task so they can spawn their own workers, but still "
        "cannot use clarify, memory, send_message, or execute_code. "
        f"Orchestrators are bounded by max_spawn_depth={max_depth} for this "
        f"user and can be disabled globally via "
        "delegation.orchestrator_enabled=false.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task."
    )


# 13.2 _build_tasks_param_description —— tasks 字段的动态描述
# ---------------------------------------------------------------------------
# 把"最多 N 个并发"这个数字直接写进 description(让 LLM 知道)
def _build_tasks_param_description() -> str:
    """Compose the 'tasks' parameter description with current concurrency limit."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    return (
        f"Batch mode: tasks to run in parallel (up to {max_children} for this "
        f"user, set via delegation.max_concurrent_children). Each gets "
        "its own subagent with isolated context and terminal session. "
        "When provided, top-level goal/context/toolsets are ignored."
    )


# 13.3 _build_role_param_description —— role 字段的动态描述
# ---------------------------------------------------------------------------
# 描述要诚实告诉 LLM 当前用户到底能不能用 'orchestrator':
#   - nesting on  → "可以,最大 N 层"
#   - nesting off → "config 关了,会被强制降级"
#   - nesting disabled by depth → "config 允许但深度不够,降级"
def _build_role_param_description() -> str:
    """Compose the 'role' parameter description with current spawn-depth limit."""
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    try:
        orchestrator_on = _get_orchestrator_enabled()
    except Exception:
        orchestrator_on = True

    if max_depth >= 2 and orchestrator_on:
        nesting_note = (
            f"Nesting IS enabled for this user (max_spawn_depth={max_depth}): "
            f"orchestrator children can themselves delegate up to {max_depth - 1} "
            "more level(s) deep."
        )
    elif max_depth >= 2 and not orchestrator_on:
        nesting_note = (
            "Nesting is currently disabled "
            "(delegation.orchestrator_enabled=false); 'orchestrator' is "
            "silently forced to 'leaf'."
        )
    else:
        nesting_note = (
            f"Nesting is OFF for this user (max_spawn_depth={max_depth}); "
            "'orchestrator' is silently forced to 'leaf'. Raise "
            "delegation.max_spawn_depth in config.yaml to enable."
        )

    return (
        "Role of the child agent. 'leaf' (default) = focused "
        "worker, cannot delegate further. 'orchestrator' = can "
        f"use delegate_task to spawn its own workers. {nesting_note}"
    )


# 13.4 _build_dynamic_schema_overrides —— 每次 get_definitions() 调一次
# ---------------------------------------------------------------------------
# 拼装成 dict 喂给 ToolEntry.dynamic_schema_overrides:
#   - 顶层 description:_build_top_level_description()
#   - tasks.description:_build_tasks_param_description()
#   - role.description: _build_role_param_description()
# 重要:深拷贝 properties —— 不能改写 DELEGATE_TASK_SCHEMA 静态字典
# Return per-call schema overrides reflecting current config.
def _build_dynamic_schema_overrides() -> dict:
    """Return per-call schema overrides reflecting current config.

    Plugged into ToolEntry.dynamic_schema_overrides so every
    get_definitions() pass rewrites the description fields to the user's
    actual limits.
    """
    overrides_params = {
        **DELEGATE_TASK_SCHEMA["parameters"],
    }
    # Deep-copy properties so we don't mutate the static schema dict.
    overrides_params["properties"] = {
        k: dict(v) for k, v in DELEGATE_TASK_SCHEMA["parameters"]["properties"].items()
    }
    overrides_params["properties"]["tasks"]["description"] = _build_tasks_param_description()
    overrides_params["properties"]["role"]["description"] = _build_role_param_description()
    return {
        "description": _build_top_level_description(),
        "parameters": overrides_params,
    }


# 13.5 DELEGATE_TASK_SCHEMA —— 静态占位 schema
# ---------------------------------------------------------------------------
# 顶层 description / tasks.description / role.description 都是**占位**的:
#   真正文字由 _build_dynamic_schema_overrides() 在每次 get_definitions() 时
#   现场拼,这样 LLM 看到的是**当前用户的**实际 limit。
# 字段意义:
#   - name:         tool 标识(delegate_task)
#   - description:  占位(每调用时覆盖)
#   - parameters.properties:
#     - goal:       必填文本(任务描述)
#     - context:    选填文本(背景信息)
#     - toolsets:   选填数组(子可用的 toolset 列表)
#     - tasks:      批任务模式(每个任务是个 dict,goal 必填)
#     - role:       'leaf' 或 'orchestrator'(子能不能再 spawn)
#     - acp_command/args: 覆盖 ACP transport
#   - parameters.required: 空 → 至少传 goal 或 tasks 任一(运行时校验)
# tasks 数组没 maxItems —— 并发上限由 runtime 在 delegate_task() 里强制
DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    # NOTE: description / tasks.description / role.description are placeholder
    # values. The real text is generated per get_definitions() call by
    # _build_dynamic_schema_overrides() (registered via
    # dynamic_schema_overrides below) so the model sees the user's actual
    # delegation.max_concurrent_children / max_spawn_depth, not the framework
    # defaults. Building these lazily (instead of at module import) also
    # avoids forcing cli.CLI_CONFIG to load before the test conftest can
    # redirect HERMES_HOME.
    "description": (
        "Spawn one or more subagents in isolated contexts. "
        "Description is rebuilt at every get_definitions() call to reflect "
        "the user's current delegation limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained -- the subagent knows nothing about your "
                    "conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints. The more "
                    "specific you are, the better the subagent performs."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Toolsets to enable for this subagent. "
                    "Default: inherits your enabled toolsets. "
                    f"Available toolsets: {_TOOLSET_LIST_STR}. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['browser'] for web interaction, "
                    "['terminal', 'file', 'web'] for full-stack tasks."
                ),
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Task goal"},
                        "context": {
                            "type": "string",
                            "description": "Task-specific context",
                        },
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Toolsets for this specific task. Available: {_TOOLSET_LIST_STR}. Use 'web' for network access, 'terminal' for shell, 'browser' for web interaction.",
                        },
                        "acp_command": {
                            "type": "string",
                            "description": (
                                "Per-task ACP command override (e.g. 'copilot'). "
                                "Overrides the top-level acp_command for this task only. "
                                "Do NOT set unless the user explicitly told you an ACP CLI is installed."
                            ),
                        },
                        "acp_args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Per-task ACP args override. Leave empty unless acp_command is set.",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["leaf", "orchestrator"],
                            "description": "Per-task role override. See top-level 'role' for semantics.",
                        },
                    },
                    "required": ["goal"],
                },
                # No maxItems — the runtime limit is configurable via
                # delegation.max_concurrent_children (default 3) and
                # enforced with a clear error in delegate_task().
                "description": "(rebuilt at get_definitions() time)",
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": "(rebuilt at get_definitions() time)",
            },
            "acp_command": {
                "type": "string",
                "description": (
                    "Override ACP command for child agents (e.g. 'copilot'). "
                    "When set, children use ACP subprocess transport instead of inheriting "
                    "the parent's transport. Requires an ACP-compatible CLI "
                    "(currently GitHub Copilot CLI via 'copilot --acp --stdio'). "
                    "See agent/copilot_acp_client.py for the implementation. "
                    "IMPORTANT: Do NOT set this unless the user has explicitly told you "
                    "a specific ACP-compatible CLI is installed and configured. "
                    "Leave empty to use the parent's default transport (Hermes subagents)."
                ),
            },
            "acp_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Arguments for the ACP command (default: ['--acp', '--stdio']). "
                    "Only used when acp_command is set. "
                    "Leave empty unless acp_command is explicitly provided."
                ),
            },
        },
        "required": [],
    },
}


# 13.6 registry.register —— 把这个 tool 真的装到 tool registry 里
# ---------------------------------------------------------------------------
# 关键点:
#   - handler:把 schema 的 args dict 翻译成本文件顶层 delegate_task() 的参数
#     (registry 调用时 kwargs 会包含 parent_agent 之类上下文)
#   - check_fn:check_delegate_requirements() → 永远 True(见 6.3)
#   - emoji:"🔀" → TUI 上显示
#   - dynamic_schema_overrides:_build_dynamic_schema_overrides
#     → 每次 LLM 要 schema 时现场拼当前 config
# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        tasks=args.get("tasks"),
        max_iterations=args.get("max_iterations"),
        acp_command=args.get("acp_command"),
        acp_args=args.get("acp_args"),
        role=args.get("role"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)
