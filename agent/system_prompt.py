"""System-prompt assembly for :class:`AIAgent`.

The agent's system prompt is built once per session and reused across all
turns — only context compression triggers a rebuild.  This keeps the
upstream prefix cache warm.  See ``hermes-agent-dev``'s
``references/system-prompt-invariant.md`` for the invariants and
``references/self-improvement-loop.md`` for how the background-review
fork inherits the cached prompt verbatim.

Three tiers are joined with ``\\n\\n``:

* ``stable``   — identity (SOUL.md or DEFAULT_AGENT_IDENTITY), tool
  guidance, computer-use guidance, nous subscription block, tool-use
  enforcement guidance + per-model operational guidance, skills prompt,
  alibaba model-name workaround, environment hints, platform hints.
* ``context``  — caller-supplied ``system_message`` plus context files
  (AGENTS.md / .cursorrules / etc.) discovered under ``TERMINAL_CWD``.
* ``volatile`` — memory snapshot, USER.md profile, external memory
  provider block, timestamp/session/model/provider line.

Pure helpers that read the agent's state.  AIAgent keeps thin forwarders.
"""
# ═══════════════════════════════════════════════════════════════
# 【学习要点】system_prompt.py 的核心设计 —— 3 层分块 + prefix cache 友好
#
# 整个文件解决的问题:
#   "怎么拼 system prompt 才能让 LLM provider 的 prefix cache 一直命中"
#
# 关键设计: 把 system prompt 拆成 3 个 tier,每个 tier 变化频率不同
#   stable   — 一次 AIAgent 实例内,跨 turn 字节级稳定 (身份/工具指导/平台提示)
#              → Anthropic/OpenAI prefix cache 命中 100% (省 75% token 费)
#   context  — session 内不变 (caller system_message + 项目 AGENTS.md)
#              → cache 大部分命中
#   volatile — 每 session 变 (memory 快照/USER.md/时间戳)
#              → cache 主动让出,每次重新计算
#
# 调用入口: run_agent.py 的 _restore_or_build_system_prompt()
#           → agent._cached_system_prompt 有值直接用(快路径)
#           → 没值调 build_system_prompt() (慢路径,首次)
#           → 压缩后调 invalidate_system_prompt() 让缓存失效
#
# 跟 conversation_loop.py 的关系:
#   run_conversation → build_messages → 取 agent._cached_system_prompt
#   → 拼到 api_messages[0] (system role) 发给 LLM
# ═══════════════════════════════════════════════════════════════


from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# 【步骤 1】从 prompt_builder 导入 11 个常量(GUIDANCE 文本块)
# 这些是硬编码在 prompt_builder.py 里的"长文本指令"
# 全部以 _GUIDANCE 结尾,代表它们会拼到 system prompt 里
#
# 分类速记:
#   身份/全局  : DEFAULT_AGENT_IDENTITY (fallback 身份)
#   工具相关  : MEMORY/SESSION_SEARCH/SKILLS/KANBAN_GUIDANCE
#              (有对应工具才注入)
#   通用准则  : HERMES_AGENT_HELP / TASK_COMPLETION / TOOL_USE_ENFORCEMENT
#   按模型分发 : GOOGLE_MODEL_OPERATIONAL / OPENAI_MODEL_EXECUTION
#              (gemini/gpt/codex/grok 才注入)
#   平台字典  : PLATFORM_HINTS (按 agent.platform 索引)
#   模型白名单: TOOL_USE_ENFORCEMENT_MODELS (auto 模式默认匹配)
# ═══════════════════════════════════════════════════════════════
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
)


def _ra():
    """Lazy reference to the ``run_agent`` module.

    Helpers like ``load_soul_md``, ``build_environment_hints``,
    ``build_context_files_prompt``, ``build_nous_subscription_prompt``,
    ``build_skills_system_prompt`` and ``get_toolset_for_tool`` are
    imported into ``run_agent``'s namespace.  Many tests
    ``patch("run_agent.load_soul_md", ...)``; if we imported them
    directly here those patches would not reach us.  Looking them up
    through ``run_agent`` on every call preserves the patch contract.
    """
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 2】_ra() lazy import 辅助 —— 测试 patch 兼容设计
    #
    # 为什么需要这个:
    #   单元测试经常用 mock.patch("run_agent.load_soul_md", ...) 改函数行为
    #   如果这里直接 `from run_agent import load_soul_md`,
    #   → 测试的 patch 改的是 run_agent 模块的属性
    #   → 这里的本地引用还是旧函数,patch 不生效
    #
    # 解法: 每次调 _ra() 都重新 import run_agent
    #   → 取到的是最新的 run_agent 命名空间
    #   → patch 改完会反映出来
    #
    # 性能: import 走 sys.modules 缓存,每次成本几乎为 0
    #
    # 使用:
    #   _r = _ra()
    #   content = _r.load_soul_md()
    #   content = _r.build_environment_hints()
    # ═══════════════════════════════════════════════════════════════
    # 2.1 lazy import: 每次调都重新走 sys.modules
    import run_agent
    return run_agent


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """Assemble the system prompt as three ordered parts.

    Returns a dict with three keys:
      * ``stable``   — identity, tool guidance, skills prompt,
        environment hints, platform hints, model-family operational
        guidance.
      * ``context``  — context files (AGENTS.md, .cursorrules, etc.)
        and caller-supplied system_message.
      * ``volatile`` — memory snapshot, user profile, external
        memory provider block, timestamp line.

    Joined into a single string by :func:`build_system_prompt` and
    cached on ``agent._cached_system_prompt`` for the lifetime of the
    AIAgent.  Hermes never re-renders parts of this string mid-
    session — that's the only way to keep upstream prompt caches
    warm across turns.
    """
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 3】build_system_prompt_parts 函数入口
    # 这是 system_prompt.py 的"主函数",返回 3 键 dict
    #
    # 调用链:
    #   build_system_prompt() (line 348)
    #     └─► build_system_prompt_parts() (本函数)
    #           ├─► stable tier (line 84-281)
    #           ├─► context tier (line 282-299)
    #           └─► volatile tier (line 301-339)
    #
    # 关键设计点:
    #   1. agent 用 Any 类型 → 解耦 class 依赖(测试方便)
    #   2. system_message 来自 caller(cli.py / gateway)
    #   3. 整个 dict 在 AIAgent 生命周期内**只算一次**
    #   4. 压缩后调 invalidate_system_prompt() 强制重算
    # ═══════════════════════════════════════════════════════════════
    # Local import to avoid pulling model_tools at module load.  Tests
    # patch ``run_agent.get_toolset_for_tool`` and similar helpers, so
    # we resolve through ``_ra()`` to honor those patches.
    # 3.1 解析 _ra() lazy reference(走 run_agent 命名空间,保 patch 兼容)
    _r = _ra()

    # ── Stable tier ────────────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4】开始装配 stable tier(一次 AIAgent 实例内,跨 turn 字节级稳定)
    #
    # stable tier 包含 (按添加顺序):
    #   1. 身份 (SOUL.md 或 DEFAULT_AGENT_IDENTITY)
    #   2. HERMES_AGENT_HELP_GUIDANCE
    #   3. TASK_COMPLETION_GUIDANCE (可选)
    #   4. 工具相关 guidance 块 (按 valid_tool_names 注入)
    #   5. COMPUTER_USE_GUIDANCE (macOS)
    #   6. Nous 订阅块
    #   7. TOOL_USE_ENFORCEMENT_GUIDANCE (按配置)
    #   8. 按 model 注入 GOOGLE/OPENAI 指引
    #   9. Skills prompt (有 skills 工具时)
    #   10. Alibaba model name 兜底
    #   11. 环境探测 (WSL/Termux + Python 工具链)
    #   12. Active profile 提示
    #   13. 平台提示 (PLATFORM_HINTS)
    #
    # 设计原则: **每个块都是可选的 / 按需注入**
    #   → 减少不必要的 token 消耗
    #   → 例如没 memory 工具就不注入 MEMORY_GUIDANCE
    # ═══════════════════════════════════════════════════════════════
    stable_parts: List[str] = []

    # Try SOUL.md as primary identity unless the caller explicitly skipped it.
    # Some execution modes (cron) still want HERMES_HOME persona while keeping
    # cwd project instructions disabled.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.1】加载 SOUL.md 作为身份(优先于默认身份)
    # 触发条件: load_soul_identity=True 或 没跳过 context 文件
    # 来源: load_soul_md() 从 ~/.hermes/SOUL.md 或项目级 SOUL.md 读
    # 用途: 让用户自定义 agent 的"人格"和"身份"
    # ═══════════════════════════════════════════════════════════════
    _soul_loaded = False
    # 4.1.1 尝试加载 SOUL.md
    if agent.load_soul_identity or not agent.skip_context_files:
        # 4.1.2 通过 _r 调 load_soul_md (走 run_agent 命名空间,test patch 友好)
        _soul_content = _r.load_soul_md()
        if _soul_content:
            # 4.1.3 SOUL.md 找到了 → 加进 stable
            stable_parts.append(_soul_content)
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 4.2】SOUL.md 没找到 → fallback 默认身份
        # DEFAULT_AGENT_IDENTITY 来自 prompt_builder.py
        # 是硬编码的 "你是 Hermes Agent" 自我介绍
        # ═══════════════════════════════════════════════════════════════
        stable_parts.append(DEFAULT_AGENT_IDENTITY)


    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.3】HERMES_AGENT_HELP_GUIDANCE — "用户问 Hermes 怎么用" 的指引
    # 总是注入(不需要条件)
    # 告诉 model: 当用户问 "Hermes 怎么用" / "怎么配置" → 调 skill 或读 docs
    # 而不是 model 自己瞎编
    # ═══════════════════════════════════════════════════════════════
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

    # Universal task-completion / no-fabrication guidance.  Applied to ALL
    # models regardless of tool_use_enforcement gating — the failure modes
    # this targets (stopping after a stub; fabricating output when a real
    # path is blocked) are not model-family specific.  Gated only by
    # config.yaml ``agent.task_completion_guidance`` (default True) so
    # users who want a leaner prompt can turn it off.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.4】TASK_COMPLETION_GUIDANCE — 防"半截答案" + 防"瞎编"
    # 触发条件:
    #   - agent._task_completion_guidance=True (默认)
    #   - agent.valid_tool_names 非空(有工具可用)
    # 防的问题:
    #   (a) model 写个 stub("下面我会读文件...") 就停了
    #   (b) 工具失败时 model 自己编一个"结果"出来
    # 配置开关: config.yaml `agent.task_completion_guidance`
    #   True (默认) | False (想让 prompt 更精简可关)
    # ═══════════════════════════════════════════════════════════════
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)

    # Tool-aware behavioral guidance: only inject when the tools are loaded
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.5】工具相关 guidance 块(按需注入)
    # 设计: 收集到一个 list,最后用空格拼成一个 stable 块
    #      (不是分别 append,避免多个换行)
    #
    # 每个 guidance 对应一个具体工具:
    #   memory 工具         → MEMORY_GUIDANCE
    #   session_search 工具 → SESSION_SEARCH_GUIDANCE
    #   skill_manage 工具   → SKILLS_GUIDANCE
    #   kanban_* 工具       → KANBAN_GUIDANCE (或 _kanban_worker_guidance)
    #   computer_use 工具   → COMPUTER_USE_GUIDANCE (单独块,多段)
    #
    # 节省 token 关键: 没这个工具就不注入对应指引
    # ═══════════════════════════════════════════════════════════════
    tool_guidance = []
    # 4.5.1 memory 工具 → MEMORY_GUIDANCE (怎么用持久化记忆)
    if "memory" in agent.valid_tool_names:
        tool_guidance.append(MEMORY_GUIDANCE)
    # 4.5.2 session_search 工具 → SESSION_SEARCH_GUIDANCE (搜索历史 session)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    # 4.5.3 skill_manage 工具 → SKILLS_GUIDANCE (管理 skills)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)
    # Kanban worker/orchestrator lifecycle — only present when the
    # dispatcher spawned this process (kanban_show check_fn gates on
    # HERMES_KANBAN_TASK env var). Normal chat sessions never see
    # this block. Resolved once at __init__ (see _kanban_worker_guidance).
    # 4.5.4 kanban worker/orchestrator 生命周期(异步任务系统)
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance:
        # 4.5.4a init 时解析好的(看 env 决定 worker 还是 orchestrator)
        tool_guidance.append(_kanban_guidance)
    elif _kanban_guidance is None and "kanban_show" in agent.valid_tool_names:
        # Fallback for code paths that bypass agent_init (rare).
        # 4.5.4b 罕见:绕过 agent_init 但有 kanban 工具 → 用默认 KANBAN_GUIDANCE
        tool_guidance.append(KANBAN_GUIDANCE)
    # 4.5.5 拼成一个块(避免多个空行)
    if tool_guidance:
        stable_parts.append(" ".join(tool_guidance))

    # Computer-use (macOS) — goes in as its own block rather than being
    # merged into tool_guidance because the content is multi-paragraph.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.6】COMPUTER_USE_GUIDANCE — macOS 自动化指导
    # 单独成一个 stable 块(不用空格拼)
    # 因为内容是多段,需要保留段落结构
    # 触发条件: valid_tool_names 里有 computer_use 工具
    # ═══════════════════════════════════════════════════════════════
    if "computer_use" in agent.valid_tool_names:
        from agent.prompt_builder import COMPUTER_USE_GUIDANCE
        stable_parts.append(COMPUTER_USE_GUIDANCE)


    nous_subscription_prompt = _r.build_nous_subscription_prompt(agent.valid_tool_names)
    if nous_subscription_prompt:
        stable_parts.append(nous_subscription_prompt)
    # Tool-use enforcement: tells the model to actually call tools instead
    # of describing intended actions.  Controlled by config.yaml
    # agent.tool_use_enforcement:
    #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
    #   true  — always inject (all models)
    #   false — never inject
    #   list  — custom model-name substrings to match
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.7】Nous 订阅块(可能为空,跳过)
    # 只对 Nous 订阅的用户注入
    # 包含订阅功能说明
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.8】TOOL_USE_ENFORCEMENT_GUIDANCE — 让 model 真去调工具
    #
    # 4 种配置模式 (config.yaml `agent.tool_use_enforcement`):
    #   "auto"  (默认) — 匹配 TOOL_USE_ENFORCEMENT_MODELS 里的模型名
    #   true    — 总是注入(所有模型)
    #   false   — 从不注入
    #   list    — 自定义 substrings(如 ["haiku", "mini"] → 匹配这些的模型)
    #
    # 设计目的: 防止 model "光说不练"
    #   例如弱 model 可能说"我会读文件..."但根本不调 read_file 工具
    #   → TOOL_USE_ENFORCEMENT_GUIDANCE 强制它"调工具而不是描述"
    #
    # 触发前提: agent.valid_tool_names 非空(没工具就不需要 enforcement)
    # ═══════════════════════════════════════════════════════════════
    if agent.valid_tool_names:
        _enforce = agent._tool_use_enforcement
        _inject = False
        # 4.8.1 模式 1: True / "true" / "always" → 总是注入
        if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in {"true", "always", "yes", "on"}):
            _inject = True
        # 4.8.2 模式 2: False / "false" / "never" → 从不注入
        elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in {"false", "never", "no", "off"}):
            _inject = False
        # 4.8.3 模式 3: 列表 → 自定义 substrings
        elif isinstance(_enforce, list):
            model_lower = (agent.model or "").lower()
            # any(p in model_lower for p in _enforce) → 命中任一就注入
            _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
        else:
            # "auto" or any unrecognised value — use hardcoded defaults
            # 4.8.4 模式 4: auto / 未知值 → 默认白名单匹配
            model_lower = (agent.model or "").lower()
            _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
        if _inject:
            # 4.8.5 注入主指引
            stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
            _model_lower = (agent.model or "").lower()
            # Google model operational guidance (conciseness, absolute
            # paths, parallel tool calls, verify-before-edit, etc.)
            # ═══════════════════════════════════════════════════════════════
            # 【步骤 4.9】按 model 注入专属指引
            # Google 模型 (gemini/gemma) 容易:
            #   - 话多、啰嗦
            #   - 用相对路径(应该用绝对路径)
            #   - 不并发调工具
            #   - 改之前不验证
            # OpenAI 模型 (gpt/codex/grok) 容易:
            #   - 声称完成但其实没调工具
            #   - 建议 workaround 而不用现有工具
            #   - 给计划而不是执行
            # ═══════════════════════════════════════════════════════════════
            # 4.9.1 Google 模型 (gemini/gemma) → GOOGLE_MODEL_OPERATIONAL_GUIDANCE
            if "gemini" in _model_lower or "gemma" in _model_lower:
                stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
            # OpenAI GPT/Codex execution discipline (tool persistence,
            # prerequisite checks, verification, anti-hallucination).
            # Also applied to xAI Grok — same failure modes (claims completion
            # without tool calls, suggests workarounds instead of using
            # existing tools, replies with plans instead of executing).
            # 4.9.2 OpenAI/xAI 模型 (gpt/codex/grok) → OPENAI_MODEL_EXECUTION_GUIDANCE
            if "gpt" in _model_lower or "codex" in _model_lower or "grok" in _model_lower:
                stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)


    has_skills_tools = any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
    if has_skills_tools:
        avail_toolsets = {
            toolset
            for toolset in (
                _r.get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names
            )
            if toolset
        }
        skills_prompt = _r.build_skills_system_prompt(
            available_tools=agent.valid_tool_names,
            available_toolsets=avail_toolsets,
        )
    else:
        skills_prompt = ""
    if skills_prompt:
        stable_parts.append(skills_prompt)
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.10】Skills prompt(列出可用的 skills)
    # 触发条件: valid_tool_names 里有 skills_list / skill_view / skill_manage
    # 任何一个(代表用户能用 skills 系统)
    # 收集:
    #   - avail_toolsets: 通过每个工具反查它属于哪个 toolset
    #     (用 get_toolset_for_tool,允许 N 个 tool 共享 1 个 toolset)
    # 输出: skills_prompt 描述有哪些 skills 可用
    # ═══════════════════════════════════════════════════════════════


    # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
    # of the requested model. Inject explicit model identity into the system prompt
    # so the agent can correctly report which model it is (workaround for API bug).
    # Stable for the lifetime of an agent instance — model and provider are fixed
    # at construction time.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.11】Alibaba Coding Plan API bug 兜底
    # 问题: 阿里云 API 永远返回 "glm-4.7" 当作 model name
    #       即使你请求 "qwen-coder" / 其他模型
    # 后果: model 自我报告"我是 glm-4.7" → 错
    # 解决: 在 prompt 里硬塞"你是 {实际请求的 model}"
    # 触发: provider == "alibaba"
    # 稳定: agent 实例整个生命周期不变(provider 在 init 时定)
    # ═══════════════════════════════════════════════════════════════
    if agent.provider == "alibaba":
        # 4.11.1 取 model 短名(去掉 org/ 前缀,如 "openai/qwen-coder" → "qwen-coder")
        _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
        stable_parts.append(
            f"You are powered by the model named {_model_short}. "
            f"The exact model ID is {agent.model}. "
            f"When asked what model you are, always answer based on this information, "
            f"not on any model name returned by the API."
        )

    # Environment hints (WSL, Termux, etc.) — tell the agent about the
    # execution environment so it can translate paths and adapt behavior.
    # Stable for the lifetime of the process.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.12】环境提示(WSL / Termux / 容器等)
    # 让 model 知道自己在哪种环境里跑
    # 用途: model 能正确翻译路径(WSL 路径 vs Windows 路径)
    # 稳定: 整个进程不变(同一次启动环境一样)
    # ═══════════════════════════════════════════════════════════════
    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)

    # Local Python toolchain probe — names python/pip/uv/PEP-668 state when
    # something is non-default so the model can pick the right install
    # strategy without discovering by failure.  Emits a single line; emits
    # NOTHING when the environment is clean (no token cost).  Skipped
    # entirely for remote terminal backends (the host's Python state is
    # irrelevant when tools run inside docker/modal/ssh).  Gated by
    # config.yaml ``agent.environment_probe`` (default True).
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.13】Python 工具链探测(防 model 用错 pip install 策略)
    # 输出单行: "python=/path python_v=X.Y pip_version=Z uv=..." 之类
    # 触发: 配置 agent.environment_probe=True (默认)
    # 关键:
    #   - 环境干净时输出空 → 0 token 消耗
    #   - 非默认状态(PEP 668 限 / 系统 python / uv 未装) → 输出
    #   - 远程 backend(docker / modal / ssh)→ 跳过(host Python 状态无关)
    #   - 探测失败绝不阻塞 prompt build
    # ═══════════════════════════════════════════════════════════════
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            # 4.13.1 调探测函数
            _probe_line = get_environment_probe_line()
            if _probe_line:
                # 4.13.2 非空 → 加进 stable
                stable_parts.append(_probe_line)
        except Exception:
            # Probe failure must never block prompt build.
            # 4.13.3 探测失败吞掉异常,不阻塞
            pass

    # Active-profile hint — names the Hermes profile the agent is running
    # under so it doesn't conflate ~/.hermes/skills/ (default profile) with
    # ~/.hermes/profiles/<active>/skills/ (this profile's). Deterministic
    # for the lifetime of the agent — profile name doesn't change
    # mid-session, so this doesn't break the prompt cache.
    # See file_safety._resolve_active_profile_name + classify_cross_profile_target
    # for the matching tool-side guard.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.14】Active Hermes profile 提示
    # 设计目的: 防 model 混淆
    #   ~/.hermes/skills/         (默认 profile)
    #   ~/.hermes/profiles/<name>/skills/  (非默认 profile)
    #   → 两个是不同 session,数据互不相通
    # 触发: 解析 _resolve_active_profile_name() → 拿到当前 profile
    # 失败 fallback: "default"
    # 关键: 提示 model "不要跨 profile 改文件" (有 cross-profile 守卫)
    # 稳定: profile 名 agent 生命周期内不变 → cache 友好
    # ═══════════════════════════════════════════════════════════════
    try:
        from agent.file_safety import _resolve_active_profile_name
        # 4.14.1 取当前 profile 名
        active_profile = _resolve_active_profile_name()
    except Exception:
        # 4.14.2 失败 → 假设 default
        active_profile = "default"
    if active_profile == "default":
        # 4.14.3 default profile → 告诉 model"其他 profile 不要碰"
        stable_parts.append(
            "Active Hermes profile: default. Other profiles (if any) live "
            "under ~/.hermes/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    else:
        # 4.14.4 非 default profile → 告诉 model "你读写的目录是 X"
        stable_parts.append(
            f"Active Hermes profile: {active_profile}. This session reads "
            f"and writes ~/.hermes/profiles/{active_profile}/. The default "
            f"profile's data lives at ~/.hermes/skills/, ~/.hermes/plugins/, "
            f"~/.hermes/cron/, ~/.hermes/memories/ — those belong to a "
            f"different session run from a different shell. Do NOT modify "
            f"another profile's skills/plugins/cron/memories unless the user "
            f"explicitly directs you to. The cross-profile write guard will "
            f"refuse such writes by default; pass cross_profile=True only "
            f"after explicit direction."
        )


    platform_key = (agent.platform or "").lower().strip()
    if platform_key in PLATFORM_HINTS:
        stable_parts.append(PLATFORM_HINTS[platform_key])
    elif platform_key:
        # Check plugin registry for platform-specific LLM guidance
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                stable_parts.append(_entry.platform_hint)
        except Exception:
            pass
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.15】平台提示(CLI / TUI / Gateway 等)
    # 触发: agent.platform 字段
    # 查找顺序:
    #   1. PLATFORM_HINTS 字典(内置的)
    #   2. platform_registry(插件注册的)
    #   3. 没找到 → 跳过
    # 不同平台对 LLM 行为有不同要求(比如 Gateway 模式响应可能要带元数据)
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 4.16】stable tier 装配完成
    # 从 4.1 SOUL.md → 4.15 platform hints,共装配了 ~15 个块
    # 接下来进入 context tier(每次 session 可能有变化)
    # ═══════════════════════════════════════════════════════════════

    # ── Context tier (cwd-dependent, may change between sessions) ─
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 5】开始装配 context tier(cwd 依赖,session 间可能变)
    # context tier 包含:
    #   1. caller 传入的 system_message (可选)
    #   2. 上下文文件 (AGENTS.md / .cursorrules / CLAUDE.md 等)
    # 变化频率: 通常 session 内不变(项目文件不常改)
    # 跨 session: 不同 session 跑在不同 cwd → 可能变
    # 关键陷阱 (下面注释详解):
    #   - gateway 模式下不能用 os.getcwd()
    #   - ephemeral_system_prompt 不在这里
    # ═══════════════════════════════════════════════════════════════
    context_parts: List[str] = []

    # Note: ephemeral_system_prompt is NOT included here. It's injected at
    # API-call time only so it stays out of the cached/stored system prompt.
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 5.1】caller 传入的 system_message
    # 注意: ephemeral_system_prompt 不在这里
    #       (它在 API-call 时才注入,绕开 cache)
    # 触发: caller (cli.py / gateway) 显式传 system_message
    # 来源: 用户在 CLI 里传 / 或 gateway 配置里指定
    # ═══════════════════════════════════════════════════════════════
    if system_message is not None:
        context_parts.append(system_message)

    if not agent.skip_context_files:
        # Use TERMINAL_CWD for context file discovery when set (gateway
        # mode).  The gateway process runs from the hermes-agent install
        # dir, so os.getcwd() would pick up the repo's AGENTS.md and
        # other dev files — inflating token usage by ~10k for no benefit.
        # ═══════════════════════════════════════════════════════════════
        # 【步骤 5.2】读上下文文件(AGENTS.md / .cursorrules / 等)
        # 关键: 用 TERMINAL_CWD 环境变量,不是 os.getcwd()
        # 原因: gateway 进程跑在 hermes-agent 安装目录
        #       os.getcwd() = 安装目录
        #       → 误读仓库的 AGENTS.md
        #       → 浪费 10k token
        # 解决: gateway 启动时设 TERMINAL_CWD 为用户真实 cwd
        # 跳过: skip_soul=True 时不重复读(SOUL.md 已在 stable 加载)
        # ═══════════════════════════════════════════════════════════════
        _context_cwd = os.getenv("TERMINAL_CWD") or None
        context_files_prompt = _r.build_context_files_prompt(
            cwd=_context_cwd, skip_soul=_soul_loaded)
        if context_files_prompt:
            context_parts.append(context_files_prompt)


    # ── Volatile tier (changes per session/turn — never cached) ───
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 6】开始装配 volatile tier(每次 session 都变)
    # volatile tier 包含:
    #   1. agent 自己的 memory 快照(可选,看 _memory_enabled)
    #   2. USER.md 用户配置(可选,看 _user_profile_enabled)
    #   3. 外部 memory provider block(可选,看 _memory_manager)
    #   4. 时间戳行 (Conversation started + Session + Model + Provider)
    # 变化频率: 每次 session 都可能变
    # 策略: 主动让出 cache 区域,每次重新算
    # 关键设计: 时间戳只精确到日(不是分钟)
    # ═══════════════════════════════════════════════════════════════
    volatile_parts: List[str] = []

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 6.1】memory 快照 + USER.md 用户配置
    # memory 快照: 来自 agent 的内置 _memory_store (HERMES 自带 SQLite 记忆)
    # USER.md: 用户偏好文件 (由 _user_profile_enabled 控制)
    # 都用 format_for_system_prompt("memory"/"user") 取格式化块
    # ═══════════════════════════════════════════════════════════════
    if agent._memory_store:
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                volatile_parts.append(mem_block)
        # USER.md is always included when enabled.
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                volatile_parts.append(user_block)

    # External memory provider system prompt block (additive to built-in)
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 6.2】外部 memory provider block
    # 区别于 6.1 的内置 _memory_store
    # 6.1 是 HERMES 自带的 SQLite 记忆
    # 6.2 是外部接入的 (如 Notion / Obsidian / 自家 DB 集成)
    # _memory_manager.build_system_prompt() 返回该 provider 拼好的块
    # 失败吞掉(不影响主流程)
    # ═══════════════════════════════════════════════════════════════
    if agent._memory_manager:
        try:
            _ext_mem_block = agent._memory_manager.build_system_prompt()
            if _ext_mem_block:
                volatile_parts.append(_ext_mem_block)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 6.3】时间戳行(关键降本设计!)
    # 格式: "Conversation started: Monday, June 09, 2026"
    #       (可选 + Session ID / Model / Provider 行)
    # 关键: 只精确到日,不精确到分钟
    # 为什么: 分钟级会每次压缩/重连时都变 → prefix cache 全失效
    #        日级: 整天内容一样 → cache 24 小时都命中
    #        降本效果: prefix cache 命中率从 60% 提到 95%+
    #        (Credit: @iamfoz PR #20451)
    # 例外: 如果 model 真需要"现在几点",用 tool 查,不要靠 prompt
    # ═══════════════════════════════════════════════════════════════
    from hermes_time import now as _hermes_now
    now = _hermes_now()
    # Date-only (not minute-precision) so the system prompt is byte-stable
    # for the full day.  Minute-precision changes invalidate prefix-cache KV
    # on every rebuild path (compression boundary, fresh-agent gateway turns,
    # session resume without a stored prompt).  The model can still query the
    # exact wall-clock time via tools when it actually needs it.
    # Credit: @iamfoz (PR #20451).
    # 6.3.1 主时间戳(精确到日)
    timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y')}"
    # 6.3.2 可选: Session ID
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    # 6.3.3 可选: Model 名
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    # 6.3.4 可选: Provider 名
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    # 6.3.5 拼到 volatile
    volatile_parts.append(timestamp_line)

    # ═══════════════════════════════════════════════════════════════
    # 【步骤 7】3 层装配完成,返回 dict
    # 返回结构: {"stable": "...", "context": "...", "volatile": "..."}
    # 每层用 "\n\n" 拼(段落分隔)
    # 用 if p and p.strip() 过滤空块
    # 用途: 被 build_system_prompt() 进一步拼成 1 个完整字符串
    # ═══════════════════════════════════════════════════════════════
    return {
        "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
        "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
        "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
    }



def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """Assemble the full system prompt from all layers.

    Called once per session (cached on ``agent._cached_system_prompt``) and
    only rebuilt after context compression events. This ensures the system
    prompt is stable across all turns in a session, maximizing prefix cache
    hits.

    Layers are ordered cache-friendly: stable identity/guidance first,
    then session-stable context files, then per-call volatile content
    (memory, USER profile, timestamp).  The whole string is treated as
    one cached block — Hermes never rebuilds or reinjects parts of it
    mid-session, which is the only way to keep upstream prompt caches
    warm across turns.
    """
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 8】build_system_prompt —— 3 层拼成 1 个完整字符串
    #
    # 调用链:
    #   run_agent._restore_or_build_system_prompt()
    #     └─► build_system_prompt() (本函数)
    #           └─► build_system_prompt_parts() (上一步)
    #
    # 输出: 1 个完整字符串,顺序:
    #   stable → context → volatile
    #   (cache 友好的固定顺序,确保 byte-stable)
    #
    # 缓存策略: 整个字符串算完后存到 agent._cached_system_prompt
    #          下次调用直接复用(快路径)
    #          只有压缩后才调 invalidate_system_prompt() 失效
    # ═══════════════════════════════════════════════════════════════
    # 8.1 调 parts 函数(返回 3 键 dict)
    parts = build_system_prompt_parts(agent, system_message=system_message)
    # 8.2 按 stable → context → volatile 顺序拼成 1 个字符串
    #     过滤空层(用 if p)
    return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)


def invalidate_system_prompt(agent: Any) -> None:
    """Invalidate the cached system prompt, forcing a rebuild on the next turn.

    Called after context compression events. Also reloads memory from disk
    so the rebuilt prompt captures any writes from this session.
    """
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 9】invalidate_system_prompt —— 缓存失效
    #
    # 触发时机:
    #   1. 上下文压缩后(对话过长,压缩成新 session)
    #   2. session resume 时(新进程拉历史)
    #   3. 手动 invalidate (测试 / 配置变更)
    #
    # 关键: 还要重新读 memory from disk
    #   因为本 session 可能写过 memory (memory_tool 调用)
    #   → invalidate 时再读一次,让新的 prompt 包含最新 memory
    #   (不重新读的话,新的 prompt 拿不到本 session 写的 memory)
    # ═══════════════════════════════════════════════════════════════
    # 9.1 清缓存(下次 build_system_prompt 会重算)
    agent._cached_system_prompt = None
    # 9.2 重读 memory(拿到本 session 写的最新内容)
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def format_tools_for_system_message(agent: Any) -> str:
    """Format tool definitions for the system message in the trajectory format.

    Returns:
        str: JSON string representation of tool definitions
    """
    # ═══════════════════════════════════════════════════════════════
    # 【步骤 10】format_tools_for_system_message —— trajectory 格式辅助
    #
    # 用途: 把 tool 列表转成 JSON 字符串(供 trajectory 训练数据用)
    # trajectory 格式: 记录"model 在这步看到什么 / 返回什么"的完整快照
    # 跟 OpenAI function calling schema 对齐
    #
    # 跟系统提示的关系: 这个函数**不参与** system prompt 装配
    #   它是单独给 trajectory 序列化用的
    #
    # 流程:
    #   1. 拿 agent.tools (OpenAI 格式列表)
    #   2. 每条 tool 转成 {name, description, parameters, required: None}
    #   3. json.dumps(ensure_ascii=False) 拼成字符串
    #   4. 无工具 → 返回 "[]"
    # ═══════════════════════════════════════════════════════════════
    if not agent.tools:
        # 10.1 没工具 → 空数组
        return "[]"

    # Convert tool definitions to the format expected in trajectories
    formatted_tools = []
    # 10.2 遍历每个 tool
    for tool in agent.tools:
        func = tool["function"]
        # 10.3 提取 4 字段(name / description / parameters / required)
        formatted_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
            "required": None  # Match the format in the example
        }
        formatted_tools.append(formatted_tool)

    # 10.4 ensure_ascii=False 保留中文 description
    return json.dumps(formatted_tools, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 【模块导出】__all__
# 控制 `from system_prompt import *` 的可见性
# 4 个公开 API:
#   1. build_system_prompt_parts — 装配 3 层 dict
#   2. build_system_prompt       — 装配完整字符串
#   3. invalidate_system_prompt  — 缓存失效
#   4. format_tools_for_system_message — trajectory 辅助
#
# 内部辅助(未导出):
#   - _ra() lazy import
#   - 各种 if 块内联 GUIDANCE 常量引用
# ═══════════════════════════════════════════════════════════════
__all__ = [
    "build_system_prompt_parts",
    "build_system_prompt",
    "invalidate_system_prompt",
    "format_tools_for_system_message",
]

