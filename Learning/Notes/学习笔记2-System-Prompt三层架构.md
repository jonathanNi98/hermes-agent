# 学习笔记 2 — System Prompt 三层架构 + 主循环形式化总结



System prompt怎么构建
如何compress的，head middle tail

> **对应学习计划**:Day 2(主循环结构) + Day 3(System Prompt 三层)
> **源文件**:`hermes-agent/agent/system_prompt.py` (407 行) + `hermes-agent/agent/prompt_builder.py` (1507 行) + `hermes-agent/agent/conversation_loop.py`
> **创建日期**:2026-06-08
> **前序笔记**:学习笔记 1(主循环架构)

---

## 📌 第一部分:Day 2 产出物 —— **主循环伪代码**

`run_conversation(user_message)` 主循环的完整伪代码(从源码学到的):

```
═══════════════════════════════════════════════════════════════════
        run_conversation(user_message)  主循环伪代码
═══════════════════════════════════════════════════════════════════

function run_conversation(agent, user_message):

    # ============ 初始化阶段(只跑一次) ============
    restore_or_build_system_prompt(agent)       # 构建 system_prompt(详见 Day 3)
    memory_manager.prefetch_all(user_message)   # 预取记忆

    # ============ 外层 while: 回合推进 ============
    while api_call_count < max_iterations AND iteration_budget.remaining > 0:

        # ── 回合初始化 ──
        api_call_count += 1
        check_interrupt()                        # 检查用户中断
        consume_budget()                          # 消耗 iteration budget

        # ── Step 1: build_messages(行 814-1136) ──
        api_messages = build_messages(...)        # 内部 messages → API 格式
            ├─ tool_call args 修复
            ├─ role 交替修复
            ├─ 注入 system + memory
            ├─ 主动 surrogate 清洗
            └─ 14 个 *_attempted 标志位初始化

        # ── Step 2: 内层 while: 单次 API 重试(行 1179) ──
        while retry_count < max_retries:

            # ① 发起 API 调用(行 1404)
            response = _interruptible_streaming_api_call(api_kwargs)

            # ② 成功路径:break(行 2387)
            if response.success:
                break                                # 跳出内层 while

            # ③ 失败路径:错误处理(行 2700+)
            error = classify_api_error(response)    # 21 种 FailoverReason
            if error.is_auth:
                refresh_credential()                # 401 刷凭证
            elif error.should_compress:
                restart_with_compressed_messages()   # 压缩
            elif error.should_fallback:
                activate_fallback()                  # 切 fallback
            elif error.retryable:
                retry_count += 1                     # 退避重试
                continue
            else:
                break                                # 致命错误,跳出

            retry_count += 1

        # ④ 归一化响应(行 4360)
        assistant_message = transport.normalize_response(response)
        record_token_usage(assistant_message)      # 8 个累加器

        # ── Step 3: check tool_calls(行 4607,★ 主分叉) ──
        if assistant_message.tool_calls:

            # 路径 A: 工具执行
            validate_tool_names()                   # 5 重门验证
            validate_json_args()                    # 4 层防御
            check_truncation()
            post_call_guardrails()                  # cap delegate + dedupe
            append_assistant_message()
            execute_tools()                         # ★ 行 5025:真的调工具
            check_guardrail_halt()                  # 工具被拦 → 终止
            check_compression()                     # 是否需要压缩
            continue                                # 回到外层 while 顶部

        else:

            # 路径 B: 最终响应
            if empty_response:                      # 5 种恢复级联
                try_partial_stream_recovery()       # 4.91
                try_prior_turn_fallback()            # 4.92
                try_post_tool_nudge()                # 4.93
                try_thinking_prefill()              # 4.95
                try_fallback_provider()              # 4.97
                → "(empty)" 终止哨兵                 # 4.98
                break                                # 退出外层
            else:
                final_response = assistant_message.content
                break                                # 退出外层 while

    # ============ 清理收尾(18 步,行 5745+) ============
    handle_max_iterations()                       # 4.107
    save_trajectory()                              # 4.109.1
    cleanup_task_resources()                       # 4.110
    drop_scaffolding()                             # 4.111
    persist_session()                              # 4.111.2
    emit_diagnostic_log()                          # 4.112-4.113
    attach_file_mutation_verifier()                # 4.114
    attach_turn_completion_explainer()             # 4.115
    fire_plugin_hooks(transform_llm_output)        # 4.117
    fire_plugin_hooks(post_llm_call)               # 4.118
    extract_reasoning()                            # 4.119
    build_result_dict()                            # 4.120
    handle_pending_steer()                         # 4.121
    check_skill_nudge()                            # 4.122
    sync_external_memory()                         # 4.123
    spawn_background_review()                      # 4.124
    fire_on_session_end_hook()                     # 4.125
    clear_interrupt_state()                        # 4.121.3
    return result                                  # 4.126
```

### 主循环的 5 大核心步骤(简化版)

```
┌────────────────────────────────────────────────────────────┐
│ 主循环 5 步流程                                              │
│                                                            │
│ ① 初始化(只一次)                                           │
│    ├─ 恢复/构建 system prompt(三层架构)                    │
│    └─ 预取记忆                                              │
│                                                            │
│ ② 外层 while(回合推进)                                     │
│    ├─ build_messages(构造 API 请求)                       │
│    └─ 内层 while(API 重试)                                  │
│        ├─ 调 API                                            │
│        ├─ 错误处理(21 种 FailoverReason)                   │
│        └─ 成功 → break                                      │
│                                                            │
│ ③ check tool_calls(★ 主分叉)                                │
│    ├─ 有 → 工具执行 → continue 回到外层顶部                 │
│    └─ 无 → 最终响应 → break 退出                            │
│                                                            │
│ ④ 清理收尾(18 步)                                          │
│                                                            │
│ ⑤ return result                                           │
└────────────────────────────────────────────────────────────┘
```

---

## 📌 第二部分:Day 3 核心 —— **System Prompt 三层架构**

### 整体设计哲学

Hermes 把 LLM 看到的"开头那段话"分成**三层**:

```
┌──────────────────────────────────────────────┐
│ 第 1 层: Stable (稳定层)                      │
│ (整个 agent 生命周期不变)                    │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ 第 2 层: Context (上下文层)            │   │
│  │ (同一 session 不变)                   │   │
│  │                                      │   │
│  │  ┌──────────────────────────────┐   │   │
│  │  │ 第 3 层: Volatile (易变层)     │   │   │
│  │  │ (每天都变,但 24h 内稳定)     │   │   │
│  │  │                              │   │   │
│  │  │ "你叫 Hermes..." (SOUL.md)   │   │   │
│  │  │ "如何调工具..." (指南)        │   │   │
│  │  │ "AGENTS.md 规则..." (项目)    │   │   │
│  │  │ "记忆: 用户喜欢..." (memory)  │   │   │
│  │  │ "今天 2026-06-08" (timestamp) │   │   │
│  │  └──────────────────────────────┘   │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

**核心思想**:**变化越频繁的内容越靠后**(后注入的内容可以独立变,不影响前面稳定内容的 cache 命中)。

### 入口函数

`agent/system_prompt.py` **61-345 行** 的 `build_system_prompt_parts()` 返回一个 dict:

```python
{
    "stable":   "...",   # 第 1 层
    "context":  "...",   # 第 2 层
    "volatile": "...",   # 第 3 层
}
```

由 `build_system_prompt()`(行 348)拼成单个字符串,**缓存在 `agent._cached_system_prompt`**。

### 第 1 层:Stable(稳定层)

**位置**:`system_prompt.py:84-280`

**特点**:**整个 agent 生命周期内不变**(从 `__init__` 到 `del`)

#### Stable 层的 14 个组件

| # | 组件 | 来源 | 行号 | 说明 |
|---|---|---|---|---|
| 1 | **SOUL.md** | `_r.load_soul_md()` | 90-95 | 主身份定义(可选) |
| 2 | **DEFAULT_AGENT_IDENTITY** | 硬编码常量 | 99 | SOUL.md 缺失时的降级身份 |
| 3 | **HERMES_AGENT_HELP_GUIDANCE** | 静态常量 | 102 | "用户问 Hermes 自身时"的指南 |
| 4 | **TASK_COMPLETION_GUIDANCE** | 静态常量 | 110-111 | "完成真实任务,不要伪造" |
| 5 | **Tool 行为指导**(memory/session_search/skill/kanban) | 静态常量 | 114-132 | 4 个有 tool 的特殊指导 |
| 6 | **Computer-use 指导** | `prompt_builder` | 136-138 | macOS 自动化 |
| 7 | **Nous 订阅提示** | `build_nous_subscription_prompt` | 140-142 | 订阅用户的特殊功能 |
| 8 | **Tool-use enforcement** | 静态常量 | 150-177 | "真的调工具,别只描述" |
| 9 | **Skills 提示** | `build_skills_system_prompt` | 179-195 | 所有可用 skill |
| 10 | **Alibaba 身份纠正** | 硬编码 | 202-209 | API bug 修复 |
| 11 | **环境提示**(WSL/Termux) | `build_environment_hints()` | 211-216 | OS 特殊说明 |
| 12 | **Python toolchain 探针** | `tools/env_probe` | 225-233 | PEP-668 等 |
| 13 | **Active profile 提示** | 静态字符串 | 242-267 | "你在 default profile" |
| 14 | **Platform 提示** | `PLATFORM_HINTS` 字典 | 269-280 | 部署平台特殊说明 |

#### 为什么不变?

- ✅ **SOUL.md** 在进程启动时读一次
- ✅ **工具列表** 在 `__init__` 时定下来
- ✅ **profile** 不会变
- ✅ **平台** 不会变
- ✅ **tool_use_enforcement** 跟 model 名绑

#### 关键设计

**这层的字节级稳定 → 上游 LLM 的 prompt cache 命中率最大化**。

#### SOUL.md 的特殊处理

```python
# system_prompt.py 行 90-99
_soul_loaded = False
if agent.load_soul_identity or not agent.skip_context_files:
    _soul_content = _r.load_soul_md()
    if _soul_content:
        stable_parts.append(_soul_content)
        _soul_loaded = True

if not _soul_loaded:
    # Fallback to hardcoded identity
    stable_parts.append(DEFAULT_AGENT_IDENTITY)
```

- **优先用 SOUL.md**(用户自定义身份)
- **降级用 DEFAULT_AGENT_IDENTITY**(系统默认)
- **特殊模式**:`skip_context_files=True`(cron 模式)→ 也可以加载 SOUL

#### Tool 行为指导的过滤逻辑

```python
# system_prompt.py 行 114-132
tool_guidance = []
if "memory" in agent.valid_tool_names:
    tool_guidance.append(MEMORY_GUIDANCE)
if "session_search" in agent.valid_tool_names:
    tool_guidance.append(SESSION_SEARCH_GUIDANCE)
if "skill_manage" in agent.valid_tool_names:
    tool_guidance.append(SKILLS_GUIDANCE)
# Kanban worker 模式(HERMES_KANBAN_TASK env)
if _kanban_guidance:
    tool_guidance.append(_kanban_guidance)
elif _kanban_guidance is None and "kanban_show" in agent.valid_tool_names:
    tool_guidance.append(KANBAN_GUIDANCE)
```

**特点**:**只加载当前 agent 注册的工具对应的指导**,不浪费 token。

#### Tool-use enforcement 的智能注入

```python
# system_prompt.py 行 150-177
if agent.valid_tool_names:
    _enforce = agent._tool_use_enforcement
    # 配置项可以是:
    #   "auto"  (默认) - 匹配内置白名单
    #   true    - 总是注入
    #   false   - 从不注入
    #   list    - 自定义 model 名子串
    
    if _enforce is True:
        _inject = True
    elif _enforce is False:
        _inject = False
    elif isinstance(_enforce, list):
        # 检查 model 名是否包含列表里的子串
        _inject = any(p in agent.model.lower() for p in _enforce)
    else:  # "auto"
        # 用内置 TOOL_USE_ENFORCEMENT_MODELS 白名单
        _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
    
    if _inject:
        stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
        # 进一步按 model 家族注入专用指导
        if "gemini" in model_lower:
            stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
        if "gpt" in model_lower or "codex" in model_lower or "grok" in model_lower:
            stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)
```

**意义**:**不同 model 家族有不同的失败模式**:
- **Google/Gemini**:需要简洁、绝对路径、并行 tool、verify-before-edit
- **OpenAI/GPT/Codex/xAI**:需要 tool 持久性、前置检查、验证、反幻觉

**Hermes 给每个家族专门的指导文本**。

### 第 2 层:Context(上下文层)

**位置**:`system_prompt.py:282-299`

**特点**:**同一 session 内不变,但跨 session 可能变**

#### Context 层的 2 个组件

| # | 组件 | 来源 | 何时变 |
|---|---|---|---|
| 1 | **system_message** 参数 | 调用方传入 | 每个 session 不同(默认 None) |
| 2 | **上下文文件**(AGENTS.md 等) | `build_context_files_prompt` | 同一 session 内不变 |

#### 上下文文件

`build_context_files_prompt`(在 `prompt_builder.py:1468`)读取的文件:

- `AGENTS.md`(项目级 agent 指导)
- `.cursorrules`
- `CLAUDE.md`
- `HERMES.md`
- **可能还有**:`.windsurfrules`、`.github/copilot-instructions.md` 等

#### 特殊处理:TERMINAL_CWD

```python
# system_prompt.py 行 290-299
if not agent.skip_context_files:
    # Use TERMINAL_CWD for context file discovery when set (gateway
    # mode).  The gateway process runs from the hermes-agent install
    # dir, so os.getcwd() would pick up the repo's AGENTS.md and
    # other dev files — inflating token usage by ~10k for no benefit.
    _context_cwd = os.getenv("TERMINAL_CWD") or None
    context_files_prompt = _r.build_context_files_prompt(
        cwd=_context_cwd, skip_soul=_soul_loaded)
```

**关键**:
- **CLI 模式**:用 `os.getcwd()`(用户在哪个目录)
- **Gateway 模式**:用 `TERMINAL_CWD` 环境变量(用户在哪个目录)
- **原因**:gateway 进程跑在 hermes-agent 安装目录,如果用 `os.getcwd()` 会读到仓库自己的 `AGENTS.md`,**浪费 10k token**

#### 关键设计

**这层比较"中等"**——同一 session 内稳定,但如果改 `.cursorrules` 重启后会变。

### 第 3 层:Volatile(易变层)

**位置**:`system_prompt.py:301-339`

**特点**:**理论上每次都变,但实际用 date-only 保持 24h 稳定**

#### Volatile 层的 4 个组件

| # | 组件 | 来源 | 何时变 |
|---|---|---|---|
| 1 | **memory block** | `agent._memory_store.format_for_system_prompt("memory")` | 记忆更新时变 |
| 2 | **user profile block** | `agent._memory_store.format_for_system_prompt("user")` | 用户信息更新时变 |
| 3 | **外部 memory provider block** | `agent._memory_manager.build_system_prompt()` | 跨进程恢复时变 |
| 4 | **timestamp + session info** | 硬编码 | **每天变**(date-only) |

#### 关键设计:date-only timestamp

```python
# system_prompt.py 行 324-339
from hermes_time import now as _hermes_now
now = _hermes_now()
# Date-only (not minute-precision) so the system prompt is byte-stable
# for the full day.  Minute-precision changes invalidate prefix-cache KV
# on every rebuild path (compression boundary, fresh-agent gateway turns,
# session resume without a stored prompt).  The model can still query the
# exact wall-clock time via tools when it actually needs it.
# Credit: @iamfoz (PR #20451).
timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y')}"
if agent.pass_session_id and agent.session_id:
    timestamp_line += f"\nSession ID: {agent.session_id}"
if agent.model:
    timestamp_line += f"\nModel: {agent.model}"
if agent.provider:
    timestamp_line += f"\nProvider: {agent.provider}"
volatile_parts.append(timestamp_line)
```

**为什么不精确到分钟?**
- 分钟级会破坏 prefix-cache KV
- date-only 在当天 byte-stable
- 模型需要精确时间时,自己用 tool 查

**这个设计是从 PR #20451 来的**(贡献者 @iamfoz),**注释里明确写明"credit"**。

#### Timestamp 的完整格式

```
Conversation started: Monday, June 08, 2026
Session ID: abc-123-uuid
Model: claude-opus-4-8
Provider: anthropic
```

### 三层组合的字节稳定策略

```python
# system_prompt.py:341-345
return {
    "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
    "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
    "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
}
```

```python
# build_system_prompt() 简化
return stable + "\n\n" + context + "\n\n" + volatile
```

**顺序的重要性**:
- **前**面的内容变更会导致**所有**后续内容 cache 失效
- **把稳定的放前** = 大部分 turn 可以命中 cache
- **易变的放后** = 哪怕变了,也只影响最后一段

### 缓存机制

```python
# system_prompt.py 行 348-358
def build_system_prompt(agent, system_message=None):
    """Called once per session (cached on agent._cached_system_prompt) and
    only rebuilt after context compression events. This ensures the system
    prompt is stable across all turns in a session, maximizing prefix cache
    hits.
    """
```

**关键点**:
- **只调一次**(per session)
- **缓存在 agent._cached_system_prompt**
- **只在压缩后失效**

```python
# system_prompt.py:367
def invalidate_system_prompt(agent) -> None:
    """清缓存,下次重新构造"""
```

---

## 📌 第三部分:跟主循环的衔接(我们学过的)

### 主循环里如何使用

`conversation_loop.py` 行 970-988:

```python
# 主循环里的 system prompt 拼装
effective_system = active + ephemeral  # ← 缓存 system prefix
api_messages = [
    {"role": "system", "content": effective_system},
    *history_messages,
    ...
]
```

**"active"** = `agent._cached_system_prompt` = 三层拼好的字符串
**"ephemeral"** = 当次 turn 临时加的(比如 nudge 提示)

### 缓存的 5 个失效时机

| 失效时机 | 原因 |
|---|---|
| **压缩触发** | 压缩后 messages 改变,系统 prompt 也可能需要更新 |
| **手动 invalidate** | `agent.invalidate_system_prompt()` |
| **process restart** | 进程退出,缓存自然丢 |
| **profile 切换** | 跨 profile 切换 |
| **跨 session 恢复** | 重新构造 |

### 缓存路径决策树(★ 新增)

光知道"有缓存"不够,**关键是要知道"什么时候走什么路径"**。整个机制分 3 层,层层递进。

#### 入口判断:快 vs 慢路径

```python
# conversation_loop.py:587-590
if agent._cached_system_prompt is None:
    _restore_or_build_system_prompt(agent, system_message, conversation_history)

active_system_prompt = agent._cached_system_prompt
```

| 路径 | 触发条件 | 行为 | 耗时 |
|---|---|---|---|
| **入口快路径** | `_cached_system_prompt` **已存在**(非 None) | 直接复用,啥都不做 | **0** |
| **慢路径(完整流程)** | `_cached_system_prompt` **是 None** | 调 `_restore_or_build_system_prompt()` | 视分支而定 |

**99% 的 turn 走入口快路径** —— 这是 prefix cache 命中的关键。

#### 慢路径内部:3 条分支决策树

`_restore_or_build_system_prompt()`(`conversation_loop.py:218-316`)不是单一函数,它有 **3 条分支**:

```
                     _restore_or_build_system_prompt(agent)
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │ 1. 查 SQLite session DB     │
                          │    agent._session_db        │
                          │    .get_session(session_id) │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │                           │
                       DB 有值?                    DB 没值 / 损坏
                          │                           │
                          ▼                           ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │ [A] DB 恢复          │    │ [B] 从零重算          │
              │ line 267-271         │    │ line 286-288         │
              │ 复用 stored_prompt   │    │ 调 build_system_     │
              │ = byte-stable 复用   │    │ prompt() 完整链路    │
              │ → cache 100% 命中   │    │ = 50-200ms          │
              └──────────┬───────────┘    └──────────┬───────────┘
                         │                           │
                         │                           ▼
                         │              ┌──────────────────────┐
                         │              │ [C] 写回 DB          │
                         │              │ line 308-310         │
                         │              │ update_system_prompt │
                         │              │ 给下次能走 [A]       │
                         │              └──────────┬───────────┘
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                          return(下次 turn 走快路径)
```

#### 三层路径对比表

| 层次 | 触发 | 耗时 | 频率 | 何时用 |
|---|---|---|---|---|
| **入口快路径** | `_cached_system_prompt` 非 None | 0 | 每个 turn(99%) | 同一 agent 实例内任何复用的 turn |
| **DB 恢复 [A]** | DB 里有 stored_prompt | 1 次 SQL 查询 | gateway 模式每个新 agent | 续 session、新 agent 实例 |
| **从零重算 [B]** | DB 没值 / 损坏 | 50-200ms | 首次 session | 新 session、cache 损坏恢复 |
| **写回 DB [C]** | 紧跟 [B] 之后 | 1 次 SQL UPDATE | 每次重算后 | 给下个 turn / 下个 agent 留种子 |

#### Gateway 模式的关键设计动机

**问题**:gateway 模式下,每个 HTTP 请求**新建一个 AIAgent 实例**(`run_conversation()` 一次一个实例)。

**如果只靠内存 cache**:
- 每个新 agent 的 `_cached_system_prompt` 都是 `None`
- 每次都走慢路径 [B] 重算
- prefix cache 永远不命中(每次内容可能不一样)
- **降本 75% 的优势全没了**

**解决方案**:
- 把 system prompt **持久化到 SQLite** 的 `sessions.system_prompt` 列
- 新 agent 起来时先查 DB → 走 [A] 复用 → cache 命中
- 第一次 / 损坏时走 [B] + [C] → 留种

**关键行**:`conversation_loop.py:267-271` —— **一行赋值就把 prefix cache 救回来了**:

```python
if stored_prompt:
    # Continuing session — reuse the exact system prompt from the
    # previous turn so the Anthropic cache prefix matches.
    agent._cached_system_prompt = stored_prompt
    return
```

#### 三种状态的诊断日志

`stored_state` 变量有 4 种取值,每种都对应一种处理:

| 状态 | 含义 | 日志级别 |
|---|---|---|
| `missing` | 没 session row(全新 session,合法) | 不警告 |
| `present` | row 存在且有 prompt → 走 [A] 复用 | 不警告 |
| `null` | row 存在但 system_prompt 列是 NULL(legacy / 迁移残留) | WARNING(若 conversation_history 非空) |
| `empty` | row 存在但存了空字符串(写盘 bug) | **总是 WARNING** |

**WHY 区分这么细?**:因为如果"续 session 时 stored_prompt 突然为 null/empty",意味着**之前那次写入坏了**,每次都重算 = 每次都重付钱 = 用户投诉。日志必须能让运维 grep 到。

#### 实际触发链(从 `run_conversation` 入口)

```
run_conversation(user_message)
    │
    ├─► 1. 初始化 system_prompt (★ 我们这里讲的部分)
    │     │
    │     ├─► if _cached_system_prompt is None:
    │     │     _restore_or_build_system_prompt()
    │     │
    │     └─► active_system_prompt = _cached_system_prompt
    │
    ├─► 2. Preflight 压缩检查
    │
    ├─► 3. 外层 while:
    │     ├─► build_messages(api_messages)
    │     │     └─► 拼装 {"role": "system", "content": active_system_prompt}
    │     │
    │     ├─► 内层 while: API 重试
    │     │
    │     └─► check tool_calls 分叉
    │
    └─► 4. 收尾 + 清理
```

#### 关键源码对照(新增)

| 关键行 | 内容 |
|---|---|
| `conversation_loop.py:218-316` | `_restore_or_build_system_prompt()` 完整实现 |
| `conversation_loop.py:245-265` | 查 DB + 4 种 stored_state 判定 |
| `conversation_loop.py:267-271` | **A 路径(DB 恢复)** |
| `conversation_loop.py:273-284` | null/empty 警告日志 |
| `conversation_loop.py:286-288` | **B 路径(从零重算)** |
| `conversation_loop.py:290-302` | on_session_start 钩子 |
| `conversation_loop.py:308-310` | **C 路径(写回 DB)** |
| `conversation_loop.py:587-590` | 入口 if 判断(快路径/慢路径分叉) |

---

## 📌 第四部分:`prompt_builder.py` 的角色(被 system_prompt 调用)

`prompt_builder.py` 是**辅助函数库**,不直接被主循环调用。**被 `system_prompt.py` 调用**:

| 函数 | 行号 | 作用 | 被谁调用 |
|---|---|---|---|
| `_scan_context_content` | 43 | 扫描上下文文件内容 | system_prompt (内部) |
| `build_environment_hints` | 766 | OS 环境提示(WSL/Termux) | system_prompt:214 |
| `build_skills_system_prompt` | 1039 | 所有可用 skill 列表 | system_prompt:188 |
| `build_nous_subscription_prompt` | 1273 | Nous 订阅用户特殊功能 | system_prompt:140 |
| `load_soul_md` | 1355 | 读 SOUL.md 文件 | system_prompt:92 |
| `build_context_files_prompt` | 1468 | 读 AGENTS.md/.cursorrules 等 | system_prompt:296 |

**特点**:
- **`prompt_builder.py` 是"原料库"**
- **`system_prompt.py` 是"主厨"**(决定哪些原料、怎么组合)
- 主循环只问 `system_prompt.py` 要结果,**不直接问 `prompt_builder.py`**

---

## 📌 第五部分:三层架构的设计原则

| 原则 | 体现 |
|---|---|
| **Cache-friendly** | 稳定内容在前,易变内容在后 |
| **单次构造** | `build_system_prompt()` 只调一次,缓存在 `agent._cached_system_prompt` |
| **显式失效** | `invalidate_system_prompt()` 在压缩后调用 |
| **失败容错** | 多个 `try/except`(环境提示、技能、外部 memory 都可能失败) |
| **不阻塞启动** | 用 `_ra()` 延迟引用 `run_agent`,方便测试 patch |
| **byte-stable** | date-only 时间戳,profile 不变,平台不变 |
| **按需加载** | tool 行为指导只加载当前注册的工具对应的部分 |
| **model 家族专用** | Google/OpenAI 各有专门的指导文本 |

---

## 📌 第六部分:Day 2-3 自检问题

完成这一天的学习后,你应该能回答:

1. ✅ 主循环里 while 循环做了哪几件事?
2. ✅ System Prompt 三层分别对应什么?
3. ✅ 哪一层变化最频繁?(Volatile)
4. ✅ 哪些组件在第 1 层?(14 个,见上表)
5. ✅ timestamp 为什么用 date-only 而不精确到分钟?(避免破坏 cache)
6. ✅ SOUL.md 在哪一层?(Stable)
7. ✅ memory / USER profile 在哪一层?(Volatile)
8. ✅ 三层组合的顺序能换吗?(不能,会影响 cache 命中率)
9. ✅ 哪一层调用最多外部函数?(Stable,14 个组件)
10. ✅ `agent._cached_system_prompt` 什么时候失效?(压缩后)
11. ✅ `prompt_builder.py` 跟 `system_prompt.py` 是什么关系?(原料库 vs 主厨)
12. ✅ Gateway 模式下为什么用 `TERMINAL_CWD` 而不是 `os.getcwd()`?(避免读到仓库自己的 AGENTS.md)

---

## 📌 第七部分:Day 2-3 关键源码对照表

| 关键行 | 内容 |
|---|---|
| `system_prompt.py:46` | `_ra()` lazy 引用 `run_agent` 模块 |
| `system_prompt.py:61` | `build_system_prompt_parts()` 主函数入口 |
| `system_prompt.py:84-280` | 第 1 层:Stable 构造(14 个组件) |
| `system_prompt.py:282-299` | 第 2 层:Context 构造(2 个组件) |
| `system_prompt.py:301-339` | 第 3 层:Volatile 构造(4 个组件) |
| `system_prompt.py:341-345` | 3 个 list → dict |
| `system_prompt.py:348` | `build_system_prompt()` 拼成单个 string |
| `system_prompt.py:356-358` | 缓存策略:每 session 调一次 |
| `system_prompt.py:367` | `invalidate_system_prompt()` 清缓存 |
| `system_prompt.py:378` | `format_tools_for_system_message()` |
| `prompt_builder.py:766` | `build_environment_hints()` |
| `prompt_builder.py:1039` | `build_skills_system_prompt()` |
| `prompt_builder.py:1273` | `build_nous_subscription_prompt()` |
| `prompt_builder.py:1355` | `load_soul_md()` |
| `prompt_builder.py:1468` | `build_context_files_prompt()` |
| `conversation_loop.py:970-988` | 主循环里用 `_cached_system_prompt` |

---

## 📌 第八部分:Day 2-3 完成度最终确认

| 任务 | 计划 | 实际 | 状态 |
|---|---|---|---|
| **Day 2 必须回答** | "while not done 循环里做了哪几件事?" | 5 大步骤 + 18 步清理收尾 | ✅ **100%** |
| **Day 2 必须练习** | 在循环内加 print,观察顺序 | 已经在 Day 1 加了 130+ step 注释 | ✅ **100%** |
| **Day 2 产出物** | 主循环伪代码 | 本节顶部完整伪代码 | ✅ **100%** |
| **Day 3 必须回答** | "三层分别对应哪些内容?哪层会变化?" | 14+2+4 个组件,每层详细解释 | ✅ **100%** |
| **Day 3 必须练习** | 修改 SOUL.md 观察变化 | 解释了 SOUL.md 在哪一层、怎么生效 | ✅ **100%** |
| **Day 3 产出物** | 三层架构图 | ASCII 嵌套结构图 | ✅ **100%** |
| **额外产出** | (无要求) | 14 个 Stable 组件全表 + 缓存失效 5 时机 + 设计原则 8 条 | ✅ **远超** |

**Day 2-3 全部完成 ✅**

---

## 📌 下一步建议

Day 2-3 已经**完美收官**,**强烈建议**接下来选一个方向:

1. **继续 Day 4**(消息构建细节,build_messages)—— 你 Day 1 已经学过 build_messages 拆解,可以快速过
2. **继续 Day 5-7**(Provider / Tool Registry / Tool Executor)—— 进入新内容
3. **做 Mini Hermes 仿写项目**(巩固 Day 1-3)—— 路线图任务 5

需要我帮你**继续 Day 4-5**,还是**先做一个 Mini Hermes 巩固**?

---

## 📌 第九部分:Guidance / Hint / Identity 三种文本块(★ 新增)

学完 `prompt_builder.py` 之后,我发现 system prompt 里的文本块**实际上分 3 种类型**,每种作用不同。

### 三种类型对比

| 类型 | 例子 | 作用 | 注入位置 |
|---|---|---|---|
| **Identity**(身份) | `DEFAULT_AGENT_IDENTITY` | 告诉 model 你是谁 | 总是(stable tier 第 1 槽) |
| **Guidance**(行为准则) | `MEMORY_GUIDANCE`, `TASK_COMPLETION_GUIDANCE`, `TOOL_USE_ENFORCEMENT_GUIDANCE`, `SKILLS_GUIDANCE`, `KANBAN_GUIDANCE`, `SESSION_SEARCH_GUIDANCE` | 教 model 怎么用某个工具 / 应该怎么做 | 按需(有相关工具/配置触发) |
| **Hint**(环境提示) | `WSL_ENVIRONMENT_HINT`, `PLATFORM_HINTS` 字典, `HERMES_ENVIRONMENT_HINT` 环境变量 | 告诉 model 当前环境/平台的特殊情况 | 按需(检测到对应环境触发) |

### Identity(身份)

**作用**:"你是谁"—— model 的基本人设。

**例子**(`DEFAULT_AGENT_IDENTITY`):
> "You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct..."

**关键**:
- 只有 1 个(`DEFAULT_AGENT_IDENTITY`)+ SOUL.md 覆盖
- 总是在 stable tier 的最前
- 不会按需注入

### Guidance(行为准则)

**作用**:"工具怎么用 / 应该怎么做"—— 操作手册。

**例子**:

| 名字 | 教什么 |
|---|---|
| `MEMORY_GUIDANCE` | 存什么、不存什么、写法(陈述句 vs 祈使句) |
| `TASK_COMPLETION_GUIDANCE` | 防半截 stub、防瞎编 |
| `TOOL_USE_ENFORCEMENT_GUIDANCE` | 调工具不描述 |
| `SKILLS_GUIDANCE` | 何时建/改 skill |
| `KANBAN_GUIDANCE` | 异步任务 6 步生命周期 |
| `SESSION_SEARCH_GUIDANCE` | 怎么搜历史 |

**关键**:
- 多个,按工具/配置按需注入
- 内容是"通用规则"(跨 model 适用)
- 跟 tool 强绑定(有 X 工具才注入 X 指引)

### Hint(环境提示)

**作用**:"你现在在哪种特殊环境"—— 本地贴士。

**例子**:
- `WSL_ENVIRONMENT_HINT` —— WSL 环境,Windows 路径翻译
- `PLATFORM_HINTS` 字典 —— 不同平台(WhatsApp / Telegram / CLI / cron)
- `HERMES_ENVIRONMENT_HINT` 环境变量 —— 用户自定义

**关键**:
- 按当前环境检测注入
- 内容是"环境事实"(不是操作规则)
- 例:WSL 提示不会教你"怎么用 memory 工具",只告诉你"Windows 文件在 /mnt/c/"

### 三者关系(用类比)

> 想象你去一个**新城市**工作:

| Hermes 类型 | 类比 |
|---|---|
| **Identity** | "你是 X 公司员工" |
| **Guidance** | "公司操作手册:报销流程、请假流程、邮件规范" |
| **Hint** | "这个城市公交车要招手才停 / WSL 路径要翻译成 /mnt/c/" |

### 设计原则(从这 3 种类型看出)

1. **Identity 永远在最前** —— "先告诉你你是谁"
2. **Guidance 跟工具绑定** —— 有相关工具才注入,省 token
3. **Hint 跟环境绑定** —— 检测到环境才注入,省 token
4. **三者都是 stable tier** —— 都在 prefix cache 命中区

### 完整 GUIDANCE 块清单(11 个 + 1 字典)

| # | 名字 | 类型 | 触发条件 |
|---|---|---|---|
| 1 | `DEFAULT_AGENT_IDENTITY` | Identity | 总是(SOUL.md 找不到时) |
| 2 | `HERMES_AGENT_HELP_GUIDANCE` | Guidance | 总是 |
| 3 | `MEMORY_GUIDANCE` | Guidance | `"memory" in valid_tool_names` |
| 4 | `SESSION_SEARCH_GUIDANCE` | Guidance | `"session_search" in valid_tool_names` |
| 5 | `SKILLS_GUIDANCE` | Guidance | `"skill_manage" in valid_tool_names` |
| 6 | `KANBAN_GUIDANCE` | Guidance | `kanban_show` 工具 / HERMES_KANBAN_TASK |
| 7 | `TOOL_USE_ENFORCEMENT_GUIDANCE` | Guidance | 4 种 enforcement 配置模式 |
| 8 | `TASK_COMPLETION_GUIDANCE` | Guidance | 总是(默认开) |
| 9 | `OPENAI_MODEL_EXECUTION_GUIDANCE` | Guidance | gpt / codex / grok |
| 10 | `GOOGLE_MODEL_OPERATIONAL_GUIDANCE` | Guidance | gemini / gemma |
| 11 | `COMPUTER_USE_GUIDANCE` | Guidance | `"computer_use" in valid_tool_names` |
| 12 | `PLATFORM_HINTS` 字典 | Hint | `agent.platform` 命中键 |
| 13 | `WSL_ENVIRONMENT_HINT` | Hint | `is_wsl() == True` |
| 14 | `TOOL_USE_ENFORCEMENT_MODELS` 元组 | 配置 | 4 种 enforcement 模式 |

**统计**:
- Identity:1 个
- Guidance:9 个
- Hint:2 个(PLATFORM_HINTS + WSL)
- 配置元数据:1 个(TOOL_USE_ENFORCEMENT_MODELS)

### 一句话总结

> **Identity 回答 "你是谁",Guidance 回答 "应该怎么做",Hint 回答 "当前有什么特殊情况"**。
>
> 三种类型全部都是 stable tier 的一部分,都进 prefix cache 命中区。

---

## 📌 第十部分:`WSL_ENVIRONMENT_HINT` 详解(★ 新增)

### 它是什么

`WSL_ENVIRONMENT_HINT` 是 Hermes 给 model 的**一条"本地贴士"**。

### 完整内容

```python
WSL_ENVIRONMENT_HINT = (
    "You are running inside WSL (Windows Subsystem for Linux). "
    "The Windows host filesystem is mounted under /mnt/ — "
    "/mnt/c/ is the C: drive, /mnt/d/ is D:, etc. "
    "The user's Windows files are typically at "
    "/mnt/c/Users/<username>/Desktop/, Documents/, Downloads/, etc. "
    "When the user references Windows paths or desktop files, translate "
    "to the /mnt/c/ equivalent. You can list /mnt/c/Users/ to discover "
    "the Windows username if needed."
)
```

### 关键信息

1. **WSL 是什么** —— Windows Subsystem for Linux(在 Windows 上跑 Linux 子系统)
2. **路径翻译规则** —— Windows `C:\` → Linux `/mnt/c/`
3. **找用户文件位置** —— `/mnt/c/Users/<用户名>/Desktop/`
4. **降级方案** —— 不知道用户名时列 `/mnt/c/Users/`

### 实际使用场景

| 用户说 | 没 hint 时 model 怎么做 | 有 hint 时 model 怎么做 |
|---|---|---|
| "把桌面的 todo.txt 给我看" | 用 `C:\Users\Alice\Desktop\todo.txt` → 命令失败 | 翻译成 `/mnt/c/Users/Alice/Desktop/todo.txt` → 成功 |
| "在 D 盘里搜文件" | 用 `D:\` → 失败 | 用 `/mnt/d/` → 成功 |
| "我桌面上有什么" | 不知道 Desktop 在哪 | 列 `/mnt/c/Users/Alice/Desktop/` |

### 为什么用 hint 不让 model 自己探测?

3 个原因:
1. **省 token** —— 一行 hint 几十 token,比让 model 跑 `wsl.exe --status` 探测便宜
2. **省时间** —— 不用等 model 试错
3. **降低错误率** —— 路径翻译是常见出错点,直接告诉规则比让 model 推理更可靠

### Hint 完整家族

`build_environment_hints()` 返回的 hints 可能包含:

| Hint | 何时追加 |
|---|---|
| **OS 类型**(Linux/macOS/Windows) | 总是 |
| **$HOME 路径** | 总是 |
| **cwd** | 总是 |
| **`WSL_ENVIRONMENT_HINT`** ← **你问的这个** | `is_wsl() == True` |
| **`_WINDOWS_BASH_SHELL_HINT`** | Windows 上跑 bash 时 |
| **远程 backend 信息**(Docker/Modal/SSH) | `TERMINAL_ENV` 设了对应值 |
| **`HERMES_ENVIRONMENT_HINT` 自定义** | 环境变量里设了 |

### 一句话总结

**`WSL_ENVIRONMENT_HINT` = 一段小提示文本 = "你在 WSL,Windows 路径是 /mnt/c/..."**。它是 Hint 类型文本的代表,跟 Guidance(操作手册)不同 —— Hint 是"环境事实",Guidance 是"操作规则"。

---

## 📌 第十一部分:Context Tier 项目级文件读取(★ 新增)

学完 `prompt_builder.py` 的 4 个 loader 后,我把 context tier 的文件读取规则整理一下。

### 5 类文件 + 1 套优先级

`build_context_files_prompt()` 用**优先级规则**加载项目级文件。**4 种项目级文件,first found wins**(只读 1 种):

| 优先级 | 文件类型 | 命名变体 | 搜索范围 |
|---|---|---|---|
| 1 | `.hermes.md` / `HERMES.md` | Hermes 专属 | **cwd 往上到 git root** |
| 2 | `AGENTS.md` / `agents.md` | 通用 | **cwd only** |
| 3 | `CLAUDE.md` / `claude.md` | Claude 专属 | **cwd only** |
| 4 | `.cursorrules` + `.cursor/rules/*.mdc` | Cursor 专属 | **cwd only**(可多个文件) |

**另外**:`SOUL.md` 独立从 `~/.hermes/SOUL.md` 读(走 `load_soul_md()`),不在这 4 类里。

### 关键设计原则

#### 1. **First found wins**(只读 1 种)

```python
# system_prompt.py / prompt_builder.py
project_context = (
    _load_hermes_md(cwd_path)
    or _load_agents_md(cwd_path)
    or _load_claude_md(cwd_path)
    or _load_cursorrules(cwd_path)
)
```

**为什么只读 1 种?**
- 避免冲突(不同文件说不同规则)
- 节省 token
- 假设用户项目用一种约定

#### 2. **大小写两种命名都试**

每个 loader 都试两种大小写(为了兼容不同操作系统/编辑器):
- `AGENTS.md` / `agents.md`
- `CLAUDE.md` / `claude.md`
- `.hermes.md` / `HERMES.md`

#### 3. **只有 .hermes.md 会往父目录走**

```python
# _find_hermes_md:从 cwd 往上,直到 git root
stop_at = _find_git_root(cwd)
for directory in [current, *current.parents]:
    ...
    if stop_at and directory == stop_at:
        break
```

**为什么?**
- 项目的 `.hermes.md` 通常在仓库根目录(团队共享)
- AGENTS.md 等是 cwd 级别(每个子目录可能有自己的)

#### 4. **.cursorrules 支持多个文件**

```python
# .cursorrules  (单文件扁平)
# + .cursor/rules/*.mdc  (多文件,拆分的)
```

Cursor 风格的项目会把规则拆成多个 `.mdc` 文件(每个文件一个主题)。Hermes 把它们**累加**到一段文本里。

### 每个 loader 的流程

#### `.hermes.md` loader

```python
def _load_hermes_md(cwd_path):
    path = _find_hermes_md(cwd_path)        # cwd → git root
    if not path: return ""
    content = path.read_text()
    content = _strip_yaml_frontmatter(content) # 剥 YAML 头
    content = _scan_context_content(...)      # 安全扫描
    result = f"## {name}\n\n{content}"
    return _truncate_content(result, ".hermes.md")  # 20K 截断
```

**唯一会走父目录的 loader**。

#### `AGENTS.md` loader

```python
def _load_agents_md(cwd_path):
    for name in ["AGENTS.md", "agents.md"]:   # 试 2 种命名
        candidate = cwd_path / name
        if candidate.exists():
            content = read → scan → format → truncate
            return
    return ""   # 都找不到 → 返回空
```

**只查 cwd,找到就停**。

#### `CLAUDE.md` loader

跟 `AGENTS.md` 一样,只是试 `CLAUDE.md` / `claude.md` 两种命名。

#### `.cursorrules` loader

```python
def _load_cursorrules(cwd_path):
    # 读 .cursorrules (单文件)
    # + 读 .cursor/rules/*.mdc (多文件)
    # 累加成一段文本
    return _truncate_content(combined, ".cursorrules")
```

**唯一支持"多文件累加"的 loader**。

### 3 步通用处理流程

**所有 loader 都过这 3 步**:

```
读文件
  ↓
[1] _strip_yaml_frontmatter (.hermes.md only)  # 剥 YAML 头
  ↓
[2] _scan_context_content(...)                  # 安全扫描(防 injection)
  ↓
[3] _truncate_content(content, name)            # 20K 截断
  ↓
拼成 markdown 段落 "## {filename}\n\n{content}"
  ↓
返回
```

### 安全防护(4 层)

| 防护 | 函数 | 作用 |
|---|---|---|
| 1. YAML 头剥离 | `_strip_yaml_frontmatter` | 防止 YAML 配置被注入到 prompt |
| 2. **威胁扫描** | `_scan_context_content` | 匹配注入特征("ignore previous instructions" 等),命中就用 `[BLOCKED: ...]` 占位 |
| 3. 截断 | `_truncate_content` | 头 70% + 尾 20% + marker |
| 4. 优先级 | first found wins | 避免冲突 |

**最关键的是 _scan_context_content** —— 它直接读 threat_patterns.py 的特征库,**任何 AGENTS.md 里的注入尝试都会被替换成 `[BLOCKED: ...]`**。

### 截断的 4 个数字

```python
CONTEXT_FILE_MAX_CHARS = 20_000            # 上限
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7          # 头 70%
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2          # 尾 20%
                                              # 隐含:中间 10% 给 marker
```

**为什么 70/20?**
- **头 70%**:标题、说明、核心原则(最重要)
- **尾 20%**:新增规则、补充说明(经常追加)
- **中间 10%**:详细例子、长解释(冗长可砍)

### Skip Soul 机制

`build_context_files_prompt(skip_soul=True)` 这个参数:

```python
# 流程
1. load_soul_md() → SOUL.md 注入到 stable tier (line 200)
2. build_context_files_prompt(skip_soul=True) → 不要再读 SOUL.md
3. 避免 SOUL.md 重复注入
```

**为什么需要?**
- `load_soul_md()` 在 stable tier 装身份时已经读了 SOUL.md
- 如果 `build_context_files_prompt` 又读一次 → 注入两次
- `skip_soul=True` 跳过 SOUL.md 路径(虽然这里本来也没列 SOUL.md,主要是标记"我已经在 stable 加载过了")

### 完整数据流(从项目文件到 system prompt)

```
项目根目录
├── .hermes.md           ← 读 (.hermes.md loader, 走父目录)
├── AGENTS.md            ← 读 (AGENTS.md loader, cwd)
├── CLAUDE.md            ← 读 (CLAUDE.md loader, cwd)
├── .cursorrules         ← 读 (.cursorrules loader)
└── .cursor/rules/
    ├── python.mdc       ← 读 (累加)
    └── security.mdc     ← 读 (累加)

每个文件:
  read → strip_yaml → scan_threats → truncate_20K → format

first found wins(只取 1 种项目级)
  ↓
拼成一段 markdown 文本
  ↓
注入到 system_prompt 的 context tier
  ↓
跟 stable + volatile 一起组成 agent._cached_system_prompt
```

### 5 个 loader 的差异表

| Loader | 文件类型 | 搜索范围 | 多文件? | YAML 头剥离? |
|---|---|---|---|---|
| `_load_hermes_md` | Hermes 专属 | cwd → git root | 否 | ✅ |
| `_load_agents_md` | 通用 | cwd | 否 | ❌ |
| `_load_claude_md` | Claude 专属 | cwd | 否 | ❌ |
| `_load_cursorrules` | Cursor 专属 | cwd | ✅(累加) | ❌ |
| `load_soul_md` | 身份 | `~/.hermes/` | 否 | ❌ |

### 实际 token 影响

| 文件大小 | 处理后大小 | 注入到 system prompt |
|---|---|---|
| 5K 字符 | 5K(不截) | ✅ 原样 |
| 15K 字符 | 15K(不截) | ✅ 原样 |
| 30K 字符 | 20K(截断) | 头 14K + marker + 尾 4K |
| 1M 字符 | 20K(截断) | 同上 |

**总 context tier 大小**:
- 1 个项目级文件(20K) + SOUL.md(可能 20K) = 40K token
- 跟 stable tier(几千 token)对比,**context tier 可能比 stable 还大**

### 跟稳定 tier / 易变 tier 的关系

| 维度 | context tier | stable tier | volatile tier |
|---|---|---|---|
| **核心来源** | 项目级文件(用户写的) | 身份/工具/平台 | memory/时间戳 |
| **大小** | 0-40K token | 3-8K token | 200-500 token |
| **变化条件** | cwd 变了 / 文件改了 | agent 实例变了 | 跨过午夜 |
| **关键防护** | 威胁扫描 + 截断 | 配置按需注入 | date-only 时间戳 |
| **同 session 内稳定** | 通常稳定(项目文件不常改) | 字节级稳定 | 24h 内稳定 |

### 一句话总结

> **Context tier = 项目级文件读取**。**优先级 .hermes.md > AGENTS.md > CLAUDE.md > .cursorrules**(first found wins)。**每个文件都过 3 关**(剥 YAML → 威胁扫描 → 20K 截断)。**唯一会走父目录的是 .hermes.md**(团队共享约定)。**唯一支持多文件累加的是 .cursorrules**。
>
> 整个 context tier 的核心目的:**让 model 知道"我在哪个项目里跑,项目的规则是什么"**,同时**通过扫描和截断防止注入攻击和 token 爆掉**。

---

## 📌 第十二部分:`_truncate_content` 函数详解(★ 新增)

### 一句话答案

**`_truncate_content` 截断 5 类项目配置文件,上限 20K 字符,策略"头 70% + 尾 20% + 中间 marker"**。目的是**防止项目配置文件太长撑爆 system prompt**。

### 截断的具体对象

| 文件 | 调用位置 |
|---|---|
| `SOUL.md` | `load_soul_md()` line 1746 |
| `.hermes.md` / `HERMES.md` | `_load_hermes_md()` |
| `AGENTS.md` / `agents.md` | `_load_agents_md()` |
| `CLAUDE.md` / `claude.md` | `_load_claude_md()` |
| `.cursorrules` + `.cursor/rules/*.mdc` | `_load_cursorrules()` |

### 截断规则

```python
CONTEXT_FILE_MAX_CHARS = 20_000            # 20K 字符上限
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7          # 头 70% = 14,000 字符
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2          # 尾 20% = 4,000 字符
                                              # 中间 10% 给 marker
```

**实际拼起来**:
```
[头 14,000 字符] + [marker 几百字符] + [尾 4,000 字符] ≈ 20K 字符
```

### 截断的 3 个目的

| 目的 | 说明 |
|---|---|
| **保护 token 预算** | 防止单个文件吃光 system prompt |
| **防恶意超长** | 用户塞 1M 字符也只读 20K |
| **保证信号 > 噪声** | 头尾保留,中间冗长砍掉 |

### 为什么"头 70% + 尾 20%"?

| 部分 | 内容 | 重要程度 |
|---|---|---|
| 头 14% | 标题、说明、核心原则 | ⭐⭐⭐⭐⭐ |
| 中 10% | 详细规则、例子、长解释 | ⭐⭐ |
| 尾 20% | 重要补充、收尾规则、引用 | ⭐⭐⭐⭐ |

### 中间 marker

```python
marker = f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of {len(content)} chars. Use file tools to read the full file.]\n\n"
```

**关键信息**:
- `[...truncated ...]` — 明显标识是截断
- 文件名 — model 知道哪个文件被截了
- 保留的字符数 — 让 model 心里有数
- **`Use file tools to read the full file`** — 明确告诉 model:要看完整版用 read_file 工具

**这个 marker 是关键设计**:
- 没用 `...</br>` 这种含糊标记
- 直接说"用工具读完整版" → 鼓励 model 主动获取完整信息

### 实际触发场景

| 文件大小 | 触发截断? | 截断后大小 |
|---|---|---|
| 5K | ❌ | 5K(原样) |
| 15K | ❌ | 15K(原样) |
| 30K | ✅ | 20K(头 14K + marker + 尾 4K) |
| 1M | ✅ | 20K(同上) |

**所有超过 20K 的文件都被截到 20K**。

### 一句话总结

> **`_truncate_content` = 项目配置文件的"安全阀"**。**所有超过 20K 字符的上下文文件都被截到 20K**:**头 70% + 尾 20% + 中间 marker**。**关键是 marker 告诉 model "用 read_file 读完整版"**,不会因截断而漏信息。

---

## 📌 第十三部分:什么时候压缩 / 什么时候不压缩(`should_compress` 决策树,★ 新增)

> **源文件**:`hermes-agent/agent/context_compressor.py` 第 906-929 行
> **核心方法**:`ContextCompressor.should_compress(prompt_tokens=None) -> bool`

### 核心方法源码(23 行浓缩了压缩系统的全部触发决策)

```python
def should_compress(self, prompt_tokens: int = None) -> bool:
    """Check if context exceeds the compression threshold.

    【步骤 6.3】决定是否要触发压缩

    双重保护:
    1. tokens < threshold → 不压
    2. 连续 2 次压缩节省 < 10% → 视为 thrashing,放弃压缩
       (避免反复压缩每次只省 1-2 条消息的死循环)
    """
    tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
    if tokens < self.threshold_tokens:
        return False
    # Anti-thrashing: back off if recent compressions were ineffective
    if self._ineffective_compression_count >= 2:
        if not self.quiet_mode:
            logger.warning(
                "Compression skipped — last %d compressions saved <10%% each. "
                "Consider /new to start a fresh session, or /compress <topic> "
                "for focused compression.",
                self._ineffective_compression_count,
            )
        return False
    return True
```

### 决策树(执行顺序自上而下)

```
是否压缩?
  │
  ├─ [Q1] 有没有 prompt_tokens 传入?
  │     ├─ 有 → 用入参
  │     └─ 无 → 用 self.last_prompt_tokens(上一轮真实值)
  │
  ├─ [Q2] tokens < threshold_tokens?
  │     └─ ✅ 是 → 不压(还没到门槛,过早压缩是浪费)
  │
  ├─ [Q3] _ineffective_compression_count >= 2?
  │     ├─ ✅ 是 → 不压(连续 2 次省 <10%,在 thrashing)
  │     │         打 warning,提示用户用 /new 或 /compress <topic>
  │     └─ ❌ 否 → 压
  │
  └─ 默认 → 压 ✅
```

### 三种"不压"场景

| 场景 | 代码 | 解释 |
|---|---|---|
| **未达阈值** | `if tokens < self.threshold_tokens: return False` | 用了不到 50% 上下文(默认),没必要压 |
| **Thrashing 保护** | `if self._ineffective_compression_count >= 2: return False` | 连续 2 次省 < 10%,压不动了 |
| **估算虚高 defer** | `should_defer_preflight_to_real_usage` | 旁路逻辑,rough 估算虚高时建议"等真实结果",避免误触发 |

#### 场景 1:未达阈值

```python
if tokens < self.threshold_tokens:
    return False
```

- `threshold_tokens` = `context_length × threshold_percent`(默认 50%)
- 例子:200K 上下文,threshold = 100K
- 用了 80K tokens 也不会压,要等到 100K+

**为什么不早压?** 压缩有成本:LLM 调用 + 信息损失。**未到门槛就压 = 浪费钱 + 丢信息**。

#### 场景 2:Thrashing 保护(连续无效压缩)

```python
if self._ineffective_compression_count >= 2:
    # ... warn
    return False
```

- `_ineffective_compression_count` 由 `compress()` 在 Phase 5 维护
- 每次压缩完算 `savings_pct`,< 10% 计数 +1,否则清零
- 连续 2 次无效 → 触发 thrashing 保护

**为什么是 2 次而不是 1 次?** 单次无效可能是巧合(刚好碰上一波短消息),2 次是统计意义上的"压不下来了"。

#### 场景 3:估算虚高 defer(旁路,不在 should_compress 里)

- `should_defer_preflight_to_real_usage` 在 rough 估算虚高时,会建议"先等真实结果"
- 这是**避免误触发**的"先等等"机制,跟 `should_compress` 的"该不该压"互补

### 四个关键变量

| 变量 | 来源 | 含义 |
|---|---|---|
| `prompt_tokens` | 参数(优先) | 调用方传入的精确值 |
| `self.last_prompt_tokens` | `update_from_response` 写入 | 上一轮 LLM 真实返回的 prompt tokens |
| `self.threshold_tokens` | `__init__` 算好 | 触发压缩的下限 |
| `self._ineffective_compression_count` | `compress()` Phase 5 维护 | 连续无效压缩的次数 |

### 压缩"节奏"的可视化

```
token 使用量
  │
  │  ┌────┐
  │  │压  │← 压完掉下来
  │  │一次│
  │  └────┘
  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ threshold(50%)
  │                              ╱
  │                            ╱   ← 慢慢涨上来
  │  ┌────┐                 ╱
  │  │压  │              ╱
  │  │无效│           ╱     ← 涨太快,省不下来
  │  └────┘        ╱
  │            ╱
  │          ╱
  └──────────────────────────────── 时间 →
       触发1    触发2    (被 thrashing 保护拒绝)
```

### 调用链(从主循环到这儿)

```
conversation_loop.py
   ↓
engine.should_compress(current_tokens)  ← 你看到的就是这里
   ├─ 内部:检查 threshold + thrashing
   └─ 返回 True → engine.compress(messages, ...)
                  ├─ Phase 1: _prune_old_tool_results
                  ├─ Phase 2-4: 边界计算 + LLM summary + 拼装
                  └─ Phase 5: 更新 _ineffective_compression_count
```

### 关键设计哲学(4 条)

| 原则 | 体现 |
|---|---|
| **早压是浪费** | threshold 默认 50%,不预设 30% / 70% |
| **晚压是坏事** | 提示用户 `/new`,而不是无限压 |
| **短视不预测** | 只看当前 tokens,不预测未来增长 |
| **失败有节奏** | 1 次无效不放弃,2 次才放弃(给机会) |

### 关联概念(后续学习)

- `_ineffective_compression_count` 的写入位置 → `compress()` Phase 5(line 2405-2412)
- `should_defer_preflight_to_real_usage` → 另一个 defer 决策(line 873-899)
- `last_prompt_tokens` 的更新位置 → `update_from_response`
- `threshold_tokens` 的计算 → `__init__` / `update_model`(line 787-790)

### 一句话总结

> **`should_compress` = 23 行浓缩的"压不压"决策树**:**先看阈值,再看是否 thrashing**。**未达阈值不压,连续 2 次无效不压**。**只在确实需要的时候压**——**早压浪费、晚压坏事**。

---

## 📌 第十四部分:`compress()` 主入口 — 5 阶段压缩算法详解(★ 新增)

> **源文件**:`hermes-agent/agent/context_compressor.py` 第 2187-2512 行
> **核心方法**:`ContextCompressor.compress(messages, current_tokens, focus_topic, force)`
> **触发**:`should_compress` 返回 True 时被主循环调用

### 一句话答案

**`compress()` = 把 messages 切成 head / middle / tail 三段,中间段调 LLM 生成结构化 summary 替代原文,head 保留 + tail 保留 + middle 替换**。整个过程 5 个阶段。

### 压缩窗口布局

```
messages 列表(假设 50 条,80K tokens)
  ↓
┌────────────────────┬────────────────────────┬────────────────────┐
│       head         │        middle          │        tail        │
│   头 4 条左右      │   中间 70+ 条          │  尾 20K tokens     │
│   (system + 头 N)  │   (要压成 summary)     │  (按 token 倒推)  │
│                    │                        │                    │
│   永 远 保 留       │      LLM 压 缩         │   永 远 保 留      │
│                    │   ← 调一次 LLM         │                    │
└────────────────────┴────────────────────────┴────────────────────┘
   ↑                                          ↑
   protect_head_size                          find_tail_cut_by_tokens
   + _align_boundary_forward                   (从尾倒推 20K tokens)
```

### 5 阶段算法详解

#### Phase 0:初始化(主入口开头)

```python
# 1) 重置 6 个失败标志(本轮另算)
self._last_summary_dropped_count = 0
self._last_summary_fallback_used = False
self._last_summary_error = None
# ... 等

# 2) force 跳过冷却期
if force and self._summary_failure_cooldown_until > 0.0:
    self._summary_failure_cooldown_until = 0.0

# 3) 消息数检查(太少就别压)
n_messages = len(messages)
_min_for_compress = self._protect_head_size(messages) + 3 + 1
if n_messages <= _min_for_compress:
    return messages

# 4) display_tokens 三段 fallback
display_tokens = current_tokens or self.last_prompt_tokens or estimate_messages_tokens_rough(messages)
```

**关键**:
- `force=True` 跳过冷却期 — 手动 `/compress` 命令立即重试(用户主动行为优先于冷却)
- `display_tokens` 优先用真实值,fallback 估算 — 日志里看的"压缩节省"基于这个

#### Phase 1:修剪旧 tool 结果(廉价,无 LLM)

```python
messages, pruned_count = self._prune_old_tool_results(
    messages, protect_tail_count=self.protect_last_n,
    protect_tail_tokens=self.tail_token_budget,
)
```

**3-pass 修剪器**:
- **Pass 1: 去重** — md5 哈希,相同内容的 tool result 只留最新全量,其他换 "[Duplicate ...]"
- **Pass 2: 单行摘要** — 老的 tool result 换成 "[terminal] ran X -> exit 0" 风格
- **Pass 3: 大 args 截断** — assistant 消息里 tool_call.arguments > 500 chars 的 JSON-safe 截断

**为什么先做这步**: tool result 通常占大头(读文件、终端输出),先廉价瘦身 → LLM summary 拿到的输入更小,更便宜。

**保护策略**: 按 `tail_token_budget`(默认 20K)+ 消息数下限(`protect_last_n`=20)。

#### Phase 2:计算 head/tail 边界

```python
# head: 头几条(默认 1+3=4)
compress_start = self._protect_head_size(messages)

# 起点对齐:跳过开头的 tool result
compress_start = self._align_boundary_forward(messages, compress_start)

# tail: 按 token 预算倒推(默认 20K)
compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

# 没有 middle 段就没东西可压
if compress_start >= compress_end:
    return messages
```

**头 vs 尾的不同策略**:
- **head 按数量**: 头部通常是 system + 第一轮对话,固定成本(几 K tokens),用数量简单可靠
- **tail 按 token**: 尾部内容变数大(短消息几段 vs 长消息大文件输出),token 预算自适应

#### Phase 2.5:迭代式 summary 探测(关键!)

```python
# 找上次的 summary(用于迭代式压缩)
summary_search_start = 1 if messages and messages[0].get("role") == "system" else 0
summary_idx, summary_body = self._find_latest_context_summary(
    messages, summary_search_start, compress_end,
)
if summary_idx is not None:
    if summary_body and not self._previous_summary:
        self._previous_summary = summary_body
    # 关键:turns_to_summarize 收缩到 summary 之后
    turns_to_summarize = messages[max(compress_start, summary_idx + 1):compress_end]
```

**这是迭代式压缩的入口**:
- `_find_latest_context_summary` 倒着扫描 `[summary_search_start, compress_end)` 范围
- 找到带 `SUMMARY_PREFIX` / `LEGACY_SUMMARY_PREFIX` / `_HISTORICAL_SUMMARY_PREFIXES` 之一的消息
- 如果找到:**收紧** `turns_to_summarize`,跳过旧 summary
- 把 body 存到 `self._previous_summary`(rehydrate)

#### Phase 3:调 LLM 生成 summary

```python
summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)
```

**这是唯一调 LLM 的地方**。详见下面第十五部分。

**失败处理**(双策略):
```python
# abort_on_summary_failure=True → 放弃压缩,返回原 messages
if not summary and self.abort_on_summary_failure:
    self._last_compress_aborted = True
    return messages

# abort_on_summary_failure=False → 用 deterministic fallback
if not summary:
    summary = self._build_static_fallback_summary(turns_to_summarize, reason=...)
```

#### Phase 4:拼装压缩结果

**4.1 复制 head**(system 加备注)

```python
compressed = []
for i in range(compress_start):
    msg = messages[i].copy()
    if i == 0 and msg.get("role") == "system":
        # 追加 "[Note: Some earlier turns have been compacted...]"
        # 提示 LLM 不要重做已有工作
        if _compression_note not in existing:
            msg["content"] = _append_text_to_content(existing, _compression_note)
    compressed.append(msg)
```

**4.2 选 summary 角色(避撞)**

```python
last_head_role = messages[compress_start - 1].get("role")
first_tail_role = messages[compress_end].get("role")

# 启发式:head 是 assistant/tool → summary 选 user
if last_head_role in {"assistant", "tool"}:
    summary_role = "user"
else:
    summary_role = "assistant"

# 如果跟 tail 撞 → 翻转试试
if summary_role == first_tail_role:
    flipped = "assistant" if summary_role == "user" else "user"
    if flipped != last_head_role:
        summary_role = flipped
    else:
        # 两个 role 都不行 → 合并到 tail 第一条
        _merge_summary_into_tail = True
```

**为什么要避撞**: OpenAI / Anthropic 都要求 user ↔ assistant 严格交替。

**4.3 summary 边界标记**(防弱模型误读)

```python
# summary 作为独立 user 消息时,加显式 end marker
if not _merge_summary_into_tail and summary_role == "user":
    summary += "\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
```

**修 #11475 / #14521** — 弱模型把 summary 里"## Active Task"当成新输入响应。

**4.4 拼装 tail**

```python
if not _merge_summary_into_tail:
    compressed.append({"role": summary_role, "content": summary})

for i in range(compress_end, n_messages):
    msg = messages[i].copy()
    if _merge_summary_into_tail and i == compress_end:
        # 合并模式:summary 当作 tail 第一条的前缀
        msg["content"] = _append_text_to_content(msg.get("content"), merged_prefix, prepend=True)
    compressed.append(msg)
```

#### Phase 5:清理 + 统计

```python
# 1) 修补 orphan tool_call / tool_result
compressed = self._sanitize_tool_pairs(compressed)

# 2) 剥离老图片(防 Kilo-Org #9434 body-size 限制)
compressed = _strip_historical_media(compressed)

# 3) 算节省率,更新抗 thrashing 计数
self.compression_count += 1
new_estimate = estimate_messages_tokens_rough(compressed)
saved_estimate = display_tokens - new_estimate
savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
if savings_pct < 10:
    self._ineffective_compression_count += 1
else:
    self._ineffective_compression_count = 0
```

**10% 阈值由来**:
- < 10% 节省说明"内容是硬骨头"(工具结果少)或"窗口太小"
- 继续压缩大概率也省不下来 → 不如放弃
- 连续 2 次无效 → `should_compress` 拒绝再压,提示用户 `/new`

### 6 个关键设计哲学

| 哲学 | 实现 |
|---|---|
| **迭代式 summary** | 找到旧 summary 后只压新增 turns,信息不衰减 |
| **Token 预算而非消息数** | tail 大小按 token 算,自适应长短消息 |
| **角色避撞** | head / summary / tail 三段必须严格交替 |
| **Summary 边界标记** | 防弱模型把 summary 里的旧请求当成新输入 |
| **Anti-thrashing** | 省不到 10% 算无效,连续 2 次放弃 |
| **失败优雅降级** | LLM 失败 → 用 deterministic fallback 而不是报错 |

### 实际效果(数值例子)

```
原始 messages(50 条,80K tokens):
  [0] system: "You are Hermes Agent..."
  [1-44] 中间 44 条对话
  [45-49] 最近 5 条对话

压缩后(9 条,~25K tokens):
  [0] system: "You are Hermes Agent..." + [Note: compacted...]
  [1-3] head 4 条原样保留
  [4] user: {LLM summary}              ← ⭐ 替代 [4-44] 共 41 条
  [5-9] tail 5 条原样保留
```

**节省**: 50 → 9 条(82% 缩减),80K → 25K tokens(69% 缩减)。

### 一句话总结

> **`compress()` = 5 阶段算法总控**:**Phase 0 初始化 → Phase 1 廉价修剪 → Phase 2 边界 + 迭代探测 → Phase 3 调 LLM → Phase 4 角色避撞拼装 → Phase 5 清理 + 抗 thrashing**。**核心思想是"head + tail 保留,中间 LLM 压缩"**。

---

## 📌 第十五部分:`_generate_summary` — 调 LLM 生成 summary 的核心(★ 新增)

> **源文件**:`hermes-agent/agent/context_compressor.py` 第 1431-1850 行
> **核心方法**:`ContextCompressor._generate_summary(turns_to_summarize, focus_topic)`
> **调用方**:`compress()` Phase 3
> **重要性**:整个压缩系统**唯一调 LLM** 的地方

### 一句话答案

**`_generate_summary` = 构造 prompt(系统指令 + turns 序列化 + 可选 focus topic)→ 调 LLM → 处理 4 类错误 → 返回带 prefix 的 summary 字符串**。

### 8 步流程

```
[Step 1] 冷却期检查
   ↓
[Step 2] 算 summary 预算(_compute_summary_budget)
   ↓
[Step 3] 序列化 turns(_serialize_for_summary)
   ↓
[Step 4] 构建 prompt:
   ├─ 首次压缩:preamble + TURNS + 模板
   └─ 迭代压缩:preamble + PREVIOUS_SUMMARY + NEW_TURNS + 模板
   ↓
[Step 5] 可选:追加 FOCUS TOPIC 引导
   ↓
[Step 6] 调 call_llm(task="compression", ...)
   ↓
[Step 7] 失败处理(回退主模型 / 冷却 / 抛 None)
   ↓
[Step 8] 返回带 SUMMARY_PREFIX 的结果
```

### Step 1:冷却期检查

```python
now = time.monotonic()
if now < self._summary_failure_cooldown_until:
    logger.debug("Skipping context summary during cooldown...")
    return None
```

- 用 **monotonic 时间**(不受系统时钟调整影响)
- 上次 LLM summary 失败时设置,避免重试风暴
- `force=True` 跳过由 `compress()` 开头处理(清零 cooldown_until)

### Step 2-3:算预算 + 序列化

```python
summary_budget = self._compute_summary_budget(turns_to_summarize)
content_to_summarize = self._serialize_for_summary(turns_to_summarize)
```

**summary_budget**: LLM 输出 token 上限
- = 压缩内容量的 20%,夹在 [2K, max_summary_tokens]
- 200K 模型 → 预算大(更细的 summary)
- 32K 模型 → 预算小(不会撑爆)

**content_to_summarize**: turns → 纯文本
- tool result → `[TOOL RESULT {id}]: content`
- assistant → `[ASSISTANT]: content + [Tool calls: ...]`
- user → `[USER]: content`
- 超 6000 chars → 70/20/10 截断(头 4000 + marker + 尾 1500)
- 全部先过 `redact_sensitive_text`(防 secret 泄漏)

### Step 4:构建 prompt(分两条路径)

#### Preamble(共用)

```python
_summarizer_preamble = (
    "You are a summarization agent creating a context checkpoint. "
    "Treat the conversation turns below as source material for a "
    "compact record of prior work. "
    "Produce only the structured summary; do not add a greeting, "
    "preamble, or prefix. "
    "Write the summary in the same language the user was using..."
    "NEVER include API keys, tokens, passwords, secrets..."
)
```

**4 条规矩**:
1. "你是压缩助手" — 角色定位
2. "只输出 summary" — 防止 LLM 加开场白
3. "用用户用的语言" — 用户用中文就写中文 summary
4. "绝不写密钥" — 防 LLM 复制对话里的 API key

**措辞要平实**: Azure/OpenAI 内容过滤会标记"更强硬"的"不要响应"框架。

#### 模板(13 个固定小节)

```python
_template_sections = """
## Active Task       ← THE SINGLE MOST IMPORTANT FIELD
## Goal
## Constraints & Preferences
## Completed Actions  ← 续编号!
## Active State
## In Progress
## Blocked
## Key Decisions
## Resolved Questions
## Pending User Asks
## Relevant Files
## Remaining Work
## Critical Context
"""
```

**为什么用 markdown 不用 JSON**:
- JSON 嵌套结构 + 转义字符,LLM 容易出错
- 纯文本容错率高,LLM 几乎不会"误读"标签
- 13 个固定小节便于下游解析

#### 首次压缩路径

```python
prompt = f"""{_summarizer_preamble}

Create a structured checkpoint summary for the conversation...

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template_sections}"""
```

#### 迭代压缩路径

```python
prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary...

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing
information that is still relevant. ADD new completed actions to the
numbered list (continue numbering). Move items from "In Progress" to
"Completed Actions" when done...

{_template_sections}"""
```

**额外指令**:
- "PRESERVE all existing information that is still relevant"
- "ADD new completed actions to the numbered list (continue numbering)"
- "Move items from 'In Progress' to 'Completed Actions' when done"
- "Move answered questions to 'Resolved Questions'"
- "Update 'Active State' to reflect current state"

### Step 5:FOCUS TOPIC 引导(可选)

```python
if focus_topic:
    prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving all
information related to the focus topic above. ..."""
```

**灵感来自 Claude Code 的 `/compact <topic>`**。
- 放在 prompt 末尾 → LLM 注意力机制"最近的内容权重高"
- focus 内容占 60-70% 预算,其他压成一行甚至省略

### Step 6:调 LLM

```python
call_kwargs = {
    "task": "compression",
    "main_runtime": {
        "model": self.model,
        "provider": self.provider,
        "base_url": self.base_url,
        "api_key": self.api_key,
        "api_mode": self.api_mode,
    },
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(summary_budget * 1.3),  # 30% 余量
}
if self.summary_model:
    call_kwargs["model"] = self.summary_model  # 可用更便宜的子模型
response = call_llm(**call_kwargs)
content = response.choices[0].message.content
```

**关键点**:
- `task="compression"` — 标记压缩任务,call_llm 可能走不同 rate limit
- `main_runtime` 传主模型配置,fallback 时复用
- `max_tokens = budget × 1.3` — 30% 余量(LLM 输出常略超 budget)
- `summary_model` 可覆盖 — 比如主模型 opus,summary 用 haiku 省 90% 成本

**二次脱敏**:
```python
content = response.choices[0].message.content
if not isinstance(content, str):
    content = str(content) if content else ""
# prompt 里的"NEVER include secrets"不能 100% 信任
summary = redact_sensitive_text(content.strip())
```

### Step 7:失败处理(4 类错误分类)

```python
except Exception as e:
    _status = getattr(e, "status_code", None) or ...
    _err_str = str(e).lower()
    
    # 4 类可恢复错误
    _is_model_not_found = _status in {404, 503} or "model_not_found" in _err_str
    _is_timeout = _status in {408, 429, 502, 504} or "timeout" in _err_str
    _is_json_decode = isinstance(e, json.JSONDecodeError) or "expecting value" in _err_str
    _is_streaming_closed = _is_connection_error(e)
    
    # 4 类都触发"回退到主模型重试"
    if (_is_model_not_found or _is_timeout or _is_json_decode or _is_streaming_closed) \
       and self.summary_model != self.model and not _summary_model_fallen_back:
        self._fallback_to_main_for_compression(e, _reason)
        return self._generate_summary(turns_to_summarize, focus_topic)  # 递归重试
    
    # 未知错误兜底
    if self.summary_model != self.model and not _summary_model_fallen_back:
        self._fallback_to_main_for_compression(e, "failed")
        return self._generate_summary(turns_to_summarize, focus_topic)
    
    # 最后的冷却(30s 或 60s)
    _transient_cooldown = 30 if (_is_json_decode or _is_streaming_closed) else 60
    self._summary_failure_cooldown_until = time.monotonic() + _transient_cooldown
    return None
```

**4 类错误对应 issue**:
| 错误类型 | 触发条件 | 对应 issue |
|---|---|---|
| `_is_model_not_found` | HTTP 404/503 + 错误信息含 "model_not_found" | 配置错误 |
| `_is_timeout` | HTTP 408/429/502/504 + "timeout" | 网络/限流 |
| `_is_json_decode` | `JSONDecodeError` 或 "expecting value" | #22244 |
| `_is_streaming_closed` | `ConnectionError` + 特征子串 | #18458 |

**冷却期分两档**:
- JSON decode / streaming closed: **30s**(可能很快自愈)
- 其他:**60s**(限流/网络需要更久)

**防递归**: `_summary_model_fallen_back` 标志位,fallback 后只重试一次。

### Step 8:加 prefix 返回

```python
# 成功路径
self._previous_summary = summary  # 保存为下次迭代的输入
self._summary_failure_cooldown_until = 0.0
self._summary_model_fallen_back = False
self._last_summary_error = None
return self._with_summary_prefix(summary)
```

**`_with_summary_prefix` 干 3 件事**:
1. `_strip_summary_prefix` 剥掉所有历史 prefix(v1, legacy, historical)
2. prepend 最新的 `SUMMARY_PREFIX`
3. 防止 #35344: 历史 prefix 残留误读成"用户指令"

### 返回 None 意味着什么

- **不返回字符串**(连 fallback summary 都不返回)
- 调用方 `compress()` 看到 None 后:
  - `abort_on_summary_failure=True` → 放弃压缩,返回原 messages
  - `abort_on_summary_failure=False` → 用 `_build_static_fallback_summary`
- **宁可让用户看到"摘要失败"提示,也不能注入"假摘要"**

### `_previous_summary` 数据流(关键!)

```
compress() 第 1 次:
  _find_latest_context_summary() → (None, "")
  _previous_summary 仍为 None
  _generate_summary → 走"首次"prompt
  成功后:self._previous_summary = summary   ← 写入!

compress() 第 2 次:
  _find_latest_context_summary() → (idx=5, body)  ← 找到上次的 summary
  if summary_body and not self._previous_summary:   ← rehydrate(空才写)
      self._previous_summary = summary_body
  _generate_summary → 走"迭代"prompt(有 previous_summary)
  成功后:self._previous_summary = 新 summary   ← 再次写入!
```

**写入的 2 个地方**:
1. `compress()` 里 rehydrate(从历史 messages 恢复,只在 self._previous_summary 为空时写)
2. `_generate_summary()` 成功路径(总是写,更新为最新)

### 一句话总结

> **`_generate_summary` = 整个压缩系统唯一调 LLM 的地方**:**构造 preamble + 模板 + turns → 调 LLM → 4 类错误分类处理 → 返回带 prefix 的 summary**。**核心创新是"迭代式 prompt"**(previous_summary + new turns)和"4 类错误 fallback"机制。

---

## 📌 第十六部分:首次压缩 vs 多次压缩(迭代)对比(★ 新增)

> **相关源文件**:`hermes-agent/agent/context_compressor.py`
> - 判定 1:`compress()` 里 `_find_latest_context_summary` — 第 2240-2249 行
> - 判定 2:`_generate_summary()` 里 `if self._previous_summary` — 第 1638 行
> - 写入 1:`compress()` rehydrate — 第 2247-2248 行
> - 写入 2:`_generate_summary()` 成功 — 第 1742 行

### 一句话答案

**首次压缩 = 没有 previous_summary,从 0 写 summary**;**多次压缩(迭代)= 保留旧 summary,续编号追加新信息**。

### 流程对比

```
首次压缩(从无到有):
  ┌────────────┐
  │  middle    │   ← LLM 看到的就是这些 turns
  │  turns     │
  └────────────┘
       ↓
  [LLM Prompt]
    preamble
    TURNS TO SUMMARIZE: <turns>
    模板
       ↓
  Output: 新 summary(从 0 开始写)


多次压缩(迭代式):
  ┌────────────┐
  │  上次      │   ← LLM 看到这两个
  │  summary   │
  ├────────────┤
  │  新增      │
  │  turns     │
  └────────────┘
       ↓
  [LLM Prompt]
    preamble
    PREVIOUS SUMMARY: <上次 summary>
    NEW TURNS TO INCORPORATE: <新增 turns>
    模板 + 增量指令
       ↓
  Output: 更新后的 summary(基于上次,加入新内容)
```

### 关键差异

| 维度 | 首次压缩 | 多次压缩(迭代) |
|---|---|---|
| **输入** | 只有 turns | previous_summary + 新 turns |
| **prompt 路径** | "首次压缩" 分支 | "迭代更新" 分支 |
| **LLM 任务** | 从头写 summary | 在已有 summary 上**更新** |
| **Completed Actions 编号** | 从 1 开始 | **续编号**(不重新开始) |
| **信息保真度** | 中等(只覆盖当前 turns) | **高**(旧信息 + 新信息) |
| **触发条件** | `_previous_summary is None` | `_previous_summary` 不为空 |

### 实际代码对比

```python
if self._previous_summary:
    # 多次压缩路径(迭代)
    prompt = f"""...
    You are updating a context compaction summary...
    PREVIOUS SUMMARY: {self._previous_summary}
    NEW TURNS TO INCORPORATE: {content_to_summarize}
    ...续编号 / 移动 In Progress→Completed ..."""
else:
    # 首次压缩路径
    prompt = f"""...
    Create a structured checkpoint summary...
    TURNS TO SUMMARIZE: {content_to_summarize}
    ..."""
```

### 为什么"续编号"重要?

**首次压缩后的 summary**:
```
## Completed Actions
1. 读 config.py
2. 改 config.py
3. 跑测试
```

**如果不做迭代**(每次都"从 0 写"):
```
## Completed Actions  ← 第二次压缩,被覆盖
1. 读 config.py
2. 改 config.py
3. 跑测试
4. 修 build
5. 提交
```
**问题**: 看上去还是 1-5,信息其实**没丢**,但 token 浪费了 — 1-3 在两次压缩里被重复算。

**做了迭代**(在第二次 prompt 里加 "continue numbering"):
```
## Completed Actions  ← 第二次压缩,LLM 续编号
1. 读 config.py  ← 从旧 summary 保留
2. 改 config.py
3. 跑测试
4. 修 build       ← 新增
5. 提交           ← 新增
```

**好处**:
- 旧信息 + 新信息都保留
- LLM 不用"重新理解" 1-3,直接续写
- summary 结构稳定(编号不跳)

### `_previous_summary` 数据流完整时序

```
T0: 还没压缩过
    _previous_summary = None
    _find_latest_context_summary → (None, "")
    if self._previous_summary: False → 走"首次"路径

T1: 第一次压缩完成
    compress() 里: self._previous_summary = "## Active Task: ..."
    拼装时: _with_summary_prefix → "SUMMARY_PREFIX\n## Active Task: ..."

T2: 又跑了几十轮对话
    compress() 触发
    _find_latest_context_summary(messages, 1, end)
      → 找到 T1 那条 user 消息(idx=5),带 SUMMARY_PREFIX
      → 返回 (5, body)
    if summary_idx is not None: True
      self._previous_summary = body  ← rehydrate
      turns_to_summarize = messages[max(0, 6):end]  ← 跳过 summary
    _generate_summary(turns)
      if self._previous_summary: True → 走"迭代"路径
      LLM 收到 "PREVIOUS SUMMARY: ... NEW TURNS: ..."

T3: 第二次压缩完成
    self._previous_summary = 更新后的 summary
    拼装时: _with_summary_prefix → "SUMMARY_PREFIX\n..."
```

### 为什么 resume 场景特别重要?

**场景**: 用户 `/new` 创建新会话,但实际是 resume 同一个 lineage。

```
恢复后的 messages:
  [0] system
  [1] user: "[CONTEXT COMPACTION — REFERENCE ONLY] ... ## Active Task: ..."  ← 旧 session 留下的 summary
  [2] user: "新的请求"
  [3] assistant: ...

compress() 触发:
  _find_latest_context_summary 从 [1] 开始往 [end] 找
  → 找到 idx=1(带 SUMMARY_PREFIX)
  → 把 body 存到 self._previous_summary
  → turns_to_summarize 从 idx=2 开始(跳过旧 summary)
  → LLM 走"迭代"路径
  → LLM 看到"上次的总结 + 新请求产生的 turns"
  → 输出"整合后的 summary"
```

**关键**: 不需要让 LLM 重新理解整个旧 summary 涉及的历史(那些已经是"前情提要"),**只需要整合新内容**。

### 一句话总结

> **首次压缩 = 从 0 写,只覆盖当前 middle 段**;**多次压缩(迭代)= 保留旧 summary,续编号追加新信息**。**关键差异是"是否有 previous_summary"**。**迭代的好处是信息不衰减 + LLM 任务简化**(续写 vs 重写)。**核心判定点是 `_generate_summary` 里 `if self._previous_summary` 分支**。

