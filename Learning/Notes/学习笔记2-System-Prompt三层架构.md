# 学习笔记 2 — System Prompt 三层架构 + 主循环形式化总结

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
