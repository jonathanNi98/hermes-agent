# 学习笔记 5 — Day 8:子 agent 构造

> 日期:2026-06
> 主题:`tools/delegate_tool.py` 子 agent 构造相关函数
> 范围:Day 8 — Delegate / Subagent

---

# 1. `_build_child_system_prompt`

## 🎯 一句话

> **给子 agent 拼一段"专注的任务"system prompt**。5 段固定结构 + 1 段可选(orchestrator 专属)。

## 📋 6 段构造流程

```
parts = [
    "段 1) 身份:You are a focused subagent working on a specific delegated task.",
    "",
    "段 2) 任务:YOUR TASK: {goal}",
    "段 3) 背景(可选):CONTEXT: {context}",
    "段 4) 工作目录(可选):WORKSPACE PATH: {workspace_path}",
    "段 5) 汇报要求:做完了要说 4 件事(做了什么/找到什么/改了哪些文件/遇到什么问题)",
]
# 段 6) role == "orchestrator" 才追加
#    WHEN to delegate / WHEN NOT to delegate
#    depth 数字 + max_spawn_depth 数字(真实值,不幻觉)
return "\n".join(parts)
```

## 🧱 段段拆解

### 段 1 — 身份(固定)

```python
"You are a focused subagent working on a specific delegated task."
```

**作用**:把子从"通用 agent"重新定位成"专注执行这一个任务的工人"。**没有"你是 Hermes" / "你可以..."** 这种开放描述 —— 故意窄化。

### 段 2 — 任务(必填)

```python
f"YOUR TASK:\n{goal}"
```

**作用**:直接告诉子"你要做什么"。**`goal` 是 LLM 调 delegate_task 时传的**(已经从父 conversation 抽象出来)。

### 段 3 — 背景(可选)

```python
if context and context.strip():
    parts.append(f"\nCONTEXT:\n{context}")
```

**作用**:子 agent **看不到父的 conversation history**,所以需要父把相关上下文**手动**塞进 `context` 参数。

**为什么 strip()**:防御性 —— 防止 LLM 传 `"   "` 这种纯空白,把它当成"有 context"但实际为空。

### 段 4 — 工作目录(可选 + 严格)

```python
if workspace_path and str(workspace_path).strip():
    parts.append(
        "\nWORKSPACE PATH:\n"
        f"{workspace_path}\n"
        "Use this exact path for local repository/workdir operations..."
    )
```

**关键设计**:
- 只在 `workspace_path` 是**真实存在的绝对路径**时才追加(由 `_resolve_workspace_hint` 在 7.2 保证)
- 如果探测不到 → **不写这个段**,而不是写个假的 `/workspace/...`

### 段 5 — 汇报要求(固定)

```
- What you did
- What you found or accomplished
- Any files you created or modified
- Any issues encountered

Important workspace rule: Never assume a repository lives at /workspace/...

Be thorough but concise -- your response is returned to the parent agent as a summary.
```

**作用**:这 4 项是给父 agent 看的 **summary 模板**,子 agent 的输出会被父用 conversation_loop 接收,所以**必须**按这个结构汇报。

### 段 6 — Orchestrator 专属(可选)

只在 `role='orchestrator'` 时追加。包含:
- `WHEN to delegate` — 2+ 独立并行 / reasoning-heavy
- `WHEN NOT to delegate` — 单步机械 / 一两个工具能搞定 / pass-through
- "Coordinate and synthesize before reporting back" — **你是责任主体,不是 worker**
- `depth 数字 + max_spawn_depth 数字` — **真实值**,由 config 算出来

## 🧠 设计要点

| 设计 | 价值 |
|---|---|
| **窄化身份**("focused subagent") | 子不会"主动发挥"做父没让做的事 |
| **背景靠显式 `context` 参数** | 强制父 LLM 思考"哪些信息子真的需要" |
| **workspace_path 必须真实存在才写** | 防止教子假路径(`/workspace/...`),TUI 不一致 |
| **汇报要求固定 4 项** | 父拿到的 summary 结构稳定,容易处理 |
| **depth 数字传真实值,不靠 LLM 推算** | 防止子自信地说"我还能再 spawn 5 层"(实际不行) |
| **Orchestrator 段追加而非替换** | 普通段的"汇报要求"对 orchestrator 也适用 |

## 🔀 Role 行为对比

| role | 段数 | 第 6 段(depth note)内容 |
|---|---|---|
| `leaf` | 5 段 | 无 |
| `orchestrator` 且 `child_depth + 1 < max_spawn_depth` | 6 段 | "Your own children can themselves be orchestrators or leaves..." |
| `orchestrator` 且 `child_depth + 1 >= max_spawn_depth` | 6 段 | "Your own children MUST be leaves..." |

**关键**:`child_depth + 1 >= max_spawn_depth` 判断的是"我的下一层就是 leaf floor"。

## 🔑 关键代码

```python
# 7.1 核心:role + depth 判断决定第 6 段内容
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
```

**为什么 `child_depth + 1`**:算的是"我的下一层",不是"我这一层"。

## 📍 调用链

```
delegate_task()  [12.x]
   ↓
_build_child_agent()  [9.5]
   ↓ 调
_build_child_system_prompt(...)
   ↓ workspace_path 来自
_resolve_workspace_hint(parent_agent)  [7.2]
   ↓ 真实存在的绝对路径 or None
```

---

# 2. `_build_child_agent`

## 🎯 一句话

> **构造一个子 AIAgent 实例**(返回,不跑)。构造完才丢给 `_run_single_child` 在 worker thread 里跑。

## 📋 11 步构造流程(主线程串行)

```
9.1  Role 解析         role + kill switch + depth 双重护栏,降级到 'leaf'
9.2  Subagent 身份     生成 sa-{task_index}-{8hex} + parent_subagent_id
9.3  父 toolset 推导   3 种来源:enabled_toolsets / 反推 valid_tool_names / 默认
9.4  子 toolset 计算   intersect(父) + 补 MCP + strip 黑名单
9.5  写 system prompt  调 7.1 + 7.2 拿 workspace_path
9.6  进度回调构造      调 8.x,subagent_id 嵌进每条事件
9.7  子循环预算        max_iterations(config 权威,忽略 caller 传的)
9.8  思考回调          包装 child_progress_cb 成 _thinking
9.9  凭证解析          override > 父继承(10 个字段逐项合并)
9.10 Reasoning effort  config override > 父继承
9.11 Fallback chain    继承父的 _fallback_chain
9.12 Provider filters  override 时清空(防 OpenRouter only=[...] 拉回)
9.13 真正构造 AIAgent  把上面所有塞进构造器
9.14 后置绑定          _print_fn / _delegate_depth / _subagent_id
9.15 凭证池共享        1) 同 provider 共享 2) 不同加载自己的 3) 没 None
9.16 注册 _active_children(给 interrupt 传播)
9.17 立刻宣告 spawn    spawn_requested 事件给 TUI
```

**主线程串行** —— 9.13 调 `AIAgent()` 会改 `model_tools._last_resolved_tool_names`(全局)。所以**不能并发构造**。

## 🧠 设计要点

| 设计 | 价值 |
|---|---|
| **主线程串行构造** | 防 `model_tools._last_resolved_tool_names` 全局污染 |
| **Role 双重护栏**(kill switch + depth) | "能不能 spawn"集中到一处决定,逻辑可预测 |
| **toolset 三层过滤**(intersect / MCP / strip) | 子不可能"获得父没有的"工具 |
| **api_mode 不在 provider 变时继承** | 修 #20558 — 子用 chat_completions 打 Anthropic 端点会 404 |
| **override_provider 时清空 ACP** | 修 #16816 — 不清会绕过 override 凭证 |
| **max_iterations 忽略 caller,只用 config** | 防 LLM 自己缩小预算让用户懵 |
| **subagent_id 在构造期就生成** | 9.2 → 9.6 → 9.16 → 9.17 全用同一个 key |
| **立刻宣告 spawn_requested** | 子可能排队几秒,TUI 节点不能等 run 开始才显示 |

## 🔑 关键代码

### 9.1 Role 解析

```python
# 唯一一处把 role 强制降级成 'leaf' 的地方
child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
max_spawn = _get_max_spawn_depth()
orchestrator_ok = _get_orchestrator_enabled() and child_depth < max_spawn
effective_role = role if (role == "orchestrator" and orchestrator_ok) else "leaf"
```

### 9.2 Subagent 身份

```python
subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
tui_depth = max(0, child_depth - 1)  # 0 = 第一层子 for the UI
```

**`subagent_id`** 格式 `sa-<task_index>-<8hex>`,被这些地方共用:
- 进度回调(每条事件都带)
- spawn_requested 事件
- `_active_subagents` 注册表(TUI 用来定位和 kill)
- 文件状态协调 `_current_task_id`

## 📍 调用链

```
LLM 调 delegate_task(tasks=[A, B, C])
   ↓
for i, t in enumerate(task_list):
   ↓
_build_child_agent(task_index=i, ...)            ← 9.x 主线程
   ↓
_run_single_child(i, goal, child, parent_agent)  ← 11.x worker thread
```

---

# 3. `delegate_task`(12.x)

## 🎯 一句话

> **顶层入口**,LLM 调 `delegate_task(...)` 时实际执行的函数。负责**守门 + 编排 + 收尾**,不亲自跑子 agent。

## 📋 完整流程(12.1 ~ 12.11)

```
12.2  入口守卫
      ├─ parent_agent 不能为 None
      └─ is_spawn_paused() → fail-fast "spawning paused"
12.3  加载 config + 规范化 role
      ├─ 读 cfg(get "delegation" 段)
      ├─ 忽略 caller 传的 max_iterations(用 config 权威值)
      ├─ _resolve_delegation_credentials(cfg, parent_agent)
      │   ├─ 路径 A:配置了 base_url → 直连 OpenAI 兼容端点
      │   ├─ 路径 B:配置了 provider  → 走 runtime provider 解析
      │   └─ 路径 C:啥都没          → 返 None 全字段,子继承父
      └─ depth ≥ max_spawn → fail-fast "depth limit reached"
12.4  规范化 task list
      ├─ 优先 tasks(可能是 JSON 字符串,_recover_tasks_from_json_string 解析)
      ├─ 其次 goal 包成 [{goal, context, toolsets, role}]
      └─ 都没有 → fail-fast
12.5  构造所有子 agent(主线程串行)
      ├─ 保存父的 _last_resolved_tool_names(防子构造污染)
      ├─ for i, t in enumerate(task_list): 串行 _build_child_agent
      │   └─ child._delegate_saved_tool_names = _parent_tool_names
      └─ finally: _last_resolved_tool_names = _parent_tool_names(权威还原)
12.6  执行
      ├─ 单任务:result = _run_single_child(0, ...)
      └─ 批任务:
          ├─ ThreadPoolExecutor(max_workers=max_children) 起 N 个 worker
          ├─ submit(_run_single_child, ...) 每个 worker 跑一个
          └─ while pending: _cf_wait(0.5s) 轮询(可响应父 interrupt)
12.7  通知父的 _memory_manager(可选)
12.8  触发 subagent_stop hook + 累加 _children_cost_total
12.9  把子成本合并到父 session(_children_cost_total → parent.session_estimated_cost_usd)
12.10 拼最终 JSON 返给 LLM
      {"results": [...], "total_duration_seconds": ...}
```

## 🧠 设计要点

| 设计 | 价值 |
|---|---|
| **任何错误都返 tool_error JSON,不抛异常** | LLM 收到的是可解析的错误,不会让父 conversation 崩 |
| **`max_iterations` 忽略 caller 传的** | config 权威 —— 防 LLM 缩小预算让用户懵 |
| **凭证覆盖三条独立路径**(base_url / provider / 继承) | 灵活支持直连端点 / 切 provider / 透明继承 |
| **`is_spawn_paused` 早返** | TUI 全局闸门,父被 interrupt 时立刻停新 spawn |
| **构造串行 + 执行并发** | 改全局的构造独占;独立的执行可并行 |
| **`while pending: _cf_wait(0.5s)` 轮询**(批任务) | 不阻塞,父被 interrupt 时能快速退出 |
| **`finally` 还原 `_last_resolved_tool_names`** | 任一子构造抛异常也能还原全局 |
| **`subagent_stop` hook 在父线程串行触发** | plugin 作者不用考虑并发 |
| **子成本递归 rollup**(可加性) | 嵌套 orchestrator→worker 树正确合并到父 |
| **`tool_progress_callback._flush()` 在成功路径调** | 最后几个 tool 名不丢 |

## 🔑 关键代码

### 12.4 规范成 task list

```python
if tasks and isinstance(tasks, list):
    task_list = tasks
elif goal and isinstance(goal, str) and goal.strip():
    task_list = [{"goal": goal, "context": context, "toolsets": toolsets, "role": top_role}]
else:
    return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")
```

### 12.5 try/finally 还原全局

```python
try:
    for i, t in enumerate(task_list):
        child = _build_child_agent(...)
        child._delegate_saved_tool_names = _parent_tool_names
        children.append((i, t, child))
finally:
    _model_tools._last_resolved_tool_names = _parent_tool_names
```

### 12.6 批任务轮询

```python
while pending:
    if getattr(parent_agent, "_interrupt_requested", False) is True:
        # 父被中断 → 给还在跑的标 "interrupted" → break
        ...

    from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED
    done, pending = _cf_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
    for future in done:
        entry = future.result()
        results.append(entry)
        # ... 打印完成行 ...
```

### 12.9 子成本合并到父

```python
if _children_cost_total > 0.0:
    parent_agent.session_estimated_cost_usd = current + _children_cost_total
    # cost_source / cost_status 升级(避免 UI 显示 "none")
    if getattr(parent_agent, "session_cost_source", "none") in {None, "", "none"}:
        parent_agent.session_cost_source = "subagent"
```

**为什么是可加性**:嵌套 orchestrator→worker 时,每层 delegate_task 都把自己直接子 fold 进自己,父再 fold orchestrator(已含子的子)。层层累加,正确反映整棵树的总成本。

## 📍 跟其他函数的关系

```
LLM 调 delegate_task(...)              ← 你现在看的位置
   ↓
_build_child_agent(task_index, ...)    ← 见 #2
   ↓
_build_child_system_prompt(goal, ...)  ← 见 #1
   ↓
_run_single_child(i, goal, child, ...) ← 真正跑(下个学习笔记看)
```

## 🎯 一句话

> **`delegate_task` = "组织者"**。它**不亲自跑**任何子 agent,只做:守门 → 规范参数 → 构造 → 调度执行 → 收尾(hook / cost / JSON)。所有"做实事"的部分都委托给 `_build_child_agent` / `_run_single_child`。

---

# 新增知识点

(以后学到的相关知识,加在这里)