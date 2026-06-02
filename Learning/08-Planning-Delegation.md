# 核心模块学习 — Delegation / Sub-agent / Cron

## 1. 这个模块解决什么问题

**问题**：复杂任务如何拆分给子 Agent 并行执行？定时任务如何调度？

**答案**：
- `delegate_tool.py` 实现子代理委派，支持单任务和批量并行模式
- `cron/scheduler.py` 实现定时任务调度，Gateway 每60秒调用 `tick()`

---

## 2. 真实源码位置（已验证）

```
tools/delegate_tool.py           ← 子代理委派（2801行）
tools/kanban_tools.py           ← 看板任务系统
tools/cronjob_tools.py          ← Cron 任务工具
cron/scheduler.py               ← Cron 调度器（tick）
cron/jobs.py                    ← Job 定义
tools/mixture_of_agents_tool.py ← MoA（混合智能体）
```

**重要发现**：
- `delegate_tool.py` 不需要专门的 Planner 类，逻辑嵌入在 `delegate_task_handler` 中
- 子代理通过 `ThreadPoolExecutor` 并行执行（最大8个 worker）
- 子代理使用独立 session_id，完全隔离父 Agent 的历史

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# tools/delegate_tool.py
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",  # 禁止递归 delegation
    "clarify",       # 禁止用户交互
    "memory",        # 禁止写共享记忆
    "send_message",  # 禁止跨平台副作用
    "execute_code",  # 子代理应 step-by-step 推理
])

def delegate_task_handler(args: dict) -> str:
    """单任务或批量委派"""
    goal = args.get("goal")
    tasks = args.get("tasks", [])      # 批量任务
    max_concurrent = args.get("max_concurrent_children", 3)
    role = args.get("role", "leaf")

def _build_child_agent(parent, goal, context, toolsets, role, task_id)
    """创建子 AIAgent 实例"""

def _run_single_child(child_task) -> dict:
    """在 ThreadPoolExecutor 中运行单个子代理"""

# tools/cronjob_tools.py
def cronjob_create_handler(args: dict) -> str
def cronjob_list_handler(args: dict) -> str
def cronjob_delete_handler(args: dict) -> str

# cron/scheduler.py
class CronScheduler:
    def tick(self):                    # 每60秒被 Gateway 调用
        """检查并执行到期的 cron jobs"""
```

---

## 4. 调用链

```
delegate_task_handler(args)
  │
  ├─► tasks 为空 → run_single_delegation(goal, role)
  │       │
  │       ├─► _build_child_agent()     # 创建独立 AIAgent
  │       │       ├─► 独立 session_id
  │       │       ├─► 继承受限 toolsets
  │       │       └─► 构建子代理 system prompt
  │       │
  │       └─► child.run_conversation(goal)
  │               └─► 返回结果摘要
  │
  └─► tasks 不为空 → run_batch_delegation(tasks, max_concurrent)
          │
          └─► ThreadPoolExecutor(max_workers=max_concurrent)
                  ├─► _run_single_child(task1)
                  ├─► _run_single_child(task2)
                  └─► _run_single_child(task3)
                  └─► 收集所有结果

Cron 调度：
gateway/run.py（每60秒）
  │
  └─► CronScheduler.tick()
          ├─► 读取 cron 表
          ├─► 检查到期 jobs
          └─► spawn_worker(job)
                  └─► 子进程执行 job
```

---

## 5. 输入和输出

```
delegate_task 单任务模式：
  输入：goal: str, role: str
  输出：{"status": "completed", "summary": "...", "api_calls": N}

delegate_task 批量模式：
  输入：tasks: List[dict], max_concurrent: int
  输出：List[dict]（每个子任务的结果）

cronjob_create：
  输入：schedule: str, command: str, task_id: str
  输出：{"status": "created", "task_id": "..."}

CronScheduler.tick：
  输入：无
  输出：触发到期的 jobs（副作用）
```

---

## 6. 和其他模块的关系

```
Delegation 依赖：
  ├─► AIAgent（创建子代理实例）
  ├─► conversation_loop.py（子代理运行）
  ├─► tool_executor.py（工具执行）
  └─► hermes_state.py（session 持久化）

Cron 依赖：
  ├─► cronjob_tools.py（job 定义）
  ├─► hermes_state.py（持久化）
  └─► gateway/run.py（触发 tick）

其他模块依赖 Delegation：
  ├─► conversation_loop.py（工具调用）
  └─► skills/（可能用 delegation 实现复杂 skill）
```

---

## 7. 设计亮点

### 亮点 1：完全隔离的子代理
```python
# tools/delegate_tool.py 注释：
# "Each child gets:
#   - A fresh conversation (no parent history)
#   - Its own task_id (own terminal session, file ops cache)
#   - A restricted toolset (configurable, with blocked tools always stripped)
#   - A focused system prompt built from the delegated goal + context"
```
子代理不继承父历史，防止污染。

### 亮点 2：ThreadPoolExecutor 并发控制
```python
with ThreadPoolExecutor(max_workers=max_concurrent_children) as executor:
    futures = [executor.submit(_run_single_child, task) for task in child_tasks]
    results = [f.result() for f in futures]
```
灵活控制并发数，平衡效率和资源。

### 亮点 3：DELEGATE_BLOCKED_TOOLS 隔离
```python
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",  # 无递归
    "clarify",        # 无用户交互
    "memory",         # 无共享记忆写入
    "send_message",   # 无跨平台副作用
    "execute_code",   # 推理而非脚本
])
```
即使父 Agent 传递了 toolsets，这些工具也会被移除。

### 亮点 4：Subagent 审批回调
```python
# tools/delegate_tool.py 注释：
# "Subagents run inside a ThreadPoolExecutor worker.
# The CLI's interactive approval callback is stored in tools/terminal_tool.py's
# threading.local(), so worker threads do NOT inherit it.
# Fix: install a non-interactive callback via ThreadPoolExecutor(initializer=)"
```
解决 worker 线程无法访问主线程 TUI 的问题。

---

## 8. 风险和不足

- **子代理隔离不完美**：文件系统路径隔离依赖配置，非系统级隔离
- **资源控制缺失**：没有 CPU/内存限制，子代理可能耗尽资源
- **结果聚合复杂**：多子代理结果聚合方式简单（仅串联合并）
- **错误处理**：子代理失败时父代理的恢复策略简单

---

## 9. 最小实现伪代码

```python
import concurrent.futures

def delegate_task_handler(args: dict) -> str:
    goal = args.get("goal")
    tasks = args.get("tasks", [])
    max_concurrent = args.get("max_concurrent_children", 3)

    if tasks:
        return json.dumps(run_batch_delegation(tasks, max_concurrent))
    else:
        return json.dumps(run_single_delegation(goal))


def run_single_delegation(goal: str, role: str = "leaf") -> dict:
    # 1. 创建子代理
    child = AIAgent(
        base_url=parent.base_url,
        api_key=parent.api_key,
        provider=parent.provider,
        model=parent.model,
        max_iterations=child_max_iters,
        enabled_toolsets=strip_blocked_tools(parent.toolsets),
        session_id=str(uuid.uuid4()),
    )

    # 2. 运行
    result = child.run_conversation(goal)

    # 3. 返回摘要
    return {
        "status": "completed",
        "summary": summarize(result),
        "api_calls": result.api_calls,
        "duration_seconds": result.duration,
    }


def run_batch_delegation(tasks: list, max_concurrent: int) -> list:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(run_single_delegation, t["goal"]) for t in tasks]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def strip_blocked_tools(toolsets):
    """移除 DELEGATE_BLOCKED_TOOLS"""
    blocked = DELEGATE_BLOCKED_TOOLS
    return {k: v for k, v in toolsets.items() if k not in blocked}
```

---

## 10. 练习题

### 练习 1：追踪子代理创建过程（入门）
```
目标：理解子代理和父代理的关系

步骤：
1. 在 delegate_task_handler() 加日志
2. 在 _build_child_agent() 加日志
3. 用 delegate_task 执行一个简单任务
4. 观察父子代理的参数差异

产出物：父子代理参数对比表
```

### 练习 2：实现并行任务拆解（进阶）
```
目标：理解批量 delegation 的并发控制

步骤：
1. 发送3个独立子任务给 delegate_task
2. 观察 ThreadPoolExecutor 的执行（加日志）
3. 验证确实是并行的（看时间戳）
4. 改变 max_concurrent，观察行为变化

产出物：并行执行时序图
```

### 练习 3：分析 cron 调度流程（高级）
```
目标：理解定时任务如何被触发

步骤：
1. 查看 cronjob_create_handler() 的保存逻辑
2. 查看 CronScheduler.tick() 的执行逻辑
3. 理解文件锁防止重复执行
4. 找一个实际 cron job 观察

产出物：Cron 调度流程图
```
