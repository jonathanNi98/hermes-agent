# 核心模块学习 — Agent Loop

## 1. 这个模块解决什么问题

**问题**：Agent 如何处理用户输入、反复调用模型、执行工具，直到任务完成？

**答案**：`agent/conversation_loop.py` 中的 `run_conversation()` 实现了这个主循环。

---

## 2. 真实源码位置（已验证）

```
agent/conversation_loop.py      ← 主循环（约3900行，已从 run_agent.py 提取）
run_agent.py                   ← AIAgent 入口类（约5600行）
                                run_conversation 现在是 thin forwarder
```

**关键发现**：`run_conversation()` 已从 `run_agent.py` 提取到 `agent/conversation_loop.py`。原 `run_agent.py` 中的实现是 thin forwarder，只做转发。这是 2025 年的重构结果。

---

## 3. 核心类 / 函数 / 方法（均已验证）

```python
# agent/conversation_loop.py
def run_conversation(agent, user_message: str) -> str:
    """主循环入口函数，约3900行"""

# 内部关键函数
def _build_messages(agent, user_message) -> List[dict]   # 构建 API messages
def _make_api_call(agent, messages) -> dict              # 模型调用（含重试）
def _execute_tool_calls_sequential(agent, tool_calls)     # 顺序执行工具
def _execute_tool_calls_concurrent(agent, tool_calls)     # 并发执行工具（ThreadPoolExecutor, max=8）
def _restore_or_build_system_prompt(agent)               # 恢复或构建 system prompt
def _compress_if_needed(agent)                           # 上下文压缩

# agent/tool_executor.py（已从 run_agent 提取）
def _execute_tool_calls_sequential(agent, tool_calls)
def _execute_tool_calls_concurrent(agent, tool_calls)

# run_agent.py
class AIAgent:
    def run_conversation(self, user_message: str) -> str:
        return conversation_loop.run_conversation(self, user_message)  # thin forwarder
```

---

## 4. 调用链

```
用户输入
  │
  ▼
cli.py / gateway/run.py
  │  (创建 AIAgent 实例)
  ▼
run_agent.py: AIAgent.run_conversation()
  │  (thin forwarder，2025年重构)
  ▼
agent/conversation_loop.py: run_conversation()
  │
  ├─► _restore_or_build_system_prompt()
  │       └─► system_prompt.py: build_system_prompt_parts()
  │               ├─► _build_stable()   → SOUL.md, 工具指导, skills
  │               ├─► _build_context()  → system_message, 文件
  │               └─► _build_volatile() → memory, timestamp
  │
  ├─► memory_manager.prefetch_all(user_message)
  │
  ├─► while not done (api_call_count < max_iterations)
  │       │
  │       ├─► _build_messages()
  │       │       ├─► system_prompt
  │       │       ├─► history messages
  │       │       ├─► memory_context
  │       │       └─► tools_schema
  │       │
  │       ├─► _make_api_call()
  │       │       └─► Provider Adapter (Anthropic/OpenAI/Gemini/etc.)
  │       │
  │       ├─► 解析 response.tool_calls
  │       │
  │       ├─► _execute_tool_calls_sequential/concurrent()
  │       │       └─► registry.dispatch(name, args)
  │       │               └─► tools/<name>.py handler
  │       │
  │       ├─► 回填工具结果 → messages
  │       │
  │       ├─► 错误处理 / 重试 / fallback
  │       │       └─► error_classifier.py: classify_api_error()
  │       │
  │       └─► _compress_if_needed()
  │               └─► context_compressor.py: should_compress() / compress()
  │
  └─► memory_manager.sync_all()
```

---

## 5. 输入和输出

```
输入：
  - user_message: str（用户输入）

输出：
  - final_response: str（最终文本响应）

副作用：
  - self.messages 历史被修改（添加 user/assistant/tool 消息）
  - self._memory_manager 可能写入记忆
  - 可能触发上下文压缩
  - 可能创建子代理（通过 delegate_task）
  - 可能触发 session 持久化（hermes_state.py）
```

---

## 6. 和其他模块的关系

```
Agent Loop 依赖：
  ├─► system_prompt.py       ← 获取 system prompt
  ├─► memory_manager.py      ← prefetch/sync 记忆
  ├─► tool_executor.py       ← 执行工具
  ├─► model_tools.py         ← 获取工具 schema
  └─► providers/*_adapter.py ← 模型调用

其他模块依赖 Agent Loop：
  ├─► cli.py                 ← 创建 AIAgent 调用 run_conversation
  ├─► gateway/run.py         ← 同上
  └─► batch_runner.py        ← 同上
```

---

## 7. 设计亮点

### 亮点 1：thin forwarder 模式（2025 重构）
```python
# run_agent.py
def run_conversation(self, user_message: str) -> str:
    return conversation_loop.run_conversation(self, user_message)
```
真正的逻辑在 `conversation_loop.py`，`run_agent.py` 只是转发。测试可以 patch `conversation_loop.run_conversation`。

### 亮点 2：Sequential + Concurrent 双模式
```python
# agent/tool_executor.py
_MAX_TOOL_WORKERS = 8  # 最大并发数

def _execute_tool_calls_concurrent(agent, tool_calls):
    with ThreadPoolExecutor(max_workers=_MAX_TOOL_WORKERS) as executor:
        futures = [executor.submit(execute_one, tc) for tc in tool_calls]
        results = [f.result() for f in futures]
```
支持并行工具调用提升效率，互不依赖的工具可以同时执行。

### 亮点 3：多层重试和 fallback
```python
# agent/error_classifier.py
class FailoverReason(Enum):
    RATE_LIMIT
    CONTEXT_OVERFLOW
    TIMEOUT
    MODEL_UNAVAILABLE
    ...
```
API 调用失败后，`classify_api_error()` 判断原因，触发对应重试策略或模型切换。

### 亮点 4：IterationBudget 控制迭代次数
```python
# agent/iteration_budget.py
class IterationBudget:
    """防止无限循环的 Budget 控制"""
```
每个 turn 有最大迭代次数，防止工具调用死循环。

---

## 8. 风险和不足

- **约 3900 行**：即使提取了，函数本身依然巨大，单一职责原则违反
- **状态管理分散**：在 `AIAgent` 实例属性中管理，`conversation_loop.py` 通过 `_ra()` 间接访问
- **测试覆盖相对较少**：核心循环的测试相比工具系统少

---

## 9. 最小实现伪代码

```python
def run_conversation(agent, user_message: str) -> str:
    """极简版 Agent Loop"""

    # 1. 构建初始消息
    messages = [
        {"role": "system", "content": agent.system_prompt},
        *agent.history,
        {"role": "user", "content": user_message},
    ]

    # 2. 主循环
    while agent.api_call_count < agent.max_iterations:
        # 调用模型
        response = agent.provider.chat(messages)

        # 检查是否有工具调用
        if not response.tool_calls:
            # 3. 最终响应
            agent.history.append({"role": "assistant", "content": response.content})
            return response.content

        # 4. 执行工具
        for tool_call in response.tool_calls:
            result = agent.tool_executor.execute(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
            })
            agent.history.append({
                "role": "tool",
                "content": result,
                "name": tool_call.function.name,
            })

        # 5. 追加 assistant 消息（携带 tool_calls）
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls,
        })

    return "Max iterations reached"
```

---

## 10. 练习题

### 练习 1：追踪一条用户消息的完整路径（入门）
```
目标：亲手画出消息从输入到 API 调用的完整路径

步骤：
1. 打开 cli.py，找到创建 AIAgent 的位置
2. 追踪到 run_conversation() 的调用
3. 在 conversation_loop.py 的 run_conversation() 开头加 print
4. 启动 hermes，发送一条简单消息
5. 观察日志输出的顺序

产出物：调用链草图（手画或 ASCII 图）
```

### 练习 2：观察工具调用循环（进阶）
```
目标：理解 while not done 循环的执行次数和退出条件

步骤：
1. 发送一个需要工具的任务（如"读取 /tmp/test.txt"）
2. 观察循环执行了几次
3. 每次循环中 messages 列表怎么变化
4. 追踪 max_iterations 是在哪里检查的

产出物：循环次数分析 + messages 状态变化表
```

### 练习 3：在 Loop 中插入自定义 Hook（高级）
```
目标：不修改原代码，在外部观察 Loop 行为

步骤：
1. 在 _make_api_call() 前后插入计时逻辑
2. 记录每次 API 调用的耗时
3. 在工具执行前后记录日志（用 logger.info）
4. 分析耗时分布

产出物：性能分析报告（哪些工具最慢？API 调用占比？）
```
