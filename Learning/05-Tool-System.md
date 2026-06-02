# 核心模块学习 — Tool Registry + Tool Executor

## 1. 这个模块解决什么问题

**问题**：80+ 工具如何注册、发现、分发、执行？如何让模型知道有哪些工具可用？

**答案**：工具通过 `registry.register()` 自注册，模型通过 `get_tool_definitions()` 获取 schema，调用时通过 `registry.dispatch()` 分发。

---

## 2. 真实源码位置（已验证）

```
tools/registry.py                  ← 工具注册中心（590行）
model_tools.py                     ← 工具 API 统一入口（1400+行）
agent/tool_executor.py             ← 工具执行器（从 run_agent 提取）
tools/<name>.py                    ← 各工具实现（80+个）
```

**重要发现**：
- `tools/registry.py` 使用 AST 静态扫描发现自注册工具，不需要运行时 import
- `model_tools.py` 是统一的 API 入口，暴露 `get_tool_definitions()` 等方法
- `tool_executor.py` 已从 `run_agent.py` 提取，支持顺序和并发两种执行模式

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# tools/registry.py
class ToolEntry:
    """工具元数据"""
    name: str
    schema: dict
    handler: Callable
    check_fn: Callable  # 可用性检查
    toolset: str

class Registry:
    def register(name, schema, handler, toolset, check_fn=None)
    def get_definitions(tool_names) -> List[dict]    # 获取 schema
    def dispatch(name, args) -> str                  # 执行工具
    def get_entry(name) -> ToolEntry
    def get_all_entries() -> List[ToolEntry]
    def discover_builtin_tools()                       # AST 扫描发现

def discover_builtin_tools(tools_dir) -> List[str]    # 模块级函数

# model_tools.py
def get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> list
def handle_function_call(function_name, function_args, task_id, user_task) -> str
def get_all_tool_names() -> list
def get_toolset_for_tool(name) -> str

# agent/tool_executor.py
_MAX_TOOL_WORKERS = 8
def _execute_tool_calls_sequential(agent, tool_calls) -> List[dict]
def _execute_tool_calls_concurrent(agent, tool_calls) -> List[dict]
```

---

## 4. 调用链

```
工具注册阶段（模块加载时）：
tools/<name>.py（模块级代码）
  │
  └─► registry.register(
          name="xxx",
          schema={...},
          handler=xxx_handler,
          toolset="xxx"
      )
  │
  ▼
tools/registry.py: Registry 实例注册到全局 registry 对象

模型获取工具 schema：
model_tools.get_tool_definitions(enabled_toolsets=...)
  │
  └─► registry.get_definitions(tool_names)
          └─► 过滤、返回 schema 列表
  │
  ▼
注入 API 请求的 tools 参数

工具调用阶段：
API 返回 tool_calls
  │
  ▼
conversation_loop.py: _execute_tool_calls_sequential/concurrent()
  │
  └─► registry.dispatch(name, args)
          │
          ├─► tool_entry.check_fn()  # 可用性检查
          └─► tool_entry.handler(args)  # 执行
                  │
                  ▼
          tools/<name>.py: handler 函数
                  │
                  ▼
          return json.dumps(result)
```

---

## 5. 输入和输出

```
注册阶段：
  输入：name, schema, handler, toolset, check_fn
  输出：ToolEntry 添加到 registry.entries

获取 schema：
  输入：enabled_toolsets, disabled_toolsets
  输出：List[dict]，每个 dict 包含 tool 的 schema

执行工具：
  输入：tool_name: str, args: dict
  输出：json.dumps(result): str

工具 handler 输入/输出：
  输入：args: dict（JSON 反序列化后的参数字典）
  输出：json.dumps(result) 字符串
```

---

## 6. 和其他模块的关系

```
Tool Registry 依赖：
  ├─► tools/<name>.py          ← 各工具 handler
  └─► toolsets.py             ← 工具集定义

其他模块依赖 Tool Registry / Executor：
  ├─► conversation_loop.py     ← 获取 schema + 执行工具
  ├─► model_tools.py           ← 暴露统一 API
  ├─► cli.py                   ← 获取工具列表
  └─► batch_runner.py          ← 同 conversation_loop
```

---

## 7. 设计亮点

### 亮点 1：AST 静态扫描自注册
```python
# tools/registry.py：
def _module_registers_tools(module_path: Path) -> bool:
    """Only inspects module-body statements so that helper modules
    which happen to call registry.register() inside a function
    are not picked up."""
    tree = ast.parse(source)
    return any(_is_registry_register_call(stmt) for stmt in tree.body)
```
不需要运行时 import 所有工具模块，扫描 + import 模式。

### 亮点 2：ToolEntry check_fn TTL 缓存
```python
# tools/registry.py：
if entry.check_fn and should_check(now, entry):
    available = entry.check_fn()
    cache_check_result(entry.name, available, now + 30)  # 30s TTL
```
避免频繁探测可用性，30秒缓存。

### 亮点 3：Generation Counter 无锁并发
```python
# tools/registry.py：
# 工具注册递增 generation，读取时复制
# 遍历 entries 时不锁，mutation 时递增 generation
```
读多写少场景下的无锁设计。

---

## 8. 风险和不足

- **单例 Registry**：全局单例，在多进程环境下可能有问题
- **工具 schema 膨胀**：80+ 工具 schema 全部传给模型，可能影响上下文效率
- **handler 错误处理**：部分工具的错误处理不一致

---

## 9. 最小实现伪代码

```python
# ===== tool_registry.py =====
class ToolEntry:
    def __init__(self, name, schema, handler, toolset, check_fn=None):
        self.name = name
        self.schema = schema
        self.handler = handler
        self.toolset = toolset
        self.check_fn = check_fn

class Registry:
    def __init__(self):
        self.entries = {}

    def register(self, name, schema, handler, toolset, check_fn=None):
        self.entries[name] = ToolEntry(name, schema, handler, toolset, check_fn)

    def get_definitions(self, tool_names=None):
        result = []
        for name, entry in self.entries.items():
            if tool_names and name not in tool_names:
                continue
            result.append(entry.schema)
        return result

    def dispatch(self, name, args):
        entry = self.entries.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = entry.handler(args)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

registry = Registry()


# ===== tool_executor.py =====
def execute_tool_calls_sequential(tool_calls):
    results = []
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        result = registry.dispatch(name, args)
        results.append({"tool_call_id": tc.id, "result": result})
    return results


# ===== tools/calculator.py =====
def calculator_handler(args):
    expr = args.get("expression")
    result = eval(expr)  # 危险！仅示例
    return {"result": result}

registry.register(
    name="calculator",
    schema={
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    handler=calculator_handler,
    toolset="utility"
)
```

---

## 10. 练习题

### 练习 1：追踪一个工具的完整调用路径（入门）
```
目标：理解工具从注册到调用的完整流程

步骤：
1. 选择一个简单工具（如 calculator）
2. 追踪 registry.register() 的调用位置
3. 追踪 get_tool_definitions() 如何获取其 schema
4. 追踪 handle_function_call() 如何执行它

产出物：完整调用链图
```

### 练习 2：添加一个新工具（进阶）
```
目标：亲手添加一个工具到系统

步骤：
1. 在 tools/ 下创建 my_tool.py
2. 定义 schema 和 handler
3. 写 registry.register() 调用
4. 启动 hermes，用 /tools 确认工具出现
5. 调用它，确认能正常工作

产出物：新工具文件 + 调用测试结果
```

### 练习 3：实现并发工具执行（高级）
```
目标：理解 concurrent 模式的实现

步骤：
1. 找到 _execute_tool_calls_concurrent() 的实现
2. 分析 ThreadPoolExecutor 的使用
3. 找一个互不依赖的工具对（如 read_file + calculate）
4. 验证并发执行确实比顺序执行快

产出物：性能对比数据
```
