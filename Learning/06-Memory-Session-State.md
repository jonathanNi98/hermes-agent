# 核心模块学习 — Memory / Session / State

## 1. 这个模块解决什么问题

**问题**：Agent 如何跨对话记住信息？Session 状态如何持久化？记忆 provider 如何插拔？

**答案**：`MemoryManager` 编排多个 provider 实现 prefetch/sync 生命周期。`SessionDB` 用 SQLite WAL 持久化 session 历史。

---

## 2. 真实源码位置（已验证）

```
agent/memory_manager.py      ← 记忆管理器（654行）
agent/memory_provider.py     ← Provider 抽象基类（320行）
tools/memory_tool.py         ← 内置记忆工具
hermes_state.py              ← SQLite 状态持久化（5000+行）
```

**重要发现**：
- `MemoryManager` 是单一集成点，替换了之前分散在各 backend 的代码
- `MemoryProvider` 是 ABC，支持外部 provider（Honcho, Hindsight, Mem0 等）
- `hermes_state.py` 使用 SQLite WAL，支持 FTS5 全文搜索
- **只有一个外部 plugin provider 允许注册**（防止 schema 膨胀）

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# agent/memory_manager.py
class MemoryManager:
    def __init__(self)
    def add_provider(provider: MemoryProvider)     # 添加外部 provider
    def build_system_prompt() -> str              # 生成 system prompt 块
    def prefetch_all(query: str) -> str           # turn 前预取
    def sync_all(user_msg, assistant_response)    # turn 后同步
    def queue_prefetch_all(user_msg)             # 队列预取
    def get_tool_schemas() -> List[dict]          # 暴露工具 schema

def build_memory_context_block(query, memories)   # 工具函数

# agent/memory_provider.py
class MemoryProvider(ABC):
    @property
    def name(self) -> str                        # 'builtin', 'honcho', etc.

    # 核心生命周期
    @abstractmethod
    def is_available(self) -> bool               # 是否可用
    @abstractmethod
    def initialize()                              # 初始化
    @abstractmethod
    def system_prompt_block() -> str             # system prompt 文本
    @abstractmethod
    def prefetch(query: str) -> str              # 预取记忆
    @abstractmethod
    def sync_turn(user, assistant)               # 同步 turn
    @abstractmethod
    def get_tool_schemas() -> List[dict]          # 工具 schema
    @abstractmethod
    def handle_tool_call(args: dict) -> str      # 处理工具调用
    @abstractmethod
    def shutdown()                               # 清理

    # 可选钩子
    def on_turn_start(turn, message, **kwargs)
    def on_session_end(messages)
    def on_session_switch(new_session_id, **kwargs)
    def on_pre_compress(messages) -> str
    def on_memory_write(action, target, content, metadata=None)
    def on_delegation(task, result, **kwargs)

# hermes_state.py
class SessionDB:
    def __init__(self, db_path=DEFAULT_DB_PATH)
    def create_session(session_id, source, model, provider)
    def add_message(session_id, role, content, metadata=None)
    def get_session_messages(session_id) -> List[dict]
    def get_session(session_id) -> dict
    def update_session_metadata(session_id, **kwargs)
    def compress_session(session_id, summary, message_ids)
```

---

## 4. 调用链

```
AIAgent 初始化：
  MemoryManager()
    │
    ├─► add_provider(plugin_provider)   # 可选
    └─► 各 provider.initialize()

对话 turn 前：
  memory_manager.prefetch_all(user_message)
    │
    └─► 各 provider.prefetch(query)
            │
            └─► 返回记忆文本片段
    │
    └─► build_memory_context_block()  # 组合成 <memory-context> 块
    │
    ▼
注入 system_prompt 或 user message

对话 turn 后：
  memory_manager.sync_all(user_msg, assistant_response)
    │
    └─► 各 provider.sync_turn(user, assistant)

状态持久化：
  hermes_state.py: SessionDB
    │
    ├─► create_session()
    ├─► add_message()
    ├─► compress_session()
    └─► WAL 模式并发读写
```

---

## 5. 输入和输出

```
MemoryManager.prefetch_all：
  输入：user_message: str
  输出：memory_context_block: str（含 <memory-context> 标签）

MemoryManager.sync_all：
  输入：user_msg: dict, assistant_response: dict
  输出：无直接返回值（内部写入 provider）

SessionDB：
  输入：session_id, role, content, metadata
  输出：持久化到 SQLite

MemoryProvider.prefetch：
  输入：query: str（用户消息）
  输出：相关记忆文本

MemoryProvider.sync_turn：
  输入：user: dict, assistant: dict
  输出：无（内部写入）
```

---

## 6. 和其他模块的关系

```
Memory Manager 依赖：
  ├─► MemoryProvider ABC        ← 各 provider 实现
  ├─► tools/memory_tool.py      ← 内置记忆工具
  └─► hermes_state.py          ← session 持久化

其他模块依赖 Memory Manager：
  ├─► conversation_loop.py     ← prefetch_all / sync_all
  ├─► system_prompt.py         ← build_system_prompt_block
  └─► context_compressor.py    ← on_pre_compress 钩子
```

---

## 7. 设计亮点

### 亮点 1：单一集成点
```python
# agent/memory_manager.py 注释：
# "Single integration point in run_agent.py.
# Replaces scattered per-backend code with one manager
# that delegates to registered providers."
```
之前各 backend 分散逻辑，现在统一管理。

### 亮点 2：prefetch/sync 分离
```python
# turn 前 prefetch：延迟优化，用户无感知
# turn 后 sync：写入，不阻塞响应
```
读写分离，优化感知延迟。

### 亮点 3：Frozen Snapshot Pattern
```python
# tools/memory_tool.py 注释：
# "If the agent crashes mid-write, the snapshot on disk is always consistent.
# The working copy is the only in-memory structure."
```
防止崩溃导致的不一致状态。

### 亮点 4：外部 Provider 限制
```python
# Only ONE external plugin provider is allowed at a time
# attempting to register a second external provider is rejected
```
防止 schema 膨胀和冲突。

---

## 8. 风险和不足

- **外部 provider 只有一个**：多 provider 不能同时激活
- **SQLite WAL 限制**：网络文件系统（NFS/SMB）上 WAL 可能失败，已回退到 DELETE 模式
- **MemoryManager 654 行**：依然较大，可考虑进一步拆分

---

## 9. 最小实现伪代码

```python
class MemoryProvider(ABC):
    @property
    def name(self): pass

    @abstractmethod
    def is_available(self): pass

    @abstractmethod
    def prefetch(self, query: str) -> str:
        """Return relevant memories as text"""

    @abstractmethod
    def sync_turn(self, user: dict, assistant: dict):
        """Write turn to memory"""


class MemoryManager:
    def __init__(self):
        self.providers = []

    def add_provider(self, provider: MemoryProvider):
        self.providers.append(provider)

    def prefetch_all(self, query: str) -> str:
        blocks = []
        for p in self.providers:
            if p.is_available():
                block = p.prefetch(query)
                if block:
                    blocks.append(block)
        if blocks:
            return "<memory-context>\n" + "\n\n".join(blocks) + "\n</memory-context>"
        return ""

    def sync_all(self, user, assistant):
        for p in self.providers:
            if p.is_available():
                p.sync_turn(user, assistant)


# 内置实现
class BuiltinMemoryProvider(MemoryProvider):
    def __init__(self):
        self.memories = []

    def prefetch(self, query: str) -> str:
        # 简单词匹配
        relevant = [m for m in self.memories if any(w in m for w in query.split())]
        return "\n".join(relevant[-5:])  # 最近5条

    def sync_turn(self, user, assistant):
        self.memories.append(f"User: {user['content']}")
        if isinstance(assistant, dict):
            self.memories.append(f"Assistant: {assistant.get('content', '')}")
        else:
            self.memories.append(f"Assistant: {assistant}")
```

---

## 10. 练习题

### 练习 1：追踪记忆的读写时机（入门）
```
目标：理解记忆在 turn 中的读写时机

步骤：
1. 在 memory_manager.prefetch_all() 加日志
2. 在 memory_manager.sync_all() 加日志
3. 运行一个多 turn 对话
4. 观察读写时机和顺序

产出物：记忆生命周期时序图
```

### 练习 2：实现一个简单 Provider（进阶）
```
目标：理解 Provider ABC 的接口

步骤：
1. 实现 MemoryProvider ABC 的所有必需方法
2. 用 JSON 文件做后端存储
3. 注册到 MemoryManager
4. 测试 prefetch 和 sync

产出物：自定义 MemoryProvider 实现
```

### 练习 3：分析 SessionDB 的压缩流程（高级）
```
目标：理解上下文压缩时 session 状态如何变化

步骤：
1. 查看 hermes_state.py 的 compress_session() 实现
2. 触发一次上下文压缩
3. 观察 session 表的变化
4. 理解 parent_session_id 链

产出物：压缩流程分析 + SQLite 操作日志
```
