# 核心模块学习 — Context / Prompt Builder

## 1. 这个模块解决什么问题

**问题**：模型需要看到什么上下文？System prompt 如何构建才能既完整又不超出 token 限制？

**答案**：三层架构分离稳定内容、动态内容和易变内容，实现缓存友好和按需更新。

---

## 2. 真实源码位置（已验证）

```
agent/system_prompt.py    ← 三层架构 stable/context/volatile（408行）
agent/prompt_builder.py   ← Prompt 构建细节（1507行）
agent/context_engine.py   ← Context Engine 抽象基类（可插拔）
agent/context_compressor.py ← 默认压缩实现（约2500行）
```

**重要发现**：三层架构定义在 `system_prompt.py`，而 `prompt_builder.py` 包含更多构建细节。注意 `context_engine.py` 定义了可插拔的上下文引擎接口，默认实现是 `context_compressor.py`。

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# agent/system_prompt.py
def _build_stable(agent)      # 稳定层：SOUL.md, 工具指导, skills, 平台提示
def _build_context(agent)      # 上下文层：system_message, AGENTS.md, .cursorrules
def _build_volatile(agent)     # 易变层：memory, timestamp, session info
def build_system_prompt_parts(agent)  # 组合三层，返回 List[str]

# agent/prompt_builder.py
def build_system_prompt_parts()  # 与 system_prompt.py 配合
# 关键常量：
MEMORY_GUIDANCE
SKILLS_GUIDANCE
TOOL_USE_ENFORCEMENT_GUIDANCE
KANBAN_GUIDANCE
SESSION_SEARCH_GUIDANCE
...

# agent/context_engine.py
class ContextEngine(ABC):      # 可插拔的上下文引擎基类
    def name(self) -> str
    def on_session_start()
    def update_from_response()
    def should_compress() -> bool
    def compress(messages) -> List[dict]
    def on_session_end()
```

---

## 4. 调用链

```
AIAgent._restore_or_build_system_prompt()
  │
  ▼
build_system_prompt_parts()
  │
  ├─► _build_stable()
  │       ├─► load_soul_md()            → SOUL.md 内容
  │       ├─► TOOL_USE_ENFORCEMENT_*   → 工具使用指导
  │       ├─► SKILLS_GUIDANCE          → 技能系统提示
  │       ├─► PLATFORM_HINTS           → 平台环境提示
  │       └─► build_nous_subscription_prompt()
  │
  ├─► _build_context()
  │       ├─► system_message（用户传入）
  │       ├─► build_context_files_prompt() → AGENTS.md, .cursorrules 等
  │       └─► _scan_context_content()     → 注入扫描
  │
  └─► _build_volatile()
          ├─► memory_manager.build_system_prompt_block() → 记忆块
          ├─► USER.md profile
          └─► timestamp / model / provider 行

三层结果用 "\n\n".join() 组合
  │
  ▼
注入 _build_messages() 的 system 消息
```

---

## 5. 输入和输出

```
输入：
  - agent._cached_system_prompt（已缓存则跳过）
  - agent.system_message（用户传入）
  - agent._memory_manager
  - TERMINAL_CWD 下的 AGENTS.md / .cursorrules

输出：
  - system_prompt_parts: List[str]（三层组合后的字符串列表）
  - 副作用：agent._cached_system_prompt 被设置（缓存在 session 级别）

压缩触发时（context_compressor.py）：
  - 输入：messages + usage 数据
  - 输出：压缩后的 messages
```

---

## 6. 和其他模块的关系

```
Prompt Builder 依赖：
  ├─► SOUL.md / DEFAULT_AGENT_IDENTITY   ← 身份定义
  ├─► memory_manager.build_system_prompt_block()  ← 记忆
  ├─► skills_tool                          ← 技能索引
  └─► AGENTS.md / .cursorrules            ← 用户上下文文件

其他模块依赖 Prompt Builder：
  ├─► conversation_loop.py                ← 每个 turn 调用
  ├─► background_review.py                ← 继承缓存的 system prompt
  └─► context_compressor.py              ← 压缩后可能重建
```

---

## 7. 设计亮点

### 亮点 1：三层分离 + session 级缓存
```python
# agent/system_prompt.py 注释：
# "The agent's system prompt is built once per session and reused across all
# turns — only context compression triggers a rebuild.
# This keeps the upstream prefix cache warm."
```
stable 层几乎不变，context 层每个 session 变一次，volatile 层每个 turn 变。缓存友好。

### 亮点 2：注入扫描（Threat Patterns）
```python
# agent/system_prompt.py：
from tools.threat_patterns import scan_for_threats as _scan_for_threats

def _scan_context_content(content: str, filename: str) -> str:
    findings = _scan_for_threats(content, scope="context")
    if findings:
        return f"[BLOCKED: {filename} contained potential prompt injection ...]"
    return content
```
防止通过 AGENTS.md 等文件进行 prompt 注入。

### 亮点 3：Context Engine 可插拔架构
```python
# agent/context_engine.py：
class ContextEngine(ABC):
    """Pluggable context engine. Default is ContextCompressor.
    Third-party engines (e.g. LCM) can replace it via plugins/."""
```
压缩策略可以替换，不锁定在单一实现。

---

## 8. 风险和不足

- **SOUL.md 依赖**：系统行为高度依赖 SOUL.md 内容，修改需谨慎
- **token 估算误差**：`estimate_messages_tokens_rough()` 是粗略估算，可能与实际有偏差
- **压缩触发时机**：需要精确的 token 计数，否则可能浪费上下文窗口

---

## 9. 最小实现伪代码

```python
def build_system_prompt(agent) -> str:
    """极简版三层 Prompt"""

    # Layer 1: Stable（几乎不变）
    stable = load_soul_md() + "\n\n"
    stable += TOOL_USE_GUIDANCE + "\n\n"
    stable += SKILLS_GUIDANCE + "\n\n"

    # Layer 2: Context（每个 session 变）
    context = agent.system_message + "\n\n"
    context += scan_and_load_context_files(".")  # AGENTS.md, .cursorrules

    # Layer 3: Volatile（每个 turn 变）
    volatile = agent.memory_manager.build_block() + "\n\n"
    volatile += f"Timestamp: {now()}\n"
    volatile += f"Model: {agent.model}\n"

    return stable + "\n\n" + context + "\n\n" + volatile


def scan_and_load_context_files(cwd: Path) -> str:
    """扫描并加载上下文文件"""
    files = ["AGENTS.md", ".cursorrules"]
    content = []
    for f in files:
        path = Path(cwd) / f
        if path.exists():
            text = path.read_text()
            text = scan_for_threats(text, scope="context")  # 注入扫描
            content.append(f"=== {f} ===\n{text}")
    return "\n\n".join(content)
```

---

## 10. 练习题

### 练习 1：观察三层内容（入门）
```
目标：确认三层分别包含什么内容

步骤：
1. 在 build_system_prompt_parts() 返回前加日志
2. 分别打印 stable / context / volatile 部分
3. 运行 hermes，观察三层输出

产出物：三层内容截图或摘录
```

### 练习 2：修改 SOUL.md 观察变化（进阶）
```
目标：理解 stable 层对 Agent 行为的影响

步骤：
1. 备份 SOUL.md
2. 修改 SOUL.md 中的某一段（如身份描述）
3. 启动 hermes，对比行为变化
4. 恢复 SOUL.md

产出物：行为差异分析
```

### 练习 3：实现自定义 Context Source（高级）
```
目标：让 Agent 能读取项目特定的上下文文件

步骤：
1. 在 prompt_builder.py 中添加新函数 build_custom_context()
2. 扫描 .myproject-rules 文件
3. 将其加入 context 层
4. 测试

产出物：自定义 context 文件 + 对比测试结果
```
