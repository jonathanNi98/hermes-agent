# 学习笔记 3 — Day 6:核心架构区分

> 日期:2026-06
> 主题:`hermes_state.py` vs `memory_tool.py` —— 录"事实"vs 录"结论"

---

## 🎯 一句话总结

> **`hermes_state.py` 录"事实",`memory_tool.py` 录"结论"。**
> 事实是系统自动记的原始数据;结论是 LLM 主动蒸馏出的"该记住的东西"。

---

## 🔁 一个 turn 的完整数据流

```
┌────────────────────────────────────────────────────┐
│  LLM 处理一个 turn                                  │
│                                                     │
│  1. 读 memory (system prompt 里)                    │
│     → "用户喜欢 TS"                                 │
│                                                     │
│  2. 调工具、写消息                                  │
│     → hermes_state 记每一条消息                      │
│     → memory tool 记 LLM 主动 add 的事实             │
│                                                     │
│  3. LLM 决定记不记这个偏好                           │
│     → 调 memory_tool.add(...)                      │
│     → memory_manager.on_memory_write 双写到 state   │
│                                                     │
│  4. 下个 turn / 明天新 session                       │
│     → memory 还在,LLM 接着用                        │
│     → hermes_state messages 还在,但只对这个 session │
└────────────────────────────────────────────────────┘
```

---

## 📊 详细对照

| 维度 | `hermes_state.py` (SessionDB) | `memory_tool.py` (MemoryStore) |
|---|---|---|
| **存的什么** | 原始消息(user / assistant / tool_call / result) | LLM 蒸馏出的事实 / 偏好 / 项目信息 |
| **一行一条** | 一条消息(message) | 一条 span(记忆单元) |
| **粒度** | 一轮对话可能 5-20 行 | 一轮对话可能 0-1 行 |
| **写的人** | **系统**自动写(LLM 完全不知情) | **LLM 自己**用 `add/replace/remove` 工具写 |
| **读的人** | `/resume` `/rewind` `/compress` 等系统命令;FTS 搜原始消息 | LLM 在 system prompt 里看到自己的"长期记忆" |
| **检索方式** | FTS5 全文搜索(字面匹配 + trigram) | 子串匹配 + LLM 决策时自动从 system prompt 拿 |
| **依赖 LLM 吗** | ❌ 完全不依赖 | ✅ LLM 决定写什么、不写什么 |
| **生命周期** | 单 session 内有效,过时被压缩/清理 | 跨 session 持久,跟着 profile 走 |

---

## 🎬 一个具体场景

**用户说**: "我比较喜欢用 TypeScript,不太喜欢 JavaScript。"

### 写入路径

```
用户输入
  ↓
[1] hermes_state.py 自动记一条 user 消息
    → messages 表:{role: "user", content: "我比较喜欢用 TypeScript,..."}
    → LLM 完全不知道这一步发生了
    
[2] LLM 处理完 → hermes_state.py 再记一条 assistant 消息
    → messages 表:{role: "assistant", content: "好的,了解了..."}
    
[3] LLM **自己决定**: "用户表达了一个偏好,值得记下来"
    → 调 memory tool:add(target="user", content="喜欢 TypeScript,不喜欢 JavaScript")
    → memory entries:["喜欢 TypeScript,不喜欢 JavaScript"]
```

### 读取路径(第二天新 session)

```
新 session 启动
  ↓
[1] hermes_state.py: 这个 session 还没有 messages
    → messages 表空(因为新 session_id)
    
[2] memory_tool.py: 把昨天的"喜欢 TS"加载进 system prompt
    → "## Memory:\n- 喜欢 TypeScript,不喜欢 JavaScript"
    → LLM 看到后,回答就自动适配
```

**关键观察**:
- 如果用户**第二天**问"我用什么语言比较好?"——LLM 之所以知道"用 TS",**完全靠 memory tool**,跟 hermes_state 没关系
- 如果用户**第二天**打 `/resume` 想看昨天的原文对话——**完全靠 hermes_state**,跟 memory tool 没关系

---

## 🧬 它们是孤立的还是有联系?

**内置 builtin provider 把它们桥接起来了**(`memory_manager.py` 的 `on_memory_write`):

```python
# memory_manager.py 里
def on_memory_write(self, action, target, content, metadata=None):
    # 1. 写 memory store(本意)
    self._builtin_store.add(target, content)
    # 2. 写 hermes_state 留痕(调试 / 重放)
    #    → 在 messages 表里记一条 tool_call/tool_result
```

**所以**:builtin 路径下,记一条 memory = 在 `messages` 表里**同时**留一份 tool_call 记录。
**好处**:`/resume` 之后,你能"看到 LLM 当时是主动记下了什么"——完整的因果链可回放。

---

## 🗂️ 用"档案柜"类比

| 比喻 | hermes_state.py | memory_tool.py |
|---|---|---|
| **角色** | 监控摄像头 | 秘书笔记本 |
| **录什么** | **所有发生过的事**(录像) | **值得记的事**(笔记) |
| **谁来录** | 自动 24/7 录 | 秘书自己挑重点记 |
| **回看** | 调监控查"那一刻发生了什么" | 翻笔记本查"老板有什么偏好" |
| **特点** | 完整、原始、量大 | 浓缩、主观、量小 |

---

## 🔧 修改哪个文件

| 你想... | 改哪个文件 |
|---|---|
| 改"对话怎么存"、加字段、改 FTS 索引 | `hermes_state.py` |
| 改"LLM 怎么决策记什么"、"记忆怎么去重/匹配" | `memory_tool.py` |
| 改"两个之间怎么同步" | `memory_manager.py` 的 `on_memory_write` |

---

## 🧠 这个区分为什么重要?

1. **读源码时知道去哪改** — bug 出现在 `/resume` 找 hermes_state,bug 出现在"LLM 没记住我的偏好"找 memory_tool
2. **加 feature 时选对文件** — 想加"对话可导出"就动 hermes_state,想加"自动提取偏好"就动 memory_tool + manager
3. **跟别人解释时能讲清楚** — 监控摄像头 vs 秘书笔记本是个一秒钟能听懂的类比
4. **理解 Day 6 后续** — Phase 2/3/4/5 走 hermes_state 内部,Day 6 还会涉及 memory 的提取/匹配策略

---

## 📍 当前位置

- ✅ Day 6 已读:`memory_provider.py` `memory_manager.py` `memory_tool.py`
- 🟡 Day 6 进行中:`hermes_state.py` — Phase 1 鸟瞰、Phase 2.1 `__init__`、Phase 2.2 `_execute_write` 已完成
- ⏳ Phase 2.3 `_init_schema`、Phase 3 FTS5+search、Phase 4 session lifecycle、Phase 5 edges 待读
- ⏳ Day 6 结尾:hindsight / mini_hermes(可选)、Day 6 练习题(10 题)

---

## 💡 关键 takeaway

> hermes_state 是"系统的记忆"(系统自动录),memory 是"LLM 的记忆"(LLM 自己蒸馏)。
> builtin provider 用 `on_memory_write` 桥接两者,让 LLM 主动记的事也能在 `/resume` 里看到决策过程。
