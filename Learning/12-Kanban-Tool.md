# 核心模块学习 — Kanban Tool(多 Agent 任务板)

> **本文件视角**:在 `08-Planning-Delegation.md` 中 kanban 只是被一带而过的三个主题之一
> (delegate / kanban / cron)。本文件专门把 kanban 抽出来,做一次完整的源码级深读。

---

## 1. 这个模块解决什么问题

**问题**:一个高层目标如何在多个 agent 之间拆解、跟踪、汇总?Worker 完成后如何给
orchestrator / 人类交付"结构化的 handoff",而不是"它跑完了 / 它挂了"这种二值结果?

**答案**:
- Kanban 是一组**注册到 model schema 的工具**,背后是个 SQLite 任务板 (`~/.hermes/kanban.db`)
- 模型在创建任务时声明 `assignee` (哪个 profile 跑) 和 `parents[]` (依赖哪些父任务)
- 一个常驻的 **dispatcher** (`gateway/run.py:5609 _kanban_dispatcher_watcher`)
  每 tick 扫描 ready 任务,spawn 对应 profile 的 worker agent
- Worker 跑完调 `kanban_complete(...)` 上报结构化结果,带 `summary` + `metadata` + `created_cards` + `artifacts`
- 一个并行的 **notifier** (`gateway/run.py:5105 _kanban_notifier_watcher`) 把终态事件投递给订阅者

---

## 2. 真实源码位置(已验证)

```
tools/kanban_tools.py              ← 9 个 tool 的定义 + 注册 (1412 行)
hermes_cli/kanban.py               ← 人类用的 CLI 入口 (`hermes kanban list/complete/...`)
hermes_cli/kanban_db.py            ← DB schema / KanbanDB 类 / 状态机
hermes_cli/kanban_decompose.py     ← 一次性分解 helper
hermes_cli/kanban_swarm.py         ← 启动 swarm 跑一批任务
hermes_cli/kanban_diagnostics.py   ← 调试用
hermes_cli/kanban_specify.py       ← 任务规范 / 模板
gateway/run.py:5609                ← 嵌入式 dispatcher watcher
gateway/run.py:5105                ← 嵌入式 notifier watcher
gateway/run.py:4687-4696           ← 启动两个后台 task 的位置
```

**重要发现**:
- Kanban 工具是**按需注册**的——普通 `hermes chat` session 看不到,只有 dispatcher spawn 的
  worker (`HERMES_KANBAN_TASK` env) 或显式配了 `kanban` toolset 的 profile 才能见
- Dispatcher 默认是**内嵌在 gateway 里的 asyncio 任务**——不用单独跑 `hermes kanban daemon`,
  但可关 (`kanban.dispatch_in_gateway: false` / `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`)
- CLI / dashboard / slash command 是**人**用的;tool 是**模型**用的。同一份 DB 同一份语义,
  两套入口互不干扰

---

## 3. 核心类 / 函数 / 方法(已验证)

### 3.1 9 个工具(tools/kanban_tools.py:1333-1411)

```python
# —— Worker surface (HERMES_KANBAN_TASK 触发) ——
KANBAN_SHOW_SCHEMA      # 读任务全状态:task / parents / children / comments / runs / events
KANBAN_COMPLETE_SCHEMA  # 标 done,带 summary + metadata + created_cards + artifacts
KANBAN_BLOCK_SCHEMA     # 卡住任务,等人类回复 (写 reason)
KANBAN_HEARTBEAT_SCHEMA # 长任务心跳,告诉 dispatcher "我还活着"
KANBAN_COMMENT_SCHEMA   # 任务评论,持久化笔记

# —— Orchestrator surface (profile 配 kanban toolset,且无 HERMES_KANBAN_TASK) ——
KANBAN_LIST_SCHEMA      # 列任务做路由(assignee / status / tenant 过滤)
KANBAN_CREATE_SCHEMA    # 创建子任务,fan-out 入口
KANBAN_UNBLOCK_SCHEMA   # 重开被 block 的任务
KANBAN_LINK_SCHEMA      # 任务间建非 parent 链接
```

注册示例(tools/kanban_tools.py:1351-1358):
```python
registry.register(
    name="kanban_complete",
    toolset="kanban",
    schema=KANBAN_COMPLETE_SCHEMA,
    handler=_handle_complete,
    check_fn=_check_kanban_mode,        # ← gating 关键
    emoji="✔",
)
```

### 3.2 Gating 逻辑(tools/kanban_tools.py:62-90)

```python
def _check_kanban_mode() -> bool:
    """Worker surface:show / complete / block / heartbeat / comment"""
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    return _profile_has_kanban_toolset()

def _check_kanban_orchestrator_mode() -> bool:
    """Orchestrator-only:list / unblock"""
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False   # ← worker 看不到 list / unblock
    return _profile_has_kanban_toolset()
```

`check_fn` 会被 registry TTL 缓存 (~30s),所以每次 tool call 不会反复 import 配置。

### 3.3 任务所有权强制(tools/kanban_tools.py:132-160)

```python
def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    """Worker 只能 mutate 自己的 task。
    防止 prompt injection 让 worker 去 complete 别人的任务(issue #19534)。
    """
    if os.environ.get("HERMES_KANBAN_TASK") != tid:
        return tool_error(...)
    ...
```

每个 mutating handler (`_handle_complete`, `_handle_block`, `_handle_heartbeat`)
第一行都调这个 guard,验证 env 里的 task id 和参数里传的一致。

### 3.4 Dispatcher 钩子(gateway/run.py:4687-4696)

```python
# Start background kanban notifier — delivers `completed`, `blocked`, ...
asyncio.create_task(self._kanban_notifier_watcher())

# Start background kanban dispatcher — spawns workers for ready tasks.
# Gated by `kanban.dispatch_in_gateway` (default True).
asyncio.create_task(self._kanban_dispatcher_watcher())
```

`_kanban_dispatcher_watcher` 每 tick 调 `kanban_db.dispatch_once`
(在 `asyncio.to_thread` 里跑,SQLite WAL 不会 block event loop)。

---

## 4. 调用链

### 4.1 一次完整 fan-out 的端到端流程

```
[1] Orchestrator profile 收到高层目标
        │
        ▼
[2] kanban_show(task_id=当前)             ── 拿上下文、parents、prior attempts
        │
        ▼
[3] kanban_create(...)  ×N                ── 每个子任务指定 assignee + parents
        │   │
        │   └─ 新任务 status="todo",等所有 parents 变 done 后自动 promote 到 "ready"
        │
        ▼
[4] kanban_complete(task_id=当前,
                     created_cards=[...]) ── kernel verify 防止 phantom refs
        │
        ▼
[5] Gateway 里的 _kanban_dispatcher_watcher 下一 tick
        │   │
        │   └─ kanban_db.dispatch_once() 扫到 ready 任务
        │       │
        │       └─ spawn AIAgent(profile=task.assignee, env=HERMES_KANBAN_TASK=...)
        │
        ▼
[6] Worker agent 启动,看到 schema 里有 kanban_* 工具
        │
        ▼
[7] Worker 干活,期间 kanban_heartbeat() 发心跳
        │
        ▼
[8] Worker 调 kanban_complete(summary=..., metadata={...},
                                created_cards=[...], artifacts=[...])
        │
        ▼
[9] 终态事件落 DB,kernel 推 cursor
        │
        ▼
[10] _kanban_notifier_watcher 看到新事件
        │   └─ 上传 artifacts → 发给订阅者
        │
        ▼
[11] Parents 任务里被声明的 synthesis 任务 auto-promote 到 ready
        │
        ▼
[12] Synthesis worker 起来,kanban_show() 拿所有子任务的 summary/metadata
```

### 4.2 Tool call → DB write 内部(以 _handle_complete 为例)

```
kanban_complete(task_id="abc", summary=..., metadata=..., created_cards=["x","y"])
    │
    ├─► _default_task_id("abc") → "abc"
    ├─► _enforce_worker_task_ownership("abc")  # 验 env 一致
    ├─► 校验 summary / metadata / created_cards / artifacts 类型
    ├─► _stamp_worker_session_metadata("abc", metadata)
    │       # 把 HERMES_SESSION_ID 写进 metadata.worker_session_id(可追溯)
    ├─► _connect(board=...) → kb, conn
    │
    └─► kb.complete_task(conn, "abc",
                          result=..., summary=..., metadata=...,
                          created_cards=["x","y"],
                          expected_run_id=_worker_run_id("abc"))
            │
            ├─► 校验 created_cards 真实存在 (否则 HallucinatedCardsError)
            ├─► 写 task.status = "done", task.completed_at = now
            ├─► 写 runs 表 + events 表 (审计)
            ├─► 检查 children,如果所有 parents 都 done → auto-promote child 到 "ready"
            └─► 通知 notifier (写 kanban_notify_subs)
```

---

## 5. 输入和输出

### 5.1 各工具的入参(选关键字段,完整 schema 在 tools/kanban_tools.py)

```
kanban_show(task_id, board=None)
    → {"task": {...}, "parents": [...], "children": [...],
       "comments": [...], "events": [...], "runs": [...],
       "worker_context": "..."}  # 预格式化的字符串,worker 可原样用

kanban_list(assignee=None, status=None, tenant=None,
            include_archived=False, limit=50, board=None)
    → {"tasks": [{id, title, status, assignee, priority, ...}], "truncated": bool}

kanban_complete(task_id, summary=None, result=None,
                metadata=None,                # 自由 dict,装 changed_files / tests_run / ...
                created_cards=None,           # 这次 fan-out 出去的所有新任务 id
                artifacts=None,               # 交付物绝对路径,gateway 上传
                board=None)
    → {"task_id": "abc", "status": "done",
       "promoted_children": ["synth-1"], ...}  # 顺便报告哪些子任务被 auto-promote

kanban_block(task_id, reason, board=None)
    → {"task_id": "abc", "status": "blocked"}

kanban_heartbeat(task_id, note=None, board=None)
    → {"task_id": "abc", "heartbeat_at": "..."}

kanban_comment(task_id, body, board=None)        # body 是 markdown
    → {"comment_id": 42}

kanban_create(title, assignee, body=None,
              parents=None,                      # 依赖的父任务 id 列表
              tenant=None, priority=0,
              workspace_kind="scratch",          # scratch | dir | worktree
              workspace_path=None,               # dir/worktree 时必填
              board=None)
    → {"task_id": "new-1", "status": "todo", "promoted_after_parents": False}

kanban_unblock(task_id, board=None)
    → {"task_id": "abc", "status": "ready"}

kanban_link(task_id, other_task_id, kind="related", board=None)
    → {"link_id": 99}
```

### 5.2 任务状态机(kanban_list 的 schema enum,kaban_tools.py:936-940)

```
triage → todo → ready → running → blocked / done / archived
                  ↑                 │
                  └─────────────────┘  (unblock 走 ready)
```

**关键转换**:
- `triage` → `todo`:进入板子的初始状态
- `todo` → `ready`:所有 `parents` 都 `done` 时**自动 promote**(kanban_create 描述里讲了这个 fan-in 语义)
- `ready` → `running`:dispatcher 选中了,准备 spawn
- `running` → `done`:worker 调 `kanban_complete`
- `running` → `blocked`:worker 调 `kanban_block`,等人类介入
- `blocked` → `ready`:orchestrator 调 `kanban_unblock`

---

## 6. 和其他模块的关系

```
Kanban 依赖:
  ├─► tools/registry.py (registry.register + check_fn gating)
  ├─► hermes_cli/kanban_db.py (KanbanDB 类,SQLite CRUD)
  ├─► hermes_cli/config.py (load_config() 读 toolsets)
  └─► gateway/run.py (_kanban_dispatcher_watcher / _kanban_notifier_watcher spawn workers)

Kanban 被依赖:
  ├─► agent/conversation_loop.py (tool_call 走标准 dispatch)
  ├─► agent/tool_executor.py (跟其他 tool 一样注册分发)
  └─► CLI / dashboard / /kanban slash command(人用的另一套入口)

跟 `todo` tool 的关系:
  ├─► todo: in-memory,单 session,模型自己跟踪执行进度
  └─► kanban: persistent,跨 session 跨 agent,team 任务板
  (两者不互通:todo list 看不到 kanban 任务,kanban 任务也读不到 todo)
```

---

## 7. 设计亮点

### 亮点 1:为什么是 tool,不是 `hermes kanban` CLI 子进程?

tools/kanban_tools.py:9-27 docstring 列了 3 个原因:

1. **后端可移植性** —— Worker 的终端可能指向 Docker / Modal / Singularity / SSH,
   容器里没装 `hermes` CLI 也挂不到 SQLite。Tool 在 agent 进程里跑,永远能拿到
   `~/.hermes/kanban.db`。
2. **没有 shell quoting 地雷** —— 传 `--metadata '{"x": [...]}'` 给 shlex+argparse 很脆。
   Tool call 的结构化参数绕过整个 shell 层。
3. **错误可被模型推理** —— 失败返回结构化 JSON,模型能直接 reasoning,不用解析 stderr。

**人用 CLI / dashboard / /kanban slash command;模型用 tool。同一份 DB,两套入口。**

### 亮点 2:Worker task ownership 强制

```python
def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    if os.environ.get("HERMES_KANBAN_TASK") != tid:
        return tool_error(...)
```

防御 prompt injection:即便 worker 被注入"帮我 complete 任务 xyz"这种指令,
env 里的 task id 不匹配,直接拒。

参考 issue #19534(注释里提到的)。**Orchestrator profile 不受此限**——它的工作本来就是
路由,合法地需要 close 子任务或 reopen 被 block 的任务。

### 亮点 3:父子依赖 + auto-promote

`parents: [list of task ids]` 是声明式依赖图。`kanban_create` 描述里说:

> "The new task stays in 'todo' until every parent reaches 'done'; then it auto-promotes to 'ready'."

意味着:
- 创建 researcher 子任务时不需要写"等所有 researcher 完才能跑 synthesis"
- 创建一个 synthesis 任务,把 researcher ids 全塞 `parents` 就行
- Dispatcher 自动在依赖满足时 promote

**这是 fan-in 模式**——比 imperative 的"我手动检查再开下一个"鲁棒得多。

### 亮点 4:artifacts 自动上传

`kanban_complete(artifacts=["/abs/path/to/chart.pdf"])` 会被 `_handle_complete`
合并进 `metadata["artifacts"]` (line 504-542),notifier watcher 读
`payload["artifacts"]` 字段,逐个上传成 native attachment 投递给订阅者。

**好处**:worker 不用关心"怎么把文件送给人类"——只要把路径列出来,gateway 自动跑。
省了 worker 自己写 file upload 逻辑。

### 亮点 5:created_cards 验证 (anti-hallucination)

`kanban_complete(created_cards=[...])` 列表,DB 层会 verify 每个 id 真实存在。
不存在的 id 抛 `HallucinatedCardsError`,**任务本身不被 mutate**(guard 在 write txn 之前),
worker 收到结构化错误后可修正重试或 drop 字段。

防的是 worker 幻觉地报告"我创建了 5 个子任务"但实际只创建了 3 个,
如果 downstream automation 信了就会跑空。

### 亮点 6:Dispatcher 嵌入 gateway,但可关

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true   # 默认,不开 daemon
```

```bash
HERMES_KANBAN_DISPATCH_IN_GATEWAY=0 hermes ...  # env 逃逸口
```

_gated 注释(gateway/run.py:5609-5630)解释:读一次 config,之后每 tick 检查 `_running`。
Gateway stop 翻转 flag,cancel pending task,in-flight `to_thread` 自然结束。

**默认嵌入**省一个进程,**可关**让用户跑独立的 `hermes kanban daemon`(多 gateway 共享一个 dispatcher 场景)。

---

## 8. 风险和不足

- **Worker 看不到 board 全貌** —— `_check_kanban_orchestrator_mode` 把 `kanban_list` 隐藏了。
  Worker 拿不到"还有哪些 sibling 任务在跑",只能 `kanban_show(自己的)`。
  设计上是合理的(防止乱动别人的任务),但 worker 偶尔需要"看一眼我等的是哪个父任务"
  时只能猜。
- **Dispatcher 是 best-effort polling** —— `_kanban_dispatcher_watcher` 是 interval-based,
  不是 push。任务 created 到 worker spawn 之间有 `dispatch_interval_seconds` 的延迟
  (默认估计 1-5s)。高 QPS 场景需要自己调。
- **artifacts 上传是 silent skip** —— 路径在 disk 上不存在时 notifier 默默跳过
  (line 1043-1044 注释)。Worker 不知道文件丢了。
- **Tenant 隔离不严格** —— `tenant` 字段是个 namespace 字符串,不是 ACL。`kanban_list(tenant=X)`
  是查询参数不是访问控制。任何拿到 DB 读权限的进程都能跨 tenant 列表。
- **Status 转移逻辑分散** —— promote / unblock / complete 各有 handler 写,没有一个
  集中的状态机模块。改转移规则要扫多处。
- **`created_cards` 失败不阻塞 complete** —— 如果想 fan-out 5 个子任务但只成功 4 个,
  `kanban_complete(created_cards=[5个])` 会抛 HallucinatedCardsError,任务**不会被标 done**。
  Worker 必须重试或 drop 字段——但 drop 后下游 automation 又不知道这 4 个存在。
  没有"部分成功"的表达。
- **Tool 注册没走 `_AGENT_LOOP_TOOLS`** —— 跟 `todo` 不一样,`kanban_*` 是普通
  registry 流程,handler 里直接连 DB。如果 `_handle_complete` 抛异常没被 tool_executor
  正确处理,DB 可能半提交。

---

## 9. 最小实现伪代码

```python
# 假设 SQLite 已建好 tasks / runs / events / comments / edges 表

import json
import os
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    title: str
    body: str
    assignee: str
    status: str            # triage|todo|ready|running|blocked|done|archived
    tenant: str
    priority: int
    workspace_kind: str
    workspace_path: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    result: str | None
    current_run_id: int | None

class KanbanDB:
    def __init__(self, path="~/.hermes/kanban.db"):
        self.path = os.path.expanduser(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'triage',
            tenant TEXT,
            priority INTEGER DEFAULT 0,
            workspace_kind TEXT DEFAULT 'scratch',
            workspace_path TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT, completed_at TEXT,
            result TEXT, current_run_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS edges (
            parent_id TEXT, child_id TEXT,
            PRIMARY KEY (parent_id, child_id)
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, profile TEXT,
            status TEXT, outcome TEXT,
            summary TEXT, error TEXT,
            metadata TEXT,
            started_at TEXT, ended_at TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, kind TEXT,
            payload TEXT, run_id INTEGER,
            created_at TEXT
        );
        """)

    def create_task(self, *, title, assignee, body=None,
                    parents=None, tenant=None, priority=0,
                    workspace_kind="scratch", workspace_path=None) -> Task:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, title, body, assignee, "triage", tenant,
             priority, workspace_kind, workspace_path,
             now, None, None, None, None),
        )
        # 初始 status = "triage",立刻 promote 到 "todo" (没 parents)
        # 有 parents → "todo",等所有 parents done 再 promote
        self.conn.execute(
            "UPDATE tasks SET status='todo' WHERE id=?", (task_id,)
        )
        for p in (parents or []):
            self.conn.execute(
                "INSERT INTO edges VALUES (?,?)", (p, task_id)
            )
        return self.get_task(self.conn, task_id)

    def complete_task(self, task_id, *, result=None, summary=None,
                      metadata=None, created_cards=None,
                      expected_run_id=None) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        # 1. verify created_cards 存在
        if created_cards:
            placeholders = ",".join("?" * len(created_cards))
            rows = self.conn.execute(
                f"SELECT id FROM tasks WHERE id IN ({placeholders})",
                created_cards,
            ).fetchall()
            found = {r[0] for r in rows}
            missing = set(created_cards) - found
            if missing:
                raise HallucinatedCardsError(missing)

        # 2. mutate task
        self.conn.execute(
            "UPDATE tasks SET status='done', completed_at=?, result=? "
            "WHERE id=?",
            (now, json.dumps({"summary": summary, "metadata": metadata}),
             task_id),
        )

        # 3. write event (notifier 读)
        self.conn.execute(
            "INSERT INTO events(task_id, kind, payload, created_at) "
            "VALUES (?,?,?,?)",
            (task_id, "completed",
             json.dumps({"summary": summary, "metadata": metadata,
                         "artifacts": (metadata or {}).get("artifacts", [])}),
             now),
        )

        # 4. promote children whose parents are all done
        self._auto_promote_children(task_id)
        return True

    def _auto_promote_children(self, parent_id):
        for child_id, in self.conn.execute(
            "SELECT child_id FROM edges WHERE parent_id=?", (parent_id,)
        ).fetchall():
            all_parents_done = self.conn.execute(
                """
                SELECT COUNT(*) FROM edges e
                JOIN tasks t ON t.id = e.parent_id
                WHERE e.child_id = ? AND t.status != 'done'
                """, (child_id,),
            ).fetchone()[0] == 0
            if all_parents_done:
                self.conn.execute(
                    "UPDATE tasks SET status='ready' WHERE id=? AND status='todo'",
                    (child_id,),
                )

    def dispatch_once(self):
        """Dispatcher 一 tick:扫 ready 任务,spawn worker"""
        ready = self.conn.execute(
            "SELECT id, title, body, assignee, tenant, priority, "
            "workspace_kind, workspace_path FROM tasks "
            "WHERE status='ready' "
            "ORDER BY priority DESC, created_at ASC LIMIT 5"
        ).fetchall()
        for row in ready:
            task_id, title, body, assignee, *_ = row
            self.conn.execute(
                "UPDATE tasks SET status='running', started_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), task_id),
            )
            # 真实实现里 spawn 的是 AIAgent 子进程
            spawn_worker(
                task_id=task_id, profile=assignee, body=body,
                env={"HERMES_KANBAN_TASK": task_id, ...},
            )


# —— Tool 层 ——
def _check_kanban_mode() -> bool:
    return bool(os.environ.get("HERMES_KANBAN_TASK"))

def _enforce_worker_task_ownership(tid: str):
    if os.environ.get("HERMES_KANBAN_TASK") != tid:
        return tool_error("ownership violation")
    return None

def kanban_complete(args: dict, kb: KanbanDB) -> str:
    tid = args.get("task_id") or os.environ.get("HERMES_KANBAN_TASK")
    err = _enforce_worker_task_ownership(tid)
    if err:
        return err
    summary = args.get("summary")
    if not summary:
        return tool_error("summary required")
    try:
        kb.complete_task(
            tid,
            summary=summary,
            metadata=args.get("metadata"),
            created_cards=args.get("created_cards"),
        )
    except HallucinatedCardsError as e:
        return tool_error(f"created_cards missing: {e.missing}")
    return json.dumps({"task_id": tid, "status": "done"})


# —— Gateway 启动 ——
async def main():
    kb = KanbanDB()
    while True:
        await asyncio.to_thread(kb.dispatch_once)
        await asyncio.sleep(2.0)  # dispatch_interval_seconds
```

---

## 10. 练习题

### 练习 1:追踪一次 fan-out 的完整数据流(入门)

```
目标:理解 orchestrator → 创建子任务 → dispatcher → worker → complete 的端到端路径

步骤:
1. 在 kanban_create handler 加日志,记 title / assignee / parents
2. 在 kanban_db.dispatch_once 加日志,记被 promote 的 child id
3. 在 _handle_complete 加日志,记 created_cards 验证结果
4. 手动用 hermes CLI 创建一个 parent + 3 个 child + 1 个 synthesis 任务
5. 把 parent 标 done,观察 synthesis 是否 auto-promote

产出物:一张时序图,标出每次状态转移的时间戳和触发点
```

### 练习 2:理解 worker ownership 强制(进阶)

```
目标:理解为什么 worker 不能 mutate 别人的任务

步骤:
1. 阅读 _enforce_worker_task_ownership (tools/kanban_tools.py:132)
2. 在 _handle_complete 入口前后加日志,看正常 vs 异常的 task_id
3. 尝试用 Python REPL 模拟两个 worker:
   - worker_a: env HERMES_KANBAN_TASK="a"
   - worker_b: env HERMES_KANBAN_TASK="b"
4. 让 worker_a 调 kanban_complete(task_id="b"),观察是否被拒
5. 理解这个 guard 防的是哪类 prompt injection 攻击

产出物:攻击场景分析(谁会想让 worker_a 动 b?为什么?)
```

### 练习 3:实验 parent auto-promote 语义(进阶)

```
目标:理解声明式依赖图 vs 命令式检查

步骤:
1. 创建任务 A,assignee=researcher,parents=[]
2. 创建任务 B,assignee=synthesizer,parents=[A]
3. 观察 B 的初始 status (应该是 "todo",不是 "ready")
4. 把 A 标 done
5. 观察 B 是否自动变 "ready"
6. 再创建 C,assignee=researcher2,parents=[A],同样把 A 标 done
7. 验证:即使 C 跟 B 互不依赖,B 也不会"等 C 一起"

产出物:解释 fan-in 语义的边界——它是"全部父任务 done"还是"任一父任务 done"?
```

### 练习 4:跟踪一次 artifact 上传(高级)

```
目标:理解 worker → DB → notifier → 订阅者的完整交付链

步骤:
1. 在 _handle_complete 的 artifacts 处理段(line 504-542)加日志
2. 在 _kanban_notifier_watcher 加日志,记读到的 payload["artifacts"]
3. 写一个 worker 调 kanban_complete(artifacts=["/tmp/test.pdf"])
4. 在 /tmp 放个真文件,跑通
5. 删掉文件,再跑一次,观察 notifier 是否 silent skip
6. 读 line 1043-1044 的注释,理解为什么是 silent 而不是 fail

产出物:交付链时序图 + silent skip 的设计取舍分析
```

### 练习 5:Dispatcher 关闭的影响(高级)

```
目标:理解 dispatch_in_gateway 开关的实际影响

步骤:
1. 创建一个 ready 任务,确认 gateway 模式下会自动 spawn worker
2. 设 HERMES_KANBAN_DISPATCH_IN_GATEWAY=0 重启 gateway
3. 观察那个 ready 任务是否变成 "永远 ready"
4. 思考:多 gateway 部署时,关掉内嵌 dispatcher 跑独立 daemon 的好处是什么?
5. (可选)读 hermes_cli/kanban.py 找独立 daemon 入口

产出物:单 gateway vs 多 gateway + daemon 模式的对比表
```

---

## 📍 跟其它学习模块的衔接

- **08-Planning-Delegation.md** —— 把 kanban 跟 delegate / cron 一起介绍过一遍。本文件是
  它的"专精版",可以替换 08 里 kanban 相关段落。建议:**先读 08 建立全景图,再读本文件
  深入**,或者反过来。
- **05-Tool-System.md** —— 解释 `registry.register(name, schema, handler, check_fn, ...)`
  的统一约定。Kanban 是 `check_fn` gating 玩得最花的一个例子。
- **02-Agent-Loop.md** + **agent/conversation_loop.py:351 run_conversation** ——
  Worker 的 conversation loop 跟普通 session 一样,只是 env 多设了 `HERMES_KANBAN_TASK`,
  让 `check_fn` 把 kanban_* 工具"点亮"。
- **00-学习路线图-总结.md** Day 11 (Delegation) —— Kanban 是 Day 11 的一部分。
  想看完整 picture,Day 8/9 (Memory) → Day 10 (Skills) → Day 11 (本文件) 顺序读。

---

## 💡 关键 takeaway

> Kanban 是 Hermes 的"team task board",不是单 agent 的 todo list。
> 9 个 tool 围绕一个 SQLite DB,声明式 parent 依赖让 fan-out / fan-in 自然发生,
> dispatcher 内嵌在 gateway 里持续把 ready 任务 spawn 成 worker。
> Worker surface 只能 mutate 自己的 task (防 prompt injection),orchestrator surface
> 才有 list / unblock 权限。Artifacts 路径 + created_cards 验证 = 端到端交付自动化。
