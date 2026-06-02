# 核心模块学习 — Safety Boundary / Permission / Sandbox

## 1. 这个模块解决什么问题

**问题**：Agent 如何防止执行危险操作（删除系统文件、执行恶意命令）？写操作是否需要审批？

**答案**：多层安全机制：路径验证、命令白名单、危险命令检测、Approval 审批状态机。

---

## 2. 真实源码位置（已验证）

```
tools/approval.py               ← 危险命令审批（2000+行）
tools/path_security.py         ← 路径验证（validate_within_dir）
tools/tirith_security.py       ← 危险命令模式检测
agent/file_safety.py           ← 文件安全策略
tools/terminal_tool.py         ← 命令执行 + approval 回调
agent/tool_result_classification.py ← 工具结果分类（read/write）
tools/credential_files.py      ← 凭证文件访问控制
```

**重要发现**：
- `approval.py` 状态机支持 pending/approved/denied，超时自动拒绝
- `path_security.py` 的 `validate_within_dir()` 被多个工具复用（skill_manager_tool, skills_tool, cronjob_tools 等）
- YOLO 模式冻结在模块导入时（`is_truthy_value(os.getenv("HERMES_YOLO_MODE")` 一次性读取）
- `~/.hermes/config.yaml` 写入已被阻止（防止 approval bypass）

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# tools/approval.py
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))
_approval_session_key: contextvars.ContextVar[str]  # per-session 审批状态

def detect_dangerous_command(command: str) -> Optional[dict]
    """检测危险命令，返回 {pattern, risk_level} 或 None"""

def request_approval(action, risk_level, details) -> str
    """请求用户审批，返回 approval_id"""

def check_approval(approval_id) -> str
    """检查审批状态"""

def _fire_approval_hook(hook_name: str, **kwargs)

# tools/path_security.py
def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """检查路径是否在允许目录内
    Returns error message if blocked, None if safe."""

def has_traversal_component(path_str: str) -> bool:
    """快速检查 .. 遍历攻击"""

# tools/tirith_security.py
DANGEROUS_PATTERNS = [...]  # 危险命令模式列表

def detect_dangerous_command(command: str) -> bool:
    """检测命令是否匹配危险模式"""

# agent/tool_result_classification.py
READ_ONLY_TOOLS = frozenset([...])
WRITE_TOOLS = frozenset([...])

def classify_tool(name: str) -> str:
    """返回 'read' / 'write' / 'unknown'"""
```

---

## 4. 调用链

```
危险操作触发：
  │
  ▼
tool_executor.execute(tool_name, args)
  │
  ├─► 工具分类：classify_tool(tool_name) → 'read' / 'write'
  │
  ├─► 路径检查：validate_within_dir(path, root)
  │       └─► 失败 → 返回错误，不执行
  │
  ├─► 危险命令检测：detect_dangerous_command(command)
  │       └─► 命中 → request_approval()
  │               │
  │               ├─► save_approval_request()
  │               ├─► notify_user_approval()
  │               └─► 等待用户响应 / 超时
  │
  └─► Approval 检查：check_approval(approval_id)
          ├─► approved → 执行
          ├─► denied → 返回拒绝错误
          └─► timeout → 超时拒绝

审批状态机：
  Pending → User Approved → 执行
          ↘ User Denied → 返回拒绝
          ↘ Timeout → 超时拒绝
```

---

## 5. 输入和输出

```
validate_within_dir：
  输入：path: Path, root: Path
  输出：None（安全）或 str（错误信息）

detect_dangerous_command：
  输入：command: str
  输出：{"pattern": "...", "risk_level": "high"} 或 None

request_approval：
  输入：action: str, risk_level: str, details: str
  输出：{"approval_id": "...", "status": "pending"}
```

---

## 6. 和其他模块的关系

```
Safety 依赖：
  ├─► hermes_cli.config        ← 审批配置
  ├─► hermes_state.py          ← 审批状态持久化（gateway）
  └─► tools/terminal_tool.py   ← approval 回调

其他模块依赖 Safety：
  ├─► tool_executor.py         ← 执行前检查
  ├─► file_tools.py            ← 路径验证
  ├─► terminal_tool.py         ← 命令白名单 + approval
  └─► cronjob_tools.py         ← 定时任务路径验证
```

---

## 7. 设计亮点

### 亮点 1：YOLO 模式冻结在导入时
```python
# tools/approval.py：
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))
```
防止运行时 skill 动态修改环境变量绕过审批。

### 亮点 2：per-session 审批状态
```python
# tools/approval.py：
_approval_session_key: contextvars.ContextVar[str]
```
Gateway 并发执行时，每个 session 有独立审批状态，线程安全。

### 亮点 3：共享的路径验证 helper
```python
# tools/path_security.py：
# "Shared path validation helpers previously duplicated across
# skill_manager_tool, skills_tool, skills_hub, cronjob_tools, and credential_files."
```
DRY 原则，多工具复用同一验证逻辑。

### 亮点 4：多层防护
```
用户输入 → classify_tool（判断类型）
  → validate_within_dir（路径检查）
  → detect_dangerous_command（命令检测）
  → request_approval（审批）
  → execute
```
纵深防御，单层绕过不会直接导致安全问题。

---

## 8. 风险和不足

- **配置驱动**：路径限制依赖 config.yaml，配置错误可导致安全漏洞
- **Approval 超时**：默认超时后默认拒绝，但需确认行为符合预期
- **子代理隔离**：子代理的 approval 回调通过 ThreadPoolExecutor initializer 传递，设计复杂
- **凭证访问**：`env_probe` 等工具的凭证访问控制需持续审计

---

## 9. 最小实现伪代码

```python
# ===== path_security.py =====
def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Returns error message if validation fails, or None if safe."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


# ===== approval.py =====
class ApprovalState(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"

def request_approval(action: str, risk_level: str, details: str) -> str:
    approval_id = str(uuid.uuid4())

    # 保存审批请求
    save_approval_request(approval_id, {
        "action": action,
        "risk_level": risk_level,
        "details": details,
        "status": ApprovalState.PENDING,
        "created_at": now(),
    })

    # 通知用户（CLI / Gateway）
    notify_user_approval(approval_id)

    # 等待响应（同步阻塞或异步回调）
    result = wait_for_approval(approval_id, timeout=120)
    return result


def check_approval(approval_id: str) -> str:
    state = get_approval_state(approval_id)
    if state == ApprovalState.APPROVED:
        return "approved"
    elif state == ApprovalState.DENIED:
        return "denied"
    elif state == ApprovalState.TIMEOUT:
        return "timeout"
    return "pending"


# ===== tool_executor.py（安全检查）=====
def execute_tool(tool_name: str, args: dict):
    # 1. 检查工具类型
    tool_type = classify_tool(tool_name)

    # 2. 写操作需要审批
    if tool_type == "write":
        if not check_auto_approve(tool_name):
            result = request_approval(
                action=f"{tool_name}: {args}",
                risk_level="high",
                details=str(args)
            )
            if result != "approved":
                return json.dumps({"error": f"Approval denied: {result}"})

    # 3. 路径验证
    if "path" in args:
        error = validate_within_dir(Path(args["path"]), allowed_root)
        if error:
            return json.dumps({"error": error})

    # 4. 执行
    return registry.dispatch(tool_name, args)
```

---

## 10. 练习题

### 练习 1：追踪路径限制的完整生效路径（入门）
```
目标：理解路径验证如何嵌入工具执行

步骤：
1. 找到 validate_within_dir() 的调用位置
2. 追踪 write_file 如何调用它
3. 尝试访问被阻止的路径，观察错误
4. 理解 relative_to() 的防护原理

产出物：路径验证调用链图
```

### 练习 2：触发并观察 Approval 流程（进阶）
```
目标：理解审批状态机

步骤：
1. 配置 require_approval for write 操作
2. 发送一个需要写入的任务
3. 观察审批请求的生成和等待
4. approve / deny / timeout 分别观察结果

产出物：Approval 状态机时序图
```

### 练习 3：审计所有安全检查点（高级）
```
目标：全面理解多层防护

步骤：
1. 找出所有调用 classify_tool() 的位置
2. 找出所有调用 validate_within_dir() 的位置
3. 找出所有调用 detect_dangerous_command() 的位置
4. 画出完整的安全检查网络图

产出物：安全检查网络图 + 漏洞分析
```
