# 核心模块学习 — Observability / Logging / Trace

## 1. 这个模块解决什么问题

**问题**：如何追踪 Agent 的行为？如何调试问题？如何统计使用量和成本？

**答案**：`hermes_logging.py` 提供集中式日志配置，`trajectory_compressor.py` 保存训练轨迹，`batch_runner.py` 支持批量评估。

---

## 2. 真实源码位置（已验证）

```
hermes_logging.py               ← 集中日志配置
agent/usage_pricing.py         ← 使用量估算
agent/account_usage.py         ← 账户使用量
trajectory_compressor.py        ← 轨迹压缩（RL 训练数据）
batch_runner.py                ← 批量评估运行器
hermes_state.py                ← Session 历史
agent/conversation_loop.py     ← 内含 trace 日志
```

**重要发现**：
- 日志使用 `RotatingFileHandler` + `RedactingFormatter`（敏感信息不落盘）
- Session Replay 通过 `hermes_state.py` 的 `get_session_messages()` 实现
- Trajectory 压缩保护训练数据的第一轮和最后几轮，中间压缩

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# hermes_logging.py
_session_context = threading.local()  # per-conversation session context

def setup_logging(
    mode: str = "cli",     # "cli" | "gateway" | "gui"
    log_dir: str = "~/.hermes/logs",
    level: str = "INFO",
    force: bool = False,
):
    """集中日志配置入口，CLI 和 Gateway 启动时调用"""

def set_session_context(session_id: str)
def clear_session_context()

class RedactingFormatter(logging.Formatter):
    """过滤敏感信息（API keys, tokens）"""

def _install_session_record_factory():
    """每条日志包含 session_id"""

# agent/usage_pricing.py
def estimate_usage_cost(model, input_tokens, output_tokens, provider) -> dict
    """估算 API 使用成本"""

def normalize_usage(usage: dict) -> dict

# trajectory_compressor.py
class TrajectoryCompressor:
    """压缩训练轨迹，保护首尾轮"""

def compress_trajectory(trajectory: List[dict], target_max_tokens: int) -> List[dict]
    """按 token 预算压缩轨迹"""

# batch_runner.py
def run_batch(tasks: List[dict], model: str, provider: str, output_dir: str)
    """批量运行评估任务"""

# hermes_state.py
def get_session_messages(session_id: str) -> List[dict]
def replay_session(session_id: str)
```

---

## 4. 调用链

```
Hermes 启动：
  │
  ├─► cli.py → setup_logging(mode="cli")
  ├─► gateway/run.py → setup_logging(mode="gateway")
  │
  ▼
  所有模块的 logger.info() / logger.debug() 等
  │
  ├─► RotatingFileHandler → ~/.hermes/logs/agent.log
  ├─► RedactingFormatter（过滤敏感信息）
  └─► session_id tag（通过 _install_session_record_factory）

Conversation 运行：
  │
  ├─► conversation_loop.py 日志：
  │       ├─► API 请求/响应（DEBUG 级别）
  │       ├─► 工具调用（INFO 级别）
  │       └─► Token 使用量（INFO 级别）
  │
  ├─► tool_executor.py 日志：
  │       ├─► 工具开始/完成
  │       └─► 工具耗时
  │
  └─► hermes_state.py：
          └─► Session 消息持久化

Trajectory 保存（batch_runner）：
  │
  └─► trajectory_compressor.py
          ├─► 保护第一轮（system, human, first assistant）
          ├─► 保护最后 N 轮
          └─► 压缩中间轮
```

---

## 5. 输入和输出

```
setup_logging：
  输入：mode, log_dir, level, force
  输出：配置生效（日志写入文件）

estimate_usage_cost：
  输入：model, input_tokens, output_tokens, provider
  输出：{"input_cost": "$X.XXX", "output_cost": "$X.XXX", "total": "$X.XXX"}

TrajectoryCompressor.compress：
  输入：trajectory: List[dict], target_max_tokens: int
  输出：压缩后的 trajectory

get_session_messages：
  输入：session_id: str
  输出：List[dict]（该 session 的所有消息）
```

---

## 6. 和其他模块的关系

```
Observability 被其他模块使用：
  ├─► conversation_loop.py     ← 日志主要产生地
  ├─► tool_executor.py         ← 工具执行日志
  ├─► tools/*.py               ← 各工具日志
  ├─► model_tools.py           ← 工具 schema 日志
  └─► cli.py / gateway/run.py  ← 启动时初始化

Observability 依赖：
  ├─► hermes_state.py         ← Session 消息查询
  └─► 使用量数据来自 API 响应
```

---

## 7. 设计亮点

### 亮点 1：RedactingFormatter 敏感信息不落盘
```python
# hermes_logging.py 注释：
# "All log files use RotatingFileHandler with RedactingFormatter
# so secrets are never written to disk."
```
API keys、tokens 等敏感信息在日志中被替换为 `[REDACTED]`。

### 亮点 2：Session Context 关联
```python
# hermes_logging.py：
_session_context = threading.local()

def set_session_context(session_id: str):
    """Call at start of conversation, clear when done.
    All log lines will include [session_id] for filtering."""
```
同一次对话的所有日志通过 session_id 关联。

### 亮点 3：Trajectory 首尾保护压缩
```python
# trajectory_compressor.py 注释：
# "1. Protect first turns (system, human, first gpt, first tool)
#  2. Protect last N turns (final actions and conclusions)
#  3. Compress MIDDLE turns only"
```
训练数据压缩时保护最重要的信号。

### 亮点 4：使用量估算
```python
# agent/usage_pricing.py：
PRICING = {
    "claude-sonnet-4": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
}
```
支持多模型的成本估算。

---

## 8. 风险和不足

- **日志量巨大**：verbose 模式下日志量极大，可能影响性能
- **Trace 分散**：没有统一的 trace ID 贯穿整个请求
- **指标不完整**：缺少 P50/P90/P99 延迟分布
- **Session Replay 有限**：不支持带工具调用状态的精确回放

---

## 9. 最小实现伪代码

```python
import logging
from logging.handlers import RotatingFileHandler
import threading
import re

# ===== hermes_logging.py 简化版 =====

class RedactingFormatter(logging.Formatter):
    REDACT_PATTERNS = [
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', re.I), 'API_KEY=[REDACTED]'),
        (re.compile(r'Bearer\s+[\w-]+'), 'Bearer [REDACTED]'),
    ]

    def format(self, record):
        msg = super().format(record)
        for pattern, replacement in self.REDACT_PATTERNS:
            msg = pattern.sub(replacement, msg)
        return msg


def setup_logging(log_dir="~/.hermes/logs", level="INFO"):
    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = RedactingFormatter(
        "%(asctime)s %(levelname)s [%(session_id)s] %(name)s: %(message)s"
    )

    # 文件 handler
    file_handler = RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))

    # 根 logger
    root = logging.getLogger()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)


# ===== trace_logger.py 简化版 =====
class JSONLTraceLogger:
    def __init__(self, trace_dir="~/.hermes/traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "session_id": get_session_context(),
        }
        trace_file = self.trace_dir / f"{date.today()}.jsonl"
        with open(trace_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_api_request(self, model, messages, tools):
        self.log("api_request", {
            "model": model,
            "message_count": len(messages),
            "tool_count": len(tools) if tools else 0,
        })

    def log_tool_call(self, tool_name, args):
        self.log("tool_call", {"tool": tool_name, "args": args})
```

---

## 10. 练习题

### 练习 1：追踪一次请求的完整日志（入门）
```
目标：从日志还原完整调用链

步骤：
1. 开启 verbose 模式运行 hermes
2. 执行一个简单任务（read file）
3. 追踪日志文件，按时间顺序还原调用
4. 找出 API 请求、工具调用、结果返回的日志

产出物：还原的调用时序图
```

### 练习 2：实现使用量统计（进阶）
```
目标：统计一次 session 的 API 使用成本

步骤：
1. 在 usage_pricing.py 中实现成本计算
2. 在每次 API 调用后累计 usage
3. 在 session 结束时输出报告
4. 运行一个多 turn 对话，查看成本报告

产出物：成本分析报告
```

### 练习 3：实现 JSONL Trace Logger（高级）
```
目标：实现结构化 trace 记录

步骤：
1. 实现 JSONLTraceLogger 类
2. 在 conversation_loop.py 中植入 trace 记录
3. 实现 trace 查询工具
4. 分析某次 session 的完整 trace

产出物：Trace 分析报告 + trace 文件
```
