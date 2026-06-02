# 核心模块学习 — Evaluation / Tests / Benchmark

## 1. 这个模块解决什么问题

**问题**：如何验证 Agent 的行为正确性？如何批量评估 Agent 能力？如何生成 RL 训练数据？

**答案**：
- `tests/` 目录有完整 pytest 套件
- `batch_runner.py` 支持批量评估任务
- `trajectory_compressor.py` 生成 RL 训练轨迹

---

## 2. 真实源码位置（已验证）

```
tests/                           ← 完整测试套件
  ├── conftest.py               ← pytest fixtures
  ├── agent/                    ← Agent 核心测试
  ├── tools/                    ← 工具测试（242+）
  ├── integration/              ← 集成测试
  ├── e2e/                     ← 端到端测试
  ├── stress/                  ← 压力测试
  └── fixtures/                ← 测试数据

batch_runner.py                 ← 批量评估运行器
trajectory_compressor.py        ← 轨迹压缩
mini_swe_runner.py             ← SWE-bench 类型评估
agent/background_review.py     ← 后台自改进
```

**重要发现**：
- `batch_runner.py` 支持 multiprocessing 并行，使用 checkpointing 容错
- `trajectory_compressor.py` 专门用于 RL 训练数据压缩，保护首尾轮
- `tests/conftest.py` 有大量 fixtures，支持 mock、临时目录等

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# batch_runner.py
def run_batch(
    tasks: List[dict],
    model: str,
    provider: str,
    output_dir: str,
    batch_size: int = 10,
    resume: bool = False,
) -> List[dict]:
    """批量运行评估任务，支持 checkpoint 和 resume"""

def _run_single_task(task: dict, agent: AIAgent) -> dict

def save_results(results: List[dict], output_dir: str)

def load_checkpoint(checkpoint_path: str) -> dict

# trajectory_compressor.py
class TrajectoryCompressor:
    def compress(self, trajectory: List[dict], target_max_tokens: int) -> List[dict]
        """按 token 预算压缩轨迹，保护首尾"""

def compress_trajectory(trajectory, target_max_tokens) -> List[dict]

# tests/conftest.py
@pytest.fixture
def temp_project(tmp_path)
@pytest.fixture
def mock_agent()
@pytest.fixture
def sample_messages()
```

---

## 4. 调用链

```
批量评估：
  │
  ▼
batch_runner.run_batch(tasks, model, provider, output_dir)
  │
  ├─► 创建 AIAgent 实例池
  │
  ├─► for task in tasks:
  │       ├─► 检查 checkpoint（resume 模式）
  │       ├─► _run_single_task(task)
  │       │       ├─► agent.run_conversation(task["input"])
  │       │       ├─► evaluate(task, response)
  │       │       └─► 返回结果
  │       ├─► 保存 checkpoint
  │       └─► 聚合结果
  │
  └─► save_results(results, output_dir)

轨迹压缩：
  │
  ▼
TrajectoryCompressor.compress(trajectory, target_max_tokens)
  │
  ├─► 保护第一轮（system, human, first gpt, first tool）
  ├─► 保护最后 N 轮
  ├─► 压缩中间轮
  └─► 返回压缩后 trajectory
```

---

## 5. 输入和输出

```
run_batch：
  输入：tasks: List[dict], model: str, provider: str, output_dir: str
  输出：results: List[dict]，每项含 task_id, input, expected, actual, success, score

TrajectoryCompressor.compress：
  输入：trajectory: List[dict], target_max_tokens: int
  输出：压缩后的 trajectory

评估任务格式：
  {
    "id": "task_001",
    "input": "用户输入",
    "expected": "期望输出",
    "validation": "contains" | "matches" | "semantic_similarity",
    "category": "file_operations",
    "difficulty": "easy" | "medium" | "hard",
    "timeout": 60,
  }
```

---

## 6. 和其他模块的关系

```
Evaluation 依赖：
  ├─► AIAgent.run_conversation()
  ├─► hermes_state.py（session 历史）
  ├─► trajectory_compressor.py（轨迹压缩）
  └─► tools/（工具行为测试）

其他模块依赖 Evaluation：
  ├─► CI/CD（提交前跑测试）
  ├─► Agent 开发（验证新功能）
  └─► RL 训练（trajectory_compressor 生成数据）
```

---

## 7. 设计亮点

### 亮点 1：Checkpoint + Resume
```python
# batch_runner.py 注释：
# "Checkpointing for fault tolerance and resumption"
```
长时批量任务支持中断恢复，不丢失进度。

### 亮点 2：多进程并行
```python
# batch_runner.py：
from multiprocessing import Pool, Lock

with Pool(processes=batch_size) as pool:
    results = pool.map(run_single_task, tasks)
```
充分利用多核并行评估。

### 亮点 3：Trajectory 首尾保护压缩
```python
# trajectory_compressor.py：
# "1. Protect first turns (system, human, first gpt, first tool)
#  2. Protect last N turns (final actions and conclusions)
#  3. Compress MIDDLE turns only"
```
RL 训练数据的质量比数量更重要。

### 亮点 4：多验证策略
```python
validation = "contains" | "matches" | "semantic_similarity" | "llm_judge"
```
支持从简单字符串匹配到 LLM 评判的多种评估方式。

---

## 8. 风险和不足

- **评估覆盖不均**：文件操作覆盖率高，复杂推理场景覆盖率低
- **Ground Truth 获取**：很多任务没有客观标准答案
- **多轮评估复杂**：需要管理 session 状态和记忆
- **Benchmark 泄漏**：训练数据可能污染评估

---

## 9. 最小实现伪代码

```python
# ===== eval_runner.py 简化版 =====

EVAL_TASKS = [
    {
        "id": "file_read_001",
        "name": "读取文件",
        "category": "file_operations",
        "difficulty": "easy",
        "input": "读取 /tmp/test.txt 的内容",
        "expected": "Hello World",
        "validation": "contains",
        "setup": {"/tmp/test.txt": "Hello World"},
        "timeout": 30,
    },
]


def evaluate_response(response: str, task: dict) -> float:
    expected = task["expected"]
    validation = task["validation"]

    if validation == "contains":
        return 1.0 if expected.lower() in response.lower() else 0.0
    elif validation == "matches":
        return 1.0 if re.search(expected, response) else 0.0
    elif validation == "semantic_similarity":
        # embedding similarity
        score = cosine_sim(get_embedding(response), get_embedding(expected))
        return score
    elif validation == "llm_judge":
        score = llm_judge(f"Input: {task['input']}\nResponse: {response}")
        return score
    return 0.0


def run_eval_suite(tasks: List[dict], agent: AIAgent, output_dir: str) -> dict:
    results = []

    for task in tqdm(tasks):
        # Setup
        for path, content in task.get("setup", {}).items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)

        # Run
        try:
            response = agent.run_conversation(task["input"])
            score = evaluate_response(response, task)
        except Exception as e:
            score = 0.0
            response = f"Error: {e}"

        results.append({
            "task_id": task["id"],
            "input": task["input"],
            "expected": task["expected"],
            "actual": response,
            "score": score,
            "success": score >= 0.8,
        })

    # Save
    save_results(results, output_dir)
    return aggregate_results(results)
```

---

## 10. 练习题

### 练习 1：运行现有测试（入门）
```
目标：熟悉测试框架

步骤：
1. 找到 tests/agent/test_conversation_loop.py
2. 运行 pytest -v tests/agent/test_conversation_loop.py
3. 查看覆盖率报告
4. 找一个简单测试，理解测试结构

产出物：测试运行结果 + 测试结构分析
```

### 练习 2：编写一个工具测试（进阶）
```
目标：学习如何测试工具

步骤：
1. 在 tests/tools/ 创建测试文件
2. 使用 pytest fixtures 创建临时文件
3. 断言工具行为正确
4. 运行测试确认通过

产出物：新工具测试文件
```

### 练习 3：设计并运行一个评估套件（高级）
```
目标：完整评估一次 Agent 能力

步骤：
1. 定义 5 个评估任务（file_ops, reasoning, memory, safety, multi_tool）
2. 实现 evaluate_response() 函数
3. 用 batch_runner 运行评估
4. 分析结果，生成报告

产出物：评估报告 + 评估任务定义
```
