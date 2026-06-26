# 学习笔记 7 — Day 14

## 1. 四个关键能力

`hermes_logging.py` 500 行,唯一公开入口 `setup_logging()`,CLI 和 gateway 启动早期各调一次。围绕 **"4 个日志文件 + 4 项能力"** 展开。

### ① `RotatingFileHandler` —— 自动轮转

日志写到一定大小自动切文件,不会无限膨胀。`agent.log` / `errors.log` / `gateway.log` / `gui.log` 都用这个 handler。

### ② `RedactingFormatter` —— **关键安全设计**

日志写入前**先脱敏**,敏感信息(API key、token)**永远不上盘**。所以你能放心把日志贴给别人排查问题。这是**最末梢的防线** —— 就算代码里有 `logger.info(f"token={token}")` 也写不出来。

### ③ Session 上下文关联

```python
set_session_context(session_id)    # 一段对话开始时
clear_session_context()             # 结束时清掉
```

调了之后,该线程所有日志自动带 `[session_id]` 前缀,排查时能按 session 过滤。背后是 `threading.local()` —— 每个线程独立存自己的 session_id,异步 gateway 多 session 并发不串。

### ④ 静默 noisy 第三方库

```python
_NOISY_LOGGERS = ("openai", "httpx", "httpcore", "asyncio", "hpack", ...)
```

把这些库拉到 WARNING 以上,不被它们的 DEBUG 噪音淹没。

### 核心招式

**"一个入口 + 文件分流 + 脱敏 + session 标记"四件套**:

- **一个入口** —— `setup_logging()` 幂等,`force=True` 可重置
- **文件分流** —— 按 logger 名字分文件(`gateway.*` → gateway.log,`hermes_cli.*` → gui.log,其他都进 agent.log)
- **脱敏** —— `RedactingFormatter` 在写入前过滤,防线在最末梢
- **session 关联** —— `threading.local()` 给多 session gateway 切片

**为什么这样设计?**

- **排查效率**:`errors.log` 一眼扫完今天所有 warning,不用在 `agent.log` 里 grep
- **审计需要**:脱敏让日志可分享、可上传 issue
- **多 session 隔离**:gateway 同时跑 N 个对话,日志能按 session 切片

### 一个简单的例子

```python
# 启动时调一次
from hermes_logging import setup_logging, set_session_context, get_logger
setup_logging(mode="gateway")              # 幂等,放心调

# 对话开始时
set_session_context("sess-abc-123")
log = get_logger(__name__)
log.info("user asked to deploy")           # 写 agent.log,带 [sess-abc-123]
log.info("token=sk-xxx")                   # 写盘前被 RedactingFormatter 替换成 sk-***

# 对话结束时
clear_session_context()
```

---

## 2. trajectory_compressor.py — 离线训练数据压缩

1508 行,**训练数据预处理工具**,不是 agent 运行时用的。

### 2.1 轨迹是什么

```json
{"messages": [
  {"role": "system",    "content": "你是 Hermes..."},
  {"role": "user",      "content": "帮我部署到 Modal"},
  {"role": "assistant", "tool_calls": [{"function": {"name": "terminal", "arguments": "ls"}}]},
  {"role": "tool",      "content": "file1.py\nfile2.py"},
  ... (可能 100+ 轮)
  {"role": "assistant", "content": "部署完成"}
]}
```

一条 JSONL = **一段 agent 在某任务上的"完整录像"**(思考 + 行动 + 结果,带因果)。

### 2.2 压缩策略(6 步)

```
1. 保护前几轮     (system, human, 第一条 gpt, 第一个 tool)  ← 必看
2. 保护最后 N 轮  (最终动作和结论)                          ← 必看
3. 只压"中间"    (从第 2 个 tool response 开始)
4. 只压到刚好达标(target_max_tokens)
5. 压缩区换成一条 human summary 消息
6. 其余 tool calls 保持完整(summary 之后模型继续工作的部分不能丢)
```

**核心**:**前后保留 + 中间摘要**,像写文章摘要 —— 头尾最重要,中间可扔。

### 2.3 轨迹的三大用途

|用途|说明|
|------|------|
|**训练数据**(主要)|每一步 agent 决策 = 示范样本,SFT 最理想;压到 16000 token 降本 5× 保留 90% 信号|
|**评估 / 评分**|跑一批任务,看 agent 用了几轮、调了哪些工具、哪步出错 → 评测报告|
|**Debug / 复现**|生产轨迹拉出来看:幻觉从哪步开始?哪个工具返回触发了错误决策?|

### 2.4 使用方式(CLI)

```bash
# 压一个目录里所有 JSONL
python trajectory_compressor.py --input=data/my_run

# 压单文件,目标 16000 tokens
python trajectory_compressor.py --input=trajectories.jsonl --target_max_tokens=16000

# 抽 15% 采样压(降本)
python trajectory_compressor.py --input=trajectories.jsonl --sample_percent=15
```

### 2.5 关键模块

- `fire` —— Google CLI 框架(参数自动绑函数)
- `rich.progress` —— 进度条(批量压很多文件时看进度)
- `agent.retry_utils.jittered_backoff` —— 重试时加抖动,防雪崩
- `utils.base_url_host_matches` —— 看 API 是不是走 OpenRouter 等代理

### 2.6 核心招式

**"完整轨迹 = 训练金矿,压缩 = 降本不减质"**:

- **完整轨迹** 价值最高(token 消耗也最高)
- **前后保留**保住训练信号的因果链
- **中间摘要**把探查/迭代/重试这些"过程噪声"压成一句
- **目标驱动**只压到刚好达标,不多压

**它不是 agent 运行时用的**(运行时是 `hermes_state.py` 里的对话压缩),而是**离线训练数据预处理工具** —— 把生产环境跑出来的轨迹压成"适合训练"的样本。

### 2.7 一句话

**轨迹 = agent 的"操作录像"**;主要用来训练(教模型怎么做)、评估(测模型做得多好)、复现(debug 歪了的行为);`trajectory_compressor.py` 把录像压成"够用就好"的训练样本。

---

## 3. batch_runner.py — 批量跑 agent 的并行执行器

1321 行,**离线评测/数据生产**工具,不是日常 agent 用的。

### 3.1 五步流水线(数据流)

```
数据集(data.jsonl)
    ↓ ① 加载 + 分批
[[batch_0: prompt_0..9], [batch_1: prompt_10..19], ...]
    ↓ ② multiprocessing.Pool 分发给 N 个 worker
[worker_0 跑 batch_0]  [worker_1 跑 batch_1]  ...
    ↓ ③ 每个 batch 跑完 → 写 checkpoint
[checkpoint_0.json, checkpoint_1.json, ...]
    ↓ ④ 同时写轨迹(从 messages 转 from/value)
[trajectories.jsonl]
    ↓ ⑤ 从 messages 抽工具统计,合并
[tool_stats.json]
```

### 3.2 五大能力详解

#### ① 数据集加载与分批

**入口** [`BatchRunner._load_dataset`](hermes-agent/batch_runner.py#L642) 和 [`_create_batches`](hermes-agent/batch_runner.py#L674):

```python
def _load_dataset(self) -> List[Dict[str, Any]]:
    with open(self.dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

def _create_batches(self):
    indexed = list(enumerate(self.dataset))   # 保留 index 是关键(checkpoint 用)
    return [indexed[i:i + self.batch_size] for i in range(0, len(indexed), self.batch_size)]
```

**输入**:

```json
{"prompt": "Fix the bug in foo.py", "expected": "..."}
{"prompt": "Add tests for bar.py", "expected": "..."}
```

#### ② 多进程并行

**入口** [`_process_batch_worker`](hermes-agent/batch_runner.py#L400) 和 [`BatchRunner.run`](hermes-agent/batch_runner.py#L810):

```python
with multiprocessing.Pool(processes=num_workers) as pool:
    results = pool.map(_process_batch_worker, batches)
```

**为什么用 multiprocessing 而不是 asyncio?**

|维度|multiprocessing|asyncio|
|---|---|---|
|并行|**真多核**(绕过 GIL)|单线程,IO 等待时切|
|适合|CPU 密集(模型推理)|IO 密集(网络)|
|启动成本|高(每个 worker 拷内存)|低|

agent 跑模型推理是 **CPU+GPU 密集**,所以 multiprocessing 才对路。

#### ③ Checkpoint 持久化

**入口** [`_save_checkpoint`](hermes-agent/batch_runner.py#L715) 和 [`_load_checkpoint`](hermes-agent/batch_runner.py#L688):

```python
def _save_checkpoint(self, checkpoint_data, lock=None):
    # 写到 ~/.hermes/runs/<run_name>/checkpoints/
    # lock 防多 worker 同时写
    ...

def _scan_completed_prompts_by_content(self) -> set:
    # 启动时扫所有 checkpoint,提取已完成的 prompt(用内容去重)
    # 这样 --resume 时跳过的不是 index,而是"内容已完成的"
    ...
```

**关键设计**:**按内容去重而非按 index** —— 数据集文件被改了顺序/插了行,resume 仍然准确。

```bash
# 跑挂了,resume
python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
# → 自动扫 ~/.hermes/runs/my_run/checkpoints/,跳过已完成的 prompt
```

#### ④ 轨迹保存(给训练用)

**入口** [`_process_single_prompt`](hermes-agent/batch_runner.py#L244):

```python
def _process_single_prompt(self, prompt_data, batch_idx, prompt_idx):
    agent = AIAgent(...)
    messages = agent.run(prompt_data["prompt"])     # 跑 prompt,拿到完整 messages
    trajectory = convert_messages_to_from_value(messages)  # 转训练格式
    # 写 JSONL
```

**`from/value` 是什么?** 模型训练格式(HF、CogComp 等历史约定):

```json
{"from": "system", "value": "你是 Hermes..."}
{"from": "human",  "value": "帮我部署"}
{"from": "gpt",    "value": "tool_call: terminal, ls"}
{"from": "tool",   "value": "file1.py file2.py"}
{"from": "gpt",    "value": "部署完成"}
```

为什么不用 `role/content`?训练框架历史上用的是 `from/value`,统一这个 `trajectory_compressor.py` 直接能吃。

#### ⑤ 工具调用统计聚合

**入口** [`_extract_tool_stats`](hermes-agent/batch_runner.py#L125) 和 [`_extract_reasoning_stats`](hermes-agent/batch_runner.py#L208):

```python
def _extract_tool_stats(messages):
    stats = {}  # {"terminal": {"calls": 50, "errors": 3, "tokens": 12000}, ...}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                stats.setdefault(tool_name, {"calls": 0, "errors": 0, "tokens": 0})
                stats[tool_name]["calls"] += 1
        if msg.get("role") == "tool" and "error" in msg.get("content", "").lower():
            stats[tc_name]["errors"] += 1
    return stats
```

**聚合输出** `tool_stats.json`:

```json
{
  "terminal":  {"calls": 4820, "errors": 23, "tokens": 1200000},
  "read_file": {"calls": 3105, "errors": 5,  "tokens": 800000},
  "edit_file": {"calls": 892,  "errors": 2,  "tokens": 400000}
}
```

**这是评估报告的原料** —— 看 agent 在任务集上"主要靠哪些工具"、"哪些工具爱出错",指导后续优化(发现 `edit_file` 失败率高 → 改提示词或加校验)。

### 3.3 核心招式

**"分批 + 并行 + checkpoint = 长任务能跑完"**,**"轨迹 + 统计 = 跑完有用"**。前后两段各管一摊,中间靠 multiprocessing 串起来。

### 3.4 和其他文件的关系

```text
batch_runner.py          (并行批量执行)
    ↓ 输出
trajectory_compressor.py (压成训练样本)
    ↓ 输出
训练流水线

# 同期还用到:
run_agent.py            # 真正的 agent 实现(AIAgent 类)
toolset_distributions   # 工具集配置(测不同工具下的表现)
hermes_state.py         # 对话状态(每个 batch 实例存自己 session)
usage_pricing.py        # 算成本(统计这次跑了多少钱)
hermes_logging.py       # 日志(批量跑的进度和错误)
```

### 3.5 一句话

**batch_runner.py = 离线评测/数据生产的并行执行器**,把数据集跑成轨迹 + 统计,服务于训练和回归测试,**不是给日常 agent 用的**。
