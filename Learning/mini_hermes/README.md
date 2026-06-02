# Mini Hermes Agent

A minimal implementation of Hermes Agent's core engineering patterns, built for learning purposes.

## Project Structure

```
mini_hermes/
├── __init__.py
├── agent.py                  # AIAgent 主类
├── conversation_loop.py       # 主循环（核心）
├── prompt_builder.py         # System prompt 三层架构
├── provider.py               # 模型调用抽象
├── tool_registry.py          # 工具注册中心
├── tool_executor.py          # 工具执行器
├── memory_manager.py         # 记忆管理
├── safety.py                # 安全检查
├── trace_logger.py          # JSONL trace logger
├── tools/
│   ├── __init__.py
│   ├── calculator.py         # 计算器
│   ├── read_file.py         # 文件读取
│   └── write_file.py        # 文件写入
└── eval_runner.py           # 评估运行器
```

## Quick Start

```python
from mini_hermes.agent import AIAgent

agent = AIAgent()
response = agent.run_conversation("What is 15 + 27?")
print(response)
```

## 核心模块对照表

| Mini Hermes | 真实 Hermes | 说明 |
|-------------|-------------|------|
| `agent.py` | `run_agent.py` | AIAgent 入口类 |
| `conversation_loop.py` | `agent/conversation_loop.py` | 主循环 |
| `prompt_builder.py` | `agent/system_prompt.py` | 三层 Prompt |
| `provider.py` | `providers/base.py` | Provider 声明式配置 |
| `tool_registry.py` | `tools/registry.py` | 工具注册中心 |
| `tool_executor.py` | `agent/tool_executor.py` | 工具执行器 |
| `memory_manager.py` | `agent/memory_manager.py` | 记忆管理器 |
| `safety.py` | `tools/path_security.py` | 路径安全 |
| `trace_logger.py` | `hermes_logging.py` | 结构化日志 |
| `eval_runner.py` | `batch_runner.py` | 批量评估 |

## 最小可用版本实现

### 已实现（必须）
- [x] Agent Loop（while not done → API call → tool call → loop）
- [x] Tool Registry（register + dispatch）
- [x] Tool Executor（sequential execution）
- [x] Simple Memory（JSON 文件持久化）
- [x] Path Restriction（validate_within_dir）
- [x] JSONL Trace Logger

### 选做（进阶）
- [ ] System Prompt 三层架构完整实现
- [ ] Provider 抽象（支持 Anthropic / OpenAI）
- [ ] Approval 状态机
- [ ] Concurrent 工具执行
- [ ] Context Compression
- [ ] Delegation / Sub-agent
- [ ] Skills 系统
- [ ] Batch eval runner

## 运行测试

```bash
# 运行评估套件
python eval_runner.py
```

## 设计原则

1. **学习优先**：代码清晰易懂，不追求性能优化
2. **模式忠实**：尽量还原真实 Hermes 的设计模式
3. **可运行**：每个模块都是可运行的，不只是伪代码
4. **简化不牺牲**：关键架构决策保留，细节简化
