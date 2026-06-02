# 核心模块学习 — Runtime / Provider

## 1. 这个模块解决什么问题

**问题**：Agent 如何支持这么多模型（Anthropic、OpenAI、Google、Azure、Gemini、Codex）？模型切换怎么做到无损的？

**答案**：ProviderProfile 声明式配置 + Adapter 模式，每个模型有自己的 adapter，运行时按配置选择。

---

## 2. 真实源码位置（已验证）

```
providers/base.py                 ← ProviderProfile 数据类（声明式配置）
agent/anthropic_adapter.py        ← Anthropic 模型适配器
agent/openai_adapter.py           ← OpenAI 模型适配器（待验证）
agent/gemini_native_adapter.py    ← Gemini 模型适配器
agent/bedrock_adapter.py          ← AWS Bedrock 适配器
agent/azure_identity_adapter.py   ← Azure 适配器
agent/gemini_cloudcode_adapter.py ← Google Cloud Code 适配器
agent/codex_responses_adapter.py  ← Codex 适配器
```

**重要发现**：adapter 在 `agent/` 目录下，不在 `providers/` 下。`providers/base.py` 只定义了 `ProviderProfile` 数据类，真正的客户端构造和调用逻辑在各个 adapter 文件中。

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# providers/base.py
@dataclass
class ProviderProfile:
    """声明式 provider 配置"""
    name: str
    api_mode: str = "chat_completions"
    base_url: str = ""
    auth_type: str = "api_key"
    fallback_models: tuple = ()
    default_aux_model: str = ""  # 辅助模型（压缩用）
    ...

# agent/anthropic_adapter.py（示例）
class AnthropicAdapter:
    """Anthropic 模型适配器"""
    def __init__(self, agent):
        self.agent = agent

    def chat(self, messages, **kwargs):
        """调用 Anthropic API"""
        ...

    def convert_to_anthropic_format(self, messages):
        """Hermes messages → Anthropic 格式"""
        ...
```

**关键发现**：`ProviderProfile` 是**声明式配置**，不持有客户端。真正的 HTTP 调用和格式转换在各个 adapter 中。

---

## 4. 调用链

```
AIAgent._make_api_call()
  │
  ├─► 根据 agent.provider 选择 adapter
  │       ├─► "anthropic"  → AnthropicAdapter
  │       ├─► "openai"    → OpenAIAdapter
  │       ├─► "gemini"    → GeminiNativeAdapter
  │       └─► ...
  │
  ▼
adapter.chat(messages)
  │
  ├─► convert_to_provider_format(messages)  # Hermes → Provider 格式
  ├─► build_request(...)
  ├─► HTTP POST / Response
  └─► convert_from_provider_format(response)  # Provider → Hermes 格式
  │
  ▼
返回 Hermes 格式的 response（含 tool_calls）
```

---

## 5. 输入和输出

```
输入（adapter.chat）：
  - messages: List[dict]（Hermes 格式）
  - tools: List[dict]（工具 schema）
  - model: str
  - max_tokens: int
  - temperature: float
  - ...

输出：
  - response: dict（含 content, tool_calls, usage 等）

ProviderProfile 配置输入：
  - base_url: str
  - api_key: str（从环境变量或配置读取）
  - model: str
  - max_tokens: int
  - ...
```

---

## 6. 和其他模块的关系

```
Provider 依赖：
  ├─► 模型 API（Anthropic/OpenAI/Gemini 官方 SDK）
  ├─► agent/usage_pricing.py    ← 使用量估算
  ├─► agent/retry_utils.py      ← 重试逻辑
  └─► hermes_logging.py        ← 日志

其他模块依赖 Provider：
  ├─► conversation_loop.py      ← _make_api_call() 调用
  ├─► context_compressor.py     ← 辅助模型调用（用 default_aux_model）
  └─► cli.py / gateway/run.py   ← 初始化时选择 provider
```

---

## 7. 设计亮点

### 亮点 1：ProviderProfile 声明式配置
```python
# providers/base.py：
@dataclass
class ProviderProfile:
    """Provider profiles are DECLARATIVE — they describe the provider's
    behavior. They do NOT own client construction, credential rotation,
    or streaming. Those stay on AIAgent."""
```
声明式配置使得添加新 provider 只需要定义新的 ProviderProfile，不需要修改核心逻辑。

### 亮点 2：辅助模型分离
```python
@dataclass
class ProviderProfile:
    default_aux_model: str = ""  # cheap model for auxiliary tasks
```
压缩、视觉等辅助任务用廉价模型，节省成本。

### 亮点 3：多模型支持
```python
# 支持的模型（部分）：
- Anthropic: claude-3-5-sonnet, claude-3-opus, etc.
- OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.
- Google: gemini-1.5-pro, gemini-1.5-flash, etc.
- AWS Bedrock: claude on Bedrock, etc.
```
统一接口，模型可切换。

---

## 8. 风险和不足

- **Adapter 代码重复**：各个 adapter 之间有一些重复的请求构建、错误处理逻辑
- **ProviderProfile 和 Adapter 分离**：配置和实现分离增加了理解成本
- **API 版本差异**：各 provider API 版本和功能差异需要 adapter 处理兼容

---

## 9. 最小实现伪代码

```python
@dataclass
class ProviderProfile:
    name: str
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    default_aux_model: str = ""


class AnthropicAdapter:
    def __init__(self, profile: ProviderProfile):
        self.profile = profile
        self.client = Anthropic(api_key=profile.api_key)

    def chat(self, messages, tools=None):
        # Hermes → Anthropic 格式
        anthropic_messages = self._convert(messages)

        response = self.client.messages.create(
            model=self.profile.model,
            max_tokens=self.profile.max_tokens,
            messages=anthropic_messages,
            tools=tools,
        )

        # Anthropic → Hermes 格式
        return {
            "content": response.content[0].text if response.content else "",
            "tool_calls": self._extract_tool_calls(response),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }


class OpenAIAdapter:
    """同上，类似结构"""
    ...


def create_adapter(provider_name: str, profile: ProviderProfile):
    adapters = {
        "anthropic": AnthropicAdapter,
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
    }
    return adapters[provider_name](profile)
```

---

## 10. 练习题

### 练习 1：对比两个 Adapter 的调用差异（入门）
```
目标：理解不同模型 API 的格式差异

步骤：
1. 打开 agent/anthropic_adapter.py
2. 对比 agent/openai_adapter.py（如果存在）
3. 找出消息格式转换的差异

产出物：格式差异对照表
```

### 练习 2：追踪模型切换流程（进阶）
```
目标：理解 provider 切换时发生了什么

步骤：
1. 在 cli.py 中尝试切换不同 provider
2. 观察 AIAgent 初始化时 adapter 的创建
3. 追踪 _make_api_call() 中的 adapter 选择逻辑

产出物：Provider 切换流程图
```

### 练习 3：实现自定义 Provider（高级）
```
目标：添加一个新模型的支持

步骤：
1. 定义 ProviderProfile
2. 实现 Adapter 类（chat 方法）
3. 注册到 adapter 映射表
4. 测试

产出物：一个新模型的 adapter 实现
```
