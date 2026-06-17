# 学习笔记 4 — Day 7:Skills 系统的 4 个核心函数

> 日期:2026-06
> 主题:`agent/skill_commands.py` 入口 4 函数的"书店"类比
> 范围:Day 7 阶段总结

---

## 🎯 一句话总结

> **Skills 系统 = 一家书店**:`scan` 进货,`get` 查库存,`resolve` 路由,`build` 打包。4 个函数一条线把"用户打 `/foo`"变成"LLM 收到完整 user message"。

---

## 📋 4 函数速查

| 函数 | 类比 | 做什么 |
|---|---|---|
| `scan_skill_commands` | 进货 | 扫所有 `SKILL.md`,注册成 `/<slug>` 命令,塞进全局缓存 |
| `get_skill_commands` | 前台 | 缓存层 —— 智能判断要不要重扫(空 / platform 变) |
| `resolve_skill_command_key` | 路由 | 用户输入 `/claude_code` → 规范 key `/claude-code` |
| `build_skill_invocation_message` | 打包 | 加载 skill + 拼 user message,送进 conversation_loop |

---

## 🏪 书店类比展开

### 1. `scan_skill_commands` —— 进货
- 扫仓库(本地 + 外部)所有 `SKILL.md`
- 抽 name / description / platform / disabled
- **规范化 slug**(小写、空格/下划线转连字符、去非法字符)
- 摆上货架:**全量替换** `_skill_commands` 缓存
- 关键设计:局部异常吞掉(一个坏 skill 不让所有书架空)

### 2. `get_skill_commands` —— 前台
- 客人来问书,先看库存
- **不每次重扫**(慢),只在两种情况重扫:
  - 缓存空
  - platform 变了(gateway 同时服务 Telegram + Discord 场景)
- 与 `skill_bundles.get_skill_bundles()` 区别:
  - bundle 走 **mtime** 检测
  - command 走 **platform** 检测(#14536)

### 3. `resolve_skill_command_key` —— 路由
- 客人说"我要 `/claude_code`"(Telegram 强制下划线)
- 但书架上注册的是 `/claude-code`
- **下划线 → 连字符**归一化,然后查表
- 返规范 key 或 None

### 4. `build_skill_invocation_message` —— 打包
- 拿到书 + 客人要夹的纸条(user_instruction)
- 调 `_load_skill_payload` 真去仓库取
- **埋点**:`bump_use(skill_name)` 记录使用次数(#17782 Curator 生命周期)
- 拼 `activation_note` 显式告诉 LLM "用户要你按 skill 内容做"
- 调 `_build_skill_message` 拼最终 user message
- 送进 conversation_loop

---

## 🔄 完整调用链

```
用户输入 /github-code-review
  ↓
1. parse_skill_invocation 解析(还没看)
  ↓
2. resolve_skill_command_key("/github-code-review")
   → 查 get_skill_commands() 表
   → 返 "/github-code-review" 或 None
  ↓
3. build_skill_invocation_message("/github-code-review")
   → _load_skill_payload 加载 SKILL.md
   → bump_use 埋点
   → 拼 activation_note + skill body
   → 返 user message 字符串
  ↓
4. conversation_loop 把这个 message 喂给 LLM
  ↓
LLM 看到"[IMPORTANT: 用户调了 ... skill,按它做]" + 完整 SKILL.md
```

---

## 🧠 4 个关键设计点

| 设计 | 价值 |
|---|---|
| **全量替换缓存**(scan) | 简单可靠,无陈旧条目 |
| **Platform 触发重扫**(get) | 多平台 gateway 场景 |
| **下划线/连字符等价**(resolve) | 兼容 Telegram bot command |
| **activation_note 前置**(build) | 显式引导 LLM 行为 |

---

## 📍 读完入口函数后的下一步

`skill_commands.py` 还有 3 个辅助函数没看:
- `_load_skill_payload` —— 加载单个 skill(被 build 调用)
- `_inject_skill_config` —— 注入 skill 特定配置
- `_build_skill_message` —— 拼单条 message(被 build 和 preload 调用)
- `build_preloaded_skills_prompt` —— `-s` CLI flag 预加载

Day 7 接下来看哪个:
- `agent/skill_preprocessing.py`(模板变量 + 内联 shell)
- 或者 Day 7 收官 + Day 8
