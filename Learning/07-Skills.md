# 核心模块学习 — Skills / Procedural Memory

## 1. 这个模块解决什么问题

**问题**：如何让 Agent 学会可复用的技能（如代码审查、PR 流程）？Skill 如何被发现和加载？

**答案**：Skills 是一组 Markdown 文件（SKILL.md），通过 frontmatter 定义元数据。`skills_tool.py` 提供查看和加载功能。`skill_bundles.py` 支持多个 skill 组合成 bundle。

---

## 2. 真实源码位置（已验证）

```
skills/                          ← Skills 目录（20+ 子目录）
tools/skills_tool.py            ← Skills 工具（查看/加载）
agent/skill_bundles.py          ← Skill Bundles（组合多个 skills）
agent/skill_utils.py            ← Skill 工具函数
agent/skill_commands.py          ← Skill 命令处理
agent/skill_preprocessing.py     ← Skill 预处理
agent/skill_bundles.py          ← Bundle 管理
skills/index-cache/             ← Skill 索引缓存
```

**重要发现**：
- Skills 格式兼容 agentskills.io 标准（YAML frontmatter）
- Skills 目录结构：`skills/<category>/<skill-name>/SKILL.md`
- `skill_bundles.py` 支持 bundle（多个 skill 组合成一个 slash 命令）

---

## 3. 核心类 / 函数 / 方法（已验证）

```python
# agent/skill_bundles.py
class SkillBundle:
    """Bundle 元数据"""
    name: str
    description: str
    skills: List[str]   # skill slug 列表
    instruction: str    # 额外引导

def get_skill_bundles() -> Dict[str, SkillBundle]
def resolve_bundle_command_key(command: str) -> Optional[str]
def build_bundle_invocation_message(bundle_slug: str) -> str
def reload_bundles() -> DiffResult
def list_bundles() -> List[BundleInfo]

# agent/skill_utils.py
def get_all_skills_dirs() -> List[Path]
def iter_skill_index_files() -> Iterator[Path]
def parse_frontmatter(content: str) -> dict
def get_disabled_skill_names() -> Set[str]
def extract_skill_conditions(skill_path: Path) -> dict
def skill_matches_platform(skill: dict) -> bool

# tools/skills_tool.py
class SkillManagerTool:
    """Skill 工具类"""
    def skills_list(args) -> str   # 列出所有 skill
    def skill_view(args) -> str    # 查看单个 skill 内容
```

**SKILL.md Frontmatter 格式（agentskills.io 兼容）：**
```yaml
---
name: skill-name              # Required, max 64 chars
description: Brief description # Required, max 1024 chars
version: 1.0.0
platforms: [macos, linux]      # Optional
prerequisites:
  env_vars: [API_KEY]
  commands: [curl, jq]
---
# Skill Title
Skill content here...
```

---

## 4. 调用链

```
用户输入 /<skill-name> 或 /<bundle-name>
  │
  ▼
cli.py / gateway: 解析 slash 命令
  │
  ├─► 检查 bundles（优先）→ skill_bundles.py
  │       └─► build_bundle_invocation_message()
  │               └─► 加载所有 bundle 内的 skills
  │
  └─► 检查 individual skills → skill_utils.py
          └─► iter_skill_index_files()
                  └─► 扫描 skills/ 目录
  │
  ▼
生成 user message，包含 skill 内容
  │
  ▼
注入 conversation_loop（作为普通 user message）
  │
  ▼
模型看到 skill 内容，执行相应指导
```

**Skill 在 System Prompt 中的位置**：
- 通过 `SKILLS_GUIDANCE` 注入 stable 层
- 通过 `/<skill-name>` 命令以 user message 方式加载

---

## 5. 输入和输出

```
Skill 加载：
  输入：/github-code-review（用户 slash 命令）
  输出：user message，包含 SKILL.md 的完整内容

Skill 列出：
  输入：无参数
  输出：所有可用 skills 的列表（名称 + 描述 + 平台）

Bundle 加载：
  输入：/backend-dev（bundle 名）
  输出：user message，包含 bundle 内所有 SKILL.md 的内容
```

---

## 6. 和其他模块的关系

```
Skills 依赖：
  ├─► skills/ 目录结构
  ├─► hermes_constants.py（SKILLS_DIR 路径）
  └─► tools/threat_patterns.py（扫描 SKILL.md 内容）

其他模块依赖 Skills：
  ├─► system_prompt.py（SKILLS_GUIDANCE 注入 stable 层）
  ├─► conversation_loop.py（slash 命令解析）
  └─► cli.py（slash 命令处理）
```

---

## 7. 设计亮点

### 亮点 1：agentskills.io 兼容格式
```yaml
---
name: skill-name
description: Brief description
platforms: [macos, linux]
prerequisites:
  env_vars: [API_KEY]
  commands: [curl, jq]
---
```
标准格式，便于从 agentskills.io 导入现成 skills。

### 亮点 2：Bundle 组合
```python
# agent/skill_bundles.py：
# "If a bundle and a skill share the same slash name,
# the bundle wins. The slash command dispatch checks
# bundles first, then falls back to skills."
```
bundle 优先于同名 skill，支持组合复用。

### 亮点 3：平台过滤
```python
def skill_matches_platform(skill: dict) -> bool:
    platforms = skill.get("platforms", [])
    if not platforms:
        return True
    return current_platform in platforms
```
Skill 可以限定平台（macos/linux/windows）。

---

## 8. 风险和不足

- **Skill 存储在本地**：没有远程同步机制（index-cache 是缓存，不是同步）
- **Frontmatter 解析**：依赖特定格式，解析失败可能静默忽略
- **Skill 版本管理**：没有版本控制，更新可能破坏现有行为

---

## 9. 最小实现伪代码

```python
from pathlib import Path
import re

FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def parse_skill(path: Path) -> dict:
    content = path.read_text()
    match = FRONTMATTER_RE.match(content)
    if match:
        frontmatter = yaml.safe_load(match.group(1))
        body = content[match.end():]
    else:
        frontmatter = {}
        body = content

    return {
        **frontmatter,
        "body": body,
        "path": path,
    }

def get_all_skills(skills_dir: Path) -> List[dict]:
    skills = []
    for skill_path in skills_dir.glob("*/SKILL.md"):
        skill = parse_skill(skill_path)
        skill["slug"] = skill_path.parent.name
        skills.append(skill)
    return skills

def invoke_skill(skill_slug: str) -> str:
    """生成 skill 的 user message 内容"""
    skills_dir = Path("~/.hermes/skills").expanduser()
    skill = parse_skill(skills_dir / skill_slug / "SKILL.md")
    return skill["body"]
```

---

## 10. 练习题

### 练习 1：分析 Skill 目录结构（入门）
```
目标：理解 skills 目录的组织方式

步骤：
1. 查看 skills/ 目录结构
2. 找一个具体 skill 的 SKILL.md
3. 解析 frontmatter
4. 理解 platform 和 prerequisites 字段

产出物：目录结构图 + 一个 skill 的 frontmatter 解析
```

### 练习 2：创建一个自定义 Skill（进阶）
```
目标：亲手创建一个可用的 Skill

步骤：
1. 在 skills/ 下创建目录 my-skill/
2. 写 SKILL.md，包含 name, description, platforms, content
3. 启动 hermes，用 /skills 查看是否出现
4. 用 /my-skill 加载它

产出物：自定义 SKILL.md 文件
```

### 练习 3：实现 Skill Bundle（高级）
```
目标：理解 bundle 的组合机制

步骤：
1. 查看 agent/skill_bundles.py 的 Bundle 格式
2. 创建 ~/.hermes/skill-bundles/my-bundle.yaml
3. 定义 bundle 包含多个 skills
4. 用 /my-bundle 测试

产出物：bundle YAML 文件 + 调用测试
```
