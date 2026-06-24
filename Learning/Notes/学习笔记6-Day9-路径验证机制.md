# 学习笔记 6 — Day 9:路径验证机制

## 1. 检查"模型给的路径"是不是在"允许的目录"里*

`path_security.py` 就干一件事:**检查"模型给的路径"是不是在"允许的目录"里**,防止模型乱跑去读/写系统文件。

整文件 22 行,核心就 2 个函数:

```python
def validate_within_dir(path, root)   # 完整检查
def has_traversal_component(path)     # 快速预筛
```

### 一个简单的例子

```python
# 允许模型操作 ~/projects/myapp 这个目录
validate_within_dir(
    Path("~/projects/myapp/src/main.py"),  # 模型想写的文件
    Path("~/projects/myapp")                # 允许的根
)
# → None (合法,放行)

validate_within_dir(
    Path("~/projects/myapp/../../etc/passwd"),  # 试图逃出去
    Path("~/projects/myapp")
)
# → "Path escapes allowed directory: ..." (被拒)
```

**核心招式**:`resolve()` 把路径"算成"真实位置,`relative_to()` 验证它在不在根目录下。
算不出来 → 抛 `ValueError` → 被 `except` 抓住 → 报错。

---

## 2. 一个点:**tirith_security.py 就干一件事 — 在命令执行前扫一遍**

`check_command_security(command)` 把命令字符串扔给 tirith 二进制,根据**退出码**返回 `allow` / `block` / `warn`(0/1/2)。JSON stdout 只用来补充 findings/summary,**不能覆盖**退出码的判定。子进程挂了 / 超时 / 退出码未知时按 `fail_open` 配置走。

文件里那一大坨 `_install_tirith` / `_resolve_tirith_path` / `ensure_installed` 全是配套设施——保证 tirith 二进制**存在、是签名过的、能跑**。

### 一个简单的例子

```python
# approval.py 在执行 shell 命令前会调它
result = check_command_security("rm -rf /tmp/foo")
# → {"action": "block",  "findings": [...], "summary": "..."}  # 危险,被拦

result = check_command_security("ls -la")
# → {"action": "allow",  "findings": [],     "summary": ""}     # 没事,放行

result = check_command_security("curl https://example.app/api")
# → {"action": "warn" → "allow", "findings": []}                # .app TLD 误报,自动降级
```

**一句话**:退出码是**唯一真相源**,JSON 只是注释;子进程出问题(fail_open / fail_closed)才决定放不放行。

---

## 3. `file_safety.py` 四个判定函数

四个函数都是**纯查询**(无副作用),真拦截在调用方(`file_operations.py` / `file_tools.py`)的 guard clause 里。

### `is_write_denied(path) → True/False`

- **True(拦)** —— 命中三类:① 精确黑名单(SSH 密钥、shell rc、`/etc/passwd`、`auth.json` / `.env` / `.anthropic_oauth.json` 等);② 目录前缀(`~/.ssh/`、`.aws/`、`.kube/`、`/etc/sudoers.d/` 等);③ Hermes 控制面(`auth.json` / `config.yaml` / `webhook_subscriptions.json` / `mcp-tokens/` / `pairing/`,**同时拦 active home + root**)。
- **False(放)** —— 上面都没命中 **且** 在 `HERMES_WRITE_SAFE_ROOT` 之内。

### `get_read_block_error(path) → 错误字符串 / None`

- **字符串(拦)** —— 三类:① Hermes 内部缓存 `skills/.hub/`(防 prompt injection);② 凭证存储 `auth.json` / `auth.lock` / `.env` / `.anthropic_oauth.json` / `mcp-tokens/` / `cache/bws_cache.json`;③ 项目级 `.env` 系列(`.env` / `.env.local` / `.env.production` / `.envrc`,任何路径都拦,建议读 `.env.example`)。
- **None(放)** —— 上面三类都不命中。

### `get_cross_profile_warning(path) → 警告字符串 / None`

- **字符串(警告)** —— `classify_cross_profile_target(path)` 返回非 None,即跨 profile 写。
- **None(放)** —— 同 profile / 路径在 Hermes 之外 / 不在 profile-scoped area。

### `classify_cross_profile_target(path) → dict / None`

- **dict(跨 profile)** —— `<root>/<area>/...` 或 `<root>/profiles/<name>/<area>/...`,其中 `area ∈ {skills, plugins, cron, memories}`,且 target_profile ≠ active_profile。
- **None(非跨 profile)** —— 路径在 `<root>` 之外(`relative_to()` 抛 ValueError)/ 在 `<root>` 下但不命中 area / target_profile == active_profile(写自己的)。

**核心招式**:检查器 + 调用方守卫拆开 —— 纯函数易测,调用方在 IO 入口挡一刀;但只防"走正门"的 agent,terminal 工具绕路就完全失效(docstring 自己写了 "defense-in-depth, not a security boundary")。

### 小例子 + 为什么被拒

假设 `HERMES_HOME = ~/.hermes/profiles/A/`(active profile = "A"),`HOME = /home/u`。

```python
# ── is_write_denied ────────────────────────────────────────
is_write_denied("~/.ssh/id_rsa")
# → True   命中 build_write_denied_prefixes("~/.ssh/"),
#          realpath 后 "/home/u/.ssh/id_rsa".startswith("/home/u/.ssh/") → True

is_write_denied("~/.hermes/auth.json")
# → True   命中 Hermes 控制面检查:control_file_names = ("auth.json", ...) ∋ "auth.json",
#          且 hermes_dirs 同时含 active home(/home/u/.hermes/profiles/A)和 root(/home/u/.hermes),
#          resolved == realpath(hermes_home/auth.json) → True

is_write_denied("/tmp/foo.txt")
# → False  没命中黑名单 + 没设 HERMES_WRITE_SAFE_ROOT → 放行

is_write_denied("/tmp/innocent")        # 假设这是个符号链接 → /home/u/.ssh/id_rsa
# → True   realpath 解开符号链接后命中 ~/.ssh/ 前缀黑名单
```

```python
# ── get_read_block_error ───────────────────────────────────
get_read_block_error(".env")
# → "Access denied: ...credential leakage..."
#   命中 _BLOCKED_PROJECT_ENV_BASENAMES(任何路径下 .env 都拦)

get_read_block_error("~/.hermes/auth.json")
# → "Access denied: ...Hermes credential store..."
#   命中 credential_file_names,resolved == (hermes_home/"auth.json").resolve()

get_read_block_error("README.md")
# → None  哪类都不命中,放行
```

```python
# ── get_cross_profile_warning + classify_cross_profile_target ───
# active_profile = "A"

classify_cross_profile_target("~/.hermes/profiles/B/skills/x.py")
# → {"active_profile": "A", "target_profile": "B", "area": "skills", ...}
#   路径在 <root>/profiles/B/skills/... → target_profile="B", area="skills"
#   target_profile("B") ≠ active_profile("A") → 跨 profile

get_cross_profile_warning("~/.hermes/profiles/B/skills/x.py")
# → "Cross-profile write blocked by soft guard: ...belongs to Hermes profile 'B'..."
#   (↑ 就是把上面的 dict 拼成警告字符串)

classify_cross_profile_target("~/.hermes/profiles/A/skills/y.py")
# → None   target_profile("A") == active_profile("A"),写自己 profile

classify_cross_profile_target("~/.hermes/skills/z.py")
# → {"active_profile": "A", "target_profile": "default", "area": "skills", ...}
#   路径在 <root>/skills/... → target_profile="default"(不是当前 active 的 A)
#   ⚠️ 这也会被警告 — 写 root 级 default profile 同样算跨 profile

classify_cross_profile_target("/tmp/z.py")
# → None   路径在 <root> 之外,relative_to() 抛 ValueError → 早退返回 None
```

---

## 4. approval.py — 命令审批状态机

`approval.py` 1645 行,核心不是个真"状态机",而是**两层判定 + 三种生效范围**的审批系统。

**两层判定**(任意一条命中就拦/警告):

|层|函数|干啥|
|---|------|------|
|**Hardline(硬拦)**|[`detect_hardline_command`](hermes-agent/tools/approval.py#L289)|12 条无差别 `rm -rf /` / `mkfs` / fork bomb / `shutdown` 等 —— **无条件拦**,没法批准|
|**Dangerous(危险)**|[`detect_dangerous_command`](hermes-agent/tools/approval.py#L503)|47 条模式(`curl ... \| sh`、写 `.ssh/`、`sudo -S` 密码爆破等) —— **可批准**,触发审批流|
|**Tirith(可选)**|`check_command_security`|调 tirith 二进制扫内容威胁(见笔记第 2 点)|

**审批生效范围**(三种"批准"持久度):

```python
_pending: dict[str, dict] = {}                 # 待审批队列(等用户/UI 回复)
_session_approved: dict[str, set] = {}         # 仅当前 session 批准
_persistent_approved: set[str] = {}            # 永久批准(写到 ~/.hermes/command_allowlist)
_session_yolo: set[str] = set()                # session 级 YOLO 模式(全批准)
```

**审批流程**(伪代码):

```text
agent 发起命令
    ↓
① detect_hardline_command → 命中? → 直接 block(没法批准)
    ↓ 没命中
② detect_dangerous_command → 命中? → pattern_key (例: "rm -rf {dir}")
    ↓ 命中
③ 查 _session_approved / _persistent_approved → 已被批准过? → 直接放行
    ↓ 没批准过
④ 提交到 _pending 队列,等用户决定
    ↓
用户回复 allow-once / allow-session / allow-permanent / deny
    ↓
   ├─ allow-once       → 跑一次,下次还要再问
   ├─ allow-session    → 写 _session_approved,本 session 内同 pattern_key 都放行
   ├─ allow-permanent  → 写 ~/.hermes/command_allowlist,所有 session 都放行
   └─ deny             → 永久 block
```

**和 tirith / file_safety 的关系**:

- tirith —— **扫描内容**(homograph URL、pipe-to-shell、terminal injection)
- file_safety —— **路径级黑名单**(SSH 密钥、`.env`)
- approval —— **命令模式 + 审批流**(`rm -rf`、`shutdown`、写敏感目录)

三者**独立、各管一摊**,在 `terminal_tool._check_all_guards` 里被串起来依次跑。

**核心招式**:三层防御各管一个维度(模式 / 内容 / 路径),审批状态用 session_key 隔离 + 三级持久度(once/session/permanent),让用户能用最小成本表达"这次 OK / 以后都 OK"的不同意图。

### 三个判定层各拦什么

#### ① Hardline(硬拦,12 条 —— 无条件 block,没法批准)

看 [`HARDLINE_PATTERNS`](hermes-agent/tools/approval.py#L218),全是"任何情况下都不能让 agent 干的事":

- `rm -rf /` / `/home` / `/etc` / `/usr` 等 → 例:`rm -rf /`、`rm -rf /*`
- `mkfs` → 例:`mkfs.ext4 /dev/sda`
- `dd of=/dev/sd*` / `> /dev/sd*` → 例:`dd if=foo of=/dev/sda`
- Fork bomb → `:(){:|:&};:`
- `kill -1` / `killall -KILL` → 杀所有进程
- `shutdown` / `reboot` / `halt` / `init 0|6` / `systemctl poweroff` → 关机重启

**特点**:命令位置锚定(`_CMDPOS`),防止 `echo reboot` 误命中;**没**走审批流,直接 block。

---

#### ② Dangerous(危险,47 条 —— 可批准,弹审批框)

看 [`DANGEROUS_PATTERNS`](hermes-agent/tools/approval.py#L336),覆盖"有合法用途但很危险"的命令:

- **递归删除** — `rm -r`、`find -exec rm` / `-delete`、`xargs rm`
- **权限/属主** — `chmod 777` / `o+w`、`chown -R root`
- **磁盘写入** — `dd if=`、`> /dev/sd*`、`mkfs`
- **SQL 破坏** — `DROP TABLE/DATABASE`、`DELETE FROM`(无 WHERE)、`TRUNCATE`
- **服务控制** — `systemctl stop/restart/disable/mask`、`kill -9 -1`、`pkill -9`
- **脚本执行** — `bash -c`、`python -e`、`curl ... | sh`、heredoc(`python << EOF`)
- **覆盖系统/凭证文件** — `> /etc/...`、`tee ~/.ssh/...`、`sed -i ~/.hermes/config.yaml`
- **Git 破坏** — `git reset --hard`、`git push --force`、`git clean -f`、`git branch -D`
- **自残保护** — `pkill hermes`、`kill $(pgrep hermes)`、`hermes gateway stop`、`hermes update`
- **Docker 生命周期** — `docker compose restart/stop/kill/down`
- **Sudo 提权** — `sudo -S`(密码爆破)、`sudo -s` / `-A`(TTY-free 提权)

**特点**:命中后生成 `pattern_key` → 查 `_session_approved` / `_persistent_approved` → 没批过就进 `_pending` 等用户决定。

---

#### ③ Tirith(可选 —— 内容级威胁,warn/block)

看 [笔记第 2 点](hermes-agent/Learning/Notes/学习笔记6-Day9-路径验证机制.md),由外部 Rust 二进制扫:

- **homograph URL**(钓鱼域名,例:`xn--gogle-1ta.com`)
- **pipe-to-interpreter**(URL 里藏 payload)
- **terminal injection**(ANSI 转义注入)
- **lookalike_tld**(`.app` 之类误报会被代码自动降级为 allow)

**特点**:判定和前两层**正交** —— 前两层看"命令字符串模式",tirith 看"命令**内容语义**"。比如 `curl https://xn--gogle-1ta.com` 前两层都放行(就是个 curl),tirith 会拦(同形异义域名)。

---

**一句话总结**:

- **Hardline** = "这事绝对不能干,问了也白问"
- **Dangerous** = "这事可以干,但先让我问问用户"
- **Tirith** = "这事的**内容**看起来可疑,让我再查查"

---

## 5. credential_files.py — 远程沙箱"喂文件"的注册中心

文件 437 行,核心职责:**当 terminal 跑在 Docker / Modal / SSH 等远程后端时,告诉后端要把哪些宿主机文件挂载/同步进沙箱**。

### 5.1 为什么需要这文件 —— 远程沙箱"干净"问题

`terminal_tool.py` 支持 6 种环境跑命令:

|后端|含义|沙箱在哪|
|---|------|---|
|`local`|直接在本机跑(默认)|无|
|`docker`|跑在 Docker 容器|本机或远端(看 Docker daemon)|
|`modal`|跑在 Modal 云沙箱|远端(AWS/GCP)|
|`ssh`|跑在远端 SSH 主机|远端|
|`daytona`|跑在 Daytona 云沙箱|远端|
|`singularity`|跑在 Singularity 容器|通常本机|

**问题**:`local` 模式所有文件都在,agent 随便读;但 `docker` / `modal` / `ssh` 模式下,容器/远端是**全新的干净环境**,宿主机上的 OAuth token、API key、上传的文档**都没了** —— 但 agent 需要用它们。

**credential_files.py 的解法**:把宿主上**必要的**文件,**精确**地挂载/同步进沙箱。

### 5.2 三类文件为什么分开

#### ① 凭证文件(必传)

```yaml
# skill frontmatter 里声明它需要哪些凭证
required_credential_files:
  - google_token.json        # 相对 HERMES_HOME
  - anthropic_oauth.json
```

注册 [`register_credential_file`](hermes-agent/tools/credential_files.py#L56):

- 拼成 `<HERMES_HOME>/google_token.json`
- **校验**必须落在 `HERMES_HOME` 内(`validate_within_dir`)
- 存在 → 加进注册表,记 `{容器路径: 宿主路径}`

**为什么校验?** 防止恶意 skill 写 `required_credential_files: ['../../.ssh/id_rsa']` —— 没校验的话,直接就把宿主 SSH 密钥挂进容器,agent 就读到了。

#### ② Skills 目录(整目录传)

```python
get_skills_directory_mount()    # Docker 用 — 整个目录 bind mount
iter_skills_files()             # Modal/Daytona 用 — 单文件一个个上传
```

**反符号链接攻击** [`_safe_skills_path`](hermes-agent/tools/credential_files.py#L250):

- 假设恶意 skill 在 `~/.hermes/skills/evil/scripts/run.sh` 放了个符号链接,指向宿主 `~/.ssh/id_rsa`
- Docker bind mount **会跟符号链接走**,等于把 SSH 密钥曝给容器
- 解决方案:遍历目录,发现任何 symlink → **复制一份"清白版"到 `/tmp/hermes-skills-safe-xxx/`**,跳过所有 symlink

```python
symlinks = [p for p in skills_dir.rglob("*") if p.is_symlink()]
if not symlinks:
    return str(skills_dir)        # 没 symlink → 直接挂(零开销)
# 有 symlink → 复制干净版本到 /tmp
```

#### ③ 缓存目录(挂用户上传/生成的文件)

```python
_CACHE_DIRS = [
    ("cache/documents",   "document_cache"),     # (新路径, 老路径,向后兼容)
    ("cache/images",      "image_cache"),
    ("cache/audio",       "audio_cache"),
    ("cache/screenshots", "browser_screenshots"),
]
```

涵盖用户上传/生成的四类文件:文档(PDF 等)、图片、音频(TTS)、截图(浏览器)。

**`to_agent_visible_cache_path()`** —— agent 给的是**宿主路径**(如 `/Users/jon/.hermes/cache/documents/report.pdf`),但容器里看不到。需要翻译成**容器路径**(`/root/.hermes/cache/documents/report.pdf`):

```python
if os.environ.get("TERMINAL_ENV", "local") != "docker":
    return host_path    # local/Modal/Daytona 不需要翻译(各自有不同的挂载机制)

for mount in get_cache_directory_mounts():
    if path 在 mount["host_path"] 下:
        return mount["container_path"] + 相对路径
```

### 5.3 注册表数据流

```text
┌─────────────────────────────────────────────────────────────┐
│ ① Skill 加载                                                 │
│   解析 frontmatter 的 required_credential_files              │
│   → register_credential_file("google_token.json")            │
│   → 校验:非绝对路径 / 非 ../ / 落 HERMES_HOME 内 / 是文件   │
│   → _get_registered()[container_path] = host_path           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ② 沙箱创建时(terminal_tool._create_environment)              │
│   get_credential_file_mounts() + get_skills_directory_mount()│
│   + get_cache_directory_mounts()                             │
│   → Docker:bind mount(-v 参数)                              │
│   → Modal/Daytona:iter_*_files() 单文件上传                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ③ 命令执行时                                                  │
│   agent 说"读 /Users/jon/.hermes/cache/report.pdf"           │
│   → to_agent_visible_cache_path() 翻译                       │
│   → "/root/.hermes/cache/report.pdf" (容器路径)             │
│   → agent 在容器里就能读了                                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Session 隔离(ContextVar)

```python
_registered_files_var: ContextVar[Dict[str, str]] = ContextVar("_registered_files")
```

为什么用 `ContextVar` 而不是普通 `dict`?

- **Gateway 模式**下多个 session 并发跑,每个 session 跑各自的 agent
- 普通 `dict` 会**跨 session 串数据**:session A 注册的文件被 session B 看到
- `ContextVar` 每个调用上下文一份,gateway pipeline 自动隔离

### 5.5 三个安全招式

|招式|防御什么|实现|
|------|---------|---------|
|**路径三层校验**|恶意 skill 用 `../.ssh/id_rsa` 越权读凭证|`validate_within_dir` + 绝对路径检查 + `..` resolve|
|**符号链接清白化**|恶意 skill 用 symlink 借 bind mount 曝宿主文件|`_safe_skills_path` 检测到 symlink 就复制剥光版本到 `/tmp`|
|**Session 隔离**|gateway 模式下 session 间串数据|`ContextVar` 自动隔离|

### 5.6 核心招式

**"挑出来 + 校验 + 挂进"三段式** —— 不是把所有文件都塞进沙箱(那太危险),而是精确挑选 + 严格校验 + 按需挂载,既保证 agent 能干活,又防止它跑出去读宿主文件。

**和 file_safety 的关系**:**反向互补**。

- `file_safety` —— **禁止**写入敏感路径(纵深防御)
- `credential_files` —— **主动**挂载指定文件进沙箱(精确投放)

两者都基于 `validate_within_dir` / `realpath` 做路径安全,目标相反 —— 一个"别写",一个"给我挂"。

---

## 6. 四个安全文件全景对照

|文件|防御维度|判定对象|判定方式|命中后|盲点|
|---|---|---|---|---|---|
|**tirith_security**|内容级威胁|命令字符串的**内容语义**|外部 Rust 二进制扫(homograph URL、pipe-to-shell、terminal injection)|`allow` / `warn` / `block`(退出码)|只看语义,不看命令模式|
|**file_safety**|路径级黑名单|解析后的**绝对路径**(`realpath` 后)|精确相等 / `startswith(prefix + sep)` / `Path.relative_to()`|`WriteResult(error=...)` 早退|`terminal` 直跑 shell 完全绕过它|
|**credential_files**|沙箱文件投放|宿主要"主动挂载"的文件路径|`validate_within_dir` + 符号链接清白化|挂进沙箱(`/root/.hermes/...`)|只管挂不管用,挂出去后 agent 怎么用是别的事|
|**approval**|命令模式 + 审批流|命令**字符串模式**(47 条正则)|硬拦 12 条 / 危险 47 条 / tirith 可选|弹审批框 → `allow-once` / `allow-session` / `allow-permanent` / `deny`|正则写不到的变形命令可绕过|

### 6.1 一句话定位

- **tirith** = "这事**内容**可疑,让我查查"(内容级扫描)
- **file_safety** = "这事**路径**不许碰"(路径黑名单)
- **credential_files** = "这事**要喂哪些文件**"(精确投放)
- **approval** = "这事**命令模式**危险,先问问用户"(模式 + 审批)

### 6.2 串起来的调用链

```text
agent 发起 shell 命令
    ↓
① approval.detect_hardline_command  → 命中? → block
    ↓ 没命中
② approval.detect_dangerous_command  → 命中? → 查 _session_approved / _persistent_approved
                                          → 没批过 → 提交 _pending → 等用户决定
    ↓ 没命中 / 已批准
③ tirith.check_command_security      → 0/1/2 退出码 → allow / block / warn
    ↓
④ 沙箱内执行(如果是 docker/modal/ssh)
    ├─ credential_files 负责"挂哪些文件进沙箱"
    └─ 如果 agent 想读写文件
        └─ file_operations.py 调 file_safety.is_write_denied() → 拦下
           (但 file_tools 直跑 terminal 就绕过这道)
```

### 6.3 互补与盲点

|维度|tirith|file_safety|credential_files|approval|
|---|---|---|---|---|
|看"模式"|✗|✗|✗|✓|
|看"内容"|✓|✗|✗|△(正则略看)|
|看"路径"|✗|✓|✓|△(正则拼出 .ssh/ 等)|
|能挂文件|✗|✗|✓|✗|
|能拦截|✓|✓|✗|✓|

**盲点分布**:

- `tirith` 看内容不看模式,`approval` 看模式不看内容 —— **正交互补**
- `file_safety` 守 `file_tools` 入口,`approval` 守 `terminal` 入口 —— **terminal 直跑 `cat ~/.ssh/x` 两个都绕不过去**(这是 file_safety 自己承认的"defense-in-depth, not a security boundary")
- `credential_files` 解决的是"沙箱里啥都没有"的可用性问题,跟前三者**职责正相反**(投放 vs 拦截)

### 6.4 核心招式

**四层防御各管一个维度,串成一条管道**:

```text
内容(tirith) ←互补→ 模式(approval) ←互补→ 路径(file_safety) ←互补→ 沙箱(credential_files)
```

- 任何一个失守,**至少还有一道兜底**
- 但 `terminal` 直跑任意 shell 是**共同盲点** —— 这是 OS-level 限制,Python 层没法彻底堵
- 真正"安全"靠用户**别在 `local` 模式跑不可信 agent** + 重要操作用 `docker` 沙箱
