"""Central registry for all hermes-agent tools.

每个工具文件在模块顶层调用 ``registry.register()`` 来声明它的
schema、handler、所属 toolset、以及可用性检测函数(check_fn)。
``model_tools.py`` 不再维护自己的一套并行数据结构,而是直接查这个 registry。

# === 核心设计理念 ===
1. 自注册(self-registering):工具文件 import 时就把自己注册进全局 registry
2. 单例:模块级 `registry = ToolRegistry()`(L544),全进程共享一份
3. 无循环依赖:registry.py 不 import model_tools 或工具文件本身
4. AST 扫描:`discover_builtin_tools` 用 AST 静态判断哪些模块"会注册工具",
   避免 import 一个不相关的模块带来的副作用(副作用 = 真的会去执行其顶层注册)

# === 导入链(circular-import safe) ===
#
#   tools/registry.py  (不 import model_tools / 工具文件)
#          ↑
#   tools/*.py         (在模块顶层 import from tools.registry)
#          ↑
#   model_tools.py     (import tools.registry + 所有工具模块)
#          ↑
#   run_agent.py, cli.py, batch_runner.py, ...
#
# 自上而下:registry 是"叶子",被所有人依赖,但不依赖任何人
"""

import ast
import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _is_registry_register_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``registry.register(...)`` call expression.

    # === 这个函数的作用 ===
    # 在不执行模块代码的前提下,只看 AST 节点结构,判断某一行
    # 是不是长这样:`registry.register(...)`。
    #
    # 之所以需要 AST 扫描,而不是直接 import:
    #   - import 会执行模块顶层代码 → 可能产生副作用(网络、磁盘)
    #   - 用 AST 扫描"零副作用"地挑出真正会注册的模块
    #   - 这就是 ``discover_builtin_tools`` 的前置过滤
    #
    # === AST 节点识别 ===
    # 目标表达式:`registry.register(...)`
    #
    # AST 结构(自顶向下):
    #   Expr                      <- 表达式语句
    #     value: Call             <- 函数调用
    #       func: Attribute       <- 属性访问
    #         attr: "register"    <- 属性名
    #         value: Name         <- 属性宿主
    #           id: "registry"    <- 名称 id
    #       args / keywords       <- 调用参数(这里不关心)
    #
    # 所以判断条件是:节点是 Expr.value 是 Call,Call.func 是
    # `registry.register` 形式的属性访问。
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """Return True when the module contains a top-level ``registry.register(...)`` call.

    # === 关键约束:"顶层"调用 ===
    # 只扫描模块体的语句(`tree.body`),不看函数/类内部。
    # 原因:有些辅助模块可能只是在某个函数里临时调一下
    # `registry.register()`,但我们想找的是"import 后就会自动注册"的那种。
    # 顶层调用 = 模块被 import 时会立即执行的注册。

    # === 容错处理 ===
    # - OSError:文件读不到(权限、不存在)
    # - SyntaxError:文件有语法错误
    # 这两种都返回 False,意味着"当作这个文件不注册工具,跳过它"。

    # === 实现方式 ===
    # 1. 读取源码
    # 2. ast.parse 转成语法树
    # 3. 遍历顶层语句(`tree.body`),看有没有任何一条是
    #    `registry.register(...)` 形式
    """
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False

    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names.

    # === 流程 4 步 ===
    # 1. 确定要扫描的目录(默认就是 tools/ 目录本身)
    # 2. glob 所有 .py 文件
    # 3. 过滤掉特殊文件 + 用 AST 判断"是否注册工具"
    # 4. 真正 import 那些候选模块(这时它们的顶层代码会跑,
    #    触发 `registry.register(...)` 把工具塞进全局 registry)
    #
    # === 过滤条件 ===
    # - `__init__.py`:Python 包标识,不是工具
    # - `registry.py`:就是本文件(import 自己会无限递归)
    # - `mcp_tool.py`:MCP 工具走单独的加载路径,不走 builtin
    # - 其他 .py 还得过 AST 那一关(辅助模块被剔除)
    #
    # === 排序 + 容错 ===
    # - 排序:为了让注册顺序稳定(便于调试、snapshot 一致)
    # - 容错:某个模块 import 失败,只 warning 不抛——避免一个坏模块
    #   把整个 agent 启动搞挂
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent
    module_names = [
        f"tools.{path.stem}"
        for path in sorted(tools_path.glob("*.py"))
        if path.name not in {"__init__.py", "registry.py", "mcp_tool.py"}
        and _module_registers_tools(path)
    ]

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


class ToolEntry:
    """Metadata for a single registered tool.

    # === 这是"一个工具"的所有信息 ===
    # 工具文件调 `registry.register(...)` 时,实际上就是把这一坨参数
    # 包成 ToolEntry 塞进 `self._tools: Dict[str, ToolEntry]` 里。

    # === 用 __slots__ 而不是普通类属性 ===
    # 节省内存:60+ 工具 × 11 字段 = 660+ 属性,如果用 __dict__ 会浪费大量内存
    # 加快属性访问:slots 走 descriptor,比 __dict__ 查表快
    # 防止误加新字段:想加新字段必须显式列在 slots 里,避免 typo

    # === 11 个字段的含义 ===
    # name              str     工具的"身份证号",模型调工具时用的名字
    # toolset           str     工具所属分类("file" / "terminal" / "mcp-xxx" 等)
    # schema            dict    OpenAI 格式的工具描述(给 LLM 看的)
    # handler           Callable 模型说"调 X"时,实际执行的函数
    # check_fn          Callable 探测工具是否可用(环境/依赖)
    # requires_env      list    需要哪些环境变量(如 ["DOCKER_HOST", "MODAL_TOKEN"])
    # is_async          bool    handler 是不是 async def(决定 dispatch 时要不要过 asyncio)
    # description       str     给人类看的简介(LLM 不一定看,UI 可能用)
    # emoji             str     给 UI 用的表情符号
    # max_result_size_chars  限制这个工具的返回值最多多少字符
    # dynamic_schema_overrides   运行时动态改 schema(见下面详细注释)
    """

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars", "dynamic_schema_overrides",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None):
        # === 基础字段:直接保存 ===
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars

        # === dynamic_schema_overrides:运行时改 schema ===
        # 可选:零参数 callable,返回一个 dict,会在 get_definitions() 时
        # 浅合并到基础 schema 上。
        #
        # 为什么需要?
        #   有些工具的 description 依赖运行时配置。比如 delegate_task:
        #   它的 description 必须告诉 LLM "当前最大并发子 agent 数 = 3、
        #   最大递归深度 = 2",如果用户改了 config.yaml,数字就得跟着变。
        #   写死成 "3" 在用户改完配置后就变成错的描述。
        #
        # 怎么用?
        #   - callable,每次 get_definitions() 都会调
        #   - 返回 dict,会被浅合并到 schema 上(顶层 key 覆盖)
        #   - 抛异常时,会 warning 并继续用静态 schema
        #
        # 缓存?
        #   不在这层缓存。外层(model_tools.get_tool_definitions)用
        #   config.yaml 的 mtime + size 当 cache key,配置变 → 缓存失效。
        self.dynamic_schema_overrides = dynamic_schema_overrides


# ---------------------------------------------------------------------------
# check_fn TTL cache
#
# === 为什么需要这个缓存? ===
# check_fn 是工具的"可用性探测"函数,典型例子:
#   - tools/terminal_tool.check_terminal_requirements  ← 探测 docker daemon
#   - browser tool 的 check                              ← 探测 playwright
#   - modal tool 的 check                                ← 探测 modal SDK
# 这些探测经常要 spawn 子进程、读文件、连网络——很贵。
#
# 现实:在长寿命的 CLI / gateway 进程里,get_definitions() 会被反复调
# (每个 turn 都会拿 schema),但外部状态(Docker 是否在跑)在"人"的时间
# 尺度上才变。所以每次都重探测是纯浪费。
#
# === 解决方案:30 秒 TTL ===
# 缓存 30 秒,既:
#   - 大幅减少探测次数(99%+ 的 get_definitions 走缓存)
#   - 30 秒后自动重探测 → env 变量翻转、用户 `hermes tools enable`
#     这类操作能在 1-2 个 turn 内生效,不需要显式 invalidate
#
# === 关键点 ===
# - 用 `time.monotonic()`(墙钟可能跳变,但单调时钟不会)
# - 锁内只做"读缓存"和"写缓存"——真正的探测在锁外(避免长任务
#   阻塞其他线程的 cache hit)
# - 探测抛异常 → 记 False(工具暂时不可用),不传播
# ---------------------------------------------------------------------------

# TTL:30 秒。看似随便选的数,其实在"省 CPU" vs "配置生效延迟"之间取的中间值
_CHECK_FN_TTL_SECONDS = 30.0

# 缓存结构:{check_fn 函数本身: (探测时间戳, 探测结果)}
# 用函数对象当 key 而非名字,避免不同模块同名函数撞车
_check_fn_cache: Dict[Callable, tuple[float, bool]] = {}

# 写缓存的锁(读用 GIL,Dict.get 是原子的;写要锁)
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    """Return bool(fn()), TTL-cached across calls. Swallows exceptions as False.

    # === 执行流程(双重检查锁式) ===
    # 1. 拿单调时钟
    # 2. 加锁 → 查缓存
    #    - 命中且未过期 → 直接返回(95%+ 的情况走这条)
    # 3. 锁外执行实际探测(耗时的 IO 放在锁外,不让其他线程等)
    # 4. 加锁 → 写回缓存
    # 5. 返回结果
    #
    # === 异常处理 ===
    # check_fn 探测可能失败(网络断、docker daemon 挂、依赖没装)。
    # 这种时候宁可"标为不可用",也不要抛异常——抛异常会污染上层逻辑。
    """
    now = time.monotonic()
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get(fn)
        if cached is not None:
            ts, value = cached
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value
    try:
        value = bool(fn())
    except Exception:
        value = False
    with _check_fn_cache_lock:
        _check_fn_cache[fn] = (now, value)
    return value


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results. Call after config changes that
    affect tool availability (e.g. ``hermes tools enable``).

    # === 什么时候调用? ===
    # 外部代码改了"会改变工具可用性"的状态时:
    #   - 用户执行 `hermes tools enable foo` (开启某个工具集)
    #   - 用户改了 ~/.hermes/config.yaml
    #   - MCP server 列表变化
    # 这时候想立即生效,不等 30 秒 TTL,就直接调一下这个函数。
    """
    with _check_fn_cache_lock:
        _check_fn_cache.clear()


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files.

    # === 这个类是整个 Tool System 的"中央数据库" ===
    # - 工具文件 import 时调 `registry.register(...)` → 塞进 self._tools
    # - main loop / model_tools 调 `registry.get_definitions(...)` → 取 schema 给 LLM
    # - 模型决定调工具 → main loop 调 `registry.dispatch(name, args)` → 执行 handler
    # - 还要支持热插拔(MCP server 推送新工具、用户 enable/disable 工具集)
    #
    # === 关键设计 ===
    # 1. 线程安全:用 RLock 保护所有 mutation
    # 2. 快照读取:reader 拿到的是 list(self._tools.values()) 的副本,
    #    即使 writer 之后改了原始 dict,reader 也不会受影响
    # 3. Generation counter:单调递增计数器,让外部缓存能
    #    "key 一下 generation,看 generation 变了就失效"——比加锁便宜
    """

    def __init__(self):
        # === 三张表 ===
        # _tools: 工具名 → ToolEntry(主表,最常用)
        self._tools: Dict[str, ToolEntry] = {}
        # _toolset_checks: 工具集名 → 探测函数(共享,避免每个 tool 重复探测)
        self._toolset_checks: Dict[str, Callable] = {}
        # _toolset_aliases: 别名 → 规范名(让"terminal" 和 "shell" 指向同一组工具)
        self._toolset_aliases: Dict[str, str] = {}

        # === 锁 ===
        # 用 RLock(可重入锁)而不是 Lock。
        # 原因:ToolRegistry 的方法之间会互相调用(比如 register 内部
        # 可能调 deregister 之类的),RLock 允许同一线程多次加锁。
        # MCP dynamic refresh 可能在别的线程 mutate 这个 registry,
        # 所以所有 mutation 必须加锁;reader 走快照,不直接读 dict。
        self._lock = threading.RLock()

        # === Generation counter(无锁并发的核心) ===
        # 每次 mutation(register / deregister / alias 变化 / MCP refresh)
        # 都 +1。
        #
        # 用途:外部 cache 可以 key 住这个 generation。
        # 比如 model_tools.get_tool_definitions 可以这样:
        #     cache[generation] = [...]
        # 然后每次取 schema 时,先看 `registry._generation` 变没变——
        # 变了就重算,没变就返缓存。完全不需要加锁,比锁快得多。
        self._generation: int = 0

    def _snapshot_state(self) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks.

        # === 为什么要"快照"而不是直接返回内部状态? ===
        # 因为 reader 拿到结果后,可能还要遍历/处理几毫秒;
        # 这期间 writer 可能插进来改 _tools 字典,导致 reader 看到
        # 半新半旧的状态(迭代器失效、KeyError 等)。
        #
        # list(...) / dict(...) 会创建新对象,之后 writer 怎么改
        # 都不影响 reader 拿到的这份副本。
        #
        # === 为什么两个一起返回? ===
        # 因为某些操作(比如 is_toolset_available)需要同时看
        # 工具列表和工具集的 check_fn,这两者必须来自同一时刻的快照,
        # 否则可能"工具有了但 check_fn 还没注册"或反之。
        """
        with self._lock:
            return list(self._tools.values()), dict(self._toolset_checks)

    def _snapshot_entries(self) -> List[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        return self._snapshot_state()[0]

    def _snapshot_toolset_checks(self) -> Dict[str, Callable]:
        """Return a stable snapshot of toolset availability checks."""
        return self._snapshot_state()[1]

    def _evaluate_toolset_check(self, toolset: str, check: Callable | None) -> bool:
        """Run a toolset check, treating missing or failing checks as unavailable/available.

        # === 三种情况的处理 ===
        # 1. check 是 None → 默认可用(没有 check_fn 的工具,默认暴露)
        # 2. check 返回 truthy → 可用
        # 3. check 返回 falsy → 不可用
        # 4. check 抛异常 → 不可用 + debug 日志(不传播异常)
        #
        # 异常被吞掉的原因:check_fn 探测外部状态,外部状态异常是常态
        # (网络瞬断、docker daemon 挂),不能让这种"外部噪声"污染
        # 工具注册流程的稳定性。
        """
        if not check:
            return True
        try:
            return bool(check())
        except Exception:
            logger.debug("Toolset %s check raised; marking unavailable", toolset)
            return False

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """Return a registered tool entry by name, or None.

        # === 走锁而不是走快照 ===
        # 这里是"点查"——按名字找一个 tool,直接 dict.get 即可。
        # 拿到的 ToolEntry 对象是引用(不是副本),如果 caller 之后
        # 改了它的属性,会污染 registry。但 caller 不应该改 ToolEntry,
        # 所以这里没做 deep copy。
        """
        with self._lock:
            return self._tools.get(name)

    def get_registered_toolset_names(self) -> List[str]:
        """Return sorted unique toolset names present in the registry.

        # 用 set comprehension 去重 + sorted 排序(让结果稳定可复现)
        # 走 snapshot 而不是锁,这样 reader 拿到的是稳定副本
        """
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for a canonical toolset name.

        # === Alias 是干什么的? ===
        # 让不同的名字指向同一组工具。比如:
        #   register_toolset_alias("shell", "terminal")
        #   register_toolset_alias("cmd",   "terminal")
        # 这样用户说"启用 shell 工具集"和"启用 terminal 工具集"是等价的。
        # 用在 UI 配置兼容、用户别名习惯等场景。

        # === 重复注册怎么办? ===
        # 同名 alias 被映射到不同 toolset → warning 但允许覆盖
        # (有时是用户改主意了,有时是 plugin 升级)
        # 每次注册都 +1 generation,让缓存失效
        """
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """Return a snapshot of ``{alias: canonical_toolset}`` mappings."""
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """Return the canonical toolset name for an alias, or None."""
        with self._lock:
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable = None,
        override: bool = False,
    ):
        """Register a tool.  Called at module-import time by each tool file.

        ``override=True`` is an explicit opt-in for plugins that intend to
        replace an existing built-in tool implementation (e.g. swap the
        default browser tool for a headed-Chrome CDP backend). Without it,
        registrations that would shadow an existing tool from a different
        toolset are rejected to prevent accidental overwrites.

        # === 12 个参数都在干什么? ===
        # name              工具名(LLM 调的工具标识,不能重)
        # toolset           所属分类
        # schema            OpenAI 格式的工具定义(给 LLM 看)
        # handler           实际执行的函数
        # check_fn          可用性探测(可选)
        # requires_env      需要的环境变量列表
        # is_async          handler 是不是 async
        # description       描述(优先用参数,否则从 schema 拿)
        # emoji             UI 表情
        # max_result_size_chars  返回值大小限制
        # dynamic_schema_overrides  运行时改 schema
        # override          是否允许覆盖同名不同 toolset 的工具

        # === 重复注册策略(4 档) ===
        # 1. 同名同 toolset → 直接覆盖(L290,常规情况)
        # 2. 同名不同 toolset,但都是 mcp-* → 允许(MCP server 刷新)
        # 3. 同名不同 toolset,且 override=True → 允许(plugin 显式接管)
        # 4. 同名不同 toolset,既不是 MCP 也不 override → 拒绝 + 报错
        #
        # 策略 4 的目的:防止插件意外覆盖内置工具(可能引入 bug 或安全风险)
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                # Allow MCP-to-MCP overwrites (legitimate: server refresh,
                # or two MCP servers with overlapping tool names).
                both_mcp = (
                    existing.toolset.startswith("mcp-")
                    and toolset.startswith("mcp-")
                )
                if both_mcp:
                    logger.debug(
                        "Tool '%s': MCP toolset '%s' overwriting MCP toolset '%s'",
                        name, toolset, existing.toolset,
                    )
                elif override:
                    # Explicit plugin opt-in: replace the existing tool.
                    # Logged at INFO so the override is auditable in agent.log.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name, toolset, existing.toolset,
                    )
                else:
                    # Reject shadowing — prevent plugins/MCP from overwriting
                    # built-in tools or vice versa.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset,
                    )
                    return
            # === 真正写入 ===
            # `description or schema.get("description", "")` 优先用参数,
            # 没有时回退到 schema 里的 description 字段。
            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides,
            )
            # === 共享 check_fn 到 toolset 级别 ===
            # 一个 toolset 可能有多个工具,但它们通常共享同一个 check_fn
            # (比如 terminal 工具集的所有工具都依赖 docker)。
            # 只在第一次见到这个 toolset 时记录,避免重复覆盖。
            if check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            # === generation +1 ===
            # 让所有外部缓存(基于 generation)失效
            self._generation += 1

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Also cleans up the toolset check if no other tools remain in the
        same toolset.  Used by MCP dynamic tool discovery to nuke-and-repave
        when a server sends ``notifications/tools/list_changed``.

        # === 什么时候调? ===
        # - MCP server 推送 `notifications/tools/list_changed`
        #   → 整组工具 nuke-and-repave(全删了重新 import)
        # - 用户 `hermes tools disable foo` 禁用某个工具
        # - plugin 卸载

        # === 连锁清理 ===
        # 如果删完某个工具后,它的 toolset 已经空了(没有别的工具属于这个 toolset),
        # 那这个 toolset 的 check_fn 和所有指向它的 alias 都没用了,
        # 一并清掉,避免"挂着空 toolset"这种状态泄漏。

        # === 锁的粒度 ===
        # 整个函数在锁内完成(快速操作,不阻塞),debug 日志放锁外
        """
        with self._lock:
            entry = self._tools.pop(name, None)
            if entry is None:
                return
            # Drop the toolset check and aliases if this was the last tool in
            # that toolset.
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in self._tools.values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset, None)
                self._toolset_aliases = {
                    alias: target
                    for alias, target in self._toolset_aliases.items()
                    if target != entry.toolset
                }
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """Return OpenAI-format tool schemas for the requested tool names.

        Only tools whose ``check_fn()`` returns True (or have no check_fn)
        are included. ``check_fn()`` results are cached for ~30 s via
        :func:`_check_fn_cached` to amortize repeat probes (check_terminal_
        requirements probes modal/docker, browser checks probe playwright,
        etc.); TTL chosen so env-var changes (``hermes tools enable foo``)
        still take effect in near-real-time without forcing a full cache
        flush on every call.

        # === 这个函数干什么? ===
        # 把"工具名集合"转成"LLM 能直接用的 OpenAI 格式 schema 列表"。
        # 整个过程要:
        #   1. 过滤:不存在的工具、check_fn 失败的 → 跳过
        #   2. 增强:补 name 字段、应用 dynamic_schema_overrides
        #   3. 包装:包成 `{"type": "function", "function": {...}}` 格式

        # === 两层 check_fn 缓存 ===
        # 第 1 层(全局):_check_fn_cache 跨调用缓存 30 秒(在 _check_fn_cached 里)
        # 第 2 层(本次):check_results 字典缓存本次调用内的探测结果
        #
        # 为什么需要第 2 层?
        #   一次 get_definitions 可能查询 30+ 工具,它们可能共享同一个
        #   check_fn(比如 terminal 工具集下 5 个工具都共用 check_terminal_requirements)。
        #   不在本次内缓存就要调 5 次 _check_fn_cached,虽然 30s 内会命中
        #   全局缓存,但每次都要走"加锁 → 查 ts → 解锁"的开销,不值。
        #   用 dict 缓存本次,只调 1 次 _check_fn_cached。
        """
        result = []
        # Per-call cache on top of the 30 s TTL — handles repeat probes of the
        # same check_fn within one definitions pass without re-reading the
        # TTL clock.
        check_results: Dict[Callable, bool] = {}

        # === 快照:一次拿到所有 entry,O(N) 一次,后续 O(1) 查 ===
        # 用 dict 索引(不要每次都线性扫描列表找 name)
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}

        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            # === check_fn 过滤 ===
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # Ensure schema always has a "name" field — use entry.name as fallback
            schema_with_name = {**entry.schema, "name": entry.name}
            # Apply runtime-dynamic overrides (e.g. delegate_task description
            # depends on current delegation.max_concurrent_children /
            # max_spawn_depth). Caller side (model_tools.get_tool_definitions)
            # already keys its memo on config.yaml mtime + size, so changes
            # to delegation.* in config invalidate the cache automatically.
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                    if isinstance(overrides, dict):
                        schema_with_name.update(overrides)
                except Exception as exc:
                    # 抛异常时降级到静态 schema,不污染整个流程
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; "
                        "using static schema",
                        name, exc,
                    )
            # === OpenAI 格式包装 ===
            # LLM 看的 schema 长这样:
            #   {"type": "function", "function": {name, description, parameters, ...}}
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """Execute a tool handler by name.

        * Async handlers are bridged automatically via ``_run_async()``.
        * All exceptions are caught and returned as ``{"error": "..."}``
          for consistent error format.

        # === 这是模型"调工具"的最终入口 ===
        # 模型说:"用 X 工具,参数是 Y" → main loop 拿到 →
        # → registry.dispatch(X, Y) → 返回 JSON 字符串给 main loop
        #
        # === 三个关键点 ===
        # 1. 不知道工具名 → 返回 "Unknown tool" 错误
        # 2. handler 是 async → 用 _run_async 桥接到同步(因为 agent 主循环是同步的)
        # 3. handler 抛异常 → 不传播,统一包成 `{"error": "..."}` JSON
        #
        # === 异常为什么不能传播? ===
        # dispatch 是在 agent 主循环里调用的,如果抛异常会直接
        # 干掉整个 turn。统一成 JSON 错误返回,让 LLM 看到错误后
        # 可以决定"重试"或"换工具",这是 ReAct 模式的一部分。
        """
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            if entry.is_async:
                # === async 桥接 ===
                # 懒加载 _run_async(在 model_tools 里),避免循环 import
                # (model_tools 反过来要用 registry)
                from model_tools import _run_async
                return _run_async(entry.handler(args, **kwargs))
            return entry.handler(args, **kwargs)
        except Exception as e:
            # === 异常处理 ===
            # logger.exception 会自动附上 traceback,但只 log 不传播
            logger.exception("Tool %s dispatch error: %s", name, e)
            # 走 sanitizer 把异常消息里的特殊 token(CDATA、fence 等)洗掉
            # 这些东西如果原样送到 LLM,会被 LLM 误解为"工具在输出结构化数据"
            # 引发解析混乱。详细见 model_tools._sanitize_tool_error。
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                # 防御:万一 sanitizer 自己也炸了,绝对不要让它"卡住"
                # 错误传播——直接用原始消息
                sanitized = raw
            return json.dumps({"error": sanitized})

    # ------------------------------------------------------------------
    # Query helpers  (replace redundant dicts in model_tools.py)
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """Return per-tool max result size, or *default* (or global default).

        # === 优先级链(3 档) ===
        # 1. 这个工具自己设了 max_result_size_chars → 用它
        # 2. 没设,但 caller 传了 default → 用 caller 的
        # 3. 都没 → 走全局默认值(tools/budget_config.py)
        #
        # 为什么不直接在 register 时设全局默认?
        #   因为有些工具(token 量大的)需要更严的限制,有些(简单查询)可以宽松。
        #   三档优先级让 caller 在不修改工具定义的情况下也能控制。
        """
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names.

        # 走 snapshot 而不是直接迭代 self._tools.values(),避免在迭代过程中
        # 被另一个线程 register/deregister 干扰
        """
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> Optional[dict]:
        """Return a tool's raw schema dict, bypassing check_fn filtering.

        Useful for token estimation and introspection where availability
        doesn't matter — only the schema content does.

        # === 跟 get_definitions 的区别 ===
        # get_definitions → 返回 OpenAI 包装格式,会跑 check_fn
        # get_schema     → 返回原始 dict,不过滤
        #
        # 用于:
        #   - token 估算(算"如果我告诉 LLM 这些工具,要花多少 token")
        #   - 自省 / 调试
        #   - 一些不关心工具是否能跑的元数据查询
        """
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """Return the toolset a tool belongs to, or None."""
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return the emoji for a tool, or *default* if unset.

        # 默认 "⚡" 闪电 emoji,任何没显式设 emoji 的工具都用这个
        # 比如"我看到工具 A 用了 📁,工具 B 没设 → 工具 B 显示 ⚡"
        """
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool.

        # 用于 UI 展示 / 调试:"所有工具 → 所属 toolset" 的映射
        # 一次性返回全部,避免 caller 反复调 get_toolset_for_tool
        """
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset's requirements are met.

        Returns False (rather than crashing) when the check function raises
        an unexpected exception (e.g. network error, missing import, bad config).

        # === 在锁内只查引用,实际探测放锁外 ===
        # 锁内:从 _toolset_checks 拿到 callable 引用(快)
        # 锁外:_evaluate_toolset_check 真正执行探测(可能耗 IO)
        # 这样锁持有时间最短,其他线程的 register / get_entry 不会阻塞
        """
        with self._lock:
            check = self._toolset_checks.get(toolset)
        return self._evaluate_toolset_check(toolset, check)

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset.

        # === 用途 ===
        # 启动时一次过:判断哪些 toolset 可用,哪些不可用,输出到
        # `hermes tools` CLI 或 agent 启动日志,给用户清晰的提示
        # (比如"你的 terminal 工具不可用,需要装 docker")
        """
        entries, toolset_checks = self._snapshot_state()
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: self._evaluate_toolset_check(toolset, toolset_checks.get(toolset))
            for toolset in toolsets
        }

    def get_available_toolsets(self) -> Dict[str, dict]:
        """Return toolset metadata for UI display.

        # === 输出格式 ===
        # {
        #   "terminal": {
        #     "available": True,
        #     "tools": ["run_command", "shell"],
        #     "description": "",
        #     "requirements": ["DOCKER_HOST", "MODAL_TOKEN"]
        #   },
        #   ...
        # }
        #
        # 跟 get_toolset_requirements 的区别:
        #   - get_available_toolsets → 含 available bool,给 UI 用
        #   - get_toolset_requirements → 含 setup_url 等更多字段,给后端
        # === 去重策略 ===
        # 用 dict 实现 set:先看 ts not in toolsets 再初始化
        # requires_env 也用 if not in 去重(因为同一 toolset 多工具可能共享 env)
        """
        toolsets: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self._evaluate_toolset_check(
                        ts, toolset_checks.get(ts)
                    ),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat.

        # === 兼容性目的 ===
        # 老代码(historical / 外部 plugin)可能依赖 TOOLSET_REQUIREMENTS
        # 这种"扁平 dict,每个 toolset 一个 entry"的格式。
        # 新代码用 get_available_toolsets,带 available bool。
        # 这个方法就是给老代码留的"形状兼容"出口。
        #
        # === 输出结构 ===
        # {
        #   "terminal": {
        #     "name": "terminal",
        #     "env_vars": ["DOCKER_HOST"],
        #     "check_fn": <callable>,
        #     "setup_url": None,
        #     "tools": ["run_command", "shell"]
        #   },
        #   ...
        # }
        """
        result: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function.

        # === 老 API 形状 ===
        # 早期代码(以及一些外部脚本)期望"分两组返回"的形态:
        #   (["file", "search"], [{"name": "terminal", "env_vars": [...], "tools": [...]}])
        # 这种"两段式"比"一个带 available bool 的 dict"更直观地表达
        # "哪组能用 / 哪组不能用",所以保留这个 API。
        #
        # === seen set 干什么? ===
        # 一次遍历所有 entry,但同一个 toolset 下的多个工具会重复触发。
        # 用 seen 集合去重,每个 toolset 只评估一次 check_fn
        # (check_fn 可能耗 IO,要少调)。
        """
        available = []
        unavailable = []
        seen = set()
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts in seen:
                continue
            seen.add(ts)
            if self._evaluate_toolset_check(ts, toolset_checks.get(ts)):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": entry.requires_env,
                    "tools": [e.name for e in entries if e.toolset == ts],
                })
        return available, unavailable


# === Module-level singleton(全进程一份) ===
# 这是为什么所有工具文件都 `from tools.registry import registry` 然后
# 调 registry.register(...)——它们 import 的是同一个对象。
# module-level 变量在 Python 解释器中是单例的(同一进程里)。
# 如果用 `def get_registry()`,会引入 lazy initialization 复杂度,
# 这里直接 module load 时就 new 一个,简单可靠。
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Helpers for tool response serialization
# ---------------------------------------------------------------------------
# === 工具返回值必须长什么样? ===
# LLM 通过 tool_call 触发 handler,handler 必须返回一个 JSON 字符串。
# 错误情况:`{"error": "..."}`
# 成功情况:`{"success": true, ...}` 或任何结构化 dict
#
# === 这两个 helper 解决什么问题? ===
# 工具代码里会大量出现:
#     return json.dumps({"error": "msg"}, ensure_ascii=False)
# 这种样板代码既啰嗦又容易写错(忘 ensure_ascii 中文乱码、忘 str() 类型错误)
# tool_error / tool_result 把它收敛到两行调用。
#
# === ensure_ascii=False 为什么重要? ===
# 默认 True 时,中文会被转成 \uXXXX,LLM 看到的全是转义码,语义损失。
# 设为 False 后保留原始 UTF-8,LLM 能直接读懂。
#
# === 用法示例 ===
#   from tools.registry import registry, tool_error, tool_result
#
#   return tool_error("something went wrong")
#   return tool_error("not found", code=404)             # 加额外字段
#   return tool_result(success=True, data=payload)       # 关键字参数
#   return tool_result({"key": "value"})                 # 直接传 dict


def tool_error(message, **extra) -> str:
    """Return a JSON error string for tool handlers.

    >>> tool_error("file not found")
    '{"error": "file not found"}'
    >>> tool_error("bad input", success=False)
    '{"error": "bad input", "success": false}'

    # === 参数 ===
    # message: 错误消息(会自动 str() 转换,接受任何类型)
    # **extra: 任意额外字段(比如 code=404, retryable=True)
    #
    # === 输出形状 ===
    #   {"error": "<message>", **extra}
    # 没有 extra 时就是 `{"error": "..."}`
    """
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg *or* keyword arguments (not both):

    >>> tool_result(success=True, count=42)
    '{"success": true, "count": 42}'
    >>> tool_result({"key": "value"})
    '{"key": "value"}'

    # === 两种调用方式(互斥) ===
    # 1. 位置参数传 dict:
    #      tool_result({"key": "value"})
    #    适用:handler 自己组装好一个 dict 想直接序列化
    #
    # 2. 关键字参数:
    #      tool_result(success=True, count=42)
    #    适用:handler 想"按字段写",省去 dict literal
    #
    # === 为什么 data is not None 而不是 if data? ===
    # 万一 caller 传 `tool_result({})`(空 dict),if data 会判 False 当作
    # 没传,然后走 kwargs 分支 → 返回 "{}" 巧合一样。
    # 但更阴险的是:传 `tool_result(False)` 时 if data 也判 False。
    # 用 `is not None` 显式区分"没传"和"传了 falsy 值"。
    #
    # === 不允许两种混用 ===
    # 注释里写 "not both",但代码没强制检查。
    # 因为 kwargs 也会被忽略(只走 data 分支),不是 bug——只是 caller
    # 的 bug,docstring 提醒一下。
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
