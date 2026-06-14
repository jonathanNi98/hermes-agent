#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import json
import logging
import random
import re
import sqlite3
import threading
import time
from pathlib import Path

from agent.memory_manager import sanitize_context
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

SCHEMA_VERSION = 14

# ---------------------------------------------------------------------------
# WAL-compatibility fallback
# ---------------------------------------------------------------------------
# SQLite's WAL mode requires shared-memory (mmap) coordination and fcntl
# byte-range locks that don't reliably work on network filesystems (NFS,
# SMB/CIFS, some FUSE mounts, WSL1).  Upstream documents this explicitly:
# https://www.sqlite.org/wal.html#sometimes_queries_return_sqlite_busy_in_wal_mode
#
# On those filesystems ``PRAGMA journal_mode=WAL`` raises
# ``sqlite3.OperationalError: locking protocol`` (SQLITE_PROTOCOL).  If we
# propagate that, every feature backed by state.db / kanban.db breaks
# silently — /resume, /title, /history, /branch, kanban dispatcher, etc.
#
# Instead, fall back to ``journal_mode=DELETE`` (the pre-WAL default) which
# works on NFS.  Concurrency drops — concurrent readers are blocked during
# a write — but the feature works.
_WAL_INCOMPAT_MARKERS = (
    "locking protocol",       # SQLITE_PROTOCOL on NFS/SMB
    "not authorized",         # Some FUSE mounts block WAL pragma outright
)

# Last SessionDB() init error, per-process.  Surfaced in /resume and
# related slash-command error strings so users know WHY the DB is
# unavailable instead of getting a bare "Session database not available."
# Only SessionDB.__init__ writes to this; kanban_db.connect() failures
# do not update it (by design — kanban failures are reported via their
# own caller's error handling, not via /resume-style slash commands).
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()

# Paths for which we've already logged a WAL-fallback WARNING.  Without
# this, kanban_db.connect() (called on every kanban operation — see
# hermes_cli/kanban_db.py for ~30 call sites) would re-log the same
# filesystem-incompat warning on every connection, filling errors.log.
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()

_FTS_TRIGGERS = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear) the most recent state.db init failure.

    Thread-safe via _last_init_error_lock.  Callers pass a message to
    record a failure or None to clear.  SessionDB.__init__ only calls
    this to SET on failure — it deliberately does NOT clear on success,
    because in a multi-threaded caller (e.g. gateway / web_server per-
    request SessionDB() instantiation), a concurrent successful open
    racing past a different thread's failure would erase the cause
    string that thread's /resume handler is about to format.  Explicit
    clears (e.g. test fixtures) are still supported by passing None.
    """
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Return the most recent state.db init failure, if any.

    Slash-command handlers (``/resume``, ``/title``, ``/history``, ``/branch``)
    call this to surface the underlying cause in their error messages when
    ``_session_db is None``.  Returns ``None`` if SessionDB initialized
    successfully (or hasn't been attempted).
    """
    return _last_init_error


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """Format a user-facing 'session DB unavailable' message with cause.

    When ``SessionDB()`` init fails, callers set ``_session_db = None`` and
    several slash commands (/resume, /title, /history, /branch) previously
    responded with a bare ``"Session database not available."`` — no
    indication of WHY.  This helper includes the captured cause (typically
    ``"locking protocol"`` from NFS/SMB) and points users at the known
    culprit so they can fix it themselves.

    Example output:
        Session database not available: locking protocol (state.db may be
        on NFS/SMB — see https://www.sqlite.org/wal.html).
    """
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = ""
    if any(marker in cause.lower() for marker in _WAL_INCOMPAT_MARKERS):
        hint = " (state.db may be on NFS/SMB/FUSE — see https://www.sqlite.org/wal.html)"
    return f"{prefix}: {cause}{hint}."


def _on_disk_journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    """Read the journal mode from the SQLite DB header on disk.

    Returns the mode string (e.g. ``"wal"``, ``"delete"``), or ``None``
    if the value cannot be determined (new DB, or PRAGMA read failed).
    """
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    mode = row[0]
    if isinstance(mode, bytes):  # defensive: sqlite3 occasionally returns bytes
        try:
            mode = mode.decode("ascii")
        except UnicodeDecodeError:
            return None
    return str(mode).strip().lower() if mode is not None else None


def apply_wal_with_fallback(
    conn: sqlite3.Connection,
    *,
    db_label: str = "state.db",
) -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the journal mode actually set (``"wal"`` or ``"delete"``).

    On WAL-incompatible filesystems (NFS, SMB, some FUSE), SQLite raises
    ``OperationalError("locking protocol")`` when setting WAL.  We fall
    back to DELETE mode — the pre-WAL default, which works on NFS — and
    log one WARNING explaining why.

    The WARNING is deduplicated per ``db_label``: repeated connections
    to the same underlying DB (e.g. kanban_db.connect() which is called
    on every kanban operation) log once per process, not once per call.
    Different db_labels log independently, so state.db and kanban.db
    each get one warning on the same NFS mount.

    Shared by :class:`SessionDB` and ``hermes_cli.kanban_db.connect`` so
    both databases get identical fallback behavior.

    Never downgrades to DELETE if the on-disk DB header reports WAL — see _on_disk_journal_mode.
    """
    # Read-only probe — no flock, no checkpoint, no WAL/SHM unlink.
    # Skipping the set-pragma prevents WAL-init from unlinking files other connections hold open.
    try:
        current_mode = conn.execute("PRAGMA journal_mode").fetchone()
        if current_mode and current_mode[0] == "wal":
            return "wal"
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return "wal"
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            # Unrelated OperationalError — don't silently swallow.
            raise
        # Don't downgrade if another process already set WAL on disk.
        existing = _on_disk_journal_mode(conn)
        if existing == "wal":
            raise
        _log_wal_fallback_once(db_label, exc)
        conn.execute("PRAGMA journal_mode=DELETE")
        return "delete"


def _log_wal_fallback_once(db_label: str, exc: Exception) -> None:
    """Log a single WARNING per (process, db_label) about WAL fallback.

    Without this dedup, NFS users running kanban (which opens a fresh
    connection on every operation — see hermes_cli/kanban_db.py) would
    fill errors.log with hundreds of identical warnings per hour.
    """
    with _wal_fallback_warned_lock:
        if db_label in _wal_fallback_warned_paths:
            return
        _wal_fallback_warned_paths.add(db_label)
    logger.warning(
        "%s: WAL journal_mode unsupported on this filesystem (%s) — "
        "falling back to journal_mode=DELETE (slower rollback-journal "
        "mode; reduces concurrency but works on NFS/SMB/FUSE). See "
        "https://www.sqlite.org/wal.html for details. This warning "
        "fires once per process per database.",
        db_label,
        exc,
    )

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
"""

# Indexes that reference columns added in later schema versions must be
# created AFTER _reconcile_columns() has had a chance to ADD them on
# existing databases. SCHEMA_SQL above is run by sqlite executescript
# which would otherwise fail on legacy DBs ("no such column: active").
DEFERRED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

# Trigram FTS5 table for CJK substring search.  The default unicode61
# tokenizer splits CJK characters into individual tokens, breaking phrase
# matching.  The trigram tokenizer creates overlapping 3-byte sequences so
# substring queries work natively for any script (CJK, Thai, etc.).
FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    # Attempt a PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._fts_enabled = False
        self._fts_unavailable_warned = False
        try:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                # Short timeout — application-level retry with random jitter
                # handles contention instead of sitting in SQLite's internal
                # busy handler for up to 30s.
                timeout=1.0,
                # auto-starts transactions on DML, which conflicts with our
                # explicit BEGIN IMMEDIATE.  None = we manage transactions
                # ourselves.
                isolation_level=None,
            )
            self._conn.row_factory = sqlite3.Row
            apply_wal_with_fallback(self._conn, db_label="state.db")
            self._conn.execute("PRAGMA foreign_keys=ON")

            self._init_schema()
        except Exception as exc:
            # Capture the cause so /resume and friends can surface WHY the
            # session DB is unavailable instead of a bare "Session database
            # not available."  Callers that catch this exception keep their
            # existing ``self._session_db = None`` degradation path.
            #
            # Note: we deliberately do NOT clear _last_init_error on the
            # success path (no else branch).  In multi-threaded callers
            # (gateway, web_server per-request SessionDB()), a concurrent
            # successful open racing past this failure would erase the
            # cause that another thread's /resume is about to format.
            # Tests that need to reset the state can call
            # ``hermes_state._set_last_init_error(None)`` explicitly.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # 1.6 __init__ — 打开 SQLite 连接、设 WAL、建 schema
    # ─────────────────────────────────────────────────────────────────────
    #
    # === 这是干什么的? ===
    # SessionDB 的入口:打开 ~/.hermes/state.db,设置 WAL 模式,
    # 初始化 schema(创建表 + 调 reconcile)。
    #
    # === 关键设计点 ===
    #   1. check_same_thread=False — 允许多线程共用 1 个连接
    #      (Hermes gateway / web_server 都有 per-request SessionDB)
    #   2. timeout=1.0 — 短超时,应用层做 jitter retry
    #      (不靠 SQLite 内部 busy handler 等 30 秒)
    #   3. isolation_level=None — 自己管 transaction
    #      (默认的 "deferred" 跟我们的 BEGIN IMMEDIATE 冲突)
    #   4. _init_schema() — 启动时建表 + 调 _reconcile_columns 加缺失列
    #      失败时 _set_last_init_error 记录原因,供 /resume 等 slash 命令用
    #
    # === 为什么失败要保留 _last_init_error? ===
    # 多线程场景下,线程 A 失败 → 线程 B 成功打开,如果在 except 里清掉 error
    # 线程 A 的 /resume 就会显示 "Session database not available" 而不知道原因
    # 故**故意**不写 else 分支,只在测试时显式 reset
    # ─────────────────────────────────────────────────────────────────────

    # ── Core write helper ──

    # 1.13 _is_fts5_unavailable_error — 判断 sqlite 报错是不是"FTS5 不可用"
    # 用途:有些 pip 安装的 python 携带的 sqlite 不带 FTS5 模块
    #      此时所有 FTS5 操作会抛 "no such module: fts5"
    # 判定:错误信息里**同时**含 "no such module" 和 "fts5" 两个关键词
    # 用 lower() 防止大小写不一致
    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        return "no such module" in err and "fts5" in err

    # 1.14 _warn_fts5_unavailable — FTS5 不可用时只警告 1 次
    # 关键:_fts_unavailable_warned flag 保证**只警告 1 次**(避免刷屏)
    # 一旦标记为不可用,后续调用 no-op
    # 警告文案里给用户装环境的链接 — https://hermes-agent.nousresearch.com
    def _warn_fts5_unavailable(self, exc: sqlite3.OperationalError) -> None:
        self._fts_enabled = False
        if self._fts_unavailable_warned:
            return
        self._fts_unavailable_warned = True
        logger.warning(
            "SQLite FTS5 unavailable for %s; full-text session search "
            "disabled. This usually means Hermes is running on an "
            "unsupported install (e.g. a pip-installed or pip-managed "
            "Python whose bundled SQLite lacks FTS5) rather than a "
            "mainline install. Some features may be missing or behave "
            "differently. Install the supported way: "
            "https://hermes-agent.nousresearch.com (underlying error: %s)",
            self.db_path,
            exc,
        )

    # 1.15 _sqlite_supports_fts5 — 探针,验证当前 sqlite 是否带 FTS5
    # 做法:CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)
    #      成功 = 带;失败 = 不带
    # 用 temp table 不污染 schema
    # DROP 掉探针表(虽然 temp 也会自动清)
    def _sqlite_supports_fts5(self, cursor: sqlite3.Cursor) -> bool:
        try:
            cursor.execute("CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)")
            cursor.execute("DROP TABLE temp._hermes_fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    # 1.16 _drop_fts_triggers — 删 FTS5 同步触发器
    # 触发器在 messages 表 INSERT/UPDATE/DELETE 时同步更新 messages_fts
    # 删触发器 → 后续写不再同步 FTS,但 messages_fts 表还在
    # 用途:_init_schema 里,如果探测到 FTS5 不可用,先删触发器
    #      这样写消息不再因 FTS 错误失败(降级)
    @staticmethod
    def _drop_fts_triggers(cursor: sqlite3.Cursor) -> None:
        for trigger in _FTS_TRIGGERS:
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    # 1.17 _fts_trigger_count — 数 FTS 触发器还剩几个
    # 配合 _drop_fts_triggers / _ensure_fts_schema:
    # 删完触发器后,这个数应该 = 0
    # 重建后会回到 _FTS_TRIGGERS 列表长度
    # 用 ? 占位符 + 列表拼 IN(?,?,?) — 避免 SQL 注入
    @staticmethod
    def _fts_trigger_count(cursor: sqlite3.Cursor) -> int:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        row = cursor.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        ).fetchone()
        return int(row[0] if not isinstance(row, sqlite3.Row) else row[0])

    # 1.18 _rebuild_fts_indexes — 从 messages 表重建 FTS 索引
    # 适用场景:
    #   1. 首次启用 FTS(触发器被删过,现在要重建)
    #   2. messages 表被批量导入(SQL 文件恢复)
    # 做法:DELETE FROM messages_fts + 重新 INSERT
    # 注意:content / tool_name / tool_calls 三个字段都进 FTS
    #  → 全文搜索能搜到工具名 + 工具结果
    @staticmethod
    def _rebuild_fts_indexes(cursor: sqlite3.Cursor) -> None:
        for table_name in ("messages_fts", "messages_fts_trigram"):
            cursor.execute(f"DELETE FROM {table_name}")
        cursor.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )
        cursor.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )

    # 1.19 _fts_table_probe — 探测 FTS5 virtual table 是否可用
    # 3 种状态用 Optional[bool] 表达:
    #   True  = 存在且能查
    #   False = 真的不存在(走 _ensure_fts_schema 创建)
    #   None  = FTS5 模块不可用(整个降级,不再尝试)
    # 通过 SELECT * FROM table LIMIT 0 探针 — 不返回数据,只验 schema
    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            return True
        except sqlite3.OperationalError as exc:
            if self._is_fts5_unavailable_error(exc):
                self._warn_fts5_unavailable(exc)
                return None
            if "no such table" in str(exc).lower():
                return False
            raise

    # 1.20 _ensure_fts_schema — 确保 FTS5 virtual table + 触发器存在
    # 关键设计:即使表已存在,也要 executescript 跑 DDL
    # 原因:之前 no-FTS5 runtime 把触发器删了,现在 FTS5 恢复了
    #      触发器要重建 — executescript 里的 CREATE TRIGGER IF NOT EXISTS 会无脑跑
    # 返 True = 成功(可用);False = FTS5 不可用(降级)
    def _ensure_fts_schema(
        self,
        cursor: sqlite3.Cursor,
        table_name: str,
        ddl: str,
    ) -> bool:
        status = self._fts_table_probe(cursor, table_name)
        if status is None:
            return False
        try:
            # Run even when the virtual table exists so any dropped or missing
            # triggers are recreated after a previous no-FTS5 runtime disabled
            # them to keep message writes working.
            cursor.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    # 1.10 _execute_write — 所有写操作的事务包装(本类最重要的方法之一)
    #
    # === 这是干什么的? ===
    # 包装 BEGIN IMMEDIATE + 业务函数 + COMMIT,加 jitter retry 防锁竞争。
    # 所有 INSERT / UPDATE / DELETE 都应该走这个,**不要**直接调 self._conn。
    #
    # === 关键设计 ===
    #   1. BEGIN IMMEDIATE — 事务**开始**就抢 WAL 写锁(不是 commit 时)
    #      → 锁竞争立刻浮面,不会到 commit 才发现
    #   2. jitter retry — 失败后等 20-150ms **随机**时间,重试
    #      → 打破 convoy pattern(大家都等同样的退避时间,继续挤)
    #   3. caller **不能**调 commit — 这层统一管
    #   4. 每 50 次写做 1 次 PASSIVE WAL checkpoint
    #      → 防止 WAL 文件无限增长
    #
    # === 错误处理 ===
    #   * 业务函数抛异常 → rollback → 重新 raise
    #   * "locked"/"busy" → sleep + retry
    #   * 其他 SQLite 错 → 直接抛(不重试)
    #   * 重试用尽 → 抛 last_err
    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(
                            self._WRITE_RETRY_MIN_S,
                            self._WRITE_RETRY_MAX_S,
                        )
                        time.sleep(jitter)
                        continue
                # Non-lock error or retries exhausted — propagate.
                raise
        # Retries exhausted (shouldn't normally reach here).
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    # 1.11 _try_wal_checkpoint — PASSIVE WAL checkpoint(每 50 次写触发 1 次)
    # PASSIVE 模式:**不阻塞**别的事务,不抢锁,只把"没人在用"的 WAL frame 落盘
    # 配合 _execute_write 的 N-write 计数,定期触发,防止 WAL 文件无限增长
    # 失败永远吞掉(best-effort)— checkpoint 不是关键路径
    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.

        Flushes committed WAL frames back into the main DB file for any
        frames that no other connection currently needs.  Keeps the WAL
        from growing unbounded when many processes hold persistent
        connections.
        """
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception:
            pass  # Best effort — never fatal.

    # 1.7 close — 关闭连接(出口前最后做 1 次 PASSIVE checkpoint)
    # 设计:进程退出前帮忙做一次 checkpoint,WAL 文件不会无限增长
    # 跟 _try_wal_checkpoint 区别:close 是**用户主动调**的(进程退出)
    # _try_wal_checkpoint 是 _execute_write 自动周期性触发
    # 错误吞掉(best-effort)— close 阶段不能因为 checkpoint 失败导致关连接失败
    def close(self):
        """Close the database connection.

        Attempts a PASSIVE WAL checkpoint first so that exiting processes
        help keep the WAL file from growing unbounded.
        """
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    # 1.8 _parse_schema_columns — 从 SCHEMA_SQL 解析"期望的列"
    # 巧妙之处:**用 in-memory SQLite 自己解析 DDL**,不写 regex
    # 原因:DEFAULT 表达式里可能有逗号、CHECK 约束里可能有括号、REFERENCES 嵌套
    #      用 regex 解析这些**一定会出 edge case bug**
    #      让 SQLite 自己 parse,提取 PRAGMA table_info — 0 边界问题
    # 配合 _reconcile_columns → 整 schema 演化是**声明式**的
    # 加新列?改 SCHEMA_SQL 一行就行
    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    # 1.9 _reconcile_columns — 声明式 schema 演化(Beets 模式)
    # 流程:从 SCHEMA_SQL 拿"期望列"→ PRAGMA table_info 拿"实际列" → 差集 ALTER ADD
    # **完全不需要**写版本号判断的 migration 块
    # 加 1 个列 → 改 SCHEMA_SQL 1 行 → 下次启动自动加上去
    # 错误吞掉(DEBUG 级别):"duplicate column name"是预期的(并发),不报警
    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Follows the Beets/sqlite-utils pattern: the CREATE TABLE definition
        in SCHEMA_SQL is the single source of truth for the desired schema.
        On every startup this method diffs the live columns (via PRAGMA
        table_info) against the declared columns, and ADDs any that are
        missing.

        This makes column additions a declarative operation — just add
        the column to SCHEMA_SQL and it appears on the next startup.
        Version-gated migration blocks are no longer needed for ADD COLUMN.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            # Get current columns from the live table
            try:
                rows = cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
            live_cols = set()
            for row in rows:
                # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
                name = row[1] if isinstance(row, (tuple, list)) else row["name"]
                live_cols.add(name)

            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}'
                        )
                    except sqlite3.OperationalError as exc:
                        # Expected: "duplicate column name" from a race or
                        # re-run.  Unexpected: "Cannot add a NOT NULL column
                        # with default value NULL" from a schema mistake.
                        # Log at DEBUG so it's visible in agent.log.
                        logger.debug(
                            "reconcile %s.%s: %s", table_name, col_name, exc,
                        )

    # 1.12 _init_schema — bootstrap 整个 schema(创建 + 调和 + 索引)
    #
    # === 这是干什么的? ===
    # SessionDB 启动时调的"建表总入口":跑 SCHEMA_SQL → reconcile 加列 → 补索引
    #
    # === 4 个步骤 ===
    #   1. cursor.executescript(SCHEMA_SQL)  一次性跑所有 CREATE TABLE
    #   2. self._reconcile_columns(cursor)  声明式加缺失列(见 1.9)
    #   3. CREATE INDEX IF NOT EXISTS ...    部分索引(reconcile 之后才能建)
    #   4. cursor.executescript(DEFERRED_INDEX_SQL)  延后索引
    #
    # === 顺序的重要性 ===
    # 索引里 WHERE 引用了 reconcile 加的列(如 ``active``)
    # 如果在 reconcile **之前** 跑 CREATE INDEX,初始启动会失败
    # → 必须在 reconcile **之后** 才能建
    #
    # === schema_version 表保留 ===
    # 数据迁移(改老行的内容)不能声明式做,需要版本号管理
    # 所以 schema_version 表还在,留给未来的 data migration
    def _init_schema(self):
        """Create tables and FTS if they don't exist, reconcile columns.

        Schema management follows the declarative reconciliation pattern
        (Beets, sqlite-utils): SCHEMA_SQL is the single source of truth.
        On existing databases, _reconcile_columns() diffs live columns
        against SCHEMA_SQL and ADDs any missing ones.  This eliminates
        the version-gated migration chain for column additions, making
        it impossible for reordered or inserted migrations to skip columns.

        The schema_version table is retained for future data migrations
        (transforming existing rows) which cannot be handled declaratively.
        """
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # ── Declarative column reconciliation ──────────────────────────
        # Diff live tables against SCHEMA_SQL and ADD any missing columns.
        # This is idempotent and self-healing: even if a version-gated
        # migration was skipped (e.g. due to version renumbering), the
        # column gets created here.
        self._reconcile_columns(cursor)

        # Indexes that reference reconciler-added columns must be created
        # AFTER _reconcile_columns runs — declaring them in SCHEMA_SQL
        # makes the initial executescript fail on legacy DBs (the index's
        # WHERE clause references a column that doesn't exist yet).
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
                "ON messages(session_id, platform_message_id) "
                "WHERE platform_message_id IS NOT NULL"
            )
        except sqlite3.OperationalError as exc:
            logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)

        # Deferred indexes that reference the reconciler-added ``active``
        # column (idx_messages_session_active) — same ordering constraint.
        cursor.executescript(DEFERRED_INDEX_SQL)

        fts5_available = self._sqlite_supports_fts5(cursor)
        fts_migrations_complete = True
        if not fts5_available:
            # Existing FTS triggers can still fire on messages INSERT/UPDATE
            # even though the current sqlite runtime cannot read the virtual
            # tables they target. Drop only the triggers so core persistence
            # continues; if a future runtime has FTS5, _ensure_fts_schema()
            # recreates them.
            self._drop_fts_triggers(cursor)

        # ── Schema version bookkeeping ─────────────────────────────────
        # Bump to current so future data migrations (if any) can gate on
        # version.  No version-gated column additions remain.
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            # Data migrations that can't be expressed declaratively (row
            # backfills, index changes tied to a specific version step) stay
            # in a version-gated chain. Column additions are handled by
            # _reconcile_columns() above and no longer need entries here.
            if current_version < 10:
                # v10: trigram FTS5 table for CJK/substring search. The
                # virtual table + triggers are created unconditionally via
                # FTS_TRIGRAM_SQL below, but existing rows need a one-time
                # backfill into the FTS index.
                if fts5_available:
                    _fts_trigram_exists = self._fts_table_probe(
                        cursor, "messages_fts_trigram"
                    )
                    if _fts_trigram_exists is False:
                        if self._ensure_fts_schema(
                            cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                        ):
                            cursor.execute(
                                "INSERT INTO messages_fts_trigram(rowid, content) "
                                "SELECT id, content FROM messages WHERE content IS NOT NULL"
                            )
                        else:
                            fts_migrations_complete = False
                    elif _fts_trigram_exists is None:
                        fts_migrations_complete = False
                else:
                    fts_migrations_complete = False
            if current_version < 11:
                # v11: re-index FTS5 tables to cover tool_name + tool_calls and
                # switch from external-content to inline mode. Existing DBs have
                # old-schema FTS tables and triggers that IF NOT EXISTS won't
                # overwrite, so we drop them explicitly and let the post-migration
                # existence checks (below) recreate them from FTS_SQL /
                # FTS_TRIGRAM_SQL, then backfill every message row. Fixes #16751.
                if fts5_available:
                    self._drop_fts_triggers(cursor)
                    for _tbl in ("messages_fts", "messages_fts_trigram"):
                        try:
                            cursor.execute(f"DROP TABLE IF EXISTS {_tbl}")
                        except sqlite3.OperationalError as exc:
                            if not self._is_fts5_unavailable_error(exc):
                                raise
                            self._warn_fts5_unavailable(exc)
                            fts5_available = False
                            fts_migrations_complete = False
                            break

                    if fts5_available:
                        # Recreate virtual tables + triggers with the new inline-mode
                        # schema that indexes content || tool_name || tool_calls.
                        if (
                            self._ensure_fts_schema(cursor, "messages_fts", FTS_SQL)
                            and self._ensure_fts_schema(
                                cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                            )
                        ):
                            # Backfill both indexes from every existing messages row.
                            cursor.execute(
                                "INSERT INTO messages_fts(rowid, content) "
                                "SELECT id, "
                                "COALESCE(content, '') || ' ' || "
                                "COALESCE(tool_name, '') || ' ' || "
                                "COALESCE(tool_calls, '') "
                                "FROM messages"
                            )
                            cursor.execute(
                                "INSERT INTO messages_fts_trigram(rowid, content) "
                                "SELECT id, "
                                "COALESCE(content, '') || ' ' || "
                                "COALESCE(tool_name, '') || ' ' || "
                                "COALESCE(tool_calls, '') "
                                "FROM messages"
                            )
                        else:
                            fts_migrations_complete = False
                else:
                    fts_migrations_complete = False
            if current_version < 12:
                # v12: messages.active flag for rewind/undo soft-deletion.
                # The declarative reconcile_columns() above adds the
                # column itself; this UPDATE is belt-and-suspenders to
                # ensure any rows that pre-existed the ADD COLUMN have
                # active=1 rather than NULL.
                try:
                    cursor.execute(
                        "UPDATE messages SET active = 1 WHERE active IS NULL"
                    )
                except sqlite3.OperationalError:
                    pass
            if current_version < SCHEMA_VERSION and fts_migrations_complete:
                cursor.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )

        # Unique title index — always ensure it exists
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                "ON sessions(title) WHERE title IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass  # Index already exists

        if fts5_available:
            # FTS5 setup. Run the DDL even when the virtual table exists so
            # CREATE TRIGGER IF NOT EXISTS repairs trigger-only degradation from
            # an earlier no-FTS5 runtime.
            triggers_need_repair = self._fts_trigger_count(cursor) < len(_FTS_TRIGGERS)
            self._fts_enabled = self._ensure_fts_schema(cursor, "messages_fts", FTS_SQL)

            # Trigram FTS5 for CJK/substring search. This is optional relative
            # to the main FTS table; if it cannot be created, CJK search falls
            # back to LIKE.
            if self._fts_enabled:
                trigram_enabled = self._ensure_fts_schema(
                    cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                )
                if trigram_enabled and triggers_need_repair:
                    self._rebuild_fts_indexes(cursor)

        self._conn.commit()

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    # 1.31 _insert_session_row — 共享的 INSERT OR IGNORE 入口
    # 跟 create_session / ensure_session 配合:
    #   * 这俩是"对外"的薄包装
    #   * 这个是"对内"的实际写库函数
    # INSERT OR IGNORE → 已存在 session_id 时静默 no-op(并发安全)
    # 字段全集:source / user_id / model / model_config / system_prompt / parent / cwd
    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
        cwd: str = None,
    ) -> None:
        """Shared INSERT OR IGNORE for session rows."""
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,
                   system_prompt, parent_session_id, cwd, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    user_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    cwd,
                    time.time(),
                ),
            )
        self._execute_write(_do)

    # 1.1 create_session — 创建一条 session 记录
    # 4 个公开方法之一,对应"开始一个新对话"
    # 实际写库在 _insert_session_row(管 schema 细节)
    # 这里就是个 thin wrapper
    #   session_id: 自己 generate 的 UUID(调用方决定)
    #   source:     "cli" / "telegram" / "discord" / "gateway" ...
    #   **kwargs:   model / title / cwd 等其他字段
    # 返 session_id(方便调用方链式调用)
    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id
    # 1.21 end_session — 标记 session 结束
    # 关键设计:**first end_reason wins**
    # 如果 session 已经被 end 过(有 ended_at),UPDATE 不生效
    # 这保护 compression 拆分的 session — 即使 stale 的 /resume 错误地又 end 一次
    # 也不会覆盖掉原本的 "compression" 原因
    # 想覆盖?先调 reopen_session() 再 end
    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended.

        No-ops when the session is already ended. The first end_reason wins:
        compression-split sessions must keep their ``end_reason = 'compression'``
        record even if a later stale ``end_session()`` call (e.g. from a
        desynced CLI session_id after ``/resume`` or ``/branch``) targets them
        with a different reason. Use ``reopen_session()`` first if you
        intentionally need to re-end a closed session with a new reason.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )
        self._execute_write(_do)

    # 1.22 reopen_session — 撤销 end_session(让 session 可被 /resume)
    # 简单:UPDATE sessions SET ended_at = NULL, end_reason = NULL
    # 不检查 ended_at 当前是什么 — 反正就是要清掉
    # 用例:/resume 之前如果目标 session 被 end 过,先 reopen
    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    # 1.23 update_session_cwd — 持久化 session 的工作目录
    # 用途:CLI 报告 cwd 后存进 session
    # 让 /resume 时能恢复到原 cwd(不用每次重新 cd)
    # 早返回:session_id 或 cwd 为空 → no-op(防写入空数据)
    def update_session_cwd(self, session_id: str, cwd: str) -> None:
        """Persist the session working directory when a frontend knows it."""
        if not session_id or not cwd:
            return

        def _do(conn):
            conn.execute("UPDATE sessions SET cwd = ? WHERE id = ?", (cwd, session_id))

        self._execute_write(_do)
    # ──────────────────────────────────────────────────────────────────────
    # Compression locks
    # ──────────────────────────────────────────────────────────────────────
    # Atomic per-session locks that prevent two compression paths from
    # racing on the same session_id and producing orphan child sessions.
    #
    # The race: ``conversation_compression.py`` rotates ``agent.session_id``
    # as a side effect of a successful compression (end old session, create
    # new). That mutation is local to the AIAgent instance — but ``state.db``
    # is shared across all instances. Two AIAgents that share the same
    # ``session_id`` at the moment they both decide to compress (most
    # commonly the parent turn's agent + a background-review fork started
    # right after the turn ended) each end the parent and create their own
    # NEW session, parented to the same old id. The gateway SessionEntry
    # only catches one rotation; the other child silently accumulates
    # writes — Damien's "parent → two orphan children" repro shape.
    #
    # The lock is keyed by ``session_id`` and is held for the duration of
    # the compress() call plus the rotation. ``holder`` identifies the
    # current owner (pid:tid:nonce) for diagnostics; the lock is recovered
    # via ``expires_at`` if the holder process crashed without releasing.
    # 1.2 try_acquire_compression_lock — 压缩互斥锁(原子的)
    # 用途:防止两个 compression path 同时压缩同一个 session
    # race:conversation_compression.py 压缩成功后会**旋转 session_id**
    #       如果两个 compressor 同时跑,都会旋转,导致 lineage 错乱
    #
    # === 实现 ===
    # 单事务里三步:
    #   1. DELETE 过期锁(stale 锁可以被新 holder 接管,防 compressor 崩溃死锁)
    #   2. INSERT OR IGNORE 新锁(主键冲突时 IGNORE)
    #   3. SELECT 验证自己是不是 holder
    # SQLite 串行化写,这三步原子
    #
    # === TTL 默认 300s ===
    # 万一 compressor 崩了不释放,5 分钟后自动过期
    # 防御:压缩半截不会永久卡住 session
    #
    # === Fail open 哲学 ===
    # 如果 lock 子系统本身坏了 → 返 False → caller 跳过压缩
    # 这是"安全失败"——不压缩还能用,锁崩了不能冒险
    def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Try to atomically acquire the compression lock for ``session_id``.

        Returns ``True`` on success (caller now owns the lock and must
        release via :meth:`release_compression_lock`).  Returns ``False``
        if another holder already owns a non-expired lock — the caller
        MUST NOT proceed with compression in that case (its rotation would
        race against the holder's, splitting the session lineage).

        Expired locks (``expires_at < now``) are reclaimed transparently:
        the stale row is deleted and the new holder acquires it. This
        prevents a crashed compressor from permanently blocking the
        session.

        Implementation: single-transaction DELETE-expired + INSERT-or-IGNORE,
        followed by a SELECT to confirm we got the row. SQLite serialises
        writes, so the whole sequence is atomic against other writers.
        """
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            # First: reclaim any expired lock for this session_id.
            conn.execute(
                "DELETE FROM compression_locks "
                "WHERE session_id = ? AND expires_at < ?",
                (session_id, now),
            )
            # Then: try to insert. INSERT OR IGNORE returns no rowcount
            # difference — verify ownership via SELECT.
            conn.execute(
                "INSERT OR IGNORE INTO compression_locks "
                "(session_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, holder, now, expires_at),
            )
            row = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row is not None and (
                row["holder"] if isinstance(row, sqlite3.Row) else row[0]
            ) == holder

        try:
            return bool(self._execute_write(_do))
        except sqlite3.Error as exc:
            logger.warning(
                "try_acquire_compression_lock(%s) failed: %s",
                session_id, exc,
            )
            # Fail open: returning False makes the caller skip compression,
            # which is the safe behaviour when the lock subsystem is broken.
            return False

    # 1.5 release_compression_lock — 释放压缩锁(必须是自己 holder)
    # 4 个公开方法之四(compress_session 在 Hermes 里实际是这套锁)
    # 关键:**必须有 holder 参数**
    # 只删 holder 匹配的——不会误删别人的锁
    # 错误兜底:release 失败只 warning,不当致命错误
    def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release the compression lock for ``session_id`` iff we own it.

        Idempotent: no-op when the lock has already expired and been
        reclaimed by a different holder, or when no lock exists. The
        ``holder`` check prevents a late-returning compressor from
        clobbering a fresh lock held by someone else.
        """
        if not session_id:
            return

        def _do(conn):
            conn.execute(
                "DELETE FROM compression_locks "
                "WHERE session_id = ? AND holder = ?",
                (session_id, holder),
            )

        try:
            self._execute_write(_do)
        except sqlite3.Error as exc:
            logger.warning(
                "release_compression_lock(%s) failed: %s",
                session_id, exc,
            )

    # 1.24 get_compression_lock_holder — 查锁 holder(诊断用)
    # 不是锁协议的一部分(协议用 try_acquire / release)
    # 用途:调试时查"现在谁拿着这个 session 的压缩锁"
    # 返 None = 没锁 / 锁已过期
    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the current (non-expired) holder for ``session_id``, or None.

        Diagnostic helper — not used by the locking protocol itself.
        """
        if not session_id:
            return None
        now = time.time()
        row = self._conn.execute(
            "SELECT holder FROM compression_locks "
            "WHERE session_id = ? AND expires_at >= ?",
            (session_id, now),
        ).fetchone()
        if row is None:
            return None
        return row["holder"] if isinstance(row, sqlite3.Row) else row[0]


    # 1.25 update_system_prompt — 存整段 system prompt 快照
    # 用途:session 启动时调一次,把拼好的完整 system prompt 存进 sessions 表
    # 之后 /history 等 UI 可以看当时是什么 prompt
    # 跟 volatile tier 的 system prompt 是不同的存
    #  (运行时 system prompt 在内存里,这个在数据库里给历史回放用)
    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    # 1.26 update_session_model — 强制设 session 的 model 列
    # 跟 update_token_counts 区别:这里是**无条件**UPDATE
    # 用户跑 /model 切了模型后,dashboard 要反映最新选择
    # 而 token 计数用 COALESCE(model, ?) 只在 NULL 时填
    def update_session_model(self, session_id: str, model: str) -> None:
        """Update the model for a session after a mid-session switch.

        Unlike ``update_token_counts`` which uses ``COALESCE(model, ?)``
        (only filling in NULL), this unconditionally sets the model column
        so that the dashboard reflects the user's latest /model choice.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET model = ? WHERE id = ?",
                (model, session_id),
            )
        self._execute_write(_do)

    # 1.27 update_token_counts — 更新 token / 费用 / api_call 计数
    #
    # === 关键设计:absolute 模式 ===
    #   absolute=False (默认) → **累加**(用 input_tokens + ?)
    #       适用:CLI 路径,每 API call 传 delta
    #   absolute=True        → **直接设**(用 input_tokens = ?)
    #       适用:gateway 路径,cached agent 已经持有累计值
    #
    # === COALESCE(model, ?) 模式 ===
    # 只在 model 为 NULL 时填入(避免覆盖用户在 /model 后改的)
    # 配合 update_session_model — 一个"只填",一个"必盖"
    #
    # === 防 race ===
    # 先 _insert_session_row(... "unknown") → 防止 cron / kanban / delegate 并发下
    # 初始 create_session 失败 → UPDATE 静默 0 行
    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        # Ensure the session row exists so the UPDATE doesn't silently affect
        # 0 rows.  Under concurrent load (cron + kanban + delegate_task) the
        # initial create_session() may have failed due to SQLite locking.
        # INSERT OR IGNORE is cheap and idempotent.
        self._insert_session_row(session_id, "unknown", model=model)
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = ?
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider,
            billing_base_url,
            billing_mode,
            model,
            api_call_count,
            session_id,
        )
        def _do(conn):
            conn.execute(sql, params)
        self._execute_write(_do)

    # 1.28 ensure_session — 幂等的"建或得"session
    # 比 create_session 更"松":
    #   * source 默认 "unknown"(不用 caller 决定)
    #   * 其他字段从 **kwargs 透传
    # 实现就是 INSERT OR IGNORE — 已存在则 no-op
    # 用途:晚到的 token counter / metadata 写入,先确保 session 行存在
    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
        **kwargs,
    ) -> str:
        """Ensure a session row exists (INSERT OR IGNORE). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    # 1.29 prune_empty_ghost_sessions — 删 TUI "ghost" session
    # Ghost 定义:
    #   * source = 'tui'
    #   * title IS NULL
    #   * ended_at IS NOT NULL
    #   * started_at > 24 小时前
    #   * 没有 messages(空 session)
    # 这种是用户开了 TUI 又立刻关的"幻影", 24h 后清掉
    # 删完还会调 _remove_session_files 清磁盘 .json / .jsonl
    def prune_empty_ghost_sessions(self, sessions_dir: "Optional[Path]" = None) -> int:
        """Remove empty TUI ghost sessions (no messages, no title, >24hr old)."""
        cutoff = time.time() - 86400  # Only sessions older than 24 hours

        def _do(conn):
            rows = conn.execute("""
                SELECT id FROM sessions
                WHERE source = 'tui'
                  AND title IS NULL
                  AND ended_at IS NOT NULL
                  AND started_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
            """, (cutoff,)).fetchall()
            ids = [r[0] if isinstance(r, (tuple, list)) else r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM sessions WHERE id IN ({placeholders})", ids
                )
            return ids

        removed_ids = self._execute_write(_do) or []
        # Clean up any on-disk session files (belt-and-suspenders)
        if sessions_dir and removed_ids:
            for sid in removed_ids:
                self._remove_session_files(sessions_dir, sid)
        return len(removed_ids)

    # 1.30 finalize_orphaned_compression_sessions — 修复 #20001
    # 修的 bug:压缩旋转后,某些 child session 永远不被 end_session
    # 现象:子 session 有 messages、没 end_reason、api_call_count=0
    # 解法:7 天后扫描,把这种 child 标 ended_at + end_reason='orphaned_compression'
    # **非破坏性**:所有 messages 保留
    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuation sessions as ended.

        Targets child sessions that were never finalized: parent is ended
        with reason='compression', child has messages but no end_reason/ended_at
        and api_call_count=0.  Non-destructive: preserves all messages and sets
        end_reason='orphaned_compression'.  Fix for #20001.
        """
        cutoff = time.time() - 604800  # 7 days

        def _do(conn):
            now = time.time()
            result = conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?,
                    end_reason = 'orphaned_compression'
                WHERE api_call_count = 0
                  AND end_reason IS NULL
                  AND ended_at IS NULL
                  AND started_at < ?
                  AND parent_session_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = sessions.parent_session_id
                        AND p.end_reason = 'compression'
                        AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages m
                      WHERE m.session_id = sessions.id
                  )
                """,
                (now, cutoff),
            )
            return result.rowcount

        return self._execute_write(_do) or 0

    # 1.63 get_session — 拿 1 个 session 完整 row
    # 简单 SELECT * WHERE id=?,返 dict 或 None
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    # 1.64 resolve_session_id — ID 或前缀 → 完整 ID
    # 用途:用户输 "abc123" 想 resume,系统能补全成 "abc123-def456-..."
    # 先 exact match,再用 LIKE prefix match
    # **多个**匹配 → 返 None(歧义,不能猜)
    # ESCAPE '\' 防 LIKE 通配符误匹配
    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = (
            session_id_or_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    # 1.33 sanitize_title — 校验 + 清洗 title
    # 干 5 件事:
    #   1. 去首尾空白
    #   2. 去 ASCII 控制字符(0x00-0x1F, 0x7F)但保留 \t \n \r
    #   3. 去 Unicode 控制字符(零宽 / RTL override / 双向等)
    #   4. 合并连续空白为 1 个空格
    #   5. 限制 MAX_TITLE_LENGTH=100
    # 返清洗后字符串 / None / ValueError(过长)
    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    # 1.38 set_session_title — 设/改 session title
    # 约束:
    #   * title 唯一(整个 sessions 表)→ 重复抛 ValueError
    #   * title 走 sanitize_title(去控制字符 + 限长)
    # 返:True = 找到且设了;False = session 不存在
    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)
        def _do(conn):
            if title:
                # Check uniqueness (allow the same session to keep its own title)
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = cursor.fetchone()
                if conflict:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict['id']}"
                    )
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            return cursor.rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    # 1.34 get_session_title — 拿 title
    # 简单 SELECT,返 title 列或 None
    # 用 self._lock 短锁(读不需要事务)
    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    # 1.35 get_session_by_title — 按 title 查 session
    # 返完整 session dict(不只是 id)
    # 用 SELECT * — call site 拿到所有列
    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ?", (title,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, title, started_at FROM sessions "
                "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
            numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    # 1.36 get_next_title_in_lineage — 计算 lineage 下一个编号 title
    # 例:base="my session", 已有 "my session" 和 "my session #2"
    #     → 返 "my session #3"
    # 步骤:剥现有 #N 后缀 → 找最大编号 → +1
    # ESCAPE '\' 防 LIKE 通配符(% / _)误匹配
    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
            existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    # 1.37 get_compression_tip — 沿压缩链走到 tip
    # 用途:压缩会把 1 个 session 拆成 parent + child + grandchild + ...
    # 查"最新那个"(tip) — UI 列表显示时只显示 tip,隐藏中间
    # 走 parent_session_id 指针,直到 NULL/没 parent 为止
    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk the compression-continuation chain forward and return the tip.

        A compression continuation is a child session where:
        1. The parent's ``end_reason = 'compression'``
        2. The child was created AFTER the parent was ended (started_at >= ended_at)

        The second condition distinguishes compression continuations from
        delegate subagents or branch children, which can also have a
        ``parent_session_id`` but were created while the parent was still live.

        Returns the session_id of the latest continuation in the chain, or the
        input ``session_id`` if it isn't part of a compression chain (or if the
        input itself doesn't exist).
        """
        current = session_id
        # Bound the walk defensively — compression chains this deep are
        # pathological and shouldn't happen in practice. 100 = plenty.
        for _ in range(100):
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT id FROM sessions "
                    "WHERE parent_session_id = ? "
                    "  AND started_at >= ("
                    "      SELECT ended_at FROM sessions "
                    "      WHERE id = ? AND end_reason = 'compression'"
                    "  ) "
                    "ORDER BY started_at DESC LIMIT 1",
                    (current, current),
                )
                row = cursor.fetchone()
            if row is None:
                return current
            current = row["id"]
        return current

    # 1.41 list_sessions_rich — 列 session 列表(富信息版)
    # 比 SELECT * 多:preview(首条 user message 摘要) + last_active + compression_tip
    # 用途:/history 命令、UI 列表展示
    # 支持 source 过滤 + exclude_sources 反向过滤 + limit
    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        min_message_count: int = 0,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.

        With ``project_compression_tips=True`` (default), sessions that are
        roots of compression chains are projected forward to their latest
        continuation — one logical conversation = one list entry, showing the
        live continuation's id/message_count/title/last_active. This prevents
        compressed continuations from being invisible to users while keeping
        delegate subagents and branches hidden. Pass ``False`` to return the
        raw root rows (useful for admin/debug UIs).

        Pass ``order_by_last_active=True`` to sort by most-recent activity
        instead of original conversation start time. For compression chains,
        the "most-recent activity" is taken from the live tip (not the root),
        so an old conversation that was compressed and continued recently
        surfaces in the correct slot. Ordering is computed at SQL level via
        a recursive CTE that walks compression-continuation edges, so LIMIT
        and OFFSET still apply efficiently.
        """
        where_clauses = []
        params = []

        if not include_children:
            # Show root sessions and branch sessions (whose parent ended with
            # end_reason='branched' before the child was created), while still
            # hiding sub-agent runs and compression continuations (which also
            # carry a parent_session_id but were spawned while the parent was
            # still live — i.e., started_at < parent.ended_at).
            where_clauses.append(
                "(s.parent_session_id IS NULL"
                " OR EXISTS (SELECT 1 FROM sessions p"
                "            WHERE p.id = s.parent_session_id"
                "            AND p.end_reason = 'branched'"
                "            AND s.started_at >= p.ended_at))"
            )

        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        if order_by_last_active:
            # Compute effective_last_active by walking each surfaced session's
            # compression-continuation chain forward in SQL and taking the MAX
            # timestamp across the chain. This lets us ORDER BY + LIMIT at SQL
            # level instead of fetching every row and sorting in Python, while
            # still surfacing old compression roots whose live tip is fresh.
            #
            # The CTE seeds from rows the outer WHERE admits (roots + branch
            # children), then recursively joins forward through
            # compression-continuation edges using the same criteria as
            # get_compression_tip (parent.end_reason='compression' AND
            # child.started_at >= parent.ended_at).
            query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {where_sql}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND child.started_at >= parent.ended_at
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX(COALESCE(
                            (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = cur_id),
                            (SELECT started_at FROM sessions ss WHERE ss.id = cur_id)
                        )) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                SELECT s.*,
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {where_sql}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
            """
            # WHERE params apply twice (CTE seed + outer select).
            params = params + params + [limit, offset]
        else:
            query = f"""
                SELECT s.*,
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            # Drop the internal ordering column so callers see a clean dict.
            s.pop("_effective_last_active", None)
            sessions.append(s)

        # Project compression roots forward to their tips. Each row whose
        # end_reason is 'compression' has a continuation child; replace the
        # surfaced fields (id, message_count, title, last_active, ended_at,
        # end_reason, preview) with the tip's values so the list entry acts
        # as the live conversation. Keep the root's started_at to preserve
        # chronological ordering by original conversation start.
        if project_compression_tips and not include_children:
            projected = []
            for s in sessions:
                if s.get("end_reason") != "compression":
                    projected.append(s)
                    continue
                tip_id = self.get_compression_tip(s["id"])
                if tip_id == s["id"]:
                    projected.append(s)
                    continue
                tip_row = self._get_session_rich_row(tip_id)
                if not tip_row:
                    projected.append(s)
                    continue
                # Preserve the root's started_at for stable sort order, but
                # surface the tip's identity and activity data.
                merged = dict(s)
                for key in (
                    "id", "ended_at", "end_reason", "message_count",
                    "tool_call_count", "title", "last_active", "preview",
                    "model", "system_prompt", "cwd",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = s["id"]
                projected.append(merged)
            sessions = projected

        return sessions

    # 1.42 _get_session_rich_row — 单 session 富信息(内部 helper)
    # 跟 list_sessions_rich 区别:这个只查 1 个
    # 给 list_sessions_rich 复用 + 单独的"看一个 session 详情"调用
    def _get_session_rich_row(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single session with the same enriched columns as
        ``list_sessions_rich`` (preview + last_active). Returns None if the
        session doesn't exist.
        """
        query = """
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE s.id = ?
        """
        with self._lock:
            cursor = self._conn.execute(query, (session_id,))
            row = cursor.fetchone()
        if not row:
            return None
        s = dict(row)
        raw = s.pop("_preview_raw", "").strip()
        if raw:
            text = raw[:60]
            s["preview"] = text + ("..." if len(raw) > 60 else "")
        else:
            s["preview"] = ""
        return s

    # =========================================================================
    # Message storage
    # =========================================================================

    # Sentinel prefix used to distinguish JSON-encoded structured content
    # (multimodal messages: lists of parts like text + image_url) from plain
    # string content. The NUL byte is not legal in normal text, so this
    # cannot collide with real user content.
    _CONTENT_JSON_PREFIX = "\x00json:"

    @classmethod
    # 1.43 _encode_content — 多模态 content 编码(sentinel + JSON)
    # 为什么需要?SQLite content 列 TEXT,但 multimodal message 是 list[dict]
    # 解决:list/dict → JSON 串 → 前面加 \x00json: sentinel
    # sentinel 用 NUL 字节开头 → 不会跟正常文本冲突(NUL 在 text 里非法)
    # 安全标量(str/int/float)原样返回
    # 配套:_decode_content 反向解析
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if content is None or isinstance(content, (str, bytes, int, float)):
            return content
        try:
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            # Last-resort fallback: stringify so persistence never fails.
            return str(content)

    @classmethod
    # 1.44 _decode_content — 反向解析 _encode_content 的 sentinel+JSON
    # 标量原样返回 / sentinel 开头 → JSON.loads / 否则原样
    # 容错:JSON 解析失败 → 返原值(不抛)— 数据损坏时降级
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to decode JSON-encoded message content; "
                    "returning raw string"
                )
                return content
        return content

    # 1.3 append_message — 追加一条 message 到 session(4 个公开方法之二)
    # 返 message row ID(给 LLM 看到 tool_call_id 之外的稳定引用)
    #
    # === 入参多 ===
    # 不仅 role + content,还有:
    #   * tool_name / tool_calls / tool_call_id (tool 消息用)
    #   * reasoning / reasoning_content / reasoning_details (Claude/o1 推理)
    #   * codex_*_items (Codex Responses 协议专属)
    #   * platform_message_id (Telegram 等外部平台的 msg_id,用来做 redact)
    #   * observed (是否从外部平台看到,影响可见性)
    #
    # === 重要细节 ===
    # * JSON 序列化所有结构化字段(reasoning_details / codex items / tool_calls)
    #   SQLite 不能直接 bind list/dict
    # * content 走 _encode_content(支持 multimodal:list of parts → JSON)
    # * 顺手更新 session.message_count(如果 role=tool 还 +tool_call_count)
    # * 全部在一个 _execute_write 事务里(原子)
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).

        ``platform_message_id`` is the external messaging platform's own
        message ID (e.g. Telegram update_id, Yuanbao msg_id).  It is
        independent of the SQLite autoincrement primary key and is used by
        platform-specific flows like yuanbao's recall guard to redact a
        message by its platform-side identifier.
        """
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = (
            json.dumps(reasoning_details)
            if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items)
            if codex_reasoning_items else None
        )
        codex_message_items_json = (
            json.dumps(codex_message_items)
            if codex_message_items else None
        )
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        # Multimodal content (list of parts) must be JSON-encoded: sqlite3
        # cannot bind list/dict parameters directly.
        stored_content = self._encode_content(content)

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_content,
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                    platform_message_id,
                    1 if observed else 0,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        return self._execute_write(_do)

    # 1.32 replace_messages — 原子"删 + 重插"整个 transcript
    # 用途:/retry /undo /compress 等需要重写整段对话的场景
    # 必须 1 个事务:删 + 插**不能**分两次
    #  → 中途失败 = 留下空 session(灾难)
    # 原子性由 _execute_write 保障(BEGIN IMMEDIATE + 1 次 commit)
    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Atomically replace every message for a session.

        Used by transcript-rewrite flows such as /retry, /undo, and /compress.
        The delete + reinsert sequence must commit as one transaction so a
        mid-rewrite failure does not leave SQLite with a partial transcript.
        """

        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )

            now_ts = time.time()
            total_messages = 0
            total_tool_calls = 0
            for msg in messages:
                role = msg.get("role", "unknown")
                tool_calls = msg.get("tool_calls")
                reasoning_details = msg.get("reasoning_details") if role == "assistant" else None
                codex_reasoning_items = (
                    msg.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    msg.get("codex_message_items") if role == "assistant" else None
                )

                reasoning_details_json = (
                    json.dumps(reasoning_details) if reasoning_details else None
                )
                codex_items_json = (
                    json.dumps(codex_reasoning_items) if codex_reasoning_items else None
                )
                codex_message_items_json = (
                    json.dumps(codex_message_items) if codex_message_items else None
                )
                tool_calls_json = json.dumps(tool_calls) if tool_calls else None
                # Accept either `platform_message_id` (new explicit name) or
                # `message_id` (yuanbao's existing convention on message dicts).
                platform_msg_id = (
                    msg.get("platform_message_id") or msg.get("message_id")
                )

                conn.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        role,
                        self._encode_content(msg.get("content")),
                        msg.get("tool_call_id"),
                        tool_calls_json,
                        msg.get("tool_name"),
                        now_ts,
                        msg.get("token_count"),
                        msg.get("finish_reason"),
                        msg.get("reasoning") if role == "assistant" else None,
                        msg.get("reasoning_content") if role == "assistant" else None,
                        reasoning_details_json,
                        codex_items_json,
                        codex_message_items_json,
                        platform_msg_id,
                        1 if msg.get("observed") else 0,
                    ),
                )
                total_messages += 1
                if tool_calls is not None:
                    total_tool_calls += (
                        len(tool_calls) if isinstance(tool_calls, list) else 1
                    )
                now_ts += 1e-6

            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, session_id),
            )

        self._execute_write(_do)

    # 1.4 get_messages — 加载 session 全部 messages(4 个公开方法之三)
    # 4 个公开方法里**最常调**——主循环 / context compressor / session_search 都靠它
    #
    # === 顺序保证 ===
    # 用 AUTOINCREMENT id 排序,不是 timestamp
    # 原因:WSL2 上 clock 偶尔回退(commit c03acca50)
    # 用 id 永远单调递增,顺序永远对
    #
    # === include_inactive 参数 ===
    # 默认 False = 只返 active=1 的
    # True = 含软删的(给 audit / debug / /undo 用)
    # 软删是 rewind_to_message 用的——不是真删,而是 active=0
    #
    # === 解码 ===
    # content 走 _decode_content(可能从 JSON 还原 multimodal)
    # tool_calls 从 JSON 字符串还原成 list
    # 失败兜底:tool_calls → []
    def get_messages(
        self, session_id: str, include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """Load messages for a session in insertion order.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted rows (e.g. for
        audit / debug views of rewound history). See
        :meth:`rewind_to_message` for the soft-delete mechanic.

        Ordered by AUTOINCREMENT id (true insertion order) rather than
        timestamp — see c03acca50 for the WSL2 clock-regression rationale.
        """
        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ?"
                f"{active_clause} ORDER BY id",
                (session_id,),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            result.append(msg)
        return result

    # 1.45 get_messages_around — 锚点窗口读 messages
    # 用途:UI 显示"某条 message 周围的内容"(前后各 N 条)
    # 例:/history 显示某条 user message 周围 5 条
    # 锚点:message_id,前后各取 before / after 条
    # 内部用 ROWID 比较,O(log n) 索引扫描
    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id.

        Returns a dict with:
          - ``window``: up to ``window`` messages before the anchor, the anchor
            itself, and up to ``window`` messages after, ordered by id ascending.
          - ``messages_before``: count of messages strictly before the anchor
            still in the session (== window unless we hit the start).
          - ``messages_after``: count of messages strictly after the anchor
            still in the session (== window unless we hit the end).

        Used by ``session_search`` for both the discovery shape (anchored on the
        FTS5 match) and the scroll shape (anchored on any message id). The
        ``messages_before`` / ``messages_after`` counts let the caller detect
        session boundaries: when either is less than ``window``, the agent has
        reached one end of the session.

        Returns an empty window when ``around_message_id`` is not a real id in
        ``session_id`` — callers decide how to surface that.
        """
        if window < 0:
            window = 0
        with self._lock:
            # Confirm the anchor exists in this session.
            anchor_exists = self._conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor_exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}

            # Two queries: anchor + before (DESC, take window+1), and after
            # (ASC, take window). Final order is id ASC.
            before_rows = self._conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = self._conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()

        # before_rows is DESC; reverse so it's ASC, then concatenate after_rows.
        rows = list(reversed(before_rows)) + list(after_rows)
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in get_messages_around, falling back to []"
                    )
                    msg["tool_calls"] = []
            result.append(msg)

        # before_rows includes the anchor itself; subtract 1 for the count of
        # messages strictly before the anchor in the returned slice.
        messages_before = max(0, len(before_rows) - 1)
        messages_after = len(after_rows)
        return {
            "window": result,
            "messages_before": messages_before,
            "messages_after": messages_after,
        }

    # 1.46 get_anchored_view — 锚点 + 头尾(3 段)
    # 在 get_messages_around 之上:不光给中间,还拼上头尾的 bookend
    # 用途:长 session 搜索时 — "开头 N 条 + 命中周围 + 结尾 N 条"
    # 3 slice 不重叠,UI 可以清晰分块展示
    def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
        """Return an anchored window plus session bookends.

        Built on top of ``get_messages_around``. Three slices:

          - ``window``: messages immediately surrounding the anchor. Filtered
            to ``keep_roles`` (tool-response noise dropped by default), EXCEPT
            the anchor itself is always preserved regardless of role.
          - ``bookend_start``: first ``bookend`` user/assistant messages of the
            session — but only those whose id is strictly before the window's
            first message id. Empty when the window already overlaps the
            session head. Empty-content messages (tool-call-only assistant
            turns) are skipped so they don't crowd out actual prose openings.
          - ``bookend_end``: last ``bookend`` user/assistant messages of the
            session, same non-overlap rule at the tail.

        Bookends let an FTS5 hit anywhere in a long session yield the goal
        (opening) and the resolution (closing) on a single call — without
        loading the whole transcript.

        Returns ``{"window": [], "messages_before": 0, "messages_after": 0,
        "bookend_start": [], "bookend_end": []}`` when the anchor isn't in
        the session.

        ``keep_roles=None`` disables role filtering (raw window + raw
        bookends).
        """
        if bookend < 0:
            bookend = 0

        # Reuse the primitive — handles anchor-existence, content decoding,
        # tool_calls deserialisation, and boundary counts.
        primitive = self.get_messages_around(
            session_id, around_message_id, window=window
        )
        window_rows = primitive["window"]
        if not window_rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }

        # Apply role filter to the window, but never drop the anchor itself.
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [
                m for m in window_rows
                if m.get("id") == around_message_id or m.get("role") in keep_set
            ]
        else:
            filtered_window = window_rows

        window_min_id = window_rows[0]["id"]
        window_max_id = window_rows[-1]["id"]

        # Fetch bookends only when there's room outside the window. SQL filters
        # by id range, role, and non-empty content — tool-call-only assistant
        # turns (content='' with tool_calls populated) are excluded so they
        # don't crowd out actual prose openings/closings.
        bookend_start_rows: List[Any] = []
        bookend_end_rows: List[Any] = []
        if bookend > 0:
            with self._lock:
                role_clause = ""
                role_params: list = []
                if keep_roles is not None:
                    role_placeholders = ",".join("?" for _ in keep_roles)
                    role_clause = f" AND role IN ({role_placeholders})"
                    role_params = list(keep_roles)

                bookend_start_rows = self._conn.execute(
                    f"SELECT * FROM messages "
                    f"WHERE session_id = ? AND id < ?{role_clause} "
                    f"AND length(content) > 0 "
                    f"ORDER BY id ASC LIMIT ?",
                    (session_id, window_min_id, *role_params, bookend),
                ).fetchall()

                bookend_end_rows = self._conn.execute(
                    f"SELECT * FROM messages "
                    f"WHERE session_id = ? AND id > ?{role_clause} "
                    f"AND length(content) > 0 "
                    f"ORDER BY id DESC LIMIT ?",
                    (session_id, window_max_id, *role_params, bookend),
                ).fetchall()
                # End rows came back DESC for the LIMIT cap; flip to ASC.
                bookend_end_rows = list(reversed(bookend_end_rows))

        def _hydrate(row) -> Dict[str, Any]:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in get_anchored_view, falling back to []"
                    )
                    msg["tool_calls"] = []
            return msg

        return {
            "window": filtered_window,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": [_hydrate(r) for r in bookend_start_rows],
            "bookend_end": [_hydrate(r) for r in bookend_end_rows],
        }

    # 1.55 resolve_resume_session_id — /resume 时把 session_id 重定向
    # 场景:用户给老 session_id 跑 /resume
    # 老 session 可能是压缩链中间节点,没 messages 了
    # 沿着 parent_session_id 走到 tip(真正有 messages 那个)
    # 返:能 resume 的 session_id
    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the first descendant in the chain that has at least one message
        row. If the original session already has messages, or no descendant
        has any, the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id

        with self._lock:
            # If this session already has messages, nothing to redirect.
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
            except Exception:
                return session_id
            if row is not None:
                return session_id

            # Walk descendants: at each step, pick the most-recently-started
                # child session; stop once we find one with messages.
            current = session_id
            seen = {current}
            for _ in range(32):
                try:
                    child_row = self._conn.execute(
                        "SELECT id FROM sessions "
                        "WHERE parent_session_id = ? "
                        "ORDER BY started_at DESC, id DESC LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if child_row is None:
                    return session_id
                child_id = child_row["id"] if hasattr(child_row, "keys") else child_row[0]
                if not child_id or child_id in seen:
                    return session_id
                seen.add(child_id)
                try:
                    msg_row = self._conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                        (child_id,),
                    ).fetchone()
                except Exception:
                    return session_id
                if msg_row is not None:
                    return child_id
                current = child_id
        return session_id

    # 1.47 get_messages_as_conversation — 转 OpenAI 对话格式
    # messages 表里存的是带 metadata 的 row,要喂 LLM 必须转成 OpenAI 格式
    # {role: "user"/"assistant"/"tool", content: ..., tool_call_id / tool_calls: ...}
    # 关键:要决定每条 message 放哪个 key(LLM 看的格式跟 DB 存的不一样)
    def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted (rewound) rows
        as well. See :meth:`rewind_to_message`.
        """
        session_ids = [session_id]
        if include_ancestors:
            session_ids = self._session_lineage_root_to_tip(session_id)

        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            rows = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, "
                "finish_reason, reasoning, reasoning_content, reasoning_details, "
                "codex_reasoning_items, codex_message_items, platform_message_id, observed "
                f"FROM messages WHERE session_id IN ({placeholders})"
                f"{active_clause} ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        messages = []
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Surface the platform-side message id (e.g. yuanbao msg_id,
            # telegram update_id) so platform-specific flows like recall
            # can match by external identifier instead of having to fall
            # back to content-match heuristics.  Exposed as ``message_id``
            # for backward compatibility with the JSONL transcript shape.
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["finish_reason"]:
                    msg["finish_reason"] = row["finish_reason"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        return messages

    # 1.56 _session_lineage_root_to_tip — 沿 parent 链走到 tip
    # 跟 get_compression_tip 区别:返**整条**链,不只 tip
    # 用途:回放历史要把所有 ancestor 都加载进来
    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]

        chain = []
        current = session_id
        seen = set()
        with self._lock:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = self._conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                current = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        return list(reversed(chain)) or [session_id]

    @staticmethod
    # 1.65 _is_duplicate_replayed_user_message — 检测重复回放
    # 场景:压缩链回放时,parent 的最后一条 user message 可能跟 child 第一条一样
    # → 重复显示,LMM 困惑
    # 比较最后一条 messages[-1] 跟当前 msg 的 content
    # 命中 → 跳过(去重)
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get("role") == "user" and prev.get("content") == content:
                return True
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return False
        return False

    # =========================================================================
    # Rewind (soft-delete) — see /rewind slash command + issue #21910
    # =========================================================================

    # 1.57 rewind_to_message — 软删某条之后所有 messages
    # 用途:/rewind N 命令
    # 实现:UPDATE messages SET active=0 WHERE id >= anchor(不真删)
    # 软删好处:
    #   * 还能 /restore_rewound 撤销
    #   * DB row 还在,FTS 索引能搜到(可选)
    #   * 不会破坏 message_id 顺序
    # bump rewind_count 计数
    def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        """Soft-delete all messages with id >= ``target_message_id`` in *session_id*.

        The target message itself becomes inactive as well so the caller
        can pre-fill it as the next user prompt without it appearing
        twice in the replayed transcript.  Rewound rows are kept on
        disk with ``active=0`` for audit / forensic inspection — use
        :meth:`get_messages` with ``include_inactive=True`` to see them.

        Returns a dict::

            {
                "rewound_count": int,    # number of rows newly flipped to active=0
                "target_message": dict,  # full row dict of the target
                "new_head_id":   int|None  # id of the last still-active row, or None
            }

        Raises ``ValueError`` if the target message does not exist in
        *session_id* or if its role is not ``"user"``.

        Always increments ``sessions.rewind_count`` — even when the
        target is already inactive — so the counter accurately reflects
        the number of rewind operations performed against the session.
        Idempotent on the ``active`` flag: re-rewinding past the same
        target is a no-op on row state but still bumps the counter.
        """

        # 1) Validate target up-front (read-only, outside the write txn).
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"message {target_message_id} not found in session {session_id}"
            )
        target_row = dict(row)
        if target_row.get("role") != "user":
            raise ValueError(
                f"rewind target must be a 'user' message (got role="
                f"{target_row.get('role')!r}, id={target_message_id})"
            )

        # Decode content for callers (prefill the prompt buffer).
        target_row["content"] = self._decode_content(target_row.get("content"))

        rewound: List[int] = []

        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            return ids

        rewound = self._execute_write(_do)

        # 2) Compute new head id (largest still-active row id in session).
        with self._lock:
            head_row = self._conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
        new_head_id = head_row[0] if head_row and head_row[0] is not None else None

        return {
            "rewound_count": len(rewound),
            "target_message": target_row,
            "new_head_id": new_head_id,
        }

    # 1.58 restore_rewound — 撤销 rewind
    # 把 active=0 且 id >= since_message_id 的 messages 翻回 active=1
    # 跟 rewind_to_message 配对
    def restore_rewound(self, session_id: str, since_message_id: int) -> int:
        """Mark inactive messages with id >= *since_message_id* active again.

        Returns the number of rows flipped back to ``active=1``.
        Intended for undo-of-rewind and test cleanup; not wired to a
        slash command in v1.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 0",
                (session_id, since_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 1 WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids)

        return self._execute_write(_do)

    # 1.59 list_recent_user_messages — 列最近 N 条 user message
    # 用途:/rewind 选单(让用户选"回到哪条")
    # 返:(message_id, 短预览)
    # 软删的(active=0)不返
    def list_recent_user_messages(
        self,
        session_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the *limit* most-recent user messages, newest first.

        Each entry is a dict with keys ``id``, ``timestamp``, ``preview``.
        ``preview`` is the first 80 characters of the message content
        (with line breaks collapsed to spaces). Used by the /rewind
        slash command picker.

        By default only active messages are returned.
        """
        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, timestamp, content FROM messages "
                "WHERE session_id = ? AND role = 'user'"
                f"{active_clause} "
                "ORDER BY id DESC LIMIT ?",
                (session_id, int(limit)),
            )
            rows = cursor.fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_content(row["content"])
            if isinstance(decoded, list):
                # Multimodal — flatten text parts.
                text_parts = [
                    p.get("text", "") for p in decoded
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                preview = " ".join(t for t in text_parts if t).strip()
                if not preview:
                    preview = "[multimodal content]"
            elif isinstance(decoded, str):
                preview = decoded
            else:
                preview = ""
            preview = " ".join(preview.split())  # collapse whitespace
            if len(preview) > 80:
                preview = preview[:77] + "..."
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "preview": preview,
                }
            )
        return result

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    # 1.66 _sanitize_fts5_query — FTS5 query 防注入
    # FTS5 MATCH 语法有特殊字符:"OR" "AND" "NEAR" "*" ".." 等
    # 用户输入直接拼 → 可能解析失败或被注入
    # 策略:
    #   1. 双引号包住的短语 → 原样
    #   2. 含 . 或 - 的 token → 双引号包
    #   3. 其它 → 当单 term
    # 配 _is_cjk_codepoint 联合做中文判断
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders.
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

        # Step 2: Strip remaining (unmatched) FTS5-special characters
        sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted, hyphenated and underscored
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()


    @staticmethod
    # 1.67 _is_cjk_codepoint — 判断 1 个 Unicode 码点是不是 CJK 家族
    # 覆盖:
    #   * 0x4E00-0x9FFF  CJK 统一汉字
    #   * 0x3400-0x4DBF  CJK 扩展 A
    #   * 0x20000-0x2A6DF CJK 扩展 B
    #   * 0x3000-0x303F  CJK 符号
    #   * 0x3040-0x309F  平假名
    #   * 0x30A0-0x30FF  片假名
    #   * 0xAC00-0xD7AF  韩文音节
    def _is_cjk_codepoint(cp: int) -> bool:
        return (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF)      # Hangul Syllables

    @staticmethod
    # 1.68 _contains_cjk — text 里有 CJK 字符?
    # 简化:遍历 codepoint,任一命中 _is_cjk_codepoint 就返 True
    def _contains_cjk(text: str) -> bool:
        """Check if text contains CJK (Chinese, Japanese, Korean) characters."""
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF):     # Hangul Syllables
                return True
        return False

    @classmethod
    # 1.69 _count_cjk — 数 CJK 字符数
    # 用 sum(1 for c in text if _is_cjk_codepoint(ord(c)))
    # 跟 len(text) 不同 — 只算 CJK,不算 ASCII
    def _count_cjk(cls, text: str) -> int:
        """Count CJK characters in text."""
        return sum(1 for ch in text if cls._is_cjk_codepoint(ord(ch)))

    # 1.48 search_messages — FTS5 + CJK 回退全文搜索
    # 主路径:FTS5 MATCH(英文/拉丁文 — 倒排索引,BM25 排序)
    # CJK 回退:中文/日文/韩文 — FTS5 的 default tokenizer 不分词
    #   改用 messages_fts_trigram(3 字符 trigram 索引)
    #   ≥3 字符用 trigram,<3 字符用 LIKE
    # 配合 _sanitize_fts5_query 转义特殊字符防注入
    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).

        ``sort`` controls temporal ordering:
          - ``None`` (default): FTS5 BM25 relevance only. Time-neutral.
          - ``"newest"``: order by message timestamp DESC, then by rank.
          - ``"oldest"``: order by message timestamp ASC, then by rank.

        The short-CJK LIKE fallback already orders by timestamp DESC and
        ignores ``sort``. The trigram CJK path honours ``sort`` like the main
        FTS5 path.

        Rewound (``active=0``) rows are excluded by default. Pass
        ``include_inactive=True`` to search every row.
        """
        if not self._fts_enabled:
            return []

        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        # Normalise sort. Anything not in the allowed set falls back to None
        # (FTS5 rank-only) so callers can pass through user input without
        # validation.
        if isinstance(sort, str):
            sort_norm = sort.strip().lower()
            if sort_norm not in ("newest", "oldest"):
                sort_norm = None
        else:
            sort_norm = None

        # ORDER BY shared across the main FTS5 path and trigram CJK path.
        # With sort set, timestamp is primary and rank is the tiebreaker.
        if sort_norm == "newest":
            order_by_sql = "ORDER BY m.timestamp DESC, rank"
        elif sort_norm == "oldest":
            order_by_sql = "ORDER BY m.timestamp ASC, rank"
        else:
            order_by_sql = "ORDER BY rank"

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]
        if not include_inactive:
            where_clauses.append("m.active = 1")

        if source_filter is not None:
            source_placeholders = ",".join("?" for _ in source_filter)
            where_clauses.append(f"s.source IN ({source_placeholders})")
            params.extend(source_filter)

        if exclude_sources is not None:
            exclude_placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({exclude_placeholders})")
            params.extend(exclude_sources)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """

        # CJK queries bypass the unicode61 FTS5 table.  The default tokenizer
        # splits CJK characters into individual tokens, so "大别山项目" becomes
        # "大 AND 别 AND 山 AND 项 AND 目" — producing false positives and
        # missing exact phrase matches.
        #
        # For queries with 3+ CJK characters, we use the trigram FTS5 table
        # (indexed substring matching with ranking and snippets).  For shorter
        # CJK queries (1-2 chars), trigram can't match (it needs ≥9 UTF-8
        # bytes = 3 CJK chars), so we fall back to LIKE.
        is_cjk = self._contains_cjk(query)
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_count = self._count_cjk(raw_query)

            # Per-token CJK length check (#20494): trigram needs >=3 CJK chars
            # per token. A query like "广西 OR 桂林 OR 漓江" has cjk_count=6
            # (>=3) but each individual token is only 2 chars — trigram returns 0.
            # Route to LIKE when any non-operator CJK token is <3 CJK chars.
            _tokens_for_check = [
                t for t in raw_query.split()
                if t.upper() not in {"AND", "OR", "NOT"} and self._contains_cjk(t)
            ]
            _any_short_cjk = any(
                self._count_cjk(t) < 3 for t in _tokens_for_check
            )

            if cjk_count >= 3 and not _any_short_cjk:
                # Trigram FTS5 path — quote each non-operator token to handle
                # FTS5 special chars (%, *, etc.) while preserving boolean
                # operators (AND, OR, NOT) for multi-term queries.
                tokens = raw_query.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in {"AND", "OR", "NOT"}:
                        parts.append(tok)
                    else:
                        parts.append('"' + tok.replace('"', '""') + '"')
                trigram_query = " ".join(parts)
                tri_where = ["messages_fts_trigram MATCH ?"]
                tri_params: list = [trigram_query]
                if not include_inactive:
                    tri_where.append("m.active = 1")
                if source_filter is not None:
                    tri_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    tri_params.extend(source_filter)
                if exclude_sources is not None:
                    tri_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    tri_params.extend(exclude_sources)
                if role_filter:
                    tri_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    tri_params.extend(role_filter)
                tri_sql = f"""
                    SELECT
                        m.id,
                        m.session_id,
                        m.role,
                        snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet,
                        m.content,
                        m.timestamp,
                        m.tool_name,
                        s.source,
                        s.model,
                        s.started_at AS session_started
                    FROM messages_fts_trigram
                    JOIN messages m ON m.id = messages_fts_trigram.rowid
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(tri_where)}
                    {order_by_sql}
                    LIMIT ? OFFSET ?
                """
                tri_params.extend([limit, offset])
                with self._lock:
                    try:
                        tri_cursor = self._conn.execute(tri_sql, tri_params)
                    except sqlite3.OperationalError:
                        matches = []
                    else:
                        matches = [dict(row) for row in tri_cursor.fetchall()]
            else:
                # Short / mixed CJK query: trigram cannot match tokens with
                # <3 CJK chars. Fall back to LIKE substring search.
                # For multi-token OR queries (e.g. "广西 OR 桂林 OR 漓江"),
                # build one LIKE condition per non-operator token so each term
                # is matched independently (#20494).
                non_op_tokens = [
                    t for t in raw_query.split()
                    if t.upper() not in {"AND", "OR", "NOT"}
                ] or [raw_query]
                token_clauses = []
                like_params: list = []
                for tok in non_op_tokens:
                    esc = tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    token_clauses.append(
                        "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"
                    )
                    like_params += [f"%{esc}%", f"%{esc}%", f"%{esc}%"]
                like_where = [f"({' OR '.join(token_clauses)})"]
                if source_filter is not None:
                    like_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    like_params.extend(role_filter)
                like_sql = f"""
                    SELECT m.id, m.session_id, m.role,
                           substr(m.content,
                                  max(1, instr(m.content, ?) - 40),
                                  120) AS snippet,
                           m.content, m.timestamp, m.tool_name,
                           s.source, s.model, s.started_at AS session_started
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY m.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                like_params.extend([limit, offset])
                # instr() for snippet uses first search token
                like_params = [non_op_tokens[0]] + like_params
                with self._lock:
                    like_cursor = self._conn.execute(like_sql, like_params)
                    matches = [dict(row) for row in like_cursor.fetchall()]
        else:
            with self._lock:
                try:
                    cursor = self._conn.execute(sql, params)
                except sqlite3.OperationalError:
                    # FTS5 query syntax error despite sanitization — return empty
                    return []
                else:
                    matches = [dict(row) for row in cursor.fetchall()]

        # Add surrounding context (1 message before + after each match).
        # Done outside the lock so we don't hold it across N sequential queries.
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute(
                        """WITH target AS (
                               SELECT session_id, timestamp, id
                               FROM messages
                               WHERE id = ?
                           )
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp < t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id < t.id)
                               ORDER BY m.timestamp DESC, m.id DESC
                               LIMIT 1
                           )
                           UNION ALL
                           SELECT role, content
                           FROM messages
                           WHERE id = ?
                           UNION ALL
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp > t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id > t.id)
                               ORDER BY m.timestamp ASC, m.id ASC
                               LIMIT 1
                           )""",
                        (match["id"], match["id"]),
                    )
                    context_msgs = []
                    for r in ctx_cursor.fetchall():
                        raw = r["content"]
                        decoded = self._decode_content(raw)
                        # Multimodal context: render a compact text-only
                        # summary for search previews.
                        if isinstance(decoded, list):
                            text_parts = [
                                p.get("text", "") for p in decoded
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            text = " ".join(t for t in text_parts if t).strip()
                            preview = text or "[multimodal content]"
                        elif isinstance(decoded, str):
                            preview = decoded
                        else:
                            preview = ""
                        context_msgs.append(
                            {"role": r["role"], "content": preview[:200]}
                        )
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        # Remove full content from result (snippet is enough, saves tokens)
        for match in matches:
            match.pop("content", None)

        return matches

    # 1.49 search_sessions — session 级别搜索
    # 跟 search_messages 区别:搜 session 不是 message
    # 计算 last_active 排序(最近活跃优先)
    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column (latest
        message timestamp for the session, falling back to ``started_at``),
        ordered by most-recently-used first.
        """
        select_with_last_active = (
            "SELECT s.*, COALESCE(m.last_active, s.started_at) AS last_active "
            "FROM sessions s "
            "LEFT JOIN ("
            "SELECT session_id, MAX(timestamp) AS last_active "
            "FROM messages GROUP BY session_id"
            ") m ON m.session_id = s.id "
        )
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "WHERE s.source = ? "
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (source, limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    # 1.50 session_count — 计数 session(可按 source 过滤)
    # 跟 SELECT COUNT(*) 的区别:可加 min_message_count 阈值
    # (过滤"空 session"做统计时用)
    def session_count(self, source: str = None, min_message_count: int = 0) -> int:
        """Count sessions, optionally filtered by source."""
        where_clauses = []
        params = []

        if source:
            where_clauses.append("source = ?")
            params.append(source)
        if min_message_count > 0:
            where_clauses.append("message_count >= ?")
            params.append(min_message_count)

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM sessions{where_sql}", params)
            return cursor.fetchone()[0]

    # 1.51 message_count — 计数 messages(全局或按 session)
    # session_id 给的话:只数该 session 的
    # 不给:数整个表(全局总数)
    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    # 1.52 export_session — 导 1 个 session 全部数据(dict)
    # 包含 session row + 所有 messages
    # 用途:备份、调试、用户导出对话
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    # 1.53 export_all — 导所有 session(可按 source 过滤)
    # 返 list[dict],每个 dict = 1 个 session + 它所有 messages
    # 用途:JSONL 备份、跨机器迁移
    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    # 1.54 clear_messages — 硬删所有 messages + 重置计数
    # 跟 soft-delete (rewind) 区别:这是物理 DELETE FROM
    # 用途:/clear 命令、清空对话但保留 session 行
    # 同时重置 message_count / tool_call_count = 0
    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    @staticmethod
    # 1.60 _remove_session_files — 清磁盘上的 session 文件
    # DB 之外,sessions/ 目录可能还有 .json / .jsonl / request_dump_* 文件
    # 删除 session 时**也要**清这些(防孤儿文件)
    # best-effort:失败不抛(不是关键路径)
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session.

        Cleans up ``{session_id}.json``, ``{session_id}.jsonl``, and any
        ``request_dump_{session_id}_*.json`` files left by the gateway.
        Silently skips files that don't exist and swallows OSError so a
        filesystem hiccup never blocks a DB operation.
        """
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            p = sessions_dir / f"{session_id}{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # request_dump files use session_id as a prefix component
        try:
            for p in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    # 1.61 delete_session — 删 1 个 session + 它的 messages
    # 重要:不级联删 child sessions — 设 parent_session_id = NULL
    # (orphan children,稍后 prune_sessions 清)
    # 还要调 _remove_session_files 清磁盘文件
    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Child sessions are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted, so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for the deleted
        session. Returns True if the session was found and deleted.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            # Orphan child sessions so FK constraint is satisfied
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return deleted

    # 1.62 prune_sessions — 批量删过期 session
    # 策略:删 ended_at > N 天的 session
    # 同样 orphan 它的 children + 清磁盘文件
    # 用途:maybe_auto_prune_and_vacuum(启动时跑)
    def prune_sessions(
        self,
        older_than_days: int = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete sessions older than N days. Returns count of deleted sessions.

        Only prunes ended sessions (not active ones).  Child sessions outside
        the prune window are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.  When *sessions_dir* is provided, also removes
        on-disk transcript files (``.json`` / ``.jsonl`` /
        ``request_dump_*``) for every pruned session, outside the DB
        transaction.
        """
        cutoff = time.time() - (older_than_days * 86400)
        removed_ids: list[str] = []

        def _do(conn):
            if source:
                cursor = conn.execute(
                    """SELECT id FROM sessions
                       WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?""",
                    (cutoff, source),
                )
            else:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                    (cutoff,),
                )
            session_ids = {row["id"] for row in cursor.fetchall()}

            if not session_ids:
                return 0

            # Orphan any sessions whose parent is about to be deleted
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            return len(session_ids)

        count = self._execute_write(_do)
        # Clean up on-disk files outside the DB transaction
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    # ── Meta key/value (for scheduler bookkeeping) ──

    # 1.39 get_meta — 从 state_meta 读 KV
    # 用途:scheduler / watchdog 存小数据(如 "last_kanban_run_ts": "...")
    # 返字符串值 / None
    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row["value"] if isinstance(row, sqlite3.Row) else row[0]

    # 1.40 set_meta — 写 state_meta KV(upsert)
    # INSERT ... ON CONFLICT DO UPDATE:存在则覆盖,不存在则新建
    # 走 _execute_write 走事务(防并发覆盖)
    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the state_meta key/value store."""
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._execute_write(_do)

    # 1.70 apply_telegram_topic_migration — Telegram DM topic 模式迁移
    # 用途:Telegram DM 用 topic 区分对话时,要建专门的 binding 表
    # + 加外键 ON DELETE CASCADE
    # 只跑 1 次,版本门控(idempotent)
    def apply_telegram_topic_migration(self) -> None:
        """Create Telegram DM topic-mode tables on explicit /topic opt-in.

        This migration is deliberately not part of automatic SessionDB startup
        reconciliation. Operators must be able to upgrade Hermes, keep the old
        Telegram bot behavior running, and only mutate topic-mode state when the
        user executes /topic to opt into the feature.

        Schema versions:
          v1 — initial shape (no ON DELETE CASCADE on session_id FK)
          v2 — session_id FK gets ON DELETE CASCADE so session pruning
               automatically clears bindings.
        """
        def _do(conn):
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT
                );

                CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, thread_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session
                ON telegram_dm_topic_bindings(session_id);

                CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user
                ON telegram_dm_topic_bindings(user_id, chat_id);
                """
            )

            # v1 → v2: rebuild telegram_dm_topic_bindings if its session_id FK
            # lacks ON DELETE CASCADE. SQLite can't ALTER a foreign key, so we
            # rebuild the table. Only runs once per DB (version gate).
            current = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                ("telegram_dm_topic_schema_version",),
            ).fetchone()
            current_version = int(current[0]) if current and str(current[0]).isdigit() else 0
            if current_version < 2:
                fk_rows = conn.execute(
                    "PRAGMA foreign_key_list('telegram_dm_topic_bindings')"
                ).fetchall()
                needs_rebuild = any(
                    row[2] == "sessions" and (row[6] or "") != "CASCADE"
                    for row in fk_rows
                )
                if needs_rebuild:
                    conn.executescript(
                        """
                        CREATE TABLE telegram_dm_topic_bindings_new (
                            chat_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            session_key TEXT NOT NULL,
                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                            managed_mode TEXT NOT NULL DEFAULT 'auto',
                            linked_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (chat_id, thread_id)
                        );
                        INSERT INTO telegram_dm_topic_bindings_new
                            SELECT chat_id, thread_id, user_id, session_key,
                                   session_id, managed_mode, linked_at, updated_at
                            FROM telegram_dm_topic_bindings;
                        DROP TABLE telegram_dm_topic_bindings;
                        ALTER TABLE telegram_dm_topic_bindings_new
                            RENAME TO telegram_dm_topic_bindings;
                        CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session
                            ON telegram_dm_topic_bindings(session_id);
                        CREATE INDEX idx_telegram_dm_topic_bindings_user
                            ON telegram_dm_topic_bindings(user_id, chat_id);
                        """
                    )

            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("telegram_dm_topic_schema_version", "2"),
            )
        self._execute_write(_do)

    # 1.80 enable_telegram_topic_mode — 启用 topic 模式(给某个 chat/user)
    # 持久化 capability flag
    def enable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        user_id: str,
        has_topics_enabled: Optional[bool] = None,
        allows_users_to_create_topics: Optional[bool] = None,
    ) -> None:
        """Enable Telegram DM topic mode for one private chat/user.

        This method intentionally owns the explicit topic migration. Ordinary
        SessionDB startup must not create these side tables.
        """
        self.apply_telegram_topic_migration()
        now = time.time()

        def _to_int(value: Optional[bool]) -> Optional[int]:
            if value is None:
                return None
            return 1 if value else 0

        def _do(conn):
            conn.execute(
                """
                INSERT INTO telegram_dm_topic_mode (
                    chat_id, user_id, enabled, activated_at, updated_at,
                    has_topics_enabled, allows_users_to_create_topics,
                    capability_checked_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    enabled = 1,
                    updated_at = excluded.updated_at,
                    has_topics_enabled = excluded.has_topics_enabled,
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,
                    capability_checked_at = excluded.capability_checked_at
                """,
                (
                    str(chat_id),
                    str(user_id),
                    now,
                    now,
                    _to_int(has_topics_enabled),
                    _to_int(allows_users_to_create_topics),
                    now,
                ),
            )
        self._execute_write(_do)

    # 1.81 disable_telegram_topic_mode — 关闭 topic 模式
    # 可选:clear_bindings=True 时,把所有 (chat_id, thread_id) 绑定清掉
    def disable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        clear_bindings: bool = True,
    ) -> None:
        """Disable Telegram DM topic mode for one private chat.

        When ``clear_bindings`` is True (default) the (chat_id, thread_id)
        bindings for this chat are also cleared so re-enabling later
        starts from a clean slate. Set to False if the operator wants to
        preserve bindings for a later re-enable.

        Never creates the topic-mode tables from scratch; if they don't
        exist there is nothing to disable and the call is a no-op.
        """
        def _do(conn):
            try:
                conn.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? "
                    "WHERE chat_id = ?",
                    (time.time(), str(chat_id)),
                )
                if clear_bindings:
                    conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE chat_id = ?",
                        (str(chat_id),),
                    )
            except sqlite3.OperationalError:
                # Tables don't exist yet — nothing to disable.
                return
        self._execute_write(_do)

    # 1.82 is_telegram_topic_mode_enabled — 查某 chat/user topic 模式开了?
    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str) -> bool:
        """Return whether Telegram DM topic mode is enabled for this chat/user."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT enabled FROM telegram_dm_topic_mode
                    WHERE chat_id = ? AND user_id = ?
                    """,
                    (str(chat_id), str(user_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        if row is None:
            return False
        enabled = row["enabled"] if isinstance(row, sqlite3.Row) else row[0]
        return bool(enabled)

    # 1.83 get_telegram_topic_binding — 按 (chat_id, thread_id) 查绑定
    def get_telegram_topic_binding(
        self,
        *,
        chat_id: str,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the session binding for a Telegram DM topic, if present."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (str(chat_id), str(thread_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    # 1.84 list_telegram_topic_bindings_for_chat — 列某 chat 全部 binding
    # 按时间倒序(最新在前)
    def list_telegram_topic_bindings_for_chat(
        self,
        *,
        chat_id: str,
    ) -> List[Dict[str, Any]]:
        """All Telegram DM topic bindings for one chat, newest first.

        Read-only; returns [] if the bindings table doesn't exist yet
        (does not trigger the topic-mode migration).
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM telegram_dm_topic_bindings "
                    "WHERE chat_id = ? ORDER BY updated_at DESC",
                    (str(chat_id),),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    # 1.85 get_telegram_topic_binding_by_session — 反向查(sessions → topic)
    # 用 UNIQUE index 加速(一个 session 只能绑一个 topic)
    def get_telegram_topic_binding_by_session(
        self,
        *,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the Telegram DM topic binding for a given session_id, if present.

        Uses the UNIQUE INDEX on telegram_dm_topic_bindings(session_id) for an
        efficient reverse lookup. Returns None when the session has no binding or
        the table does not exist yet.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    # 1.86 bind_telegram_topic — 绑 1 个 session 到 1 个 Telegram topic
    # 拒绝"双绑":同一 session 只能绑一个 topic(UNIQUE 约束)
    def bind_telegram_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        user_id: str,
        session_key: str,
        session_id: str,
        managed_mode: str = "auto",
    ) -> None:
        """Bind one Telegram DM topic thread to one Hermes session.

        A Hermes session may only be linked to one Telegram topic in MVP.
        Rebinding the same topic to the same session is idempotent; trying to
        link the same session to a different topic raises ValueError.
        """
        self.apply_telegram_topic_migration()
        now = time.time()
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        user_id = str(user_id)
        session_key = str(session_key)
        session_id = str(session_id)

        def _do(conn):
            existing_session = conn.execute(
                """
                SELECT chat_id, thread_id FROM telegram_dm_topic_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if existing_session is not None:
                linked_chat = existing_session["chat_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[0]
                linked_thread = existing_session["thread_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[1]
                if str(linked_chat) != chat_id or str(linked_thread) != thread_id:
                    raise ValueError("session is already linked to another Telegram topic")

            conn.execute(
                """
                INSERT INTO telegram_dm_topic_bindings (
                    chat_id, thread_id, user_id, session_key, session_id,
                    managed_mode, linked_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_key = excluded.session_key,
                    session_id = excluded.session_id,
                    managed_mode = excluded.managed_mode,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    thread_id,
                    user_id,
                    session_key,
                    session_id,
                    managed_mode,
                    now,
                    now,
                ),
            )
        self._execute_write(_do)

    # 1.87 is_telegram_session_linked_to_topic — session 已绑 topic?
    # 只读,不做迁移
    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        """Return True if a Hermes session is already bound to any Telegram DM topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables have not been created yet (i.e. nobody has run
        ``/topic`` in this profile), the session is by definition unbound
        and we return False.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    LIMIT 1
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return row is not None

    # 1.88 list_unlinked_telegram_sessions_for_user — 列未绑 topic 的老 session
    # 用途:用户开了 topic 模式后,帮它把老 session 选一个绑
    def list_unlinked_telegram_sessions_for_user(
        self,
        *,
        chat_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List previous Telegram sessions for this user that are not bound to a topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables are absent, fall back to a simpler query that
        just returns this user's Telegram sessions — there can't be any
        bindings yet.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.*,
                        COALESCE(
                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_dm_topic_bindings b
                          WHERE b.session_id = s.id
                      )
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                # telegram_dm_topic_bindings doesn't exist yet — no bindings
                # means every telegram session for this user is "unlinked".
                rows = self._conn.execute(
                    """
                    SELECT s.*,
                        COALESCE(
                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            session = dict(row)
            raw = str(session.pop("_preview_raw", "") or "").strip()
            session["preview"] = raw[:60] + ("..." if len(raw) > 60 else "") if raw else ""
            sessions.append(session)
        return sessions

    # ── Space reclamation ──

    # FTS5 virtual tables whose b-tree segments we merge on optimize. The
    # trigram table is created lazily / may be disabled, so we probe before
    # touching it (see optimize_fts).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram")

    def _fts_table_exists(self, name: str) -> bool:
        """True if an FTS5 virtual table is queryable in this DB."""
        try:
            self._conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    # 1.71 optimize_fts — FTS5 b-tree segment 合并
    # FTS5 写多了会产生很多小 segment → 搜索变慢
    # INSERT INTO messages_fts(messages_fts, 'optimize') 触发合并
    # 返修了几个 FTS index(应该是 1)
    def optimize_fts(self) -> int:
        """Merge fragmented FTS5 b-tree segments into one per index.

        FTS5 indexes grow as a series of incremental segments — one per
        ``INSERT`` batch driven by the message triggers. Over tens of
        thousands of messages these segments accumulate, which both bloats
        the ``*_data`` shadow tables and slows ``MATCH`` queries that must
        scan every segment. The special ``'optimize'`` command rewrites each
        index as a single merged segment.

        This is purely a maintenance operation — it changes neither search
        results nor ``snippet()`` output, only on-disk layout and query
        speed. It is complementary to VACUUM: ``optimize`` compacts the FTS
        index internally, then VACUUM returns the freed pages to the OS.

        Skips any FTS table that does not exist (e.g. the trigram index when
        disabled via ``HERMES_DISABLE_FTS_TRIGRAM`` or not yet created), so
        it is safe to call unconditionally.

        Returns the number of FTS indexes that were optimized.
        """
        optimized = 0
        with self._lock:
            for tbl in self._FTS_TABLES:
                if not self._fts_table_exists(tbl):
                    continue
                try:
                    # The column name in the INSERT must match the table name
                    # for FTS5 special commands.
                    self._conn.execute(
                        f"INSERT INTO {tbl}({tbl}) VALUES('optimize')"
                    )
                    optimized += 1
                except sqlite3.OperationalError as exc:
                    logger.warning(
                        "FTS optimize failed for %s: %s", tbl, exc
                    )
        return optimized

    # 1.72 vacuum — 回收 SQLite 文件空间
    # 大量 DELETE 后,DB 文件不会自动缩(VACUUM 才会)
    # 慢、占 I/O,不该频繁跑
    # 配合 optimize_fts 用 — 清理完碎片再 vacuum
    def vacuum(self) -> int:
        """Run VACUUM to reclaim disk space after large deletes.

        SQLite does not shrink the database file when rows are deleted —
        freed pages just get reused on the next insert. After a prune that
        removed hundreds of sessions, the file stays bloated unless we
        explicitly VACUUM.

        VACUUM rewrites the entire DB, so it's expensive (seconds per
        100MB) and cannot run inside a transaction. It also acquires an
        exclusive lock, so callers must ensure no other writers are
        active. Safe to call at startup before the gateway/CLI starts
        serving traffic.

        FTS5 segments are merged first via :meth:`optimize_fts` so the
        subsequent VACUUM reclaims the pages freed by the merge. This is a
        layout-only optimization — search results are unchanged.

        Returns the number of FTS indexes that were optimized (0 if the
        merge step failed or no FTS tables exist).
        """
        # Merge FTS5 segments before VACUUM so the freed pages are returned
        # to the OS in the same pass. optimize_fts() manages its own lock.
        optimized = 0
        try:
            optimized = self.optimize_fts()
        except Exception as exc:
            logger.warning("FTS optimize before VACUUM failed: %s", exc)
        # VACUUM cannot be executed inside a transaction.
        with self._lock:
            # Best-effort WAL checkpoint first, then VACUUM.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.execute("VACUUM")
        return optimized

    # 1.73 maybe_auto_prune_and_vacuum — 启动时自动维护
    # 跑顺序:
    #   1. prune_empty_ghost_sessions(24h 前的空 TUI)
    #   2. finalize_orphaned_compression_sessions(7d 前的孤儿)
    #   3. 周期到了才 vacuum(防止每次启动都跑)
    # 启动慢的时候可能就是这个跑的
    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
        sessions_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Idempotent auto-maintenance: prune old sessions + optional VACUUM.

        Records the last run timestamp in state_meta so subsequent calls
        within ``min_interval_hours`` no-op. Designed to be called once at
        startup from long-lived entrypoints (CLI, gateway, cron scheduler).

        When *sessions_dir* is provided, on-disk transcript files
        (``.json`` / ``.jsonl`` / ``request_dump_*``) for pruned sessions
        are removed as part of the same sweep (issue #3015).

        Never raises. On any failure, logs a warning and returns a dict
        with ``"error"`` set.

        Returns a dict with keys:
          - ``"skipped"`` (bool) — true if within min_interval_hours of last run
          - ``"pruned"`` (int)   — number of sessions deleted
          - ``"vacuumed"`` (bool) — true if VACUUM ran
          - ``"error"`` (str, optional) — present only on failure
        """
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            # Skip if another process/call did maintenance recently.
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run

            pruned = self.prune_sessions(
                older_than_days=retention_days,
                sessions_dir=sessions_dir,
            )
            result["pruned"] = pruned

            # Only VACUUM if we actually freed rows — VACUUM on a tight DB
            # is wasted I/O. Threshold keeps small DBs from paying the cost.
            if vacuum and pruned > 0:
                try:
                    self.vacuum()
                    result["vacuumed"] = True
                except Exception as exc:
                    logger.warning("state.db VACUUM failed: %s", exc)

            # Record the attempt even if pruned == 0, so we don't retry
            # every startup within the min_interval_hours window.
            self.set_meta("last_auto_prune", str(now))

            if pruned > 0:
                logger.info(
                    "state.db auto-maintenance: pruned %d session(s) older than %d days%s",
                    pruned,
                    retention_days,
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            # Maintenance must never block startup. Log and return error marker.
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)

        return result

    # ── Handoff (cross-platform session transfer) ──────────────────────────
    #
    # State machine:
    #   None       — no handoff in flight
    #   "pending"  — CLI requested handoff, gateway hasn't picked it up yet
    #   "running"  — gateway is processing (session switch + synthetic turn)
    #   "completed"— gateway successfully delivered the synthetic turn
    #   "failed"   — gateway hit an error; reason in handoff_error
    #
    # The CLI writes "pending" then poll-waits for terminal state. The gateway
    # watcher transitions pending→running→{completed,failed}.

    # 1.74 request_handoff — 跨平台交接请求
    # 用途:CLI 的对话要"交接"到 Telegram / Discord 等
    # 状态机:None → pending → running → completed / failed
    # 返 True = 成功标记 pending(其他状态会拒绝)
    def request_handoff(self, session_id: str, platform: str) -> bool:
        """Mark a session as pending handoff to the given platform.

        Returns True if the row was found and not already in flight; False if
        the session is already in a non-terminal handoff state.
        """
        def _do(conn):
            cur = conn.execute(
                "UPDATE sessions "
                "SET handoff_state = 'pending', "
                "    handoff_platform = ?, "
                "    handoff_error = NULL "
                "WHERE id = ? AND (handoff_state IS NULL "
                "                  OR handoff_state IN ('completed', 'failed'))",
                (platform, session_id),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    # 1.75 get_handoff_state — 查交接状态
    # 返:{state, platform, error} / None(没交接)
    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read the current handoff state for a session.

        Returns ``{"state", "platform", "error"}`` or None if the session has
        no handoff record.
        """
        try:
            cur = self._conn.execute(
                "SELECT handoff_state, handoff_platform, handoff_error "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "state": row["handoff_state"],
                "platform": row["handoff_platform"],
                "error": row["handoff_error"],
            }
        except Exception:
            return None

    # 1.76 list_pending_handoffs — gateway watcher 找待处理交接
    # 返所有 state=pending 的,按时间排(最老的先)
    # gateway 周期性扫这个表,自己 claim 然后处理
    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """Return all sessions in handoff_state='pending', oldest first.

        Used by the gateway's handoff watcher.
        """
        try:
            cur = self._conn.execute(
                "SELECT * FROM sessions "
                "WHERE handoff_state = 'pending' "
                "ORDER BY started_at ASC"
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    # 1.77 claim_handoff — gateway 原子认领交接
    # pending → running 转换,原子
    # 返 True = 认领成功(自己来跑)/ False = 别人抢先了
    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""
        def _do(conn):
            cur = conn.execute(
                "UPDATE sessions SET handoff_state = 'running' "
                "WHERE id = ? AND handoff_state = 'pending'",
                (session_id,),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    # 1.78 complete_handoff — 标记交接完成
    # running → completed,清 error
    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET handoff_state = 'completed', "
                "handoff_error = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    # 1.79 fail_handoff — 标记交接失败
    # running → failed,记 error(截断 500 字符)
    def fail_handoff(self, session_id: str, error: str) -> None:
        """Mark a handoff as failed and record the reason."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET handoff_state = 'failed', "
                "handoff_error = ? WHERE id = ?",
                (error[:500], session_id),
            )
        self._execute_write(_do)
