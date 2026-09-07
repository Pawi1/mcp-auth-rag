"""SQLite connection helper."""

import sqlite3
import threading

from config import DB_TIMEOUT

_local = threading.local()


def connect(path) -> sqlite3.Connection:
    """Return this thread's connection to `path`, opening it on first use.

    Opening a connection costs ~85us, far more than any query this server
    runs, and a fresh connection to a WAL database also has to map the -shm
    index every time. Connections are therefore kept open; callers must not
    close them.

    One connection is held per thread, so worker threads (password hashing,
    DNS) get their own rather than sharing across threads, which sqlite3
    forbids. Switching path closes the previous one, which keeps tests that
    point at a fresh database per case from leaking descriptors.
    """
    key = str(path)
    cached = getattr(_local, "conn", None)

    if cached is not None and _local.path == key:
        if cached.in_transaction:
            cached.rollback()
        cached.row_factory = None
        return cached

    if cached is not None:
        cached.close()
        _local.conn = None
        _local.path = None

    conn = sqlite3.connect(key, timeout=DB_TIMEOUT)
    _local.conn = conn
    _local.path = key
    return conn


def enable_wal(conn: sqlite3.Connection) -> None:
    """WAL is persisted in the database file, so this only has to run at schema setup."""
    conn.execute("PRAGMA journal_mode=WAL")
