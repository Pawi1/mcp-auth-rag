"""SQLite connection helper."""

import sqlite3

from config import DB_TIMEOUT


def connect(path) -> sqlite3.Connection:
    """Open a connection to `path`. Callers pass their own module-level DB_PATH."""
    return sqlite3.connect(str(path), timeout=DB_TIMEOUT)


def enable_wal(conn: sqlite3.Connection) -> None:
    """WAL is persisted in the database file, so this only has to run at schema setup."""
    conn.execute("PRAGMA journal_mode=WAL")
