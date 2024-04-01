## Adapted from SkyPilot
import contextlib
import functools
import os
import pathlib
import sqlite3
from typing import Callable, Optional, Union

_DB_PATH = os.path.expanduser("~/.sky/skylet_config.db")
os.makedirs(pathlib.Path(_DB_PATH).parents[0], exist_ok=True)

# https://github.com/skypilot-org/skypilot/blob/master/sky/utils/db_utils.py#L4
@contextlib.contextmanager
def safe_cursor(db_path: str):
    """A newly created, auto-committing, auto-closing cursor."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        conn.commit()
        conn.close()


_table_created = False


# https://github.com/skypilot-org/skypilot/blob/master/sky/skylet/configs.py
def ensure_table(func: Callable):
    """Ensure the table exists before calling the function.

    Since this module will be imported whenever `sky` is imported (due to
    Python's package importing logic), we should avoid creating the table
    until it's actually needed to avoid too many concurrent commit to the
    database.
    It solves the database locked problem in #1576.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _table_created
        if not _table_created:
            with safe_cursor(_DB_PATH) as c:  # Call it 'c' to avoid pylint complaining.
                # Use WAL mode to avoid locking problem in #1507.
                # Reference: https://stackoverflow.com/a/39265148
                c.execute("PRAGMA journal_mode=WAL")
                c.execute(
                    """\
                    CREATE TABLE IF NOT EXISTS config (
                        key TEXT PRIMARY KEY,
                        value TEXT)"""
                )
        _table_created = True
        return func(*args, **kwargs)

    return wrapper


@ensure_table
def get_config(key: str) -> Optional[bytes]:
    with safe_cursor(_DB_PATH) as cursor:
        rows = cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        for (value,) in rows:
            return value
        return None


@ensure_table
def set_config(key: str, value: Union[bytes, str]) -> None:
    with safe_cursor(_DB_PATH) as cursor:
        cursor.execute(
            """\
            INSERT OR REPLACE INTO config VALUES (?, ?)
            """,
            (key, value),
        )
