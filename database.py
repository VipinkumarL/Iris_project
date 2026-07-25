import sqlite3
import os
from contextlib import closing


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "database.db")


def resolve_db_path(db_path=None):
    db_path = db_path or DEFAULT_DATABASE_PATH
    db_path = os.path.expanduser(str(db_path))

    if os.path.isabs(db_path):
        return db_path

    return os.path.join(PROJECT_ROOT, db_path)


def get_connection(db_path="database.db"):
    connection = sqlite3.connect(resolve_db_path(db_path), timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path="database.db"):
    resolved_path = resolve_db_path(db_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    with closing(get_connection(resolved_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                hash TEXT,
                pattern TEXT,
                signature TEXT
            )
            """
        )

        cur.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cur.fetchall()]

        if "pattern" not in columns:
            cur.execute("ALTER TABLE users ADD COLUMN pattern TEXT")

        if "signature" not in columns:
            cur.execute("ALTER TABLE users ADD COLUMN signature TEXT")

        conn.commit()
