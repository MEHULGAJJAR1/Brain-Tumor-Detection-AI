"""Small SQLite data-access layer using one connection per Flask request."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import current_app, g


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "database" / "schema.sql"


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        database_path = Path(current_app.config["DATABASE_PATH"])
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        g.db = connection
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db(app) -> None:
    """Create the persistent schema if this is a new installation."""
    with app.app_context():
        connection = get_db()
        connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
    init_db(app)
