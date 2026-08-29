"""
database.py
------------
Small helper module that owns all direct MySQL access for the backend.

We use plain `mysql-connector-python` with hand-written SQL rather than a
full ORM (like SQLAlchemy models) because there's only one table and a
couple of simple queries - a lightweight connection helper keeps the
project easy to follow for learning purposes. If the project grows,
swapping this out for SQLAlchemy later would be straightforward.
"""

from typing import Any

import mysql.connector
from mysql.connector import Error as MySQLError

from config import settings


def get_connection():
    """Open a new connection to the MySQL database.

    A fresh connection is opened per request rather than kept as a single
    long-lived global connection - this is the simplest way to avoid
    "MySQL server has gone away" errors when the backend is idle for a
    while, and FastAPI's request/response cycle makes it cheap to do.
    """
    return mysql.connector.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
    )


def save_prediction(record: dict[str, Any]) -> int:
    """Insert one prediction record and return its new row id.

    `record` is expected to already use the snake_case column names that
    match the `churn_predictions` table in schema.sql.
    """
    columns = ", ".join(record.keys())
    placeholders = ", ".join(["%s"] * len(record))
    query = f"INSERT INTO churn_predictions ({columns}) VALUES ({placeholders})"

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, list(record.values()))
        conn.commit()
        return cursor.lastrowid
    except MySQLError as exc:
        conn.rollback()
        raise exc
    finally:
        cursor.close()
        conn.close()


def get_recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    """Fetch the most recent predictions, newest first, for the dashboard's
    history view.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)  # returns rows as dicts
        cursor.execute(
            "SELECT * FROM churn_predictions ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
