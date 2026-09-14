import sys
from pathlib import Path

# Add project root directory to sys.path so config can be imported from src/ scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import config


def get_connection(db_path=None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    target_path = db_path or config.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=None, schema_path=None) -> None:
    """Initializes the database schema if tables do not exist."""
    schema_file = schema_path or config.SCHEMA_PATH
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = get_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def insert_response(
    prompt_id: int, platform: str, raw_text: str, db_path=None
) -> int:
    """Inserts a new response and returns its auto-generated response ID."""
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO responses (prompt_id, platform, raw_text, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (prompt_id, platform, raw_text, timestamp),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_mention(
    response_id: int,
    brand_name: str,
    position: int,
    sentiment: str,
    db_path=None,
) -> int:
    """Inserts a detected brand mention into the mentions table."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO mentions (response_id, brand_name, position, sentiment)
            VALUES (?, ?, ?, ?)
            """,
            (response_id, brand_name, position, sentiment),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_citation(
    response_id: int,
    source_url: Optional[str],
    source_domain: str,
    db_path=None,
) -> int:
    """Inserts an extracted citation/domain into the citations table."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO citations (response_id, source_url, source_domain)
            VALUES (?, ?, ?)
            """,
            (response_id, source_url, source_domain),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_existing_response(
    prompt_id: int, platform: str, db_path=None
) -> Optional[Dict[str, Any]]:
    """Checks if a response already exists for a given prompt_id and platform."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM responses
            WHERE prompt_id = ? AND platform = ?
            ORDER BY id DESC LIMIT 1
            """,
            (prompt_id, platform),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_response(prompt_id: int, platform: str, db_path=None) -> None:
    """Deletes existing response and associated mentions/citations for re-runs."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM responses WHERE prompt_id = ? AND platform = ?",
            (prompt_id, platform),
        )
        rows = cursor.fetchall()
        for row in rows:
            response_id = row["id"]
            cursor.execute("DELETE FROM mentions WHERE response_id = ?", (response_id,))
            cursor.execute("DELETE FROM citations WHERE response_id = ?", (response_id,))
            cursor.execute("DELETE FROM responses WHERE id = ?", (response_id,))
        conn.commit()
    finally:
        conn.close()


def fetch_all_responses(db_path=None) -> List[Dict[str, Any]]:
    """Fetches all stored responses."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM responses")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_all_mentions(db_path=None) -> List[Dict[str, Any]]:
    """Fetches all stored brand mentions."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM mentions")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_all_citations(db_path=None) -> List[Dict[str, Any]]:
    """Fetches all stored citations."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM citations")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def clear_database(db_path=None) -> None:
    """Clears all records from tables."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM citations;")
        conn.execute("DELETE FROM mentions;")
        conn.execute("DELETE FROM responses;")
        conn.commit()
    finally:
        conn.close()
