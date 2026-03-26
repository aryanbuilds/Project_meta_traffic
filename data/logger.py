"""SQLite logger for decisions and corridor events."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/decisions.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                timestamp REAL,
                junction_id TEXT,
                phase TEXT,
                duration_s INTEGER,
                reasoning TEXT,
                emergency INTEGER,
                pce_north REAL,
                pce_south REAL,
                pce_east REAL,
                pce_west REAL,
                avg_wait_s REAL,
                latency_ms REAL,
                controller_type TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS corridor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                ambulance_id TEXT,
                destination TEXT,
                origin_direction TEXT,
                route_text TEXT,
                duration_s INTEGER
            )
            """
        )


def log_corridor_event(
    ambulance_id: str,
    destination: str,
    origin_direction: str,
    route_text: str,
    duration_s: int,
) -> None:
    init_db()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO corridor_events
            (timestamp, ambulance_id, destination, origin_direction, route_text, duration_s)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), ambulance_id, destination, origin_direction, route_text, duration_s),
        )
