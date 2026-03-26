"""SQLite logger for decisions and corridor events."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

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


def log_decision(
    *,
    step: int,
    junction_id: str,
    phase: str,
    duration_s: int,
    reasoning: str,
    emergency: bool,
    pce_north: float,
    pce_south: float,
    pce_east: float,
    pce_west: float,
    avg_wait_s: float,
    latency_ms: float,
    controller_type: str,
) -> None:
    init_db()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO decisions
            (step, timestamp, junction_id, phase, duration_s, reasoning, emergency,
             pce_north, pce_south, pce_east, pce_west, avg_wait_s, latency_ms, controller_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(step),
                time.time(),
                junction_id,
                phase,
                int(duration_s),
                reasoning,
                int(bool(emergency)),
                float(pce_north),
                float(pce_south),
                float(pce_east),
                float(pce_west),
                float(avg_wait_s),
                float(latency_ms),
                controller_type,
            ),
        )


def fetch_recent_decisions(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    safe_limit = max(1, min(int(limit), 500))
    with _conn() as con:
        rows = con.execute(
            """
            SELECT *
            FROM decisions
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_kpi_summary() -> dict[str, Any]:
    init_db()
    with _conn() as con:
        agg = con.execute(
            """
            SELECT
                COUNT(*) AS decisions,
                AVG(avg_wait_s) AS avg_wait_s,
                AVG(latency_ms) AS avg_latency_ms,
                AVG(CASE WHEN emergency = 1 THEN 1.0 ELSE 0.0 END) AS emergency_rate,
                AVG(CASE WHEN controller_type = 'rule_based' THEN 1.0 ELSE 0.0 END) AS fallback_rate
            FROM decisions
            """
        ).fetchone()
        corridor_count = con.execute("SELECT COUNT(*) AS c FROM corridor_events").fetchone()

    return {
        "decisions": int(agg["decisions"] or 0),
        "avg_wait_s": float(agg["avg_wait_s"] or 0.0),
        "avg_latency_ms": float(agg["avg_latency_ms"] or 0.0),
        "emergency_rate": float(agg["emergency_rate"] or 0.0),
        "fallback_rate": float(agg["fallback_rate"] or 0.0),
        "corridor_events": int(corridor_count["c"] or 0),
    }
