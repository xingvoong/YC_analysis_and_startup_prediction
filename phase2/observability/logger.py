"""
Prediction logger — writes each /predict call to a SQLite store.
Imported by the FastAPI app.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            desc_len    INTEGER,
            score       INTEGER,
            level       TEXT,
            model_ver   TEXT
        )
    """)
    con.commit()
    return con


def log_prediction(score: int, level: str, desc_len: int, model_version: str):
    with _conn() as con:
        con.execute(
            "INSERT INTO predictions (ts, desc_len, score, level, model_ver) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), desc_len, score, level, model_version),
        )
