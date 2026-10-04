import json
import sqlite3
from datetime import datetime, timedelta, timezone

from app.database import init_db, recent_analyses


def test_recent_analyses_keeps_records_older_than_three_days_in_storage(tmp_path):
    database = tmp_path / "history.sqlite3"
    init_db(database)
    now = datetime.now(timezone.utc)
    records = [
        ("expired", now - timedelta(days=3, seconds=1)),
        ("recent", now - timedelta(days=2)),
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """INSERT INTO analyses(input_text, language, result_json, created_at)
               VALUES (?, ?, ?, ?)""",
            [
                (text, "am", json.dumps({"main_topic": text}), created_at.isoformat())
                for text, created_at in records
            ],
        )

    items = recent_analyses(database)

    assert {item["input_text"] for item in items} == {"expired", "recent"}