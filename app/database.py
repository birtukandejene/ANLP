import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def get_connection(path):
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                language TEXT NOT NULL,
                result_json TEXT NOT NULL,
                embedding_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                note TEXT DEFAULT '',
                item_type TEXT NOT NULL DEFAULT 'analysis',
                item_text TEXT NOT NULL DEFAULT '',
                corrected_text TEXT NOT NULL DEFAULT '',
                trust_score REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'pending',
                input_text TEXT NOT NULL DEFAULT '',
                prediction_json TEXT NOT NULL DEFAULT '{}',
                embedding_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            );
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(feedback)").fetchall()}
        if "item_type" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN item_type TEXT NOT NULL DEFAULT 'analysis'")
        if "item_text" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN item_text TEXT NOT NULL DEFAULT ''")
        if "corrected_text" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN corrected_text TEXT NOT NULL DEFAULT ''")
        if "trust_score" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN trust_score REAL NOT NULL DEFAULT 0.0")
        if "status" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        for column, declaration in (
            ("input_text", "TEXT NOT NULL DEFAULT ''"),
            ("prediction_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("embedding_json", "TEXT"),
        ):
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE feedback ADD COLUMN {column} {declaration}"
                )
        analysis_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(analyses)").fetchall()
        }
        if "embedding_json" not in analysis_columns:
            connection.execute("ALTER TABLE analyses ADD COLUMN embedding_json TEXT")
        connection.execute(
            """UPDATE feedback
               SET input_text = COALESCE(NULLIF(input_text, ''), (
                   SELECT input_text FROM analyses WHERE analyses.id = feedback.analysis_id
               )),
               prediction_json = COALESCE(NULLIF(prediction_json, '{}'), (
                   SELECT result_json FROM analyses WHERE analyses.id = feedback.analysis_id
               )),
               embedding_json = COALESCE(embedding_json, (
                   SELECT embedding_json FROM analyses WHERE analyses.id = feedback.analysis_id
               ))
               WHERE input_text = '' OR prediction_json = '{}' OR embedding_json IS NULL"""
        )


def save_analysis(path, input_text, result, embedding=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(path) as connection:
        cursor = connection.execute(
            """INSERT INTO analyses(
                input_text, language, result_json, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                input_text,
                result.get("language", ""),
                json.dumps(result, ensure_ascii=False),
                json.dumps(embedding) if embedding is not None else None,
                created_at,
            ),
        )
        return cursor.lastrowid


def save_feedback(path, analysis_id, feedback_items):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(path) as connection:
        analysis = connection.execute(
            "SELECT input_text, result_json, embedding_json FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
        if analysis is None:
            raise ValueError("Analysis not found.")
        stored_items = []
        for item in feedback_items:
            corrected_text = item.get("corrected_text", item["item_text"]).strip()
            trust_score = 0.0
            if item["rating"] == 1:
                trust_score = 0.70
                if corrected_text != item["item_text"]:
                    trust_score += 0.15
                if item["item_type"] == "keyword" and corrected_text in analysis["input_text"]:
                    trust_score += 0.05
            stored_items.append((
                analysis_id,
                item["rating"],
                "",
                item["item_type"],
                item["item_text"],
                corrected_text,
                min(trust_score, 0.95),
                "stored",
                analysis["input_text"],
                analysis["result_json"],
                analysis["embedding_json"],
                created_at,
            ))
        connection.executemany(
            """INSERT INTO feedback(
                analysis_id, rating, note, item_type, item_text, corrected_text,
                trust_score, status, input_text, prediction_json, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            stored_items,
        )


def feedback_progress(path):
    thresholds = {"main_topic": 3, "subtopic": 3, "keyword": 3}
    with get_connection(path) as connection:
        rows = connection.execute(
            """SELECT item_type, COUNT(*) AS count
               FROM feedback
               WHERE status = 'stored'
               GROUP BY item_type"""
        ).fetchall()
    counts = {item_type: 0 for item_type in thresholds}
    counts.update({row["item_type"]: row["count"] for row in rows if row["item_type"] in counts})
    return {
        "counts": counts,
        "thresholds": thresholds,
        "ready": all(counts[item_type] >= threshold for item_type, threshold in thresholds.items()),
    }


def recent_analyses(path, limit=20):
    with get_connection(path) as connection:
        rows = connection.execute(
            "SELECT id, input_text, result_json, created_at FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": row["id"], "input_text": row["input_text"], "result": json.loads(row["result_json"]), "created_at": row["created_at"]}
        for row in rows
    ]


def delete_analysis(path, analysis_id):
    with get_connection(path) as connection:
        cursor = connection.execute(
            "DELETE FROM analyses WHERE id = ?",
            (analysis_id,),
        )
        return cursor.rowcount > 0
