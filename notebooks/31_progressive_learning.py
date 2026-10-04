#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
28_progressive_learning.py

SAFE PROGRESSIVE LEARNING LAYER FOR THE AMHARIC NLP SYSTEM

Purpose
-------
Store user interactions and feedback in a database without immediately
contaminating the trusted training dataset.

Learning flow
-------------
1. User submits text.
2. Existing model makes a prediction.
3. Prediction is saved to SQLite.
4. User may:
      - accept the prediction
      - correct the domain
      - correct the topic
      - correct the subtopic
      - provide a completely new label
5. Feedback is stored as PENDING.
6. Validation rules determine whether the example is:
      - pending
      - approved
      - rejected
7. Approved examples become candidate training examples.
8. A later rebuild process can combine:
      trusted annotated dataset
             +
      approved database examples
             ↓
      new semantic index / supervised model

IMPORTANT
---------
This script DOES NOT modify the original Excel training dataset.

It also DOES NOT immediately retrain the model after every user input.

The database is the source for progressive-learning examples.

Database
--------
data/progressive_learning.db

Tables
------
interactions
feedback
learning_examples
learning_runs

Status flow
-----------
PENDING
   ↓
VALIDATING
   ↓
APPROVED
   or
REJECTED
    initialize_database(database_file)
Trust levels
------------
0.00 - raw user input only
0.25 - model prediction only
0.70 - user accepted prediction
0.85 - explicit user correction
0.95 - strongly validated/repeated correction

The actual trust score is calculated using several pieces of evidence.

This script is intentionally independent of a fixed number of domains.
Domains and topics are learned dynamically from the supplied data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "progressive_learning.db"
)

DATABASE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# --------------------------------------------------------------------------
# Safety thresholds
# --------------------------------------------------------------------------

# User explicitly confirmed the model prediction.
ACCEPTED_TRUST_THRESHOLD = 0.70

# User explicitly corrected the model.
CORRECTION_TRUST_THRESHOLD = 0.85

# Repeated consistent feedback can increase confidence.
REPEATED_FEEDBACK_BONUS = 0.05

# Maximum trust value.
MAX_TRUST_SCORE = 0.98

# Minimum score required for automatic approval.
AUTO_APPROVE_THRESHOLD = 0.90

# Minimum number of supporting confirmations for a strong automatic
# promotion when feedback is not an explicit correction.
MIN_SUPPORTING_CONFIRMATIONS = 2

# --------------------------------------------------------------------------
# Dataset safety
# --------------------------------------------------------------------------

# Very short texts are usually not useful as training examples.
MIN_TEXT_LENGTH = 3

# Avoid storing excessively large user submissions as training examples.
MAX_TEXT_LENGTH = 20_000

# --------------------------------------------------------------------------
# Dynamic labels
# --------------------------------------------------------------------------

# There is intentionally NO:
#
# DOMAINS = [...]
#
# and NO:
#
# TOPICS = [...]
#
# Labels are whatever the user/model/database provides.


# ============================================================================
# STATUS VALUES
# ============================================================================

STATUS_PENDING = "pending"
STATUS_VALIDATING = "validating"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

FEEDBACK_NONE = "none"
FEEDBACK_ACCEPT = "accept"
FEEDBACK_CORRECT = "correct"
FEEDBACK_REJECT = "reject"


# ============================================================================
# TIME
# ============================================================================

def utc_now() -> str:
    """
    Return an ISO-8601 UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

AMHARIC_TOKEN_RE = re.compile(
    r"[\u1200-\u137F]+"
)


def normalize_text(text: Any) -> str:
    """
    Normalize whitespace while preserving the original language/content.
    """
    if text is None:
        return ""

    text = str(text)

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def validate_text(text: str) -> str:
    """
    Validate and normalize incoming user text.
    """
    text = normalize_text(text)

    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"Text is too short. Minimum length: "
            f"{MIN_TEXT_LENGTH}"
        )

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text is too long. Maximum length: "
            f"{MAX_TEXT_LENGTH}"
        )

    return text


# ============================================================================
# LABEL NORMALIZATION
# ============================================================================

def normalize_label(value: Any) -> Optional[str]:
    """
    Normalize optional domain/topic/subtopic labels.

    Empty values become None.
    """
    if value is None:
        return None

    value = normalize_text(value)

    if not value:
        return None

    return value


def normalize_topics(
    topics: Any,
) -> list[str]:
    """
    Convert topics into a unique list.

    Supports:
        ["topic1", "topic2"]

    and:
        "topic1|topic2"
    """
    if topics is None:
        return []

    if isinstance(topics, (list, tuple, set)):
        raw_topics = list(topics)
    else:
        raw_topics = str(topics).split("|")

    result = []

    for topic in raw_topics:
        topic = normalize_label(topic)

        if topic and topic not in result:
            result.append(topic)

    return result


# ============================================================================
# HASHING
# ============================================================================

def text_hash(text: str) -> str:
    """
    Stable SHA-256 hash used for duplicate detection.

    The actual user text is still stored in the database.
    """
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================================
# DATABASE
# ============================================================================

def get_connection(
    database_file: Path = DATABASE_FILE,
) -> sqlite3.Connection:
    """
    Open SQLite connection.
    """
    connection = sqlite3.connect(
        database_file
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL"
    )

    return connection


def initialize_database(
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Create all progressive-learning tables.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        # ------------------------------------------------------------------
        # Interactions
        # ------------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                text TEXT NOT NULL,

                text_hash TEXT NOT NULL,

                predicted_domain TEXT,

                predicted_topics TEXT,

                predicted_subtopics TEXT,

                predicted_keywords TEXT NOT NULL DEFAULT '[]',

                prediction_confidence REAL,

                semantic_similarity REAL,

                model_version TEXT,

                created_at TEXT NOT NULL,

                metadata_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_interactions_text_hash
            ON interactions(text_hash)
            """
        )

        # ------------------------------------------------------------------
        # Feedback
        # ------------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                interaction_id INTEGER NOT NULL,

                feedback_type TEXT NOT NULL,

                user_domain TEXT,

                user_topics TEXT,

                user_subtopics TEXT,

                user_keywords TEXT NOT NULL DEFAULT '[]',

                comment TEXT,

                trust_score REAL NOT NULL DEFAULT 0.0,

                status TEXT NOT NULL DEFAULT 'pending',

                validation_reason TEXT,

                created_at TEXT NOT NULL,

                validated_at TEXT,

                FOREIGN KEY(interaction_id)
                    REFERENCES interactions(id)
                    ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_feedback_status
            ON feedback(status)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_feedback_interaction
            ON feedback(interaction_id)
            """
        )

        # ------------------------------------------------------------------
        # Learning examples
        # ------------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                interaction_id INTEGER,

                feedback_id INTEGER,

                text TEXT NOT NULL,

                text_hash TEXT NOT NULL,

                domain TEXT,

                topics TEXT,

                subtopics TEXT,

                keywords TEXT NOT NULL DEFAULT '[]',

                label_type TEXT NOT NULL DEFAULT 'topic',

                source TEXT NOT NULL,

                trust_score REAL NOT NULL,

                status TEXT NOT NULL DEFAULT 'pending',

                approved_at TEXT,

                rejected_at TEXT,

                rejection_reason TEXT,

                used_in_training INTEGER NOT NULL DEFAULT 0,

                training_run_id INTEGER,

                created_at TEXT NOT NULL,

                FOREIGN KEY(interaction_id)
                    REFERENCES interactions(id)
                    ON DELETE SET NULL,

                FOREIGN KEY(feedback_id)
                    REFERENCES feedback(id)
                    ON DELETE SET NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_learning_examples_status
            ON learning_examples(status)
            """
        )

        for table, column, declaration in (
            ("interactions", "predicted_keywords", "TEXT NOT NULL DEFAULT '[]'"),
            ("feedback", "user_keywords", "TEXT NOT NULL DEFAULT '[]'"),
            ("learning_examples", "keywords", "TEXT NOT NULL DEFAULT '[]'"),
            ("learning_examples", "label_type", "TEXT NOT NULL DEFAULT 'topic'"),
        ):
            columns = {
                row[1]
                for row in cursor.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if column not in columns:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_feedback_imports (
                dashboard_feedback_id INTEGER PRIMARY KEY,
                learner_feedback_id INTEGER NOT NULL,
                learning_example_id INTEGER,
                imported_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_learning_examples_hash
            ON learning_examples(text_hash)
            """
        )

        # ------------------------------------------------------------------
        # Learning runs
        # ------------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_version TEXT NOT NULL,

                started_at TEXT NOT NULL,

                completed_at TEXT,

                approved_examples INTEGER NOT NULL DEFAULT 0,

                used_examples INTEGER NOT NULL DEFAULT 0,

                notes TEXT
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================================
# PREDICTION STORAGE
# ============================================================================

def save_interaction(
    text: str,
    predicted_domain: Optional[str] = None,
    predicted_topics: Optional[list[str]] = None,
    predicted_subtopics: Optional[list[str]] = None,
    predicted_keywords: Optional[list[str]] = None,
    prediction_confidence: Optional[float] = None,
    semantic_similarity: Optional[float] = None,
    model_version: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    database_file: Path = DATABASE_FILE,
) -> int:
    """
    Save a model interaction.

    This does NOT mean the data is trusted.

    It only records what the model predicted.
    """

    text = validate_text(text)

    predicted_domain = normalize_label(
        predicted_domain
    )

    predicted_topics = normalize_topics(
        predicted_topics
    )

    predicted_subtopics = normalize_topics(
        predicted_subtopics
    )

    predicted_keywords = normalize_topics(
        predicted_keywords
    )

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO interactions (
                text,
                text_hash,
                predicted_domain,
                predicted_topics,
                predicted_subtopics,
                predicted_keywords,
                prediction_confidence,
                semantic_similarity,
                model_version,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                text_hash(text),
                predicted_domain,
                json.dumps(
                    predicted_topics,
                    ensure_ascii=False,
                ),
                json.dumps(
                    predicted_subtopics,
                    ensure_ascii=False,
                ),
                json.dumps(
                    predicted_keywords,
                    ensure_ascii=False,
                ),
                prediction_confidence,
                semantic_similarity,
                model_version,
                utc_now(),
                json.dumps(
                    metadata or {},
                    ensure_ascii=False,
                ),
            ),
        )

        interaction_id = cursor.lastrowid

        connection.commit()

        return int(interaction_id)

    finally:
        connection.close()


# ============================================================================
# FEEDBACK TRUST
# ============================================================================

def calculate_trust_score(
    feedback_type: str,
    prediction_confidence: Optional[float],
    user_domain: Optional[str],
    predicted_domain: Optional[str],
    user_topics: list[str],
    predicted_topics: list[str],
    supporting_feedback_count: int = 0,
) -> float:
    """
    Calculate a conservative trust score.

    Important:
        A model prediction by itself has ZERO training authority.

    User feedback is what increases trust.
    """

    feedback_type = (
        feedback_type or FEEDBACK_NONE
    ).lower()

    confidence = (
        float(prediction_confidence)
        if prediction_confidence is not None
        else 0.0
    )

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    score = 0.0

    # ----------------------------------------------------------------------
    # Explicit user correction
    # ----------------------------------------------------------------------

    if feedback_type == FEEDBACK_CORRECT:
        score = CORRECTION_TRUST_THRESHOLD

        # Explicit domain correction.
        if (
            user_domain
            and predicted_domain
            and user_domain != predicted_domain
        ):
            score += 0.05

        # Explicit topic correction.
        if user_topics:
            predicted_set = set(
                predicted_topics
            )

            user_set = set(
                user_topics
            )

            if user_set != predicted_set:
                score += 0.03

    # ----------------------------------------------------------------------
    # User accepts model prediction
    # ----------------------------------------------------------------------

    elif feedback_type == FEEDBACK_ACCEPT:
        score = (
            ACCEPTED_TRUST_THRESHOLD
            + 0.15 * confidence
        )

    # ----------------------------------------------------------------------
    # Explicit rejection
    # ----------------------------------------------------------------------

    elif feedback_type == FEEDBACK_REJECT:
        score = 0.0

    else:
        score = 0.0

    # ----------------------------------------------------------------------
    # Repeated supporting evidence
    # ----------------------------------------------------------------------

    if supporting_feedback_count > 0:
        bonus = min(
            0.10,
            supporting_feedback_count
            * REPEATED_FEEDBACK_BONUS,
        )

        score += bonus

    return float(
        min(
            MAX_TRUST_SCORE,
            max(0.0, score),
        )
    )


# ============================================================================
# FEEDBACK
# ============================================================================

def submit_feedback(
    interaction_id: int,
    feedback_type: str,
    user_domain: Optional[str] = None,
    user_topics: Optional[list[str]] = None,
    user_subtopics: Optional[list[str]] = None,
    user_keywords: Optional[list[str]] = None,
    comment: Optional[str] = None,
    database_file: Path = DATABASE_FILE,
) -> int:
    """
    Save user feedback.

    The example remains pending unless validation criteria are satisfied.
    """

    feedback_type = (
        feedback_type or FEEDBACK_NONE
    ).lower()

    allowed = {
        FEEDBACK_ACCEPT,
        FEEDBACK_CORRECT,
        FEEDBACK_REJECT,
    }

    if feedback_type not in allowed:
        raise ValueError(
            "feedback_type must be one of: "
            "accept, correct, reject"
        )

    user_domain = normalize_label(
        user_domain
    )

    user_topics = normalize_topics(
        user_topics
    )

    user_subtopics = normalize_topics(
        user_subtopics
    )

    user_keywords = normalize_topics(
        user_keywords
    )

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM interactions
            WHERE id = ?
            """,
            (interaction_id,),
        )

        interaction = cursor.fetchone()

        if interaction is None:
            raise ValueError(
                f"Interaction not found: "
                f"{interaction_id}"
            )

        predicted_topics = normalize_topics(
            json.loads(
                interaction["predicted_topics"]
                or "[]"
            )
        )

        predicted_subtopics = normalize_topics(
            json.loads(
                interaction["predicted_subtopics"]
                or "[]"
            )
        )

        # --------------------------------------------------------------
        # Count previous supporting feedback for same text.
        # --------------------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM feedback f
            JOIN interactions i
              ON i.id = f.interaction_id
            WHERE i.text_hash = ?
              AND f.status = 'approved'
              AND f.feedback_type IN ('accept', 'correct')
            """,
            (
                interaction["text_hash"],
            ),
        )

        supporting_count = int(
            cursor.fetchone()[0]
        )

        # --------------------------------------------------------------
        # Calculate trust.
        # --------------------------------------------------------------

        trust_score = calculate_trust_score(
            feedback_type=feedback_type,
            prediction_confidence=interaction[
                "prediction_confidence"
            ],
            user_domain=user_domain,
            predicted_domain=interaction[
                "predicted_domain"
            ],
            user_topics=user_topics,
            predicted_topics=predicted_topics,
            supporting_feedback_count=supporting_count,
        )

        # --------------------------------------------------------------
        # Determine initial status.
        # --------------------------------------------------------------

        if feedback_type == FEEDBACK_REJECT:
            status = STATUS_REJECTED

            validation_reason = (
                "User explicitly rejected "
                "the prediction."
            )

        elif (
            feedback_type == FEEDBACK_CORRECT
            and trust_score >= CORRECTION_TRUST_THRESHOLD
        ):
            status = STATUS_APPROVED

            validation_reason = (
                "Explicit user correction "
                "passed the correction trust threshold."
            )

        elif (
            feedback_type == FEEDBACK_ACCEPT
            and trust_score >= AUTO_APPROVE_THRESHOLD
            and supporting_count
            >= MIN_SUPPORTING_CONFIRMATIONS
        ):
            status = STATUS_APPROVED

            validation_reason = (
                "User acceptance has repeated "
                "supporting evidence."
            )

        else:
            status = STATUS_PENDING

            validation_reason = (
                "Stored as pending; additional "
                "evidence is required."
            )

        now = utc_now()

        cursor.execute(
            """
            INSERT INTO feedback (
                interaction_id,
                feedback_type,
                user_domain,
                user_topics,
                user_subtopics,
                user_keywords,
                comment,
                trust_score,
                status,
                validation_reason,
                created_at,
                validated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                feedback_type,
                user_domain,
                json.dumps(
                    user_topics,
                    ensure_ascii=False,
                ),
                json.dumps(
                    user_subtopics,
                    ensure_ascii=False,
                ),
                json.dumps(
                    user_keywords,
                    ensure_ascii=False,
                ),
                comment,
                trust_score,
                status,
                validation_reason,
                now,
                now if status != STATUS_PENDING
                else None,
            ),
        )

        feedback_id = cursor.lastrowid

        # --------------------------------------------------------------
        # If accepted/corrected, create a learning example.
        # --------------------------------------------------------------

        if feedback_type in {
            FEEDBACK_ACCEPT,
            FEEDBACK_CORRECT,
        }:

            if feedback_type == FEEDBACK_CORRECT:
                learning_domain = (
                    user_domain
                    or interaction[
                        "predicted_domain"
                    ]
                )

                learning_topics = (
                    user_topics
                    or predicted_topics
                )

                learning_subtopics = (
                    user_subtopics
                    or predicted_subtopics
                )

                source = "user_correction"
                learning_keywords = user_keywords or normalize_topics(
                    json.loads(interaction["predicted_keywords"] or "[]")
                )

            else:
                learning_domain = (
                    interaction[
                        "predicted_domain"
                    ]
                )

                learning_topics = predicted_topics

                learning_subtopics = (
                    predicted_subtopics
                )

                source = "user_confirmation"
                learning_keywords = normalize_topics(
                    json.loads(interaction["predicted_keywords"] or "[]")
                )

            label_type = "keyword" if user_keywords else "topic"

            cursor.execute(
                """
                INSERT INTO learning_examples (
                    interaction_id,
                    feedback_id,
                    text,
                    text_hash,
                    domain,
                    topics,
                    subtopics,
                    keywords,
                    label_type,
                    source,
                    trust_score,
                    status,
                    approved_at,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    feedback_id,
                    interaction["text"],
                    interaction["text_hash"],
                    learning_domain,
                    json.dumps(
                        learning_topics,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        learning_subtopics,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        learning_keywords,
                        ensure_ascii=False,
                    ),
                    label_type,
                    source,
                    trust_score,
                    status,
                    now
                    if status == STATUS_APPROVED
                    else None,
                    now,
                ),
            )

        connection.commit()

        return int(feedback_id)

    finally:
        connection.close()


# ============================================================================
# MANUAL VALIDATION
# ============================================================================

def approve_feedback(
    feedback_id: int,
    reason: str = "Manually validated.",
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Manually approve a feedback record.

    This is the strongest safety mechanism.

    A human/admin can explicitly promote an uncertain example.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM feedback
            WHERE id = ?
            """,
            (feedback_id,),
        )

        feedback = cursor.fetchone()

        if feedback is None:
            raise ValueError(
                f"Feedback not found: "
                f"{feedback_id}"
            )

        now = utc_now()

        cursor.execute(
            """
            UPDATE feedback
            SET status = ?,
                validation_reason = ?,
                validated_at = ?
            WHERE id = ?
            """,
            (
                STATUS_APPROVED,
                reason,
                now,
                feedback_id,
            ),
        )

        cursor.execute(
            """
            UPDATE learning_examples
            SET status = ?,
                approved_at = ?
            WHERE feedback_id = ?
            """,
            (
                STATUS_APPROVED,
                now,
                feedback_id,
            ),
        )

        connection.commit()

    finally:
        connection.close()


def reject_feedback(
    feedback_id: int,
    reason: str = "Manually rejected.",
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Manually reject a feedback record.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        now = utc_now()

        cursor.execute(
            """
            UPDATE feedback
            SET status = ?,
                validation_reason = ?,
                validated_at = ?
            WHERE id = ?
            """,
            (
                STATUS_REJECTED,
                reason,
                now,
                feedback_id,
            ),
        )

        cursor.execute(
            """
            UPDATE learning_examples
            SET status = ?,
                rejected_at = ?,
                rejection_reason = ?
            WHERE feedback_id = ?
            """,
            (
                STATUS_REJECTED,
                now,
                reason,
                feedback_id,
            ),
        )

        connection.commit()

    finally:
        connection.close()


def get_pending_feedback(
    database_file: Path = DATABASE_FILE,
) -> list[dict[str, Any]]:
    connection = get_connection(database_file)
    try:
        rows = connection.execute(
            """
            SELECT f.id, f.feedback_type, f.user_domain, f.user_topics,
                   f.user_subtopics, f.user_keywords, f.comment,
                   f.trust_score, i.text
            FROM feedback f
            JOIN interactions i ON i.id = f.interaction_id
            WHERE f.status = 'pending'
            ORDER BY f.created_at, f.id
            """
        ).fetchall()
        return [
            {
                "id": row["id"],
                "feedback_type": row["feedback_type"],
                "user_domain": row["user_domain"],
                "user_topics": normalize_topics(json.loads(row["user_topics"] or "[]")),
                "user_subtopics": normalize_topics(json.loads(row["user_subtopics"] or "[]")),
                "user_keywords": normalize_topics(json.loads(row["user_keywords"] or "[]")),
                "comment": row["comment"],
                "trust_score": row["trust_score"],
                "text": row["text"],
            }
            for row in rows
        ]
    finally:
        connection.close()


# ============================================================================
# APPROVED TRAINING DATA
# ============================================================================

def get_approved_learning_examples(
    unused_only: bool = True,
    database_file: Path = DATABASE_FILE,
) -> list[dict[str, Any]]:
    """
    Return approved learning examples.

    These are the ONLY database records that should be allowed
    into the future training/index-building pipeline.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        if unused_only:

            cursor.execute(
                """
                SELECT *
                FROM learning_examples
                WHERE status = 'approved'
                  AND used_in_training = 0
                ORDER BY approved_at ASC
                """
            )

        else:

            cursor.execute(
                """
                SELECT *
                FROM learning_examples
                WHERE status = 'approved'
                ORDER BY approved_at ASC
                """
            )

        rows = cursor.fetchall()

        result = []

        for row in rows:

            result.append(
                {
                    "id": row["id"],
                    "interaction_id": row[
                        "interaction_id"
                    ],
                    "feedback_id": row[
                        "feedback_id"
                    ],
                    "text": row["text"],
                    "domain": row["domain"],
                    "topics": normalize_topics(
                        json.loads(
                            row["topics"]
                            or "[]"
                        )
                    ),
                    "subtopics": normalize_topics(
                        json.loads(
                            row["subtopics"]
                            or "[]"
                        )
                    ),
                    "keywords": normalize_topics(
                        json.loads(
                            row["keywords"]
                            or "[]"
                        )
                    ),
                    "label_type": row["label_type"],
                    "source": row["source"],
                    "trust_score": row[
                        "trust_score"
                    ],
                    "approved_at": row[
                        "approved_at"
                    ],
                }
            )

        return result

    finally:
        connection.close()


def import_dashboard_feedback(
    dashboard_database: Path,
    learner_database: Path = DATABASE_FILE,
) -> dict[str, int]:
    """Import dashboard labels without treating them as reviewed gold data."""

    if not dashboard_database.exists():
        raise FileNotFoundError(
            f"Dashboard database not found: {dashboard_database}"
        )

    initialize_database(learner_database)
    source = sqlite3.connect(dashboard_database)
    source.row_factory = sqlite3.Row
    target = get_connection(learner_database)
    imported = 0
    skipped = 0
    try:
        rows = source.execute(
            """
                 SELECT f.id AS dashboard_feedback_id, f.analysis_id,
                     f.rating, f.status AS dashboard_status,
                     f.item_type, f.item_text, f.corrected_text,
                   f.created_at, a.input_text, a.result_json
            FROM feedback f
            JOIN analyses a ON a.id = f.analysis_id
            ORDER BY f.id
            """
        ).fetchall()
        active_ids = {int(row["dashboard_feedback_id"]) for row in rows}
        imported_ids = {
            int(row[0])
            for row in target.execute(
                "SELECT dashboard_feedback_id FROM dashboard_feedback_imports"
            ).fetchall()
        }
        stale_ids = imported_ids.difference(active_ids)
        for stale_id in stale_ids:
            mapping = target.execute(
                """
                SELECT learner_feedback_id
                FROM dashboard_feedback_imports
                WHERE dashboard_feedback_id = ?
                """,
                (stale_id,),
            ).fetchone()
            target.execute(
                """
                UPDATE feedback
                SET status = 'rejected',
                    validation_reason = 'Dashboard feedback was replaced or deleted.'
                WHERE id = ?
                """,
                (mapping["learner_feedback_id"],),
            )
            target.execute(
                """
                UPDATE learning_examples
                SET status = 'rejected',
                    rejected_at = ?,
                    rejection_reason = 'Dashboard feedback was replaced or deleted.'
                WHERE feedback_id = ? AND used_in_training = 0
                """,
                (utc_now(), mapping["learner_feedback_id"]),
            )

        for row in rows:
            existing = target.execute(
                "SELECT 1 FROM dashboard_feedback_imports WHERE dashboard_feedback_id = ?",
                (row["dashboard_feedback_id"],),
            ).fetchone()
            if existing:
                skipped += 1
                continue

            result = json.loads(row["result_json"] or "{}")
            subtopics = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in result.get("subtopics", [])
            ]
            keywords = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in result.get("keywords", [])
            ]
            item_type = row["item_type"]
            predicted_domain = normalize_label(
                result.get("domain") or result.get("main_topic")
            )
            predicted_topics = normalize_topics(
                [result.get("main_topic", "")]
            )
            predicted_subtopics = normalize_topics(subtopics)
            predicted_keywords = normalize_topics(keywords)
            corrected = normalize_text(row["corrected_text"] or row["item_text"])
            positive = int(row["rating"]) == 1
            changed = positive and corrected != normalize_text(row["item_text"])
            feedback_type = (
                FEEDBACK_REJECT if not positive
                else FEEDBACK_CORRECT if changed
                else FEEDBACK_ACCEPT
            )

            cursor = target.execute(
                """
                INSERT INTO interactions (
                    text, text_hash, predicted_domain, predicted_topics,
                    predicted_subtopics, predicted_keywords,
                    prediction_confidence, semantic_similarity, model_version,
                    created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validate_text(row["input_text"]),
                    text_hash(row["input_text"]),
                    predicted_domain,
                    json.dumps(predicted_topics, ensure_ascii=False),
                    json.dumps(predicted_subtopics, ensure_ascii=False),
                    json.dumps(predicted_keywords, ensure_ascii=False),
                    result.get("topic_confidence"),
                    None,
                    "dashboard-feedback-import",
                    row["created_at"] or utc_now(),
                    json.dumps({"dashboard_analysis_id": row["analysis_id"]}),
                ),
            )
            interaction_id = int(cursor.lastrowid)

            user_domain = None
            user_topics: list[str] = []
            user_subtopics: list[str] = []
            user_keywords: list[str] = []
            if positive:
                if item_type == "main_topic":
                    user_domain = corrected
                    user_topics = predicted_subtopics
                elif item_type == "subtopic":
                    user_subtopics = [corrected]
                elif item_type == "keyword":
                    user_keywords = normalize_topics(corrected.split("|"))

            score = (
                CORRECTION_TRUST_THRESHOLD if changed
                else ACCEPTED_TRUST_THRESHOLD if positive
                else 0.0
            )
            is_approved = positive and row["dashboard_status"] == STATUS_APPROVED
            reason = (
                "Dashboard feedback was explicitly approved."
                if is_approved
                else "Dashboard feedback requires review."
                if positive
                else "Dashboard feedback rejected the prediction."
            )
            status = (
                STATUS_APPROVED if is_approved
                else STATUS_PENDING if positive
                else STATUS_REJECTED
            )
            validated_at = (
                row["created_at"] or utc_now()
                if status != STATUS_PENDING
                else None
            )
            feedback_cursor = target.execute(
                """
                INSERT INTO feedback (
                    interaction_id, feedback_type, user_domain, user_topics,
                    user_subtopics, user_keywords, comment, trust_score,
                    status, validation_reason, created_at, validated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    feedback_type,
                    user_domain,
                    json.dumps(user_topics, ensure_ascii=False),
                    json.dumps(user_subtopics, ensure_ascii=False),
                    json.dumps(user_keywords, ensure_ascii=False),
                    f"Dashboard {item_type}: {row['item_text']} -> {corrected}",
                    score,
                    status,
                    reason,
                    row["created_at"] or utc_now(),
                    validated_at,
                ),
            )
            learning_example_id = None
            if positive:
                learning_domain = user_domain or predicted_domain
                learning_topics = (
                    [corrected] if item_type == "subtopic"
                    else user_topics if item_type == "main_topic"
                    else []
                )
                learning_subtopics = user_subtopics if item_type == "subtopic" else []
                learning_keywords = user_keywords if item_type == "keyword" else []
                example_cursor = target.execute(
                    """
                    INSERT INTO learning_examples (
                        interaction_id, feedback_id, text, text_hash, domain,
                        topics, subtopics, keywords, label_type, source,
                        trust_score, status, approved_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        int(feedback_cursor.lastrowid),
                        row["input_text"],
                        text_hash(row["input_text"]),
                        learning_domain,
                        json.dumps(learning_topics, ensure_ascii=False),
                        json.dumps(learning_subtopics, ensure_ascii=False),
                        json.dumps(learning_keywords, ensure_ascii=False),
                        "keyword" if item_type == "keyword" else "topic",
                        "dashboard_feedback",
                        score,
                        status,
                        validated_at if status == STATUS_APPROVED else None,
                        row["created_at"] or utc_now(),
                    ),
                )
                learning_example_id = int(example_cursor.lastrowid)

            target.execute(
                """
                INSERT INTO dashboard_feedback_imports (
                    dashboard_feedback_id, learner_feedback_id,
                    learning_example_id, imported_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    row["dashboard_feedback_id"],
                    int(feedback_cursor.lastrowid),
                    learning_example_id,
                    utc_now(),
                ),
            )
            imported += 1

        target.commit()
        return {"imported": imported, "skipped": skipped}
    except Exception:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()


# ============================================================================
# EXPORT APPROVED DATA
# ============================================================================

def export_approved_examples(
    output_file: Path,
    database_file: Path = DATABASE_FILE,
) -> int:
    """
    Export approved database examples to JSONL.

    This file is a training bridge.

    The original annotated dataset remains untouched.
    """

    examples = get_approved_learning_examples(
        unused_only=True,
        database_file=database_file,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return len(examples)


def stage_approved_training_data(
    base_dataset: Path,
    output_dataset: Path,
    database_file: Path = DATABASE_FILE,
) -> dict[str, int]:
    """Create a separate augmented workbook from approved compatible labels."""

    import pandas as pd

    if not base_dataset.exists():
        raise FileNotFoundError(f"Training dataset not found: {base_dataset}")

    frame = pd.read_excel(base_dataset, engine="openpyxl")
    if not {"text", "domain", "topics"}.issubset(frame.columns):
        raise ValueError("Training dataset must contain text, domain, and topics columns.")

    examples = get_approved_learning_examples(
        unused_only=True,
        database_file=database_file,
    )
    existing_texts = {
        normalize_text(value).casefold()
        for value in frame["text"].dropna().tolist()
    }
    rows_by_text: dict[str, dict[str, Any]] = {}
    skipped = 0
    for example in examples:
        text_key = normalize_text(example["text"]).casefold()
        if not text_key or text_key in existing_texts:
            skipped += 1
            continue
        row = rows_by_text.setdefault(
            text_key,
            {column: None for column in frame.columns},
        )
        row["text"] = example["text"]

        if example["label_type"] == "keyword":
            if "keywords" not in frame.columns or not example["keywords"]:
                skipped += 1
                continue
            if any(keyword not in example["text"] for keyword in example["keywords"]):
                skipped += 1
                continue
            prior_keywords = normalize_topics(row.get("keywords"))
            row["keywords"] = " | ".join(
                normalize_topics(prior_keywords + example["keywords"])
            )
        else:
            if not example["domain"] or not example["topics"]:
                skipped += 1
                continue
            if row["domain"] and row["domain"] != example["domain"]:
                skipped += 1
                continue
            row["domain"] = example["domain"]
            prior_topics = normalize_topics(row.get("topics"))
            row["topics"] = " | ".join(
                normalize_topics(prior_topics + example["topics"])
            )

    staged_rows = []
    for text_key, row in rows_by_text.items():
        if not row.get("domain") or not normalize_topics(row.get("topics")):
            skipped += 1
            continue
        existing_texts.add(text_key)
        staged_rows.append(row)

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    staged = pd.concat([frame, pd.DataFrame(staged_rows, columns=frame.columns)], ignore_index=True)
    staged.to_excel(output_dataset, index=False, engine="openpyxl")
    return {"base_rows": len(frame), "added_rows": len(staged_rows), "skipped_rows": skipped}


def evaluate_gate(
    baseline_file: Path,
    candidate_file: Path,
    metric: str,
    minimum_documents: int = 100,
) -> dict[str, Any]:
    """Return a non-regression decision from held-out evaluation reports."""

    with baseline_file.open(encoding="utf-8") as file:
        baseline = json.load(file)
    with candidate_file.open(encoding="utf-8") as file:
        candidate = json.load(file)
    baseline_metrics = baseline.get("metrics", baseline)
    candidate_metrics = candidate.get("metrics", candidate)

    def metric_value(report: dict[str, Any]) -> Any:
        value: Any = report.get("metrics", report)
        for part in metric.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    baseline_score = metric_value(baseline)
    candidate_score = metric_value(candidate)
    evaluated = candidate.get(
        "documents_evaluated",
        candidate_metrics.get(
            "documents_evaluated",
            candidate_metrics.get("dataset", {}).get("validation_documents", 0),
        ),
    )
    if baseline_score is None or candidate_score is None:
        raise ValueError(f"Both reports must contain the {metric!r} metric.")

    passed = (
        int(evaluated) >= minimum_documents
        and float(candidate_score) >= float(baseline_score)
    )
    return {
        "passed": passed,
        "metric": metric,
        "baseline": float(baseline_score),
        "candidate": float(candidate_score),
        "documents_evaluated": int(evaluated),
        "minimum_documents": minimum_documents,
        "reason": "Candidate meets the non-regression gate." if passed
        else "Candidate regressed or evaluated too few documents.",
    }


def command_complete_run(
    evaluation_file: Path,
    database_file: Path,
) -> None:
    with evaluation_file.open(encoding="utf-8") as file:
        evaluation = json.load(file)
    if not evaluation.get("passed"):
        raise ValueError("Evaluation gate did not pass; examples remain unused.")

    examples = get_approved_learning_examples(
        unused_only=True,
        database_file=database_file,
    )
    run_id = start_learning_run(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        notes=json.dumps(evaluation, ensure_ascii=False),
        database_file=database_file,
    )
    mark_examples_used([example["id"] for example in examples], run_id, database_file)
    finish_learning_run(run_id, len(examples), len(examples), database_file)


def run_candidate_index_build(
    training_data: Path,
    output_index: Path,
) -> None:
    environment = os.environ.copy()
    environment["ANLP_INDEX_TRAIN_PATH"] = str(training_data.resolve())
    environment["ANLP_INDEX_OUTPUT_PATH"] = str(output_index.resolve())
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "notebooks" / "26_build_amharic_semantic_index.py")],
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )


def run_candidate_evaluation(
    index_path: Path,
    training_data: Path,
    validation_data: Path,
    results_directory: Path,
) -> None:
    environment = os.environ.copy()
    environment["ANLP_EVAL_INDEX_PATH"] = str(index_path.resolve())
    environment["ANLP_EVAL_TRAIN_PATH"] = str(training_data.resolve())
    environment["ANLP_EVAL_VALIDATION_PATH"] = str(validation_data.resolve())
    environment["ANLP_EVAL_RESULTS_DIR"] = str(results_directory.resolve())
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "notebooks" / "30_amharic_document_retrieval_evaluation.py")],
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )


# ============================================================================
# MARK EXAMPLES AS USED
# ============================================================================

def mark_examples_used(
    example_ids: list[int],
    training_run_id: int,
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Mark approved examples as consumed by a training run.
    """

    if not example_ids:
        return

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        placeholders = ",".join(
            "?" for _ in example_ids
        )

        query = f"""
            UPDATE learning_examples
            SET used_in_training = 1,
                training_run_id = ?
            WHERE id IN ({placeholders})
        """

        cursor.execute(
            query,
            [
                training_run_id,
                *example_ids,
            ],
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================================
# LEARNING RUNS
# ============================================================================

def start_learning_run(
    run_version: str,
    notes: Optional[str] = None,
    database_file: Path = DATABASE_FILE,
) -> int:
    """
    Create a new learning-run record.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO learning_runs (
                run_version,
                started_at,
                notes
            )
            VALUES (?, ?, ?)
            """,
            (
                run_version,
                utc_now(),
                notes,
            ),
        )

        run_id = cursor.lastrowid

        connection.commit()

        return int(run_id)

    finally:
        connection.close()


def finish_learning_run(
    run_id: int,
    approved_examples: int,
    used_examples: int,
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Complete a learning run.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE learning_runs
            SET completed_at = ?,
                approved_examples = ?,
                used_examples = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                approved_examples,
                used_examples,
                run_id,
            ),
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================================
# DATABASE STATISTICS
# ============================================================================

def print_database_status(database_file: Path = DATABASE_FILE) -> None:
    """
    Print progressive-learning database statistics.
    """

    connection = get_connection(database_file)

    try:
        cursor = connection.cursor()

        print()
        print("=" * 80)
        print("PROGRESSIVE LEARNING DATABASE")
        print("=" * 80)

        print(
            f"Database: {database_file}"
        )

        # --------------------------------------------------------------
        # Interactions
        # --------------------------------------------------------------

        cursor.execute(
            "SELECT COUNT(*) FROM interactions"
        )

        interaction_count = cursor.fetchone()[0]

        print(
            f"Interactions:       "
            f"{interaction_count:,}"
        )

        # --------------------------------------------------------------
        # Feedback
        # --------------------------------------------------------------

        cursor.execute(
            """
            SELECT status, COUNT(*)
            FROM feedback
            GROUP BY status
            """
        )

        feedback_counts = {
            row[0]: row[1]
            for row in cursor.fetchall()
        }

        print(
            f"Pending feedback:   "
            f"{feedback_counts.get(STATUS_PENDING, 0):,}"
        )

        print(
            f"Approved feedback:  "
            f"{feedback_counts.get(STATUS_APPROVED, 0):,}"
        )

        print(
            f"Rejected feedback:  "
            f"{feedback_counts.get(STATUS_REJECTED, 0):,}"
        )

        # --------------------------------------------------------------
        # Learning examples
        # --------------------------------------------------------------

        cursor.execute(
            """
            SELECT status, COUNT(*)
            FROM learning_examples
            GROUP BY status
            """
        )

        learning_counts = {
            row[0]: row[1]
            for row in cursor.fetchall()
        }

        print(
            f"Pending examples:   "
            f"{learning_counts.get(STATUS_PENDING, 0):,}"
        )

        print(
            f"Approved examples:  "
            f"{learning_counts.get(STATUS_APPROVED, 0):,}"
        )

        print(
            f"Rejected examples:  "
            f"{learning_counts.get(STATUS_REJECTED, 0):,}"
        )

        # --------------------------------------------------------------
        # Unused approved examples
        # --------------------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM learning_examples
            WHERE status = 'approved'
              AND used_in_training = 0
            """
        )

        unused = cursor.fetchone()[0]

        print(
            f"Ready for learning: "
            f"{unused:,}"
        )

        print("=" * 80)

    finally:
        connection.close()


# ============================================================================
# COMMAND LINE
# ============================================================================

def command_init(database_file: Path = DATABASE_FILE) -> None:

    initialize_database(database_file)

    print(
        f"[SUCCESS] Database initialized:\n"
        f"{database_file}"
    )


def command_status(database_file: Path = DATABASE_FILE) -> None:

    initialize_database(database_file)

    print_database_status(database_file)


def command_export(
    output: Optional[str],
    database_file: Path = DATABASE_FILE,
) -> None:

    initialize_database(database_file)

    if output:

        output_file = Path(output)

        if not output_file.is_absolute():
            output_file = (
                PROJECT_ROOT
                / output_file
            )

    else:

        output_file = (
            PROJECT_ROOT
            / "data"
            / "learning"
            / "approved_examples.jsonl"
        )

    count = export_approved_examples(
        output_file,
        database_file,
    )

    print(
        f"[SUCCESS] Exported "
        f"{count:,} approved examples."
    )

    print(
        f"[OUTPUT] {output_file}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build CLI.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Safe progressive-learning database "
            "for the Amharic NLP system."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # --------------------------------------------------------------
    # init
    # --------------------------------------------------------------

    subparsers.add_parser(
        "init",
        help="Initialize the database.",
    ).add_argument("--db", type=Path, default=DATABASE_FILE)

    # --------------------------------------------------------------
    # status
    # --------------------------------------------------------------

    subparsers.add_parser(
        "status",
        help="Show database statistics.",
    ).add_argument("--db", type=Path, default=DATABASE_FILE)

    # --------------------------------------------------------------
    # export
    # --------------------------------------------------------------

    export_parser = subparsers.add_parser(
        "export",
        help=(
            "Export approved unused examples "
            "to JSONL."
        ),
    )

    export_parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path.",
    )
    export_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    import_parser = subparsers.add_parser(
        "import-dashboard",
        help="Import dashboard feedback as pending/rejected learner feedback.",
    )
    import_parser.add_argument("--dashboard-db", type=Path, required=True)
    import_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    pending_parser = subparsers.add_parser(
        "pending",
        help="List pending feedback for human review.",
    )
    pending_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    for command, help_text in (
        ("approve", "Approve one imported feedback item after human review."),
        ("reject", "Reject one imported feedback item after human review."),
    ):
        review_parser = subparsers.add_parser(command, help=help_text)
        review_parser.add_argument("feedback_id", type=int)
        review_parser.add_argument("--reason", default="Manually reviewed.")
        review_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    stage_parser = subparsers.add_parser(
        "stage-data",
        help="Build a separate workbook from approved feedback examples.",
    )
    stage_parser.add_argument("--base", type=Path, required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    build_parser = subparsers.add_parser(
        "build-candidate-index",
        help="Build a semantic index from staged data into a non-live path.",
    )
    build_parser.add_argument("--train", type=Path, required=True)
    build_parser.add_argument("--output-index", type=Path, required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate-candidate",
        help="Evaluate a candidate index on a fixed held-out split.",
    )
    evaluate_parser.add_argument("--index", type=Path, required=True)
    evaluate_parser.add_argument("--train", type=Path, required=True)
    evaluate_parser.add_argument("--validation", type=Path, required=True)
    evaluate_parser.add_argument("--results", type=Path, required=True)

    gate_parser = subparsers.add_parser(
        "evaluate-gate",
        help="Compare baseline/candidate reports before consuming examples.",
    )
    gate_parser.add_argument("--baseline", type=Path, required=True)
    gate_parser.add_argument("--candidate", type=Path, required=True)
    gate_parser.add_argument("--metric", required=True)
    gate_parser.add_argument("--minimum-documents", type=int, default=100)
    gate_parser.add_argument("--output", type=Path, required=True)

    complete_parser = subparsers.add_parser(
        "complete-run",
        help="Mark examples used only when the supplied gate report passes.",
    )
    complete_parser.add_argument("--evaluation", type=Path, required=True)
    complete_parser.add_argument("--db", type=Path, default=DATABASE_FILE)

    return parser


# ============================================================================
# DEMO API
# ============================================================================

def example_usage() -> None:
    """
    Example integration with Script 27.

    Your real inference system should call these functions after prediction.
    """

    initialize_database()

    # --------------------------------------------------------------
    # 1. Model makes prediction.
    # --------------------------------------------------------------

    interaction_id = save_interaction(
        text=(
            "ሰው ሰራሽ አእምሮ "
            "በትምህርት ዘርፍ እየተጠቀሙበት ነው።"
        ),

        predicted_domain="Education",

        predicted_topics=[
            "Artificial Intelligence",
            "Education Technology",
        ],

        predicted_subtopics=[
            "Machine Learning",
        ],

        prediction_confidence=0.81,

        semantic_similarity=0.87,

        model_version=(
            "amharic_semantic_hybrid_v2"
        ),
    )

    print(
        f"[INFO] Interaction saved: "
        f"{interaction_id}"
    )

    # --------------------------------------------------------------
    # 2. User explicitly corrects the model.
    #
    # This is NOT the same as raw user input.
    # --------------------------------------------------------------

    feedback_id = submit_feedback(
        interaction_id=interaction_id,

        feedback_type=FEEDBACK_CORRECT,

        user_domain="Education",

        user_topics=[
            "Artificial Intelligence"
        ],

        user_subtopics=[
            "Educational Technology"
        ],

        comment=(
            "The domain is correct but "
            "the subtopic should be Educational Technology."
        ),
    )

    print(
        f"[INFO] Feedback saved: "
        f"{feedback_id}"
    )

    print_database_status()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = build_argument_parser()

    args = parser.parse_args()

    if args.command == "init":

        command_init(args.db)

    elif args.command == "status":

        command_status(args.db)

    elif args.command == "export":

        command_export(
            args.output,
            args.db,
        )

    elif args.command == "import-dashboard":
        print(json.dumps(
            import_dashboard_feedback(args.dashboard_db, args.db),
            ensure_ascii=False,
        ))

    elif args.command == "pending":
        initialize_database(args.db)
        print(json.dumps(get_pending_feedback(args.db), ensure_ascii=False, indent=2))

    elif args.command in {"approve", "reject"}:
        initialize_database(args.db)
        reviewer = approve_feedback if args.command == "approve" else reject_feedback
        reviewer(args.feedback_id, args.reason, args.db)

    elif args.command == "stage-data":
        initialize_database(args.db)
        report = stage_approved_training_data(args.base, args.output, args.db)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == "build-candidate-index":
        run_candidate_index_build(args.train, args.output_index)

    elif args.command == "evaluate-candidate":
        run_candidate_evaluation(
            args.index,
            args.train,
            args.validation,
            args.results,
        )

    elif args.command == "evaluate-gate":
        report = evaluate_gate(
            args.baseline,
            args.candidate,
            args.metric,
            args.minimum_documents,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(2)

    elif args.command == "complete-run":
        initialize_database(args.db)
        command_complete_run(args.evaluation, args.db)

    else:

        parser.print_help()

        print()
        print(
            "Example integration:"
        )

        print()

        example_usage()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

