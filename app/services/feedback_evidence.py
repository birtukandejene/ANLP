import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


class FeedbackEvidenceStore:
    """Retrieve accumulated feedback as bounded, relevance-weighted evidence."""

    def __init__(self, database_path, similarity_threshold=0.68, max_records=200):
        self.database_path = Path(database_path)
        self.similarity_threshold = similarity_threshold
        self.max_records = max_records

    def retrieve(self, query_embedding, embed_text=None):
        query = self._normalize(query_embedding)
        if query is None or not self.database_path.exists():
            return self._empty_evidence()

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT id, analysis_id, item_type, item_text, corrected_text,
                          rating, input_text, prediction_json, embedding_json
                   FROM feedback
                   WHERE item_type IN ('main_topic', 'subtopic', 'keyword')
                   ORDER BY id DESC"""
            ).fetchall()

            missing_by_analysis = {}
            for row in rows:
                if row["embedding_json"]:
                    continue
                if row["analysis_id"] not in missing_by_analysis:
                    missing_by_analysis[row["analysis_id"]] = row["input_text"]

            if missing_by_analysis and embed_text:
                for analysis_id, text in missing_by_analysis.items():
                    try:
                        vector = self._normalize(embed_text(text))
                    except Exception:
                        vector = None
                    if vector is None:
                        continue
                    encoded = json.dumps(vector.tolist())
                    connection.execute(
                        "UPDATE analyses SET embedding_json = ? WHERE id = ?",
                        (encoded, analysis_id),
                    )
                    connection.execute(
                        "UPDATE feedback SET embedding_json = ? WHERE analysis_id = ?",
                        (encoded, analysis_id),
                    )
                rows = connection.execute(
                    """SELECT id, analysis_id, item_type, item_text, corrected_text,
                              rating, input_text, prediction_json, embedding_json
                       FROM feedback
                       WHERE item_type IN ('main_topic', 'subtopic', 'keyword')
                              ORDER BY id DESC"""
                ).fetchall()

        vote_totals = {
            "main_topic": defaultdict(float),
            "subtopic": defaultdict(float),
            "keyword": defaultdict(float),
        }
        vote_mass = defaultdict(float)
        relevant_rows = []

        for row in rows:
            stored = self._normalize(json.loads(row["embedding_json"] or "null"))
            if stored is None or stored.shape != query.shape:
                continue
            similarity = float(np.dot(query, stored))
            if similarity < self.similarity_threshold:
                continue

            relevant_rows.append((similarity, row))

        relevant_rows.sort(key=lambda item: item[0], reverse=True)
        relevant_rows = relevant_rows[:self.max_records]
        matches = []

        for similarity, row in relevant_rows:
            weight = max(0.0, similarity) * 0.70
            item_type = row["item_type"]
            original = str(row["item_text"] or "").strip()
            corrected = str(row["corrected_text"] or "").strip()
            changed = bool(corrected and corrected.casefold() != original.casefold())
            rating = int(row["rating"])

            if rating > 0:
                target = corrected if changed else original
                self._add_vote(vote_totals[item_type], vote_mass, item_type, target, weight)
                if changed:
                    self._add_vote(vote_totals[item_type], vote_mass, item_type, original, -weight)
            else:
                self._add_vote(vote_totals[item_type], vote_mass, item_type, original, -weight)
                if changed:
                    self._add_vote(vote_totals[item_type], vote_mass, item_type, corrected, weight)

            matches.append({"feedback_id": row["id"], "similarity": similarity})

        evidence = self._empty_evidence()
        for item_type, totals in vote_totals.items():
            denominator = max(vote_mass[item_type], 1.0)
            evidence[item_type] = {
                label: max(-1.0, min(1.0, score / denominator))
                for label, score in totals.items()
                if label
            }
        evidence["matches"] = matches
        return evidence

    @staticmethod
    def _add_vote(scores, masses, item_type, label, weight):
        if not label:
            return
        scores[label] += weight
        masses[item_type] += abs(weight)

    @staticmethod
    def _normalize(vector):
        if vector is None:
            return None
        try:
            values = np.asarray(vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(values))
        if not norm or not np.isfinite(norm) or not np.all(np.isfinite(values)):
            return None
        return values / norm

    @staticmethod
    def _empty_evidence():
        return {"main_topic": {}, "subtopic": {}, "keyword": {}, "matches": []}
