import importlib.util
import json
from pathlib import Path

from app.database import delete_analysis, init_db, save_analysis, save_feedback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "notebooks" / "31_progressive_learning.py"
SPEC = importlib.util.spec_from_file_location("progressive_learning", MODULE_PATH)
progressive_learning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progressive_learning)


def test_feedback_is_append_only_and_survives_analysis_deletion(tmp_path):
    database = tmp_path / "feedback.sqlite3"
    init_db(database)
    analysis_id = save_analysis(
        database,
        "የተለመደ የጽሑፍ ምሳሌ።",
        {"main_topic": "ትምህርት", "keywords": [{"text": "ትምህርት"}]},
        embedding=[1.0, 0.0],
    )

    for corrected_topic in ("ትምህርት", "ሳይንስ"):
        save_feedback(
            database,
            analysis_id,
            [{
                "item_type": "main_topic",
                "item_text": "ትምህርት",
                "corrected_text": corrected_topic,
                "rating": 1,
            }],
        )

    assert delete_analysis(database, analysis_id)
    import sqlite3
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT input_text, prediction_json, embedding_json, corrected_text FROM feedback ORDER BY id"
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "የተለመደ የጽሑፍ ምሳሌ።"
    assert '"main_topic": "ትምህርት"' in rows[0][1]
    assert rows[0][2] == "[1.0, 0.0]"
    assert [row[3] for row in rows] == ["ትምህርት", "ሳይንስ"]


def test_keyword_feedback_is_a_temporary_ranking_signal():
    keyword_spec = importlib.util.spec_from_file_location(
        "keyword_extraction_feedback_evidence",
        PROJECT_ROOT / "notebooks" / "27_keyword_extraction.py",
    )
    keyword_module = importlib.util.module_from_spec(keyword_spec)
    keyword_spec.loader.exec_module(keyword_module)
    extractor = object.__new__(keyword_module.RasyosefKeywordExtractor)
    extractor.annotation_lexicon = {}
    extractor.encode = lambda texts: keyword_module.np.tile(
        keyword_module.np.array([1.0, 0.0]),
        (len(texts), 1),
    )

    without_feedback = extractor.rank_candidates(
        "አማርኛ ቋንቋ",
        ["አማርኛ", "ቋንቋ"],
        [],
    )
    with_feedback = extractor.rank_candidates(
        "አማርኛ ቋንቋ",
        ["አማርኛ", "ቋንቋ"],
        [],
        feedback_scores={"ቋንቋ": 1.0},
    )

    assert without_feedback[0]["final_score"] == without_feedback[1]["final_score"]
    assert with_feedback[0]["keyword"] == "ቋንቋ"
    assert extractor.annotation_lexicon == {}


def test_evaluation_gate_requires_non_regression_and_enough_documents(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"metrics": {"micro_f1": 0.7}}), encoding="utf-8")
    candidate.write_text(
        json.dumps({"metrics": {"micro_f1": 0.72}, "documents_evaluated": 120}),
        encoding="utf-8",
    )

    decision = progressive_learning.evaluate_gate(
        baseline,
        candidate,
        "micro_f1",
        minimum_documents=100,
    )

    assert decision["passed"] is True
    candidate.write_text(
        json.dumps({"metrics": {"micro_f1": 0.69}, "documents_evaluated": 120}),
        encoding="utf-8",
    )
    assert not progressive_learning.evaluate_gate(
        baseline,
        candidate,
        "micro_f1",
        minimum_documents=100,
    )["passed"]


