import ast
import json
import sqlite3
from pathlib import Path

from app.database import init_db, save_analysis, save_feedback
from app.services.feedback_evidence import FeedbackEvidenceStore
from app.services.nlp_service import NLPService


def _save_label(database, text, embedding, item_type, item_text, rating, corrected=None):
    analysis_id = save_analysis(
        database,
        text,
        {"main_topic": "ትምህርት", "subtopics": [], "keywords": []},
        embedding=embedding,
    )
    save_feedback(
        database,
        analysis_id,
        [{
            "item_type": item_type,
            "item_text": item_text,
            "corrected_text": corrected or item_text,
            "rating": rating,
        }],
    )


def test_retrieval_filters_unrelated_feedback_and_aggregates_conflicts(tmp_path):
    database = tmp_path / "feedback.sqlite3"
    init_db(database)
    _save_label(database, "የትምህርት ምሳሌ አንድ።", [1.0, 0.0], "main_topic", "ትምህርት", 1)
    _save_label(database, "የትምህርት ምሳሌ ሁለት።", [0.98, 0.2], "main_topic", "ሳይንስ", 1)
    _save_label(database, "የስፖርት ምሳሌ።", [0.0, 1.0], "main_topic", "ስፖርት", 1)
    _save_label(database, "ተመሳሳይ ርዕስ።", [1.0, 0.0], "main_topic", "ሳይንስ", -1)

    evidence = FeedbackEvidenceStore(database, similarity_threshold=0.68).retrieve(
        [1.0, 0.0]
    )

    assert len(evidence["matches"]) == 3
    assert "ስፖርት" not in evidence["main_topic"]
    assert evidence["main_topic"]["ትምህርት"] > 0
    assert evidence["main_topic"]["ሳይንስ"] < 0
    assert all(-1.0 <= score <= 1.0 for score in evidence["main_topic"].values())


def test_retrieval_keeps_relevant_older_feedback_ahead_of_newer_unrelated_rows(tmp_path):
    database = tmp_path / "feedback.sqlite3"
    init_db(database)
    _save_label(database, "ቀድሞ የተመዘገበ ግቤት።", [1.0, 0.0], "main_topic", "ቋንቋ", 1)
    for index in range(4):
        _save_label(
            database,
            f"የተለየ አዲስ ግቤት {index}።",
            [0.0, 1.0],
            "main_topic",
            "ስፖርት",
            1,
        )

    evidence = FeedbackEvidenceStore(
        database,
        similarity_threshold=0.68,
        max_records=2,
    ).retrieve([1.0, 0.0])

    assert len(evidence["matches"]) == 1
    assert evidence["main_topic"]["ቋንቋ"] > 0
    assert "ስፖርት" not in evidence["main_topic"]


def test_missing_legacy_feedback_embeddings_are_backfilled_on_retrieval(tmp_path):
    database = tmp_path / "feedback.sqlite3"
    init_db(database)
    analysis_id = save_analysis(
        database,
        "የቀድሞ የትምህርት ግቤት።",
        {"main_topic": "ትምህርት"},
    )
    save_feedback(
        database,
        analysis_id,
        [{
            "item_type": "main_topic",
            "item_text": "ትምህርት",
            "corrected_text": "ሳይንስ",
            "rating": 1,
        }],
    )
    calls = []

    evidence = FeedbackEvidenceStore(database).retrieve(
        [1.0, 0.0],
        embed_text=lambda text: calls.append(text) or [1.0, 0.0],
    )

    assert calls == ["የቀድሞ የትምህርት ግቤት።"]
    assert evidence["main_topic"]["ሳይንስ"] > 0
    with sqlite3.connect(database) as connection:
        stored_embedding = connection.execute(
            "SELECT embedding_json FROM feedback WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()[0]
    assert json.loads(stored_embedding) == [1.0, 0.0]


def test_analysis_passes_retrieved_feedback_to_task_extractors(tmp_path):
    database = tmp_path / "feedback.sqlite3"
    init_db(database)
    _save_label(database, "ተመሳሳይ ርዕስ እና ቃላት።", [1.0, 0.0], "main_topic", "ቋንቋ", 1)
    _save_label(database, "ተመሳሳይ ርዕስ እና ቃላት።", [1.0, 0.0], "subtopic", "ቴክኖሎጂ", 1)
    _save_label(database, "ተመሳሳይ ርዕስ እና ቃላት።", [1.0, 0.0], "keyword", "አማርኛ", 1)

    class Detector:
        def detect(self, text):
            return {"is_amharic": True, "language": "am"}

    class TopicExtractor:
        def __init__(self):
            self.feedback = None

        def embed(self, text):
            return [1.0, 0.0]

        def extract(self, text, feedback_evidence=None, query_embedding=None):
            self.feedback = feedback_evidence
            assert query_embedding == [1.0, 0.0]
            return {"main_topic": "ቋንቋ", "subtopics": []}

    class KeywordExtractor:
        def __init__(self):
            self.feedback = None

        def extract(self, text, feedback_scores=None):
            self.feedback = feedback_scores
            return {"keywords": []}

    topic = TopicExtractor()
    keyword = KeywordExtractor()
    service = NLPService(
        topic,
        keyword,
        feedback_store=FeedbackEvidenceStore(database),
        detector=Detector(),
    )

    result = service.analyze("ተመሳሳይ ጽሑፍ ለትንተና።")

    assert topic.feedback["main_topic"]["ቋንቋ"] > 0
    assert topic.feedback["subtopic"]["ቴክኖሎጂ"] > 0
    assert keyword.feedback["አማርኛ"] > 0
    assert result["feedback_evidence_count"] == 3
    assert result["_input_embedding"] == [1.0, 0.0]


def test_feedback_only_subtopic_is_included_in_subtopic_evidence_candidates():
    module_path = Path(__file__).resolve().parents[1] / "notebooks" / "27_amharic_semantic_hybrid_inference.py"
    source = module_path.read_text(encoding="utf-8")
    syntax_tree = ast.parse(source)
    function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_subtopic_evidence"
    )
    namespace = {
        "normalize_scores": lambda scores: scores,
        "SUBTOPIC_FEEDBACK_WEIGHT": 0.2,
        "SUBTOPIC_NEIGHBOR_WEIGHT": 0.0,
        "SUBTOPIC_CENTROID_WEIGHT": 0.0,
        "SUBTOPIC_SUPERVISED_WEIGHT": 0.0,
        "SUBTOPIC_COOCCURRENCE_WEIGHT": 0.0,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"),
        namespace,
    )

    evidence = namespace["get_subtopic_evidence"](
        None,
        [],
        "main topic",
        {},
        {},
        feedback_scores={"corrected subtopic": 1.0},
    )

    assert evidence[4]["corrected subtopic"] > 0
