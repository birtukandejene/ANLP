import importlib.util
from pathlib import Path

from app.services.keyword_extractor import ExistingKeywordExtractor


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "27_keyword_extraction.py"
SPEC = importlib.util.spec_from_file_location("keyword_extraction", MODULE_PATH)
keyword_extraction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(keyword_extraction)


def test_year_abbreviation_does_not_create_single_letter_keyword():
    text = "በ2025 ዓ.ም እድገት አሳይቷል። በ2026 ዓ/ም ይቀጥላል።"

    tokens = keyword_extraction.tokenize_amharic(
        keyword_extraction.normalize_text(text)
    )

    assert "ዓ" not in tokens
    assert "ም" not in tokens


def test_standalone_amharic_letter_is_not_removed_as_abbreviation():
    assert keyword_extraction.normalize_text("ዓ ም") == "ዓ ም"


def test_keyword_selection_uses_top_semantic_candidates_with_60_percent_limit():
    extractor = object.__new__(keyword_extraction.RasyosefKeywordExtractor)
    ranked_candidates = [
        {"keyword": word, "final_score": score}
        for word, score in [
            ("አንድ", 0.99),
            ("ሁለት", 0.90),
            ("ሶስት", 0.80),
            ("አራት", 0.70),
            ("አምስት", 0.60),
            ("ስድስት", 0.50),
            ("ሰባት", 0.40),
        ]
    ]

    selected = extractor.select_keywords(
        ranked_candidates,
        top_k=10,
        input_word_count=10,
    )

    assert [item["keyword"] for item in selected] == [
        "አንድ",
        "ሁለት",
        "ሶስት",
        "አራት",
        "አምስት",
        "ስድስት",
    ]


def test_positive_feedback_keywords_join_candidate_pool_for_future_ranking():
    extractor = object.__new__(keyword_extraction.RasyosefKeywordExtractor)
    captured_candidates = {}
    extractor.build_candidates = lambda text: (["የጽሑፍ ቃል"], [], ["የጽሑፍ ቃል"])

    def capture_candidates(text, candidates, phrases, feedback_scores):
        captured_candidates["items"] = candidates
        return []

    extractor.rank_candidates = capture_candidates
    extractor.select_keywords = lambda ranked, top_k, input_word_count: []

    extractor.extract_keywords(
        "የጽሑፍ ቃል",
        top_k=5,
        feedback_scores={"ተመሳሳይ ጽንሰ-ሀሳብ": 0.8, "ውድቅ ቃል": -0.4},
    )

    assert "ተመሳሳይ ጽንሰ-ሀሳብ" in captured_candidates["items"]
    assert "ውድቅ ቃል" not in captured_candidates["items"]


def test_feedback_splits_saved_keyword_lists_and_ignores_review_placeholder():
    captured_feedback = {}

    class StubExtractor:
        def extract_keywords(self, text, top_k, feedback_scores=None):
            captured_feedback.update(feedback_scores or {})
            return {"details": []}

    adapter = ExistingKeywordExtractor("unused", "unused")
    adapter.extractor = StubExtractor()

    adapter.extract(
        "የጽሑፍ ቃል",
        feedback_scores={
            "ቃል አንድ | ቃል ሁለት": 0.8,
            "Review all keywords": 0.7,
        },
    )

    assert captured_feedback == {
        "ቃል አንድ".casefold(): 0.8,
        "ቃል ሁለት".casefold(): 0.8,
    }