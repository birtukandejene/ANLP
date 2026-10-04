from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


class AppConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
    DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", BASE_DIR / "data" / "dashboard.sqlite3"))
    LEARNER_DATABASE_PATH = BASE_DIR / "data" / "progressive_learning.db"
    TOPIC_HYBRID_MODULE_PATH = BASE_DIR / "notebooks" / "27_amharic_semantic_hybrid_inference.py"
    TOPIC_HYBRID_INDEX_PATH = BASE_DIR / "models" / "amharic_semantic_index.joblib"
    TOPIC_HYBRID_SUBTOPIC_MODEL_PATH = BASE_DIR / "models" / "final_domain_aware_subtopic_model.joblib"
    KEYWORD_MODULE_PATH = BASE_DIR / "notebooks" / "27_keyword_extraction.py"
    KEYWORD_TOP_K = 10
    TOPIC_TOP_K = 5
    FEEDBACK_SIMILARITY_THRESHOLD = 0.68
    MAX_INPUT_CHARS = 12000
