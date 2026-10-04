import importlib.util
import os
import re
from pathlib import Path


def normalize_topic_text(text):
    text = str(text or "")

    def replace_punctuation(match):
        start, end = match.span()
        if start > 0 and end < len(text) and text[start - 1].isalpha() and text[end].isalpha():
            return match.group()
        return " "

    text = re.sub(r"[፡።፣፤፥፦፧፨\.,!?\(\)\[\]\{\}\"'“”:/\\=+%#@*&^~`<>|•]+", replace_punctuation, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HybridTopicExtractor:
    """Adapter for the project's open-topic hybrid inference script."""

    def __init__(self, module_path, index_path, subtopic_model_path, top_k=5):
        self.module_path = Path(module_path)
        self.index_path = Path(index_path)
        self.subtopic_model_path = Path(subtopic_model_path)
        self.top_k = top_k
        self.module = None

    def _load(self):
        if self.module is not None:
            return self.module
        required = [self.module_path, self.index_path, self.subtopic_model_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Hybrid topic backend requires missing files: " + ", ".join(missing)
            )
        spec = importlib.util.spec_from_file_location("project_hybrid_topic_inference", self.module_path)
        module = importlib.util.module_from_spec(spec)
        previous_mode = os.environ.get("AMHARIC_HYBRID_NON_INTERACTIVE")
        os.environ["AMHARIC_HYBRID_NON_INTERACTIVE"] = "1"
        try:
            spec.loader.exec_module(module)
        finally:
            if previous_mode is None:
                os.environ.pop("AMHARIC_HYBRID_NON_INTERACTIVE", None)
            else:
                os.environ["AMHARIC_HYBRID_NON_INTERACTIVE"] = previous_mode
        self.module = module
        return module

    def embed(self, text):
        module = self._load()
        return module.encode_query(normalize_topic_text(text))

    def extract(self, text, feedback_evidence=None, query_embedding=None):
        text = normalize_topic_text(text)
        result = self._load().predict(
            text,
            feedback_evidence=feedback_evidence,
            query_embedding=query_embedding,
        )
        if not result:
            return {"main_topic": "", "subtopics": [], "topic_candidates": [], "confidence": 0.0, "backend": "27_amharic_semantic_hybrid_inference"}
        candidates = sorted(result.get("final_main_topic_scores", {}).items(), key=lambda item: item[1], reverse=True)
        main_score = float(result.get("final_main_topic_score", 0.0))
        return {
            "main_topic": result.get("final_main_topic", ""),
            "subtopics": [{"text": label, "score": round(float(result.get("combined_subtopic_scores", {}).get(label, 0.0)), 4)} for label in result.get("final_subtopics", [])[:self.top_k]],
            "topic_candidates": [{"text": label, "score": round(float(score), 4)} for label, score in candidates[:self.top_k]],
            "domain": "",
            "confidence": round(main_score, 4),
            "confidence_label": result.get("confidence", ""),
            "backend": "27_amharic_semantic_hybrid_inference",
        }
