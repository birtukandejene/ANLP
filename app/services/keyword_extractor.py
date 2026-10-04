import importlib.util
from pathlib import Path


class ExistingKeywordExtractor:
    """Lazy adapter around the project's existing Rasyosef extractor."""

    def __init__(self, module_path, stopwords_path, top_k=10):
        self.module_path = Path(module_path)
        self.stopwords_path = Path(stopwords_path)
        self.top_k = top_k
        self.extractor = None
        self.module = None

    def _load(self):
        if self.extractor is not None:
            return
        spec = importlib.util.spec_from_file_location("project_keyword_extractor", self.module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stopwords = module.load_stopwords(self.stopwords_path)
        training = module.load_annotated_dataset(module.TRAIN_PATH)
        lexicon = module.build_annotation_lexicon(training)
        self.module = module
        self.extractor = module.RasyosefKeywordExtractor(
            model_name=module.MODEL_NAME,
            stopwords=stopwords,
            annotation_lexicon=lexicon,
        )

    def extract(self, text, feedback_scores=None):
        self._load()
        normalized_feedback = {}
        for keyword, score in (feedback_scores or {}).items():
            for label in str(keyword).split('|'):
                normalized_label = label.strip()
                if not normalized_label or normalized_label.casefold() == "review all keywords":
                    continue
                normalized_feedback[normalized_label.casefold()] = float(score)
        result = self.extractor.extract_keywords(
            text,
            self.top_k,
            feedback_scores=normalized_feedback,
        )
        keywords = []
        for item in result.get("details", []):
            keywords.append({
                "text": item["keyword"],
                "score": item.get("final_score", 0.0),
                "semantic_score": item.get("semantic_score", 0.0),
                "confidence": item.get("annotation_prior", 0.0),
            })
        return {"keywords": keywords, "backend": "rasyosef_keyword_extractor"}
