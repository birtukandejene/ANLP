from ..language_detector import AmharicLanguageDetector


class NLPService:
    def __init__(self, topic_extractor, keyword_extractor, detector=None, feedback_store=None):
        self.topic_extractor = topic_extractor
        self.keyword_extractor = keyword_extractor
        self.feedback_store = feedback_store
        self.detector = detector or AmharicLanguageDetector()

    def analyze(self, text):
        detection = self.detector.detect(text)
        if not detection["is_amharic"]:
            return {
                "language": detection["language"],
                "language_detection": detection,
                "valid": False,
                "error": "Please enter Amharic language.",
            }

        query_embedding = self.topic_extractor.embed(text)
        feedback_evidence = (
            self.feedback_store.retrieve(
                query_embedding,
                embed_text=self.topic_extractor.embed,
            )
            if self.feedback_store
            else {"main_topic": {}, "subtopic": {}, "keyword": {}, "matches": []}
        )
        topic_result = self.topic_extractor.extract(
            text,
            feedback_evidence=feedback_evidence,
            query_embedding=query_embedding,
        )
        keyword_result = self.keyword_extractor.extract(
            text,
            feedback_scores=feedback_evidence["keyword"],
        )
        return {
            "language": "am",
            "language_detection": detection,
            "valid": True,
            "main_topic": topic_result.get("main_topic", ""),
            "subtopics": topic_result.get("subtopics", []),
            "topic_candidates": topic_result.get("topic_candidates", []),
            "domain": topic_result.get("domain", ""),
            "topic_confidence": topic_result.get("confidence", 0.0),
            "keywords": keyword_result.get("keywords", []),
            "feedback_evidence_count": len(feedback_evidence["matches"]),
            "_input_embedding": (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            ),
            "backends": {
                "topic": topic_result.get("backend", ""),
                "keyword": keyword_result.get("backend", ""),
            },
        }
