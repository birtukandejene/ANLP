from pathlib import Path

from flask import jsonify, render_template, request

from .database import delete_analysis, feedback_progress, recent_analyses, save_analysis, save_feedback
from .services.keyword_extractor import ExistingKeywordExtractor
from .services.feedback_evidence import FeedbackEvidenceStore
from .services.nlp_service import NLPService
from .services.topic_extractor import HybridTopicExtractor


def register_routes(app):
    topic_extractor = HybridTopicExtractor(
        app.config["TOPIC_HYBRID_MODULE_PATH"],
        app.config["TOPIC_HYBRID_INDEX_PATH"],
        app.config["TOPIC_HYBRID_SUBTOPIC_MODEL_PATH"],
        app.config["TOPIC_TOP_K"],
    )
    service = NLPService(
        topic_extractor=topic_extractor,
        keyword_extractor=ExistingKeywordExtractor(
            app.config["KEYWORD_MODULE_PATH"],
            Path(app.root_path).parent / "data" / "raw" / "amharic_stopwords.txt",
            app.config["KEYWORD_TOP_K"],
        ),
        feedback_store=FeedbackEvidenceStore(
            app.config["DATABASE_PATH"],
            app.config.get("FEEDBACK_SIMILARITY_THRESHOLD", 0.68),
        ),
    )

    @app.get("/")
    def index():
        return render_template("welcome.html")

    @app.get("/workbench")
    def workbench():
        return render_template("index.html", max_input_chars=app.config["MAX_INPUT_CHARS"])

    @app.get("/about")
    def about():
        return render_template("about.html", max_input_chars=app.config["MAX_INPUT_CHARS"])

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return jsonify({"error": "Enter Amharic text to analyze."}), 400
        if len(text) > app.config["MAX_INPUT_CHARS"]:
            return jsonify({"error": f"Text exceeds {app.config['MAX_INPUT_CHARS']:,} characters."}), 400
        try:
            result = service.analyze(text)
            if result.get("valid"):
                embedding = result.pop("_input_embedding", None)
                analysis_id = save_analysis(
                    app.config["DATABASE_PATH"],
                    text,
                    result,
                    embedding=embedding,
                )
                result["analysis_id"] = analysis_id
            return jsonify(result)
        except Exception as exc:
            app.logger.exception("NLP analysis failed")
            return jsonify({"error": "The NLP backend could not complete this analysis.", "detail": str(exc)}), 503

    @app.get("/api/history")
    def history():
        return jsonify({"items": recent_analyses(app.config["DATABASE_PATH"])})

    @app.delete("/api/history/<int:analysis_id>")
    def delete_history_item(analysis_id):
        deleted = delete_analysis(app.config["DATABASE_PATH"], analysis_id)
        if not deleted:
            return jsonify({"error": "Analysis not found."}), 404
        return jsonify({"deleted": True, "analysis_id": analysis_id})

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True) or {}
        try:
            analysis_id = int(payload["analysis_id"])
            feedback_items = payload["feedback"]
            allowed_types = {"main_topic", "subtopic", "keyword"}
            if not isinstance(feedback_items, list) or not feedback_items:
                raise ValueError
            normalized = []
            for item in feedback_items:
                item_type = str(item["item_type"])
                item_text = str(item["item_text"]).strip()
                corrected_text = str(item.get("corrected_text", item_text)).strip()
                rating = int(item["rating"])
                if item_type not in allowed_types or not item_text or (not corrected_text and item_type != "keyword") or rating not in (-1, 1):
                    raise ValueError
                normalized.append({
                    "item_type": item_type,
                    "item_text": item_text,
                    "corrected_text": corrected_text,
                    "rating": rating,
                })
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "analysis_id and labeled predictions are required."}), 400
        save_feedback(app.config["DATABASE_PATH"], analysis_id, normalized)
        return jsonify({"saved": True, "count": len(normalized), "progress": feedback_progress(app.config["DATABASE_PATH"])})

    @app.get("/api/feedback/progress")
    def feedback_status():
        return jsonify(feedback_progress(app.config["DATABASE_PATH"]))
