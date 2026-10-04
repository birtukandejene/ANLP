import re

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[\u1200-\u137F]+|[A-Za-z]+")


class AmharicLanguageDetector:
    """Conservative detector: symbols/numbers are neutral; language words decide."""

    def detect(self, text):
        words = WORD_RE.findall(text or "")
        ethiopic_words = [word for word in words if ETHIOPIC_RE.search(word)]
        latin_words = [word for word in words if LATIN_RE.search(word)]
        natural_words = len(ethiopic_words) + len(latin_words)
        ethiopic_ratio = len(ethiopic_words) / natural_words if natural_words else 0.0
        is_amharic = bool(ethiopic_words) and ethiopic_ratio >= 0.55
        return {
            "language": "am" if is_amharic else "unknown",
            "is_amharic": is_amharic,
            "confidence": round(ethiopic_ratio, 3),
            "ethiopic_words": len(ethiopic_words),
            "latin_words": len(latin_words),
        }
