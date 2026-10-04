"""
================================================================================
AMHARIC SEMANTIC KEYWORD EXTRACTION
================================================================================

Model:
    rasyosef/embedding-amharic-base

Rules:
    1. Keywords MUST come exactly from the input text.
    2. Stopwords are excluded.
    3. Single words are the default candidates.
    4. Meaningful known paired phrases can remain together.
    5. Arbitrary n-grams are NOT generated.
    6. The entire input is NEVER returned as a keyword.
    7. If a phrase is selected, its component words are not returned separately.
    8. Rasyosef embeddings are used for semantic ranking.

Stopwords:
    data/raw/amharic_stopwords.txt

Run:
    python notebooks/27_keyword_extraction.py
================================================================================
"""

import json
import re
import sys
import time
import ast
import os
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = str(BASE_DIR / "models" / "embedding-amharic-base")

STOPWORDS_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "amharic_stopwords.txt"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_FILE = (
    RESULTS_DIR
    / "rasyosef_keyword_test_results.json"
)

TRAIN_PATH = BASE_DIR / "data" / "train" / "train_processed.xlsx"

VALIDATION_PATH = (
    BASE_DIR / "data" / "train" / "validation_processed.xlsx"
)

TEST_PATH = BASE_DIR / "data" / "train" / "test_processed.xlsx"

METRICS_FILE = RESULTS_DIR / "rasyosef_keyword_metrics.json"

ANNOTATED_KEYWORD_MIN_FREQUENCY = 1

DEFAULT_TOP_K = 10


# ==============================================================================
# MEANINGFUL AMHARIC PHRASE LEXICON
# ==============================================================================
#
# IMPORTANT:
#
# This is NOT an n-gram generator.
#
# These are known or meaningful phrases that should be treated as one concept.
#
# You can add more phrases as your project grows.
#
# A phrase is only used if it appears EXACTLY in the input text.
#
# ==============================================================================

PHRASE_LEXICON = {

    # --------------------------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------------------------

    "አዲስ አበባ",
    "ኒው ዮርክ",
    "ኒው ዮርክ ሲቲ",

    "ደቡብ አፍሪካ",
    "ሰሜን አሜሪካ",
    "ደቡብ አሜሪካ",

    "ምስራቅ አፍሪካ",
    "ምዕራብ አፍሪካ",
    "ሰሜን አፍሪካ",

    "አዲስ አበባ",
    "ኪነ ጥበብ",
    "ስነ ጥበብ",
    "ስነ ህይወት",
    "ስነ ምግባር",
    "አይነ አፋር",
    "ልብ ወለድ",
    # --------------------------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------------------------

    "ትምህርት ቤት",
    "ከፍተኛ ትምህርት",
    "ሰው ሰራሽ አእምሮ",
    
    # --------------------------------------------------------------------------

  
}


# ==============================================================================
# TEXT NORMALIZATION
# ==============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize Amharic text.

    This function does NOT stem words.

    It only:
        - removes extra spaces
        - normalizes whitespace
        - strips surrounding spaces
    """

    text = str(text)
    text = re.sub(
        r"(?<![\u1200-\u137F])ዓ[./]ም(?![\u1200-\u137F])",
        " ",
        text,
    )
    text = re.sub(r"[፡።፣፤፥፦፧፨\.,!?\(\)\[\]\{\}\"'“”:/\\=+%#@*&^~`<>|•]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def strip_keyword_punctuation(value: str) -> str:
    """Remove punctuation while preserving actual word content."""
    text = str(value or "")
    text = re.sub(r"[^\u1200-\u137F0-9A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_keyword_candidate(value: str) -> bool:
    """Return True only for text that contains real word characters, not punctuation-only junk."""
    cleaned = strip_keyword_punctuation(value)
    if not cleaned:
        return False
    if re.fullmatch(r"[\s\W_]+", cleaned, flags=re.UNICODE):
        return False
    return bool(re.search(r"[\u1200-\u137F0-9A-Za-z]", cleaned))


def parse_annotation_list(value) -> List[str]:
    """Parse the keyword annotation formats used by the train splits."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = normalize_text(value)
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                values = parsed if isinstance(parsed, (list, tuple, set)) else [text]
            except (ValueError, SyntaxError):
                values = re.split(r"\||;|,|\n", text)
        else:
            values = re.split(r"\||;|,|\n", text)

    result = []
    seen = set()
    for item in values:
        keyword = normalize_text(item)
        keyword = strip_keyword_punctuation(keyword)
        key = keyword.casefold()
        if keyword and key not in seen:
            result.append(keyword)
            seen.add(key)
    return result


def load_annotated_dataset(path: Path) -> pd.DataFrame:
    """Load one processed split and retain rows with keyword annotations."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_excel(path, engine="openpyxl")
    required = {"text", "keywords"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    dataframe = dataframe[["text", "keywords"]].copy()
    dataframe["text"] = dataframe["text"].apply(normalize_text)
    dataframe["keyword_list"] = dataframe["keywords"].apply(parse_annotation_list)
    dataframe = dataframe[
        (dataframe["text"].str.len() > 0)
        & dataframe["keyword_list"].map(bool)
    ].reset_index(drop=True)
    return dataframe


def build_annotation_lexicon(dataframe: pd.DataFrame) -> Dict[str, float]:
    """Build a frequency prior from training annotations only."""

    counts = Counter(
        keyword.casefold()
        for keywords in dataframe["keyword_list"]
        for keyword in keywords
    )
    counts = {
        keyword: count
        for keyword, count in counts.items()
        if count >= ANNOTATED_KEYWORD_MIN_FREQUENCY
    }
    maximum = max(counts.values(), default=1)
    return {keyword: count / maximum for keyword, count in counts.items()}


# ==============================================================================
# TOKENIZATION
# ==============================================================================

def tokenize_amharic(text: str) -> List[str]:
    """
    Extract words from Amharic text.

    The tokenizer preserves Ethiopic characters.

    Examples:

        አዲስ አበባ
        -> ["አዲስ", "አበባ"]

        በኢትዮጵያ፣ ግብርና
        -> ["በኢትዮጵያ", "ግብርና"]
    """

    pattern = r"[\u1200-\u137F]+"

    tokens = re.findall(
        pattern,
        text
    )

    return tokens


# ==============================================================================
# STOPWORD LOADING
# ==============================================================================

def load_stopwords(path: Path) -> set:
    """
    Load Amharic stopwords.

    Supports:
        one stopword per line
    """

    print()

    print("=" * 100)
    print("LOADING STOPWORDS")
    print("=" * 100)

    print()
    print("Path:")
    print(path)

    print()

    if not path.exists():

        print(
            f"ERROR: Stopwords file not found:\n{path}"
        )

        sys.exit(1)

    stopwords = set()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            word = normalize_text(line)

            if word:
                stopwords.add(word)

    print(
        f"Loaded {len(stopwords)} stopwords."
    )

    return stopwords


# ==============================================================================
# PHRASE DETECTION
# ==============================================================================

def find_exact_phrases(
    text: str
) -> List[str]:
    """
    Find meaningful phrases from PHRASE_LEXICON.

    IMPORTANT:

    We DO NOT generate n-grams.

    We only check whether a phrase from the
    predefined phrase lexicon occurs in the input.

    Example:

        Input:
            "ትምህርት ቤት በአዲስ አበባ"

        Detected:
            "ትምህርት ቤት"
            "አዲስ አበባ"
    """

    normalized_text = normalize_text(text)

    detected = []

    for phrase in PHRASE_LEXICON:

        normalized_phrase = normalize_text(
            phrase
        )

        pattern = (
            r"(?<!\S)"
            + re.escape(normalized_phrase)
            + r"(?!\S)"
        )

        if re.search(
            pattern,
            normalized_text
        ):

            detected.append(
                normalized_phrase
            )

    detected = sorted(
        detected,
        key=lambda x: len(x.split()),
        reverse=True
    )

    return detected


# ==============================================================================
# FIND PHRASE POSITIONS
# ==============================================================================

def get_phrase_word_positions(
    tokens: List[str],
    phrase: str
) -> List[int]:
    """
    Find the positions of a phrase inside tokenized input.

    Example:

        tokens:
            ["ትምህርት", "ቤት", "በ", "አዲስ", "አበባ"]

        phrase:
            "ትምህርት ቤት"

        returns:
            [0, 1]
    """

    phrase_tokens = tokenize_amharic(
        phrase
    )

    positions = []

    phrase_length = len(
        phrase_tokens
    )

    for i in range(
        len(tokens) - phrase_length + 1
    ):

        segment = (
            tokens[
                i:
                i + phrase_length
            ]
        )

        if segment == phrase_tokens:

            positions.extend(
                range(
                    i,
                    i + phrase_length
                )
            )

    return positions


# ==============================================================================
# CHECK STOPWORD-ONLY PHRASE
# ==============================================================================

def is_stopword_only_phrase(
    phrase: str,
    stopwords: set
) -> bool:
    """
    Return True if ALL words in a phrase
    are stopwords.
    """

    words = tokenize_amharic(
        phrase
    )

    if not words:
        return True

    return all(
        word in stopwords
        for word in words
    )


# ==============================================================================
# COSINE SIMILARITY
# ==============================================================================

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate cosine similarity.
    """

    denominator = (
        np.linalg.norm(vector_a)
        *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        /
        denominator
    )


# ==============================================================================
# RASYosef KEYWORD EXTRACTOR
# ==============================================================================

class RasyosefKeywordExtractor:

    def __init__(
        self,
        model_name: str,
        stopwords: set,
        annotation_lexicon: Dict[str, float] = None
    ):

        self.model_name = model_name

        self.stopwords = stopwords

        self.annotation_lexicon = annotation_lexicon or {}

        self.annotation_phrases = {
            keyword
            for keyword in self.annotation_lexicon
            if len(keyword.split()) > 1
        }

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print()

        print("=" * 100)
        print(
            "INITIALIZING RASYosef KEYWORD EXTRACTOR"
        )
        print("=" * 100)

        print()

        print(
            f"Model       : {self.model_name}"
        )

        print(
            f"Stopwords   : {STOPWORDS_PATH}"
        )

        print(
            f"Device      : {self.device}"
        )

        print()

        print(
            "Loading Rasyosef embedding model..."
        )

        start_time = time.time()

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        loading_time = (
            time.time()
            -
            start_time
        )

        print()

        print(
            f"Model loaded in {loading_time:.2f} seconds."
        )

        embedding_dimension = (
            self.get_embedding_dimension()
        )

        print(
            f"Embedding dimension: "
            f"{embedding_dimension}"
        )

        print()

        print(
            "Rasyosef keyword extractor is ready."
        )


    # ==========================================================================
    # GET EMBEDDING DIMENSION
    # ==========================================================================

    def get_embedding_dimension(
        self
    ) -> int:

        """
        Get embedding dimension.

        Supports newer and older versions
        of sentence-transformers.
        """

        if hasattr(
            self.model,
            "get_embedding_dimension"
        ):

            return (
                self.model
                .get_embedding_dimension()
            )

        if hasattr(
            self.model,
            "get_sentence_embedding_dimension"
        ):

            return (
                self.model
                .get_sentence_embedding_dimension()
            )

        return 768


    # ==========================================================================
    # ENCODE TEXT
    # ==========================================================================

    def encode(
        self,
        texts: List[str]
    ) -> np.ndarray:

        """
        Create normalized embeddings.
        """

        embeddings = self.model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        return embeddings


    # ==========================================================================
    # CREATE SINGLE WORD CANDIDATES
    # ==========================================================================

    def get_single_word_candidates(
        self,
        text: str,
        phrase_words_positions: set
    ) -> List[str]:

        """
        Create single-word candidates.

        Rules:

            - exact words from input
            - stopwords excluded
            - duplicates removed
            - words belonging to meaningful
              detected phrases are excluded

        Example:

            Input:
                ትምህርት ቤት በአዲስ አበባ

            Detected phrases:
                ትምህርት ቤት
                አዲስ አበባ

            Single word candidates:
                none from those phrase components

        This prevents:

            ትምህርት
            ቤት
            አዲስ
            አበባ

        from competing with the complete phrases.
        """

        tokens = tokenize_amharic(
            text
        )

        candidates = []

        seen = set()

        for index, word in enumerate(
            tokens
        ):

            # --------------------------------------------------------------
            # Skip words that belong to
            # detected meaningful phrases.
            # --------------------------------------------------------------

            if (
                index
                in phrase_words_positions
            ):

                continue

            if not is_valid_keyword_candidate(word):
                continue

            # --------------------------------------------------------------
            # Skip stopwords
            # --------------------------------------------------------------

            if word in self.stopwords:

                continue

            # --------------------------------------------------------------
            # Skip duplicates
            # --------------------------------------------------------------

            if word in seen:

                continue

            # --------------------------------------------------------------
            # Add candidate
            # --------------------------------------------------------------

            seen.add(word)

            candidates.append(word)

        return candidates


    # ==========================================================================
    # BUILD CANDIDATES
    # ==========================================================================

    def build_candidates(
        self,
        text: str
    ) -> Tuple[
        List[str],
        List[str],
        List[str]
    ]:

        """
        Build candidates.

        Candidate types:

            1. Meaningful detected phrases
            2. Single words

        We DO NOT create arbitrary bigrams
        or trigrams.
        """

        tokens = tokenize_amharic(
            text
        )

        # ------------------------------------------------------------------
        # Detect meaningful phrases
        # ------------------------------------------------------------------

        detected_phrases = find_exact_phrases(text)

        normalized_text = normalize_text(text)
        for phrase in sorted(self.annotation_phrases, key=len, reverse=True):
            if re.search(
                r"(?<!\S)" + re.escape(phrase) + r"(?!\S)",
                normalized_text,
            ):
                if phrase not in detected_phrases:
                    detected_phrases.append(phrase)

        # ------------------------------------------------------------------
        # Remove stopword-only phrases
        # ------------------------------------------------------------------

        valid_phrases = []

        for phrase in detected_phrases:

            if not is_stopword_only_phrase(
                phrase,
                self.stopwords
            ):

                valid_phrases.append(
                    phrase
                )

        # ------------------------------------------------------------------
        # Find positions occupied
        # by meaningful phrases
        # ------------------------------------------------------------------

        phrase_positions = set()

        for phrase in valid_phrases:

            positions = (
                get_phrase_word_positions(
                    tokens,
                    phrase
                )
            )

            for position in positions:

                phrase_positions.add(
                    position
                )

        # ------------------------------------------------------------------
        # Create single word candidates
        # ------------------------------------------------------------------

        single_words = (
            self.get_single_word_candidates(
                text,
                phrase_positions
            )
        )

        # ------------------------------------------------------------------
        # Combine candidates
        # ------------------------------------------------------------------

        candidates = []

        seen = set()

        # Add phrases first.

        for phrase in valid_phrases:

            if phrase not in seen:

                seen.add(phrase)

                candidates.append(
                    phrase
                )

        # Add individual words.

        for word in single_words:

            if word not in seen:

                seen.add(word)

                candidates.append(
                    word
                )

        return (

            candidates,

            valid_phrases,

            single_words

        )


    # ==========================================================================
    # SEMANTIC RANKING
    # ==========================================================================

    def rank_candidates(
        self,
        text: str,
        candidates: List[str],
        phrases: List[str],
        feedback_scores: Dict[str, float] = None,
    ) -> List[Dict]:

        """
        Rank candidates using Rasyosef embeddings.

        Strategy:

            1. Embed the complete input.
            2. Embed all candidate words/phrases.
            3. Calculate cosine similarity.
            4. Normalize semantic scores.
            5. Give a SMALL bonus to meaningful phrases.

        The phrase bonus is intentionally small.

        It helps:

            አዲስ አበባ

        remain together without forcing every
        phrase to rank first.
        """

        if not candidates:

            return []

        feedback_scores = feedback_scores or {}

        # ------------------------------------------------------------------
        # Encode complete input
        # ------------------------------------------------------------------

        document_embedding = (
            self.encode([text])[0]
        )

        # ------------------------------------------------------------------
        # Encode candidates
        # ------------------------------------------------------------------

        candidate_embeddings = (
            self.encode(candidates)
        )

        cosine_scores = []

        for embedding in candidate_embeddings:

            score = cosine_similarity(

                document_embedding,

                embedding

            )

            cosine_scores.append(
                score
            )

        cosine_scores = np.array(
            cosine_scores
        )

        # ------------------------------------------------------------------
        # Semantic normalization
        # ------------------------------------------------------------------

        minimum = cosine_scores.min()

        maximum = cosine_scores.max()

        if maximum == minimum:

            semantic_scores = np.ones(
                len(cosine_scores)
            )

        else:

            semantic_scores = (

                (
                    cosine_scores
                    -
                    minimum
                )

                /

                (
                    maximum
                    -
                    minimum
                )

            )

        # ------------------------------------------------------------------
        # Rank results
        # ------------------------------------------------------------------

        results = []

        phrase_set = set(
            phrases
        )

        for index, candidate in enumerate(
            candidates
        ):

            cosine = float(
                cosine_scores[index]
            )

            semantic = float(
                semantic_scores[index]
            )

            # --------------------------------------------------------------
            # Small phrase bonus
            # --------------------------------------------------------------

            if candidate in phrase_set:

                phrase_bonus = 0.03

            else:

                phrase_bonus = 0.0

            annotation_prior = self.annotation_lexicon.get(
                candidate.casefold(),
                0.0,
            )

            # --------------------------------------------------------------
            # Final score
            #
            # Semantic ranking is dominant.
            #
            # --------------------------------------------------------------

            final_score = (

                semantic
                +
                phrase_bonus
                +
                0.08 * annotation_prior
                +
                0.16 * feedback_scores.get(candidate.casefold(), 0.0)

            )

            results.append({

                "keyword":
                    candidate,

                "final_score":
                    round(
                        final_score,
                        4
                    ),

                "semantic_score":
                    round(
                        semantic,
                        4
                    ),

                "cosine_similarity":
                    round(
                        cosine,
                        4
                    ),

                "is_phrase":
                    candidate
                    in phrase_set,

                "annotation_prior":
                    round(annotation_prior, 4),

                "word_count":
                    len(
                        tokenize_amharic(
                            candidate
                        )
                    )

            })

        # ------------------------------------------------------------------
        # Sort
        # ------------------------------------------------------------------

        results.sort(

            key=lambda item:

            item["final_score"],

            reverse=True

        )

        return results


    # ==========================================================================
    # CHECK OVERLAP
    # ==========================================================================

    def candidates_overlap(
        self,
        candidate_a: str,
        candidate_b: str
    ) -> bool:

        """
        Check whether two candidates overlap.

        Example:

            candidate_a:
                አዲስ አበባ

            candidate_b:
                አዲስ

        -> True
        """

        words_a = set(
            tokenize_amharic(
                candidate_a
            )
        )

        words_b = set(
            tokenize_amharic(
                candidate_b
            )
        )

        return bool(

            words_a
            &
            words_b

        )


    # ==========================================================================
    # SELECT FINAL KEYWORDS
    # ==========================================================================

    def select_keywords(
        self,
        ranked_candidates: List[Dict],
        top_k: int,
        input_word_count: int = None,
    ) -> List[Dict]:

        """
        Select final keywords.

        Rules:
            - use the actual input word count as the base for the 60% rule
            - cap at maximum 10 keywords for long text
            - prevent overlapping phrase/word duplicates
        """

        if not ranked_candidates:
            return []

        if input_word_count is None:
            input_word_count = max(1, len(ranked_candidates))

        dynamic_limit = max(1, int(input_word_count * 0.6))
        effective_limit = min(10, max(1, dynamic_limit), max(1, int(top_k)))

        selected = []

        for candidate in ranked_candidates:

            keyword = candidate["keyword"]

            if not is_valid_keyword_candidate(keyword):
                continue

            overlaps = False

            for selected_item in selected:
                selected_keyword = selected_item["keyword"]
                if self.candidates_overlap(keyword, selected_keyword):
                    overlaps = True
                    break

            if overlaps:
                continue

            selected.append(candidate)

            if len(selected) >= effective_limit:
                break

        return selected


    # ==========================================================================
    # EXTRACT KEYWORDS
    # ==========================================================================

    def extract_keywords(
        self,
        text: str,
        top_k: int,
        feedback_scores: Dict[str, float] = None,
    ) -> Dict:

        """
        Complete extraction pipeline.
        """

        start_time = time.time()

        # ------------------------------------------------------------------
        # Normalize input
        # ------------------------------------------------------------------

        text = normalize_text(
            text
        )

        if not text:

            return {

                "input": text,

                "keywords": [],

                "details": [],

                "candidate_count": 0,

                "extraction_time":
                    0.0

            }

        # ------------------------------------------------------------------
        # Build candidates
        # ------------------------------------------------------------------

        (

            candidates,

            phrases,

            single_words

        ) = self.build_candidates(
            text
        )

        candidate_keys = {
            str(candidate).casefold()
            for candidate in candidates
        }
        for keyword, score in (feedback_scores or {}).items():
            candidate = str(keyword).strip()
            candidate_key = candidate.casefold()
            if score > 0 and candidate and candidate_key not in candidate_keys:
                candidates.append(candidate)
                candidate_keys.add(candidate_key)

        # ------------------------------------------------------------------
        # Rank candidates
        # ------------------------------------------------------------------

        ranked_candidates = (
            self.rank_candidates(

                text,

                candidates,

                phrases,

                feedback_scores,

            )
        )

        # ------------------------------------------------------------------
        # Select final keywords
        # ------------------------------------------------------------------

        input_word_count = len(
            tokenize_amharic(text)
        )

        selected_keywords = (
            self.select_keywords(
                ranked_candidates,
                top_k,
                input_word_count=input_word_count,
            )
        )

        extraction_time = (

            time.time()
            -
            start_time

        )

        return {

            "input":
                text,

            "keywords": [

                item["keyword"]

                for item
                in selected_keywords

            ],

            "details":
                selected_keywords,

            "candidate_count":
                len(candidates),

            "phrase_candidates":
                phrases,

            "single_word_candidates":
                single_words,

            "extraction_time":
                round(
                    extraction_time,
                    4
                )

        }


# ==============================================================================
# INTEGRITY CHECK
# ==============================================================================

def check_integrity(
    text: str,
    keywords: List[str],
    stopwords: set
) -> Tuple[bool, bool]:

    """
    Check:

        1. Every keyword occurs exactly in the input.
        2. No keyword is stopword-only.
    """

    normalized_text = normalize_text(
        text
    )

    exact_match_pass = True

    stopword_pass = True

    for keyword in keywords:

        # ------------------------------------------------------------------
        # Exact phrase check
        # ------------------------------------------------------------------

        pattern = (

            r"(?<!\S)"

            +

            re.escape(keyword)

            +

            r"(?!\S)"

        )

        if not re.search(

            pattern,

            normalized_text

        ):

            exact_match_pass = False

        # ------------------------------------------------------------------
        # Stopword-only check
        # ------------------------------------------------------------------

        if is_stopword_only_phrase(

            keyword,

            stopwords

        ):

            stopword_pass = False

    return (

        exact_match_pass,

        stopword_pass

    )


def average_precision_at_k(predicted: List[str], truth: List[str], top_k: int) -> float:
    truth_set = {keyword.casefold() for keyword in truth}
    if not truth_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, keyword in enumerate(predicted[:top_k], start=1):
        if keyword.casefold() in truth_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(truth_set), top_k) if hits else 0.0


def evaluate_dataset(
    extractor: RasyosefKeywordExtractor,
    dataframe: pd.DataFrame,
    top_k: int,
    dataset_name: str,
) -> Dict[str, float]:
    """Evaluate predictions against human annotations for one split."""

    precisions = []
    recalls = []
    f1_scores = []
    average_precisions = []

    print(f"Evaluating {dataset_name}: {len(dataframe):,} documents")
    for row in dataframe.itertuples(index=False):
        prediction = extractor.extract_keywords(row.text, top_k)["keywords"]
        predicted_set = {keyword.casefold() for keyword in prediction}
        truth_set = {keyword.casefold() for keyword in row.keyword_list}
        true_positive = len(predicted_set & truth_set)

        precision = true_positive / len(predicted_set) if predicted_set else 0.0
        recall = true_positive / len(truth_set) if truth_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        average_precisions.append(
            average_precision_at_k(prediction, row.keyword_list, top_k)
        )

    metrics = {
        "dataset": dataset_name,
        "documents": len(dataframe),
        f"precision_at_{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"f1_at_{top_k}": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "mAP": float(np.mean(average_precisions)) if average_precisions else 0.0,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def run_annotated_evaluation(
    extractor: RasyosefKeywordExtractor,
    top_k: int,
    evaluation_split: str,
) -> Dict[str, float]:
    """Train the annotation prior on train and score a held-out split."""

    train_df = load_annotated_dataset(TRAIN_PATH)
    split_paths = {
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
    }
    evaluation_df = load_annotated_dataset(split_paths[evaluation_split])
    print(f"Training annotation labels: {len(train_df):,} documents")
    return evaluate_dataset(
        extractor,
        evaluation_df,
        top_k,
        evaluation_split,
    )


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(
    result: Dict
) -> None:

    """
    Save extraction result as JSON.
    """

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        RESULTS_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            result,

            file,

            ensure_ascii=False,

            indent=4

        )


# ==============================================================================
# PRINT RESULTS
# ==============================================================================

def print_results(
    result: Dict,
    stopwords: set
) -> None:

    """
    Print extraction results.
    """

    text = result["input"]

    keywords = result["keywords"]

    details = result["details"]

    (

        exact_match_pass,

        stopword_pass

    ) = check_integrity(

        text,

        keywords,

        stopwords

    )

    print()

    print("-" * 100)

    print("INPUT")

    print("-" * 100)

    print()

    print(text)

    print()

    print("-" * 100)

    print("EXTRACTED KEYWORDS")

    print("-" * 100)

    print()

    if not keywords:

        print(
            "No valid keywords found."
        )

    else:

        for index, keyword in enumerate(

            keywords,

            start=1

        ):

            print(

                f"{index}. {keyword}"

            )

    print()

    print("-" * 100)

    print("RANKING DETAILS")

    print("-" * 100)

    print()

    if not details:

        print(
            "No ranking details available."
        )

    else:

        for index, item in enumerate(

            details,

            start=1

        ):

            keyword = (
                item["keyword"]
            )

            final_score = (
                item["final_score"]
            )

            semantic_score = (
                item["semantic_score"]
            )

            cosine_score = (
                item[
                    "cosine_similarity"
                ]
            )

            is_phrase = (
                item["is_phrase"]
            )

            word_count = (
                item["word_count"]
            )

            print(

                f"{index}. "

                f"{keyword:<35} "

                f"final={final_score:.4f}  "

                f"semantic={semantic_score:.4f}  "

                f"cosine={cosine_score:.4f}  "

                f"phrase={is_phrase}  "

                f"words={word_count}"

            )

    print()

    print("-" * 100)

    print("CANDIDATE INFORMATION")

    print("-" * 100)

    print()

    print(

        f"Meaningful phrase candidates: "
        f"{len(result.get('phrase_candidates', []))}"

    )

    for phrase in result.get(
        "phrase_candidates",
        []
    ):

        print(
            f"  - {phrase}"
        )

    print()

    print(

        f"Single-word candidates: "
        f"{len(result.get('single_word_candidates', []))}"

    )

    print()

    print("-" * 100)

    print("INTEGRITY CHECK")

    print("-" * 100)

    print()

    print(

        "Exact words/phrases from input : "

        +

        (

            "PASS"

            if exact_match_pass

            else "FAIL"

        )

    )

    print(

        "No stopword-only keywords       : "

        +

        (

            "PASS"

            if stopword_pass

            else "FAIL"

        )

    )

    print()

    print(

        f"Semantic candidates evaluated: "
        f"{result['candidate_count']}"

    )

    print()

    print(

        f"Extraction time: "
        f"{result['extraction_time']:.3f} seconds"

    )


# ==============================================================================
# INTERACTIVE PROGRAM
# ==============================================================================

def run_interactive_mode(
    extractor: RasyosefKeywordExtractor,
    stopwords: set
) -> None:

    """
    Interactive keyword extraction.
    """

    print()

    print("=" * 100)

    print(
        "INTERACTIVE AMHARIC KEYWORD EXTRACTION"
    )

    print("=" * 100)

    print()

    print(

        "Enter an Amharic topic/text and the system "
        "will extract semantic keywords from the input."

    )

    print()

    print("IMPORTANT:")

    print()

    print(
        "- Output words/phrases always come from your input."
    )

    print(
        "- Stopword-only keywords are excluded."
    )

    print(
        "- Single words are the default keyword candidates."
    )

    print(
        "- Meaningful paired phrases can remain together."
    )

    print(
        "- Arbitrary n-grams are NOT generated."
    )

    print(
        "- The entire input is never returned as a keyword."
    )

    print(
        "- If a phrase is selected, its component words are excluded."
    )

    print()

    print("Examples:")

    print()

    print(
        "    አዲስ አበባ"
    )

    print(
        "    ትምህርት ቤት"
    )

    print(
        "    ኒው ዮርክ"
    )

    print(
        "    የአየር ንብረት"
    )

    print()

    print(
        "can remain as complete phrases "
        "when they exist in the phrase lexicon."
    )

    print()

    print(
        "Commands: q / quit / exit"
    )

    print()

    while True:

        print("-" * 100)

        print()

        try:

            text = input(
                "Enter topic/text: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()

            print(
                "Exiting keyword extractor."
            )

            break

        # ------------------------------------------------------------------
        # Exit
        # ------------------------------------------------------------------

        if text.lower() in {

            "q",
            "quit",
            "exit"

        }:

            print()

            print(
                "Exiting keyword extractor."
            )

            break

        # ------------------------------------------------------------------
        # Empty input
        # ------------------------------------------------------------------

        if not text:

            print()

            print(
                "Please enter Amharic text."
            )

            print()

            continue

        # ------------------------------------------------------------------
        # Top K
        # ------------------------------------------------------------------

        print()

        top_k_input = input(

            f"How many keywords? "
            f"[default={DEFAULT_TOP_K}]: "

        ).strip()

        if not top_k_input:

            top_k = DEFAULT_TOP_K

        else:

            try:

                top_k = int(
                    top_k_input
                )

                if top_k <= 0:

                    print()

                    print(

                        "Top-K must be greater than 0. "
                        f"Using default={DEFAULT_TOP_K}."

                    )

                    top_k = DEFAULT_TOP_K

            except ValueError:

                print()

                print(

                    "Invalid number. "
                    f"Using default={DEFAULT_TOP_K}."

                )

                top_k = DEFAULT_TOP_K

        # ------------------------------------------------------------------
        # Extract
        # ------------------------------------------------------------------

        print()

        print(
            "Extracting..."
        )

        print()

        result = (

            extractor.extract_keywords(

                text,

                top_k

            )

        )

        # ------------------------------------------------------------------
        # Print
        # ------------------------------------------------------------------

        print_results(

            result,

            stopwords

        )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------

        save_results(
            result
        )

        print()

        print(
            "Saved result to:"
        )

        print()

        print(
            RESULTS_FILE
        )

        print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        choices=["validation", "test"],
        help="Evaluate against an annotated held-out split instead of interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of keywords used for extraction and evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N annotated documents.",
    )
    args = parser.parse_args()

    print()

    print("=" * 100)

    print(
        "INITIALIZING RASYosef KEYWORD EXTRACTOR"
    )

    print("=" * 100)

    print()

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Stopwords   : {STOPWORDS_PATH}"
    )

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print(
        f"Device      : {device}"
    )

    print()

    # --------------------------------------------------------------------------
    # Load stopwords
    # --------------------------------------------------------------------------

    stopwords = load_stopwords(

        STOPWORDS_PATH

    )

    train_dataframe = load_annotated_dataset(TRAIN_PATH)
    annotation_lexicon = build_annotation_lexicon(train_dataframe)

    print(
        f"Loaded {len(train_dataframe):,} annotated training documents "
        f"and {len(annotation_lexicon):,} keyword labels."
    )

    # --------------------------------------------------------------------------
    # Initialize extractor
    # --------------------------------------------------------------------------

    extractor = RasyosefKeywordExtractor(

        model_name=MODEL_NAME,

        stopwords=stopwords,

        annotation_lexicon=annotation_lexicon,

    )

    if args.evaluate:
        evaluation_dataframe = load_annotated_dataset(
            {
                "validation": VALIDATION_PATH,
                "test": TEST_PATH,
            }[args.evaluate]
        )
        if args.limit is not None:
            evaluation_dataframe = evaluation_dataframe.head(args.limit)
        metrics = evaluate_dataset(
            extractor,
            evaluation_dataframe,
            args.top_k,
            args.evaluate,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {METRICS_FILE}")
        return

    # --------------------------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------------------------

    run_interactive_mode(

        extractor,

        stopwords

    )


# ==============================================================================
# PROGRAM ENTRY
# ==============================================================================

if __name__ == "__main__":

    main()