"""
================================================================================
FINAL AMHARIC SEMANTIC KEYWORD EXTRACTION EVALUATION
================================================================================

Evaluates:
    notebooks/27_keyword_extraction.py

Dataset:
    data/train/validation_processed.xlsx

Model:
    rasyosef/embedding-amharic-base

Evaluation:
    - Precision@5
    - Recall@5
    - F1@5
    - Exact Match Accuracy
    - Mean Average Precision (mAP)

Rules:
    1. Keywords are extracted ONLY from the input text.
    2. Stopwords are excluded.
    3. Single words are default candidates.
    4. Known meaningful phrases remain together.
    5. Arbitrary n-grams are NOT generated.
    6. Predictions are compared with dataset gold keywords.

Run:

    python notebooks/31_evaluate_semantic_keyword_extraction.py

================================================================================
"""

import ast
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_NAME = "rasyosef/embedding-amharic-base"

BASE_DIR = Path(__file__).resolve().parent.parent

VALIDATION_FILE = (
    BASE_DIR
    / "data"
    / "train"
    / "validation_processed.xlsx"
)

STOPWORDS_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "amharic_stopwords.txt"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "semantic_keyword_evaluation"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "semantic_keyword_predictions.xlsx"
)

METRICS_FILE = (
    RESULTS_DIR
    / "semantic_keyword_metrics.json"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "semantic_keyword_summary.txt"
)

TOP_K = 5

BATCH_SIZE = 64


# ==============================================================================
# PHRASE LEXICON
# ==============================================================================
#
# IMPORTANT:
#
# This is NOT arbitrary n-gram generation.
#
# These phrases are treated as one concept only when they appear
# exactly in the input text.
#
# ==============================================================================

PHRASE_LEXICON = {

    # Locations
    "አዲስ አበባ",
    "ኒው ዮርክ",
    "ኒው ዮርክ ሲቲ",

    "ደቡብ አፍሪካ",
    "ሰሜን አሜሪካ",
    "ደቡብ አሜሪካ",

    "ምስራቅ አፍሪካ",
    "ምዕራብ አፍሪካ",
    "ሰሜን አፍሪካ",

    # General concepts
    "ኪነ ጥበብ",
    "ስነ ጥበብ",
    "ስነ ህይወት",
    "ስነ ምግባር",
    "አይነ አፋር",
    "ልብ ወለድ",

    # Education / Technology
    "ትምህርት ቤት",
    "ከፍተኛ ትምህርት",
    "ሰው ሰራሽ አእምሮ",
}


# ==============================================================================
# TEXT NORMALIZATION
# ==============================================================================

def normalize_text(text: str) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================================================================
# TOKENIZATION
# ==============================================================================

def tokenize_amharic(text: str) -> List[str]:

    pattern = r"[\u1200-\u137F]+"

    return re.findall(
        pattern,
        normalize_text(text)
    )


# ==============================================================================
# LOAD STOPWORDS
# ==============================================================================

def load_stopwords(path: Path) -> set:

    print()
    print("=" * 100)
    print("LOADING STOPWORDS")
    print("=" * 100)

    print()
    print("Path:")
    print(path)

    if not path.exists():

        print()
        print("ERROR: Stopwords file not found:")
        print(path)

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

    print()
    print(f"Loaded stopwords: {len(stopwords):,}")

    return stopwords


# ==============================================================================
# PARSE DATASET KEYWORDS
# ==============================================================================

def parse_keywords(value) -> List[str]:

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # --------------------------------------------------------------------------
    # Try Python list format
    # Example:
    #
    # ['ኢትዮጵያ', 'ግብርና']
    # --------------------------------------------------------------------------

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            return [

                normalize_text(item)

                for item in parsed

                if normalize_text(item)

            ]

    except Exception:
        pass

    # --------------------------------------------------------------------------
    # JSON list format
    # --------------------------------------------------------------------------

    try:

        parsed = json.loads(value)

        if isinstance(parsed, list):

            return [

                normalize_text(item)

                for item in parsed

                if normalize_text(item)

            ]

    except Exception:
        pass

    # --------------------------------------------------------------------------
    # Common separators
    # --------------------------------------------------------------------------

    separators = [
        ",",
        "፣",
        ";",
        "|",
    ]

    keywords = [value]

    for separator in separators:

        new_keywords = []

        for keyword in keywords:

            new_keywords.extend(
                keyword.split(separator)
            )

        keywords = new_keywords

    keywords = [

        normalize_text(keyword)

        for keyword in keywords

        if normalize_text(keyword)

    ]

    return keywords


# ==============================================================================
# FIND EXACT PHRASES
# ==============================================================================

def find_exact_phrases(
    text: str
) -> List[str]:

    normalized_text = normalize_text(text)

    detected = []

    for phrase in PHRASE_LEXICON:

        phrase = normalize_text(phrase)

        pattern = (
            r"(?<!\S)"
            +
            re.escape(phrase)
            +
            r"(?!\S)"
        )

        if re.search(
            pattern,
            normalized_text
        ):

            detected.append(phrase)

    detected = sorted(
        detected,
        key=lambda x: len(x.split()),
        reverse=True
    )

    return detected


# ==============================================================================
# FIND PHRASE POSITIONS
# ==============================================================================

def get_phrase_positions(
    tokens: List[str],
    phrase: str
) -> List[int]:

    phrase_tokens = tokenize_amharic(phrase)

    positions = []

    if not phrase_tokens:
        return positions

    phrase_length = len(phrase_tokens)

    for i in range(
        len(tokens) - phrase_length + 1
    ):

        segment = tokens[
            i:i + phrase_length
        ]

        if segment == phrase_tokens:

            positions.extend(
                range(
                    i,
                    i + phrase_length
                )
            )

    return positions


# ==============================================================================
# STOPWORD-ONLY PHRASE CHECK
# ==============================================================================

def is_stopword_only_phrase(
    phrase: str,
    stopwords: set
) -> bool:

    words = tokenize_amharic(phrase)

    if not words:
        return True

    return all(
        word in stopwords
        for word in words
    )


# ==============================================================================
# SEMANTIC KEYWORD EXTRACTOR
# ==============================================================================

class SemanticKeywordExtractor:


    def __init__(
        self,
        model_name: str,
        stopwords: set
    ):

        self.model_name = model_name

        self.stopwords = stopwords

        self.device = (

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        print()
        print("=" * 100)
        print("LOADING SEMANTIC KEYWORD EXTRACTION MODEL")
        print("=" * 100)

        print()
        print(f"Model : {self.model_name}")
        print(f"Device: {self.device}")

        print()
        print("Loading model...")

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
            f"Model loaded in "
            f"{loading_time:.2f} seconds."
        )


    # ==========================================================================
    # ENCODE
    # ==========================================================================

    def encode(
        self,
        texts: List[str]
    ) -> np.ndarray:

        return self.model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False,

            batch_size=BATCH_SIZE

        )


    # ==========================================================================
    # BUILD CANDIDATES
    # ==========================================================================

    def build_candidates(
        self,
        text: str
    ) -> Tuple[List[str], List[str]]:

        tokens = tokenize_amharic(text)

        # ----------------------------------------------------------------------
        # Find known meaningful phrases
        # ----------------------------------------------------------------------

        detected_phrases = find_exact_phrases(text)

        valid_phrases = [

            phrase

            for phrase
            in detected_phrases

            if not is_stopword_only_phrase(

                phrase,

                self.stopwords

            )

        ]

        # ----------------------------------------------------------------------
        # Find positions occupied by phrases
        # ----------------------------------------------------------------------

        phrase_positions = set()

        for phrase in valid_phrases:

            positions = get_phrase_positions(

                tokens,

                phrase

            )

            phrase_positions.update(
                positions
            )

        # ----------------------------------------------------------------------
        # Single word candidates
        # ----------------------------------------------------------------------

        single_words = []

        seen_words = set()

        for index, word in enumerate(tokens):

            # Skip phrase component words

            if index in phrase_positions:
                continue

            # Skip stopwords

            if word in self.stopwords:
                continue

            # Remove duplicates

            if word in seen_words:
                continue

            seen_words.add(word)

            single_words.append(word)

        # ----------------------------------------------------------------------
        # Combine candidates
        # ----------------------------------------------------------------------

        candidates = []

        seen = set()

        for phrase in valid_phrases:

            if phrase not in seen:

                seen.add(phrase)

                candidates.append(phrase)

        for word in single_words:

            if word not in seen:

                seen.add(word)

                candidates.append(word)

        return candidates, valid_phrases


    # ==========================================================================
    # CHECK OVERLAP
    # ==========================================================================

    def candidates_overlap(

        self,

        candidate_a: str,

        candidate_b: str

    ) -> bool:

        words_a = set(
            tokenize_amharic(candidate_a)
        )

        words_b = set(
            tokenize_amharic(candidate_b)
        )

        return bool(
            words_a & words_b
        )


    # ==========================================================================
    # EXTRACT KEYWORDS
    # ==========================================================================

    def extract_keywords(

        self,

        text: str,

        top_k: int = TOP_K

    ) -> Tuple[List[str], List[float]]:

        text = normalize_text(text)

        if not text:
            return [], []

        candidates, phrases = (

            self.build_candidates(text)

        )

        if not candidates:
            return [], []

        # ----------------------------------------------------------------------
        # Embed document
        # ----------------------------------------------------------------------

        document_embedding = (

            self.encode([text])[0]

        )

        # ----------------------------------------------------------------------
        # Embed candidates
        # ----------------------------------------------------------------------

        candidate_embeddings = (

            self.encode(candidates)

        )

        # ----------------------------------------------------------------------
        # Cosine similarity
        #
        # Embeddings are normalized,
        # so dot product = cosine similarity.
        # ----------------------------------------------------------------------

        scores = np.dot(

            candidate_embeddings,

            document_embedding

        )

        phrase_set = set(phrases)

        # ----------------------------------------------------------------------
        # Add small phrase bonus
        # ----------------------------------------------------------------------

        results = []

        for candidate, score in zip(

            candidates,

            scores

        ):

            final_score = float(score)

            if candidate in phrase_set:

                final_score += 0.03

            results.append(

                (
                    candidate,

                    final_score
                )

            )

        # ----------------------------------------------------------------------
        # Sort
        # ----------------------------------------------------------------------

        results.sort(

            key=lambda x: x[1],

            reverse=True

        )

        # ----------------------------------------------------------------------
        # Select without overlap
        # ----------------------------------------------------------------------

        selected_keywords = []

        selected_scores = []

        for candidate, score in results:

            overlaps = False

            for selected in selected_keywords:

                if self.candidates_overlap(

                    candidate,

                    selected

                ):

                    overlaps = True

                    break

            if overlaps:
                continue

            selected_keywords.append(candidate)

            selected_scores.append(
                round(float(score), 4)
            )

            if len(selected_keywords) >= top_k:
                break

        return (

            selected_keywords,

            selected_scores

        )


# ==============================================================================
# NORMALIZE KEYWORD FOR EVALUATION
# ==============================================================================

def normalize_keyword(
    keyword: str
) -> str:

    keyword = normalize_text(keyword)

    return keyword


# ==============================================================================
# PRECISION / RECALL / F1
# ==============================================================================

def calculate_metrics_at_k(

    predicted: List[str],

    gold: List[str],

    k: int

) -> Tuple[float, float, float]:

    predicted = [

        normalize_keyword(keyword)

        for keyword in predicted[:k]

    ]

    gold = [

        normalize_keyword(keyword)

        for keyword in gold

    ]

    predicted_set = set(predicted)

    gold_set = set(gold)

    if not predicted_set:

        precision = 0.0

    else:

        correct = len(
            predicted_set & gold_set
        )

        precision = (
            correct
            /
            len(predicted_set)
        )

    if not gold_set:

        recall = 0.0

    else:

        correct = len(
            predicted_set & gold_set
        )

        recall = (
            correct
            /
            len(gold_set)
        )

    if precision + recall == 0:

        f1 = 0.0

    else:

        f1 = (

            2
            *
            precision
            *
            recall

            /

            (
                precision
                +
                recall
            )

        )

    return (

        precision,

        recall,

        f1

    )


# ==============================================================================
# EXACT MATCH
# ==============================================================================

def calculate_exact_match(

    predicted: List[str],

    gold: List[str]

) -> int:

    predicted_set = {

        normalize_keyword(keyword)

        for keyword
        in predicted

    }

    gold_set = {

        normalize_keyword(keyword)

        for keyword
        in gold

    }

    return int(
        predicted_set == gold_set
    )


# ==============================================================================
# AVERAGE PRECISION
# ==============================================================================

def calculate_average_precision(

    predicted: List[str],

    gold: List[str],

    k: int

) -> float:

    gold_set = {

        normalize_keyword(keyword)

        for keyword
        in gold

    }

    if not gold_set:
        return 0.0

    correct = 0

    precision_sum = 0.0

    seen = set()

    for index, keyword in enumerate(

        predicted[:k],

        start=1

    ):

        keyword = normalize_keyword(keyword)

        if keyword in seen:
            continue

        seen.add(keyword)

        if keyword in gold_set:

            correct += 1

            precision_at_position = (

                correct
                /
                index

            )

            precision_sum += (

                precision_at_position

            )

    return (

        precision_sum

        /

        min(

            len(gold_set),

            k

        )

    )


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N validation documents.",
    )
    args = parser.parse_args()

    start_time = time.time()

    print()
    print("=" * 100)
    print("FINAL AMHARIC SEMANTIC KEYWORD EXTRACTION EVALUATION")
    print("=" * 100)

    print()
    print("Model:")
    print(MODEL_NAME)

    print()
    print("Validation file:")
    print(VALIDATION_FILE)

    print()
    print(f"Top-K: {TOP_K}")

    # --------------------------------------------------------------------------
    # Check files
    # --------------------------------------------------------------------------

    if not VALIDATION_FILE.exists():

        print()
        print("ERROR: Validation file not found:")
        print(VALIDATION_FILE)

        sys.exit(1)

    # --------------------------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("LOADING VALIDATION DATASET")
    print("=" * 100)

    validation_df = pd.read_excel(

        VALIDATION_FILE

    )

    print()
    print(f"Original documents: {len(validation_df):,}")

    print()
    print("Columns:")

    print(
        list(validation_df.columns)
    )

    # --------------------------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------------------------

    required_columns = [

        "text",

        "keywords"

    ]

    for column in required_columns:

        if column not in validation_df.columns:

            print()
            print(
                f"ERROR: Required column not found: {column}"
            )

            sys.exit(1)

    # --------------------------------------------------------------------------
    # Remove empty rows
    # --------------------------------------------------------------------------

    validation_df["text"] = (

        validation_df["text"]
        .fillna("")
        .astype(str)

    )

    validation_df["keywords"] = (

        validation_df["keywords"]
        .fillna("")
        .astype(str)

    )

    validation_df = validation_df[

        validation_df["text"].str.strip() != ""

    ].copy()

    if args.limit is not None:
        validation_df = validation_df.head(args.limit).copy()

    print()
    print(
        f"Documents after cleaning: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------------------------
    # Load stopwords
    # --------------------------------------------------------------------------

    stopwords = load_stopwords(

        STOPWORDS_PATH

    )

    # --------------------------------------------------------------------------
    # Initialize extractor
    # --------------------------------------------------------------------------

    extractor = SemanticKeywordExtractor(

        MODEL_NAME,

        stopwords

    )

    # --------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("RUNNING KEYWORD EVALUATION")
    print("=" * 100)

    print()
    print(
        f"Evaluating "
        f"{len(validation_df):,} documents..."
    )

    precisions = []

    recalls = []

    f1_scores = []

    exact_matches = []

    average_precisions = []

    prediction_rows = []

    total_documents = len(validation_df)

    # --------------------------------------------------------------------------
    # Loop through validation documents
    # --------------------------------------------------------------------------

    for row_number, (

        index,

        row

    ) in enumerate(

        validation_df.iterrows(),

        start=1

    ):

        text = normalize_text(

            row["text"]

        )

        gold_keywords = parse_keywords(

            row["keywords"]

        )

        # ----------------------------------------------------------------------
        # Extract predicted keywords
        # ----------------------------------------------------------------------

        predicted_keywords, scores = (

            extractor.extract_keywords(

                text,

                TOP_K

            )

        )

        # ----------------------------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------------------------

        precision, recall, f1 = (

            calculate_metrics_at_k(

                predicted_keywords,

                gold_keywords,

                TOP_K

            )

        )

        exact_match = (

            calculate_exact_match(

                predicted_keywords,

                gold_keywords

            )

        )

        average_precision = (

            calculate_average_precision(

                predicted_keywords,

                gold_keywords,

                TOP_K

            )

        )

        precisions.append(precision)

        recalls.append(recall)

        f1_scores.append(f1)

        exact_matches.append(exact_match)

        average_precisions.append(

            average_precision

        )

        # ----------------------------------------------------------------------
        # Save row result
        # ----------------------------------------------------------------------

        result_row = {

            "dataset_index": index,

            "text": text,

            "gold_keywords": " | ".join(

                gold_keywords

            ),

            "predicted_keywords": " | ".join(

                predicted_keywords

            ),

            "prediction_scores": " | ".join(

                map(str, scores)

            ),

            f"precision_at_{TOP_K}":

                round(precision, 4),

            f"recall_at_{TOP_K}":

                round(recall, 4),

            f"f1_at_{TOP_K}":

                round(f1, 4),

            "exact_match":

                exact_match,

            "average_precision":

                round(

                    average_precision,

                    4

                )

        }

        # Add original metadata if available

        for column in [

            "domain",

            "topics",

            "id",

            "annotation_id"

        ]:

            if column in validation_df.columns:

                result_row[column] = (

                    row[column]

                )

        prediction_rows.append(

            result_row

        )

        # ----------------------------------------------------------------------
        # Progress
        # ----------------------------------------------------------------------

        if (

            row_number % 50 == 0

            or

            row_number == total_documents

        ):

            elapsed = (

                time.time()

                -

                start_time

            )

            print(

                f"Processed "

                f"{row_number:,}"

                f"/"

                f"{total_documents:,} "

                f"({row_number / total_documents * 100:.1f}%) "

                f"| "

                f"Elapsed: "

                f"{elapsed / 60:.2f} minutes"

            )

    # --------------------------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------------------------

    mean_precision = float(

        np.mean(precisions)

    )

    mean_recall = float(

        np.mean(recalls)

    )

    mean_f1 = float(

        np.mean(f1_scores)

    )

    exact_match_accuracy = float(

        np.mean(exact_matches)

    )

    mean_average_precision = float(

        np.mean(average_precisions)

    )

    execution_time = (

        time.time()

        -

        start_time

    )

    # --------------------------------------------------------------------------
    # Print final results
    # --------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL KEYWORD EXTRACTION RESULTS")
    print("=" * 100)

    print()

    print(
        f"Precision@{TOP_K} : "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall@{TOP_K}    : "
        f"{mean_recall:.4f}"
    )

    print(
        f"F1@{TOP_K}        : "
        f"{mean_f1:.4f}"
    )

    print()

    print(
        f"Exact Match Accuracy : "
        f"{exact_match_accuracy:.4f}"
    )

    print()

    print(
        f"mAP                  : "
        f"{mean_average_precision:.4f}"
    )

    # --------------------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------------------

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    print()
    print("=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    predictions_df = pd.DataFrame(

        prediction_rows

    )

    print()
    print("Saving predictions...")

    predictions_df.to_excel(

        PREDICTIONS_FILE,

        index=False

    )

    print()
    print("Predictions saved:")
    print(PREDICTIONS_FILE)

    # --------------------------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------------------------

    metrics = {

        "model":

            MODEL_NAME,

        "validation_file":

            str(VALIDATION_FILE),

        "validation_documents":

            total_documents,

        "top_k":

            TOP_K,

        "precision_at_k":

            round(

                mean_precision,

                6

            ),

        "recall_at_k":

            round(

                mean_recall,

                6

            ),

        "f1_at_k":

            round(

                mean_f1,

                6

            ),

        "exact_match_accuracy":

            round(

                exact_match_accuracy,

                6

            ),

        "mean_average_precision":

            round(

                mean_average_precision,

                6

            ),

        "execution_time_seconds":

            round(

                execution_time,

                2

            ),

        "execution_time_minutes":

            round(

                execution_time / 60,

                2

            )

    }

    print()
    print("Saving metrics...")

    with open(

        METRICS_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            ensure_ascii=False,

            indent=4

        )

    print()
    print("Metrics saved:")
    print(METRICS_FILE)

    # --------------------------------------------------------------------------
    # Save summary
    # --------------------------------------------------------------------------

    summary = f"""
================================================================================
FINAL AMHARIC SEMANTIC KEYWORD EXTRACTION EVALUATION
================================================================================

MODEL
--------------------------------------------------------------------------------

{MODEL_NAME}


DATASET
--------------------------------------------------------------------------------

Validation file:
{VALIDATION_FILE}

Validation documents:
{total_documents:,}


KEYWORD RESULTS
--------------------------------------------------------------------------------

Precision@{TOP_K}: {mean_precision:.4f}

Recall@{TOP_K}: {mean_recall:.4f}

F1@{TOP_K}: {mean_f1:.4f}

Exact Match Accuracy: {exact_match_accuracy:.4f}

mAP: {mean_average_precision:.4f}


EXECUTION TIME
--------------------------------------------------------------------------------

Seconds: {execution_time:.2f}

Minutes: {execution_time / 60:.2f}


RESULT FILES
--------------------------------------------------------------------------------

Predictions:
{PREDICTIONS_FILE}

Metrics:
{METRICS_FILE}

================================================================================
"""

    with open(

        SUMMARY_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(summary)

    print()
    print("Summary saved:")
    print(SUMMARY_FILE)

    # --------------------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("SEMANTIC KEYWORD EVALUATION COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Validation documents: "
        f"{total_documents:,}"
    )

    print()

    print("-" * 80)
    print("KEYWORD RESULTS")
    print("-" * 80)

    print()

    print(
        f"Precision@{TOP_K}: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{mean_recall:.4f}"
    )

    print(
        f"F1@{TOP_K}: "
        f"{mean_f1:.4f}"
    )

    print(
        f"Exact Match Accuracy: "
        f"{exact_match_accuracy:.4f}"
    )

    print(
        f"mAP: "
        f"{mean_average_precision:.4f}"
    )

    print()

    print("-" * 80)
    print("EXECUTION TIME")
    print("-" * 80)

    print()

    print(
        f"{execution_time:.2f} seconds"
    )

    print(
        f"{execution_time / 60:.2f} minutes"
    )

    print()

    print("=" * 100)
    print("RESULT FILES")
    print("=" * 100)

    print()
    print(PREDICTIONS_FILE)
    print()
    print(METRICS_FILE)
    print()
    print(SUMMARY_FILE)
    print()


# ==============================================================================
# PROGRAM ENTRY
# ==============================================================================

if __name__ == "__main__":

    main()