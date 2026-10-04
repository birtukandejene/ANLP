# ============================================================
# 30_hybrid_keyword_validation.py
#
# FINAL HYBRID KEYWORD VALIDATION
#
# PROJECT:
# Amharic NLP Domain / Topic / Keyword System
#
# METHOD:
#
# 1. Load the semantic index built from training data.
#
# 2. Load validation data.
#
# 3. Encode validation documents.
#
# 4. Predict domain using domain centroids.
#
# 5. Retrieve similar training documents from the predicted
#    domain.
#
# 6. Collect annotated keywords from retrieved documents.
#
# 7. Score keywords using semantic retrieval frequency.
#
# 8. Extract lexical candidate keywords from the input text.
#
# 9. Preserve important multi-word Amharic phrases.
#
# 10. Combine semantic and lexical keyword candidates.
#
# 11. Evaluate:
#
#     Precision@5
#     Recall@5
#     F1@5
#     Exact Match Accuracy
#     mAP
#
# INPUT:
#
# data/train/train_processed.xlsx
# data/train/validation_processed.xlsx
# models/amharic_semantic_index.joblib
#
# OUTPUT:
#
# results/hybrid_keyword_validation/
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import sys
import json
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")


# ============================================================
# INSTALL MISSING PACKAGES
# ============================================================

import subprocess
import importlib


def install_if_missing(package_name, import_name=None):

    if import_name is None:
        import_name = package_name

    try:

        importlib.import_module(import_name)

    except ImportError:

        print()
        print(
            f"Installing missing package: {package_name}"
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name,
            ]
        )


install_if_missing(
    "sentence-transformers",
    "sentence_transformers"
)

install_if_missing("pandas")

install_if_missing("numpy")

install_if_missing("openpyxl")

install_if_missing("joblib")

install_if_missing(
    "scikit-learn",
    "sklearn"
)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch

from collections import Counter

from sentence_transformers import (
    SentenceTransformer
)


# ============================================================
# START TIMER
# ============================================================

START_TIME = time.time()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "train_processed.xlsx"
)


VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "validation_processed.xlsx"
)


SEMANTIC_INDEX_PATH = (
    PROJECT_ROOT
    / "models"
    / "amharic_semantic_index.joblib"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "hybrid_keyword_validation"
)


PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "hybrid_keyword_predictions.xlsx"
)


METRICS_PATH = (
    OUTPUT_DIR
    / "hybrid_keyword_metrics.json"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "hybrid_keyword_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "rasyosef/embedding-amharic-base"
)


BATCH_SIZE = 16


TOP_K_DOCUMENTS = 10


TOP_K_KEYWORDS = 5


TEXT_COLUMN = "text"

DOMAIN_COLUMN = "domain"

KEYWORDS_COLUMN = "keywords"


# ============================================================
# HYBRID WEIGHTS
# ============================================================

# Semantic retrieval score.

SEMANTIC_WEIGHT = 0.75


# Lexical score.

LEXICAL_WEIGHT = 0.25


# ============================================================
# IMPORTANT MULTI-WORD PHRASES
#
# This is NOT the main keyword database.
#
# It only prevents important phrases from being split.
#
# Add only meaningful phrases.
#
# ============================================================

PHRASE_LEXICON = {

    # --------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------

    "አዲስ አበባ",

    "ኒው ዮርክ",

    "ኒው ዮርክ ሲቲ",

    "ደቡብ አፍሪካ",

    "ሰሜን አሜሪካ",

    "ደቡብ አሜሪካ",

    "ምስራቅ አፍሪካ",

    "ምዕራብ አፍሪካ",

    "ሰሜን አፍሪካ",

    "አዲስ አበባ ከተማ",


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    "ትምህርት ቤት",

    "ከፍተኛ ትምህርት",

    "የትምህርት ሥርዓት",


    # --------------------------------------------------------
    # GOVERNMENT / POLITICS
    # --------------------------------------------------------

    "መንግሥት አስተዳደር",

    "ፌዴራል መንግሥት",

    "ክልል መንግሥት",

    "የሰብአዊ መብት",

    "ሰብአዊ መብት",


    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    "ሰው ሰራሽ አእምሮ",

}


# ============================================================
# AMHARIC STOPWORDS
#
# Small basic list.
#
# These are used only to reduce meaningless lexical words.
#
# ============================================================

STOPWORDS = {

    "እና",

    "ወይም",

    "ነው",

    "ናቸው",

    "ነበር",

    "ነበሩ",

    "ይህ",

    "ይህን",

    "ይህንን",

    "ያ",

    "ያን",

    "እነዚህ",

    "እነዚያ",

    "ከ",

    "ለ",

    "በ",

    "ወደ",

    "እስከ",

    "ላይ",

    "ውስጥ",

    "ስለ",

    "ጋር",

    "ነገር",

    "ብቻ",

    "ደግሞ",

    "ግን",

    "ነገር",

}


# ============================================================
# DISPLAY HEADER
# ============================================================

print()

print("=" * 80)

print(
    "FINAL HYBRID AMHARIC KEYWORD VALIDATION"
)

print("=" * 80)


print()

print("Project root:")

print(PROJECT_ROOT)


print()

print("Training data:")

print(TRAIN_PATH)


print()

print("Validation data:")

print(VALIDATION_PATH)


print()

print("Semantic index:")

print(SEMANTIC_INDEX_PATH)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [

    TRAIN_PATH,

    VALIDATION_PATH,

    SEMANTIC_INDEX_PATH,

]


for path in required_files:

    if not path.exists():

        print()

        print("=" * 80)

        print("ERROR")

        print("=" * 80)

        print()

        print(
            "Required file not found:"
        )

        print(path)

        sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if value is None:

        return ""


    if isinstance(
        value,
        float
    ) and np.isnan(value):

        return ""


    text = str(value)


    text = " ".join(
        text.strip().split()
    )


    return text


# ============================================================
# KEYWORD PARSER
#
# Supports:
#
# keyword1|keyword2
#
# keyword1, keyword2
#
# keyword1; keyword2
#
# ============================================================

def parse_keywords(value):

    if value is None:

        return []


    if isinstance(
        value,
        float
    ) and np.isnan(value):

        return []


    text = clean_text(value)


    if not text:

        return []


    separators = [

        "|",

        ",",

        ";",

    ]


    for separator in separators:

        text = text.replace(
            separator,
            "|"
        )


    keywords = []


    for keyword in text.split("|"):

        keyword = clean_text(keyword)


        if not keyword:

            continue


        if keyword not in keywords:

            keywords.append(keyword)


    return keywords


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(

        vector,

        dtype=np.float32

    )


    norm = np.linalg.norm(vector)


    if norm == 0:

        return vector


    return vector / norm


# ============================================================
# MATRIX NORMALIZATION
# ============================================================

def normalize_matrix(matrix):

    matrix = np.asarray(

        matrix,

        dtype=np.float32

    )


    norms = np.linalg.norm(

        matrix,

        axis=1,

        keepdims=True

    )


    norms[norms == 0] = 1.0


    return matrix / norms


# ============================================================
# LOAD SEMANTIC INDEX
# ============================================================

print()

print("=" * 80)

print(
    "LOADING SEMANTIC INDEX"
)

print("=" * 80)


semantic_index = joblib.load(

    SEMANTIC_INDEX_PATH

)


train_embeddings = np.asarray(

    semantic_index["embeddings"],

    dtype=np.float32

)


train_embeddings = normalize_matrix(

    train_embeddings

)


train_domains = (

    semantic_index["domains"]

)


domain_centroids = (

    semantic_index["domain_centroids"]

)


print()

print(
    "Index loaded successfully."
)


print()

print(
    "Indexed documents:",
    f"{len(train_embeddings):,}"
)


print()

print(
    "Embedding dimension:",
    train_embeddings.shape[1]
)


print()

print(
    "Semantic model:",
    semantic_index.get(
        "semantic_model_name",
        MODEL_NAME
    )
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()

print("=" * 80)

print(
    "LOADING TRAINING DATA"
)

print("=" * 80)


train_df = pd.read_excel(

    TRAIN_PATH,

    engine="openpyxl"

)


print()

print(
    "Training rows:",
    f"{len(train_df):,}"
)


print()

print(
    "Training columns:"
)

print(
    list(train_df.columns)
)


# ============================================================
# CHECK TRAINING COLUMNS
# ============================================================

required_columns = [

    TEXT_COLUMN,

    DOMAIN_COLUMN,

    KEYWORDS_COLUMN,

]


missing_columns = [

    column

    for column in required_columns

    if column not in train_df.columns

]


if missing_columns:

    print()

    print(
        "ERROR: Missing training columns:"
    )

    print(missing_columns)

    sys.exit(1)


# ============================================================
# CLEAN TRAINING DATA
# ============================================================

train_df[TEXT_COLUMN] = (

    train_df[TEXT_COLUMN]

    .apply(clean_text)

)


train_df[DOMAIN_COLUMN] = (

    train_df[DOMAIN_COLUMN]

    .apply(clean_text)

)


train_df[KEYWORDS_COLUMN] = (

    train_df[KEYWORDS_COLUMN]

    .apply(parse_keywords)

)


train_df = train_df[

    train_df[TEXT_COLUMN].str.len() > 0

].copy()


train_df = train_df[

    train_df[DOMAIN_COLUMN].str.len() > 0

].copy()


train_df = train_df.reset_index(

    drop=True

)


# ============================================================
# ALIGN TRAINING DATA WITH SEMANTIC INDEX
# ============================================================

if len(train_df) != len(train_embeddings):

    print()

    print("=" * 80)

    print("ERROR")

    print("=" * 80)

    print()

    print(
        "Training Excel rows do not match semantic index."
    )


    print()

    print(
        "Training rows:",
        len(train_df)
    )


    print(
        "Index embeddings:",
        len(train_embeddings)
    )


    print()

    print(
        "The semantic index must be built from the same"
    )

    print(
        "training data in the same order."
    )

    sys.exit(1)


# ============================================================
# TRAINING KEYWORDS
# ============================================================

train_keywords = (

    train_df[KEYWORDS_COLUMN]

    .tolist()

)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print()

print("=" * 80)

print(
    "LOADING VALIDATION DATA"
)

print("=" * 80)


validation_df = pd.read_excel(

    VALIDATION_PATH,

    engine="openpyxl"

)


print()

print(
    "Validation rows:",
    f"{len(validation_df):,}"
)


print()

print(
    "Validation columns:"
)

print(
    list(validation_df.columns)
)


# ============================================================
# CHECK VALIDATION COLUMNS
# ============================================================

missing_columns = [

    column

    for column in required_columns

    if column not in validation_df.columns

]


if missing_columns:

    print()

    print(
        "ERROR: Missing validation columns:"
    )

    print(missing_columns)

    sys.exit(1)


# ============================================================
# CLEAN VALIDATION DATA
# ============================================================

validation_df[TEXT_COLUMN] = (

    validation_df[TEXT_COLUMN]

    .apply(clean_text)

)


validation_df[DOMAIN_COLUMN] = (

    validation_df[DOMAIN_COLUMN]

    .apply(clean_text)

)


validation_df[KEYWORDS_COLUMN] = (

    validation_df[KEYWORDS_COLUMN]

    .apply(parse_keywords)

)


validation_df = validation_df[

    validation_df[TEXT_COLUMN].str.len() > 0

].copy()


validation_df = validation_df[

    validation_df[DOMAIN_COLUMN].str.len() > 0

].copy()


validation_df = validation_df.reset_index(

    drop=True

)


validation_texts = (

    validation_df[TEXT_COLUMN]

    .tolist()

)


true_domains = (

    validation_df[DOMAIN_COLUMN]

    .tolist()

)


true_keywords = (

    validation_df[KEYWORDS_COLUMN]

    .tolist()

)


print()

print(
    "Final validation documents:",
    f"{len(validation_texts):,}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()

print("=" * 80)

print(
    "LOADING AMHARIC EMBEDDING MODEL"
)

print("=" * 80)


device = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print(
    "Model:",
    MODEL_NAME
)


print()

print(
    "Device:",
    device
)


if device == "cuda":

    try:

        print()

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    except Exception:

        pass


else:

    print()

    print(
        "Running on CPU."
    )


print()

print(
    "Loading model..."
)


model = SentenceTransformer(

    MODEL_NAME,

    device=device

)


print()

print(
    "Model loaded successfully."
)


# ============================================================
# ENCODE VALIDATION DOCUMENTS
# ============================================================

print()

print("=" * 80)

print(
    "ENCODING VALIDATION DOCUMENTS"
)

print("=" * 80)


if hasattr(
    model,
    "encode_query"
):

    validation_embeddings = (

        model.encode_query(

            validation_texts,

            batch_size=BATCH_SIZE,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True,

        )

    )


else:

    validation_embeddings = (

        model.encode(

            validation_texts,

            batch_size=BATCH_SIZE,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True,

        )

    )


validation_embeddings = normalize_matrix(

    validation_embeddings

)


print()

print(
    "Embedding shape:",
    validation_embeddings.shape
)


# ============================================================
# PREPARE DOMAIN CENTROIDS
# ============================================================

print()

print("=" * 80)

print(
    "PREPARING DOMAIN VECTORS"
)

print("=" * 80)


domain_names = sorted(

    domain_centroids.keys()

)


domain_matrix = np.vstack(

    [

        domain_centroids[domain]

        for domain in domain_names

    ]

)


domain_matrix = normalize_matrix(

    domain_matrix

)


print()

print(
    "Domains:",
    len(domain_names)
)


# ============================================================
# NORMALIZE KEYWORDS
# ============================================================

def normalize_keyword(keyword):

    keyword = clean_text(keyword)


    return keyword.lower()


# ============================================================
# PROTECT PHRASES
#
# Replaces spaces inside known phrases temporarily.
#
# Example:
#
# ሰው ሰራሽ አእምሮ
#
# becomes:
#
# ሰው__ሰራሽ__አእምሮ
#
# ============================================================

def protect_phrases(text):

    text = clean_text(text)


    protected = text


    sorted_phrases = sorted(

        PHRASE_LEXICON,

        key=len,

        reverse=True

    )


    for phrase in sorted_phrases:

        placeholder = phrase.replace(

            " ",

            "__"

        )


        protected = protected.replace(

            phrase,

            placeholder

        )


    return protected


# ============================================================
# RESTORE PHRASES
# ============================================================

def restore_phrase(token):

    return token.replace(

        "__",

        " "

    )


# ============================================================
# LEXICAL KEYWORD EXTRACTION
# ============================================================

def extract_lexical_keywords(text):

    text = protect_phrases(text)


    tokens = text.split()


    candidates = []


    for token in tokens:

        token = clean_text(token)


        if not token:

            continue


        token = restore_phrase(token)


        normalized = normalize_keyword(token)


        if not normalized:

            continue


        if normalized in STOPWORDS:

            continue


        if len(normalized) < 2:

            continue


        if token not in candidates:

            candidates.append(token)


    return candidates


# ============================================================
# DOMAIN PREDICTION
# ============================================================

def predict_domain(embedding):

    scores = (

        domain_matrix

        @ embedding

    )


    best_index = int(

        np.argmax(scores)

    )


    domain = (

        domain_names[best_index]

    )


    score = float(

        scores[best_index]

    )


    return (

        domain,

        score

    )


# ============================================================
# RETRIEVE SIMILAR DOCUMENTS
#
# Retrieval is restricted to predicted domain.
#
# ============================================================

def retrieve_documents(

    embedding,

    predicted_domain

):

    candidate_indices = [

        i

        for i, domain in enumerate(train_domains)

        if domain == predicted_domain

    ]


    if not candidate_indices:

        candidate_indices = list(

            range(

                len(train_embeddings)

            )

        )


    candidate_embeddings = (

        train_embeddings[
            candidate_indices
        ]

    )


    similarities = (

        candidate_embeddings

        @ embedding

    )


    top_k = min(

        TOP_K_DOCUMENTS,

        len(candidate_indices)

    )


    if top_k <= 0:

        return []


    top_positions = np.argpartition(

        -similarities,

        top_k - 1

    )[:top_k]


    top_positions = top_positions[

        np.argsort(

            -similarities[top_positions]

        )

    ]


    results = []


    for position in top_positions:

        index = candidate_indices[

            int(position)

        ]


        similarity = float(

            similarities[

                int(position)

            ]

        )


        results.append(

            (

                index,

                similarity

            )

        )


    return results


# ============================================================
# SEMANTIC KEYWORD RETRIEVAL
#
# Keywords from similar documents receive a score based on:
#
# similarity × rank weight
#
# ============================================================

def get_semantic_keyword_scores(

    retrieved_documents

):

    keyword_scores = Counter()


    for rank, (

        document_index,

        similarity

    ) in enumerate(

        retrieved_documents,

        start=1

    ):


        keywords = (

            train_keywords[
                document_index
            ]

        )


        rank_weight = (

            1.0 / rank

        )


        for keyword in keywords:

            keyword = clean_text(keyword)


            if not keyword:

                continue


            score = (

                max(
                    similarity,
                    0.0
                )

                * rank_weight

            )


            keyword_scores[keyword] += score


    return keyword_scores


# ============================================================
# LEXICAL KEYWORD SCORING
#
# A lexical candidate gets a score when it matches:
#
# 1. Retrieved annotated keywords.
#
# 2. Part of a retrieved keyword.
#
# ============================================================

def get_lexical_keyword_scores(

    text,

    semantic_keyword_scores

):

    lexical_candidates = (

        extract_lexical_keywords(text)

    )


    lexical_scores = Counter()


    semantic_keywords = list(

        semantic_keyword_scores.keys()

    )


    normalized_semantic = {

        normalize_keyword(keyword):

        keyword

        for keyword in semantic_keywords

    }


    for candidate in lexical_candidates:

        normalized_candidate = (

            normalize_keyword(candidate)

        )


        if not normalized_candidate:

            continue


        if (

            normalized_candidate

            in normalized_semantic

        ):

            original_keyword = (

                normalized_semantic[
                    normalized_candidate
                ]

            )


            lexical_scores[
                original_keyword
            ] += 1.0


        for semantic_keyword in semantic_keywords:

            normalized_keyword = (

                normalize_keyword(
                    semantic_keyword
                )

            )


            if (

                normalized_candidate
                == normalized_keyword

            ):

                continue


            if (

                normalized_candidate

                in normalized_keyword

                or

                normalized_keyword

                in normalized_candidate

            ):

                lexical_scores[
                    semantic_keyword
                ] += 0.50


    return lexical_scores


# ============================================================
# NORMALIZE SCORE DICTIONARY
# ============================================================

def normalize_scores(scores):

    if not scores:

        return {}


    maximum = max(

        scores.values()

    )


    if maximum <= 0:

        return {

            key: 0.0

            for key in scores

        }


    return {

        key:

        float(value / maximum)

        for key, value in scores.items()

    }


# ============================================================
# HYBRID KEYWORD PREDICTION
# ============================================================

def predict_keywords(

    text,

    embedding,

    predicted_domain

):


    # --------------------------------------------------------
    # Retrieve similar documents.
    # --------------------------------------------------------

    retrieved_documents = (

        retrieve_documents(

            embedding,

            predicted_domain

        )

    )


    # --------------------------------------------------------
    # Semantic keyword scores.
    # --------------------------------------------------------

    semantic_scores = (

        get_semantic_keyword_scores(

            retrieved_documents

        )

    )


    # --------------------------------------------------------
    # Lexical keyword scores.
    # --------------------------------------------------------

    lexical_scores = (

        get_lexical_keyword_scores(

            text,

            semantic_scores

        )

    )


    # --------------------------------------------------------
    # Normalize scores.
    # --------------------------------------------------------

    semantic_scores = (

        normalize_scores(

            semantic_scores

        )

    )


    lexical_scores = (

        normalize_scores(

            lexical_scores

        )

    )


    # --------------------------------------------------------
    # Combine scores.
    # --------------------------------------------------------

    all_keywords = (

        set(

            semantic_scores.keys()

        )

        |

        set(

            lexical_scores.keys()

        )

    )


    hybrid_scores = {}


    for keyword in all_keywords:

        semantic_score = (

            semantic_scores.get(

                keyword,

                0.0

            )

        )


        lexical_score = (

            lexical_scores.get(

                keyword,

                0.0

            )

        )


        final_score = (

            SEMANTIC_WEIGHT

            * semantic_score

            +

            LEXICAL_WEIGHT

            * lexical_score

        )


        hybrid_scores[keyword] = (

            float(final_score)

        )


    # --------------------------------------------------------
    # Sort keywords.
    # --------------------------------------------------------

    ranked_keywords = sorted(

        hybrid_scores.items(),

        key=lambda item:

        item[1],

        reverse=True

    )


    # --------------------------------------------------------
    # Keep Top-K.
    # --------------------------------------------------------

    predicted_keywords = [

        keyword

        for keyword, score

        in ranked_keywords[

            :TOP_K_KEYWORDS

        ]

    ]


    return (

        predicted_keywords,

        retrieved_documents,

        hybrid_scores

    )


# ============================================================
# AVERAGE PRECISION
# ============================================================

def average_precision(

    predicted,

    true

):

    if not true:

        return 0.0


    true_set = set(

        normalize_keyword(keyword)

        for keyword in true

    )


    correct = 0


    precision_sum = 0.0


    for rank, keyword in enumerate(

        predicted,

        start=1

    ):


        normalized = (

            normalize_keyword(keyword)

        )


        if normalized in true_set:

            correct += 1


            precision_sum += (

                correct / rank

            )


    if correct == 0:

        return 0.0


    return (

        precision_sum

        / len(true_set)

    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_keywords(

    predictions,

    truths

):

    precision_scores = []

    recall_scores = []

    f1_scores = []

    exact_matches = []

    average_precisions = []


    for predicted, true in zip(

        predictions,

        truths

    ):


        predicted_set = set(

            normalize_keyword(keyword)

            for keyword in predicted

        )


        true_set = set(

            normalize_keyword(keyword)

            for keyword in true

        )


        if not true_set:

            continue


        true_positive = len(

            predicted_set

            &

            true_set

        )


        precision = (

            true_positive

            / len(predicted_set)

            if predicted_set

            else 0.0

        )


        recall = (

            true_positive

            / len(true_set)

            if true_set

            else 0.0

        )


        if precision + recall > 0:

            f1 = (

                2

                * precision

                * recall

                / (

                    precision

                    + recall

                )

            )

        else:

            f1 = 0.0


        exact_match = (

            predicted_set

            == true_set

        )


        ap = (

            average_precision(

                predicted,

                true

            )

        )


        precision_scores.append(

            precision

        )


        recall_scores.append(

            recall

        )


        f1_scores.append(

            f1

        )


        exact_matches.append(

            float(exact_match)

        )


        average_precisions.append(

            ap

        )


    results = {


        "precision_at_5":

        float(

            np.mean(

                precision_scores

            )

        )


        if precision_scores

        else 0.0,


        "recall_at_5":

        float(

            np.mean(

                recall_scores

            )

        )


        if recall_scores

        else 0.0,


        "f1_at_5":

        float(

            np.mean(

                f1_scores

            )

        )


        if f1_scores

        else 0.0,


        "exact_match_accuracy":

        float(

            np.mean(

                exact_matches

            )

        )


        if exact_matches

        else 0.0,


        "map":

        float(

            np.mean(

                average_precisions

            )

        )


        if average_precisions

        else 0.0,

    }


    return results


# ============================================================
# RUN PREDICTION
# ============================================================

print()

print("=" * 80)

print(
    "RUNNING HYBRID KEYWORD PREDICTION"
)

print("=" * 80)


predicted_domains = []

domain_scores = []

predicted_keywords_all = []

retrieved_document_indices_all = []

retrieved_similarity_scores_all = []


total_documents = (

    len(validation_texts)

)


for i in range(

    total_documents

):


    if (

        i == 0

        or

        (i + 1) % 10 == 0

        or

        i + 1 == total_documents

    ):

        print(

            f"Processing: "

            f"{i + 1:,}"

            f"/"

            f"{total_documents:,}"

        )


    text = (

        validation_texts[i]

    )


    embedding = (

        validation_embeddings[i]

    )


    # --------------------------------------------------------
    # Predict domain.
    # --------------------------------------------------------

    predicted_domain, domain_score = (

        predict_domain(

            embedding

        )

    )


    # --------------------------------------------------------
    # Predict keywords.
    # --------------------------------------------------------

    (

        predicted_keywords,

        retrieved_documents,

        hybrid_scores

    ) = (

        predict_keywords(

            text,

            embedding,

            predicted_domain

        )

    )


    predicted_domains.append(

        predicted_domain

    )


    domain_scores.append(

        domain_score

    )


    predicted_keywords_all.append(

        predicted_keywords

    )


    retrieved_indices = [

        index

        for index, score

        in retrieved_documents

    ]


    retrieved_scores = [

        score

        for index, score

        in retrieved_documents

    ]


    retrieved_document_indices_all.append(

        retrieved_indices

    )


    retrieved_similarity_scores_all.append(

        retrieved_scores

    )


# ============================================================
# KEYWORD EVALUATION
# ============================================================

print()

print("=" * 80)

print(
    "KEYWORD EVALUATION"
)

print("=" * 80)


keyword_metrics = (

    evaluate_keywords(

        predicted_keywords_all,

        true_keywords

    )

)


print()

print(

    f"Precision@{TOP_K_KEYWORDS} "

    f": "

    f"{keyword_metrics['precision_at_5']:.4f}"

)


print(

    f"Recall@{TOP_K_KEYWORDS} "

    f": "

    f"{keyword_metrics['recall_at_5']:.4f}"

)


print(

    f"F1@{TOP_K_KEYWORDS} "

    f": "

    f"{keyword_metrics['f1_at_5']:.4f}"

)


print()

print(

    "Exact Match Accuracy "

    f": "

    f"{keyword_metrics['exact_match_accuracy']:.4f}"

)


print()

print(

    "mAP "

    f": "

    f"{keyword_metrics['map']:.4f}"

)


# ============================================================
# DOMAIN ACCURACY
#
# Included for reference.
#
# ============================================================

domain_correct = sum(

    predicted == true

    for predicted, true

    in zip(

        predicted_domains,

        true_domains

    )

)


domain_accuracy = (

    domain_correct

    / len(true_domains)

)


print()

print("=" * 80)

print(
    "DOMAIN REFERENCE RESULT"
)

print("=" * 80)


print()

print(

    "Domain Accuracy: "

    f"{domain_accuracy:.4f}"

)


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

print()

print("=" * 80)

print(
    "CREATING PREDICTION RESULTS"
)

print("=" * 80)


results_df = validation_df.copy()


results_df[

    "predicted_domain"

] = predicted_domains


results_df[

    "domain_similarity"

] = domain_scores


results_df[

    "true_keywords"

] = [

    " | ".join(keywords)

    for keywords

    in true_keywords

]


results_df[

    "predicted_keywords"

] = [

    " | ".join(keywords)

    for keywords

    in predicted_keywords_all

]


results_df[

    "retrieved_training_indices"

] = [

    " | ".join(

        str(index)

        for index in indices

    )

    for indices

    in retrieved_document_indices_all

]


results_df[

    "retrieved_similarity_scores"

] = [

    " | ".join(

        f"{score:.4f}"

        for score in scores

    )

    for scores

    in retrieved_similarity_scores_all

]


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print()

print(
    "Saving predictions..."
)


results_df.to_excel(

    PREDICTIONS_PATH,

    index=False,

    engine="openpyxl"

)


print()

print(
    "Predictions saved:"
)

print(PREDICTIONS_PATH)


# ============================================================
# BUILD METRICS
# ============================================================

execution_time = (

    time.time()

    - START_TIME

)


metrics = {


    "project":

    "Amharic NLP Hybrid Keyword Extraction",


    "semantic_model":

    MODEL_NAME,


    "validation_documents":

    int(

        len(validation_texts)

    ),


    "top_k_documents":

    TOP_K_DOCUMENTS,


    "top_k_keywords":

    TOP_K_KEYWORDS,


    "semantic_weight":

    SEMANTIC_WEIGHT,


    "lexical_weight":

    LEXICAL_WEIGHT,


    "phrase_lexicon_size":

    len(PHRASE_LEXICON),


    "domain_accuracy":

    float(domain_accuracy),


    "keyword_metrics":

    keyword_metrics,


    "execution_time_seconds":

    float(execution_time),


    "execution_time_minutes":

    float(

        execution_time / 60

    ),

}


# ============================================================
# SAVE METRICS JSON
# ============================================================

print()

print(
    "Saving metrics..."
)


with open(

    METRICS_PATH,

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

print(
    "Metrics saved:"
)

print(METRICS_PATH)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = f"""

================================================================================
FINAL HYBRID AMHARIC KEYWORD VALIDATION
================================================================================

VALIDATION DOCUMENTS

{len(validation_texts):,}


CONFIGURATION

Embedding Model:

{MODEL_NAME}


Top Retrieved Documents:

{TOP_K_DOCUMENTS}


Top Predicted Keywords:

{TOP_K_KEYWORDS}


Semantic Weight:

{SEMANTIC_WEIGHT}


Lexical Weight:

{LEXICAL_WEIGHT}


Phrase Lexicon Size:

{len(PHRASE_LEXICON)}


================================================================================
DOMAIN RESULT
================================================================================

Accuracy:

{domain_accuracy:.4f}


================================================================================
KEYWORD RESULTS
================================================================================

Precision@{TOP_K_KEYWORDS}:

{keyword_metrics['precision_at_5']:.4f}


Recall@{TOP_K_KEYWORDS}:

{keyword_metrics['recall_at_5']:.4f}


F1@{TOP_K_KEYWORDS}:

{keyword_metrics['f1_at_5']:.4f}


Exact Match Accuracy:

{keyword_metrics['exact_match_accuracy']:.4f}


mAP:

{keyword_metrics['map']:.4f}


================================================================================
EXECUTION TIME
================================================================================

Seconds:

{execution_time:.2f}


Minutes:

{execution_time / 60:.2f}


================================================================================
OUTPUT FILES
================================================================================

Predictions:

{PREDICTIONS_PATH}


Metrics:

{METRICS_PATH}


Summary:

{SUMMARY_PATH}


================================================================================
"""


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(

    SUMMARY_PATH,

    "w",

    encoding="utf-8"

) as file:


    file.write(

        summary

    )


print()

print(
    "Summary saved:"
)

print(SUMMARY_PATH)


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("=" * 80)

print(
    "FINAL HYBRID KEYWORD VALIDATION COMPLETE"
)

print("=" * 80)


print()

print(

    "Validation documents:",

    f"{len(validation_texts):,}"

)


print()

print("-" * 80)

print(
    "DOMAIN RESULT"
)

print("-" * 80)


print()

print(

    f"Accuracy: "

    f"{domain_accuracy:.4f}"

)


print()

print("-" * 80)

print(
    "KEYWORD RESULTS"
)

print("-" * 80)


print()

print(

    f"Precision@{TOP_K_KEYWORDS}: "

    f"{keyword_metrics['precision_at_5']:.4f}"

)


print(

    f"Recall@{TOP_K_KEYWORDS}: "

    f"{keyword_metrics['recall_at_5']:.4f}"

)


print(

    f"F1@{TOP_K_KEYWORDS}: "

    f"{keyword_metrics['f1_at_5']:.4f}"

)


print(

    "Exact Match Accuracy: "

    f"{keyword_metrics['exact_match_accuracy']:.4f}"

)


print(

    "mAP: "

    f"{keyword_metrics['map']:.4f}"

)


print()

print("-" * 80)

print(
    "EXECUTION TIME"
)

print("-" * 80)


print()

print(

    f"{execution_time:.2f} seconds"

)


print(

    f"{execution_time / 60:.2f} minutes"

)


print()

print("=" * 80)

print(
    "RESULT FILES"
)

print("=" * 80)


print()

print(PREDICTIONS_PATH)


print()

print(METRICS_PATH)


print()

print(SUMMARY_PATH)


print()

print("=" * 80)

print(
    "NEXT STEP"
)

print("=" * 80)


print()

print(
    "Proceed to dashboard development."
)


print()
