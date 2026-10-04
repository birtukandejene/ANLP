# ============================================================
# 30_amharic_document_retrieval_evaluation.py
#
# FAST HIERARCHICAL DOCUMENT RETRIEVAL
# FOR AMHARIC DOMAIN + TOPIC + KEYWORD PREDICTION
#
# MODEL:
#   rasyosef/embedding-amharic-base
#
# INPUT:
#   models/amharic_semantic_index.joblib
#   data/train/train_processed.xlsx
#   data/train/validation_processed.xlsx
#
# OUTPUT:
#   results/document_retrieval_validation/
#
# PREDICTS:
#   1. Domain
#   2. Top-K Topics
#   3. Top-K Keywords
#
# EVALUATES:
#
# DOMAIN:
#   Accuracy
#   Precision
#   Recall
#   F1
#
# TOPICS:
#   Precision@K
#   Recall@K
#   F1@K
#   Exact Match Accuracy
#   mAP
#
# KEYWORDS:
#   Precision@K
#   Recall@K
#   F1@K
#   Exact Match Accuracy
#   mAP
#
# IMPORTANT:
#   This script uses vectorized NumPy matrix operations.
#   It is designed to be significantly faster than
#   Python nested-loop retrieval.
# ============================================================


import sys
import os
import subprocess
import importlib
from pathlib import Path
import warnings
import time
import json
import re
from collections import defaultdict

warnings.filterwarnings("ignore")


# ============================================================
# INSTALL MISSING PACKAGES
# ============================================================

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
install_if_missing("scikit-learn", "sklearn")


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


SEMANTIC_INDEX_PATH = Path(os.environ.get(
    "ANLP_EVAL_INDEX_PATH",
    str(PROJECT_ROOT / "models" / "amharic_semantic_index.joblib"),
))


TRAIN_PATH = Path(os.environ.get(
    "ANLP_EVAL_TRAIN_PATH",
    str(PROJECT_ROOT / "data" / "train" / "train_processed.xlsx"),
))


VALIDATION_PATH = Path(os.environ.get(
    "ANLP_EVAL_VALIDATION_PATH",
    str(PROJECT_ROOT / "data" / "train" / "validation_processed.xlsx"),
))


RESULTS_DIR = Path(os.environ.get(
    "ANLP_EVAL_RESULTS_DIR",
    str(PROJECT_ROOT / "results" / "document_retrieval_validation"),
))


PREDICTIONS_PATH = (
    RESULTS_DIR
    / "document_retrieval_predictions.xlsx"
)


METRICS_PATH = (
    RESULTS_DIR
    / "document_retrieval_metrics.json"
)


SUMMARY_PATH = (
    RESULTS_DIR
    / "document_retrieval_summary.txt"
)


DOMAIN_REPORT_PATH = (
    RESULTS_DIR
    / "domain_classification_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "rasyosef/embedding-amharic-base"
)


TEXT_COLUMN = "text"

DOMAIN_COLUMN = "domain"

TOPICS_COLUMN = "topics"

KEYWORDS_COLUMN = "keywords"


# ------------------------------------------------------------
# EMBEDDING
# ------------------------------------------------------------

BATCH_SIZE = 32


# ------------------------------------------------------------
# DOMAIN RETRIEVAL
#
# Number of candidate domains.
#
# Using more than one domain can improve topic/keyword recall.
# ------------------------------------------------------------

TOP_DOMAIN_K = 3


# ------------------------------------------------------------
# DOCUMENT RETRIEVAL
#
# Number of similar training documents.
# ------------------------------------------------------------

TOP_DOCUMENT_K = 30


# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

TOP_TOPIC_K = 5

TOP_KEYWORD_K = 5


# ------------------------------------------------------------
# SIMILARITY
#
# Ignore very weak document matches.
# ------------------------------------------------------------

MIN_DOCUMENT_SIMILARITY = 0.25


# ------------------------------------------------------------
# WEIGHTING
#
# Topic and keyword scores are based on:
#
# cosine_similarity ^ SIMILARITY_POWER
#
# Larger values emphasize highly similar documents.
# ------------------------------------------------------------

SIMILARITY_POWER = 2.0


# ------------------------------------------------------------
# DOMAIN BONUS
#
# Documents from the best predicted domain receive
# a small bonus.
# ------------------------------------------------------------

BEST_DOMAIN_BONUS = 1.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):

        if np.isnan(value):
            return ""

    return " ".join(
        str(value).strip().split()
    )


# ============================================================
# PARSE MULTI-LABEL VALUES
#
# Supports:
#
# topic1|topic2
#
# topic1,topic2
#
# ["topic1", "topic2"]
#
# ["topic1"]
# ============================================================

def parse_labels(value):

    if value is None:
        return []

    if isinstance(value, float):

        if np.isnan(value):
            return []

    text = str(value).strip()

    if not text:
        return []

    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .replace("'", "")
    )

    parts = re.split(
        r"\||,|;",
        text
    )

    labels = []

    for item in parts:

        item = clean_text(item)

        if item:

            if item not in labels:

                labels.append(item)

    return labels


# ============================================================
# NORMALIZE MATRIX
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

    return (
        matrix / norms
    )


# ============================================================
# NORMALIZE VECTOR
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        vector
    )

    if norm == 0:

        return vector

    return vector / norm


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, dataset_name):

    print()
    print("=" * 80)

    print(
        f"LOADING {dataset_name.upper()} DATA"
    )

    print("=" * 80)

    print()
    print("File:")

    print(path)


    if not path.exists():

        print()
        print("ERROR: File not found.")

        print(path)

        sys.exit(1)


    df = pd.read_excel(
        path,
        engine="openpyxl"
    )


    print()
    print(
        "Original rows:",
        f"{len(df):,}"
    )


    print()
    print("Columns:")

    print(
        list(df.columns)
    )


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [

        TEXT_COLUMN,

        DOMAIN_COLUMN,

        TOPICS_COLUMN,

        KEYWORDS_COLUMN,

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        print()

        print(
            "ERROR: Missing columns:"
        )

        print(
            missing_columns
        )

        sys.exit(1)


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df[TEXT_COLUMN] = (

        df[TEXT_COLUMN]

        .apply(clean_text)

    )


    df[DOMAIN_COLUMN] = (

        df[DOMAIN_COLUMN]

        .apply(clean_text)

    )


    # --------------------------------------------------------
    # REMOVE EMPTY TEXT
    # --------------------------------------------------------

    before = len(df)


    df = df[

        df[TEXT_COLUMN].str.len() > 0

    ].copy()


    print()

    print(
        "Removed empty texts:",
        before - len(df)
    )


    # --------------------------------------------------------
    # REMOVE EMPTY DOMAIN
    # --------------------------------------------------------

    before = len(df)


    df = df[

        df[DOMAIN_COLUMN].str.len() > 0

    ].copy()


    print()

    print(
        "Removed empty domains:",
        before - len(df)
    )


    # --------------------------------------------------------
    # PARSE LABELS
    # --------------------------------------------------------

    df[TOPICS_COLUMN] = (

        df[TOPICS_COLUMN]

        .apply(parse_labels)

    )


    df[KEYWORDS_COLUMN] = (

        df[KEYWORDS_COLUMN]

        .apply(parse_labels)

    )


    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )


    print()

    print(
        "Final documents:",
        f"{len(df):,}"
    )


    return df


# ============================================================
# ENCODE VALIDATION TEXTS
# ============================================================

def encode_texts(model, texts):

    print()

    print("=" * 80)

    print(
        "ENCODING VALIDATION DOCUMENTS"
    )

    print("=" * 80)


    print()

    print(
        "Documents:",
        f"{len(texts):,}"
    )


    print()

    print(
        "Batch size:",
        BATCH_SIZE
    )


    print()


    start_time = time.time()


    if hasattr(
        model,
        "encode_document"
    ):

        embeddings = (

            model.encode_document(

                texts,

                batch_size=BATCH_SIZE,

                show_progress_bar=True,

                convert_to_numpy=True,

                normalize_embeddings=True,

            )

        )

    else:

        embeddings = (

            model.encode(

                texts,

                batch_size=BATCH_SIZE,

                show_progress_bar=True,

                convert_to_numpy=True,

                normalize_embeddings=True,

            )

        )


    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )


    embeddings = normalize_matrix(
        embeddings
    )


    elapsed = (
        time.time()
        - start_time
    )


    print()

    print(
        f"Encoding time: {elapsed:.2f} seconds"
    )


    print()

    print(
        "Embedding shape:",
        embeddings.shape
    )


    return embeddings


# ============================================================
# BUILD DOMAIN REFERENCE MATRIX
# ============================================================

def build_domain_matrix(domain_centroids):

    domain_names = sorted(
        domain_centroids.keys()
    )


    domain_vectors = np.vstack(

        [

            domain_centroids[
                domain
            ]

            for domain in domain_names

        ]

    )


    domain_vectors = normalize_matrix(
        domain_vectors
    )


    return (

        domain_names,

        domain_vectors,

    )


# ============================================================
# PREPARE TRAINING DOCUMENT STRUCTURE
# ============================================================

def prepare_training_structure(
    train_embeddings,
    train_domains,
    train_topics,
    train_keywords,
):

    print()

    print("=" * 80)

    print(
        "PREPARING TRAINING DOCUMENT RETRIEVAL INDEX"
    )

    print("=" * 80)


    domain_to_indices = defaultdict(list)


    for index, domain in enumerate(

        train_domains

    ):

        domain_to_indices[
            domain
        ].append(index)


    # Convert lists to NumPy arrays

    for domain in domain_to_indices:

        domain_to_indices[
            domain
        ] = np.asarray(

            domain_to_indices[
                domain
            ],

            dtype=np.int32

        )


    print()

    print(
        "Training documents:",
        f"{len(train_embeddings):,}"
    )


    print()

    print(
        "Domains:",
        len(domain_to_indices)
    )


    print()

    for domain in sorted(
        domain_to_indices.keys()
    ):

        print(

            f"  {domain}: "
            f"{len(domain_to_indices[domain]):,}"

        )


    return domain_to_indices


# ============================================================
# GET TOP DOMAINS
# ============================================================

def get_top_domains(

    query_vector,

    domain_names,

    domain_vectors,

    top_k,

):

    similarities = (

        domain_vectors
        @
        query_vector

    )


    top_k = min(
        top_k,
        len(domain_names)
    )


    top_indices = np.argpartition(

        -similarities,

        top_k - 1

    )[:top_k]


    top_indices = (

        top_indices[

            np.argsort(

                -similarities[
                    top_indices
                ]

            )

        ]

    )


    results = []


    for index in top_indices:

        results.append(

            (

                domain_names[index],

                float(
                    similarities[index]
                ),

            )

        )


    return results


# ============================================================
# RETRIEVE SIMILAR TRAINING DOCUMENTS
# ============================================================

def retrieve_documents(

    query_vector,

    candidate_domains,

    domain_to_indices,

    train_embeddings,

    top_k,

):

    # --------------------------------------------------------
    # COLLECT DOCUMENT INDICES
    # --------------------------------------------------------

    candidate_indices = []


    for domain, score in candidate_domains:

        if domain in domain_to_indices:

            candidate_indices.extend(

                domain_to_indices[
                    domain
                ].tolist()

            )


    if not candidate_indices:

        return (

            np.array(
                [],
                dtype=np.int32
            ),

            np.array(
                [],
                dtype=np.float32
            ),

        )


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    candidate_indices = np.asarray(

        list(
            dict.fromkeys(
                candidate_indices
            )
        ),

        dtype=np.int32

    )


    # --------------------------------------------------------
    # VECTOR SIMILARITY
    #
    # This is the fast part.
    #
    # Matrix multiplication:
    #
    # candidate_embeddings @ query_vector
    # --------------------------------------------------------

    candidate_embeddings = (

        train_embeddings[
            candidate_indices
        ]

    )


    similarities = (

        candidate_embeddings
        @
        query_vector

    )


    # --------------------------------------------------------
    # FILTER WEAK DOCUMENTS
    # --------------------------------------------------------

    valid_mask = (

        similarities
        >=
        MIN_DOCUMENT_SIMILARITY

    )


    candidate_indices = (

        candidate_indices[
            valid_mask
        ]

    )


    similarities = (

        similarities[
            valid_mask
        ]

    )


    if len(candidate_indices) == 0:

        return (

            np.array(
                [],
                dtype=np.int32
            ),

            np.array(
                [],
                dtype=np.float32
            ),

        )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_k = min(

        top_k,

        len(candidate_indices)

    )


    top_positions = np.argpartition(

        -similarities,

        top_k - 1

    )[:top_k]


    top_positions = (

        top_positions[

            np.argsort(

                -similarities[
                    top_positions
                ]

            )

        ]

    )


    return (

        candidate_indices[
            top_positions
        ],

        similarities[
            top_positions
        ],

    )


# ============================================================
# WEIGHTED LABEL VOTING
#
# Used for BOTH:
#
# Topics
# Keywords
# ============================================================

def weighted_label_voting(

    retrieved_indices,

    similarities,

    train_labels,

    train_domains,

    best_domain,

    top_k,

):

    label_scores = defaultdict(float)


    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if len(retrieved_indices) == 0:

        return []


    # --------------------------------------------------------
    # WEIGHTED VOTING
    # --------------------------------------------------------

    for document_index, similarity in zip(

        retrieved_indices,

        similarities,

    ):


        similarity = float(
            similarity
        )


        # ----------------------------------------------------
        # SIMILARITY WEIGHT
        # ----------------------------------------------------

        weight = (

            max(
                similarity,
                0.0
            )

            **

            SIMILARITY_POWER

        )


        # ----------------------------------------------------
        # BEST DOMAIN BONUS
        # ----------------------------------------------------

        if (

            train_domains[
                document_index
            ]

            ==

            best_domain

        ):

            weight *= (
                BEST_DOMAIN_BONUS
            )


        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        labels = (

            train_labels[
                document_index
            ]

        )


        for label in labels:

            label = clean_text(
                label
            )


            if not label:

                continue


            label_scores[
                label
            ] += weight


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked = sorted(

        label_scores.items(),

        key=lambda item: item[1],

        reverse=True,

    )


    # --------------------------------------------------------
    # RETURN TOP K
    # --------------------------------------------------------

    return [

        label

        for label, score in ranked[:top_k]

    ]


# ============================================================
# DOMAIN PREDICTION
# ============================================================

def predict_domain(

    query_vector,

    domain_names,

    domain_vectors,

):

    similarities = (

        domain_vectors
        @
        query_vector

    )


    best_index = int(

        np.argmax(
            similarities
        )

    )


    return (

        domain_names[
            best_index
        ],

        float(
            similarities[
                best_index
            ]
        ),

    )


# ============================================================
# MULTI-LABEL PRECISION@K
# ============================================================

def precision_at_k(

    true_labels,

    predicted_labels,

    k,

):

    predicted = predicted_labels[:k]


    if not predicted:

        return 0.0


    true_set = set(
        true_labels
    )


    correct = sum(

        1

        for label in predicted

        if label in true_set

    )


    return (

        correct
        /
        len(predicted)

    )


# ============================================================
# MULTI-LABEL RECALL@K
# ============================================================

def recall_at_k(

    true_labels,

    predicted_labels,

    k,

):

    if not true_labels:

        return 0.0


    true_set = set(
        true_labels
    )


    predicted_set = set(

        predicted_labels[:k]

    )


    correct = len(

        true_set
        &
        predicted_set

    )


    return (

        correct
        /
        len(true_set)

    )


# ============================================================
# MULTI-LABEL F1@K
# ============================================================

def f1_at_k(

    true_labels,

    predicted_labels,

    k,

):

    precision = precision_at_k(

        true_labels,

        predicted_labels,

        k,

    )


    recall = recall_at_k(

        true_labels,

        predicted_labels,

        k,

    )


    if (

        precision + recall
        ==
        0

    ):

        return 0.0


    return (

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


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(

    true_labels,

    predicted_labels,

):

    return int(

        set(true_labels)

        ==

        set(predicted_labels)

    )


# ============================================================
# AVERAGE PRECISION
# ============================================================

def average_precision_at_k(

    true_labels,

    predicted_labels,

    k,

):

    true_set = set(
        true_labels
    )


    if not true_set:

        return 0.0


    score = 0.0

    correct = 0


    predictions = (
        predicted_labels[:k]
    )


    for rank, label in enumerate(

        predictions,

        start=1

    ):


        if label in true_set:

            correct += 1


            precision = (

                correct
                /
                rank

            )


            score += precision


    denominator = min(

        len(true_set),

        k

    )


    if denominator == 0:

        return 0.0


    return (

        score
        /
        denominator

    )


# ============================================================
# EVALUATE MULTI-LABEL RESULTS
# ============================================================

def evaluate_multilabel(

    true_lists,

    predicted_lists,

    k,

):

    precisions = []

    recalls = []

    f1_scores = []

    exact_matches = []

    average_precisions = []


    for true_labels, predicted_labels in zip(

        true_lists,

        predicted_lists,

    ):


        precisions.append(

            precision_at_k(

                true_labels,

                predicted_labels,

                k,

            )

        )


        recalls.append(

            recall_at_k(

                true_labels,

                predicted_labels,

                k,

            )

        )


        f1_scores.append(

            f1_at_k(

                true_labels,

                predicted_labels,

                k,

            )

        )


        exact_matches.append(

            exact_match(

                true_labels,

                predicted_labels,

            )

        )


        average_precisions.append(

            average_precision_at_k(

                true_labels,

                predicted_labels,

                k,

            )

        )


    return {

        "precision_at_k":

            float(
                np.mean(
                    precisions
                )
            ),

        "recall_at_k":

            float(
                np.mean(
                    recalls
                )
            ),

        "f1_at_k":

            float(
                np.mean(
                    f1_scores
                )
            ),

        "exact_match_accuracy":

            float(
                np.mean(
                    exact_matches
                )
            ),

        "mAP":

            float(
                np.mean(
                    average_precisions
                )
            ),

    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()


    print()

    print("=" * 80)

    print(
        "AMHARIC FAST HIERARCHICAL DOCUMENT RETRIEVAL"
    )

    print("=" * 80)


    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )


    print()

    print(
        "Semantic index:"
    )

    print(
        SEMANTIC_INDEX_PATH
    )


    print()

    print(
        "Training data:"
    )

    print(
        TRAIN_PATH
    )


    print()

    print(
        "Validation data:"
    )

    print(
        VALIDATION_PATH
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [

        SEMANTIC_INDEX_PATH,

        TRAIN_PATH,

        VALIDATION_PATH,

    ]


    for file_path in required_files:

        if not file_path.exists():

            print()

            print(
                "ERROR: Required file not found."
            )

            print(
                file_path
            )

            sys.exit(1)


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    # ========================================================
    # LOAD SEMANTIC INDEX
    # ========================================================

    print()

    print("=" * 80)

    print(
        "LOADING SEMANTIC INDEX"
    )

    print("=" * 80)


    semantic_index = joblib.load(

        SEMANTIC_INDEX_PATH

    )


    print()

    print(
        "Index loaded successfully."
    )


    train_embeddings = np.asarray(

        semantic_index[
            "embeddings"
        ],

        dtype=np.float32,

    )


    train_embeddings = normalize_matrix(

        train_embeddings

    )


    print()

    print(

        "Documents indexed:",

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

            MODEL_NAME,

        )

    )


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    train_df = load_dataset(

        TRAIN_PATH,

        "training",

    )


    validation_df = load_dataset(

        VALIDATION_PATH,

        "validation",

    )


    # ========================================================
    # IMPORTANT CONSISTENCY CHECK
    # ========================================================

    if (

        len(train_df)

        !=

        len(train_embeddings)

    ):

        print()

        print("=" * 80)

        print(
            "ERROR: TRAINING DATA AND SEMANTIC INDEX DO NOT MATCH"
        )

        print("=" * 80)


        print()

        print(

            "Training Excel documents:",

            len(train_df)

        )


        print(

            "Semantic index documents:",

            len(train_embeddings)

        )


        print()

        print(
            "The index must be built from the exact same "
            "training dataset and document order."
        )


        sys.exit(1)


    # ========================================================
    # EXTRACT TRAINING DATA
    # ========================================================

    train_domains = (

        train_df[
            DOMAIN_COLUMN
        ].tolist()

    )


    train_topics = (

        train_df[
            TOPICS_COLUMN
        ].tolist()

    )


    train_keywords = (

        train_df[
            KEYWORDS_COLUMN
        ].tolist()

    )


    # ========================================================
    # EXTRACT VALIDATION DATA
    # ========================================================

    validation_texts = (

        validation_df[
            TEXT_COLUMN
        ].tolist()

    )


    true_domains = (

        validation_df[
            DOMAIN_COLUMN
        ].tolist()

    )


    true_topics = (

        validation_df[
            TOPICS_COLUMN
        ].tolist()

    )


    true_keywords = (

        validation_df[
            KEYWORDS_COLUMN
        ].tolist()

    )


    # ========================================================
    # LOAD DOMAIN CENTROIDS
    # ========================================================

    print()

    print("=" * 80)

    print(
        "PREPARING DOMAIN REFERENCE VECTORS"
    )

    print("=" * 80)


    domain_centroids = (

        semantic_index[
            "domain_centroids"
        ]

    )


    (

        domain_names,

        domain_vectors,

    ) = build_domain_matrix(

        domain_centroids

    )


    print()

    print(
        "Domains:",
        len(domain_names)
    )


    # ========================================================
    # PREPARE DOCUMENT INDEX
    # ========================================================

    domain_to_indices = (

        prepare_training_structure(

            train_embeddings,

            train_domains,

            train_topics,

            train_keywords,

        )

    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print("=" * 80)

    print(
        "LOADING AMHARIC EMBEDDING MODEL"
    )

    print("=" * 80)


    print()

    print(
        "Model:",
        MODEL_NAME
    )


    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

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
            "CUDA GPU was not detected."
        )

        print(
            "Running on CPU."
        )


    print()

    print(
        "Loading model..."
    )


    model = SentenceTransformer(

        MODEL_NAME,

        device=device,

    )


    print()

    print(
        "Model loaded successfully."
    )


    # ========================================================
    # ENCODE VALIDATION DATA
    # ========================================================

    validation_embeddings = (

        encode_texts(

            model,

            validation_texts,

        )

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print()

    print("=" * 80)

    print(
        "RUNNING HIERARCHICAL DOCUMENT RETRIEVAL"
    )

    print("=" * 80)


    predicted_domains = []

    predicted_domain_scores = []

    predicted_topics = []

    predicted_keywords = []

    top_domain_details = []

    retrieved_document_counts = []

    retrieved_similarity_max = []


    total_documents = len(

        validation_embeddings

    )


    for index in range(

        total_documents

    ):


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (

            index == 0

            or

            (index + 1) % 100 == 0

            or

            index + 1 == total_documents

        ):

            print(

                f"Processing: "
                f"{index + 1:,}"
                f"/"
                f"{total_documents:,}"

            )


        query_vector = (

            validation_embeddings[
                index
            ]

        )


        # ====================================================
        # DOMAIN PREDICTION
        # ====================================================

        predicted_domain, domain_score = (

            predict_domain(

                query_vector,

                domain_names,

                domain_vectors,

            )

        )


        predicted_domains.append(

            predicted_domain

        )


        predicted_domain_scores.append(

            domain_score

        )


        # ====================================================
        # TOP CANDIDATE DOMAINS
        # ====================================================

        candidate_domains = (

            get_top_domains(

                query_vector,

                domain_names,

                domain_vectors,

                TOP_DOMAIN_K,

            )

        )


        top_domain_details.append(

            " | ".join(

                [

                    f"{domain} ({score:.4f})"

                    for domain, score

                    in candidate_domains

                ]

            )

        )


        # ====================================================
        # DOCUMENT RETRIEVAL
        # ====================================================

        retrieved_indices, similarities = (

            retrieve_documents(

                query_vector,

                candidate_domains,

                domain_to_indices,

                train_embeddings,

                TOP_DOCUMENT_K,

            )

        )


        retrieved_document_counts.append(

            int(
                len(retrieved_indices)
            )

        )


        if len(similarities) > 0:

            retrieved_similarity_max.append(

                float(
                    np.max(
                        similarities
                    )
                )

            )

        else:

            retrieved_similarity_max.append(

                0.0

            )


        # ====================================================
        # TOPIC PREDICTION
        # ====================================================

        topics = (

            weighted_label_voting(

                retrieved_indices,

                similarities,

                train_topics,

                train_domains,

                predicted_domain,

                TOP_TOPIC_K,

            )

        )


        predicted_topics.append(

            topics

        )


        # ====================================================
        # KEYWORD PREDICTION
        # ====================================================

        keywords = (

            weighted_label_voting(

                retrieved_indices,

                similarities,

                train_keywords,

                train_domains,

                predicted_domain,

                TOP_KEYWORD_K,

            )

        )


        predicted_keywords.append(

            keywords

        )


    # ========================================================
    # DOMAIN EVALUATION
    # ========================================================

    print()

    print("=" * 80)

    print(
        "DOMAIN EVALUATION"
    )

    print("=" * 80)


    domain_accuracy = accuracy_score(

        true_domains,

        predicted_domains,

    )


    domain_precision = precision_score(

        true_domains,

        predicted_domains,

        average="weighted",

        zero_division=0,

    )


    domain_recall = recall_score(

        true_domains,

        predicted_domains,

        average="weighted",

        zero_division=0,

    )


    domain_f1 = f1_score(

        true_domains,

        predicted_domains,

        average="weighted",

        zero_division=0,

    )


    print()

    print(

        f"Accuracy : {domain_accuracy:.4f}"

    )

    print(

        f"Precision: {domain_precision:.4f}"

    )

    print(

        f"Recall   : {domain_recall:.4f}"

    )

    print(

        f"F1 Score : {domain_f1:.4f}"

    )


    # ========================================================
    # DOMAIN CLASSIFICATION REPORT
    # ========================================================

    domain_report = classification_report(

        true_domains,

        predicted_domains,

        zero_division=0,

        digits=4,

    )


    print()

    print("-" * 80)

    print(
        "DOMAIN CLASSIFICATION REPORT"
    )

    print("-" * 80)

    print()

    print(
        domain_report
    )


    # ========================================================
    # TOPIC EVALUATION
    # ========================================================

    print()

    print("=" * 80)

    print(
        "TOPIC EVALUATION"
    )

    print("=" * 80)


    topic_metrics = (

        evaluate_multilabel(

            true_topics,

            predicted_topics,

            TOP_TOPIC_K,

        )

    )


    print()

    print(

        f"Precision@{TOP_TOPIC_K}"
        f"         : "
        f"{topic_metrics['precision_at_k']:.4f}"

    )

    print(

        f"Recall@{TOP_TOPIC_K}"
        f"            : "
        f"{topic_metrics['recall_at_k']:.4f}"

    )

    print(

        f"F1@{TOP_TOPIC_K}"
        f"                : "
        f"{topic_metrics['f1_at_k']:.4f}"

    )

    print(

        "Exact Match Accuracy"
        f"   : "
        f"{topic_metrics['exact_match_accuracy']:.4f}"

    )

    print(

        "mAP"
        f"                    : "
        f"{topic_metrics['mAP']:.4f}"

    )


    # ========================================================
    # KEYWORD EVALUATION
    # ========================================================

    print()

    print("=" * 80)

    print(
        "KEYWORD EVALUATION"
    )

    print("=" * 80)


    keyword_metrics = (

        evaluate_multilabel(

            true_keywords,

            predicted_keywords,

            TOP_KEYWORD_K,

        )

    )


    print()

    print(

        f"Precision@{TOP_KEYWORD_K}"
        f"         : "
        f"{keyword_metrics['precision_at_k']:.4f}"

    )

    print(

        f"Recall@{TOP_KEYWORD_K}"
        f"            : "
        f"{keyword_metrics['recall_at_k']:.4f}"

    )

    print(

        f"F1@{TOP_KEYWORD_K}"
        f"                : "
        f"{keyword_metrics['f1_at_k']:.4f}"

    )

    print(

        "Exact Match Accuracy"
        f"   : "
        f"{keyword_metrics['exact_match_accuracy']:.4f}"

    )

    print(

        "mAP"
        f"                    : "
        f"{keyword_metrics['mAP']:.4f}"

    )


    # ========================================================
    # DATASET OVERLAP ANALYSIS
    # ========================================================

    train_topic_set = set(

        topic

        for topic_list

        in train_topics

        for topic

        in topic_list

    )


    validation_topic_set = set(

        topic

        for topic_list

        in true_topics

        for topic

        in topic_list

    )


    common_topics = (

        train_topic_set

        &

        validation_topic_set

    )


    unseen_topics = (

        validation_topic_set

        -

        train_topic_set

    )


    overlap_percentage = (

        (
            len(common_topics)

            /

            len(validation_topic_set)

        )

        *

        100

        if validation_topic_set

        else 0.0

    )


    print()

    print("=" * 80)

    print(
        "DATASET TOPIC OVERLAP"
    )

    print("=" * 80)


    print()

    print(

        "Training unique topics:",

        f"{len(train_topic_set):,}"

    )


    print(

        "Validation unique topics:",

        f"{len(validation_topic_set):,}"

    )


    print(

        "Common topics:",

        f"{len(common_topics):,}"

    )


    print(

        "Unseen validation topics:",

        f"{len(unseen_topics):,}"

    )


    print(

        "Validation topic overlap:",

        f"{overlap_percentage:.2f}%"

    )


    # ========================================================
    # SEEN TOPIC DOCUMENT EVALUATION
    #
    # Evaluate only validation documents whose topics
    # all exist in training.
    # ========================================================

    seen_true_topics = []

    seen_predicted_topics = []


    for true_list, predicted_list in zip(

        true_topics,

        predicted_topics,

    ):

        if (

            set(true_list)

            <=

            train_topic_set

        ):

            seen_true_topics.append(

                true_list

            )


            seen_predicted_topics.append(

                predicted_list

            )


    if seen_true_topics:

        seen_topic_metrics = (

            evaluate_multilabel(

                seen_true_topics,

                seen_predicted_topics,

                TOP_TOPIC_K,

            )

        )

    else:

        seen_topic_metrics = {

            "precision_at_k": 0.0,

            "recall_at_k": 0.0,

            "f1_at_k": 0.0,

            "exact_match_accuracy": 0.0,

            "mAP": 0.0,

        }


    print()

    print(

        "Validation documents with ALL "
        "topics seen in training:",

        f"{len(seen_true_topics):,}"

    )


    print()

    print(

        f"Seen-topic document F1@{TOP_TOPIC_K}: "
        f"{seen_topic_metrics['f1_at_k']:.4f}"

    )


    # ========================================================
    # CREATE PREDICTION DATAFRAME
    # ========================================================

    print()

    print("=" * 80)

    print(
        "SAVING RESULTS"
    )

    print("=" * 80)


    prediction_df = pd.DataFrame({

        "text":

            validation_texts,


        "true_domain":

            true_domains,


        "predicted_domain":

            predicted_domains,


        "domain_score":

            predicted_domain_scores,


        "candidate_domains":

            top_domain_details,


        "true_topics":

            [

                " | ".join(labels)

                for labels

                in true_topics

            ],


        "predicted_topics":

            [

                " | ".join(labels)

                for labels

                in predicted_topics

            ],


        "true_keywords":

            [

                " | ".join(labels)

                for labels

                in true_keywords

            ],


        "predicted_keywords":

            [

                " | ".join(labels)

                for labels

                in predicted_keywords

            ],


        "retrieved_documents":

            retrieved_document_counts,


        "maximum_similarity":

            retrieved_similarity_max,

    })


    prediction_df.to_excel(

        PREDICTIONS_PATH,

        index=False,

        engine="openpyxl",

    )


    print()

    print(
        "Predictions saved:"
    )

    print(
        PREDICTIONS_PATH
    )


    # ========================================================
    # METRICS
    # ========================================================

    execution_time = (

        time.time()

        -

        start_time

    )


    metrics = {


        "configuration": {

            "model":

                MODEL_NAME,


            "top_domain_k":

                TOP_DOMAIN_K,


            "top_document_k":

                TOP_DOCUMENT_K,


            "top_topic_k":

                TOP_TOPIC_K,


            "top_keyword_k":

                TOP_KEYWORD_K,


            "minimum_document_similarity":

                MIN_DOCUMENT_SIMILARITY,


            "similarity_power":

                SIMILARITY_POWER,


            "best_domain_bonus":

                BEST_DOMAIN_BONUS,

        },


        "dataset": {

            "training_documents":

                int(len(train_df)),


            "validation_documents":

                int(len(validation_df)),

        },


        "domain": {

            "accuracy":

                float(domain_accuracy),


            "precision":

                float(domain_precision),


            "recall":

                float(domain_recall),


            "f1":

                float(domain_f1),

        },


        "topics": {

            "precision_at_k":

                topic_metrics[
                    "precision_at_k"
                ],


            "recall_at_k":

                topic_metrics[
                    "recall_at_k"
                ],


            "f1_at_k":

                topic_metrics[
                    "f1_at_k"
                ],


            "exact_match_accuracy":

                topic_metrics[
                    "exact_match_accuracy"
                ],


            "mAP":

                topic_metrics[
                    "mAP"
                ],

        },


        "keywords": {

            "precision_at_k":

                keyword_metrics[
                    "precision_at_k"
                ],


            "recall_at_k":

                keyword_metrics[
                    "recall_at_k"
                ],


            "f1_at_k":

                keyword_metrics[
                    "f1_at_k"
                ],


            "exact_match_accuracy":

                keyword_metrics[
                    "exact_match_accuracy"
                ],


            "mAP":

                keyword_metrics[
                    "mAP"
                ],

        },


        "topic_overlap": {

            "training_unique_topics":

                int(
                    len(train_topic_set)
                ),


            "validation_unique_topics":

                int(
                    len(validation_topic_set)
                ),


            "common_topics":

                int(
                    len(common_topics)
                ),


            "unseen_validation_topics":

                int(
                    len(unseen_topics)
                ),


            "validation_topic_overlap_percentage":

                float(
                    overlap_percentage
                ),


            "documents_with_all_topics_seen":

                int(
                    len(seen_true_topics)
                ),


            "seen_topic_precision_at_k":

                seen_topic_metrics[
                    "precision_at_k"
                ],


            "seen_topic_recall_at_k":

                seen_topic_metrics[
                    "recall_at_k"
                ],


            "seen_topic_f1_at_k":

                seen_topic_metrics[
                    "f1_at_k"
                ],


            "seen_topic_mAP":

                seen_topic_metrics[
                    "mAP"
                ],

        },


        "execution_time_seconds":

            float(
                execution_time
            ),

    }


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(

        METRICS_PATH,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            metrics,

            file,

            ensure_ascii=False,

            indent=4,

        )


    print()

    print(
        "Metrics saved:"
    )

    print(
        METRICS_PATH
    )


    # ========================================================
    # SAVE DOMAIN REPORT
    # ========================================================

    with open(

        DOMAIN_REPORT_PATH,

        "w",

        encoding="utf-8",

    ) as file:

        file.write(
            domain_report
        )


    print()

    print(
        "Domain report saved:"
    )

    print(
        DOMAIN_REPORT_PATH
    )


    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    summary = f"""

================================================================================
AMHARIC HIERARCHICAL DOCUMENT RETRIEVAL EVALUATION
================================================================================

VALIDATION DOCUMENTS: {len(validation_df):,}

TRAINING DOCUMENTS: {len(train_df):,}


================================================================================
CONFIGURATION
================================================================================

Model: {MODEL_NAME}

Top candidate domains: {TOP_DOMAIN_K}

Top retrieved documents: {TOP_DOCUMENT_K}

Top topics: {TOP_TOPIC_K}

Top keywords: {TOP_KEYWORD_K}

Minimum document similarity: {MIN_DOCUMENT_SIMILARITY}

Similarity power: {SIMILARITY_POWER}

Best domain bonus: {BEST_DOMAIN_BONUS}


================================================================================
DOMAIN RESULTS
================================================================================

Accuracy : {domain_accuracy:.4f}

Precision: {domain_precision:.4f}

Recall   : {domain_recall:.4f}

F1 Score : {domain_f1:.4f}


================================================================================
TOPIC RESULTS
================================================================================

Precision@{TOP_TOPIC_K}: {topic_metrics['precision_at_k']:.4f}

Recall@{TOP_TOPIC_K}: {topic_metrics['recall_at_k']:.4f}

F1@{TOP_TOPIC_K}: {topic_metrics['f1_at_k']:.4f}

Exact Match Accuracy: {topic_metrics['exact_match_accuracy']:.4f}

mAP: {topic_metrics['mAP']:.4f}


================================================================================
KEYWORD RESULTS
================================================================================

Precision@{TOP_KEYWORD_K}: {keyword_metrics['precision_at_k']:.4f}

Recall@{TOP_KEYWORD_K}: {keyword_metrics['recall_at_k']:.4f}

F1@{TOP_KEYWORD_K}: {keyword_metrics['f1_at_k']:.4f}

Exact Match Accuracy: {keyword_metrics['exact_match_accuracy']:.4f}

mAP: {keyword_metrics['mAP']:.4f}


================================================================================
TOPIC DATASET OVERLAP
================================================================================

Training unique topics: {len(train_topic_set):,}

Validation unique topics: {len(validation_topic_set):,}

Common topics: {len(common_topics):,}

Unseen validation topics: {len(unseen_topics):,}

Validation topic overlap: {overlap_percentage:.2f}%

Validation documents with ALL topics seen:
{len(seen_true_topics):,}

Seen-topic Precision@{TOP_TOPIC_K}:
{seen_topic_metrics['precision_at_k']:.4f}

Seen-topic Recall@{TOP_TOPIC_K}:
{seen_topic_metrics['recall_at_k']:.4f}

Seen-topic F1@{TOP_TOPIC_K}:
{seen_topic_metrics['f1_at_k']:.4f}

Seen-topic mAP:
{seen_topic_metrics['mAP']:.4f}


================================================================================
EXECUTION TIME
================================================================================

{execution_time:.2f} seconds

{execution_time / 60:.2f} minutes


================================================================================
RESULT FILES
================================================================================

Predictions:
{PREDICTIONS_PATH}

Metrics:
{METRICS_PATH}

Domain report:
{DOMAIN_REPORT_PATH}

================================================================================
"""


    with open(

        SUMMARY_PATH,

        "w",

        encoding="utf-8",

    ) as file:

        file.write(
            summary
        )


    print()

    print(
        "Summary saved:"
    )

    print(
        SUMMARY_PATH
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 80)

    print(
        "DOCUMENT RETRIEVAL EVALUATION COMPLETE"
    )

    print("=" * 80)


    print()

    print(

        "Validation documents:",

        f"{len(validation_df):,}"

    )


    print()

    print("-" * 80)

    print(
        "DOMAIN RESULTS"
    )

    print("-" * 80)


    print()

    print(
        f"Accuracy : {domain_accuracy:.4f}"
    )

    print(
        f"Precision: {domain_precision:.4f}"
    )

    print(
        f"Recall   : {domain_recall:.4f}"
    )

    print(
        f"F1 Score : {domain_f1:.4f}"
    )


    print()

    print("-" * 80)

    print(
        "TOPIC RESULTS"
    )

    print("-" * 80)


    print()

    print(

        f"Precision@{TOP_TOPIC_K}: "
        f"{topic_metrics['precision_at_k']:.4f}"

    )

    print(

        f"Recall@{TOP_TOPIC_K}: "
        f"{topic_metrics['recall_at_k']:.4f}"

    )

    print(

        f"F1@{TOP_TOPIC_K}: "
        f"{topic_metrics['f1_at_k']:.4f}"

    )

    print(

        f"Exact Match Accuracy: "
        f"{topic_metrics['exact_match_accuracy']:.4f}"

    )

    print(

        f"mAP: "
        f"{topic_metrics['mAP']:.4f}"

    )


    print()

    print("-" * 80)

    print(
        "KEYWORD RESULTS"
    )

    print("-" * 80)


    print()

    print(

        f"Precision@{TOP_KEYWORD_K}: "
        f"{keyword_metrics['precision_at_k']:.4f}"

    )

    print(

        f"Recall@{TOP_KEYWORD_K}: "
        f"{keyword_metrics['recall_at_k']:.4f}"

    )

    print(

        f"F1@{TOP_KEYWORD_K}: "
        f"{keyword_metrics['f1_at_k']:.4f}"

    )

    print(

        f"Exact Match Accuracy: "
        f"{keyword_metrics['exact_match_accuracy']:.4f}"

    )

    print(

        f"mAP: "
        f"{keyword_metrics['mAP']:.4f}"

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

    print(
        PREDICTIONS_PATH
    )

    print()

    print(
        METRICS_PATH
    )

    print()

    print(
        SUMMARY_PATH
    )

    print()

    print(
        DOMAIN_REPORT_PATH
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()