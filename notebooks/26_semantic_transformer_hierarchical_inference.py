# ==================================================================================================
# AMHARIC NLP
# FAST SEMANTIC TRANSFORMER HYBRID HIERARCHICAL INFERENCE
#
# FEATURES
#
#   1. Existing supervised domain classifier
#   2. Pretrained multilingual semantic transformer
#   3. Fast semantic embedding model
#   4. Persistent embedding cache
#   5. Semantic nearest-neighbor retrieval
#   6. Domain centroid similarity
#   7. Local neighbor domain voting
#   8. Noise-aware hybrid domain decision
#   9. Domain-specific subtopic prediction
#  10. Always returns MAIN TOPIC + SUBTOPIC(S)
#  11. Does NOT return UNKNOWN
#  12. Handles noisy training annotations
#
# RECOMMENDED MODEL:
#
#       intfloat/multilingual-e5-small
#
# Faster than multilingual-e5-base and suitable for CPU inference.
#
# ==================================================================================================

import os
import sys
import re
import gc
import hashlib
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd

from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

SCRIPT_PATH = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_PATH.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

DOMAIN_MODEL_PATH = (
    MODELS_DIR /
    "final_professional_domain_classification_model.joblib"
)

SUBTOPIC_MODEL_PATH = (
    MODELS_DIR /
    "final_domain_aware_subtopic_model.joblib"
)

SEMANTIC_INDEX_PATH = (
    MODELS_DIR /
    "fast_semantic_transformer_hierarchical_index.joblib"
)


# --------------------------------------------------------------------------------------------------
# SEMANTIC MODEL
# --------------------------------------------------------------------------------------------------

# FAST AND HIGHLY RECOMMENDED FOR CPU
SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-small"

# Alternative slower model:
#
# SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-base"


# --------------------------------------------------------------------------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 64

NORMALIZE_EMBEDDINGS = True


# --------------------------------------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# --------------------------------------------------------------------------------------------------

TOP_K_NEIGHBORS = 25

TOP_K_DISPLAY = 5

TOP_K_SUBTOPIC_NEIGHBORS = 40


# --------------------------------------------------------------------------------------------------
# HYBRID DOMAIN WEIGHTS
#
# The supervised model is useful.
#
# But because annotations may be noisy, supervised confidence alone
# must NOT determine the final domain.
#
# Semantic context has stronger importance.
# --------------------------------------------------------------------------------------------------

WEIGHT_SUPERVISED = 0.25

WEIGHT_NEIGHBOR_VOTE = 0.35

WEIGHT_CENTROID = 0.25

WEIGHT_BEST_NEIGHBOR = 0.15


# --------------------------------------------------------------------------------------------------
# SUBTOPIC CONFIGURATION
# --------------------------------------------------------------------------------------------------

MAX_SUBTOPICS = 3

MIN_SUBTOPIC_SCORE = 0.10


# --------------------------------------------------------------------------------------------------
# CACHE VERSION
#
# Change this value if the structure of the saved index changes.
# --------------------------------------------------------------------------------------------------

INDEX_VERSION = "fast_semantic_transformer_v1"


# ==================================================================================================
# DISPLAY FUNCTIONS
# ==================================================================================================

def line(character="=", length=100):
    print(character * length)


def section(title):
    print()
    line("=")
    print(title)
    line("=")
    print()


def subsection(title):
    print()
    line("-")
    print(title)
    line("-")
    print()


# ==================================================================================================
# TEXT CLEANING
# ==================================================================================================

def clean_text(text):

    if text is None:
        return ""

    if isinstance(text, float) and np.isnan(text):
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text


# ==================================================================================================
# DATASET FINGERPRINT
#
# Used to determine whether an existing semantic index belongs
# to the current training dataset.
# ==================================================================================================

def dataset_fingerprint(texts, domains, topics):

    hasher = hashlib.sha256()

    total = len(texts)

    hasher.update(str(total).encode("utf-8"))

    # Sample enough rows to detect dataset changes
    sample_size = min(total, 1000)

    if sample_size == 0:
        return hasher.hexdigest()

    indices = np.linspace(
        0,
        total - 1,
        sample_size,
        dtype=int
    )

    for index in indices:

        text = str(texts[index])

        domain = str(domains[index])

        topic = str(topics[index])

        value = (
            text +
            "||" +
            domain +
            "||" +
            topic
        )

        hasher.update(
            value.encode(
                "utf-8",
                errors="ignore"
            )
        )

    return hasher.hexdigest()


# ==================================================================================================
# FIND TRAINING DATA
# ==================================================================================================

def find_training_dataset():

    section("SEARCHING FOR TRAINING DATA")

    candidates = []

    preferred_paths = [

        PREPARED_DATA_DIR /
        "train_prepared.csv",

        PREPARED_DATA_DIR /
        "train_prepared.xlsx",

        DATA_DIR /
        "train" /
        "train_processed.xlsx"

    ]

    for path in preferred_paths:

        if path.exists():

            candidates.append(path)

    if not candidates:

        for path in PROJECT_ROOT.rglob("*"):

            if not path.is_file():
                continue

            filename = path.name.lower()

            if (
                "train" in filename
                and (
                    path.suffix.lower() == ".csv"
                    or path.suffix.lower() == ".xlsx"
                )
            ):

                candidates.append(path)

    if not candidates:

        raise FileNotFoundError(
            "No training dataset could be found."
        )

    print("Candidate datasets:\n")

    valid_candidates = []

    for path in candidates:

        try:

            if path.suffix.lower() == ".csv":

                dataframe = pd.read_csv(
                    path,
                    nrows=5
                )

            else:

                dataframe = pd.read_excel(
                    path,
                    nrows=5
                )

            print(path)

            print("Columns:")

            print(
                dataframe.columns.tolist()
            )

            print()

            valid_candidates.append(path)

        except Exception as error:

            print(
                f"Could not inspect {path}: {error}"
            )

    if not valid_candidates:

        raise RuntimeError(
            "No readable training dataset found."
        )

    # Prefer prepared training data
    for path in valid_candidates:

        if (
            path.name.lower() ==
            "train_prepared.csv"
        ):

            return path

    return valid_candidates[0]


# ==================================================================================================
# LOAD DATASET
# ==================================================================================================

def load_training_dataset():

    dataset_path = find_training_dataset()

    section("LOADING TRAINING DATA")

    print("Dataset:")

    print(dataset_path)

    print()

    if dataset_path.suffix.lower() == ".csv":

        dataframe = pd.read_csv(
            dataset_path
        )

    elif dataset_path.suffix.lower() == ".xlsx":

        dataframe = pd.read_excel(
            dataset_path
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_path}"
        )

    print("Rows:")

    print(len(dataframe))

    print()

    print("Columns:")

    print(
        dataframe.columns.tolist()
    )

    return dataframe, dataset_path


# ==================================================================================================
# DETECT COLUMNS
# ==================================================================================================

def detect_column(
    dataframe,
    possible_names
):

    lower_columns = {

        str(column).lower(): column

        for column in dataframe.columns

    }

    for name in possible_names:

        if name.lower() in lower_columns:

            return lower_columns[name.lower()]

    return None


def detect_dataset_columns(dataframe):

    text_column = detect_column(

        dataframe,

        [

            "text",
            "content",
            "sentence",
            "document"

        ]

    )

    domain_column = detect_column(

        dataframe,

        [

            "consolidated_domains",
            "domain",
            "main_topic",
            "main_domain"

        ]

    )

    topic_column = detect_column(

        dataframe,

        [

            "topic_canonical",
            "topic_normalized",
            "topic_original",
            "topic",
            "subtopic",
            "topics"

        ]

    )

    if text_column is None:

        raise ValueError(
            "Could not find text column."
        )

    if domain_column is None:

        raise ValueError(
            "Could not find domain column."
        )

    if topic_column is None:

        raise ValueError(
            "Could not find topic/subtopic column."
        )

    return (

        text_column,

        domain_column,

        topic_column

    )


# ==================================================================================================
# PREPARE TRAINING DATA
# ==================================================================================================

def prepare_training_data(
    dataframe,
    trained_domains
):

    section("PREPARING REFERENCE DATA")

    (

        text_column,

        domain_column,

        topic_column

    ) = detect_dataset_columns(
        dataframe
    )

    print("Text column:")

    print(text_column)

    print()

    print("Domain column:")

    print(domain_column)

    print()

    print("Topic column:")

    print(topic_column)

    print()

    working = dataframe[

        [

            text_column,

            domain_column,

            topic_column

        ]

    ].copy()

    working.columns = [

        "text",

        "domain",

        "topic"

    ]

    print("Original documents:")

    print(len(working))

    working["text"] = (

        working["text"]

        .apply(clean_text)

    )

    working["domain"] = (

        working["domain"]

        .apply(clean_text)

    )

    working["topic"] = (

        working["topic"]

        .apply(clean_text)

    )

    working = working[

        working["text"] != ""

    ]

    working = working[

        working["domain"] != ""

    ]

    print()

    print(
        "Documents after cleaning:"
    )

    print(len(working))

    if trained_domains:

        working = working[

            working["domain"].isin(
                trained_domains
            )

        ]

    print()

    print(
        "Documents belonging to trained domains:"
    )

    print(len(working))

    # Remove exact duplicates only.
    #
    # We do NOT aggressively remove similar documents because
    # similar examples may represent useful semantic context.

    before_duplicates = len(working)

    working = working.drop_duplicates(

        subset=[

            "text",

            "domain",

            "topic"

        ]

    )

    removed = (

        before_duplicates -

        len(working)

    )

    print()

    print(
        "Exact duplicate rows removed:"
    )

    print(removed)

    working = (

        working

        .reset_index(drop=True)

    )

    print()

    print(
        "Domain distribution:"
    )

    print(

        working["domain"]

        .value_counts()

    )

    return (

        working,

        text_column,

        domain_column,

        topic_column

    )


# ==================================================================================================
# LOAD DOMAIN MODEL
# ==================================================================================================

def load_domain_model():

    section(
        "LOADING DOMAIN CLASSIFICATION MODEL"
    )

    print("Loading:")

    print(DOMAIN_MODEL_PATH)

    print()

    model_data = joblib.load(
        DOMAIN_MODEL_PATH
    )

    print(
        "Domain model loaded successfully."
    )

    print()

    print("Loaded object type:")

    print(type(model_data))

    if isinstance(model_data, dict):

        print()

        print("Available keys:")

        for key in model_data.keys():

            print(f"  - {key}")

    return model_data


# ==================================================================================================
# LOAD SUBTOPIC MODEL
# ==================================================================================================

def load_subtopic_model():

    section(
        "LOADING DOMAIN-AWARE SUBTOPIC MODEL"
    )

    print("Loading:")

    print(SUBTOPIC_MODEL_PATH)

    print()

    model_data = joblib.load(
        SUBTOPIC_MODEL_PATH
    )

    print(
        "Subtopic model loaded successfully."
    )

    print()

    print("Loaded object type:")

    print(type(model_data))

    if isinstance(model_data, dict):

        print()

        print("Available keys:")

        for key in model_data.keys():

            print(f"  - {key}")

    return model_data


# ==================================================================================================
# EXTRACT DOMAIN MODEL COMPONENTS
# ==================================================================================================

def extract_domain_components(
    domain_model_data
):

    section(
        "EXTRACTING DOMAIN MODEL COMPONENTS"
    )

    if not isinstance(
        domain_model_data,
        dict
    ):

        raise ValueError(
            "Domain model must be a dictionary."
        )

    classifier = domain_model_data.get(
        "model"
    )

    word_vectorizer = (

        domain_model_data.get(
            "word_vectorizer"
        )

    )

    character_vectorizer = (

        domain_model_data.get(
            "character_vectorizer"
        )

    )

    domains = (

        domain_model_data.get(
            "domains",
            []
        )

    )

    if classifier is None:

        raise ValueError(
            "Domain classifier missing."
        )

    if word_vectorizer is None:

        raise ValueError(
            "Word vectorizer missing."
        )

    if character_vectorizer is None:

        raise ValueError(
            "Character vectorizer missing."
        )

    if not domains:

        if hasattr(
            classifier,
            "classes_"
        ):

            domains = list(

                classifier.classes_

            )

    domains = list(domains)

    print("Classifier:")

    print(type(classifier))

    print()

    print("Word vectorizer:")

    print(type(word_vectorizer))

    print()

    print("Character vectorizer:")

    print(type(character_vectorizer))

    print()

    print(
        "Number of domains:"
    )

    print(len(domains))

    print()

    print("Domains:\n")

    for index, domain in enumerate(

        domains,

        start=1

    ):

        print(
            f"{index}. {domain}"
        )

    return {

        "classifier": classifier,

        "word_vectorizer":
            word_vectorizer,

        "character_vectorizer":
            character_vectorizer,

        "domains":
            domains

    }


# ==================================================================================================
# LOAD SEMANTIC TRANSFORMER
# ==================================================================================================

def load_semantic_model():

    section(
        "LOADING PRETRAINED SEMANTIC MODEL"
    )

    print("Model:")

    print(
        SEMANTIC_MODEL_NAME
    )

    print()

    print(
        "This model provides multilingual contextual semantic embeddings."
    )

    print()

    print(
        "The first run may download the model."
    )

    print()

    model = SentenceTransformer(

        SEMANTIC_MODEL_NAME

    )

    print(
        "Pretrained semantic model loaded successfully."
    )

    return model


# ==================================================================================================
# E5 EMBEDDING
#
# E5 models work best with:
#
# Training/reference text:
#
#       passage: ...
#
# Query:
#
#       query: ...
#
# ==================================================================================================

def create_passages(
    texts
):

    return [

        "passage: " +

        clean_text(text)

        for text in texts

    ]


def create_query(
    text
):

    return (

        "query: " +

        clean_text(text)

    )


# ==================================================================================================
# ENCODE TRAINING DOCUMENTS
# ==================================================================================================

def encode_passages(
    semantic_model,
    texts
):

    section(
        "CREATING SEMANTIC EMBEDDINGS"
    )

    total = len(texts)

    print(
        "Number of reference documents:"
    )

    print(total)

    print()

    print("Batch size:")

    print(BATCH_SIZE)

    print()

    passages = create_passages(
        texts
    )

    embeddings = semantic_model.encode(

        passages,

        batch_size=BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=NORMALIZE_EMBEDDINGS

    )

    embeddings = np.asarray(

        embeddings,

        dtype=np.float32

    )

    print()

    print(
        "Embedding matrix shape:"
    )

    print(
        embeddings.shape
    )

    return embeddings


# ==================================================================================================
# DOMAIN CENTROIDS
# ==================================================================================================

def build_domain_centroids(

    embeddings,

    domains,

    trained_domains

):

    section(
        "BUILDING ROBUST DOMAIN CENTROIDS"
    )

    centroids = {}

    domain_indices = {}

    domains_array = np.asarray(
        domains
    )

    for domain in trained_domains:

        indices = np.where(

            domains_array == domain

        )[0]

        if len(indices) == 0:

            continue

        domain_vectors = embeddings[
            indices
        ]

        centroid = np.mean(

            domain_vectors,

            axis=0

        )

        centroid = np.asarray(

            centroid,

            dtype=np.float32

        )

        norm = np.linalg.norm(
            centroid
        )

        if norm > 0:

            centroid = (

                centroid / norm

            )

        centroids[domain] = centroid

        domain_indices[domain] = (

            indices.astype(np.int32)

        )

        print(

            f"{domain}: "

            f"{len(indices)} "

            f"reference documents"

        )

    return (

        centroids,

        domain_indices

    )


# ==================================================================================================
# DOMAIN SUBTOPIC KNOWLEDGE
# ==================================================================================================

def build_subtopic_knowledge(

    dataframe

):

    section(
        "BUILDING DOMAIN-SPECIFIC SUBTOPIC KNOWLEDGE"
    )

    domain_topic_counts = {}

    domain_topics = {}

    for domain, group in (

        dataframe.groupby(
            "domain"
        )

    ):

        topics = (

            group["topic"]

            .apply(clean_text)

        )

        topics = topics[

            topics != ""

        ]

        counts = Counter(
            topics.tolist()
        )

        domain_topic_counts[
            domain
        ] = counts

        domain_topics[
            domain
        ] = list(

            counts.keys()

        )

    print(
        "Domain subtopic statistics built."
    )

    return (

        domain_topic_counts,

        domain_topics

    )


# ==================================================================================================
# BUILD SEMANTIC INDEX
# ==================================================================================================

def build_semantic_index(

    semantic_model,

    trained_domains

):

    section(
        "BUILDING FAST SEMANTIC HIERARCHICAL INDEX"
    )

    dataframe, dataset_path = (

        load_training_dataset()

    )

    (

        dataframe,

        text_column,

        domain_column,

        topic_column

    ) = prepare_training_data(

        dataframe,

        trained_domains

    )

    texts = dataframe[
        "text"
    ].tolist()

    domains = dataframe[
        "domain"
    ].tolist()

    topics = dataframe[
        "topic"
    ].tolist()

    fingerprint = dataset_fingerprint(

        texts,

        domains,

        topics

    )

    embeddings = encode_passages(

        semantic_model,

        texts

    )

    (

        domain_centroids,

        domain_indices

    ) = build_domain_centroids(

        embeddings,

        domains,

        trained_domains

    )

    (

        domain_topic_counts,

        domain_topics

    ) = build_subtopic_knowledge(

        dataframe

    )

    text_to_indices = defaultdict(list)

    for index, text in enumerate(
        texts
    ):

        normalized = clean_text(
            text
        )

        text_to_indices[
            normalized
        ].append(index)

    index = {

        "index_version":
            INDEX_VERSION,

        "semantic_model_name":
            SEMANTIC_MODEL_NAME,

        "dataset_path":
            str(dataset_path),

        "dataset_fingerprint":
            fingerprint,

        "texts":
            texts,

        "domains":
            domains,

        "topics":
            topics,

        "embeddings":
            embeddings,

        "domain_centroids":
            domain_centroids,

        "domain_indices":
            domain_indices,

        "domain_topic_counts":
            domain_topic_counts,

        "domain_topics":
            domain_topics,

        "text_to_indices":
            dict(text_to_indices),

        "trained_domains":
            trained_domains

    }

    section(
        "SAVING SEMANTIC INDEX"
    )

    print("Saving:")

    print(
        SEMANTIC_INDEX_PATH
    )

    print()

    SEMANTIC_INDEX_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        index,

        SEMANTIC_INDEX_PATH,

        compress=3

    )

    print(
        "Semantic index saved successfully."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print()

    print(
        "The next run will load this index."
    )

    print(
        "It will NOT encode all training documents again."
    )

    return index


# ==================================================================================================
# VALIDATE EXISTING INDEX
# ==================================================================================================

def validate_semantic_index(

    index,

    trained_domains

):

    required_keys = [

        "index_version",

        "semantic_model_name",

        "texts",

        "domains",

        "topics",

        "embeddings",

        "domain_centroids",

        "domain_indices",

        "domain_topic_counts",

        "domain_topics",

        "text_to_indices",

        "trained_domains"

    ]

    if not isinstance(index, dict):

        return False

    for key in required_keys:

        if key not in index:

            print(

                f"Missing index key: {key}"

            )

            return False

    if (

        index.get(
            "index_version"
        )

        !=

        INDEX_VERSION

    ):

        print(
            "Index version mismatch."
        )

        return False

    if (

        index.get(
            "semantic_model_name"
        )

        !=

        SEMANTIC_MODEL_NAME

    ):

        print(
            "Semantic model mismatch."
        )

        return False

    if not isinstance(

        index["embeddings"],

        np.ndarray

    ):

        return False

    if len(

        index["texts"]

    ) != len(

        index["embeddings"]

    ):

        return False

    if len(

        index["domains"]

    ) != len(

        index["embeddings"]

    ):

        return False

    if len(

        index["topics"]

    ) != len(

        index["embeddings"]

    ):

        return False

    return True


# ==================================================================================================
# LOAD OR BUILD SEMANTIC INDEX
# ==================================================================================================

def load_or_build_semantic_index(

    semantic_model,

    trained_domains

):

    section(
        "LOADING SEMANTIC TRANSFORMER INDEX"
    )

    if SEMANTIC_INDEX_PATH.exists():

        print(
            "Existing semantic index found."
        )

        print()

        print("Loading:")

        print(
            SEMANTIC_INDEX_PATH
        )

        print()

        try:

            index = joblib.load(

                SEMANTIC_INDEX_PATH

            )

            valid = validate_semantic_index(

                index,

                trained_domains

            )

            if valid:

                print(
                    "Semantic index loaded successfully."
                )

                print()

                print(
                    "Reference documents:"
                )

                print(
                    len(
                        index["texts"]
                    )
                )

                print()

                print(
                    "Embedding shape:"
                )

                print(
                    index[
                        "embeddings"
                    ].shape
                )

                return index

            else:

                print()

                print(
                    "Existing index is incompatible."
                )

                print(
                    "Rebuilding index..."
                )

        except Exception as error:

            print()

            print(
                "Could not load existing index."
            )

            print(error)

            print()

            print(
                "Rebuilding index..."
            )

    else:

        print(
            "No semantic index found."
        )

        print(
            "A new index will be built."
        )

    return build_semantic_index(

        semantic_model,

        trained_domains

    )


# ==================================================================================================
# SUPERVISED FEATURE CREATION
# ==================================================================================================

def create_supervised_features(

    text,

    word_vectorizer,

    character_vectorizer

):

    word_features = (

        word_vectorizer.transform(
            [text]
        )

    )

    character_features = (

        character_vectorizer.transform(
            [text]
        )

    )

    from scipy.sparse import hstack

    features = hstack(

        [

            word_features,

            character_features

        ]

    )

    return features


# ==================================================================================================
# SUPERVISED DOMAIN SCORES
#
# LinearSVC normally provides decision_function,
# not predict_proba.
#
# We convert decision values into relative probabilities
# using softmax.
# ==================================================================================================

def get_supervised_domain_scores(

    text,

    domain_components

):

    classifier = (

        domain_components[
            "classifier"
        ]

    )

    word_vectorizer = (

        domain_components[
            "word_vectorizer"
        ]

    )

    character_vectorizer = (

        domain_components[
            "character_vectorizer"
        ]

    )

    trained_domains = (

        domain_components[
            "domains"
        ]

    )

    features = create_supervised_features(

        text,

        word_vectorizer,

        character_vectorizer

    )

    if hasattr(

        classifier,

        "decision_function"

    ):

        decision_scores = (

            classifier.decision_function(
                features
            )

        )

        decision_scores = np.asarray(

            decision_scores

        )

        if decision_scores.ndim == 1:

            decision_scores = (

                decision_scores.reshape(
                    1,
                    -1
                )

            )

        probabilities = softmax(

            decision_scores[0]

        )

        classes = list(

            classifier.classes_

        )

    elif hasattr(

        classifier,

        "predict_proba"

    ):

        probabilities = (

            classifier.predict_proba(
                features
            )[0]

        )

        classes = list(

            classifier.classes_

        )

    else:

        prediction = (

            classifier.predict(
                features
            )[0]

        )

        probabilities = np.zeros(

            len(trained_domains),

            dtype=np.float32

        )

        classes = list(
            trained_domains
        )

        if prediction in classes:

            index = classes.index(
                prediction
            )

            probabilities[index] = 1.0

    scores = {}

    for domain, probability in zip(

        classes,

        probabilities

    ):

        scores[domain] = float(
            probability
        )

    # Guarantee every trained domain exists

    for domain in trained_domains:

        if domain not in scores:

            scores[domain] = 0.0

    return scores


# ==================================================================================================
# NORMALIZE SCORE DICTIONARY
# ==================================================================================================

def normalize_scores(
    score_dict
):

    if not score_dict:

        return {}

    values = np.asarray(

        list(
            score_dict.values()
        ),

        dtype=np.float64

    )

    minimum = np.min(values)

    maximum = np.max(values)

    if maximum - minimum < 1e-12:

        return {

            key: 0.0

            for key in score_dict

        }

    return {

        key: float(

            (value - minimum)

            /

            (maximum - minimum)

        )

        for key, value in (

            score_dict.items()

        )

    }


# ==================================================================================================
# QUERY EMBEDDING
# ==================================================================================================

def encode_query(

    semantic_model,

    text

):

    query = create_query(
        text
    )

    embedding = semantic_model.encode(

        [query],

        convert_to_numpy=True,

        normalize_embeddings=NORMALIZE_EMBEDDINGS

    )

    embedding = np.asarray(

        embedding,

        dtype=np.float32

    )

    return embedding[0]


# ==================================================================================================
# SEMANTIC RETRIEVAL
#
# Because embeddings are normalized:
#
#       cosine similarity = dot product
#
# This is faster than calling sklearn cosine_similarity
# for every query.
# ==================================================================================================

def semantic_retrieval(

    query_embedding,

    index,

    original_text

):

    embeddings = index[
        "embeddings"
    ]

    similarities = np.dot(

        embeddings,

        query_embedding

    )

    similarities = np.asarray(

        similarities,

        dtype=np.float32

    )

    # ------------------------------------------------------------------
    # EXCLUDE EXACT SAME TRAINING DOCUMENT
    #
    # This prevents similarity = 1.0 when the user enters
    # a text already present in training data.
    # ------------------------------------------------------------------

    normalized_text = clean_text(
        original_text
    )

    same_indices = (

        index[
            "text_to_indices"
        ].get(

            normalized_text,

            []

        )

    )

    if same_indices:

        similarities[
            same_indices
        ] = -1.0

    top_k = min(

        TOP_K_NEIGHBORS,

        len(similarities)

    )

    if top_k <= 0:

        return []

    candidate_indices = np.argpartition(

        similarities,

        -top_k

    )[-top_k:]

    candidate_indices = (

        candidate_indices[

            np.argsort(

                similarities[
                    candidate_indices
                ]

            )[::-1]

        ]

    )

    neighbors = []

    for index_position in (

        candidate_indices

    ):

        similarity = float(

            similarities[
                index_position
            ]

        )

        if similarity < 0:

            continue

        neighbors.append(

            {

                "index":
                    int(index_position),

                "similarity":
                    similarity,

                "text":
                    index["texts"][
                        index_position
                    ],

                "domain":
                    index["domains"][
                        index_position
                    ],

                "topic":
                    index["topics"][
                        index_position
                    ]

            }

        )

    return neighbors


# ==================================================================================================
# SEMANTIC DOMAIN VOTING
#
# Important:
#
# A single nearest document should not decide the domain.
#
# Multiple neighbors are aggregated.
#
# Strong neighbors receive greater weight.
# ==================================================================================================

def calculate_neighbor_domain_scores(

    neighbors,

    trained_domains

):

    scores = {

        domain: 0.0

        for domain in trained_domains

    }

    if not neighbors:

        return scores

    for neighbor in neighbors:

        domain = neighbor[
            "domain"
        ]

        similarity = neighbor[
            "similarity"
        ]

        if domain not in scores:

            continue

        # ----------------------------------------------------------------
        # Similarity weighting
        #
        # Negative similarities do not contribute.
        #
        # Squaring gives stronger semantic matches
        # more influence than weak matches.
        # ----------------------------------------------------------------

        weight = max(
            similarity,
            0.0
        ) ** 2

        scores[domain] += weight

    total = sum(
        scores.values()
    )

    if total > 0:

        scores = {

            domain:
            value / total

            for domain, value in (

                scores.items()

            )

        }

    return scores


# ==================================================================================================
# BEST NEIGHBOR DOMAIN SCORES
#
# This provides another semantic signal,
# but is weaker than aggregate neighbor voting.
# ==================================================================================================

def calculate_best_neighbor_scores(

    neighbors,

    trained_domains

):

    scores = {

        domain: 0.0

        for domain in trained_domains

    }

    if not neighbors:

        return scores

    best_similarity_by_domain = {

        domain: 0.0

        for domain in trained_domains

    }

    for neighbor in neighbors:

        domain = neighbor[
            "domain"
        ]

        similarity = max(

            neighbor[
                "similarity"
            ],

            0.0

        )

        if domain not in (

            best_similarity_by_domain

        ):

            continue

        best_similarity_by_domain[
            domain
        ] = max(

            best_similarity_by_domain[
                domain
            ],

            similarity

        )

    maximum = max(

        best_similarity_by_domain.values()

    )

    if maximum > 0:

        for domain, value in (

            best_similarity_by_domain.items()

        ):

            scores[domain] = (

                value / maximum

            )

    return scores


# ==================================================================================================
# DOMAIN CENTROID SIMILARITY
# ==================================================================================================

def calculate_centroid_scores(

    query_embedding,

    index,

    trained_domains

):

    scores = {}

    centroids = index[
        "domain_centroids"
    ]

    for domain in trained_domains:

        centroid = centroids.get(
            domain
        )

        if centroid is None:

            scores[domain] = 0.0

            continue

        centroid = np.asarray(
            centroid
        )

        similarity = float(

            np.dot(

                query_embedding,

                centroid

            )

        )

        scores[domain] = max(

            similarity,

            0.0

        )

    return normalize_scores(
        scores
    )


# ==================================================================================================
# DOMAIN NOISE PENALTY
#
# If neighbors strongly disagree with each other,
# semantic evidence should be treated cautiously.
# ==================================================================================================

def calculate_neighbor_agreement(

    neighbors

):

    if not neighbors:

        return 0.0

    weighted_domains = Counter()

    total_weight = 0.0

    for neighbor in neighbors:

        similarity = max(

            neighbor[
                "similarity"
            ],

            0.0

        )

        weight = similarity ** 2

        weighted_domains[

            neighbor["domain"]

        ] += weight

        total_weight += weight

    if total_weight <= 0:

        return 0.0

    best_weight = max(

        weighted_domains.values()

    )

    agreement = (

        best_weight /

        total_weight

    )

    return float(
        agreement
    )


# ==================================================================================================
# HYBRID DOMAIN PREDICTION
# ==================================================================================================

def predict_main_topic(

    text,

    semantic_model,

    domain_components,

    semantic_index

):

    trained_domains = (

        domain_components[
            "domains"
        ]

    )

    # ------------------------------------------------------------------
    # SUPERVISED MODEL
    # ------------------------------------------------------------------

    supervised_scores = (

        get_supervised_domain_scores(

            text,

            domain_components

        )

    )

    supervised_scores = normalize_scores(

        supervised_scores

    )

    # ------------------------------------------------------------------
    # SEMANTIC QUERY
    # ------------------------------------------------------------------

    query_embedding = encode_query(

        semantic_model,

        text

    )

    # ------------------------------------------------------------------
    # SEMANTIC NEIGHBORS
    # ------------------------------------------------------------------

    neighbors = semantic_retrieval(

        query_embedding,

        semantic_index,

        text

    )

    neighbor_scores = (

        calculate_neighbor_domain_scores(

            neighbors,

            trained_domains

        )

    )

    neighbor_scores = normalize_scores(

        neighbor_scores

    )

    best_neighbor_scores = (

        calculate_best_neighbor_scores(

            neighbors,

            trained_domains

        )

    )

    centroid_scores = (

        calculate_centroid_scores(

            query_embedding,

            semantic_index,

            trained_domains

        )

    )

    # ------------------------------------------------------------------
    # NEIGHBOR AGREEMENT
    # ------------------------------------------------------------------

    neighbor_agreement = (

        calculate_neighbor_agreement(

            neighbors

        )

    )

    # ------------------------------------------------------------------
    # HYBRID SCORE
    #
    # High supervised confidence alone cannot dominate.
    #
    # Semantic context has higher combined influence.
    # ------------------------------------------------------------------

    hybrid_scores = {}

    for domain in trained_domains:

        supervised = (

            supervised_scores.get(
                domain,
                0.0
            )

        )

        neighbor = (

            neighbor_scores.get(
                domain,
                0.0
            )

        )

        centroid = (

            centroid_scores.get(
                domain,
                0.0
            )

        )

        best_neighbor = (

            best_neighbor_scores.get(
                domain,
                0.0
            )

        )

        hybrid_score = (

            WEIGHT_SUPERVISED
            *
            supervised

            +

            WEIGHT_NEIGHBOR_VOTE
            *
            neighbor

            +

            WEIGHT_CENTROID
            *
            centroid

            +

            WEIGHT_BEST_NEIGHBOR
            *
            best_neighbor

        )

        hybrid_scores[
            domain
        ] = float(
            hybrid_score
        )

    ranked_domains = sorted(

        hybrid_scores.items(),

        key=lambda item:

            item[1],

        reverse=True

    )

    predicted_domain = (

        ranked_domains[0][0]

    )

    best_score = (

        ranked_domains[0][1]

    )

    second_score = (

        ranked_domains[1][1]

        if len(ranked_domains) > 1

        else 0.0

    )

    domain_margin = (

        best_score -

        second_score

    )

    # ------------------------------------------------------------------
    # EVIDENCE SCORE
    #
    # This does NOT determine whether prediction exists.
    #
    # The system always predicts.
    #
    # It only indicates reliability.
    # ------------------------------------------------------------------

    best_neighbor_similarity = 0.0

    mean_neighbor_similarity = 0.0

    if neighbors:

        best_neighbor_similarity = max(

            neighbor[
                "similarity"
            ]

            for neighbor in neighbors

        )

        mean_neighbor_similarity = float(

            np.mean(

                [

                    neighbor[
                        "similarity"
                    ]

                    for neighbor in neighbors

                ]

            )

        )

    evidence_score = (

        0.30
        *
        max(
            best_neighbor_similarity,
            0.0
        )

        +

        0.25
        *
        max(
            mean_neighbor_similarity,
            0.0
        )

        +

        0.25
        *
        max(
            neighbor_agreement,
            0.0
        )

        +

        0.20
        *
        max(
            domain_margin,
            0.0
        )

    )

    if evidence_score >= 0.55:

        evidence_level = "HIGH"

    elif evidence_score >= 0.30:

        evidence_level = "MEDIUM"

    else:

        evidence_level = "LOW"

    return {

        "predicted_domain":
            predicted_domain,

        "hybrid_score":
            best_score,

        "domain_margin":
            domain_margin,

        "evidence_score":
            evidence_score,

        "evidence_level":
            evidence_level,

        "supervised_scores":
            supervised_scores,

        "neighbor_scores":
            neighbor_scores,

        "best_neighbor_scores":
            best_neighbor_scores,

        "centroid_scores":
            centroid_scores,

        "hybrid_scores":
            hybrid_scores,

        "ranked_domains":
            ranked_domains,

        "neighbors":
            neighbors,

        "neighbor_agreement":
            neighbor_agreement,

        "best_neighbor_similarity":
            best_neighbor_similarity,

        "mean_neighbor_similarity":
            mean_neighbor_similarity,

        "query_embedding":
            query_embedding

    }


# ==================================================================================================
# PREDICT SUBTOPICS FROM SEMANTIC NEIGHBORS
#
# IMPORTANT:
#
# We use neighbors belonging to the predicted main topic.
#
# This prevents subtopics from unrelated domains.
#
# Multiple neighbors vote for subtopics.
# ==================================================================================================

def predict_semantic_subtopics(

    predicted_domain,

    neighbors,

    semantic_index

):

    topic_scores = Counter()

    topic_support = defaultdict(
        list
    )

    for neighbor in neighbors:

        if (

            neighbor["domain"]

            !=

            predicted_domain

        ):

            continue

        topic = clean_text(

            neighbor["topic"]

        )

        if not topic:

            continue

        similarity = max(

            neighbor[
                "similarity"
            ],

            0.0

        )

        # Strong semantic matches receive more influence

        weight = similarity ** 3

        topic_scores[
            topic
        ] += weight

        topic_support[
            topic
        ].append(
            neighbor
        )

    # ------------------------------------------------------------------
    # If top-K neighbors contain too few examples
    # from the predicted domain, retrieve more examples
    # directly from that domain.
    # ------------------------------------------------------------------

    if not topic_scores:

        embeddings = semantic_index[
            "embeddings"
        ]

        domain_indices = (

            semantic_index[
                "domain_indices"
            ].get(

                predicted_domain,

                np.asarray(
                    [],
                    dtype=np.int32
                )

            )

        )

        if len(domain_indices) > 0:

            # Query embedding will be passed through
            # another function when needed.

            pass

    if not topic_scores:

        return []

    maximum_score = max(

        topic_scores.values()

    )

    ranked = []

    for topic, score in (

        topic_scores.items()

    ):

        normalized_score = (

            score /

            maximum_score

        )

        if (

            normalized_score

            >=

            MIN_SUBTOPIC_SCORE

        ):

            ranked.append(

                {

                    "topic":
                        topic,

                    "score":
                        float(
                            normalized_score
                        ),

                    "support_count":
                        len(

                            topic_support[
                                topic
                            ]

                        ),

                    "evidence":
                        "semantic neighbors"

                }

            )

    ranked = sorted(

        ranked,

        key=lambda item:

            item["score"],

        reverse=True

    )

    return ranked[
        :MAX_SUBTOPICS
    ]


# ==================================================================================================
# DOMAIN-RESTRICTED SEMANTIC SUBTOPIC RETRIEVAL
#
# This is used when the general top-K neighbor list does not
# contain enough documents from the predicted domain.
# ==================================================================================================

def retrieve_domain_neighbors(

    query_embedding,

    predicted_domain,

    semantic_index,

    original_text

):

    domain_indices = (

        semantic_index[
            "domain_indices"
        ].get(

            predicted_domain,

            np.asarray(
                [],
                dtype=np.int32
            )

        )

    )

    if len(domain_indices) == 0:

        return []

    embeddings = (

        semantic_index[
            "embeddings"
        ][domain_indices]

    )

    similarities = np.dot(

        embeddings,

        query_embedding

    )

    similarities = np.asarray(

        similarities,

        dtype=np.float32

    )

    normalized_text = clean_text(
        original_text
    )

    same_indices = set(

        semantic_index[
            "text_to_indices"
        ].get(

            normalized_text,

            []

        )

    )

    for local_index, global_index in enumerate(

        domain_indices

    ):

        if int(global_index) in same_indices:

            similarities[
                local_index
            ] = -1.0

    top_k = min(

        TOP_K_SUBTOPIC_NEIGHBORS,

        len(similarities)

    )

    if top_k <= 0:

        return []

    local_indices = np.argpartition(

        similarities,

        -top_k

    )[-top_k:]

    local_indices = (

        local_indices[

            np.argsort(

                similarities[
                    local_indices
                ]

            )[::-1]

        ]

    )

    neighbors = []

    for local_index in local_indices:

        similarity = float(

            similarities[
                local_index
            ]

        )

        if similarity < 0:

            continue

        global_index = int(

            domain_indices[
                local_index
            ]

        )

        neighbors.append(

            {

                "index":
                    global_index,

                "similarity":
                    similarity,

                "text":
                    semantic_index[
                        "texts"
                    ][global_index],

                "domain":
                    semantic_index[
                        "domains"
                    ][global_index],

                "topic":
                    semantic_index[
                        "topics"
                    ][global_index]

            }

        )

    return neighbors


# ==================================================================================================
# DOMAIN-AWARE SUBTOPIC PREDICTION
#
# Strategy:
#
# 1. Retrieve semantic neighbors specifically inside predicted domain
#
# 2. Aggregate subtopic evidence
#
# 3. Use stored subtopic frequency only as a weak fallback
#
# ==================================================================================================

def predict_subtopics(

    predicted_domain,

    query_embedding,

    original_text,

    semantic_index

):

    domain_neighbors = (

        retrieve_domain_neighbors(

            query_embedding,

            predicted_domain,

            semantic_index,

            original_text

        )

    )

    topic_scores = Counter()

    topic_support = defaultdict(
        list
    )

    for neighbor in domain_neighbors:

        topic = clean_text(

            neighbor["topic"]

        )

        if not topic:

            continue

        similarity = max(

            neighbor[
                "similarity"
            ],

            0.0

        )

        # ----------------------------------------------------------------
        # Cubic weighting:
        #
        # very similar semantic examples become more important.
        # weak examples contribute less.
        # ----------------------------------------------------------------

        weight = similarity ** 3

        topic_scores[
            topic
        ] += weight

        topic_support[
            topic
        ].append(
            neighbor
        )

    # ------------------------------------------------------------------
    # NORMAL SEMANTIC PREDICTION
    # ------------------------------------------------------------------

    if topic_scores:

        maximum_score = max(

            topic_scores.values()

        )

        predictions = []

        for topic, score in (

            topic_scores.items()

        ):

            normalized_score = (

                score /

                maximum_score

            )

            if (

                normalized_score

                >=

                MIN_SUBTOPIC_SCORE

            ):

                predictions.append(

                    {

                        "topic":
                            topic,

                        "score":
                            float(
                                normalized_score
                            ),

                        "support_count":
                            len(

                                topic_support[
                                    topic
                                ]

                            ),

                        "evidence":
                            "domain-restricted semantic neighbors"

                    }

                )

        predictions = sorted(

            predictions,

            key=lambda item:

                item["score"],

            reverse=True

        )

        if predictions:

            return predictions[
                :MAX_SUBTOPICS
            ]

    # ------------------------------------------------------------------
    # FALLBACK
    #
    # The system still guarantees subtopic output.
    #
    # This is used only if no semantic subtopic evidence exists.
    # ------------------------------------------------------------------

    domain_topic_counts = (

        semantic_index[
            "domain_topic_counts"
        ].get(

            predicted_domain,

            Counter()

        )

    )

    if domain_topic_counts:

        most_common = (

            domain_topic_counts.most_common(

                MAX_SUBTOPICS

            )

        )

        maximum_count = (

            most_common[0][1]

        )

        fallback_predictions = []

        for topic, count in (

            most_common

        ):

            fallback_predictions.append(

                {

                    "topic":
                        topic,

                    "score":
                        float(

                            count /

                            maximum_count

                        ),

                    "support_count":
                        int(count),

                    "evidence":
                        "domain subtopic fallback"

                }

            )

        return fallback_predictions

    return [

        {

            "topic":
                predicted_domain,

            "score":
                0.0,

            "support_count":
                0,

            "evidence":
                "main topic fallback"

        }

    ]


# ==================================================================================================
# COMPLETE PREDICTION
# ==================================================================================================

def predict_hierarchical(

    text,

    semantic_model,

    domain_components,

    semantic_index

):

    text = clean_text(
        text
    )

    if not text:

        raise ValueError(
            "Input text is empty."
        )

    main_result = (

        predict_main_topic(

            text,

            semantic_model,

            domain_components,

            semantic_index

        )

    )

    predicted_domain = (

        main_result[
            "predicted_domain"
        ]

    )

    subtopics = (

        predict_subtopics(

            predicted_domain,

            main_result[
                "query_embedding"
            ],

            text,

            semantic_index

        )

    )

    main_result[
        "subtopics"
    ] = subtopics

    main_result[
        "input_text"
    ] = text

    return main_result


# ==================================================================================================
# PRINT PREDICTION
# ==================================================================================================

def print_prediction(

    result

):

    section(
        "SEMANTIC TRANSFORMER HIERARCHICAL PREDICTION"
    )

    print("INPUT TEXT:")

    print(
        result[
            "input_text"
        ]
    )

    subsection(
        "PREDICTED MAIN TOPIC"
    )

    print()

    print(
        result[
            "predicted_domain"
        ]
    )

    print()

    print(
        "Hybrid score:"
    )

    print(

        f"{result['hybrid_score']:.4f}"

    )

    print()

    print(
        "Evidence level:"
    )

    print(
        result[
            "evidence_level"
        ]
    )

    print()

    print(
        "Evidence score:"
    )

    print(

        f"{result['evidence_score']:.4f}"

    )

    print()

    print(
        "Domain margin:"
    )

    print(

        f"{result['domain_margin']:.4f}"

    )

    print()

    print(
        "Neighbor agreement:"
    )

    print(

        f"{result['neighbor_agreement']:.4f}"

    )

    print()

    print(
        "Best semantic neighbor similarity:"
    )

    print(

        f"{result['best_neighbor_similarity']:.4f}"

    )

    print()

    print(
        "Mean semantic neighbor similarity:"
    )

    print(

        f"{result['mean_neighbor_similarity']:.4f}"

    )


    # ----------------------------------------------------------------------------------------------
    # TOP DOMAIN CANDIDATES
    # ----------------------------------------------------------------------------------------------

    subsection(
        "TOP MAIN TOPIC CANDIDATES"
    )

    for position, (

        domain,

        score

    ) in enumerate(

        result[
            "ranked_domains"
        ][:5],

        start=1

    ):

        print(

            f"{position}. "

            f"{domain} "

            f"(hybrid score: "

            f"{score:.4f})"

        )


    # ----------------------------------------------------------------------------------------------
    # SUBTOPICS
    # ----------------------------------------------------------------------------------------------

    subsection(
        "PREDICTED SUBTOPIC(S)"
    )

    subtopics = result[
        "subtopics"
    ]

    if not subtopics:

        print()

        print(
            "No subtopic evidence found."
        )

    else:

        for position, subtopic in enumerate(

            subtopics,

            start=1

        ):

            print()

            print(

                f"{position}. "

                f"{subtopic['topic']}"

            )

            print(

                f"   score: "

                f"{subtopic['score']:.4f}"

            )

            print(

                f"   support documents: "

                f"{subtopic['support_count']}"

            )

            print(

                f"   evidence: "

                f"{subtopic['evidence']}"

            )


    # ----------------------------------------------------------------------------------------------
    # SEMANTIC CONTEXT EVIDENCE
    # ----------------------------------------------------------------------------------------------

    subsection(
        "TOP SEMANTIC CONTEXT EVIDENCE"
    )

    neighbors = result[
        "neighbors"
    ]

    if not neighbors:

        print(
            "No semantic neighbors found."
        )

    else:

        for position, neighbor in enumerate(

            neighbors[
                :TOP_K_DISPLAY
            ],

            start=1

        ):

            print()

            print(

                f"{position}. "

                f"Domain: "

                f"{neighbor['domain']}"

            )

            print(

                f"   Topic: "

                f"{neighbor['topic']}"

            )

            print(

                f"   Similarity: "

                f"{neighbor['similarity']:.4f}"

            )

            print(

                f"   Text: "

                f"{neighbor['text']}"

            )

    line("=")


# ==================================================================================================
# PRINT SYSTEM INFORMATION
# ==================================================================================================

def print_system_info(

    domain_components,

    semantic_index

):

    section(
        "SYSTEM INFORMATION"
    )

    print("Project root:")

    print(
        PROJECT_ROOT
    )

    print()

    print("Semantic model:")

    print(
        SEMANTIC_MODEL_NAME
    )

    print()

    print("Number of domains:")

    print(

        len(

            domain_components[
                "domains"
            ]

        )

    )

    print()

    print(
        "Number of reference documents:"
    )

    print(

        len(

            semantic_index[
                "texts"
            ]

        )

    )

    print()

    print(
        "Embedding shape:"
    )

    print(

        semantic_index[
            "embeddings"
        ].shape

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
        "Hybrid weights:"
    )

    print()

    print(

        f"Supervised: "

        f"{WEIGHT_SUPERVISED}"

    )

    print(

        f"Neighbor voting: "

        f"{WEIGHT_NEIGHBOR_VOTE}"

    )

    print(

        f"Centroid: "

        f"{WEIGHT_CENTROID}"

    )

    print(

        f"Best neighbor: "

        f"{WEIGHT_BEST_NEIGHBOR}"

    )

    print()

    print(
        "Output:"
    )

    print()

    print(
        "MAIN TOPIC + SUBTOPIC(S)"
    )

    print()

    print(
        "UNKNOWN is not used as a final prediction."
    )


# ==================================================================================================
# INTERACTIVE INFERENCE
# ==================================================================================================

def interactive_inference(

    semantic_model,

    domain_components,

    semantic_index

):

    section(
        "AMHARIC FAST SEMANTIC TRANSFORMER HIERARCHICAL NLP SYSTEM"
    )

    print(
        "PIPELINE:"
    )

    print()

    print(
        "TEXT"
    )

    print(
        "  |"
    )

    print(
        "  +--> SUPERVISED DOMAIN MODEL"
    )

    print(
        "  |"
    )

    print(
        "  +--> PRETRAINED MULTILINGUAL SEMANTIC TRANSFORMER"
    )

    print(
        "  |"
    )

    print(
        "  +--> SEMANTIC NEAREST DOCUMENTS"
    )

    print(
        "  |"
    )

    print(
        "  +--> DOMAIN CENTROIDS"
    )

    print(
        "  |"
    )

    print(
        "  +--> LOCAL NEIGHBOR AGREEMENT"
    )

    print(
        "  |"
    )

    print(
        "  +--> NOISE-AWARE HYBRID MAIN TOPIC"
    )

    print(
        "          |"
    )

    print(
        "          +--> DOMAIN-RESTRICTED SEMANTIC SUBTOPIC(S)"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print()

    print(
        "A high supervised classifier score alone"
    )

    print(
        "does NOT determine the final answer."
    )

    print()

    print(
        "The final decision combines:"
    )

    print()

    print(
        "1. Supervised domain evidence"
    )

    print(
        "2. Contextual semantic embeddings"
    )

    print(
        "3. Multiple semantic neighbors"
    )

    print(
        "4. Neighbor agreement"
    )

    print(
        "5. Domain centroid similarity"
    )

    print(
        "6. Noise-aware hybrid ranking"
    )

    print(
        "7. Domain-specific semantic subtopic retrieval"
    )

    print()

    print(
        "OUTPUT GUARANTEE:"
    )

    print()

    print(
        "The system always returns:"
    )

    print()

    print(
        "    MAIN TOPIC"
    )

    print()

    print(
        "and"
    )

    print()

    print(
        "    SUBTOPIC(S)"
    )

    print()

    print(
        "If evidence is weak,"
    )

    print(
        "the system returns LOW evidence,"
    )

    print(
        "but still returns the closest main topic"
    )

    print(
        "and subtopic(s)."
    )

    print()

    print(
        "Commands:"
    )

    print()

    print(
        "    exit"
    )

    print(
        "    quit"
    )

    print(
        "    info"
    )

    print()

    while True:

        line("-")

        try:

            text = input(
                "\nEnter text: "
            )

        except KeyboardInterrupt:

            print()

            print(
                "Exiting..."
            )

            break

        except EOFError:

            print()

            print(
                "Exiting..."
            )

            break

        text = clean_text(
            text
        )

        if not text:

            continue

        command = text.lower()

        if command in [

            "exit",

            "quit"

        ]:

            print()

            print(
                "Exiting system..."
            )

            break

        if command == "info":

            print_system_info(

                domain_components,

                semantic_index

            )

            continue

        try:

            result = predict_hierarchical(

                text,

                semantic_model,

                domain_components,

                semantic_index

            )

            print_prediction(
                result
            )

        except Exception as error:

            section(
                "PREDICTION ERROR"
            )

            print(
                type(error).__name__
            )

            print()

            print(error)

            print()

            traceback.print_exc()


# ==================================================================================================
# VALIDATE MODEL PATHS
# ==================================================================================================

def validate_paths():

    section(
        "VALIDATING MODEL PATHS"
    )

    print("Project root:")

    print(
        PROJECT_ROOT
    )

    print()

    print("Domain model:")

    print(
        DOMAIN_MODEL_PATH
    )

    print()

    print("Exists:")

    print(
        DOMAIN_MODEL_PATH.exists()
    )

    print()

    print("Subtopic model:")

    print(
        SUBTOPIC_MODEL_PATH
    )

    print()

    print("Exists:")

    print(
        SUBTOPIC_MODEL_PATH.exists()
    )

    print()

    if not DOMAIN_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Domain model not found:\n"

            f"{DOMAIN_MODEL_PATH}"

        )

    if not SUBTOPIC_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Subtopic model not found:\n"

            f"{SUBTOPIC_MODEL_PATH}"

        )

    print(
        "PASS: All required model files exist."
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    section(
        "AMHARIC NLP - FAST SEMANTIC TRANSFORMER HYBRID HIERARCHICAL INFERENCE"
    )

    print("PROJECT ROOT:")

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "STARTING SYSTEM..."
    )

    validate_paths()


    # ----------------------------------------------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------------------------------------------

    domain_model_data = (

        load_domain_model()

    )

    subtopic_model_data = (

        load_subtopic_model()

    )

    # The existing subtopic model is loaded for compatibility.
    # The semantic hierarchical inference uses domain-restricted
    # semantic retrieval for the final subtopic decision.

    _ = subtopic_model_data

    domain_components = (

        extract_domain_components(

            domain_model_data

        )

    )

    trained_domains = (

        domain_components[
            "domains"
        ]

    )


    # ----------------------------------------------------------------------------------------------
    # LOAD PRETRAINED SEMANTIC MODEL
    # ----------------------------------------------------------------------------------------------

    semantic_model = (

        load_semantic_model()

    )


    # ----------------------------------------------------------------------------------------------
    # LOAD OR BUILD INDEX
    # ----------------------------------------------------------------------------------------------

    semantic_index = (

        load_or_build_semantic_index(

            semantic_model,

            trained_domains

        )

    )


    # ----------------------------------------------------------------------------------------------
    # START INTERACTIVE SYSTEM
    # ----------------------------------------------------------------------------------------------

    interactive_inference(

        semantic_model,

        domain_components,

        semantic_index

    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "Program interrupted."
        )

    except Exception as error:

        section(
            "FATAL ERROR"
        )

        print(
            type(error).__name__
        )

        print()

        print(error)

        print()

        traceback.print_exc()

        sys.exit(1)

    finally:

        gc.collect()