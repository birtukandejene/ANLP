# ============================================================
#
# 27_amharic_semantic_hybrid_inference.py
#
# OPEN-TOPIC LOCAL AMHARIC SEMANTIC CLASSIFIER
#
# IMPORTANT:
#   - NO hard-coded 19-domain restriction for MAIN TOPIC.
#   - Main topic is selected from ALL topics found in the
#     semantic index.
#   - Every available topic is eligible.
#   - The old supervised domain classifier is NOT used to
#     force the main topic into the old 19-domain list.
#
# Uses:
#   1. rasyosef/embedding-amharic-base
#   2. Semantic nearest-neighbor evidence
#   3. Global topic semantic centroids
#   4. All indexed topics
#   5. Existing domain-aware supervised subtopic models
#   6. Topic co-occurrence for subtopic selection
#
# INPUT:
#   One Amharic sentence at a time
#
# REQUIRED:
#   models/amharic_semantic_index.joblib
#   models/final_domain_aware_subtopic_model.joblib
#
# OPTIONAL:
#   models/final_professional_domain_classification_model.joblib
#
# RUN:
#   python notebooks\27_amharic_semantic_hybrid_inference.py
#
# ============================================================

import os
import sys
import re
import subprocess
import importlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ============================================================
# INSTALL REQUIRED PACKAGES
# ============================================================

def install_if_missing(package_name, import_name=None):
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing missing package: {package_name}")
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
    "sentence_transformers",
)

install_if_missing("joblib")
install_if_missing("numpy")
install_if_missing("scipy")
install_if_missing("scikit-learn", "sklearn")
install_if_missing("pandas")


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import joblib
import torch

from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SEMANTIC_INDEX_PATH = (
    MODELS_DIR / "amharic_semantic_index.joblib"
)

DOMAIN_MODEL_PATH = (
    MODELS_DIR
    / "final_professional_domain_classification_model.joblib"
)

SUBTOPIC_MODEL_PATH = (
    MODELS_DIR
    / "final_domain_aware_subtopic_model.joblib"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

SEMANTIC_MODEL_NAME = (
    "rasyosef/embedding-amharic-base"
)

SEMANTIC_NEIGHBORS = 30

# Main topic fusion.
#
# Semantic nearest neighbors are strongest because they directly
# compare the input against real indexed Amharic examples.
#
# Topic centroid evidence evaluates ALL available topic centroids.
#
# Supervised evidence is weaker because the existing supervised
# models were originally domain-aware.
MAIN_TOPIC_NEIGHBOR_WEIGHT = 0.50
MAIN_TOPIC_CENTROID_WEIGHT = 0.30
MAIN_TOPIC_SUPERVISED_WEIGHT = 0.20
MAIN_TOPIC_FEEDBACK_WEIGHT = 0.15

# Subtopic fusion.
SUBTOPIC_NEIGHBOR_WEIGHT = 0.45
SUBTOPIC_CENTROID_WEIGHT = 0.25
SUBTOPIC_SUPERVISED_WEIGHT = 0.20
SUBTOPIC_COOCCURRENCE_WEIGHT = 0.10
SUBTOPIC_FEEDBACK_WEIGHT = 0.15

MAX_SUBTOPICS = 3

SHOW_NEIGHBORS = 5
SHOW_MAIN_TOPIC_CANDIDATES = 15
SHOW_SUBTOPIC_CANDIDATES = 10

MIN_SUBTOPIC_RELATIVE_SCORE = 0.45


# ============================================================
# LOAD SEMANTIC INDEX
# ============================================================

print()
print("=" * 72)
print("OPEN-TOPIC AMHARIC SEMANTIC HYBRID INFERENCE")
print("=" * 72)
print()

print("Project:")
print(PROJECT_ROOT)


if not SEMANTIC_INDEX_PATH.exists():
    print()
    print("ERROR:")
    print("Semantic index was not found.")
    print()
    print("Expected:")
    print(SEMANTIC_INDEX_PATH)
    print()
    print(
        "Run the semantic index builder first."
    )
    sys.exit(1)


if not SUBTOPIC_MODEL_PATH.exists():
    print()
    print("ERROR:")
    print("Subtopic model was not found:")
    print(SUBTOPIC_MODEL_PATH)
    sys.exit(1)


print()
print("Loading semantic index...")

semantic_index = joblib.load(
    SEMANTIC_INDEX_PATH
)


print()
print("Loading supervised subtopic model...")

subtopic_bundle = joblib.load(
    SUBTOPIC_MODEL_PATH
)


# ============================================================
# OPTIONAL SUPERVISED DOMAIN MODEL
#
# It is loaded only if available.
#
# IMPORTANT:
# It is NEVER allowed to restrict the main topic to its
# original 19 domain labels.
# ============================================================

domain_bundle = None

if DOMAIN_MODEL_PATH.exists():
    try:
        print()
        print("Loading optional supervised domain model...")
        domain_bundle = joblib.load(
            DOMAIN_MODEL_PATH
        )
    except Exception as error:
        print()
        print(
            "WARNING: Could not load optional domain model."
        )
        print(error)
        domain_bundle = None
else:
    print()
    print(
        "Optional supervised domain model not found."
    )
    print(
        "Continuing with open-topic semantic classification."
    )


# ============================================================
# CHECK INDEX
# ============================================================

required_index_keys = [
    "semantic_model_name",
    "embeddings",
    "texts",
    "domains",
    "topics",
    "domain_centroids",
    "topic_centroids",
]

missing_keys = [
    key
    for key in required_index_keys
    if key not in semantic_index
]

if missing_keys:
    print()
    print("ERROR:")
    print("Semantic index is missing keys:")
    print(missing_keys)
    sys.exit(1)


# ============================================================
# EXTRACT INDEX DATA
# ============================================================

TRAIN_EMBEDDINGS = np.asarray(
    semantic_index["embeddings"],
    dtype=np.float32,
)

TRAIN_TEXTS = list(
    semantic_index["texts"]
)

TRAIN_DOMAINS = list(
    semantic_index["domains"]
)

TRAIN_TOPICS = list(
    semantic_index["topics"]
)

DOMAIN_CENTROIDS = (
    semantic_index["domain_centroids"]
)

TOPIC_CENTROIDS = (
    semantic_index["topic_centroids"]
)


# ============================================================
# DYNAMIC LABEL DISCOVERY
#
# NO HARDCODED MAIN-TOPIC LIST.
#
# Every topic occurring in the semantic index becomes eligible
# for main-topic prediction.
# ============================================================

def extract_topic_list(value):
    """
    Convert the topic field of one indexed document into a
    clean list of topic strings.
    """

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    if isinstance(value, (list, tuple, set)):
        result = []

        for item in value:
            if item is None:
                continue

            item = str(item).strip()

            if item:
                result.append(item)

        return result

    return [str(value).strip()]


ALL_TOPICS = set()

for topic_value in TRAIN_TOPICS:
    for topic in extract_topic_list(topic_value):
        if topic:
            ALL_TOPICS.add(topic)

ALL_TOPICS = sorted(
    ALL_TOPICS
)

INDEX_DOMAINS = sorted(
    {
        str(domain).strip()
        for domain in TRAIN_DOMAINS
        if str(domain).strip()
    }
)


print()
print("-" * 72)
print("INDEX INFORMATION")
print("-" * 72)
print()

print(
    "Indexed documents:",
    f"{len(TRAIN_TEXTS):,}",
)

print(
    "Embedding dimension:",
    TRAIN_EMBEDDINGS.shape[1],
)

print(
    "Discovered domains:",
    len(INDEX_DOMAINS),
)

print(
    "ALL AVAILABLE TOPICS:",
    len(ALL_TOPICS),
)

print()
print(
    "Main-topic restriction:"
)
print(
    "NONE - ALL INDEXED TOPICS ARE ELIGIBLE"
)


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

model_name_from_index = semantic_index.get(
    "semantic_model_name",
    SEMANTIC_MODEL_NAME,
)
local_semantic_model_path = MODELS_DIR / "embedding-amharic-base"
if local_semantic_model_path.is_dir():
    model_name_from_index = str(local_semantic_model_path)

print()
print("Semantic model:")
print(model_name_from_index)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:")
print(device)


if device == "cuda":
    try:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    except Exception:
        pass


print()
print("Loading semantic encoder...")

semantic_model = SentenceTransformer(
    model_name_from_index,
    device=device,
)


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def normalize_vector(vector):
    """
    Safely normalize a numeric embedding vector.

    Also handles centroid dictionaries such as:

        {
            "embedding": [...],
            "count": 100
        }

    This prevents:
        float() argument must be a string or a real number,
        not 'dict'
    """

    if isinstance(vector, dict):

        for key in [
            "embedding",
            "centroid",
            "vector",
            "mean",
        ]:
            if key in vector:
                vector = vector[key]
                break

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm <= 0:
        return vector

    return vector / norm


def normalize_scores(score_dict):
    """
    Convert non-negative scores to values summing to 1.
    """

    if not score_dict:
        return {}

    values = np.array(
        [
            max(float(value), 0.0)
            for value in score_dict.values()
        ],
        dtype=np.float64,
    )

    total = values.sum()

    if total <= 0:
        uniform = 1.0 / len(score_dict)

        return {
            key: uniform
            for key in score_dict
        }

    return {
        key: float(value / total)
        for key, value in zip(
            score_dict.keys(),
            values,
        )
    }


def softmax(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return values

    values = values - np.max(values)

    exp_values = np.exp(values)

    total = exp_values.sum()

    if total <= 0:
        return (
            np.ones_like(values)
            / len(values)
        )

    return exp_values / total


def sigmoid(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.clip(
        values,
        -50,
        50,
    )

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def clean_text(text):
    text = str(text)

    def replace_punctuation(match):
        start, end = match.span()
        if start > 0 and end < len(text) and text[start - 1].isalpha() and text[end].isalpha():
            return match.group()
        return " "

    text = re.sub(
        r"[፡።፣፤፥፦፧፨\.,!?\(\)\[\]\{\}\"'“”:/\\=+%#@*&^~`<>|•]+",
        replace_punctuation,
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# CENTROID EXTRACTION
# ============================================================

def extract_centroid_vector(value):
    """
    Extract an actual numeric vector from several possible
    centroid formats.
    """

    if isinstance(value, dict):

        for key in [
            "embedding",
            "centroid",
            "vector",
            "mean",
        ]:
            if key in value:
                return normalize_vector(
                    value[key]
                )

        return None

    try:
        vector = np.asarray(
            value,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            return None

        return normalize_vector(
            vector
        )

    except Exception:
        return None


# ============================================================
# FLATTEN ALL TOPIC CENTROIDS
#
# Supports either:
#
#   {
#       topic: vector
#   }
#
# or:
#
#   {
#       domain: {
#           topic: vector
#       }
#   }
#
# or centroid dictionaries.
# ============================================================

def flatten_topic_centroids(raw_centroids):
    flattened = {}

    if not isinstance(
        raw_centroids,
        dict,
    ):
        return flattened

    for first_key, first_value in raw_centroids.items():

        # ----------------------------------------------------
        # Case 1:
        # topic -> vector
        # ----------------------------------------------------

        direct_vector = extract_centroid_vector(
            first_value
        )

        if direct_vector is not None:

            topic = str(
                first_key
            ).strip()

            if topic:
                flattened[topic] = direct_vector

            continue

        # ----------------------------------------------------
        # Case 2:
        # domain -> {topic -> vector}
        # ----------------------------------------------------

        if isinstance(
            first_value,
            dict,
        ):

            for topic, centroid in (
                first_value.items()
            ):

                vector = extract_centroid_vector(
                    centroid
                )

                if vector is None:
                    continue

                topic = str(
                    topic
                ).strip()

                if not topic:
                    continue

                flattened[topic] = vector

    return flattened


GLOBAL_TOPIC_CENTROIDS = (
    flatten_topic_centroids(
        TOPIC_CENTROIDS
    )
)


# ============================================================
# ENCODE QUERY
# ============================================================

def encode_query(text):

    if hasattr(
        semantic_model,
        "encode_query",
    ):

        embedding = (
            semantic_model.encode_query(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    else:

        embedding = (
            semantic_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )[0]

    return normalize_vector(
        embedding
    )


# ============================================================
# SEMANTIC NEAREST NEIGHBORS
# ============================================================

def retrieve_neighbors(
    query_embedding,
    k=SEMANTIC_NEIGHBORS,
):

    similarities = (
        TRAIN_EMBEDDINGS
        @ query_embedding
    )

    k = min(
        int(k),
        len(similarities),
    )

    if k <= 0:
        return []

    candidate_indices = np.argpartition(
        -similarities,
        k - 1,
    )[:k]

    candidate_indices = (
        candidate_indices[
            np.argsort(
                -similarities[
                    candidate_indices
                ]
            )
        ]
    )

    results = []

    for index in candidate_indices:

        index = int(index)

        results.append(
            {
                "index": index,
                "similarity": float(
                    similarities[index]
                ),
                "text": TRAIN_TEXTS[index],
                "domain": TRAIN_DOMAINS[index],
                "topics": extract_topic_list(
                    TRAIN_TOPICS[index]
                ),
            }
        )

    return results


# ============================================================
# MAIN TOPIC - NEAREST-NEIGHBOR EVIDENCE
#
# IMPORTANT:
#   We DO NOT filter by domain.
#
# Every topic from every retrieved document can vote.
# ============================================================

def get_main_topic_neighbor_scores(
    neighbors,
):

    scores = {}
    best_similarity = {}
    topic_examples = {}

    for item in neighbors:

        similarity = max(
            float(item["similarity"]),
            0.0,
        )

        weight = similarity ** 3

        for topic in item["topics"]:

            topic = str(
                topic
            ).strip()

            if not topic:
                continue

            scores[topic] = (
                scores.get(
                    topic,
                    0.0,
                )
                + weight
            )

            old_similarity = (
                best_similarity.get(
                    topic,
                    -1.0,
                )
            )

            if similarity > old_similarity:

                best_similarity[topic] = (
                    similarity
                )

                topic_examples[topic] = (
                    item
                )

    return (
        normalize_scores(scores),
        best_similarity,
        topic_examples,
    )


# ============================================================
# MAIN TOPIC - GLOBAL CENTROID EVIDENCE
#
# IMPORTANT:
#   Every topic centroid is evaluated.
#   No domain restriction.
# ============================================================

def get_global_topic_centroid_scores(
    query_embedding,
):

    scores = {}

    for topic, raw_centroid in (
        GLOBAL_TOPIC_CENTROIDS.items()
    ):

        centroid = extract_centroid_vector(
            raw_centroid
        )

        if centroid is None:
            continue

        if len(centroid) != len(
            query_embedding
        ):
            continue

        similarity = float(
            np.dot(
                query_embedding,
                centroid,
            )
        )

        scores[topic] = max(
            similarity,
            0.0,
        )

    return normalize_scores(
        scores
    )


# ============================================================
# SUPERVISED SUBTOPIC MODEL HELPERS
# ============================================================

def get_classifier_from_subtopic_model(
    domain_model,
):

    if domain_model is None:
        return None

    if (
        hasattr(
            domain_model,
            "predict_proba",
        )
        or hasattr(
            domain_model,
            "decision_function",
        )
    ):
        return domain_model

    if isinstance(
        domain_model,
        dict,
    ):

        for key in [
            "model",
            "classifier",
            "estimator",
        ]:

            candidate = (
                domain_model.get(
                    key
                )
            )

            if candidate is None:
                continue

            if (
                hasattr(
                    candidate,
                    "predict_proba",
                )
                or hasattr(
                    candidate,
                    "decision_function",
                )
            ):
                return candidate

    return None


# ============================================================
# GLOBAL SUPERVISED TOPIC EVIDENCE
#
# The existing subtopic system is domain-aware.
#
# Instead of selecting ONE domain first, we evaluate the
# supervised models available for ALL discovered domains.
#
# Their topic outputs are merged into one global topic score.
#
# This is secondary evidence only.
# ============================================================

def get_global_supervised_topic_scores(
    text,
):

    if not isinstance(
        subtopic_bundle,
        dict,
    ):
        return {}

    domain_models = (
        subtopic_bundle.get(
            "domain_models",
            {},
        )
    )

    if not isinstance(
        domain_models,
        dict,
    ):
        return {}

    word_vectorizer = (
        subtopic_bundle.get(
            "word_vectorizer"
        )
    )

    character_vectorizer = (
        subtopic_bundle.get(
            "character_vectorizer"
        )
    )

    if (
        word_vectorizer is None
        or character_vectorizer is None
    ):
        return {}

    try:

        word_features = (
            word_vectorizer.transform(
                [text]
            )
        )

        char_features = (
            character_vectorizer.transform(
                [text]
            )
        )

        features = hstack(
            [
                word_features,
                char_features,
            ],
            format="csr",
        )

    except Exception:
        return {}

    model_topic_scores = []

    for domain, domain_model in (
        domain_models.items()
    ):

        classifier = (
            get_classifier_from_subtopic_model(
                domain_model
            )
        )

        if classifier is None:
            continue

        classes = getattr(
            classifier,
            "classes_",
            None,
        )

        if classes is None:
            continue

        classes = list(classes)

        try:

            if hasattr(
                classifier,
                "predict_proba",
            ):

                values = (
                    classifier.predict_proba(
                        features
                    )
                )

                values = np.asarray(
                    values
                )

                if values.ndim == 2:
                    values = values[0]

            elif hasattr(
                classifier,
                "decision_function",
            ):

                values = (
                    classifier.decision_function(
                        features
                    )
                )

                values = np.asarray(
                    values
                )

                if values.ndim == 2:
                    values = values[0]

                values = sigmoid(
                    values
                )

            else:
                continue

        except Exception:
            continue

        local_scores = {}

        for i, label in enumerate(
            classes
        ):

            if i >= len(values):
                continue

            topic = str(
                label
            ).strip()

            if not topic:
                continue

            local_scores[topic] = max(
                float(values[i]),
                0.0,
            )

        if local_scores:
            model_topic_scores.append(
                normalize_scores(
                    local_scores
                )
            )

    if not model_topic_scores:
        return {}

    # --------------------------------------------------------
    # Average evidence from every available domain model.
    # --------------------------------------------------------

    aggregate = {}

    for local_scores in model_topic_scores:

        for topic, score in (
            local_scores.items()
        ):

            aggregate[topic] = (
                aggregate.get(
                    topic,
                    0.0,
                )
                + score
            )

    return normalize_scores(
        aggregate
    )


# ============================================================
# FUSE MAIN TOPIC SCORES
# ============================================================

def fuse_main_topic_scores(
    neighbor_scores,
    centroid_scores,
    supervised_scores,
    feedback_scores=None,
):

    feedback_scores = {
        label: score
        for label, score in (feedback_scores or {}).items()
        if score
    }
    feedback_weight = (
        MAIN_TOPIC_FEEDBACK_WEIGHT
        if any(feedback_scores.values())
        else 0.0
    )

    all_topics = (
        set(ALL_TOPICS)
        | set(neighbor_scores.keys())
        | set(centroid_scores.keys())
        | set(supervised_scores.keys())
        | set(feedback_scores.keys())
    )

    final_scores = {}

    for topic in all_topics:

        neighbor_score = (
            neighbor_scores.get(
                topic,
                0.0,
            )
        )

        centroid_score = (
            centroid_scores.get(
                topic,
                0.0,
            )
        )

        supervised_score = (
            supervised_scores.get(
                topic,
                0.0,
            )
        )

        existing_evidence = (
            MAIN_TOPIC_NEIGHBOR_WEIGHT
            * neighbor_score
            +
            MAIN_TOPIC_CENTROID_WEIGHT
            * centroid_score
            +
            MAIN_TOPIC_SUPERVISED_WEIGHT
            * supervised_score
        )
        final_scores[topic] = (
            (1.0 - feedback_weight) * existing_evidence
            + feedback_weight * feedback_scores.get(topic, 0.0)
        )

    return final_scores


# ============================================================
# TOP TWO
# ============================================================

def get_top_two(scores):

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if not ordered:
        return (
            None,
            0.0,
            None,
            0.0,
        )

    top_label = ordered[0][0]
    top_score = float(
        ordered[0][1]
    )

    if len(ordered) > 1:

        second_label = (
            ordered[1][0]
        )

        second_score = float(
            ordered[1][1]
        )

    else:

        second_label = None
        second_score = 0.0

    return (
        top_label,
        top_score,
        second_label,
        second_score,
    )


# ============================================================
# SUBTOPIC EVIDENCE
#
# Once the main topic is selected, find related topics from
# the same semantic neighborhood.
#
# We DO NOT require subtopics to belong to one of the old
# 19 domains.
# ============================================================

def get_subtopic_evidence(
    query_embedding,
    neighbors,
    selected_main_topic,
    global_centroid_scores,
    global_supervised_scores,
    feedback_scores=None,
):

    feedback_scores = {
        label: score
        for label, score in (feedback_scores or {}).items()
        if score
    }
    feedback_weight = (
        SUBTOPIC_FEEDBACK_WEIGHT
        if any(feedback_scores.values())
        else 0.0
    )

    neighbor_scores = {}
    cooccurrence_scores = {}
    best_similarity = {}
    topic_examples = {}

    # --------------------------------------------------------
    # Look at documents that support the selected main topic.
    # --------------------------------------------------------

    for item in neighbors:

        topics = item["topics"]

        if selected_main_topic not in topics:
            continue

        similarity = max(
            float(item["similarity"]),
            0.0,
        )

        weight = similarity ** 3

        for topic in topics:

            topic = str(
                topic
            ).strip()

            if not topic:
                continue

            if topic == selected_main_topic:
                continue

            neighbor_scores[topic] = (
                neighbor_scores.get(
                    topic,
                    0.0,
                )
                + weight
            )

            cooccurrence_scores[topic] = (
                cooccurrence_scores.get(
                    topic,
                    0.0,
                )
                + weight
            )

            old_similarity = (
                best_similarity.get(
                    topic,
                    -1.0,
                )
            )

            if similarity > old_similarity:

                best_similarity[topic] = (
                    similarity
                )

                topic_examples[topic] = (
                    item
                )

    neighbor_scores = normalize_scores(
        neighbor_scores
    )

    cooccurrence_scores = normalize_scores(
        cooccurrence_scores
    )

    # --------------------------------------------------------
    # Candidate topic pool.
    #
    # Include:
    #   - topics co-occurring with main topic
    #   - topics from nearest neighbors
    #   - all global centroid topics
    #   - all supervised topics
    #
    # But never return the main topic itself as its own
    # subtopic.
    # --------------------------------------------------------

    candidates = (
        set(neighbor_scores.keys())
        |
        set(cooccurrence_scores.keys())
        |
        set(global_centroid_scores.keys())
        |
        set(global_supervised_scores.keys())
        |
        set(feedback_scores.keys())
    )

    candidates.discard(
        selected_main_topic
    )

    # --------------------------------------------------------
    # Combine evidence.
    # --------------------------------------------------------

    centroid_scores = {
        topic: global_centroid_scores.get(
            topic,
            0.0,
        )
        for topic in candidates
    }

    supervised_scores = {
        topic: global_supervised_scores.get(
            topic,
            0.0,
        )
        for topic in candidates
    }

    centroid_scores = normalize_scores(
        centroid_scores
    )

    supervised_scores = normalize_scores(
        supervised_scores
    )

    combined_scores = {}

    semantic_support = {}

    for topic in candidates:

        neighbor_score = (
            neighbor_scores.get(
                topic,
                0.0,
            )
        )

        centroid_score = (
            centroid_scores.get(
                topic,
                0.0,
            )
        )

        supervised_score = (
            supervised_scores.get(
                topic,
                0.0,
            )
        )

        cooccurrence_score = (
            cooccurrence_scores.get(
                topic,
                0.0,
            )
        )

        semantic_score = (
            SUBTOPIC_NEIGHBOR_WEIGHT
            * neighbor_score
            +
            SUBTOPIC_CENTROID_WEIGHT
            * centroid_score
        )

        existing_evidence = (
            semantic_score
            +
            SUBTOPIC_SUPERVISED_WEIGHT
            * supervised_score
            +
            SUBTOPIC_COOCCURRENCE_WEIGHT
            * cooccurrence_score
        )
        final_score = (
            (1.0 - feedback_weight) * existing_evidence
            + feedback_weight * feedback_scores.get(topic, 0.0)
        )

        semantic_support[topic] = (
            semantic_score
        )

        combined_scores[topic] = (
            final_score
        )

    return (
        neighbor_scores,
        centroid_scores,
        supervised_scores,
        cooccurrence_scores,
        combined_scores,
        semantic_support,
        best_similarity,
        topic_examples,
    )


# ============================================================
# SELECT FINAL SUBTOPICS
# ============================================================

def select_final_subtopics(
    combined_scores,
):

    if not combined_scores:
        return []

    ordered = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_score = ordered[0][1]

    selected = []

    for topic, score in ordered:

        if len(selected) >= MAX_SUBTOPICS:
            break

        if best_score > 0:

            relative_score = (
                score / best_score
            )

        else:

            relative_score = 0.0

        if (
            not selected
            or relative_score
            >= MIN_SUBTOPIC_RELATIVE_SCORE
        ):

            selected.append(
                topic
            )

    return selected


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    final_main_topic,
    final_main_score,
    second_main_score,
    semantic_top_topic,
    supervised_top_topic,
    best_similarity,
):

    margin = (
        final_main_score
        - second_main_score
    )

    agreement = 0.0

    if (
        final_main_topic
        == semantic_top_topic
    ):
        agreement += 0.6

    if (
        supervised_top_topic is not None
        and final_main_topic
        == supervised_top_topic
    ):
        agreement += 0.4

    margin_score = min(
        max(
            margin * 8.0,
            0.0,
        ),
        1.0,
    )

    similarity_score = min(
        max(
            best_similarity,
            0.0,
        ),
        1.0,
    )

    evidence_score = (
        0.45
        * similarity_score
        +
        0.30
        * margin_score
        +
        0.25
        * agreement
    )

    if evidence_score >= 0.68:
        confidence = "HIGH"

    elif evidence_score >= 0.45:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    return (
        confidence,
        evidence_score,
        margin,
    )


# ============================================================
# PREDICT
# ============================================================

def predict(text, feedback_evidence=None, query_embedding=None):

    text = clean_text(text)

    if not text:
        return None

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    if query_embedding is None:
        query_embedding = encode_query(text)
    else:
        query_embedding = normalize_vector(query_embedding)
    feedback_evidence = feedback_evidence or {}

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    neighbors = retrieve_neighbors(
        query_embedding,
        SEMANTIC_NEIGHBORS,
    )

    if not neighbors:
        return None

    # ========================================================
    # MAIN TOPIC EVIDENCE
    #
    # NO DOMAIN FILTER.
    # ALL TOPICS ARE EVALUATED.
    # ========================================================

    (
        semantic_main_topic_scores,
        topic_best_similarity,
        topic_examples,
    ) = get_main_topic_neighbor_scores(
        neighbors
    )

    centroid_main_topic_scores = (
        get_global_topic_centroid_scores(
            query_embedding
        )
    )

    supervised_main_topic_scores = (
        get_global_supervised_topic_scores(
            text
        )
    )

    # ========================================================
    # FINAL MAIN TOPIC
    # ========================================================

    final_main_topic_scores = (
        fuse_main_topic_scores(
            semantic_main_topic_scores,
            centroid_main_topic_scores,
            supervised_main_topic_scores,
            feedback_evidence.get("main_topic"),
        )
    )

    (
        final_main_topic,
        final_main_topic_score,
        second_main_topic,
        second_main_topic_score,
    ) = get_top_two(
        final_main_topic_scores
    )

    if final_main_topic is None:
        return None

    # ========================================================
    # SEMANTIC TOP TOPIC
    # ========================================================

    (
        semantic_top_topic,
        semantic_top_topic_score,
        _,
        _,
    ) = get_top_two(
        semantic_main_topic_scores
    )

    # ========================================================
    # SUPERVISED TOP TOPIC
    # ========================================================

    (
        supervised_top_topic,
        supervised_top_topic_score,
        _,
        _,
    ) = get_top_two(
        supervised_main_topic_scores
    )

    # ========================================================
    # BEST SIMILARITY
    # ========================================================

    best_similarity = max(
        item["similarity"]
        for item in neighbors
    )

    # ========================================================
    # SUBTOPIC EVIDENCE
    # ========================================================

    (
        neighbor_subtopic_scores,
        centroid_subtopic_scores,
        supervised_subtopic_scores,
        cooccurrence_subtopic_scores,
        combined_subtopic_scores,
        semantic_subtopic_support,
        subtopic_best_similarity,
        subtopic_examples,
    ) = get_subtopic_evidence(
        query_embedding,
        neighbors,
        final_main_topic,
        centroid_main_topic_scores,
        supervised_main_topic_scores,
        feedback_evidence.get("subtopic"),
    )

    # ========================================================
    # FINAL SUBTOPICS
    # ========================================================

    final_subtopics = (
        select_final_subtopics(
            combined_subtopic_scores
        )
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    (
        confidence,
        evidence_score,
        main_topic_margin,
    ) = calculate_confidence(
        final_main_topic,
        final_main_topic_score,
        second_main_topic_score,
        semantic_top_topic,
        supervised_top_topic,
        best_similarity,
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    warnings_list = []

    if (
        supervised_top_topic is not None
        and supervised_top_topic
        != semantic_top_topic
    ):
        warnings_list.append(
            "Supervised and semantic topic evidence disagree."
        )

    if (
        final_main_topic
        != semantic_top_topic
    ):
        warnings_list.append(
            "Final main topic differs from the strongest "
            "semantic-neighbor topic."
        )

    if (
        supervised_top_topic is not None
        and final_main_topic
        != supervised_top_topic
    ):
        warnings_list.append(
            "Final main topic differs from the strongest "
            "supervised topic."
        )

    if best_similarity < 0.40:
        warnings_list.append(
            "Nearest semantic examples are not very strong matches."
        )

    if not final_subtopics:
        warnings_list.append(
            "No sufficiently supported subtopic was found."
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "text": text,

        # MAIN TOPIC
        "final_main_topic":
            final_main_topic,

        "final_main_topic_score":
            final_main_topic_score,

        "second_main_topic":
            second_main_topic,

        "second_main_topic_score":
            second_main_topic_score,

        # SUBTOPICS
        "final_subtopics":
            final_subtopics,

        # CONFIDENCE
        "confidence":
            confidence,

        "evidence_score":
            evidence_score,

        "best_similarity":
            best_similarity,

        "main_topic_margin":
            main_topic_margin,

        # NEIGHBORS
        "neighbors":
            neighbors,

        # MAIN TOPIC EVIDENCE
        "semantic_main_topic_scores":
            semantic_main_topic_scores,

        "centroid_main_topic_scores":
            centroid_main_topic_scores,

        "supervised_main_topic_scores":
            supervised_main_topic_scores,

        "final_main_topic_scores":
            final_main_topic_scores,

        "semantic_top_topic":
            semantic_top_topic,

        "semantic_top_topic_score":
            semantic_top_topic_score,

        "supervised_top_topic":
            supervised_top_topic,

        "supervised_top_topic_score":
            supervised_top_topic_score,

        # SUBTOPIC EVIDENCE
        "neighbor_subtopic_scores":
            neighbor_subtopic_scores,

        "centroid_subtopic_scores":
            centroid_subtopic_scores,

        "supervised_subtopic_scores":
            supervised_subtopic_scores,

        "cooccurrence_subtopic_scores":
            cooccurrence_subtopic_scores,

        "combined_subtopic_scores":
            combined_subtopic_scores,

        "semantic_subtopic_support":
            semantic_subtopic_support,

        "subtopic_best_similarity":
            subtopic_best_similarity,

        # EXAMPLES
        "topic_examples":
            topic_examples,

        "subtopic_examples":
            subtopic_examples,

        # WARNINGS
        "warnings":
            warnings_list,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print()
    print()
    print("=" * 72)
    print("FINAL OPEN-TOPIC AMHARIC CLASSIFICATION RESULT")
    print("=" * 72)

    # ========================================================
    # INPUT
    # ========================================================

    print()
    print("INPUT TEXT")
    print("-" * 72)
    print(result["text"])

    # ========================================================
    # MAIN TOPIC
    # ========================================================

    print()
    print("FINAL MAIN TOPIC PREDICTED:")
    print(
        result["final_main_topic"]
    )

    # ========================================================
    # SUBTOPICS
    # ========================================================

    print()
    print("FINAL SUBTOPICS PREDICTED:")
    print()

    if result["final_subtopics"]:

        for i, topic in enumerate(
            result["final_subtopics"],
            start=1,
        ):
            print(
                f"{i}. {topic}"
            )

    else:
        print(
            "No sufficiently supported subtopic found."
        )

    # ========================================================
    # SEMANTIC NEIGHBORS
    # ========================================================

    print()
    print("SEMANTIC EVIDENCE")
    print("-" * 72)

    for i, item in enumerate(
        result["neighbors"][:SHOW_NEIGHBORS],
        start=1,
    ):

        similarity = item["similarity"]
        domain = item["domain"]
        topics = item["topics"]

        if topics:
            topic_text = " | ".join(
                topics
            )
        else:
            topic_text = "(no topic)"

        marker = ""

        if (
            result["final_main_topic"]
            in topics
        ):
            marker = (
                "  <-- MAIN TOPIC SUPPORTED"
            )

        print()
        print(
            f"{i}. similarity = "
            f"{similarity:.4f}"
        )

        print(
            f"   original domain = "
            f"{domain}"
        )

        print(
            f"   indexed topic(s) = "
            f"{topic_text}{marker}"
        )

        print(
            f"   text = {item['text']}"
        )

    # ========================================================
    # MAIN TOPIC EVIDENCE
    # ========================================================

    print()
    print("MAIN TOPIC EVIDENCE")
    print("-" * 72)

    print()
    print(
        "Semantic nearest-neighbor top topic:"
    )

    print(
        f"  {result['semantic_top_topic']} "
        f"({result['semantic_top_topic_score']:.4f})"
    )

    print()
    print(
        "Global semantic-centroid top topic:"
    )

    centroid_top_topic, centroid_top_score, _, _ = (
        get_top_two(
            result[
                "centroid_main_topic_scores"
            ]
        )
    )

    print(
        f"  {centroid_top_topic} "
        f"({centroid_top_score:.4f})"
    )

    print()
    print(
        "Global supervised top topic:"
    )

    if result["supervised_top_topic"] is not None:

        print(
            f"  {result['supervised_top_topic']} "
            f"({result['supervised_top_topic_score']:.4f})"
        )

    else:
        print(
            "  No supervised topic evidence available."
        )

    print()
    print(
        "FINAL OPEN-TOPIC MAIN TOPIC:"
    )

    print(
        f"  {result['final_main_topic']}"
    )

    print()
    print(
        "Final main-topic score:"
    )

    print(
        f"  {result['final_main_topic_score']:.4f}"
    )

    print()
    print(
        "Second main-topic:"
    )

    print(
        f"  {result['second_main_topic']} "
        f"({result['second_main_topic_score']:.4f})"
    )

    print()
    print(
        "Main-topic decision margin:"
    )

    print(
        f"  {result['main_topic_margin']:.4f}"
    )

    print()
    print(
        "Best semantic similarity:"
    )

    print(
        f"  {result['best_similarity']:.4f}"
    )

    # ========================================================
    # TOP MAIN-TOPIC CANDIDATES
    # ========================================================

    print()
    print("TOP MAIN-TOPIC CANDIDATES")
    print("-" * 72)

    ordered_main_topics = sorted(
        result[
            "final_main_topic_scores"
        ].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (topic, score) in enumerate(
        ordered_main_topics[
            :SHOW_MAIN_TOPIC_CANDIDATES
        ],
        start=1,
    ):

        semantic_score = (
            result[
                "semantic_main_topic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        centroid_score = (
            result[
                "centroid_main_topic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        supervised_score = (
            result[
                "supervised_main_topic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        marker = ""

        if (
            topic
            == result["final_main_topic"]
        ):
            marker = "  <-- SELECTED"

        print()
        print(
            f"{rank}. {topic}{marker}"
        )

        print(
            f"   final = {score:.4f}"
        )

        print(
            f"   neighbor = {semantic_score:.4f}"
        )

        print(
            f"   centroid = {centroid_score:.4f}"
        )

        print(
            f"   supervised = {supervised_score:.4f}"
        )

    # ========================================================
    # SUBTOPIC EVIDENCE
    # ========================================================

    print()
    print("SUBTOPIC EVIDENCE")
    print("-" * 72)

    ordered_subtopics = sorted(
        result[
            "combined_subtopic_scores"
        ].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for topic, score in (
        ordered_subtopics[
            :SHOW_SUBTOPIC_CANDIDATES
        ]
    ):

        neighbor_score = (
            result[
                "neighbor_subtopic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        centroid_score = (
            result[
                "centroid_subtopic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        supervised_score = (
            result[
                "supervised_subtopic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        cooccurrence_score = (
            result[
                "cooccurrence_subtopic_scores"
            ].get(
                topic,
                0.0,
            )
        )

        semantic_support = (
            result[
                "semantic_subtopic_support"
            ].get(
                topic,
                0.0,
            )
        )

        marker = ""

        if (
            topic
            in result["final_subtopics"]
        ):
            marker = "  <-- SELECTED"

        print()
        print(
            f"{topic}{marker}"
        )

        print(
            f"  final score: "
            f"{score:.4f}"
        )

        print(
            f"  semantic neighbor: "
            f"{neighbor_score:.4f}"
        )

        print(
            f"  semantic centroid: "
            f"{centroid_score:.4f}"
        )

        print(
            f"  supervised: "
            f"{supervised_score:.4f}"
        )

        print(
            f"  co-occurrence: "
            f"{cooccurrence_score:.4f}"
        )

        print(
            f"  semantic support: "
            f"{semantic_support:.4f}"
        )

        example = (
            result[
                "subtopic_examples"
            ].get(
                topic
            )
        )

        if example is not None:

            print(
                f"  example: "
                f"{example['text']}"
            )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    print()
    print("CONFIDENCE")
    print("-" * 72)

    print(
        result["confidence"]
    )

    print(
        f"Evidence score: "
        f"{result['evidence_score']:.4f}"
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    print()
    print("ANNOTATION / MODEL WARNING")
    print("-" * 72)

    if result["warnings"]:

        for warning in result["warnings"]:

            print(
                f"WARNING: {warning}"
            )

    else:

        print(
            "None."
        )

    print()
    print("=" * 72)


# ============================================================
# INFO COMMAND
# ============================================================

def print_info():

    print()
    print("=" * 72)
    print("OPEN-TOPIC SYSTEM INFORMATION")
    print("=" * 72)

    print()
    print(
        "Semantic model:"
    )

    print(
        model_name_from_index
    )

    print()
    print(
        "Indexed documents:"
    )

    print(
        f"{len(TRAIN_TEXTS):,}"
    )

    print()
    print(
        "Embedding dimension:"
    )

    print(
        TRAIN_EMBEDDINGS.shape[1]
    )

    print()
    print(
        "Discovered domains:"
    )

    print(
        len(INDEX_DOMAINS)
    )

    print()
    print(
        "ALL AVAILABLE TOPICS:"
    )

    print(
        len(ALL_TOPICS)
    )

    print()
    print(
        "Main topic restriction:"
    )

    print(
        "NONE"
    )

    print()
    print(
        "Every topic in the semantic index is eligible."
    )

    print()
    print(
        "Semantic neighbors:"
    )

    print(
        SEMANTIC_NEIGHBORS
    )

    print()
    print(
        "Main-topic weights:"
    )

    print(
        f"  semantic neighbors = "
        f"{MAIN_TOPIC_NEIGHBOR_WEIGHT}"
    )

    print(
        f"  semantic centroid = "
        f"{MAIN_TOPIC_CENTROID_WEIGHT}"
    )

    print(
        f"  supervised = "
        f"{MAIN_TOPIC_SUPERVISED_WEIGHT}"
    )

    print()
    print(
        "Subtopic weights:"
    )

    print(
        f"  semantic neighbors = "
        f"{SUBTOPIC_NEIGHBOR_WEIGHT}"
    )

    print(
        f"  semantic centroid = "
        f"{SUBTOPIC_CENTROID_WEIGHT}"
    )

    print(
        f"  supervised = "
        f"{SUBTOPIC_SUPERVISED_WEIGHT}"
    )

    print(
        f"  co-occurrence = "
        f"{SUBTOPIC_COOCCURRENCE_WEIGHT}"
    )

    print()
    print(
        "Maximum subtopics:"
    )

    print(
        MAX_SUBTOPICS
    )

    print()
    print("=" * 72)


# ============================================================
# INTERACTIVE LOOP
# ============================================================

print()
print()
print("=" * 72)
print("READY FOR OPEN-TOPIC AMHARIC INPUT")
print("=" * 72)

print()
print(
    "Type an Amharic sentence and press ENTER."
)

print()
print(
    "IMPORTANT:"
)

print(
    "The main topic is NOT restricted to the old 19 domains."
)

print(
    "Every topic discovered in the semantic index can win."
)

print()
print(
    "Commands:"
)

print(
    "  info = show system information"
)

print(
    "  exit = quit"
)

print(
    "  quit = quit"
)


while not os.environ.get("AMHARIC_HYBRID_NON_INTERACTIVE"):

    try:

        print()

        user_input = input(
            "INPUT > "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):

        print()
        print(
            "Exiting..."
        )

        break

    if not user_input:
        continue

    command = (
        user_input.lower()
    )

    if command in [
        "exit",
        "quit",
    ]:

        print()
        print(
            "Exiting..."
        )

        break

    if command == "info":

        print_info()

        continue

    try:

        result = predict(
            user_input
        )

        if result is None:

            print()
            print(
                "Could not classify the input."
            )

            continue

        print_result(
            result
        )

    except Exception as error:

        print()
        print("=" * 72)
        print("PREDICTION ERROR")
        print("=" * 72)
        print()

        print(
            type(error).__name__
        )

        print()

        print(
            str(error)
        )

        print()

        print(
            "The system did not hide the error."
        )

        print(
            "Paste the complete error output here "
            "if this happens."
        )

