# =============================================================================
# AMHARIC NLP
# SEMANTIC TRANSFORMER + SUPERVISED HYBRID
# HIERARCHICAL INFERENCE
#
# FINAL OUTPUT:
#
#     FINAL MAIN TOPIC PREDICTED
#     FINAL SUBTOPIC PREDICTED
#
# The system always selects the best available main topic and subtopic.
# =============================================================================


import re
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_NAME = "AMHARIC SEMANTIC TRANSFORMER HYBRID HIERARCHICAL INFERENCE"


# =============================================================================
# PRETRAINED SEMANTIC MODEL
# =============================================================================

SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-base"


# =============================================================================
# PROJECT PATHS
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR = PROJECT_ROOT / "data"

PREPARED_DATA_DIR = DATA_DIR / "prepared"


# =============================================================================
# MODEL PATHS
# =============================================================================

DOMAIN_MODEL_PATH = (
    MODELS_DIR
    / "final_professional_domain_classification_model.joblib"
)

SUBTOPIC_MODEL_PATH = (
    MODELS_DIR
    / "final_domain_aware_subtopic_model.joblib"
)

SEMANTIC_INDEX_PATH = (
    MODELS_DIR
    / "semantic_transformer_hierarchical_index.joblib"
)


# =============================================================================
# TRAINING DATA
# =============================================================================

TRAIN_DATA_PATH = (
    PREPARED_DATA_DIR
    / "train_prepared.csv"
)


# =============================================================================
# PARAMETERS
# =============================================================================

SEMANTIC_BATCH_SIZE = 32

SEMANTIC_NEIGHBORS = 15

MAX_REFERENCE_DOCUMENTS = None


# =============================================================================
# HYBRID WEIGHTS
# =============================================================================

SUPERVISED_WEIGHT = 0.40

NEIGHBOR_WEIGHT = 0.40

CENTROID_WEIGHT = 0.20


# =============================================================================
# SUBTOPIC WEIGHTS
# =============================================================================

MAX_SUBTOPICS = 5

SUBTOPIC_MODEL_WEIGHT = 0.65

SUBTOPIC_SEMANTIC_WEIGHT = 0.35


# =============================================================================
# TEXT CLEANING
# =============================================================================


def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =============================================================================
# VALIDATE MODEL PATHS
# =============================================================================


def validate_model_paths():

    print()
    print("Loading models...")

    if not DOMAIN_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nDomain model not found:\n{DOMAIN_MODEL_PATH}"
        )

    if not SUBTOPIC_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nSubtopic model not found:\n{SUBTOPIC_MODEL_PATH}"
        )

    print("Models found.")


# =============================================================================
# LOAD MODELS
# =============================================================================


def load_models():

    print("Loading domain model...")

    domain_bundle = joblib.load(
        DOMAIN_MODEL_PATH
    )

    print("Domain model loaded.")

    print("Loading subtopic model...")

    subtopic_bundle = joblib.load(
        SUBTOPIC_MODEL_PATH
    )

    print("Subtopic model loaded.")

    return (
        domain_bundle,
        subtopic_bundle
    )


# =============================================================================
# EXTRACT DOMAIN COMPONENTS
# =============================================================================


def extract_domain_components(domain_bundle):

    if not isinstance(domain_bundle, dict):

        raise ValueError(
            "Domain model must be a dictionary."
        )

    classifier = domain_bundle.get("model")

    word_vectorizer = domain_bundle.get(
        "word_vectorizer"
    )

    character_vectorizer = domain_bundle.get(
        "character_vectorizer"
    )

    domains = domain_bundle.get("domains")

    if classifier is None:

        raise ValueError(
            "Domain classifier not found."
        )

    if word_vectorizer is None:

        raise ValueError(
            "Word vectorizer not found."
        )

    if character_vectorizer is None:

        raise ValueError(
            "Character vectorizer not found."
        )

    if domains is None:

        raise ValueError(
            "Domain list not found."
        )

    return {

        "classifier":
            classifier,

        "word_vectorizer":
            word_vectorizer,

        "character_vectorizer":
            character_vectorizer,

        "domains":
            list(domains)

    }


# =============================================================================
# LOAD SEMANTIC MODEL
# =============================================================================


def load_semantic_model():

    print()
    print("Loading semantic transformer...")

    print(
        SEMANTIC_MODEL_NAME
    )

    model = SentenceTransformer(
        SEMANTIC_MODEL_NAME
    )

    print(
        "Semantic transformer loaded."
    )

    return model


# =============================================================================
# FIND TRAINING DATASET
# =============================================================================


def find_training_dataset():

    candidates = []

    if TRAIN_DATA_PATH.exists():

        candidates.append(
            TRAIN_DATA_PATH
        )

    if PREPARED_DATA_DIR.exists():

        for path in PREPARED_DATA_DIR.glob(
            "*.csv"
        ):

            if path not in candidates:

                candidates.append(path)

    if not candidates:

        raise FileNotFoundError(
            "No training CSV dataset found."
        )

    valid_candidates = []

    for path in candidates:

        try:

            dataframe = pd.read_csv(path)

            columns = dataframe.columns.tolist()

            if (
                "text" in columns
                and
                (
                    "consolidated_domains" in columns
                    or
                    "domain" in columns
                )
            ):

                valid_candidates.append(
                    path
                )

        except Exception:

            continue

    if TRAIN_DATA_PATH in valid_candidates:

        return TRAIN_DATA_PATH

    if valid_candidates:

        return valid_candidates[0]

    raise ValueError(
        "Could not find a valid training dataset."
    )


# =============================================================================
# LOAD TRAINING DATA
# =============================================================================


def load_training_data(trained_domains):

    dataset_path = find_training_dataset()

    dataframe = pd.read_csv(
        dataset_path
    )

    text_column = "text"

    if "consolidated_domains" in dataframe.columns:

        domain_column = "consolidated_domains"

    elif "domain" in dataframe.columns:

        domain_column = "domain"

    else:

        raise ValueError(
            "No domain column found."
        )

    topic_column = None

    topic_candidates = [

        "topic_canonical",

        "topic_normalized",

        "topic",

        "topics"

    ]

    for candidate in topic_candidates:

        if candidate in dataframe.columns:

            topic_column = candidate

            break

    columns_to_keep = [

        text_column,
        domain_column

    ]

    if topic_column is not None:

        columns_to_keep.append(
            topic_column
        )

    dataframe = dataframe[
        columns_to_keep
    ].copy()

    dataframe[text_column] = (
        dataframe[text_column]
        .astype(str)
        .apply(clean_text)
    )

    dataframe = dataframe[
        dataframe[text_column].str.len() > 0
    ].copy()

    dataframe[domain_column] = (
        dataframe[domain_column]
        .astype(str)
        .apply(clean_text)
    )

    dataframe = dataframe[
        dataframe[domain_column].isin(
            trained_domains
        )
    ].copy()

    if topic_column is not None:

        dataframe[topic_column] = (
            dataframe[topic_column]
            .fillna("")
            .astype(str)
            .apply(clean_text)
        )

    dataframe = dataframe.drop_duplicates(
        subset=[text_column]
    ).reset_index(
        drop=True
    )

    if MAX_REFERENCE_DOCUMENTS is not None:

        dataframe = dataframe.sample(

            min(
                MAX_REFERENCE_DOCUMENTS,
                len(dataframe)
            ),

            random_state=42

        ).reset_index(
            drop=True
        )

    if len(dataframe) == 0:

        raise ValueError(
            "No valid training documents found."
        )

    return {

        "dataframe":
            dataframe,

        "dataset_path":
            dataset_path,

        "text_column":
            text_column,

        "domain_column":
            domain_column,

        "topic_column":
            topic_column

    }


# =============================================================================
# CREATE E5 EMBEDDINGS
# =============================================================================


def create_e5_embeddings(
    model,
    texts,
    prefix="passage: "
):

    prepared_texts = [

        prefix + clean_text(text)

        for text in texts

    ]

    embeddings = model.encode(

        prepared_texts,

        batch_size=SEMANTIC_BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True

    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# =============================================================================
# BUILD DOMAIN CENTROIDS
# =============================================================================


def build_domain_centroids(
    embeddings,
    domains,
    trained_domains
):

    centroids = {}

    domain_counts = {}

    for domain in trained_domains:

        indices = [

            i

            for i, value
            in enumerate(domains)

            if value == domain

        ]

        if not indices:

            continue

        domain_embeddings = embeddings[
            indices
        ]

        centroid = np.mean(

            domain_embeddings,

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

            centroid = centroid / norm

        centroids[domain] = centroid

        domain_counts[domain] = len(
            indices
        )

    return (
        centroids,
        domain_counts
    )


# =============================================================================
# BUILD NEAREST NEIGHBOR INDEX
# =============================================================================


def build_neighbor_index(embeddings):

    neighbor_model = NearestNeighbors(

        n_neighbors=min(

            SEMANTIC_NEIGHBORS,

            len(embeddings)

        ),

        metric="cosine",

        algorithm="auto"

    )

    neighbor_model.fit(
        embeddings
    )

    return neighbor_model


# =============================================================================
# BUILD TOPIC KNOWLEDGE
# =============================================================================


def build_topic_knowledge(
    dataframe,
    domain_column,
    topic_column
):

    knowledge = {}

    if topic_column is None:

        return knowledge

    for domain in dataframe[
        domain_column
    ].unique():

        domain_dataframe = dataframe[

            dataframe[domain_column]
            == domain

        ]

        topic_counts = Counter()

        for topic in domain_dataframe[
            topic_column
        ]:

            topic = clean_text(topic)

            if not topic:

                continue

            possible_topics = re.split(

                r"[,;|]",

                topic

            )

            for value in possible_topics:

                value = clean_text(value)

                if value:

                    topic_counts[value] += 1

        knowledge[domain] = dict(
            topic_counts
        )

    return knowledge


# =============================================================================
# BUILD SEMANTIC INDEX
# =============================================================================


def build_semantic_index(
    semantic_model,
    trained_domains
):

    print()
    print("Building semantic index...")

    training_data = load_training_data(
        trained_domains
    )

    dataframe = training_data[
        "dataframe"
    ]

    text_column = training_data[
        "text_column"
    ]

    domain_column = training_data[
        "domain_column"
    ]

    topic_column = training_data[
        "topic_column"
    ]

    texts = dataframe[
        text_column
    ].tolist()

    domains = dataframe[
        domain_column
    ].tolist()

    if topic_column is not None:

        topics = dataframe[
            topic_column
        ].tolist()

    else:

        topics = [

            ""

            for _ in texts

        ]

    print(
        f"Creating embeddings for {len(texts)} documents..."
    )

    embeddings = create_e5_embeddings(

        semantic_model,

        texts,

        prefix="passage: "

    )

    centroids, domain_counts = (

        build_domain_centroids(

            embeddings,

            domains,

            trained_domains

        )

    )

    neighbor_model = build_neighbor_index(
        embeddings
    )

    topic_knowledge = build_topic_knowledge(

        dataframe,

        domain_column,

        topic_column

    )

    index = {

        "version":
            "semantic_transformer_v1",

        "semantic_model_name":
            SEMANTIC_MODEL_NAME,

        "dataset_path":
            str(
                training_data[
                    "dataset_path"
                ]
            ),

        "texts":
            texts,

        "domains":
            domains,

        "topics":
            topics,

        "embeddings":
            embeddings,

        "centroids":
            centroids,

        "domain_counts":
            domain_counts,

        "neighbor_model":
            neighbor_model,

        "topic_knowledge":
            topic_knowledge,

        "trained_domains":
            trained_domains

    }

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Saving semantic index...")

    joblib.dump(

        index,

        SEMANTIC_INDEX_PATH,

        compress=3

    )

    print(
        "Semantic index saved."
    )

    return index


# =============================================================================
# LOAD OR BUILD SEMANTIC INDEX
# =============================================================================


def load_or_build_index(
    semantic_model,
    trained_domains
):

    rebuild = False

    if not SEMANTIC_INDEX_PATH.exists():

        rebuild = True

    else:

        print()
        print(
            "Loading existing semantic index..."
        )

        try:

            index = joblib.load(
                SEMANTIC_INDEX_PATH
            )

            required_keys = [

                "texts",

                "domains",

                "topics",

                "embeddings",

                "centroids",

                "neighbor_model",

                "trained_domains"

            ]

            missing = [

                key

                for key in required_keys

                if key not in index

            ]

            if missing:

                rebuild = True

            else:

                print(
                    "Semantic index loaded."
                )

        except Exception:

            rebuild = True

    if rebuild:

        index = build_semantic_index(

            semantic_model,

            trained_domains

        )

    return index


# =============================================================================
# CREATE SUPERVISED FEATURES
# =============================================================================


def create_supervised_features(
    text,
    word_vectorizer,
    character_vectorizer
):

    texts = [

        clean_text(text)

    ]

    word_features = (
        word_vectorizer.transform(
            texts
        )
    )

    character_features = (
        character_vectorizer.transform(
            texts
        )
    )

    features = hstack(

        [

            word_features,

            character_features

        ]

    )

    return features


# =============================================================================
# SOFTMAX
# =============================================================================


def stable_softmax(values):

    values = np.asarray(

        values,

        dtype=np.float64

    )

    values = values - np.max(
        values
    )

    exp_values = np.exp(
        values
    )

    total = np.sum(
        exp_values
    )

    if total == 0:

        return (

            np.ones_like(values)

            / len(values)

        )

    return exp_values / total


# =============================================================================
# SUPERVISED DOMAIN SCORES
# =============================================================================


def get_supervised_domain_scores(
    text,
    domain_components
):

    classifier = domain_components[
        "classifier"
    ]

    word_vectorizer = domain_components[
        "word_vectorizer"
    ]

    character_vectorizer = domain_components[
        "character_vectorizer"
    ]

    domains = domain_components[
        "domains"
    ]

    features = create_supervised_features(

        text,

        word_vectorizer,

        character_vectorizer

    )

    classifier_classes = list(
        classifier.classes_
    )

    if hasattr(
        classifier,
        "predict_proba"
    ):

        probabilities = classifier.predict_proba(
            features
        )[0]

    elif hasattr(
        classifier,
        "decision_function"
    ):

        decision_scores = classifier.decision_function(
            features
        )

        decision_scores = np.asarray(
            decision_scores
        )

        if decision_scores.ndim > 1:

            decision_scores = decision_scores[0]

        probabilities = stable_softmax(
            decision_scores
        )

    else:

        prediction = classifier.predict(
            features
        )[0]

        probabilities = np.zeros(
            len(classifier_classes)
        )

        prediction_index = classifier_classes.index(
            prediction
        )

        probabilities[
            prediction_index
        ] = 1.0

    scores = {

        domain: 0.0

        for domain in domains

    }

    for label, probability in zip(

        classifier_classes,

        probabilities

    ):

        if label in scores:

            scores[label] = float(
                probability
            )

    predicted_index = int(

        np.argmax(
            probabilities
        )

    )

    predicted_domain = classifier_classes[
        predicted_index
    ]

    return {

        "scores":
            scores,

        "predicted_domain":
            predicted_domain

    }


# =============================================================================
# CREATE QUERY EMBEDDING
# =============================================================================


def create_query_embedding(
    semantic_model,
    text
):

    embedding = create_e5_embeddings(

        semantic_model,

        [text],

        prefix="query: "

    )

    return embedding[0]


# =============================================================================
# SEMANTIC NEIGHBOR ANALYSIS
# =============================================================================


def analyze_semantic_neighbors(
    query_embedding,
    semantic_index,
    trained_domains
):

    neighbor_model = semantic_index[
        "neighbor_model"
    ]

    texts = semantic_index[
        "texts"
    ]

    domains = semantic_index[
        "domains"
    ]

    topics = semantic_index[
        "topics"
    ]

    number_of_neighbors = min(

        SEMANTIC_NEIGHBORS,

        len(texts)

    )

    distances, indices = neighbor_model.kneighbors(

        query_embedding.reshape(
            1,
            -1
        ),

        n_neighbors=number_of_neighbors

    )

    distances = distances[0]

    indices = indices[0]

    similarities = 1.0 - distances

    domain_votes = {

        domain: 0.0

        for domain in trained_domains

    }

    neighbor_records = []

    for similarity, index in zip(

        similarities,

        indices

    ):

        domain = domains[index]

        topic = topics[index]

        similarity = float(
            similarity
        )

        weight = max(
            similarity,
            0.0
        ) ** 2

        if domain in domain_votes:

            domain_votes[
                domain
            ] += weight

        neighbor_records.append({

            "domain":
                domain,

            "topic":
                topic,

            "similarity":
                similarity

        })

    total_vote = sum(
        domain_votes.values()
    )

    if total_vote > 0:

        domain_scores = {

            domain:
            value / total_vote

            for domain, value

            in domain_votes.items()

        }

    else:

        domain_scores = {

            domain: 0.0

            for domain
            in trained_domains

        }

    return {

        "domain_scores":
            domain_scores,

        "neighbors":
            neighbor_records

    }


# =============================================================================
# DOMAIN CENTROID ANALYSIS
# =============================================================================


def analyze_domain_centroids(
    query_embedding,
    semantic_index,
    trained_domains
):

    centroids = semantic_index[
        "centroids"
    ]

    query_embedding = np.asarray(

        query_embedding,

        dtype=np.float32

    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm > 0:

        query_embedding = (

            query_embedding
            /
            query_norm

        )

    scores = {}

    for domain in trained_domains:

        centroid = centroids.get(
            domain
        )

        if centroid is None:

            scores[domain] = 0.0

            continue

        centroid = np.asarray(

            centroid,

            dtype=np.float32

        )

        centroid_norm = np.linalg.norm(
            centroid
        )

        if centroid_norm > 0:

            centroid = centroid / centroid_norm

        similarity = float(

            np.dot(

                query_embedding,

                centroid

            )

        )

        similarity = max(
            similarity,
            0.0
        )

        scores[domain] = similarity

    total = sum(
        scores.values()
    )

    if total > 0:

        scores = {

            domain:
            value / total

            for domain, value

            in scores.items()

        }

    return scores


# =============================================================================
# NORMALIZE SCORES
# =============================================================================


def normalize_score_dictionary(
    scores,
    domains
):

    values = np.array(

        [

            scores.get(
                domain,
                0.0
            )

            for domain in domains

        ],

        dtype=np.float64

    )

    total = np.sum(values)

    if total <= 0:

        values = (

            np.ones(len(domains))

            / len(domains)

        )

    else:

        values = values / total

    return {

        domain:
        float(value)

        for domain, value

        in zip(
            domains,
            values
        )

    }


# =============================================================================
# HYBRID DOMAIN PREDICTION
# =============================================================================


def make_hybrid_domain_prediction(

    supervised_result,

    semantic_neighbor_result,

    centroid_scores,

    trained_domains

):

    supervised_scores = normalize_score_dictionary(

        supervised_result[
            "scores"
        ],

        trained_domains

    )

    neighbor_scores = normalize_score_dictionary(

        semantic_neighbor_result[
            "domain_scores"
        ],

        trained_domains

    )

    centroid_scores = normalize_score_dictionary(

        centroid_scores,

        trained_domains

    )

    hybrid_scores = {}

    for domain in trained_domains:

        hybrid_score = (

            SUPERVISED_WEIGHT
            *
            supervised_scores.get(
                domain,
                0.0
            )

            +

            NEIGHBOR_WEIGHT
            *
            neighbor_scores.get(
                domain,
                0.0
            )

            +

            CENTROID_WEIGHT
            *
            centroid_scores.get(
                domain,
                0.0
            )

        )

        hybrid_scores[
            domain
        ] = float(
            hybrid_score
        )

    predicted_domain = max(

        hybrid_scores,

        key=hybrid_scores.get

    )

    return {

        "predicted_domain":
            predicted_domain,

        "scores":
            hybrid_scores

    }


# =============================================================================
# PREDICT SUBTOPICS FROM EXISTING MODEL
# =============================================================================


def predict_model_subtopics(

    text,

    domain,

    subtopic_bundle

):

    if not isinstance(
        subtopic_bundle,
        dict
    ):

        return []

    domain_models = subtopic_bundle.get(

        "domain_models",

        {}

    )

    domain_model = domain_models.get(
        domain
    )

    if domain_model is None:

        return []

    word_vectorizer = subtopic_bundle.get(
        "word_vectorizer"
    )

    character_vectorizer = subtopic_bundle.get(
        "character_vectorizer"
    )

    if (

        word_vectorizer is None

        or

        character_vectorizer is None

    ):

        return []

    try:

        texts = [

            clean_text(text)

        ]

        word_features = word_vectorizer.transform(
            texts
        )

        character_features = character_vectorizer.transform(
            texts
        )

        features = hstack(

            [

                word_features,

                character_features

            ]

        )

        if isinstance(
            domain_model,
            dict
        ):

            classifier = domain_model.get(
                "model"
            )

        else:

            classifier = domain_model

        if classifier is None:

            return []

        predictions = []

        if hasattr(
            classifier,
            "predict_proba"
        ):

            probabilities = classifier.predict_proba(
                features
            )[0]

            classes = list(
                classifier.classes_
            )

            for topic, score in zip(

                classes,

                probabilities

            ):

                predictions.append({

                    "topic":
                        str(topic),

                    "score":
                        float(score)

                })

        elif hasattr(
            classifier,
            "decision_function"
        ):

            decision_scores = classifier.decision_function(
                features
            )

            decision_scores = np.asarray(
                decision_scores
            )

            if decision_scores.ndim > 1:

                decision_scores = decision_scores[0]

            probabilities = stable_softmax(
                decision_scores
            )

            classes = list(
                classifier.classes_
            )

            for topic, score in zip(

                classes,

                probabilities

            ):

                predictions.append({

                    "topic":
                        str(topic),

                    "score":
                        float(score)

                })

        return sorted(

            predictions,

            key=lambda item:
                item["score"],

            reverse=True

        )

    except Exception:

        return []


# =============================================================================
# SEMANTIC SUBTOPIC PREDICTION
# =============================================================================


def predict_semantic_subtopics(

    predicted_domain,

    semantic_neighbors

):

    topic_scores = defaultdict(
        float
    )

    for neighbor in semantic_neighbors:

        if (

            neighbor["domain"]

            !=

            predicted_domain

        ):

            continue

        topic = clean_text(
            neighbor.get(
                "topic",
                ""
            )
        )

        if not topic:

            continue

        similarity = float(
            neighbor["similarity"]
        )

        weight = max(
            similarity,
            0.0
        ) ** 2

        possible_topics = re.split(

            r"[,;|]",

            topic

        )

        for value in possible_topics:

            value = clean_text(
                value
            )

            if value:

                topic_scores[
                    value
                ] += weight

    if not topic_scores:

        return []

    total = sum(
        topic_scores.values()
    )

    predictions = []

    for topic, score in topic_scores.items():

        normalized_score = (

            score / total

            if total > 0

            else 0.0

        )

        predictions.append({

            "topic":
                topic,

            "score":
                float(normalized_score)

        })

    return sorted(

        predictions,

        key=lambda item:
            item["score"],

        reverse=True

    )


# =============================================================================
# COMBINE SUBTOPIC PREDICTIONS
# =============================================================================


def combine_subtopic_predictions(

    domain,

    model_predictions,

    semantic_predictions,

    topic_knowledge

):

    combined = defaultdict(
        float
    )

    for prediction in model_predictions:

        topic = clean_text(
            prediction["topic"]
        )

        if topic:

            combined[topic] += (

                SUBTOPIC_MODEL_WEIGHT

                *

                float(
                    prediction["score"]
                )

            )

    for prediction in semantic_predictions:

        topic = clean_text(
            prediction["topic"]
        )

        if topic:

            combined[topic] += (

                SUBTOPIC_SEMANTIC_WEIGHT

                *

                float(
                    prediction["score"]
                )

            )

    predictions = [

        {

            "topic":
                topic,

            "score":
                score

        }

        for topic, score
        in combined.items()

    ]

    predictions = sorted(

        predictions,

        key=lambda item:
            item["score"],

        reverse=True

    )

    # Fallback if no subtopic prediction exists
    if not predictions:

        domain_topics = topic_knowledge.get(

            domain,

            {}

        )

        if domain_topics:

            best_topic = max(

                domain_topics,

                key=domain_topics.get

            )

            predictions.append({

                "topic":
                    best_topic,

                "score":
                    0.0

            })

    return predictions[
        :MAX_SUBTOPICS
    ]


# =============================================================================
# ENSURE SUBTOPIC OUTPUT
# =============================================================================


def ensure_subtopic_output(

    predicted_domain,

    subtopics

):

    if subtopics:

        return subtopics

    return [

        {

            "topic":
                predicted_domain,

            "score":
                0.0

        }

    ]


# =============================================================================
# MAIN HIERARCHICAL PREDICTION
# =============================================================================


def predict_hierarchical(

    text,

    domain_components,

    subtopic_bundle,

    semantic_model,

    semantic_index

):

    text = clean_text(text)

    trained_domains = domain_components[
        "domains"
    ]

    # -------------------------------------------------------------------------
    # 1. SUPERVISED DOMAIN PREDICTION
    # -------------------------------------------------------------------------

    supervised_result = get_supervised_domain_scores(

        text,

        domain_components

    )

    # -------------------------------------------------------------------------
    # 2. CREATE SEMANTIC QUERY EMBEDDING
    # -------------------------------------------------------------------------

    query_embedding = create_query_embedding(

        semantic_model,

        text

    )

    # -------------------------------------------------------------------------
    # 3. SEMANTIC NEIGHBORS
    # -------------------------------------------------------------------------

    semantic_neighbor_result = analyze_semantic_neighbors(

        query_embedding,

        semantic_index,

        trained_domains

    )

    # -------------------------------------------------------------------------
    # 4. DOMAIN CENTROIDS
    # -------------------------------------------------------------------------

    centroid_scores = analyze_domain_centroids(

        query_embedding,

        semantic_index,

        trained_domains

    )

    # -------------------------------------------------------------------------
    # 5. FINAL MAIN TOPIC
    # -------------------------------------------------------------------------

    hybrid_result = make_hybrid_domain_prediction(

        supervised_result,

        semantic_neighbor_result,

        centroid_scores,

        trained_domains

    )

    predicted_domain = hybrid_result[
        "predicted_domain"
    ]

    # -------------------------------------------------------------------------
    # 6. SUBTOPIC MODEL PREDICTION
    # -------------------------------------------------------------------------

    model_subtopics = predict_model_subtopics(

        text,

        predicted_domain,

        subtopic_bundle

    )

    # -------------------------------------------------------------------------
    # 7. SEMANTIC SUBTOPIC PREDICTION
    # -------------------------------------------------------------------------

    semantic_subtopics = predict_semantic_subtopics(

        predicted_domain,

        semantic_neighbor_result[
            "neighbors"
        ]

    )

    # -------------------------------------------------------------------------
    # 8. COMBINE SUBTOPICS
    # -------------------------------------------------------------------------

    subtopics = combine_subtopic_predictions(

        predicted_domain,

        model_subtopics,

        semantic_subtopics,

        semantic_index.get(

            "topic_knowledge",

            {}

        )

    )

    # -------------------------------------------------------------------------
    # 9. GUARANTEE SUBTOPIC
    # -------------------------------------------------------------------------

    subtopics = ensure_subtopic_output(

        predicted_domain,

        subtopics

    )

    return {

        "text":
            text,

        "main_topic":
            predicted_domain,

        "subtopics":
            subtopics

    }


# =============================================================================
# PRINT FINAL RESULT
# =============================================================================


def print_prediction(result):

    print()

    print("=" * 70)

    print(
        "FINAL HIERARCHICAL PREDICTION"
    )

    print("=" * 70)

    print()

    print(
        "INPUT TEXT:"
    )

    print(
        result["text"]
    )

    print()

    print("-" * 70)

    print(
        "FINAL MAIN TOPIC PREDICTED:"
    )

    print("-" * 70)

    print()

    print(
        result["main_topic"]
    )

    print()

    print("-" * 70)

    print(
        "FINAL SUBTOPIC PREDICTED:"
    )

    print("-" * 70)

    print()

    # Print only the best subtopic
    if result["subtopics"]:

        print(
            result["subtopics"][0]["topic"]
        )

    else:

        print(
            result["main_topic"]
        )

    print()

    print("=" * 70)

    print()


# =============================================================================
# INTERACTIVE INFERENCE
# =============================================================================


def interactive_inference(

    domain_components,

    subtopic_bundle,

    semantic_model,

    semantic_index

):

    print()

    print("=" * 70)

    print(
        "AMHARIC NLP HIERARCHICAL PREDICTION SYSTEM"
    )

    print("=" * 70)

    print()

    print(
        "The system predicts:"
    )

    print()

    print(
        "1. FINAL MAIN TOPIC"
    )

    print(
        "2. FINAL SUBTOPIC"
    )

    print()

    print(
        "Commands:"
    )

    print(
        "exit"
    )

    print(
        "quit"
    )

    print()

    while True:

        try:

            text = input(
                "Enter Amharic text: "
            )

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            print("Exiting system...")
            break

        text = clean_text(text)

        if not text:

            print(
                "Please enter some text."
            )

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

        try:

            result = predict_hierarchical(

                text,

                domain_components,

                subtopic_bundle,

                semantic_model,

                semantic_index

            )

            print_prediction(
                result
            )

        except Exception as error:

            print()

            print(
                "PREDICTION ERROR"
            )

            print()

            print(
                type(error).__name__
            )

            print(
                str(error)
            )

            print()


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():

    print()

    print("=" * 70)

    print(
        SCRIPT_NAME
    )

    print("=" * 70)

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    # -------------------------------------------------------------------------
    # VALIDATE MODELS
    # -------------------------------------------------------------------------

    validate_model_paths()

    # -------------------------------------------------------------------------
    # LOAD EXISTING MODELS
    # -------------------------------------------------------------------------

    domain_bundle, subtopic_bundle = load_models()

    # -------------------------------------------------------------------------
    # EXTRACT DOMAIN COMPONENTS
    # -------------------------------------------------------------------------

    domain_components = extract_domain_components(
        domain_bundle
    )

    print()

    print(
        f"Number of main topics: "
        f"{len(domain_components['domains'])}"
    )

    # -------------------------------------------------------------------------
    # LOAD SEMANTIC MODEL
    # -------------------------------------------------------------------------

    semantic_model = load_semantic_model()

    # -------------------------------------------------------------------------
    # LOAD OR BUILD SEMANTIC INDEX
    # -------------------------------------------------------------------------

    semantic_index = load_or_build_index(

        semantic_model,

        domain_components[
            "domains"
        ]

    )

    # -------------------------------------------------------------------------
    # START INTERACTIVE SYSTEM
    # -------------------------------------------------------------------------

    interactive_inference(

        domain_components,

        subtopic_bundle,

        semantic_model,

        semantic_index

    )


# =============================================================================
# RUN SCRIPT
# =============================================================================


if __name__ == "__main__":

    main()