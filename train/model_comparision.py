import os
import re
import json
import time
import pickle
import warnings

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# MEMORY / CPU SETTINGS
# ----------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.decomposition import (
    LatentDirichletAllocation
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_recall_fscore_support
)

from scipy.sparse import hstack


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

MAX_WORD_FEATURES = 15000
MAX_CHAR_FEATURES = 10000

LDA_TOPICS = 10

TOP_K_VALUES = [3, 5, 7, 10]

OUTPUT_DIR = Path("models/comparison")

DATA_DIR = Path("data/train")

STOPWORDS_FILE = Path(
    "data/raw/amharic_stopwords.txt"
)


# ======================================================================
# TEXT UTILITIES
# ======================================================================

def normalize_text(text):
    """
    Basic text normalization.

    IMPORTANT:
    This function DOES NOT perform:
        - lemmatization
        - stemming
        - morphological analysis

    Original word forms remain intact.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def tokenize_amharic(text):
    """
    Extract tokens.

    Keeps:
        - Amharic words
        - English words
        - numbers

    No stemming or lemmatization.
    """

    text = normalize_text(text)

    tokens = re.findall(

        r"[\u1200-\u137F]+|[A-Za-z]+|\d+",

        text
    )

    return tokens


def clean_for_model(text, stopwords):
    """
    Remove stopwords while preserving
    original word forms.
    """

    tokens = tokenize_amharic(text)

    stopword_set = set(stopwords)

    cleaned = []

    for token in tokens:

        token_lower = token.lower()

        if token_lower in stopword_set:
            continue

        if len(token) < 2:
            continue

        cleaned.append(token)

    return " ".join(cleaned)


# ======================================================================
# DATA LOADING
# ======================================================================

def load_stopwords():

    if not STOPWORDS_FILE.exists():

        print(
            "\n⚠️ Stopword file not found:"
        )

        print(
            STOPWORDS_FILE
        )

        return []


    stopwords = []


    with open(

        STOPWORDS_FILE,

        "r",

        encoding="utf-8"

    ) as file:

        for line in file:

            word = line.strip()

            if word:

                stopwords.append(
                    word
                )


    return stopwords


def find_file(name):

    possible_paths = [

        DATA_DIR / name,

        Path("data") / name,

        Path(name)

    ]


    for path in possible_paths:

        if path.exists():

            return path


    raise FileNotFoundError(

        f"\nCould not find file: {name}\n"

        f"Checked:\n"

        + "\n".join(
            str(p)
            for p in possible_paths
        )

    )


def load_excel_file(name):

    path = find_file(name)

    print(
        f"Loading: {path.name}"
    )

    df = pd.read_excel(path)

    return df


def detect_text_column(df):

    candidates = [

        "processed_text",

        "text",

        "clean_text",

        "content",

        "document",

        "sentence"

    ]


    for column in candidates:

        if column in df.columns:

            return column


    raise ValueError(

        "\nCould not detect text column.\n"

        f"Available columns:\n"

        f"{list(df.columns)}"

    )


def detect_keyword_column(df):

    candidates = [

        "keywords",

        "keyword",

        "keyphrases",

        "keyphrase",

        "gold_keywords",

        "labels"

    ]


    for column in candidates:

        if column in df.columns:

            return column


    raise ValueError(

        "\nCould not detect keyword column.\n"

        f"Available columns:\n"

        f"{list(df.columns)}"

    )


def parse_keywords(value):

    """
    Convert keyword column into list.

    Supports:

        word1|word2|word3

        word1,word2,word3

        ['word1', 'word2']

    """

    if pd.isna(value):

        return []


    value = str(value).strip()


    if not value:

        return []


    # Try Python-like list

    if value.startswith("[") and value.endswith("]"):

        try:

            import ast

            result = ast.literal_eval(value)

            if isinstance(result, list):

                return [

                    normalize_text(x)

                    for x in result

                    if normalize_text(x)

                ]

        except Exception:

            pass


    # Pipe separated

    if "|" in value:

        return [

            normalize_text(x)

            for x in value.split("|")

            if normalize_text(x)

        ]


    # Comma separated

    if "," in value:

        return [

            normalize_text(x)

            for x in value.split(",")

            if normalize_text(x)

        ]


    return [value]


def load_data():

    train_df = load_excel_file(
        "train_processed.xlsx"
    )

    val_df = load_excel_file(
        "validation_processed.xlsx"
    )

    test_df = load_excel_file(
        "test_processed.xlsx"
    )


    train_text_col = detect_text_column(
        train_df
    )

    val_text_col = detect_text_column(
        val_df
    )

    test_text_col = detect_text_column(
        test_df
    )


    train_keyword_col = detect_keyword_column(
        train_df
    )

    val_keyword_col = detect_keyword_column(
        val_df
    )

    test_keyword_col = detect_keyword_column(
        test_df
    )


    train_texts = [

        normalize_text(x)

        for x in train_df[
            train_text_col
        ].tolist()

    ]


    val_texts = [

        normalize_text(x)

        for x in val_df[
            val_text_col
        ].tolist()

    ]


    test_texts = [

        normalize_text(x)

        for x in test_df[
            test_text_col
        ].tolist()

    ]


    train_keywords = [

        parse_keywords(x)

        for x in train_df[
            train_keyword_col
        ].tolist()

    ]


    val_keywords = [

        parse_keywords(x)

        for x in val_df[
            val_keyword_col
        ].tolist()

    ]


    test_keywords = [

        parse_keywords(x)

        for x in test_df[
            test_keyword_col
        ].tolist()

    ]


    return {

        "train_texts":
        train_texts,

        "val_texts":
        val_texts,

        "test_texts":
        test_texts,

        "train_keywords":
        train_keywords,

        "val_keywords":
        val_keywords,

        "test_keywords":
        test_keywords,

        "columns": {

            "train_text":
            train_text_col,

            "train_keywords":
            train_keyword_col

        }

    }


# ======================================================================
# KEYWORD NORMALIZATION FOR EVALUATION
# ======================================================================

def normalize_keyword(keyword):

    keyword = normalize_text(keyword)

    keyword = keyword.lower()

    return keyword


def keyword_match(predicted, gold):

    """
    Exact keyword matching.

    Case insensitive.

    Does NOT perform:
        - stemming
        - lemmatization
        - fuzzy matching
    """

    predicted = normalize_keyword(
        predicted
    )

    gold = normalize_keyword(
        gold
    )

    return predicted == gold


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_predictions(
    predictions,
    gold_keywords,
    k
):

    precisions = []

    recalls = []

    f1_scores = []

    average_precisions = []


    for predicted, gold in zip(

        predictions,

        gold_keywords

    ):

        predicted = predicted[:k]


        gold_normalized = [

            normalize_keyword(x)

            for x in gold

        ]


        predicted_normalized = [

            normalize_keyword(x)

            for x in predicted

        ]


        gold_set = set(
            gold_normalized
        )


        # Remove duplicates while preserving order

        unique_predicted = []

        seen = set()


        for keyword in predicted_normalized:

            if keyword not in seen:

                unique_predicted.append(
                    keyword
                )

                seen.add(keyword)


        predicted_set = set(
            unique_predicted
        )


        correct = len(

            predicted_set.intersection(
                gold_set
            )

        )


        # Precision

        if len(unique_predicted) > 0:

            precision = (

                correct /

                len(unique_predicted)

            )

        else:

            precision = 0.0


        # Recall

        if len(gold_set) > 0:

            recall = (

                correct /

                len(gold_set)

            )

        else:

            recall = 0.0


        # F1

        if precision + recall > 0:

            f1 = (

                2 *

                precision *

                recall

                /

                (

                    precision +

                    recall

                )

            )

        else:

            f1 = 0.0


        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )


        # ----------------------------------------------------------
        # Average Precision
        # ----------------------------------------------------------

        hits = 0

        precision_sum = 0.0


        for rank, keyword in enumerate(

            unique_predicted,

            start=1

        ):

            if keyword in gold_set:

                hits += 1

                precision_at_rank = (

                    hits /

                    rank

                )

                precision_sum += (

                    precision_at_rank

                )


        if len(gold_set) > 0:

            denominator = min(

                len(gold_set),

                k

            )


            if denominator > 0:

                average_precision = (

                    precision_sum /

                    denominator

                )

            else:

                average_precision = 0.0

        else:

            average_precision = 0.0


        average_precisions.append(

            average_precision

        )


    return {

        "precision":

        float(np.mean(precisions)),

        "recall":

        float(np.mean(recalls)),

        "f1":

        float(np.mean(f1_scores)),

        "map":

        float(np.mean(
            average_precisions
        ))

    }


# ======================================================================
# CANDIDATE GENERATION
# ======================================================================

def generate_candidates(
    text,
    stopwords,
    max_ngram=3
):

    """
    Generate extractive keyword candidates.

    No morphological analysis.

    Candidate examples:

        ሚራንዳ
        ኬሬ
        ሚራንዳ ኬሬ

    This allows the model to learn
    multi-word names and phrases.
    """

    tokens = tokenize_amharic(text)

    stopword_set = set(

        x.lower()

        for x in stopwords

    )


    candidates = []


    # --------------------------------------------------------------
    # Unigrams
    # --------------------------------------------------------------

    for token in tokens:

        if token.lower() in stopword_set:

            continue

        if len(token) < 2:

            continue

        candidates.append(
            token
        )


    # --------------------------------------------------------------
    # N-GRAMS
    # --------------------------------------------------------------

    for n in range(

        2,

        max_ngram + 1

    ):

        for i in range(

            len(tokens) - n + 1

        ):

            phrase_tokens = tokens[
                i:i + n
            ]


            if all(

                token.lower()
                not in stopword_set

                for token in phrase_tokens

            ):

                phrase = " ".join(

                    phrase_tokens

                )


                candidates.append(

                    phrase

                )


    # --------------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------------

    unique_candidates = []

    seen = set()


    for candidate in candidates:

        normalized = normalize_keyword(
            candidate
        )


        if normalized not in seen:

            unique_candidates.append(
                candidate
            )

            seen.add(
                normalized
            )


    return unique_candidates


# ======================================================================
# FEATURE EXTRACTION FOR SUPERVISED MODEL
# ======================================================================

def candidate_features(
    text,
    candidate,
    word_vectorizer,
    char_vectorizer
):

    """
    Build lightweight contextual features.

    Features:

    1. Word TF-IDF score
    2. Character TF-IDF similarity
    3. Candidate frequency
    4. Candidate position
    5. Candidate length
    6. Candidate word count

    This is not morphological analysis.

    It uses document/context statistics.
    """

    text_tokens = tokenize_amharic(
        text
    )


    candidate_tokens = tokenize_amharic(
        candidate
    )


    if not candidate_tokens:

        return None


    # --------------------------------------------------------------
    # Position
    # --------------------------------------------------------------

    try:

        candidate_position = (

            text.find(candidate)

            /

            max(len(text), 1)

        )

    except Exception:

        candidate_position = 1.0


    # --------------------------------------------------------------
    # Frequency
    # --------------------------------------------------------------

    candidate_count = text.count(
        candidate
    )


    frequency = (

        candidate_count

        /

        max(len(text_tokens), 1)

    )


    # --------------------------------------------------------------
    # Length
    # --------------------------------------------------------------

    character_length = (

        len(candidate)

        /

        50.0

    )


    word_count = (

        len(candidate_tokens)

        /

        3.0

    )


    # --------------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------------

    candidate_vector = word_vectorizer.transform(

        [candidate]

    )


    text_vector = word_vectorizer.transform(

        [text]

    )


    word_score = (

        text_vector.multiply(
            candidate_vector
        ).sum()

    )


    # --------------------------------------------------------------
    # Character representation
    # --------------------------------------------------------------

    candidate_char_vector = (

        char_vectorizer.transform(
            [candidate]
        )

    )


    text_char_vector = (

        char_vectorizer.transform(
            [text]
        )

    )


    char_score = (

        text_char_vector.multiply(
            candidate_char_vector
        ).sum()

    )


    return [

        float(word_score),

        float(char_score),

        float(frequency),

        float(candidate_position),

        float(character_length),

        float(word_count)

    ]


# ======================================================================
# SUPERVISED KEYWORD RANKER
# ======================================================================

def run_supervised_ranker(

    train_texts,
    train_keywords,

    val_texts,
    val_keywords,

    test_texts,
    test_keywords,

    stopwords,

    output_dir

):

    print("\n" + "=" * 70)

    print(
        "🤖 MODEL 1 - SUPERVISED CONTEXTUAL KEYWORD RANKER"
    )

    print("=" * 70)


    start_time = time.time()


    # ==============================================================
    # TF-IDF REPRESENTATIONS
    # ==============================================================

    print(
        "\nTraining word TF-IDF..."
    )


    word_vectorizer = TfidfVectorizer(

        max_features=

        MAX_WORD_FEATURES,

        token_pattern=

        r"[\u1200-\u137F]+|[A-Za-z]+|\d+",

        ngram_range=(1, 3),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True

    )


    word_vectorizer.fit(

        train_texts

    )


    print(

        "Word vocabulary:",

        len(

            word_vectorizer.get_feature_names_out()

        )

    )


    print(
        "\nTraining character TF-IDF..."
    )


    char_vectorizer = TfidfVectorizer(

        analyzer="char_wb",

        ngram_range=(3, 5),

        max_features=

        MAX_CHAR_FEATURES,

        min_df=2,

        sublinear_tf=True

    )


    char_vectorizer.fit(

        train_texts

    )


    print(

        "Character vocabulary:",

        len(

            char_vectorizer.get_feature_names_out()

        )

    )


    # ==============================================================
    # BUILD TRAINING DATA
    # ==============================================================

    print(
        "\nBuilding supervised training examples..."
    )


    X = []

    y = []


    max_negatives_per_document = 15


    for index, (

        text,

        gold

    ) in enumerate(

        zip(

            train_texts,

            train_keywords

        )

    ):


        candidates = generate_candidates(

            text,

            stopwords,

            max_ngram=3

        )


        gold_set = set(

            normalize_keyword(x)

            for x in gold

        )


        positives = []

        negatives = []


        for candidate in candidates:


            normalized = normalize_keyword(

                candidate

            )


            if normalized in gold_set:

                positives.append(

                    candidate

                )

            else:

                negatives.append(

                    candidate

                )


        # ----------------------------------------------------------
        # Limit negatives for balanced training
        # ----------------------------------------------------------

        if len(negatives) > (

            max_negatives_per_document

        ):

            np.random.seed(

                RANDOM_STATE + index

            )


            selected_indices = np.random.choice(

                len(negatives),

                max_negatives_per_document,

                replace=False

            )


            negatives = [

                negatives[i]

                for i in selected_indices

            ]


        selected_candidates = (

            positives +

            negatives

        )


        for candidate in selected_candidates:


            features = candidate_features(

                text,

                candidate,

                word_vectorizer,

                char_vectorizer

            )


            if features is None:

                continue


            X.append(

                features

            )


            label = int(

                normalize_keyword(
                    candidate
                )

                in

                gold_set

            )


            y.append(

                label

            )


        if (

            index + 1

        ) % 2000 == 0:


            print(

                f"   Processed "

                f"{index + 1:,}/"

                f"{len(train_texts):,}"

            )


    X = np.array(

        X,

        dtype=np.float32

    )


    y = np.array(

        y,

        dtype=np.int32

    )


    print(

        f"\nTraining examples: "

        f"{len(X):,}"

    )


    print(

        f"Positive examples: "

        f"{int(np.sum(y == 1)):,}"

    )


    print(

        f"Negative examples: "

        f"{int(np.sum(y == 0)):,}"

    )


    # ==============================================================
    # TRAIN MODEL
    # ==============================================================

    print(
        "\nTraining Logistic Regression..."
    )


    model = LogisticRegression(

        max_iter=300,

        class_weight="balanced",

        solver="liblinear",

        random_state=RANDOM_STATE

    )


    model.fit(

        X,

        y

    )


    print(
        "✓ Training completed"
    )


    # ==============================================================
    # PREDICTION FUNCTION
    # ==============================================================

    def predict_documents(

        texts,

        top_k

    ):


        predictions = []


        for index, text in enumerate(

            texts

        ):


            candidates = generate_candidates(

                text,

                stopwords,

                max_ngram=3

            )


            candidate_scores = []


            for candidate in candidates:


                features = candidate_features(

                    text,

                    candidate,

                    word_vectorizer,

                    char_vectorizer

                )


                if features is None:

                    continue


                probability = (

                    model.predict_proba(

                        np.array(

                            [features]

                        )

                    )[0][1]

                )


                candidate_scores.append(

                    (

                        candidate,

                        probability

                    )

                )


            candidate_scores.sort(

                key=lambda x: x[1],

                reverse=True

            )


            selected = []


            for candidate, score in (

                candidate_scores

            ):


                # --------------------------------------------------
                # Avoid redundant phrases
                # --------------------------------------------------

                redundant = False


                for existing in selected:


                    if (

                        normalize_keyword(candidate)

                        ==

                        normalize_keyword(existing)

                    ):

                        redundant = True

                        break


                if not redundant:

                    selected.append(

                        candidate

                    )


                if len(selected) >= top_k:

                    break


            predictions.append(

                selected

            )


            if (

                index + 1

            ) % 1000 == 0:


                print(

                    f"   Predicted "

                    f"{index + 1:,}/"

                    f"{len(texts):,}"

                )


        return predictions


    # ==============================================================
    # VALIDATION
    # ==============================================================

    validation_results = {}


    print("\n" + "-" * 70)

    print(
        "VALIDATION RESULTS"
    )

    print("-" * 70)


    for k in TOP_K_VALUES:


        print(

            f"\nTesting Top-{k}..."

        )


        predictions = predict_documents(

            val_texts,

            k

        )


        metrics = evaluate_predictions(

            predictions,

            val_keywords,

            k

        )


        validation_results[

            str(k)

        ] = metrics


        print(

            f"Precision@{k}: "

            f"{metrics['precision']:.4f}"

        )


        print(

            f"Recall@{k}:    "

            f"{metrics['recall']:.4f}"

        )


        print(

            f"F1@{k}:        "

            f"{metrics['f1']:.4f}"

        )


        print(

            f"mAP@{k}:       "

            f"{metrics['map']:.4f}"

        )


    # ==============================================================
    # SELECT BEST K BY MAP
    # ==============================================================

    best_k = max(

        TOP_K_VALUES,

        key=lambda k:

        validation_results[
            str(k)
        ]["map"]

    )


    print("\n" + "=" * 70)

    print(
        "🏆 BEST VALIDATION CONFIGURATION"
    )

    print("=" * 70)


    best_metrics = validation_results[
        str(best_k)
    ]


    print(
        f"\nBest Top-K: {best_k}"
    )


    print(

        f"Validation Precision: "

        f"{best_metrics['precision']:.4f}"

    )


    print(

        f"Validation Recall: "

        f"{best_metrics['recall']:.4f}"

    )


    print(

        f"Validation F1: "

        f"{best_metrics['f1']:.4f}"

    )


    print(

        f"Validation mAP: "

        f"{best_metrics['map']:.4f}"

    )


    # ==============================================================
    # TEST
    # ==============================================================

    print("\n" + "=" * 70)

    print(
        "🧪 FINAL TEST RESULTS"
    )

    print("=" * 70)


    test_predictions = predict_documents(

        test_texts,

        best_k

    )


    test_metrics = evaluate_predictions(

        test_predictions,

        test_keywords,

        best_k

    )


    print(

        f"\nTest Precision: "

        f"{test_metrics['precision']:.4f}"

    )


    print(

        f"Test Recall: "

        f"{test_metrics['recall']:.4f}"

    )


    print(

        f"Test F1: "

        f"{test_metrics['f1']:.4f}"

    )


    print(

        f"Test mAP: "

        f"{test_metrics['map']:.4f}"

    )


    # ==============================================================
    # SAVE
    # ==============================================================

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )


    results = {

        "model":

        "Supervised Contextual Keyword Ranker",

        "best_top_k":

        best_k,

        "validation_results":

        validation_results,

        "test_results":

        test_metrics,

        "training_time_seconds":

        time.time() - start_time

    }


    with open(

        output_dir /

        "supervised_keyword_results.json",

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            results,

            file,

            ensure_ascii=False,

            indent=4

        )


    # Save predictions

    prediction_rows = []


    for text, gold, predicted in zip(

        test_texts,

        test_keywords,

        test_predictions

    ):


        prediction_rows.append({

            "text":

            text,

            "gold_keywords":

            " | ".join(gold),

            "predicted_keywords":

            " | ".join(predicted)

        })


    predictions_df = pd.DataFrame(

        prediction_rows

    )


    predictions_df.to_excel(

        output_dir /

        "supervised_predictions.xlsx",

        index=False

    )


    # Save model

    model_data = {

        "model":

        model,

        "word_vectorizer":

        word_vectorizer,

        "char_vectorizer":

        char_vectorizer,

        "stopwords":

        stopwords,

        "best_top_k":

        best_k

    }


    with open(

        output_dir /

        "supervised_keyword_model.pkl",

        "wb"

    ) as file:


        pickle.dump(

            model_data,

            file

        )


    print(
        "\n💾 Supervised model saved."
    )


    return {

        "model":

        "Supervised Contextual Keyword Ranker",

        "validation":

        validation_results[
            str(best_k)
        ],

        "test":

        test_metrics,

        "best_k":

        best_k

    }


# ======================================================================
# MODEL 2 - AMHARIC ADAPTED LDA
# ======================================================================

def run_lda(

    train_texts,

    train_keywords,

    val_texts,

    val_keywords,

    test_texts,

    test_keywords,

    stopwords,

    output_dir

):

    print("\n" + "=" * 70)

    print(
        "🤖 MODEL 2 - AMHARIC-ADAPTED LDA"
    )

    print("=" * 70)


    start_time = time.time()


    # ==============================================================
    # CLEAN TEXT
    # ==============================================================

    print(
        "\nPreparing texts..."
    )


    train_clean = [

        clean_for_model(

            text,

            stopwords

        )

        for text in train_texts

    ]


    val_clean = [

        clean_for_model(

            text,

            stopwords

        )

        for text in val_texts

    ]


    test_clean = [

        clean_for_model(

            text,

            stopwords

        )

        for text in test_texts

    ]


    # ==============================================================
    # VECTORIZER
    # ==============================================================

    print(
        "\nBuilding Amharic vocabulary..."
    )


    vectorizer = CountVectorizer(

        token_pattern=

        r"[\u1200-\u137F]+|[A-Za-z]+|\d+",

        max_features=10000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95

    )


    train_matrix = vectorizer.fit_transform(

        train_clean

    )


    feature_names = (

        vectorizer.get_feature_names_out()

    )


    print(

        f"\nVocabulary size: "

        f"{len(feature_names):,}"

    )


    # ==============================================================
    # LDA
    # ==============================================================

    print(

        f"\nTraining LDA with "

        f"{LDA_TOPICS} topics..."

    )


    lda = LatentDirichletAllocation(

        n_components=LDA_TOPICS,

        max_iter=10,

        learning_method="batch",

        random_state=RANDOM_STATE,

        n_jobs=1,

        verbose=0

    )


    lda.fit(

        train_matrix

    )


    print(
        "\n✓ LDA training completed"
    )


    # ==============================================================
    # TOPIC WORDS
    # ==============================================================

    topic_keywords = {}


    for topic_id, weights in enumerate(

        lda.components_

    ):


        indices = np.argsort(

            weights

        )[::-1][:30]


        topic_keywords[

            topic_id

        ] = [

            feature_names[i]

            for i in indices

        ]


    # ==============================================================
    # PREDICTION
    # ==============================================================

    def predict_lda(

        texts,

        top_k

    ):


        cleaned = [

            clean_for_model(

                text,

                stopwords

            )

            for text in texts

        ]


        matrix = vectorizer.transform(

            cleaned

        )


        topic_probabilities = lda.transform(

            matrix

        )


        predictions = []


        for text, probabilities in zip(

            cleaned,

            topic_probabilities

        ):


            text_lower = normalize_keyword(

                text

            )


            ranked_topics = np.argsort(

                probabilities

            )[::-1]


            selected = []


            for topic_id in ranked_topics:


                for keyword in topic_keywords[

                    topic_id

                ]:


                    # Extractive condition

                    if normalize_keyword(

                        keyword

                    ) in text_lower:


                        if keyword not in selected:

                            selected.append(

                                keyword

                            )


                    if len(selected) >= top_k:

                        break


                if len(selected) >= top_k:

                    break


            predictions.append(

                selected

            )


        return predictions


    # ==============================================================
    # VALIDATION
    # ==============================================================

    validation_results = {}


    print(
        "\nPredicting validation keywords..."
    )


    for k in TOP_K_VALUES:


        predictions = predict_lda(

            val_texts,

            k

        )


        metrics = evaluate_predictions(

            predictions,

            val_keywords,

            k

        )


        validation_results[

            str(k)

        ] = metrics


        print(f"\nTop-{k}")

        print(

            f"Precision@{k}: "

            f"{metrics['precision']:.4f}"

        )

        print(

            f"Recall@{k}:    "

            f"{metrics['recall']:.4f}"

        )

        print(

            f"F1@{k}:        "

            f"{metrics['f1']:.4f}"

        )

        print(

            f"mAP@{k}:       "

            f"{metrics['map']:.4f}"

        )


    best_k = max(

        TOP_K_VALUES,

        key=lambda k:

        validation_results[
            str(k)
        ]["map"]

    )


    # ==============================================================
    # TEST
    # ==============================================================

    print(
        "\nPredicting test keywords..."
    )


    test_predictions = predict_lda(

        test_texts,

        best_k

    )


    test_metrics = evaluate_predictions(

        test_predictions,

        test_keywords,

        best_k

    )


    print("\n" + "=" * 70)

    print(
        "🧪 LDA TEST RESULTS"
    )

    print("=" * 70)


    print(

        f"\nBest Top-K: "

        f"{best_k}"

    )


    print(

        f"Precision: "

        f"{test_metrics['precision']:.4f}"

    )


    print(

        f"Recall:    "

        f"{test_metrics['recall']:.4f}"

    )


    print(

        f"F1:        "

        f"{test_metrics['f1']:.4f}"

    )


    print(

        f"mAP:       "

        f"{test_metrics['map']:.4f}"

    )


    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    results = {

        "model":

        "Amharic-Adapted LDA",

        "topics":

        LDA_TOPICS,

        "best_top_k":

        best_k,

        "validation_results":

        validation_results,

        "test_results":

        test_metrics,

        "training_time_seconds":

        time.time() - start_time

    }


    with open(

        output_dir /

        "amharic_lda_results.json",

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            results,

            file,

            ensure_ascii=False,

            indent=4

        )


    # --------------------------------------------------------------
    # SAVE MODEL
    #
    # IMPORTANT:
    # No local tokenizer function is stored.
    # Therefore pickle will NOT fail.
    # --------------------------------------------------------------

    with open(

        output_dir /

        "amharic_lda_model.pkl",

        "wb"

    ) as file:


        pickle.dump(

            {

                "vectorizer":

                vectorizer,

                "lda":

                lda,

                "topic_keywords":

                topic_keywords,

                "stopwords":

                stopwords

            },

            file

        )


    print(
        "\n💾 LDA results and model saved."
    )


    return {

        "model":

        "Amharic-Adapted LDA",

        "validation":

        validation_results[
            str(best_k)
        ],

        "test":

        test_metrics,

        "best_k":

        best_k

    }


# ======================================================================
# MODEL 3 - KEYBERT
# ======================================================================

def run_keybert(

    val_texts,

    val_keywords,

    test_texts,

    test_keywords,

    stopwords,

    output_dir

):

    print("\n" + "=" * 70)

    print(
        "🤖 MODEL 3 - MULTILINGUAL KEYBERT"
    )

    print("=" * 70)


    try:

        from keybert import KeyBERT


    except ImportError:

        print(

            "\n⚠️ KeyBERT is not installed."

        )

        print(

            "Skipping KeyBERT."

        )

        return None


    try:


        print(

            "\nLoading KeyBERT embedding model..."

        )


        # This may fail on low memory machines.

        model = KeyBERT(

            model=

            "paraphrase-multilingual-MiniLM-L12-v2"

        )


        print(
            "✓ KeyBERT loaded"
        )


        # ----------------------------------------------------------
        # LIMIT DATA FOR MEMORY SAFETY
        # ----------------------------------------------------------

        # Remove this limit later if your computer
        # has sufficient RAM.

        MAX_EVALUATION_DOCUMENTS = 1000


        val_subset_texts = val_texts[

            :MAX_EVALUATION_DOCUMENTS

        ]


        val_subset_keywords = val_keywords[

            :MAX_EVALUATION_DOCUMENTS

        ]


        print(

            f"\nEvaluating "

            f"{len(val_subset_texts):,} "

            f"validation documents..."

        )


        predictions = []


        for index, text in enumerate(

            val_subset_texts

        ):


            keywords = model.extract_keywords(

                text,

                keyphrase_ngram_range=(1, 3),

                stop_words=stopwords,

                top_n=10

            )


            extracted = [

                keyword

                for keyword, score

                in keywords

            ]


            predictions.append(

                extracted

            )


            if (

                index + 1

            ) % 100 == 0:


                print(

                    f"   Processed "

                    f"{index + 1:,}/"

                    f"{len(val_subset_texts):,}"

                )


        validation_results = {}


        for k in TOP_K_VALUES:


            metrics = evaluate_predictions(

                predictions,

                val_subset_keywords,

                k

            )


            validation_results[

                str(k)

            ] = metrics


            print(f"\nTop-{k}")

            print(

                f"Precision: "

                f"{metrics['precision']:.4f}"

            )

            print(

                f"Recall: "

                f"{metrics['recall']:.4f}"

            )

            print(

                f"F1: "

                f"{metrics['f1']:.4f}"

            )

            print(

                f"mAP: "

                f"{metrics['map']:.4f}"

            )


        best_k = max(

            TOP_K_VALUES,

            key=lambda k:

            validation_results[
                str(k)
            ]["map"]

        )


        # ----------------------------------------------------------
        # TEST
        # ----------------------------------------------------------

        test_subset_texts = test_texts[

            :MAX_EVALUATION_DOCUMENTS

        ]


        test_subset_keywords = test_keywords[

            :MAX_EVALUATION_DOCUMENTS

        ]


        print(

            f"\nEvaluating "

            f"{len(test_subset_texts):,} "

            f"test documents..."

        )


        test_predictions = []


        for text in test_subset_texts:


            keywords = model.extract_keywords(

                text,

                keyphrase_ngram_range=(1, 3),

                stop_words=stopwords,

                top_n=10

            )


            extracted = [

                keyword

                for keyword, score

                in keywords

            ]


            test_predictions.append(

                extracted

            )


        test_metrics = evaluate_predictions(

            test_predictions,

            test_subset_keywords,

            best_k

        )


        return {

            "model":

            "Multilingual KeyBERT",

            "validation":

            validation_results[
                str(best_k)
            ],

            "test":

            test_metrics,

            "best_k":

            best_k

        }


    except MemoryError:


        print(

            "\n⚠️ KeyBERT skipped due to insufficient memory."

        )


        return None


    except Exception as error:


        print(

            "\n⚠️ KeyBERT failed safely:"

        )

        print(error)


        return None


# ======================================================================
# BERTOPIC
# ======================================================================

def run_bertopic():

    """
    BERTopic is intentionally optional.

    Your previous system produced:

        llvmlite.dll errors
        MemoryError
        Windows paging file errors

    Therefore this script does not automatically
    execute BERTopic.

    BERTopic should be tested separately on a
    machine with sufficient RAM.
    """

    print("\n" + "=" * 70)

    print(
        "🤖 MODEL 4 - BERTOPIC"
    )

    print("=" * 70)


    print(

        "\n⚠️ BERTopic is disabled "

        "for this Windows low-memory run."

    )


    print(

        "Reason: previous environment showed "

        "BERTopic / llvmlite / paging-file failures."

    )


    print(

        "\nThis is NOT considered a crash."

    )


    print(

        "It can be enabled later on a "

        "higher-memory environment."

    )


    return None


# ======================================================================
# SAVE COMPARISON
# ======================================================================

def save_comparison(

    model_results,

    output_dir

):

    rows = []


    for result in model_results:


        if result is None:

            continue


        row = {

            "model":

            result["model"],

            "best_top_k":

            result["best_k"],

            "validation_precision":

            result["validation"]["precision"],

            "validation_recall":

            result["validation"]["recall"],

            "validation_f1":

            result["validation"]["f1"],

            "validation_map":

            result["validation"]["map"],

            "test_precision":

            result["test"]["precision"],

            "test_recall":

            result["test"]["recall"],

            "test_f1":

            result["test"]["f1"],

            "test_map":

            result["test"]["map"]

        }


        rows.append(

            row

        )


    if not rows:

        print(
            "\n⚠️ No model results available."
        )

        return


    results_df = pd.DataFrame(

        rows

    )


    results_df = results_df.sort_values(

        "validation_map",

        ascending=False

    )


    results_df.to_csv(

        output_dir /

        "model_comparison.csv",

        index=False,

        encoding="utf-8-sig"

    )


    results_df.to_excel(

        output_dir /

        "model_comparison.xlsx",

        index=False

    )


    print("\n" + "=" * 70)

    print(
        "🏆 MODEL COMPARISON SUMMARY"
    )

    print("=" * 70)


    for _, row in results_df.iterrows():


        print(

            f"\n{row['model']}"

        )


        print(

            f"Best Top-K: "

            f"{row['best_top_k']}"

        )


        print(

            f"Validation F1: "

            f"{row['validation_f1']:.4f}"

        )


        print(

            f"Validation mAP: "

            f"{row['validation_map']:.4f}"

        )


        print(

            f"Test F1: "

            f"{row['test_f1']:.4f}"

        )


        print(

            f"Test mAP: "

            f"{row['test_map']:.4f}"

        )


    best_model = results_df.iloc[0]


    print("\n" + "=" * 70)

    print(
        "🎯 RECOMMENDED MODEL"
    )

    print("=" * 70)


    print(

        f"\nModel: "

        f"{best_model['model']}"

    )


    print(

        f"Validation mAP: "

        f"{best_model['validation_map']:.4f}"

    )


    print(

        f"Test mAP: "

        f"{best_model['test_map']:.4f}"

    )


    print(

        f"\n💾 Comparison saved to:"

    )

    print(

        output_dir.resolve()

    )


# ======================================================================
# MAIN
# ======================================================================

def compare_models():

    print("\n" + "=" * 70)

    print(
        "📊 PHASE 4 - AMHARIC KEYWORD & TOPIC MODEL COMPARISON"
    )

    print("=" * 70)


    # ==============================================================
    # OUTPUT DIRECTORY
    # ==============================================================

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ==============================================================
    # STOPWORDS
    # ==============================================================

    stopwords = load_stopwords()


    print(

        f"\n📝 Stopwords loaded: "

        f"{len(stopwords):,}"

    )


    # ==============================================================
    # LOAD DATA
    # ==============================================================

    data = load_data()


    train_texts = data[

        "train_texts"

    ]


    val_texts = data[

        "val_texts"

    ]


    test_texts = data[

        "test_texts"

    ]


    train_keywords = data[

        "train_keywords"

    ]


    val_keywords = data[

        "val_keywords"

    ]


    test_keywords = data[

        "test_keywords"

    ]


    print("\n" + "=" * 70)

    print(
        "📊 DATA SPLITS"
    )

    print("=" * 70)


    print(

        f"\nTrain:      "

        f"{len(train_texts):,}"

    )


    print(

        f"Validation: "

        f"{len(val_texts):,}"

    )


    print(

        f"Test:       "

        f"{len(test_texts):,}"

    )


    print("\nDetected columns:")


    print(

        f"\nTrain text: "

        f"{data['columns']['train_text']}"

    )


    print(

        f"Train keywords: "

        f"{data['columns']['train_keywords']}"

    )


    # ==============================================================
    # RUN MODELS
    # ==============================================================

    model_results = []


    # --------------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------------

    try:


        supervised_result = run_supervised_ranker(

            train_texts,

            train_keywords,

            val_texts,

            val_keywords,

            test_texts,

            test_keywords,

            stopwords,

            OUTPUT_DIR

        )


        model_results.append(

            supervised_result

        )


    except Exception as error:


        print(

            "\n⚠️ Supervised model failed:"

        )


        print(error)


    # --------------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------------

    try:


        lda_result = run_lda(

            train_texts,

            train_keywords,

            val_texts,

            val_keywords,

            test_texts,

            test_keywords,

            stopwords,

            OUTPUT_DIR

        )


        model_results.append(

            lda_result

        )


    except Exception as error:


        print(

            "\n⚠️ LDA failed safely:"

        )


        print(error)


    # --------------------------------------------------------------
    # MODEL 3
    # --------------------------------------------------------------

    keybert_result = run_keybert(

        val_texts,

        val_keywords,

        test_texts,

        test_keywords,

        stopwords,

        OUTPUT_DIR

    )


    if keybert_result is not None:

        model_results.append(

            keybert_result

        )


    # --------------------------------------------------------------
    # MODEL 4
    # --------------------------------------------------------------

    bertopic_result = run_bertopic()


    if bertopic_result is not None:

        model_results.append(

            bertopic_result

        )


    # ==============================================================
    # SAVE COMPARISON
    # ==============================================================

    save_comparison(

        model_results,

        OUTPUT_DIR

    )


    print("\n" + "=" * 70)

    print(
        "✅ PHASE 4 COMPLETE"
    )

    print("=" * 70)


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":

    compare_models()