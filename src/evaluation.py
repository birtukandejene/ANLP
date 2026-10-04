import re
import numpy as np


PUNCTUATION = ".,!?;:።፣፤፥፦፧፨\"'“”‘’()[]{}"


def parse_annotations(value):
    """
    Converts:
        'keyword1 | keyword2 | keyword3'

    Into:
        ['keyword1', 'keyword2', 'keyword3']
    """

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    items = str(value).split("|")

    return [
        item.strip()
        for item in items
        if item.strip()
    ]


def normalize_keyword(keyword):
    """
    STRICT normalization.

    This does NOT perform:
    - stemming
    - lemmatization
    - morphological normalization

    We preserve the Amharic surface form because
    the official metric is exact keyword matching.
    """

    if keyword is None:
        return ""

    keyword = str(keyword).strip()

    # Normalize repeated whitespace
    keyword = re.sub(r"\s+", " ", keyword)

    # Remove only surrounding punctuation
    keyword = keyword.strip(PUNCTUATION)

    return keyword


def unique_ranked_predictions(predicted, k=5):

    unique_predictions = []

    seen = set()

    for keyword in predicted:

        keyword = normalize_keyword(keyword)

        if not keyword:
            continue

        if keyword not in seen:

            seen.add(keyword)

            unique_predictions.append(keyword)

        if len(unique_predictions) >= k:
            break

    return unique_predictions


def average_precision_at_k(predicted, gold, k=5):
    """
    Average Precision@K.

    AP@K rewards:
    - correct predictions
    - correct ranking

    Higher-ranked correct keywords receive
    more credit.
    """

    predicted = unique_ranked_predictions(
        predicted,
        k
    )

    gold_set = {
        normalize_keyword(x)
        for x in gold
        if normalize_keyword(x)
    }

    if not gold_set:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, prediction in enumerate(
        predicted,
        start=1
    ):

        if prediction in gold_set:

            hits += 1

            precision = hits / rank

            precision_sum += precision

    denominator = min(
        len(gold_set),
        k
    )

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def mean_average_precision(
    predictions,
    gold_labels,
    k=5
):

    scores = []

    for predicted, gold in zip(
        predictions,
        gold_labels
    ):

        score = average_precision_at_k(
            predicted,
            gold,
            k
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def precision_at_k(predicted, gold, k=5):

    predicted = unique_ranked_predictions(
        predicted,
        k
    )

    gold_set = {
        normalize_keyword(x)
        for x in gold
        if normalize_keyword(x)
    }

    if not predicted:
        return 0.0

    correct = sum(
        1
        for prediction in predicted
        if prediction in gold_set
    )

    return correct / len(predicted)


def recall_at_k(predicted, gold, k=5):

    predicted = unique_ranked_predictions(
        predicted,
        k
    )

    gold_set = {
        normalize_keyword(x)
        for x in gold
        if normalize_keyword(x)
    }

    if not gold_set:
        return 0.0

    correct = sum(
        1
        for prediction in predicted
        if prediction in gold_set
    )

    return correct / len(gold_set)


def f1_at_k(predicted, gold, k=5):

    precision = precision_at_k(
        predicted,
        gold,
        k
    )

    recall = recall_at_k(
        predicted,
        gold,
        k
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def evaluate_predictions(
    predictions,
    gold_labels,
    k=5
):

    map_score = mean_average_precision(
        predictions,
        gold_labels,
        k
    )

    precision_scores = []

    recall_scores = []

    f1_scores = []

    for predicted, gold in zip(
        predictions,
        gold_labels
    ):

        precision_scores.append(
            precision_at_k(
                predicted,
                gold,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                predicted,
                gold,
                k
            )
        )

        f1_scores.append(
            f1_at_k(
                predicted,
                gold,
                k
            )
        )

    return {

        "mAP@5":
            float(map_score),

        "Precision@5":
            float(np.mean(
                precision_scores
            )),

        "Recall@5":
            float(np.mean(
                recall_scores
            )),

        "F1@5":
            float(np.mean(
                f1_scores
            ))
    }