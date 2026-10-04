import pandas as pd

from src.evaluation import parse_annotations


def load_dataset(path):

    df = pd.read_excel(path)

    print(f"Loaded: {path}")
    print(f"Shape: {df.shape}")

    required_columns = [
        "text",
        "keywords",
        "topics"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(

            f"Missing columns: "
            f"{missing_columns}"
        )

    # Remove rows without text

    df = df.dropna(
        subset=["text"]
    ).copy()

    # Convert text to string

    df["text"] = df[
        "text"
    ].astype(str)

    # Parse keyword annotations

    df["keyword_list"] = df[
        "keywords"
    ].apply(
        parse_annotations
    )

    # Parse topic annotations

    df["topic_list"] = df[
        "topics"
    ].apply(
        parse_annotations
    )

    # Word count

    df["word_count"] = df[
        "text"
    ].apply(
        lambda x: len(x.split())
    )

    # Text length group

    def get_length_group(count):

        if count <= 5:
            return "short"

        elif count <= 30:
            return "medium"

        return "long"

    df["text_length_group"] = df[
        "word_count"
    ].apply(
        get_length_group
    )

    return df