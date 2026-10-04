# ============================================================
# 29_analyze_amharic_dataset.py
#
# AMHARIC ANNOTATED DATASET ANALYSIS
#
# PURPOSE:
#
# Analyze the real structure of the annotated dataset before
# improving the semantic topic prediction system.
#
# This script analyzes:
#
#   1. Training data
#   2. Validation data
#   3. Domains
#   4. Topics
#   5. Keywords
#   6. Multi-label topics
#   7. Topic frequency
#   8. Rare topics
#   9. Unseen validation topics
#   10. Topic overlap between train and validation
#
# INPUT:
#
#   data/train/train_processed.xlsx
#   data/train/validation_processed.xlsx
#
# OUTPUT:
#
#   results/dataset_analysis/
#
#       dataset_analysis_summary.txt
#       dataset_analysis.json
#       train_topic_frequency.xlsx
#       validation_topic_frequency.xlsx
#       rare_topics.xlsx
#       unseen_validation_topics.xlsx
#       domain_statistics.xlsx
#
# RUN:
#
#   python notebooks/29_analyze_amharic_dataset.py
#
# ============================================================


import sys
import subprocess
import importlib
import json
from pathlib import Path
from collections import Counter

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# INSTALL MISSING PACKAGES
# ============================================================

def install_if_missing(
    package_name,
    import_name=None,
):

    if import_name is None:

        import_name = package_name

    try:

        importlib.import_module(
            import_name
        )

    except ImportError:

        print()

        print(
            f"Installing missing package: "
            f"{package_name}"
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
    "pandas"
)

install_if_missing(
    "openpyxl"
)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np


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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_analysis"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "dataset_analysis_summary.txt"
)


JSON_PATH = (
    OUTPUT_DIR
    / "dataset_analysis.json"
)


TRAIN_TOPIC_PATH = (
    OUTPUT_DIR
    / "train_topic_frequency.xlsx"
)


VALIDATION_TOPIC_PATH = (
    OUTPUT_DIR
    / "validation_topic_frequency.xlsx"
)


RARE_TOPIC_PATH = (
    OUTPUT_DIR
    / "rare_topics.xlsx"
)


UNSEEN_TOPIC_PATH = (
    OUTPUT_DIR
    / "unseen_validation_topics.xlsx"
)


DOMAIN_STATISTICS_PATH = (
    OUTPUT_DIR
    / "domain_statistics.xlsx"
)


# ============================================================
# COLUMN NAMES
# ============================================================

TEXT_COLUMN = "text"

DOMAIN_COLUMN = "domain"

TOPICS_COLUMN = "topics"

KEYWORDS_COLUMN = "keywords"


# ============================================================
# SETTINGS
# ============================================================

RARE_TOPIC_THRESHOLD = 5

VERY_RARE_TOPIC_THRESHOLD = 2


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(
    title,
):

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_section(
    title,
):

    print()

    print("-" * 80)

    print(title)

    print("-" * 80)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    value,
):

    if value is None:

        return ""


    if isinstance(
        value,
        float,
    ):

        if np.isnan(value):

            return ""


    return " ".join(
        str(value).strip().split()
    )


# ============================================================
# LABEL PARSER
#
# Supports:
#
# topic1 | topic2
#
# topic1
#
# Python-style lists:
#
# ['topic1', 'topic2']
#
# ============================================================

def parse_labels(
    value,
):

    if value is None:

        return []


    if isinstance(
        value,
        float,
    ):

        if np.isnan(value):

            return []


    # --------------------------------------------------------
    # Already a list
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        result = []

        for item in value:

            item = clean_text(
                item
            )

            if (
                item
                and item not in result
            ):

                result.append(
                    item
                )

        return result


    # --------------------------------------------------------
    # Convert to text
    # --------------------------------------------------------

    text = clean_text(
        value
    )


    if not text:

        return []


    # --------------------------------------------------------
    # Remove common brackets
    # --------------------------------------------------------

    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )


    # --------------------------------------------------------
    # Parse pipe-separated labels
    # --------------------------------------------------------

    if "|" in text:

        parts = text.split("|")

    # --------------------------------------------------------
    # Parse comma-separated labels
    # --------------------------------------------------------

    elif "," in text:

        parts = text.split(",")

    # --------------------------------------------------------
    # Single label
    # --------------------------------------------------------

    else:

        parts = [text]


    result = []


    for item in parts:

        item = clean_text(
            item
        )


        if (
            item
            and item not in result
        ):

            result.append(
                item
            )


    return result


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    path,
    dataset_name,
):

    print_header(
        f"LOADING {dataset_name.upper()} DATA"
    )


    print()

    print(
        "File:"
    )

    print(
        path
    )


    if not path.exists():

        print()

        print(
            "ERROR: File not found."
        )

        sys.exit(1)


    df = pd.read_excel(
        path,
        engine="openpyxl",
    )


    print()

    print(
        "Original rows:",
        f"{len(df):,}"
    )


    print()

    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )


    return df


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def check_columns(
    df,
    dataset_name,
):

    required_columns = [

        TEXT_COLUMN,

        DOMAIN_COLUMN,

        TOPICS_COLUMN,

        KEYWORDS_COLUMN,

    ]


    missing_columns = [

        column

        for column
        in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        print()

        print(
            f"ERROR: {dataset_name} "
            f"is missing required columns:"
        )

        print(
            missing_columns
        )

        sys.exit(1)


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(
    df,
    dataset_name,
):

    print_header(
        f"PREPARING {dataset_name.upper()} DATA"
    )


    df = df.copy()


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df[TEXT_COLUMN] = (

        df[TEXT_COLUMN]

        .apply(
            clean_text
        )

    )


    # --------------------------------------------------------
    # Clean domain
    # --------------------------------------------------------

    df[DOMAIN_COLUMN] = (

        df[DOMAIN_COLUMN]

        .apply(
            clean_text
        )

    )


    # --------------------------------------------------------
    # Parse topics
    # --------------------------------------------------------

    df["_topic_list"] = (

        df[TOPICS_COLUMN]

        .apply(
            parse_labels
        )

    )


    # --------------------------------------------------------
    # Parse keywords
    # --------------------------------------------------------

    df["_keyword_list"] = (

        df[KEYWORDS_COLUMN]

        .apply(
            parse_labels
        )

    )


    # --------------------------------------------------------
    # Remove empty text
    # --------------------------------------------------------

    before = len(df)


    df = df[

        df[TEXT_COLUMN].str.len() > 0

    ].copy()


    removed = (

        before
        - len(df)

    )


    print()

    print(
        "Removed empty texts:",
        removed
    )


    # --------------------------------------------------------
    # Remove empty domain
    # --------------------------------------------------------

    before = len(df)


    df = df[

        df[DOMAIN_COLUMN].str.len() > 0

    ].copy()


    removed = (

        before
        - len(df)

    )


    print()

    print(
        "Removed empty domains:",
        removed
    )


    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = (

        df
        .reset_index(
            drop=True
        )

    )


    print()

    print(
        "Final documents:",
        f"{len(df):,}"
    )


    return df


# ============================================================
# COUNT ALL LABELS
# ============================================================

def count_labels(
    label_lists,
):

    counter = Counter()


    for labels in label_lists:

        for label in labels:

            label = clean_text(
                label
            )


            if label:

                counter[
                    label
                ] += 1


    return counter


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(
    df,
    dataset_name,
):

    print_header(
        f"{dataset_name.upper()} DATASET ANALYSIS"
    )


    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    document_count = (

        len(df)

    )


    domain_counts = (

        df[DOMAIN_COLUMN]
        .value_counts()

    )


    unique_domains = (

        sorted(
            df[DOMAIN_COLUMN]
            .unique()
        )

    )


    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    topic_counter = (

        count_labels(
            df["_topic_list"]
        )

    )


    unique_topics = (

        set(
            topic_counter.keys()
        )

    )


    total_topic_annotations = (

        sum(
            len(labels)

            for labels
            in df["_topic_list"]
        )

    )


    documents_with_topics = (

        sum(

            1

            for labels
            in df["_topic_list"]

            if len(labels) > 0

        )

    )


    documents_without_topics = (

        document_count

        - documents_with_topics

    )


    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    keyword_counter = (

        count_labels(
            df["_keyword_list"]
        )

    )


    unique_keywords = (

        set(
            keyword_counter.keys()
        )

    )


    total_keyword_annotations = (

        sum(
            len(labels)

            for labels
            in df["_keyword_list"]
        )

    )


    documents_with_keywords = (

        sum(

            1

            for labels
            in df["_keyword_list"]

            if len(labels) > 0

        )

    )


    documents_without_keywords = (

        document_count

        - documents_with_keywords

    )


    # --------------------------------------------------------
    # Multi-label topics
    # --------------------------------------------------------

    topic_lengths = [

        len(labels)

        for labels
        in df["_topic_list"]

    ]


    single_topic_documents = (

        sum(

            1

            for length
            in topic_lengths

            if length == 1

        )

    )


    multi_topic_documents = (

        sum(

            1

            for length
            in topic_lengths

            if length > 1

        )

    )


    # --------------------------------------------------------
    # Multi-label keywords
    # --------------------------------------------------------

    keyword_lengths = [

        len(labels)

        for labels
        in df["_keyword_list"]

    ]


    single_keyword_documents = (

        sum(

            1

            for length
            in keyword_lengths

            if length == 1

        )

    )


    multi_keyword_documents = (

        sum(

            1

            for length
            in keyword_lengths

            if length > 1

        )

    )


    # --------------------------------------------------------
    # Rare topics
    # --------------------------------------------------------

    topics_frequency_1 = [

        topic

        for topic, count
        in topic_counter.items()

        if count == 1

    ]


    topics_frequency_2 = [

        topic

        for topic, count
        in topic_counter.items()

        if count <= 2

    ]


    topics_frequency_5 = [

        topic

        for topic, count
        in topic_counter.items()

        if count <= 5

    ]


    topics_frequency_10 = [

        topic

        for topic, count
        in topic_counter.items()

        if count <= 10

    ]


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_topics_per_document = (

        total_topic_annotations
        / document_count

        if document_count > 0

        else 0

    )


    average_keywords_per_document = (

        total_keyword_annotations
        / document_count

        if document_count > 0

        else 0

    )


    statistics = {

        "dataset":

            dataset_name,


        "documents":

            int(
                document_count
            ),


        "domains":

            int(
                len(
                    unique_domains
                )
            ),


        "unique_topics":

            int(
                len(
                    unique_topics
                )
            ),


        "unique_keywords":

            int(
                len(
                    unique_keywords
                )
            ),


        "total_topic_annotations":

            int(
                total_topic_annotations
            ),


        "total_keyword_annotations":

            int(
                total_keyword_annotations
            ),


        "documents_with_topics":

            int(
                documents_with_topics
            ),


        "documents_without_topics":

            int(
                documents_without_topics
            ),


        "documents_with_keywords":

            int(
                documents_with_keywords
            ),


        "documents_without_keywords":

            int(
                documents_without_keywords
            ),


        "single_topic_documents":

            int(
                single_topic_documents
            ),


        "multi_topic_documents":

            int(
                multi_topic_documents
            ),


        "single_keyword_documents":

            int(
                single_keyword_documents
            ),


        "multi_keyword_documents":

            int(
                multi_keyword_documents
            ),


        "average_topics_per_document":

            float(
                average_topics_per_document
            ),


        "average_keywords_per_document":

            float(
                average_keywords_per_document
            ),


        "topics_occurring_once":

            int(
                len(
                    topics_frequency_1
                )
            ),


        "topics_occurring_2_or_less":

            int(
                len(
                    topics_frequency_2
                )
            ),


        "topics_occurring_5_or_less":

            int(
                len(
                    topics_frequency_5
                )
            ),


        "topics_occurring_10_or_less":

            int(
                len(
                    topics_frequency_10
                )
            ),

    }


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        "Documents:",
        f"{document_count:,}"
    )


    print()

    print(
        "Domains:",
        f"{len(unique_domains):,}"
    )


    print()

    print(
        "Unique topics:",
        f"{len(unique_topics):,}"
    )


    print()

    print(
        "Unique keywords:",
        f"{len(unique_keywords):,}"
    )


    print()

    print(
        "Total topic annotations:",
        f"{total_topic_annotations:,}"
    )


    print()

    print(
        "Average topics/document:",
        f"{average_topics_per_document:.4f}"
    )


    print()

    print(
        "Single-topic documents:",
        f"{single_topic_documents:,}"
    )


    print()

    print(
        "Multi-topic documents:",
        f"{multi_topic_documents:,}"
    )


    print()

    print(
        "Topics occurring once:",
        f"{len(topics_frequency_1):,}"
    )


    print()

    print(
        "Topics occurring 2 or fewer times:",
        f"{len(topics_frequency_2):,}"
    )


    print()

    print(
        "Topics occurring 5 or fewer times:",
        f"{len(topics_frequency_5):,}"
    )


    print()

    print(
        "Topics occurring 10 or fewer times:",
        f"{len(topics_frequency_10):,}"
    )


    return (

        statistics,

        topic_counter,

        keyword_counter,

        domain_counts,

        unique_topics,

        unique_keywords,

    )


# ============================================================
# ANALYZE TOPIC DISTRIBUTION
# ============================================================

def create_topic_frequency_dataframe(
    topic_counter,
):

    rows = []


    for topic, count in (

        topic_counter.items()

    ):

        rows.append(

            {

                "topic":

                    topic,


                "frequency":

                    int(
                        count
                    ),


                "frequency_group":

                    get_frequency_group(
                        count
                    ),

            }

        )


    result = pd.DataFrame(
        rows
    )


    if len(result) > 0:

        result = (

            result
            .sort_values(

                by=[
                    "frequency",
                    "topic",
                ],

                ascending=[
                    False,
                    True,
                ],

            )

            .reset_index(
                drop=True
            )

        )


    return result


# ============================================================
# FREQUENCY GROUP
# ============================================================

def get_frequency_group(
    count,
):

    if count == 1:

        return "1 example"


    if count == 2:

        return "2 examples"


    if count <= 5:

        return "3-5 examples"


    if count <= 10:

        return "6-10 examples"


    if count <= 20:

        return "11-20 examples"


    if count <= 50:

        return "21-50 examples"


    return "More than 50 examples"


# ============================================================
# DOMAIN STATISTICS
# ============================================================

def create_domain_statistics(
    df,
):

    rows = []


    grouped = (

        df
        .groupby(
            DOMAIN_COLUMN
        )

    )


    for domain, group in grouped:


        document_count = (

            len(group)

        )


        topic_counter = (

            count_labels(
                group["_topic_list"]
            )

        )


        keyword_counter = (

            count_labels(
                group["_keyword_list"]
            )

        )


        total_topics = (

            sum(
                len(labels)

                for labels
                in group["_topic_list"]
            )

        )


        total_keywords = (

            sum(
                len(labels)

                for labels
                in group["_keyword_list"]
            )

        )


        rare_topics = (

            sum(

                1

                for count
                in topic_counter.values()

                if count <= RARE_TOPIC_THRESHOLD

            )

        )


        rows.append(

            {

                "domain":

                    domain,


                "documents":

                    int(
                        document_count
                    ),


                "unique_topics":

                    int(
                        len(
                            topic_counter
                        )
                    ),


                "unique_keywords":

                    int(
                        len(
                            keyword_counter
                        )
                    ),


                "total_topic_annotations":

                    int(
                        total_topics
                    ),


                "total_keyword_annotations":

                    int(
                        total_keywords
                    ),


                "average_topics_per_document":

                    float(
                        total_topics
                        / document_count
                    ),


                "average_keywords_per_document":

                    float(
                        total_keywords
                        / document_count
                    ),


                "rare_topics_5_or_less":

                    int(
                        rare_topics
                    ),

            }

        )


    result = pd.DataFrame(
        rows
    )


    if len(result) > 0:

        result = (

            result
            .sort_values(

                by="documents",

                ascending=False,

            )

            .reset_index(
                drop=True
            )

        )


    return result


# ============================================================
# TRAIN / VALIDATION OVERLAP
# ============================================================

def analyze_train_validation_overlap(

    train_topics,

    validation_topics,

):

    print_header(
        "TRAIN / VALIDATION TOPIC OVERLAP"
    )


    train_topics = set(
        train_topics
    )


    validation_topics = set(
        validation_topics
    )


    common_topics = (

        train_topics
        &
        validation_topics

    )


    unseen_validation_topics = (

        validation_topics
        -
        train_topics

    )


    train_only_topics = (

        train_topics
        -
        validation_topics

    )


    validation_topic_count = (

        len(
            validation_topics
        )

    )


    if validation_topic_count > 0:

        overlap_percentage = (

            len(common_topics)
            /
            validation_topic_count

        ) * 100

    else:

        overlap_percentage = 0.0


    print()

    print(
        "Train unique topics:",
        f"{len(train_topics):,}"
    )


    print()

    print(
        "Validation unique topics:",
        f"{len(validation_topics):,}"
    )


    print()

    print(
        "Common topics:",
        f"{len(common_topics):,}"
    )


    print()

    print(
        "Unseen validation topics:",
        f"{len(unseen_validation_topics):,}"
    )


    print()

    print(
        "Validation topic overlap:",
        f"{overlap_percentage:.2f}%"
    )


    return {

        "train_unique_topics":

            int(
                len(train_topics)
            ),


        "validation_unique_topics":

            int(
                len(validation_topics)
            ),


        "common_topics":

            int(
                len(common_topics)
            ),


        "unseen_validation_topics":

            int(
                len(
                    unseen_validation_topics
                )
            ),


        "train_only_topics":

            int(
                len(
                    train_only_topics
                )
            ),


        "validation_topic_overlap_percentage":

            float(
                overlap_percentage
            ),

    }, unseen_validation_topics


# ============================================================
# DOCUMENT LEVEL UNSEEN TOPIC ANALYSIS
# ============================================================

def analyze_unseen_validation_documents(

    validation_df,

    train_topics,

):

    train_topics = set(
        train_topics
    )


    rows = []


    documents_with_unseen_topics = 0


    for _, row in (

        validation_df.iterrows()

    ):


        topics = row[
            "_topic_list"
        ]


        unseen_topics = [

            topic

            for topic in topics

            if topic not in train_topics

        ]


        if unseen_topics:

            documents_with_unseen_topics += 1


            rows.append(

                {

                    "text":

                        row[
                            TEXT_COLUMN
                        ],


                    "domain":

                        row[
                            DOMAIN_COLUMN
                        ],


                    "validation_topics":

                        " | ".join(
                            topics
                        ),


                    "unseen_topics":

                        " | ".join(
                            unseen_topics
                        ),

                }

            )


    return (

        documents_with_unseen_topics,

        pd.DataFrame(
            rows
        ),

    )


# ============================================================
# RARE TOPIC DATAFRAME
# ============================================================

def create_rare_topic_dataframe(

    topic_counter,

):

    rows = []


    for topic, count in (

        topic_counter.items()

    ):


        if count <= RARE_TOPIC_THRESHOLD:


            if count <= VERY_RARE_TOPIC_THRESHOLD:

                category = (

                    "VERY RARE"

                )

            else:

                category = (

                    "RARE"

                )


            rows.append(

                {

                    "topic":

                        topic,


                    "frequency":

                        int(
                            count
                        ),


                    "category":

                        category,

                }

            )


    result = pd.DataFrame(
        rows
    )


    if len(result) > 0:

        result = (

            result
            .sort_values(

                by=[
                    "frequency",
                    "topic",
                ],

                ascending=[
                    True,
                    True,
                ],

            )

            .reset_index(
                drop=True
            )

        )


    return result


# ============================================================
# DUPLICATE TEXT ANALYSIS
# ============================================================

def analyze_duplicate_texts(

    df,

    dataset_name,

):

    print_header(
        f"{dataset_name.upper()} DUPLICATE ANALYSIS"
    )


    duplicate_count = (

        df[TEXT_COLUMN]
        .duplicated()
        .sum()

    )


    unique_text_count = (

        df[TEXT_COLUMN]
        .nunique()

    )


    print()

    print(
        "Total documents:",
        f"{len(df):,}"
    )


    print()

    print(
        "Unique texts:",
        f"{unique_text_count:,}"
    )


    print()

    print(
        "Duplicate texts:",
        f"{duplicate_count:,}"
    )


    return {

        "documents":

            int(
                len(df)
            ),


        "unique_texts":

            int(
                unique_text_count
            ),


        "duplicate_texts":

            int(
                duplicate_count
            ),

    }


# ============================================================
# TOPIC PER DOMAIN ANALYSIS
#
# This is very important.
#
# It checks whether the same topic occurs under multiple
# domains.
# ============================================================

def analyze_topic_domain_relationship(

    df,

):

    print_header(
        "TOPIC / DOMAIN RELATIONSHIP ANALYSIS"
    )


    topic_domains = {}


    for _, row in (

        df.iterrows()

    ):


        domain = clean_text(

            row[
                DOMAIN_COLUMN
            ]

        )


        topics = (

            row[
                "_topic_list"
            ]

        )


        for topic in topics:


            if topic not in topic_domains:

                topic_domains[
                    topic
                ] = set()


            topic_domains[
                topic
            ].add(
                domain
            )


    topics_in_multiple_domains = [

        topic

        for topic, domains
        in topic_domains.items()

        if len(domains) > 1

    ]


    print()

    print(
        "Total unique topics:",
        f"{len(topic_domains):,}"
    )


    print()

    print(
        "Topics appearing in multiple domains:",
        f"{len(topics_in_multiple_domains):,}"
    )


    rows = []


    for topic, domains in (

        topic_domains.items()

    ):


        if len(domains) > 1:


            rows.append(

                {

                    "topic":

                        topic,


                    "number_of_domains":

                        len(domains),


                    "domains":

                        " | ".join(
                            sorted(domains)
                        ),

                }

            )


    result = pd.DataFrame(
        rows
    )


    if len(result) > 0:

        result = (

            result
            .sort_values(

                by="number_of_domains",

                ascending=False,

            )

            .reset_index(
                drop=True
            )

        )


    return (

        {

            "unique_topics":

                int(
                    len(
                        topic_domains
                    )
                ),


            "topics_in_multiple_domains":

                int(
                    len(
                        topics_in_multiple_domains
                    )
                ),

        },

        result,

    )


# ============================================================
# CREATE SUMMARY TEXT
# ============================================================

def create_summary_text(

    train_statistics,

    validation_statistics,

    overlap_statistics,

    train_duplicate_statistics,

    validation_duplicate_statistics,

    unseen_document_count,

    topic_domain_statistics,

):

    lines = []


    lines.append(
        "=" * 80
    )

    lines.append(
        "AMHARIC ANNOTATED DATASET ANALYSIS SUMMARY"
    )

    lines.append(
        "=" * 80
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "TRAINING DATA"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        train_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "VALIDATION DATA"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        validation_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "TRAIN / VALIDATION TOPIC OVERLAP"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        overlap_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "TRAIN DUPLICATE ANALYSIS"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        train_duplicate_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    lines.append("")

    lines.append(
        "VALIDATION DUPLICATE ANALYSIS"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        validation_duplicate_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    # --------------------------------------------------------
    # Unseen documents
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "UNSEEN VALIDATION TOPICS"
    )

    lines.append(
        "-" * 80
    )


    lines.append(

        "Validation documents containing "
        "at least one unseen topic: "

        +

        str(
            unseen_document_count
        )

    )


    # --------------------------------------------------------
    # Topic / Domain relationship
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "TOPIC / DOMAIN RELATIONSHIP"
    )

    lines.append(
        "-" * 80
    )


    for key, value in (

        topic_domain_statistics.items()

    ):

        lines.append(

            f"{key}: {value}"

        )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "INTERPRETATION GUIDE"
    )

    lines.append(
        "=" * 80
    )


    lines.append("")

    lines.append(
        "1. A large number of topics occurring only once "
        "or a few times indicates sparse topic labels."
    )


    lines.append("")

    lines.append(
        "2. Sparse topic labels make exact topic "
        "classification difficult."
    )


    lines.append("")

    lines.append(
        "3. Unseen validation topics cannot normally be "
        "predicted by a closed-label classifier."
    )


    lines.append("")

    lines.append(
        "4. If validation documents contain unseen topics, "
        "exact-match accuracy may be artificially low."
    )


    lines.append("")

    lines.append(
        "5. If many documents have multiple topics, "
        "evaluation should use multi-label Precision, "
        "Recall, F1, and mAP."
    )


    lines.append("")

    lines.append(
        "6. If the same topic appears in multiple domains, "
        "strict domain restriction may hurt topic prediction."
    )


    lines.append("")

    lines.append(
        "7. The results of this analysis should be used "
        "before changing the semantic inference architecture."
    )


    return "\n".join(
        lines
    )


# ============================================================
# START
# ============================================================

print_header(
    "AMHARIC ANNOTATED DATASET ANALYSIS"
)


print()

print(
    "Project root:"
)

print(
    PROJECT_ROOT
)


print()

print(
    "Training file:"
)

print(
    TRAIN_PATH
)


print()

print(
    "Validation file:"
)

print(
    VALIDATION_PATH
)


print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_df = load_dataset(

    TRAIN_PATH,

    "Training",

)


check_columns(

    train_df,

    "Training data",

)


train_df = prepare_dataset(

    train_df,

    "Training",

)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

validation_df = load_dataset(

    VALIDATION_PATH,

    "Validation",

)


check_columns(

    validation_df,

    "Validation data",

)


validation_df = prepare_dataset(

    validation_df,

    "Validation",

)


# ============================================================
# ANALYZE TRAINING DATA
# ============================================================

(

    train_statistics,

    train_topic_counter,

    train_keyword_counter,

    train_domain_counts,

    train_unique_topics,

    train_unique_keywords,

) = analyze_dataset(

    train_df,

    "Training",

)


# ============================================================
# ANALYZE VALIDATION DATA
# ============================================================

(

    validation_statistics,

    validation_topic_counter,

    validation_keyword_counter,

    validation_domain_counts,

    validation_unique_topics,

    validation_unique_keywords,

) = analyze_dataset(

    validation_df,

    "Validation",

)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

train_duplicate_statistics = (

    analyze_duplicate_texts(

        train_df,

        "Training",

    )

)


validation_duplicate_statistics = (

    analyze_duplicate_texts(

        validation_df,

        "Validation",

    )

)


# ============================================================
# TRAIN / VALIDATION TOPIC OVERLAP
# ============================================================

(

    overlap_statistics,

    unseen_validation_topics,

) = analyze_train_validation_overlap(

    train_unique_topics,

    validation_unique_topics,

)


# ============================================================
# UNSEEN VALIDATION DOCUMENTS
# ============================================================

print_header(
    "UNSEEN VALIDATION DOCUMENT ANALYSIS"
)


(

    unseen_document_count,

    unseen_documents_df,

) = analyze_unseen_validation_documents(

    validation_df,

    train_unique_topics,

)


print()

print(
    "Validation documents containing "
    "at least one unseen topic:",
    f"{unseen_document_count:,}"
)


# ============================================================
# TOPIC / DOMAIN RELATIONSHIP
# ============================================================

(

    topic_domain_statistics,

    multi_domain_topics_df,

) = analyze_topic_domain_relationship(

    train_df

)


# ============================================================
# CREATE FREQUENCY DATAFRAMES
# ============================================================

print_header(
    "CREATING FREQUENCY TABLES"
)


train_topic_frequency_df = (

    create_topic_frequency_dataframe(

        train_topic_counter

    )

)


validation_topic_frequency_df = (

    create_topic_frequency_dataframe(

        validation_topic_counter

    )

)


rare_topics_df = (

    create_rare_topic_dataframe(

        train_topic_counter

    )

)


print()

print(
    "Training topic frequency rows:",
    f"{len(train_topic_frequency_df):,}"
)


print()

print(
    "Validation topic frequency rows:",
    f"{len(validation_topic_frequency_df):,}"
)


print()

print(
    "Rare training topics:",
    f"{len(rare_topics_df):,}"
)


# ============================================================
# DOMAIN STATISTICS
# ============================================================

print_header(
    "DOMAIN STATISTICS"
)


domain_statistics_df = (

    create_domain_statistics(

        train_df

    )

)


print()

print(
    "Domains analyzed:",
    f"{len(domain_statistics_df):,}"
)


print()

print(
    domain_statistics_df.to_string(

        index=False

    )

)


# ============================================================
# SAVE TRAIN TOPIC FREQUENCY
# ============================================================

print_header(
    "SAVING RESULTS"
)


print()

print(
    "Saving training topic frequency..."
)


train_topic_frequency_df.to_excel(

    TRAIN_TOPIC_PATH,

    index=False,

)


print(
    TRAIN_TOPIC_PATH
)


# ============================================================
# SAVE VALIDATION TOPIC FREQUENCY
# ============================================================

print()

print(
    "Saving validation topic frequency..."
)


validation_topic_frequency_df.to_excel(

    VALIDATION_TOPIC_PATH,

    index=False,

)


print(
    VALIDATION_TOPIC_PATH
)


# ============================================================
# SAVE RARE TOPICS
# ============================================================

print()

print(
    "Saving rare topics..."
)


rare_topics_df.to_excel(

    RARE_TOPIC_PATH,

    index=False,

)


print(
    RARE_TOPIC_PATH
)


# ============================================================
# SAVE UNSEEN VALIDATION TOPICS
# ============================================================

print()

print(
    "Saving unseen validation topics..."
)


unseen_topics_rows = []


for topic in sorted(
    unseen_validation_topics
):

    unseen_topics_rows.append(

        {

            "topic":

                topic

        }

    )


unseen_topics_df = (

    pd.DataFrame(

        unseen_topics_rows

    )

)


with pd.ExcelWriter(

    UNSEEN_TOPIC_PATH,

    engine="openpyxl",

) as writer:


    unseen_topics_df.to_excel(

        writer,

        sheet_name="unseen_topics",

        index=False,

    )


    unseen_documents_df.to_excel(

        writer,

        sheet_name="unseen_documents",

        index=False,

    )


print(
    UNSEEN_TOPIC_PATH
)


# ============================================================
# SAVE DOMAIN STATISTICS
# ============================================================

print()

print(
    "Saving domain statistics..."
)


with pd.ExcelWriter(

    DOMAIN_STATISTICS_PATH,

    engine="openpyxl",

) as writer:


    domain_statistics_df.to_excel(

        writer,

        sheet_name="domain_statistics",

        index=False,

    )


    multi_domain_topics_df.to_excel(

        writer,

        sheet_name="multi_domain_topics",

        index=False,

    )


print(
    DOMAIN_STATISTICS_PATH
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary_text = (

    create_summary_text(

        train_statistics,

        validation_statistics,

        overlap_statistics,

        train_duplicate_statistics,

        validation_duplicate_statistics,

        unseen_document_count,

        topic_domain_statistics,

    )

)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

print()

print(
    "Saving text summary..."
)


with open(

    SUMMARY_PATH,

    "w",

    encoding="utf-8",

) as file:


    file.write(

        summary_text

    )


print(
    SUMMARY_PATH
)


# ============================================================
# SAVE JSON
# ============================================================

print()

print(
    "Saving JSON results..."
)


json_data = {

    "training":

        train_statistics,


    "validation":

        validation_statistics,


    "topic_overlap":

        overlap_statistics,


    "training_duplicates":

        train_duplicate_statistics,


    "validation_duplicates":

        validation_duplicate_statistics,


    "validation_documents_with_unseen_topics":

        int(
            unseen_document_count
        ),


    "topic_domain_relationship":

        topic_domain_statistics,


    "rare_topic_threshold":

        RARE_TOPIC_THRESHOLD,


    "very_rare_topic_threshold":

        VERY_RARE_TOPIC_THRESHOLD,

}


with open(

    JSON_PATH,

    "w",

    encoding="utf-8",

) as file:


    json.dump(

        json_data,

        file,

        ensure_ascii=False,

        indent=4,

    )


print(
    JSON_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "DATASET ANALYSIS COMPLETE"
)


print()

print(
    "TRAINING DOCUMENTS:",
    f"{train_statistics['documents']:,}"
)


print()

print(
    "VALIDATION DOCUMENTS:",
    f"{validation_statistics['documents']:,}"
)


print()

print(
    "TRAINING DOMAINS:",
    f"{train_statistics['domains']:,}"
)


print()

print(
    "TRAINING UNIQUE TOPICS:",
    f"{train_statistics['unique_topics']:,}"
)


print()

print(
    "VALIDATION UNIQUE TOPICS:",
    f"{validation_statistics['unique_topics']:,}"
)


print()

print(
    "TRAINING TOPICS OCCURRING ONCE:",
    f"{train_statistics['topics_occurring_once']:,}"
)


print()

print(
    "TRAINING TOPICS OCCURRING "
    "5 OR FEWER TIMES:",
    f"{train_statistics['topics_occurring_5_or_less']:,}"
)


print()

print(
    "UNSEEN VALIDATION TOPICS:",
    f"{overlap_statistics['unseen_validation_topics']:,}"
)


print()

print(
    "VALIDATION TOPIC OVERLAP:",
    f"{overlap_statistics['validation_topic_overlap_percentage']:.2f}%"
)


print()

print(
    "VALIDATION DOCUMENTS WITH "
    "UNSEEN TOPICS:",
    f"{unseen_document_count:,}"
)


print()

print(
    "TOPICS APPEARING IN "
    "MULTIPLE DOMAINS:",
    f"{topic_domain_statistics['topics_in_multiple_domains']:,}"
)


print()

print("=" * 80)

print(
    "RESULT FILES"
)

print("=" * 80)


print()

print(
    SUMMARY_PATH
)


print()

print(
    JSON_PATH
)


print()

print(
    TRAIN_TOPIC_PATH
)


print()

print(
    VALIDATION_TOPIC_PATH
)


print()

print(
    RARE_TOPIC_PATH
)


print()

print(
    UNSEEN_TOPIC_PATH
)


print()

print(
    DOMAIN_STATISTICS_PATH
)


print()

print("=" * 80)

print(
    "NEXT STEP"
)

print("=" * 80)


print()

print(
    "Use the analysis results to determine:"
)


print()

print(
    "1. Whether topic labels are sparse."
)


print()

print(
    "2. Whether validation contains unseen topics."
)


print()

print(
    "3. Whether the problem is truly multi-label."
)


print()

print(
    "4. Whether domain restriction should be used."
)


print()

print(
    "5. Whether the current global topic inference"
)


print(
    "   should be replaced with hierarchical retrieval."
)


print()

print("=" * 80)

print()