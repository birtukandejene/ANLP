# ============================================================
# 26_build_amharic_semantic_index.py
#
# LOCAL WINDOWS VERSION
#
# Builds the Amharic semantic index using:
#   rasyosef/embedding-amharic-base
#
# INPUT:
#   data/train/train_processed.xlsx
#
# OUTPUT:
#   models/amharic_semantic_index.joblib
# ============================================================

import sys
import os
import subprocess
import importlib
from pathlib import Path
import warnings

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

install_if_missing(
    "openpyxl"
)

install_if_missing(
    "joblib"
)

install_if_missing(
    "pandas"
)

install_if_missing(
    "numpy"
)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = Path(
    os.environ.get(
        "ANLP_INDEX_TRAIN_PATH",
        str(PROJECT_ROOT / "data" / "train" / "train_processed.xlsx"),
    )
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_PATH = Path(
    os.environ.get(
        "ANLP_INDEX_OUTPUT_PATH",
        str(OUTPUT_DIR / "amharic_semantic_index.joblib"),
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "rasyosef/embedding-amharic-base"
)

BATCH_SIZE = 16

TEXT_COLUMN = "text"

DOMAIN_COLUMN = "domain"

TOPICS_COLUMN = "topics"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return " ".join(
        str(value).strip().split()
    )


# ============================================================
# TOPIC PARSER
# ============================================================

def parse_topics(value):

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    result = []

    for topic in text.split("|"):

        topic = topic.strip()

        if topic and topic not in result:

            result.append(topic)

    return result


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm <= 0:
        return vector

    return vector / norm


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
# ENCODING
# ============================================================

def encode_documents(
    model,
    texts
):

    print()
    print("=" * 70)
    print("ENCODING TRAINING DOCUMENTS")
    print("=" * 70)

    print()
    print(
        "Documents:",
        f"{len(texts):,}"
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print()

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


    return embeddings


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("AMHARIC SEMANTIC INDEX BUILDER")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Training file:")
print(TRAIN_PATH)

print()
print("Output file:")
print(OUTPUT_PATH)


# ============================================================
# CHECK TRAINING FILE
# ============================================================

if not TRAIN_PATH.exists():

    print()
    print("ERROR: Training file not found.")

    print()
    print("Expected:")
    print(TRAIN_PATH)

    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EXCEL
# ============================================================

print()
print("-" * 70)
print("LOADING TRAINING DATA")
print("-" * 70)

df = pd.read_excel(
    TRAIN_PATH,
    engine="openpyxl"
)

print()
print(
    "Original rows:",
    f"{len(df):,}"
)

print()
print("Columns:")
print(list(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    TEXT_COLUMN,
    DOMAIN_COLUMN,
    TOPICS_COLUMN,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print("ERROR: Missing columns:")
    print(missing_columns)

    sys.exit(1)


# ============================================================
# CLEAN TEXT AND DOMAIN
# ============================================================

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .apply(clean_text)
)

df[DOMAIN_COLUMN] = (
    df[DOMAIN_COLUMN]
    .apply(clean_text)
)


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

before = len(df)

df = df[
    df[TEXT_COLUMN].str.len() > 0
].copy()

print()
print(
    "Removed empty texts:",
    before - len(df)
)


# ============================================================
# REMOVE EMPTY DOMAIN
# ============================================================

before = len(df)

df = df[
    df[DOMAIN_COLUMN].str.len() > 0
].copy()

print(
    "Removed rows without domain:",
    before - len(df)
)


# ============================================================
# IMPORTANT FIX
#
# DO NOT convert topics to lists before drop_duplicates().
#
# pandas cannot hash Python lists.
#
# We first create a string version specifically for duplicate
# detection.
# ============================================================

df["_topics_for_duplicate_check"] = (
    df[TOPICS_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE EXACT DUPLICATE RECORDS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        TEXT_COLUMN,
        DOMAIN_COLUMN,
        "_topics_for_duplicate_check",
    ],
    keep="first"
).copy()

print()
print(
    "Removed exact duplicate records:",
    before - len(df)
)


# ============================================================
# REMOVE TEMPORARY COLUMN
# ============================================================

df = df.drop(
    columns=[
        "_topics_for_duplicate_check"
    ]
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(
    drop=True
)


# ============================================================
# PARSE TOPICS ONLY NOW
# ============================================================

df[TOPICS_COLUMN] = (
    df[TOPICS_COLUMN]
    .apply(parse_topics)
)


# ============================================================
# FINAL DATA
# ============================================================

texts = (
    df[TEXT_COLUMN]
    .tolist()
)

domains = (
    df[DOMAIN_COLUMN]
    .tolist()
)

topics = (
    df[TOPICS_COLUMN]
    .tolist()
)


print()
print("=" * 70)
print("FINAL TRAINING DATA")
print("=" * 70)

print()
print(
    "Documents:",
    f"{len(texts):,}"
)

print()
print(
    "Domains:",
    len(set(domains))
)

print()
print("Domain distribution:")

for domain in sorted(
    set(domains)
):

    count = sum(
        1
        for item in domains
        if item == domain
    )

    print(
        f"  {domain}: {count:,}"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING AMHARIC EMBEDDING MODEL")
print("=" * 70)

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

    print(
        "This is slower but completely supported."
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

embeddings = encode_documents(
    model,
    texts
)


# ============================================================
# VERIFY EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("EMBEDDING INFORMATION")
print("=" * 70)

print()
print(
    "Shape:",
    embeddings.shape
)

print()
print(
    "Dtype:",
    embeddings.dtype
)

print()
print(
    "Memory:",
    f"{embeddings.nbytes / (1024 ** 2):.2f} MB"
)


# ============================================================
# BUILD DOMAIN CENTROIDS
# ============================================================

print()
print("=" * 70)
print("BUILDING DOMAIN CENTROIDS")
print("=" * 70)


domain_sums = {}

domain_counts = {}


for i, domain in enumerate(
    domains
):

    if domain not in domain_sums:

        domain_sums[domain] = (
            np.zeros(
                embeddings.shape[1],
                dtype=np.float32
            )
        )

        domain_counts[domain] = 0


    domain_sums[domain] += (
        embeddings[i]
    )

    domain_counts[domain] += 1


domain_centroids = {}


for domain in domain_sums:

    centroid = (
        domain_sums[domain]
        / domain_counts[domain]
    )

    centroid = normalize_vector(
        centroid
    )

    domain_centroids[domain] = (
        centroid
    )


print()
print(
    "Domain centroids:",
    len(domain_centroids)
)


# ============================================================
# BUILD TOPIC CENTROIDS
# ============================================================

print()
print("=" * 70)
print("BUILDING SUBTOPIC CENTROIDS")
print("=" * 70)


topic_sums = {}

topic_counts = {}


for i, domain in enumerate(
    domains
):

    document_topics = topics[i]


    for topic in document_topics:

        topic = str(
            topic
        ).strip()


        if not topic:
            continue


        key = (
            domain,
            topic
        )


        if key not in topic_sums:

            topic_sums[key] = (
                np.zeros(
                    embeddings.shape[1],
                    dtype=np.float32
                )
            )

            topic_counts[key] = 0


        topic_sums[key] += (
            embeddings[i]
        )

        topic_counts[key] += 1


topic_centroids = {}


for (
    domain,
    topic
), vector_sum in (
    topic_sums.items()
):

    if domain not in topic_centroids:

        topic_centroids[domain] = {}


    centroid = (
        vector_sum
        / topic_counts[
            (domain, topic)
        ]
    )


    centroid = normalize_vector(
        centroid
    )


    topic_centroids[
        domain
    ][topic] = centroid


print()
print(
    "Domain/subtopic centroid pairs:",
    len(topic_sums)
)


# ============================================================
# TOPIC FREQUENCIES
# ============================================================

topic_frequencies = {}


for (
    domain,
    topic
), count in (
    topic_counts.items()
):

    if domain not in topic_frequencies:

        topic_frequencies[domain] = {}


    topic_frequencies[
        domain
    ][topic] = int(count)


# ============================================================
# BUILD FINAL INDEX
# ============================================================

print()
print("=" * 70)
print("BUILDING FINAL SEMANTIC INDEX")
print("=" * 70)


semantic_index = {

    "index_version":
        "amharic_semantic_hybrid_v1",

    "semantic_model_name":
        MODEL_NAME,

    "embedding_dimension":
        int(
            embeddings.shape[1]
        ),

    "document_count":
        int(
            len(texts)
        ),

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

    "domain_counts":
        domain_counts,

    "topic_centroids":
        topic_centroids,

    "topic_frequencies":
        topic_frequencies,
}


# ============================================================
# SAVE
# ============================================================

print()
print("Saving index...")

joblib.dump(
    semantic_index,
    OUTPUT_PATH,
    compress=3
)


# ============================================================
# FINAL RESULT
# ============================================================

file_size_mb = (
    OUTPUT_PATH.stat().st_size
    / (1024 ** 2)
)


print()
print("=" * 70)
print("SEMANTIC INDEX BUILD COMPLETE")
print("=" * 70)

print()
print(
    "Documents indexed:",
    f"{len(texts):,}"
)

print()
print(
    "Embedding dimension:",
    embeddings.shape[1]
)

print()
print(
    "Domains:",
    len(domain_centroids)
)

print()
print(
    "Domain/subtopic centroids:",
    len(topic_sums)
)

print()
print(
    "Index file:"
)

print(
    OUTPUT_PATH
)

print()
print(
    "Index size:",
    f"{file_size_mb:.2f} MB"
)

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print()
print(
    "Run:"
)

print()
print(
    "python notebooks\\27_amharic_semantic_hybrid_inference.py"
)

print()