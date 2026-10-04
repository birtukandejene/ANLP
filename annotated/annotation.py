from pathlib import Path

import json
import os
import re
import time
import traceback
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from groq import Groq
from google import genai
from google.genai import types


# ============================================================
# FILE CONFIGURATION
# ============================================================

# IMPORTANT:
# This is the completed 25k dataset.
# It is used ONLY as the GOLD/reference dataset.
GOLD_FILE = Path(
    "data/annotation/amharic_annotation_25000_normalized.xlsx"
)

# This is the NEW 23k dataset that will be annotated.
WORKING_FILE = Path(
    "data/annotation/amharic_annotation_balanced.xlsx"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.1-flash-lite"


# ============================================================
# COLUMN CONFIGURATION
# ============================================================

TEXT_COLUMN = "text"
KEYWORDS_COLUMN = "keywords"
DOMAIN_COLUMN = "domain"
TOPICS_COLUMN = "topics"
NOTES_COLUMN = "notes"
STATUS_COLUMN = "status"
PROVIDER_COLUMN = "annotation_provider"


# ============================================================
# TURBO PERFORMANCE CONFIGURATION
# ============================================================

# Number of rows in one API request.
BATCH_SIZE = 30

# Simultaneous Groq requests.
GROQ_WORKERS = 12

# Simultaneous Gemini requests.
GEMINI_WORKERS = 8

# Save after this many successful annotations.
SAVE_EVERY = 600


# Maximum retries for temporary API failures.
MAX_RETRIES = 3

# Initial retry delay.
INITIAL_RETRY_DELAY = 1.0

# Maximum retry delay.
MAX_RETRY_DELAY = 30.0

# Optional pacing.
GROQ_MIN_INTERVAL = 0.0
GEMINI_MIN_INTERVAL = 0.0

# If a whole batch fails validation, retry its rows individually.
FALLBACK_TO_INDIVIDUAL = False


# ============================================================
# DAILY QUOTA FLAGS
# ============================================================

GROQ_DAILY_EXHAUSTED = False
GEMINI_DAILY_EXHAUSTED = False


# ============================================================
# THREAD STATE
# ============================================================

GROQ_LOCK = threading.Lock()
GEMINI_LOCK = threading.Lock()

LAST_GROQ_REQUEST = 0.0
LAST_GEMINI_REQUEST = 0.0

QUOTA_LOCK = threading.Lock()


# ============================================================
# REGEX
# ============================================================

AMHARIC_PATTERN = re.compile(
    r"[\u1200-\u137F]"
)

ENGLISH_PATTERN = re.compile(
    r"[A-Za-z]"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_spaces(value):
    if not isinstance(value, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def contains_amharic(value):
    if not isinstance(value, str):
        return False

    return bool(
        AMHARIC_PATTERN.search(value)
    )


def contains_english(value):
    if not isinstance(value, str):
        return False

    return bool(
        ENGLISH_PATTERN.search(value)
    )


def is_nonempty(value):
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, list):
        return len(value) > 0

    text = str(value).strip()

    if not text:
        return False

    if text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return False

    return True


# ============================================================
# KEYWORD RULES
# ============================================================

def sentence_word_count(text):
    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"\S+",
            text,
        )
    )


def minimum_reasonable_keywords(text):
    count = sentence_word_count(text)

    if count <= 4:
        return 1

    if count <= 7:
        return 2

    if count <= 10:
        return 3

    if count <= 15:
        return 4

    return 5


def keyword_target(text):
    count = sentence_word_count(text)

    if count <= 4:
        return 1

    if count <= 7:
        return 3

    if count <= 10:
        return 4

    if count <= 15:
        return 5

    if count <= 25:
        return 6

    if count <= 40:
        return 7

    return 8


# ============================================================
# DOMAIN NORMALIZATION
# ============================================================

def normalize_domain(domain):
    if not isinstance(domain, str):
        return ""

    return normalize_spaces(domain)


# ============================================================
# GOLD DOMAIN LOADING
# ============================================================

def load_gold_domains():
    print()
    print("=" * 70)
    print("LOADING 25K GOLD DATASET")
    print("=" * 70)

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Gold dataset not found:\n{GOLD_FILE}"
        )

    gold_df = pd.read_excel(
        GOLD_FILE
    )

    print(
        f"Gold rows: {len(gold_df):,}"
    )

    if DOMAIN_COLUMN not in gold_df.columns:
        raise ValueError(
            f"Gold dataset does not contain "
            f"'{DOMAIN_COLUMN}' column."
        )

    domains = set()

    for value in gold_df[DOMAIN_COLUMN]:

        if not is_nonempty(value):
            continue

        domain = normalize_domain(
            str(value)
        )

        if domain:
            domains.add(domain)

    if not domains:
        raise ValueError(
            "No valid domains were found "
            "in the gold dataset."
        )

    domains = sorted(domains)

    print(
        f"Unique gold domains: {len(domains)}"
    )

    print()
    print("ALLOWED GOLD DOMAINS")
    print("-" * 70)

    for domain in domains:
        print(
            f"  - {domain}"
        )

    print("=" * 70)

    return domains


# ============================================================
# GOLD DOMAIN DISTRIBUTION
# ============================================================

def show_gold_domain_distribution(gold_df):
    if DOMAIN_COLUMN not in gold_df.columns:
        return

    print()
    print("=" * 70)
    print("25K GOLD DOMAIN DISTRIBUTION")
    print("=" * 70)

    counts = (
        gold_df[DOMAIN_COLUMN]
        .dropna()
        .astype(str)
        .map(normalize_spaces)
        .value_counts()
    )

    total = counts.sum()

    for domain, count in counts.items():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{domain:<30} "
            f"{count:>7,} "
            f"({percentage:>6.2f}%)"
        )

    print("=" * 70)


# ============================================================
# KEYWORDS
# ============================================================

def clean_keywords(keywords, text):

    if not isinstance(keywords, list):
        return []

    cleaned = []

    for keyword in keywords:

        if not isinstance(keyword, str):
            continue

        keyword = normalize_spaces(
            keyword
        )

        if not keyword:
            continue

        # Exact literal occurrence.
        if keyword not in text:
            continue

        if keyword not in cleaned:
            cleaned.append(keyword)

    return cleaned


# ============================================================
# VALIDATION
# ============================================================

def validate_annotation(
    annotation,
    text,
    allowed_domains,
):

    if not isinstance(annotation, dict):
        return False, "Not a dictionary."

    required = [
        KEYWORDS_COLUMN,
        DOMAIN_COLUMN,
        TOPICS_COLUMN,
        NOTES_COLUMN,
    ]

    for field in required:

        if field not in annotation:
            return False, (
                f"Missing field: {field}"
            )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    keywords = clean_keywords(
        annotation[KEYWORDS_COLUMN],
        text,
    )

    if not keywords:
        return False, (
            "No valid extractive keywords."
        )

    if len(keywords) > 8:
        return False, (
            "More than 8 keywords."
        )

    minimum = minimum_reasonable_keywords(
        text
    )

    if len(keywords) < minimum:
        return False, (
            f"Too few keywords: "
            f"{len(keywords)} < {minimum}"
        )

    # --------------------------------------------------------
    # DOMAIN
    # --------------------------------------------------------

    domain = normalize_domain(
        annotation[DOMAIN_COLUMN]
    )

    if not domain:
        return False, "Empty domain."

    # CRITICAL:
    # Domain MUST exist in the 25k gold dataset.
    if domain not in allowed_domains:
        return False, (
            f"Domain not in 25k gold domain set: "
            f"{domain}"
        )

    # Domain must be Amharic.
    if contains_english(domain):
        return False, (
            "English in domain."
        )

    if not contains_amharic(domain):
        return False, (
            "Domain is not Amharic."
        )

    # --------------------------------------------------------
    # TOPICS
    # --------------------------------------------------------

    topics = annotation[TOPICS_COLUMN]

    if not isinstance(topics, list):
        return False, (
            "Topics must be a list."
        )

    topics = [
        normalize_spaces(topic)
        for topic in topics
        if isinstance(topic, str)
        and normalize_spaces(topic)
    ]

    if not topics:
        return False, "No topics."

    if len(topics) > 4:
        return False, (
            "Too many topics."
        )

    for topic in topics:

        if contains_english(topic):
            return False, (
                f"English in topic: {topic}"
            )

        if not contains_amharic(topic):
            return False, (
                f"Topic not Amharic: {topic}"
            )

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    notes = normalize_spaces(
        annotation[NOTES_COLUMN]
    )

    if notes:

        if contains_english(notes):
            return False, (
                "English in notes."
            )

        if not contains_amharic(notes):
            return False, (
                "Notes not Amharic."
            )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    annotation[KEYWORDS_COLUMN] = keywords
    annotation[DOMAIN_COLUMN] = domain
    annotation[TOPICS_COLUMN] = topics
    annotation[NOTES_COLUMN] = notes

    return True, ""


# ============================================================
# GEMINI JSON SCHEMA
# ============================================================

ANNOTATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1,
            "maxItems": 8,
        },
        "domain": {
            "type": "string",
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1,
            "maxItems": 4,
        },
        "notes": {
            "type": "string",
        },
    },
    "required": [
        "id",
        "keywords",
        "domain",
        "topics",
        "notes",
    ],
}


ANNOTATION_BATCH_SCHEMA = {
    "type": "array",
    "items": ANNOTATION_ITEM_SCHEMA,
}


# ============================================================
# BATCH PROMPT
# ============================================================

def build_batch_prompt(
    items,
    allowed_domains,
):

    lines = []

    for item in items:

        row_id = item["id"]
        text = item["text"]

        words = sentence_word_count(
            text
        )

        target = keyword_target(
            text
        )

        minimum = minimum_reasonable_keywords(
            text
        )

        lines.append(
            f"""
ID: {row_id}

WORDS: {words}

KEYWORD TARGET: {target}

KEYWORD MINIMUM: {minimum}

TEXT:
{text}
"""
        )

    sentences = "\n".join(lines)

    domain_list = "\n".join(
        f"- {domain}"
        for domain in allowed_domains
    )

    return f"""
You are an expert Amharic NLP annotator.

You will annotate MULTIPLE Amharic sentences.

Return exactly ONE annotation for every ID.

============================================================
KEYWORDS
============================================================

Extract important keywords directly from the ORIGINAL sentence.

CRITICAL RULE:

Every keyword MUST occur literally in that sentence.

NEVER invent keywords.

Do not translate keywords.

Do not paraphrase keywords.

English words are allowed as keywords ONLY if they literally
occur in the original sentence.

Use the keyword target and minimum supplied for each sentence.

============================================================
DOMAIN
============================================================

Return exactly ONE main domain.

CRITICAL GOLD DATASET RULE:

The domain MUST be selected from the following domain labels
used in the existing 25,000-sample GOLD dataset.

NEVER create a new domain.

NEVER invent a domain.

NEVER translate a domain into another label.

NEVER use an English domain.

Use the EXACT spelling of one of the following domains.

ALLOWED GOLD DOMAINS:

{domain_list}

Choose exactly ONE domain from this list.

============================================================
TOPICS
============================================================

Return 1-4 conceptual topics.

Topics MUST be written in Amharic.

Do not use English unless the English word literally belongs
to the original sentence and is required as part of the topic.

Prefer natural Amharic conceptual topics.

============================================================
NOTES
============================================================

Notes MUST be in Amharic.

If there is no special note, return an empty string.

============================================================
OUTPUT
============================================================

Return ONLY a JSON array.

Every input ID MUST appear exactly once.

Example:

[
  {{
    "id": 123,
    "keywords": ["..."],
    "domain": "ፖለቲካ",
    "topics": ["..."],
    "notes": ""
  }}
]

============================================================
SENTENCES
============================================================

{sentences}
"""


# ============================================================
# ERROR DETECTION
# ============================================================

def is_rate_limit_error(exc):

    text = str(exc).lower()

    return (
        "429" in text
        or "rate_limit_exceeded" in text
        or "rate limit" in text
        or "resource_exhausted" in text
        or "too many requests" in text
        or "quota" in text
        or "tokens per minute" in text
        or "requests per minute" in text
    )


def is_daily_quota_error(exc):

    text = str(exc).lower()

    if "tokens per day" in text:
        return True

    if "tpd" in text:
        return True

    if "daily quota" in text:
        return True

    if "quota exceeded" in text:
        return True

    if "daily" in text and "quota" in text:
        return True

    return False


# ============================================================
# RATE LIMIT HELPERS
# ============================================================

def wait_for_groq_slot():

    global LAST_GROQ_REQUEST

    if GROQ_MIN_INTERVAL <= 0:
        return

    with GROQ_LOCK:

        now = time.time()

        elapsed = (
            now - LAST_GROQ_REQUEST
        )

        if elapsed < GROQ_MIN_INTERVAL:

            time.sleep(
                GROQ_MIN_INTERVAL - elapsed
            )

        LAST_GROQ_REQUEST = time.time()


def wait_for_gemini_slot():

    global LAST_GEMINI_REQUEST

    if GEMINI_MIN_INTERVAL <= 0:
        return

    with GEMINI_LOCK:

        now = time.time()

        elapsed = (
            now - LAST_GEMINI_REQUEST
        )

        if elapsed < GEMINI_MIN_INTERVAL:

            time.sleep(
                GEMINI_MIN_INTERVAL - elapsed
            )

        LAST_GEMINI_REQUEST = time.time()


# ============================================================
# QUOTA FLAGS
# ============================================================

def groq_is_available():

    with QUOTA_LOCK:
        return not GROQ_DAILY_EXHAUSTED


def gemini_is_available():

    with QUOTA_LOCK:
        return not GEMINI_DAILY_EXHAUSTED


def disable_groq():

    global GROQ_DAILY_EXHAUSTED

    with QUOTA_LOCK:
        GROQ_DAILY_EXHAUSTED = True


def disable_gemini():

    global GEMINI_DAILY_EXHAUSTED

    with QUOTA_LOCK:
        GEMINI_DAILY_EXHAUSTED = True


# ============================================================
# JSON CLEANING
# ============================================================

def extract_json(text):

    if not isinstance(text, str):
        raise ValueError(
            "Response is not text."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Empty model response."
        )

    # Remove markdown fences.
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        # Find first JSON array.
        start = text.find("[")
        end = text.rfind("]")

        if start >= 0 and end > start:

            candidate = text[
                start:end + 1
            ]

            return json.loads(
                candidate
            )

        # Find JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:

            candidate = text[
                start:end + 1
            ]

            return json.loads(
                candidate
            )

        raise


# ============================================================
# GROQ BATCH CALL
# ============================================================

def call_groq_batch(
    client,
    items,
):

    wait_for_groq_slot()

    prompt = build_batch_prompt(
        items,
        ALLOWED_DOMAINS,
    )

    completion = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Amharic NLP "
                    "annotator. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.1,

        response_format={
            "type": "json_object"
        },
    )

    content = (
        completion
        .choices[0]
        .message
        .content
    )

    if not content:
        raise ValueError(
            "Groq returned empty response."
        )

    parsed = extract_json(
        content
    )

    if isinstance(parsed, dict):

        for key in [
            "annotations",
            "results",
            "data",
        ]:

            if key in parsed:

                parsed = parsed[key]
                break

    if not isinstance(parsed, list):

        raise ValueError(
            "Groq batch response is not a list."
        )

    return parsed


# ============================================================
# GEMINI BATCH CALL
# ============================================================

def call_gemini_batch(
    client,
    items,
):

    wait_for_gemini_slot()

    prompt = build_batch_prompt(
        items,
        ALLOWED_DOMAINS,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,

        contents=prompt,

        config=types.GenerateContentConfig(
            temperature=0.1,

            response_mime_type="application/json",

            response_json_schema=(
                ANNOTATION_BATCH_SCHEMA
            ),
        ),
    )

    if not response.text:

        raise ValueError(
            "Gemini returned empty response."
        )

    parsed = extract_json(
        response.text
    )

    if not isinstance(parsed, list):

        raise ValueError(
            "Gemini batch response "
            "is not a list."
        )

    return parsed


# ============================================================
# NORMALIZE BATCH RESULTS
# ============================================================

def normalize_batch_results(
    results,
    items,
):

    if not isinstance(results, list):

        raise ValueError(
            "Results must be a list."
        )

    expected_ids = {
        int(item["id"])
        for item in items
    }

    normalized = {}

    for result in results:

        if not isinstance(result, dict):
            continue

        if "id" not in result:
            continue

        try:

            row_id = int(
                result["id"]
            )

        except Exception:

            continue

        if row_id not in expected_ids:
            continue

        normalized[
            row_id
        ] = result

    return normalized


# ============================================================
# VALIDATE BATCH
# ============================================================

def validate_batch_results(
    results,
    items,
):

    normalized = (
        normalize_batch_results(
            results,
            items,
        )
    )

    valid_results = {}
    errors = {}

    for item in items:

        row_id = int(
            item["id"]
        )

        text = item["text"]

        annotation = normalized.get(
            row_id
        )

        if annotation is None:

            errors[row_id] = (
                "Missing result."
            )

            continue

        valid, error = (
            validate_annotation(
                annotation,
                text,
                ALLOWED_DOMAINS,
            )
        )

        if not valid:

            errors[row_id] = error

            continue

        valid_results[
            row_id
        ] = annotation

    return (
        valid_results,
        errors,
    )


# ============================================================
# RETRY CALCULATION
# ============================================================

def retry_delay(
    attempt,
    exc=None,
):

    if exc is not None:

        text = str(exc)

        match = re.search(
            r"(\d+)m([\d.]+)s",
            text,
            re.IGNORECASE,
        )

        if match:

            seconds = (
                float(match.group(1))
                * 60
                + float(match.group(2))
            )

            return min(
                seconds,
                MAX_RETRY_DELAY,
            )

        match = re.search(
            r"([\d.]+)s",
            text,
            re.IGNORECASE,
        )

        if match:

            seconds = float(
                match.group(1)
            )

            return min(
                seconds,
                MAX_RETRY_DELAY,
            )

    delay = (
        INITIAL_RETRY_DELAY
        * (2 ** attempt)
    )

    return min(
        delay,
        MAX_RETRY_DELAY,
    )


# ============================================================
# GROQ BATCH WORKER
# ============================================================

def groq_worker(
    client,
    batch_number,
    items,
):

    if not groq_is_available():

        return {
            "batch_number": batch_number,
            "provider": "Groq",
            "status": "DISABLED",
            "results": {},
            "errors": {},
        }

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            results = call_groq_batch(
                client,
                items,
            )

            valid_results, errors = (
                validate_batch_results(
                    results,
                    items,
                )
            )

            return {
                "batch_number": batch_number,
                "provider": "Groq",
                "status": "SUCCESS",
                "results": valid_results,
                "errors": errors,
            }

        except Exception as exc:

            if is_daily_quota_error(
                exc
            ):

                disable_groq()

                print(
                    "\n[Groq] DAILY QUOTA REACHED."
                )

                return {
                    "batch_number": batch_number,
                    "provider": "Groq",
                    "status": "DAILY_QUOTA",
                    "results": {},
                    "errors": {},
                }

            if (
                is_rate_limit_error(exc)
                and attempt < MAX_RETRIES
            ):

                delay = retry_delay(
                    attempt,
                    exc,
                )

                print(
                    f"\n[Groq] rate limit "
                    f"batch {batch_number} "
                    f"-> retry in "
                    f"{delay:.1f}s"
                )

                time.sleep(delay)

                continue

            return {
                "batch_number": batch_number,
                "provider": "Groq",
                "status": "FAILED",
                "results": {},
                "errors": {
                    item["id"]: str(exc)
                    for item in items
                },
            }

    return {
        "batch_number": batch_number,
        "provider": "Groq",
        "status": "FAILED",
        "results": {},
        "errors": {},
    }


# ============================================================
# GEMINI BATCH WORKER
# ============================================================

def gemini_worker(
    client,
    batch_number,
    items,
):

    if not gemini_is_available():

        return {
            "batch_number": batch_number,
            "provider": "Gemini",
            "status": "DISABLED",
            "results": {},
            "errors": {},
        }

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            results = call_gemini_batch(
                client,
                items,
            )

            valid_results, errors = (
                validate_batch_results(
                    results,
                    items,
                )
            )

            return {
                "batch_number": batch_number,
                "provider": "Gemini",
                "status": "SUCCESS",
                "results": valid_results,
                "errors": errors,
            }

        except Exception as exc:

            if is_daily_quota_error(
                exc
            ):

                disable_gemini()

                print(
                    "\n[Gemini] DAILY QUOTA REACHED."
                )

                return {
                    "batch_number": batch_number,
                    "provider": "Gemini",
                    "status": "DAILY_QUOTA",
                    "results": {},
                    "errors": {},
                }

            if (
                is_rate_limit_error(exc)
                and attempt < MAX_RETRIES
            ):

                delay = retry_delay(
                    attempt,
                    exc,
                )

                print(
                    f"\n[Gemini] rate limit "
                    f"batch {batch_number} "
                    f"-> retry in "
                    f"{delay:.1f}s"
                )

                time.sleep(delay)

                continue

            return {
                "batch_number": batch_number,
                "provider": "Gemini",
                "status": "FAILED",
                "results": {},
                "errors": {
                    item["id"]: str(exc)
                    for item in items
                },
            }

    return {
        "batch_number": batch_number,
        "provider": "Gemini",
        "status": "FAILED",
        "results": {},
        "errors": {},
    }


# ============================================================
# INDIVIDUAL FALLBACK
# ============================================================

def annotate_single_groq(
    client,
    item,
):

    return groq_worker(
        client,
        item["id"],
        [item],
    )


def annotate_single_gemini(
    client,
    item,
):

    return gemini_worker(
        client,
        item["id"],
        [item],
    )


# ============================================================
# EXCEL HELPERS
# ============================================================

def keywords_to_excel(keywords):

    if not isinstance(
        keywords,
        list,
    ):
        return ""

    return " | ".join(
        normalize_spaces(k)
        for k in keywords
        if normalize_spaces(k)
    )


def topics_to_excel(topics):

    if not isinstance(
        topics,
        list,
    ):
        return ""

    return " | ".join(
        normalize_spaces(t)
        for t in topics
        if normalize_spaces(t)
    )


def row_is_annotated(row):

    status = str(
        row.get(
            STATUS_COLUMN,
            "",
        )
    ).strip().lower()

    if status == "annotated":
        return True

    keywords = row.get(
        KEYWORDS_COLUMN,
        "",
    )

    domain = row.get(
        DOMAIN_COLUMN,
        "",
    )

    topics = row.get(
        TOPICS_COLUMN,
        "",
    )

    return (
        is_nonempty(keywords)
        and is_nonempty(domain)
        and is_nonempty(topics)
    )


# ============================================================
# STATUS
# ============================================================

def normalize_statuses(df):

    # FIX: pandas may infer an empty Excel status column as float64.
    # Force all annotation columns to object before assigning strings
    # such as "Pending" and "Annotated".
    for column in [
        KEYWORDS_COLUMN,
        DOMAIN_COLUMN,
        TOPICS_COLUMN,
        NOTES_COLUMN,
        STATUS_COLUMN,
        PROVIDER_COLUMN,
    ]:
        if column not in df.columns:
            df[column] = pd.Series(
                [""] * len(df),
                index=df.index,
                dtype="object",
            )
        else:
            df[column] = df[column].astype("object")

    annotated = 0
    pending = 0
    blank = 0

    for index in df.index:

        text = df.at[
            index,
            TEXT_COLUMN,
        ]

        if not is_nonempty(text):

            df.at[
                index,
                STATUS_COLUMN,
            ] = ""

            blank += 1

            continue

        if row_is_annotated(
            df.loc[index]
        ):

            df.at[
                index,
                STATUS_COLUMN,
            ] = "Annotated"

            annotated += 1

        else:

            df.at[
                index,
                STATUS_COLUMN,
            ] = "Pending"

            pending += 1

    return (
        annotated,
        pending,
        blank,
    )


# ============================================================
# FILE CHECK
# ============================================================

def check_file_writable(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    try:

        with open(path, "r+b"):
            pass

    except PermissionError:

        print()
        print("=" * 70)
        print("ERROR: EXCEL FILE IS LOCKED")
        print("=" * 70)

        print(
            f"Close the workbook:\n{path}"
        )

        print(
            "Then run the script again."
        )

        print("=" * 70)

        raise


# ============================================================
# SAVE
# ============================================================

def save_dataframe(df):

    path = WORKING_FILE

    parent = path.parent

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_file_writable(path)

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            prefix=path.stem + "_save_",
            suffix=".xlsx",
            dir=parent,
            delete=False,
        ) as tmp:

            temp_path = Path(
                tmp.name
            )

        print(
            "\n    Saving workbook..."
        )

        df.to_excel(
            temp_path,
            index=False,
        )

        os.replace(
            temp_path,
            path,
        )

        temp_path = None

        print(
            "    Progress saved."
        )

    finally:

        if (
            temp_path
            and temp_path.exists()
        ):

            try:
                temp_path.unlink()

            except Exception:
                pass


# ============================================================
# LOAD WORKBOOK
# ============================================================

def load_workbook():

    print(
        "Loading new 23k annotation file..."
    )

    if not WORKING_FILE.exists():

        raise FileNotFoundError(
            f"Working file not found:\n"
            f"{WORKING_FILE}"
        )

    df = pd.read_excel(
        WORKING_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    if TEXT_COLUMN not in df.columns:

        raise ValueError(
            f"Required column "
            f"'{TEXT_COLUMN}' not found."
        )

    for column in [
        KEYWORDS_COLUMN,
        DOMAIN_COLUMN,
        TOPICS_COLUMN,
        NOTES_COLUMN,
        STATUS_COLUMN,
        PROVIDER_COLUMN,
    ]:

        if column not in df.columns:

            df[column] = ""

    # FIX: Excel/pandas can infer empty annotation columns as float64.
    # Convert them to object so later string assignments are always safe.
    for column in [
        KEYWORDS_COLUMN,
        DOMAIN_COLUMN,
        TOPICS_COLUMN,
        NOTES_COLUMN,
        STATUS_COLUMN,
        PROVIDER_COLUMN,
    ]:
        if column not in df.columns:
            df[column] = pd.Series(
                [""] * len(df),
                index=df.index,
                dtype="object",
            )
        else:
            df[column] = df[column].astype("object")

    return df


# ============================================================
# CREATE BATCHES
# ============================================================

def create_batches(
    df,
    pending_indices,
):

    batches = []

    current = []

    for index in pending_indices:

        text = normalize_spaces(
            str(
                df.at[
                    index,
                    TEXT_COLUMN,
                ]
            )
        )

        if not text:
            continue

        current.append(
            {
                "id": int(index),
                "text": text,
            }
        )

        if len(current) >= BATCH_SIZE:

            batches.append(current)

            current = []

    if current:
        batches.append(current)

    return batches


# ============================================================
# APPLY RESULT
# ============================================================

def apply_annotation(
    df,
    row_id,
    annotation,
    provider,
):

    df.at[
        row_id,
        KEYWORDS_COLUMN,
    ] = keywords_to_excel(
        annotation[
            KEYWORDS_COLUMN
        ]
    )

    df.at[
        row_id,
        DOMAIN_COLUMN,
    ] = annotation[
        DOMAIN_COLUMN
    ]

    df.at[
        row_id,
        TOPICS_COLUMN,
    ] = topics_to_excel(
        annotation[
            TOPICS_COLUMN
        ]
    )

    df.at[
        row_id,
        NOTES_COLUMN,
    ] = annotation[
        NOTES_COLUMN
    ]

    df.at[
        row_id,
        STATUS_COLUMN,
    ] = "Annotated"

    df.at[
        row_id,
        PROVIDER_COLUMN,
    ] = provider


# ============================================================
# PROCESS BATCHES CONCURRENTLY
# ============================================================

def process_batches(
    df,
    batches,
    groq_client,
    gemini_client,
):

    global GROQ_DAILY_EXHAUSTED
    global GEMINI_DAILY_EXHAUSTED

    successful = 0
    failed = 0

    groq_success = 0
    gemini_success = 0

    unsaved_successes = 0

    start_time = time.time()

    total_rows = sum(
        len(batch)
        for batch in batches
    )

    completed_rows = 0

    print()
    print("=" * 70)
    print("23K GOLD-CONSTRAINED TURBO PROCESSING")
    print("=" * 70)

    print(
        f"Batches          : {len(batches):,}"
    )

    print(
        f"Rows             : {total_rows:,}"
    )

    print(
        f"Batch size       : {BATCH_SIZE}"
    )

    print(
        f"Groq workers     : {GROQ_WORKERS}"
    )

    print(
        f"Gemini workers   : {GEMINI_WORKERS}"
    )

    print(
        f"Gold domains     : {len(ALLOWED_DOMAINS)}"
    )

    print("=" * 70)

    pending_batches = list(
        enumerate(
            batches,
            start=1,
        )
    )

    groq_executor = (
        ThreadPoolExecutor(
            max_workers=GROQ_WORKERS,
            thread_name_prefix="groq",
        )
        if groq_client
        else None
    )

    gemini_executor = (
        ThreadPoolExecutor(
            max_workers=GEMINI_WORKERS,
            thread_name_prefix="gemini",
        )
        if gemini_client
        else None
    )

    futures = {}

    try:

        # ====================================================
        # INITIAL SUBMISSION
        # ====================================================

        for batch_number, items in pending_batches:

            if (
                groq_executor
                and groq_is_available()
            ):

                future = (
                    groq_executor.submit(
                        groq_worker,
                        groq_client,
                        batch_number,
                        items,
                    )
                )

                futures[future] = (
                    batch_number,
                    "Groq",
                    items,
                )

            if (
                gemini_executor
                and gemini_is_available()
            ):

                future = (
                    gemini_executor.submit(
                        gemini_worker,
                        gemini_client,
                        batch_number,
                        items,
                    )
                )

                futures[future] = (
                    batch_number,
                    "Gemini",
                    items,
                )

        # ====================================================
        # PROCESS COMPLETED REQUESTS
        # ====================================================

        finished_batches = set()

        for future in as_completed(
            futures
        ):

            (
                batch_number,
                provider,
                items,
            ) = futures[future]

            try:

                result = future.result()

            except Exception as exc:

                result = {
                    "batch_number": batch_number,
                    "provider": provider,
                    "status": "FAILED",
                    "results": {},
                    "errors": {
                        item["id"]: str(exc)
                        for item in items
                    },
                }

            status = result[
                "status"
            ]

            # ------------------------------------------------
            # DAILY QUOTA
            # ------------------------------------------------

            if status == "DAILY_QUOTA":

                if provider == "Groq":

                    print(
                        "\n!!! GROQ DAILY "
                        "QUOTA EXHAUSTED !!!"
                    )

                    GROQ_DAILY_EXHAUSTED = True

                else:

                    print(
                        "\n!!! GEMINI DAILY "
                        "QUOTA EXHAUSTED !!!"
                    )

                    GEMINI_DAILY_EXHAUSTED = True

                continue

            # ------------------------------------------------
            # SUCCESSFUL RESULTS
            # ------------------------------------------------

            if status == "SUCCESS":

                results = result[
                    "results"
                ]

                if batch_number in finished_batches:
                    continue

                if results:

                    finished_batches.add(
                        batch_number
                    )

                    for (
                        row_id,
                        annotation
                    ) in results.items():

                        if row_id not in df.index:
                            continue

                        if row_is_annotated(
                            df.loc[row_id]
                        ):
                            continue

                        apply_annotation(
                            df,
                            row_id,
                            annotation,
                            provider,
                        )

                        successful += 1
                        unsaved_successes += 1
                        completed_rows += 1

                        if provider == "Groq":
                            groq_success += 1
                        else:
                            gemini_success += 1

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    rate = (
                        completed_rows / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        total_rows
                        - completed_rows
                    )

                    eta = (
                        remaining / rate
                        if rate > 0
                        else 0
                    )

                    print(
                        f"\rProgress: "
                        f"{completed_rows:,}/"
                        f"{total_rows:,} | "
                        f"{rate:.2f} rows/sec | "
                        f"ETA: "
                        f"{format_seconds(eta)} | "
                        f"Groq: {groq_success:,} | "
                        f"Gemini: {gemini_success:,}",
                        end="",
                        flush=True,
                    )

                    if (
                        unsaved_successes
                        >= SAVE_EVERY
                    ):

                        save_dataframe(df)

                        unsaved_successes = 0

                continue

            # ------------------------------------------------
            # FAILED
            # ------------------------------------------------

            if status == "FAILED":
                continue

        # ====================================================
        # DETERMINE STILL-PENDING ROWS
        # ====================================================

        still_pending = []

        for batch in batches:

            for item in batch:

                row_id = item["id"]

                if not row_is_annotated(
                    df.loc[row_id]
                ):

                    still_pending.append(
                        item
                    )

        # ====================================================
        # NO INDIVIDUAL FALLBACK
        # ====================================================
        # IMPORTANT: Never turn failed batch rows into one-by-one API
        # requests. Save the current progress and let the next run
        # retry the remaining rows in normal batches.

        if still_pending:
            print()
            print()
            print("=" * 70)
            print("BATCH RUN FINISHED - NO INDIVIDUAL RETRIES")
            print("=" * 70)
            print(f"Remaining rows: {len(still_pending):,}")
            print("Current annotations will be saved now.")
            print("Run the script again to continue remaining rows in batches.")
            print("=" * 70)

        if False and (
            still_pending
            and FALLBACK_TO_INDIVIDUAL
            and (
                groq_is_available()
                or gemini_is_available()
            )
        ):

            print()
            print()

            print("=" * 70)
            print(
                "RETRYING REMAINING ROWS INDIVIDUALLY"
            )
            print("=" * 70)

            print(
                f"Remaining rows: "
                f"{len(still_pending):,}"
            )

            print("=" * 70)

            fallback_futures = {}

            fallback_groq_executor = (
                ThreadPoolExecutor(
                    max_workers=GROQ_WORKERS,
                    thread_name_prefix="groq-single",
                )
                if (
                    groq_client
                    and groq_is_available()
                )
                else None
            )

            fallback_gemini_executor = (
                ThreadPoolExecutor(
                    max_workers=GEMINI_WORKERS,
                    thread_name_prefix="gemini-single",
                )
                if (
                    gemini_client
                    and gemini_is_available()
                )
                else None
            )

            try:

                for item in still_pending:

                    if (
                        fallback_groq_executor
                        and groq_is_available()
                    ):

                        future = (
                            fallback_groq_executor.submit(
                                annotate_single_groq,
                                groq_client,
                                item,
                            )
                        )

                        fallback_futures[
                            future
                        ] = (
                            item,
                            "Groq",
                        )

                    elif (
                        fallback_gemini_executor
                        and gemini_is_available()
                    ):

                        future = (
                            fallback_gemini_executor.submit(
                                annotate_single_gemini,
                                gemini_client,
                                item,
                            )
                        )

                        fallback_futures[
                            future
                        ] = (
                            item,
                            "Gemini",
                        )

                for future in as_completed(
                    fallback_futures
                ):

                    item, provider = (
                        fallback_futures[
                            future
                        ]
                    )

                    try:

                        result = future.result()

                    except Exception:

                        continue

                    results = result.get(
                        "results",
                        {},
                    )

                    for (
                        row_id,
                        annotation
                    ) in results.items():

                        if row_id not in df.index:
                            continue

                        if row_is_annotated(
                            df.loc[row_id]
                        ):
                            continue

                        apply_annotation(
                            df,
                            row_id,
                            annotation,
                            provider,
                        )

                        successful += 1
                        unsaved_successes += 1
                        completed_rows += 1

                        if provider == "Groq":
                            groq_success += 1
                        else:
                            gemini_success += 1

                        if (
                            unsaved_successes
                            >= SAVE_EVERY
                        ):

                            save_dataframe(df)

                            unsaved_successes = 0

            finally:

                if fallback_groq_executor:

                    fallback_groq_executor.shutdown(
                        wait=True
                    )

                if fallback_gemini_executor:

                    fallback_gemini_executor.shutdown(
                        wait=True
                    )

        # ====================================================
        # COUNT FINAL FAILURES
        # ====================================================

        for batch in batches:

            for item in batch:

                row_id = item["id"]

                if not row_is_annotated(
                    df.loc[row_id]
                ):

                    failed += 1

    finally:

        if groq_executor:

            groq_executor.shutdown(
                wait=True
            )

        if gemini_executor:

            gemini_executor.shutdown(
                wait=True
            )

    return (
        successful,
        failed,
        groq_success,
        gemini_success,
        unsaved_successes,
    )


# ============================================================
# FORMAT TIME
# ============================================================

def format_seconds(seconds):

    if seconds <= 0:
        return "--"

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs}s"
        )

    if minutes > 0:

        return (
            f"{minutes}m "
            f"{secs}s"
        )

    return f"{secs}s"


# ============================================================
# MAIN
# ============================================================

def main():

    global GROQ_DAILY_EXHAUSTED
    global GEMINI_DAILY_EXHAUSTED
    global ALLOWED_DOMAINS

    print()
    print("=" * 70)
    print("AMHARIC 23K ANNOTATION")
    print("25K GOLD DOMAIN-CONSTRAINED VERSION")
    print("=" * 70)

    print()
    print(
        f"Gold file    : {GOLD_FILE}"
    )

    print(
        f"Working file : {WORKING_FILE}"
    )

    print(
        f"Groq model   : {GROQ_MODEL}"
    )

    print(
        f"Gemini model : {GEMINI_MODEL}"
    )

    print()

    print("=" * 70)
    print("PERFORMANCE CONFIGURATION")
    print("=" * 70)

    print(
        f"Batch size       : {BATCH_SIZE}"
    )

    print(
        f"Groq workers     : {GROQ_WORKERS}"
    )

    print(
        f"Gemini workers   : {GEMINI_WORKERS}"
    )

    print(
        f"Save every       : {SAVE_EVERY}"
    )

    print(
        f"Max retries      : {MAX_RETRIES}"
    )

    print(
        f"Groq interval    : {GROQ_MIN_INTERVAL}s"
    )

    print(
        f"Gemini interval  : {GEMINI_MIN_INTERVAL}s"
    )

    print("Individual fallback: DISABLED (save + stop; resume next run)")

    print("=" * 70)

    # ========================================================
    # LOAD GOLD DATASET
    # ========================================================

    if not GOLD_FILE.exists():

        raise FileNotFoundError(
            f"25k gold dataset not found:\n"
            f"{GOLD_FILE}"
        )

    gold_df = pd.read_excel(
        GOLD_FILE
    )

    print()
    print(
        f"Loaded GOLD dataset: "
        f"{len(gold_df):,} rows"
    )

    show_gold_domain_distribution(
        gold_df
    )

    # ========================================================
    # EXTRACT GOLD DOMAINS
    # ========================================================

    ALLOWED_DOMAINS = load_gold_domains()

    # ========================================================
    # API KEYS
    # ========================================================

    groq_key = os.getenv(
        "GROQ_API_KEY"
    )

    gemini_key = os.getenv(
        "GEMINI_API_KEY"
    )

    print()
    print("=" * 70)
    print("API CONFIGURATION")
    print("=" * 70)

    print(
        "Groq API key   : "
        + (
            "SET"
            if groq_key
            else "NOT SET"
        )
    )

    print(
        "Gemini API key : "
        + (
            "SET"
            if gemini_key
            else "NOT SET"
        )
    )

    print("=" * 70)

    if not groq_key and not gemini_key:

        raise RuntimeError(
            "Neither API key is configured."
        )

    # ========================================================
    # CHECK WORKING FILE
    # ========================================================

    print()
    print(
        "Checking 23k workbook..."
    )

    check_file_writable(
        WORKING_FILE
    )

    print(
        "23k workbook is accessible."
    )

    # ========================================================
    # CLIENTS
    # ========================================================

    groq_client = None
    gemini_client = None

    if groq_key:

        groq_client = Groq(
            api_key=groq_key.strip()
        )

    if gemini_key:

        gemini_client = genai.Client(
            api_key=gemini_key.strip()
        )

    # ========================================================
    # LOAD 23K WORKBOOK
    # ========================================================

    df = load_workbook()

    # ========================================================
    # NORMALIZE STATUS
    # ========================================================

    annotated, pending, blank = (
        normalize_statuses(df)
    )

    print()
    print("=" * 70)
    print("CURRENT 23K FILE STATUS")
    print("=" * 70)

    print(
        f"Total rows : {len(df):,}"
    )

    print(
        f"Annotated  : {annotated:,}"
    )

    print(
        f"Pending    : {pending:,}"
    )

    print(
        f"Blank text : {blank:,}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE NORMALIZED STATUS
    # ========================================================

    print()
    print(
        "Saving normalized statuses..."
    )

    save_dataframe(df)

    # ========================================================
    # FIND PENDING ROWS
    # ========================================================

    pending_indices = []

    for index in df.index:

        text = df.at[
            index,
            TEXT_COLUMN,
        ]

        if not is_nonempty(text):
            continue

        if row_is_annotated(
            df.loc[index]
        ):
            continue

        pending_indices.append(
            index
        )

    print()
    print("=" * 70)
    print("RESUME PLAN")
    print("=" * 70)

    print(
        f"Rows to process : "
        f"{len(pending_indices):,}"
    )

    print(
        f"Rows skipped    : "
        f"{len(df) - len(pending_indices):,}"
    )

    print(
        f"Gold domains    : "
        f"{len(ALLOWED_DOMAINS)}"
    )

    print("=" * 70)

    if not pending_indices:

        print()
        print(
            "Nothing is pending."
        )

        return

    # ========================================================
    # CREATE BATCHES
    # ========================================================

    batches = create_batches(
        df,
        pending_indices,
    )

    print()
    print(
        f"Created {len(batches):,} "
        f"API batches."
    )

    # ========================================================
    # PROCESS
    # ========================================================

    (
        successful,
        failed,
        groq_success,
        gemini_success,
        unsaved_successes,
    ) = process_batches(
        df,
        batches,
        groq_client,
        gemini_client,
    )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    if unsaved_successes > 0:

        save_dataframe(df)

    # ========================================================
    # FINAL STATUS
    # ========================================================

    annotated, pending, blank = (
        normalize_statuses(df)
    )

    save_dataframe(df)

    # ========================================================
    # FINAL DOMAIN DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("23K DATASET DOMAIN DISTRIBUTION")
    print("=" * 70)

    if DOMAIN_COLUMN in df.columns:

        counts = (
            df[DOMAIN_COLUMN]
            .dropna()
            .astype(str)
            .map(normalize_spaces)
            .value_counts()
        )

        total = counts.sum()

        for domain, count in counts.items():

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            print(
                f"{domain:<30} "
                f"{count:>7,} "
                f"({percentage:>6.2f}%)"
            )

    print("=" * 70)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Successful this run : "
        f"{successful:,}"
    )

    print(
        f"Failed this run    : "
        f"{failed:,}"
    )

    print()

    print(
        f"Groq successful    : "
        f"{groq_success:,}"
    )

    print(
        f"Gemini successful  : "
        f"{gemini_success:,}"
    )

    print()

    print(
        f"Total Annotated    : "
        f"{annotated:,}"
    )

    print(
        f"Total Pending      : "
        f"{pending:,}"
    )

    print(
        f"Blank text         : "
        f"{blank:,}"
    )

    print()

    print(
        "Gold domain restriction:"
    )

    print(
        "  ALL output domains MUST "
        "belong to the 25k gold dataset."
    )

    print()

    print(
        "Quota state:"
    )

    print(
        "  Groq   : "
        + (
            "DAILY LIMIT REACHED"
            if GROQ_DAILY_EXHAUSTED
            else "AVAILABLE"
        )
    )

    print(
        "  Gemini : "
        + (
            "DAILY LIMIT REACHED"
            if GEMINI_DAILY_EXHAUSTED
            else "AVAILABLE"
        )
    )

    print()

    print(
        f"Gold file: "
        f"{GOLD_FILE}"
    )

    print(
        f"Working file: "
        f"{WORKING_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("INTERRUPTED")
        print("=" * 70)

        print(
            "The process was interrupted."
        )

        print(
            "Previously checkpointed "
            "progress is safe."
        )

        print("=" * 70)

    except PermissionError:

        print()
        print("=" * 70)
        print("PERMISSION ERROR")
        print("=" * 70)

        print(
            "The Excel workbook is probably open."
        )

        print(
            f"Close:\n{WORKING_FILE}"
        )

        print(
            "Then run the script again."
        )

        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("UNEXPECTED ERROR")
        print("=" * 70)

        print(
            exc
        )

        traceback.print_exc()

        print("=" * 70)