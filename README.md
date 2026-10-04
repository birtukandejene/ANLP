# Amharic Archive: Project Documentation

Amharic Archive is a Flask-based workbench for analyzing Amharic documents. It combines an open-topic semantic topic system, a supervised subtopic component, semantic keyword extraction, a language gate, persistent analysis history, and feedback-aware ranking.

This document describes the project from corpus preparation through evaluation, inference, feedback, and deployment. It separates steps supported by checked-in scripts from steps whose original execution history or training code is not present in this repository.

## Contents

- [Project at a glance](#project-at-a-glance)
- [End-to-end workflow](#end-to-end-workflow)
- [Runtime architecture](#runtime-architecture)
- [Repository map](#repository-map)
- [Setup and local use](#setup-and-local-use)
- [API reference](#api-reference)
- [Tests and evaluation](#tests-and-evaluation)
- [Deployment](#deployment)
- [Limitations and reproducibility](#limitations-and-reproducibility)

## Project at a glance

The repository contains data preparation utilities, annotation and balancing tools, prepared train/validation/test files, analysis and evaluation scripts, model artifacts, and the Flask application.

At runtime, a user submits one document. The application checks whether it is predominantly Ethiopic-script text, obtains an embedding, retrieves relevant saved feedback, predicts an open-set main topic and related subtopics, extracts up to 10 keywords, stores valid analyses in SQLite, and displays the result in the browser. Feedback can influence later similar analyses, but does not retrain the stored models.

### Main runtime components

- **Web application:** Flask application factory in `app/`, browser UI in `templates/` and `static/`.
- **Language gate:** a lightweight Ethiopic/Latin word-ratio detector.
- **Topic and subtopic inference:** `notebooks/27_amharic_semantic_hybrid_inference.py`, the semantic index, and the domain-aware subtopic classifier.
- **Keyword extraction:** `notebooks/27_keyword_extraction.py`, the local Rasyosef embedding model, stopwords, and a lexicon built from annotated training keywords.
- **Persistence and feedback:** SQLite analysis history and labeled prediction feedback; relevant feedback is retrieved by embedding similarity during future analysis.

## End-to-end workflow

The following describes the lifecycle from source text to deployed inference. The repository contains working artifacts and some steps, but not a complete automated pipeline that rebuilds every artifact from raw data.

### 1. Source corpus and raw files

Raw source data belongs under `data/raw/`. The current workspace includes a resegmented JSONL corpus and `amharic_stopwords.txt`. The JSONL utility expects records with a `text` field and can preserve other record metadata.

The README does not claim the checked-in raw files are the complete source corpus: sources, collection criteria, licenses, and provenance should be documented alongside the data by the project owner before redistribution.

### 2. Cleaning and text normalization

`src/preprocessing.py` implements optional processing operations:

1. Normalize selected Ethiopic character variants.
2. Expand a fixed list of common Amharic abbreviations.
3. Remove URLs, HTML tags, selected punctuation/noise, and repeated whitespace.
4. Optionally split on configured Amharic and Latin punctuation boundaries.

`src/reprocess_existing_data.py` reads a JSONL file in chunks, applies normalization, abbreviation expansion, and cleaning without sentence segmentation, and writes a timestamped JSONL file plus a processing report under `data/cleaned/`. Run it from the repository root:

```powershell
python run_preprocess.py
```

The script selects the first `*.jsonl` file returned from `data/raw/`; it does not currently provide an input-file command-line option. The function signature has an `input_file` parameter, but its current implementation overwrites that value with the first discovered file, so do not rely on passing a custom input path without fixing the implementation. Processing is row-preserving: it writes `processed_text` and retains metadata rather than creating the Excel train/validation/test splits.

### 3. Annotation and dataset balancing

Annotation workbooks and scripts are under `annotated/` and `data/annotation/`. They support filling labels, applying spreadsheet validation, balancing/downsampling, and comparing dataset variants. Existing workbooks include domain/topic and keyword labels; exact schemas vary by stage. The annotation utility in `annotated/annotation.py` uses external Groq and Gemini services (`GROQ_API_KEY` and `GEMINI_API_KEY`), so running it requires provider credentials, network access, and quota. Treat machine-proposed labels as annotation candidates and review them against the project label definitions; API output is not ground truth by itself.

Balancing utilities may expand selected domains, deduplicate additions by IDs, or downsample each domain to a configured maximum. For example, `annotated/balance.py` and `annotated/downsampling.py` contain hard-coded local Windows paths and constants, so inspect and edit those settings before running them on another machine. Preserve an untouched source copy and record the chosen workbook, label definitions, balancing method/seed, and any manual changes. These scripts are project utilities rather than one unified pipeline.

### 4. Prepared splits

The runtime and evaluation code expect Excel splits in `data/train/`:

- `train_processed.xlsx`
- `validation_processed.xlsx`
- `test_processed.xlsx`

The checked-in semantic-index builder uses `text`, `domain`, and `topics` columns in the training workbook. The keyword extractor uses `text` and `keywords`. Topic and keyword evaluation scripts have their own required columns and parsing rules. Verify the headers and label formats against the selected script before running it.

Prepared CSVs also exist under `data/prepared/`, but the deployed inference path currently reads the Excel training split for its keyword annotation lexicon. Dataset split creation is not orchestrated by a single checked-in command. Before treating any split as an evaluation set, remove duplicate/near-duplicate documents across splits, split by source or document group where appropriate, freeze the split and random seed, and retain a manifest of counts and label versions. The repository includes `notebooks/29_analyze_amharic_dataset.py` for distribution, rare/unseen-topic, and train/validation-overlap reports. Run this audit before fitting or comparing models; do not tune on the final test split.

### 5. Build the semantic topic index

`notebooks/26_build_amharic_semantic_index.py` encodes the training texts with `rasyosef/embedding-amharic-base` and writes `models/amharic_semantic_index.joblib`. It stores document texts, domains, topics, normalized embeddings, and domain/topic centroids and counts. From the repository root:

```powershell
python notebooks/26_build_amharic_semantic_index.py
```

The script accepts `ANLP_INDEX_TRAIN_PATH` and `ANLP_INDEX_OUTPUT_PATH` environment variables. It may download model files unless they are already cached locally. Build from the final intended training split; rebuilding overwrites the configured output index.

### 6. Train or supply the supervised subtopic model

The runtime also requires `models/final_domain_aware_subtopic_model.joblib`. The hybrid inference script combines its predictions with semantic neighbor evidence, topic centroids, co-occurrence, and feedback evidence. This repository does not provide a verified end-to-end trainer for that exact final artifact. `src/train_model.py` is empty; scripts under `train/` are separate/legacy modeling experiments and should not be assumed to reproduce the shipped model.

If rebuilding this artifact, recover and document its exact training code, feature schema, labels, and dependency versions. Ensure it matches the feature construction and expected estimator interface in the hybrid inference script.

### 7. Evaluate candidate models and artifacts

The repository contains evaluation and analysis scripts under `notebooks/`, including dataset analysis, keyword validation, hybrid retrieval evaluation, and semantic validation. A few examples:

```powershell
python notebooks/28_amharic_semantic_validation_evaluation.py --limit 100
python notebooks/30_amharic_document_retrieval_evaluation.py
python notebooks/30_hybrid_keyword_validation.py
python notebooks/29_analyze_amharic_dataset.py
```

These scripts may install missing Python packages themselves, load large transformer models, and write reports beneath `results/`. Check each script's header/configuration for the expected columns, environment variables, output paths, and metrics. Validation results are only meaningful for the exact split and artifacts used; they are not a substitute for a documented held-out test evaluation. `notebooks/30_hybrid_keyword_validation.py` evaluates an alternative hybrid retrieval/lexical keyword path; it is not the keyword extractor wired into the Flask adapter.

### 8. Run interactive inference

The Flask application factory in `app/__init__.py` initializes SQLite and registers routes. `app/services/nlp_service.py` coordinates language detection, embeddings, feedback retrieval, topic inference, and keyword extraction. Model adapters in `app/services/` load the existing inference scripts lazily.

On the first valid request, the topic backend loads the index, subtopic model, and embedding model; the keyword backend loads its stopwords, training split, annotation lexicon, and embedding model. Keep the required files accessible from the project root.

For each accepted document:

1. Enforce the maximum input length and run the Amharic language gate.
2. Embed the normalized text with the topic backend.
3. Retrieve up to 200 similar saved feedback records with cosine similarity at or above 0.68.
4. Combine topic-index, centroid, supervised subtopic, co-occurrence, and relevant feedback evidence.
5. Rank keywords using semantic relevance, phrase rules, stopword filtering, the train-label frequency prior, and relevant feedback.
6. Save the text, result, embedding, and timestamp to SQLite.

Only valid analyses are saved. Feedback changes the evidence used for later similar inputs; it does not alter model weights or rebuild the semantic index.

### 9. Review and provide feedback

In the workbench, users can review the predicted main topic, subtopics, and keywords, mark items as correct or needing correction, and save the resulting labels. Feedback is stored in SQLite with the source analysis and embedding. The current progress endpoint reports counts for main-topic, subtopic, and keyword feedback; its `ready` value means each type has at least three stored labels, not that a retraining pipeline has run.

### 10. Optional progressive-learning candidate workflow

`notebooks/31_progressive_learning.py` implements a separate workflow for importing dashboard feedback, reviewing it, staging approved examples, building a candidate semantic index, evaluating that candidate, and gating it against a baseline. It does not run automatically when dashboard feedback is saved. Passing the gate does not automatically replace the live index or train the missing supervised subtopic artifact. Explore its commands with:

```powershell
python notebooks/31_progressive_learning.py --help
```

The operational sequence is:

1. Initialize the separate learner database with `init`.
2. Import records from the dashboard database with `import-dashboard`.
3. Inspect pending feedback and have a reviewer approve or reject each item.
4. Stage approved examples into a new workbook, keeping trusted base data unchanged.
5. Build a candidate index to a separate output path and evaluate it against a fixed validation split.
6. Compare candidate and baseline metric reports with `evaluate-gate`; only a passing report can be marked complete.
7. Have an operator review, version, and deploy an approved artifact separately. Keep the previous artifact for rollback.

Example commands (adjust paths for the local data and reports):

```powershell
python notebooks/31_progressive_learning.py init
python notebooks/31_progressive_learning.py import-dashboard --dashboard-db data/dashboard.sqlite3
python notebooks/31_progressive_learning.py pending
python notebooks/31_progressive_learning.py approve <feedback-id>
# Or reject an item that fails review:
python notebooks/31_progressive_learning.py reject <feedback-id> --reason "Label is unsupported by the source text."
python notebooks/31_progressive_learning.py stage-data --base data/train/train_processed.xlsx --output data/candidate/train_processed.xlsx
python notebooks/31_progressive_learning.py build-candidate-index --train data/candidate/train_processed.xlsx --output-index models/candidate/amharic_semantic_index.joblib
python notebooks/31_progressive_learning.py evaluate-candidate --index models/candidate/amharic_semantic_index.joblib --train data/candidate/train_processed.xlsx --validation data/train/validation_processed.xlsx --results results/candidate
python notebooks/31_progressive_learning.py evaluate-gate --baseline <baseline-metrics.json> --candidate results/candidate/document_retrieval_metrics.json --metric <metric-path> --minimum-documents 100 --output results/candidate/gate.json
python notebooks/31_progressive_learning.py complete-run --evaluation results/candidate/gate.json
```

Replace angle-bracket values with real values: `pending` prints the learner feedback IDs, `--baseline` must point to the baseline report from the same evaluation protocol, and `--metric` must name a numeric path present in both JSON reports. `evaluate-gate` requires at least 100 evaluated documents by default and writes `passed: true` only when the candidate does not regress on the chosen metric. `complete-run` refuses to mark examples used when the gate fails. Create candidate output directories as needed, and never point a candidate build at the live production artifact path.

### 11. Deploy and monitor

Railway configuration builds dependencies from `requirements-deploy.txt` and runs Gunicorn with one worker. Production must include all runtime model/data artifacts, configure a persistent database location, and allocate enough memory for the transformer models. See [Deployment](#deployment).

## Runtime architecture

```text
Browser
  |  GET /workbench; POST /api/analyze
  v
Flask routes (app/routes.py)
  v
NLPService (app/services/nlp_service.py)
  +--> AmharicLanguageDetector
  +--> HybridTopicExtractor
  |      +--> semantic index + embedding model
  |      +--> supervised subtopic model
  +--> FeedbackEvidenceStore (SQLite, embedding similarity)
  +--> ExistingKeywordExtractor
         +--> Rasyosef embeddings + stopwords + train keyword lexicon
  v
SQLite analysis/feedback records --> JSON response --> browser UI
```

Main-topic selection is open-topic: candidates come from topics in the semantic index rather than being forced into a fixed 19-domain list. The optional domain classifier artifact is not required by the Flask adapter's required-file check. Topic and keyword scores are ranking/evidence scores, not calibrated probabilities.

### Component responsibilities

- `app/routes.py` validates request shape and length, maps HTTP paths, calls the service, persists valid analyses, and translates backend exceptions to HTTP responses.
- `app/services/nlp_service.py` sequences language detection, topic embedding, feedback retrieval, topic/subtopic inference, and keyword extraction.
- `app/services/topic_extractor.py` lazy-loads the hybrid inference module and required joblib artifacts. The topic embedding is also stored with a valid analysis for later feedback matching.
- `app/services/feedback_evidence.py` normalizes embeddings, computes cosine similarity, keeps up to 200 relevant records above threshold, and aggregates positive/negative labels by task type.
- `app/services/keyword_extractor.py` lazy-loads the keyword module, stopwords, and training keyword labels, builds a frequency prior, and passes relevant keyword feedback into ranking.
- `app/database.py` initializes/migrates SQLite tables and stores analysis/feedback rows. The app uses SQLite directly; it does not provide multi-user tenancy or a separate database-server abstraction.

The browser handles text entry, output review/editing, analysis history, and feedback submission. Persistent database state is separate from model artifacts: deleting the database resets stored history/feedback, while replacing an index or model changes inference behavior.

## Repository map

| Path | Purpose |
| --- | --- |
| `app/` | Flask factory, config, database, routes, detector, and model adapters/services |
| `templates/`, `static/` | Welcome, workbench, About pages, styles, scripts, and assets |
| `src/` | JSONL loading, preprocessing, evaluation, and legacy/project utilities |
| `annotated/` | Annotation, balancing, downsampling, and comparison utilities |
| `data/raw/` | Raw/resegmented source files and stopwords |
| `data/cleaned/` | Timestamped cleaned JSONL and processing reports |
| `data/annotation/` | Annotation and balancing workbooks/reports |
| `data/prepared/` | Prepared CSV variants |
| `data/train/` | Processed train, validation, and test workbooks consumed by runtime/evaluation scripts |
| `notebooks/` | Index builder, inference backends, evaluation, analysis, and progressive-learning utilities |
| `models/` | Serialized model/index artifacts and local embedding model |
| `results/`, `reports/`, `outputs/` | Generated evaluations, analyses, and experiment outputs |
| `tests/` | Focused tests for feedback retrieval, history, keyword behavior, and progressive learning |
| `dashboard.py` | Local Flask development entry point |
| `wsgi.py` | WSGI application entry point |
| `railway.json` | Railway build/deploy configuration |

## Setup and local use

### Requirements

- Python 3.11 is the currently configured workspace interpreter. The repository does not declare a formal supported Python range.
- Pip dependencies from `requirements.txt`.
- Runtime artifacts listed under [Runtime artifacts](#runtime-artifacts).
- Sufficient disk space, RAM, and (for uncached model downloads) network access for transformer dependencies.

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To run the app:

```powershell
python dashboard.py
```

Open `http://127.0.0.1:5000`. The root page is the welcome page; use **Get Started** or open `/workbench` for analysis. `/about` provides the user-facing workflow guide.

Set `SECRET_KEY` to a private random value outside local development. `DATABASE_PATH` can point to a writable SQLite file; by default it is `data/dashboard.sqlite3`.

### Runtime artifacts

Required by the current app:

- `notebooks/27_amharic_semantic_hybrid_inference.py`
- `models/amharic_semantic_index.joblib`
- `models/final_domain_aware_subtopic_model.joblib`
- `models/embedding-amharic-base/` (or a usable cached model location)
- `notebooks/27_keyword_extraction.py`
- `data/raw/amharic_stopwords.txt`
- `data/train/train_processed.xlsx`

The topic backend and keyword backend both use the local `rasyosef/embedding-amharic-base` model. The keyword adapter builds its annotation-frequency lexicon from the training workbook at first use. Missing files or incompatible artifact schemas cause analysis to return an error rather than fabricated predictions.

### Configuration

Defaults in `app/config.py`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `KEYWORD_TOP_K` | 10 | Requested maximum keyword count; the extractor can return fewer based on candidate availability and text length |
| `TOPIC_TOP_K` | 5 | Maximum candidate/subtopic count passed to the adapter |
| `FEEDBACK_SIMILARITY_THRESHOLD` | 0.68 | Minimum cosine similarity for feedback retrieval |
| `MAX_INPUT_CHARS` | 12,000 | Maximum accepted request text length |
| `DATABASE_PATH` | `data/dashboard.sqlite3` | SQLite database path; environment-overridable |
| `SECRET_KEY` | development-only value | Flask secret; set securely in deployment |

The index-builder and evaluation scripts also accept selected `ANLP_*` path environment variables; see their source configuration for the exact variables.

## API reference

All API request and response bodies are JSON.

### `GET /`

Returns the welcome page.

### `GET /workbench`

Returns the analysis workbench.

### `GET /about`

Returns the user-facing About and feedback guide.

### `POST /api/analyze`

Request:

```json
{
  "text": "የኢትዮጵያ ኢኮኖሚ በ2025 ዓ.ም እድገት አሳይቷል።"
}
```

A valid response includes `analysis_id`, `language`, `valid`, `main_topic`, `subtopics`, `topic_candidates`, `topic_confidence`, `keywords`, `feedback_evidence_count`, and backend identifiers. Scores are task-specific ranking evidence, not probabilities. If text is non-Amharic, the endpoint returns HTTP 200 with `valid: false` and an error message; this is a language rejection, not a server error.

Status codes:

- `200`: valid prediction or language rejection.
- `400`: missing text or text longer than 12,000 characters.
- `503`: inference backend/artifact failure. The response includes a general error and a diagnostic `detail`; do not expose diagnostic details to untrusted clients in a hardened public deployment.

### `GET /api/history`

Returns up to the 20 most recently stored analyses as `{"items": [...]}`. The endpoint does not currently paginate or filter by user.

### `DELETE /api/history/<analysis_id>`

Deletes the analysis row and returns `{"deleted": true, "analysis_id": ...}`. Returns `404` when the ID is not present. Feedback rows are not configured with database-level cascading deletion.

### `POST /api/feedback`

Saves one or more reviewed predictions associated with an existing analysis:

```json
{
  "analysis_id": 12,
  "feedback": [
    {
      "item_type": "main_topic",
      "item_text": "የተተነበየው ርዕስ",
      "corrected_text": "የተስተካከለው ርዕስ",
      "rating": 1
    }
  ]
}
```

`item_type` must be `main_topic`, `subtopic`, or `keyword`; `rating` must be `1` (accepted/corrected) or `-1` (rejected). The `feedback` list must be non-empty and `analysis_id` must exist. The UI may send an additional `keywords` field; the route currently ignores it. Success returns `saved`, count, and progress counts/thresholds.

### `GET /api/feedback/progress`

Returns the number of stored feedback items by supported type, the current threshold of three for each type, and whether those count thresholds are met. This is a collection-progress indicator only.

## Tests and evaluation

Run the automated tests from the repository root:

```powershell
python -m pytest
```

Tests cover focused behaviors such as similarity-filtered feedback retrieval, feedback-aware inference wiring, keyword candidate rules, history behavior, and progressive-learning utilities. They do not by themselves validate the quality of the shipped model artifacts or the full production deployment.

Evaluation scripts write generated reports and predictions under `results/` (and, depending on the script, other configured output directories). Review each script's CLI/help and required input columns before running it. Preserve the exact split and artifact versions associated with reported metrics.

## Deployment

`railway.json` configures a Nixpacks build using `requirements-deploy.txt`, then starts:

```text
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 wsgi:app
```

Before deployment:

1. Ensure the source/build has every file listed under [Runtime artifacts](#runtime-artifacts), including the serialized index/classifier, embedding model, stopwords, and training workbook. Large model files may exceed common repository limits; use an artifact store and download/checksum them during build if they are not committed.
2. Configure a persistent Railway volume mounted at `/data` and set `DATABASE_PATH=/data/dashboard.sqlite3` so analyses and feedback survive redeploys.
3. Set a strong `SECRET_KEY` using the platform's secret-variable mechanism.
4. Allow enough RAM and disk for model loading and inference. The configured single Gunicorn worker reduces duplicate model memory use; the `600` second timeout accommodates cold loading but does not make inference cheap.
5. Verify `/` health checks, then test an Amharic analysis, history persistence after restart, feedback submission, and missing-artifact behavior in the deployed service.

`requirements-deploy.txt` omits development-only tools such as Jupyter and includes Gunicorn. Keep deployment dependencies aligned with the Python/runtime versions supported by the hosting image and the serialized artifacts.

## Limitations and reproducibility

- **No single full rebuild command:** raw collection, cleaning, annotation, balancing, split generation, model training, evaluation, and packaging are not joined into one reproducible pipeline.
- **Final subtopic model provenance:** the app requires `final_domain_aware_subtopic_model.joblib`, but this repository does not contain a verified trainer for that exact artifact. `src/train_model.py` is empty and `train/` contains separate experiments.
- **Dataset provenance and licensing:** the repository does not establish complete source attribution, usage rights, or collection methodology for every corpus and annotation workbook. Confirm these before publication or redistribution.
- **Schema variation:** workbooks differ by stage and script. Text, domain, topics, and keyword columns/serialization must match the particular builder or evaluator.
- **Language detection is heuristic:** acceptance is based on an Ethiopic-word presence and 55% Ethiopic word ratio. It is not a full language identification system and is not designed to distinguish Amharic from every other Ethiopic-script language.
- **Ranking scores are not calibrated confidence:** semantic similarity, combined scores, annotation priors, and confidence labels should not be interpreted as statistically calibrated probabilities.
- **Single-document inference:** the current API analyzes one text at a time, has a 12,000-character limit, and has no batch API, authentication, per-user history, or rate limiting.
- **Two feedback paths are separate:** the Flask API stores feedback and uses similar records immediately as retrieval-time evidence. The progressive-learning CLI is a separate import/review/stage/evaluate workflow. There is no automatic connection that imports, approves, trains, passes a gate, or deploys dashboard feedback. Neither path retrains model weights on each request.
- **Feedback and history privacy:** submitted text and feedback are persisted as plain SQLite data. There is no built-in retention schedule, access-control layer, encryption-at-rest management, or user separation. Deploy only with an appropriate privacy and security policy.
- **Operational model costs:** cold starts load sizeable transformer artifacts; memory, disk, startup time, and CPU/GPU availability affect latency. One worker limits duplicate memory but limits request concurrency.
- **Artifact compatibility:** joblib/pickle artifacts depend on compatible Python and library versions and should only be loaded from trusted sources.
- **Legacy and experimental files:** some scripts are standalone experiments, hard-code local Windows paths, or install packages at runtime. They are not necessarily production-ready or part of the active Flask request path.
- **Preprocessing input parameter:** `reprocess_existing_data` currently ignores a caller-supplied `input_file` because it unconditionally selects the first raw JSONL path. Its wrapper also relies on the repository root as the working directory.
- **External annotation dependencies:** the annotation utility requires provider API keys, network connectivity, available quota, and human quality control. Keep credentials out of source control and do not treat model-generated labels as verified without review.
- **Progressive learning promotes only an index candidate:** the checked-in candidate workflow rebuilds/evaluates the semantic index, but does not reproduce final supervised subtopic-model training or perform live artifact promotion.

## License and data notice

No project license or complete dataset-rights statement is established by this README. Add the appropriate software license and data/model attribution before distributing the application or its bundled corpora and artifacts.