import re
import ast
import pandas as pd
import numpy as np

# ============================================================
# CONFIG - EDIT THIS SECTION
# ============================================================

TRAIN_PATH      = r"C:\Users\user\Desktop\anlp\data\train\train_processed.xlsx"
TEST_PATH       = r"C:\Users\user\Desktop\anlp\data\train\test_processed.xlsx"
VALIDATION_PATH = r"C:\Users\user\Desktop\anlp\data\train\validation_processed.xlsx"

# Column names in your xlsx files - CHANGE THESE if different
TEXT_COL     = "text"        # the raw/clean Amharic document text
KEYWORDS_COL = "keywords"    # ground-truth keyphrases (see KEYWORDS_SEP below)
TOPIC_COL    = "topic"       # ground-truth topic tag (optional, used for reference only)

# How ground-truth keywords are stored in the KEYWORDS_COL cell.
# "sep"  -> a delimited string, e.g. "ትምህርት; መንግስት; በጀት"  (set KEYWORDS_SEP)
# "list" -> a Python-list-looking string, e.g. "['ትምህርት', 'መንግስት']"
KEYWORDS_FORMAT = "sep"
KEYWORDS_SEP = ";"

# Embedding model used for KeyBERT + BERTopic.
# Multilingual, works reasonably on Amharic out of the box:
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Alternative worth A/B testing (Africa-focused, may do better on Amharic):
# EMBEDDING_MODEL = "Davlan/afro-xlmr-base"

TOP_K = 10          # how many keywords/keyphrases to extract per document
NGRAM_RANGE = (1, 3)
N_LDA_TOPICS = 10    # tune this - try a few values and compare coherence
N_DOCS_TO_EVAL = None  # set an int (e.g. 50) to quickly test on a subset first

# Minimal starter Amharic stopword list - EXPAND THIS, it is not exhaustive.
AMHARIC_STOPWORDS = {
    "የ", "እና", "በ", "ለ", "ላይ", "ነው", "ናቸው", "ውስጥ", "ከ", "እንደ",
    "ግን", "አለ", "ነበር", "ስለ", "እስከ", "ይህ", "እነዚህ", "ሁሉ", "ጋር", "አንድ",
    "ወይም", "ነገር", "ግዜ", "እንዲሁም", "ደግሞ", "ብቻ", "እንጂ", "ማለት",
}


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_excel(path)
    missing = [c for c in [TEXT_COL, KEYWORDS_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in {path}. "
            f"Available columns: {list(df.columns)}. "
            f"Update TEXT_COL/KEYWORDS_COL in CONFIG."
        )
    df = df.dropna(subset=[TEXT_COL, KEYWORDS_COL]).reset_index(drop=True)
    return df


def parse_gt_keywords(cell):
    if KEYWORDS_FORMAT == "list":
        try:
            kws = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            kws = [cell]
    else:
        kws = str(cell).split(KEYWORDS_SEP)
    return [k.strip() for k in kws if k.strip()]


def basic_clean(text):
    text = str(text)
    text = re.sub(r"[፡።፣፤፥፦፧፨\.\,\!\?\(\)\[\]\"'“”]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. KEYWORD EXTRACTION - KeyBERT
# ============================================================

def run_keybert(texts, model):
    from keybert import KeyBERT
    kb = KeyBERT(model=model)
    all_keywords = []
    for doc in texts:
        cleaned = basic_clean(doc)
        try:
            kws = kb.extract_keywords(
                cleaned,
                keyphrase_ngram_range=NGRAM_RANGE,
                stop_words=None,           # sklearn stopword lists don't cover Amharic
                use_mmr=True,
                diversity=0.5,
                top_n=TOP_K,
            )
            kws = [w for w, _ in kws if w not in AMHARIC_STOPWORDS]
        except Exception as e:
            print(f"KeyBERT failed on a doc: {e}")
            kws = []
        all_keywords.append(kws)
    return all_keywords


# ============================================================
# 3. TOPIC MODELING - LDA (gensim)
# ============================================================

def run_lda(texts, n_topics=N_LDA_TOPICS):
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel

    tokenized = [
        [t for t in basic_clean(doc).split() if t not in AMHARIC_STOPWORDS and len(t) > 1]
        for doc in texts
    ]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        random_state=42,
        alpha="auto",
        eta="auto",
    )

    coherence_model = CoherenceModel(
        model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v"
    )
    coherence = coherence_model.get_coherence()

    # per-doc "keywords" = top words of the doc's dominant topic
    doc_keywords = []
    for bow in corpus:
        if not bow:
            doc_keywords.append([])
            continue
        topic_dist = lda.get_document_topics(bow)
        top_topic = max(topic_dist, key=lambda x: x[1])[0]
        words = [w for w, _ in lda.show_topic(top_topic, topn=TOP_K)]
        doc_keywords.append(words)

    diversity = topic_diversity([lda.show_topic(i, topn=10) for i in range(n_topics)])
    return doc_keywords, coherence, diversity, lda


# ============================================================
# 4. TOPIC MODELING - BERTopic
# ============================================================

def run_bertopic(texts, embedding_model_name):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    cleaned_texts = [basic_clean(t) for t in texts]
    sbert = SentenceTransformer(embedding_model_name)
    topic_model = BERTopic(embedding_model=sbert, verbose=False)
    topics, _ = topic_model.fit_transform(cleaned_texts)

    doc_keywords = []
    for t in topics:
        if t == -1:
            doc_keywords.append([])
            continue
        words = [w for w, _ in topic_model.get_topic(t)][:TOP_K]
        doc_keywords.append(words)

    tokenized = [[tok for tok in d.split() if tok not in AMHARIC_STOPWORDS] for d in cleaned_texts]
    coherence = compute_coherence_from_topics(
        topic_model.get_topics(), tokenized
    )
    topic_words = [
        [w for w, _ in topic_model.get_topic(tid)][:10]
        for tid in topic_model.get_topics().keys() if tid != -1
    ]
    diversity = topic_diversity(topic_words)
    return doc_keywords, coherence, diversity, topic_model


def compute_coherence_from_topics(topics_dict, tokenized_texts):
    from gensim import corpora
    from gensim.models.coherencemodel import CoherenceModel
    dictionary = corpora.Dictionary(tokenized_texts)
    topic_word_lists = [
        [w for w, _ in words][:10] for tid, words in topics_dict.items() if tid != -1
    ]
    topic_word_lists = [t for t in topic_word_lists if t]
    if not topic_word_lists:
        return 0.0
    cm = CoherenceModel(
        topics=topic_word_lists, texts=tokenized_texts, dictionary=dictionary, coherence="c_v"
    )
    return cm.get_coherence()


def topic_diversity(topics_words_list):
    """Fraction of unique words across all topics' top-10 words. Higher = more diverse."""
    all_words = [w for topic in topics_words_list for w in [x[0] if isinstance(x, tuple) else x for x in topic]]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


# ============================================================
# 5. EVALUATION - mAP@k for keyword extraction
# ============================================================

def average_precision_at_k(predicted, ground_truth, k):
    predicted_k = predicted[:k]
    if not ground_truth:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(predicted_k, start=1):
        if p in ground_truth:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / min(len(ground_truth), k)


def mean_average_precision(all_predicted, all_ground_truth, k):
    scores = [
        average_precision_at_k(pred, gt, k)
        for pred, gt in zip(all_predicted, all_ground_truth)
    ]
    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("Loading validation set...")
    val_df = load_data(VALIDATION_PATH)
    if N_DOCS_TO_EVAL:
        val_df = val_df.head(N_DOCS_TO_EVAL)

    texts = val_df[TEXT_COL].tolist()
    ground_truth = [parse_gt_keywords(c) for c in val_df[KEYWORDS_COL].tolist()]

    results = []

    # --- KeyBERT ---
    print("\nRunning KeyBERT...")
    kb_preds = run_keybert(texts, EMBEDDING_MODEL)
    kb_map5 = mean_average_precision(kb_preds, ground_truth, 5)
    kb_map10 = mean_average_precision(kb_preds, ground_truth, 10)
    results.append({
        "model": "KeyBERT",
        "mAP@5": round(kb_map5, 4),
        "mAP@10": round(kb_map10, 4),
        "coherence_c_v": None,
        "topic_diversity": None,
        "meets_0.75_target": kb_map5 >= 0.75 or kb_map10 >= 0.75,
    })

    # --- LDA ---
    print("Running LDA...")
    lda_preds, lda_coherence, lda_diversity, _ = run_lda(texts)
    lda_map5 = mean_average_precision(lda_preds, ground_truth, 5)
    lda_map10 = mean_average_precision(lda_preds, ground_truth, 10)
    results.append({
        "model": "LDA",
        "mAP@5": round(lda_map5, 4),
        "mAP@10": round(lda_map10, 4),
        "coherence_c_v": round(lda_coherence, 4),
        "topic_diversity": round(lda_diversity, 4),
        "meets_0.75_target": lda_map5 >= 0.75 or lda_map10 >= 0.75,
    })

    # --- BERTopic ---
    print("Running BERTopic...")
    bt_preds, bt_coherence, bt_diversity, _ = run_bertopic(texts, EMBEDDING_MODEL)
    bt_map5 = mean_average_precision(bt_preds, ground_truth, 5)
    bt_map10 = mean_average_precision(bt_preds, ground_truth, 10)
    results.append({
        "model": "BERTopic",
        "mAP@5": round(bt_map5, 4),
        "mAP@10": round(bt_map10, 4),
        "coherence_c_v": round(bt_coherence, 4),
        "topic_diversity": round(bt_diversity, 4),
        "meets_0.75_target": bt_map5 >= 0.75 or bt_map10 >= 0.75,
    })

    report_df = pd.DataFrame(results)
    print("\n=== Phase 4 Model Comparison Report ===")
    print(report_df.to_string(index=False))

    out_path = "comparison_report.csv"
    report_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()