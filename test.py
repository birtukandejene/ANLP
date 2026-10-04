from pathlib import Path

print("=" * 80)
print("AMHARIC EMBEDDING MODEL TEST")
print("=" * 80)

MODEL_PATH = Path("models/embedding-amharic-base")

print(f"\nModel path:\n{MODEL_PATH.resolve()}")
print(f"Exists: {MODEL_PATH.exists()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH.resolve()}")

print("\nSTEP 1: Importing SentenceTransformer...", flush=True)

from sentence_transformers import SentenceTransformer

print("STEP 2: Import successful.", flush=True)

print("\nSTEP 3: Loading local Amharic embedding model...", flush=True)
print("This time NO Hugging Face download should be required.", flush=True)

model = SentenceTransformer(
    str(MODEL_PATH.resolve()),
    device="cpu"
)

print("\nSTEP 4: Model loaded successfully.", flush=True)

dimension = model.get_sentence_embedding_dimension()

print(f"Embedding dimension: {dimension}", flush=True)

texts = [
    "የፕሮጀክቱ ስራ አስኪያጅ አቶ ዳዊት ገብረ እግዚአብሄር እንደገለጹት የማከፋፈያ ጣቢያው የግንባታ፣ የፍተሻ እና የሙከራ ስራ ሙሉ በሙሉ ተጠናቆ ለአገልግሎት ዝግጁ ሆኗል።",

    "የማከፋፈያ ጣቢያው ግንባታ ተጠናቆ ለአገልግሎት ዝግጁ ሆኗል።",

    "እግዚአብሄር የተመረጡትን ህዝቦቹን ከባርነት ነጻ ለማድረግ የሰራውን ስራ ይተርክልናል።",

    "በኑሮአችንና በአገልግሎታችን ሊገለጥ የሚገባው የእግዚአብሄር ፍቅር ነው።",

    "የኢትዮጵያ ብሔራዊ እግር ኳስ ቡድን ለቀጣዩ ጨዋታ ከፍተኛ ዝግጅት እያደረገ ነው።",

    "የሸቀጦች ዋጋ መጨመር በኢትዮጵያ ኢኮኖሚ ላይ ከፍተኛ ተፅዕኖ እያሳደረ ነው።",
]

print("\nSTEP 5: Encoding test sentences...", flush=True)

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nSTEP 6: Encoding finished.", flush=True)

print(f"Embedding shape: {embeddings.shape}")

if embeddings.shape[1] != 768:
    print(
        f"\nWARNING: Expected 768 dimensions, "
        f"but received {embeddings.shape[1]}."
    )

print("\n" + "=" * 80)
print("SEMANTIC SIMILARITY TEST")
print("=" * 80)

query = embeddings[0]

# Since embeddings are normalized, dot product = cosine similarity.
scores = embeddings @ query

ranking = scores.argsort()[::-1]

for rank, idx in enumerate(ranking, start=1):
    print("\n" + "-" * 80)
    print(f"RANK {rank}")
    print(f"Similarity: {scores[idx]:.4f}")
    print(f"Text: {texts[idx]}")

print("\n" + "=" * 80)
print("PAIRWISE CHECKS")
print("=" * 80)

pairs = [
    (0, 1, "PROJECT vs PROJECT"),
    (0, 2, "PROJECT vs RELIGION"),
    (0, 3, "PROJECT vs RELIGION"),
    (0, 4, "PROJECT vs SPORT"),
    (0, 5, "PROJECT vs ECONOMY"),
]

for a, b, label in pairs:
    similarity = float(embeddings[a] @ embeddings[b])
    print(f"{label}: {similarity:.4f}")

print("\nTEST COMPLETE.")