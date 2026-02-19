import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ============================
# CONFIG
# ============================

INPUT = "data/processed/universal_disease_train_v6.csv"
OUT = "data/processed/universal_embeddings_v6.npy"
OUT_META = "data/processed/universal_labels_v6.csv"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(INPUT)
assert {"text", "label"}.issubset(df.columns)

print("✅ Samples:", len(df))
print("✅ Unique diseases:", df["label"].nunique())
print("\nLabel distribution:")
print(df["label"].value_counts())

# ============================
# EMBEDDINGS
# ============================

print("\n🔹 Loading MPNet model...")
model = SentenceTransformer(MODEL_NAME)

print("🔹 Encoding texts...")
embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32
)

# ============================
# SAVE
# ============================

np.save(OUT , embeddings)
df[["label"]].to_csv(OUT_META, index=False)

print("\n✅ Embeddings saved")
print("📐 Shape:", embeddings.shape)
print("💾 Files:")
print(" -", OUT)
print(" -", OUT_META)
