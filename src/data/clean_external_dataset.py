import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

INPUT = "data/processed/combined_diseases_train.csv"
OUT_DIR = Path("data/processed/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("🔄 Loading data...")
df = pd.read_csv(INPUT)

print("🔄 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("🔄 Encoding text...")
embeddings = model.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

# Save embeddings + labels
np.save(OUT_DIR / "X.npy", embeddings)
np.save(OUT_DIR / "y.npy", df["label"].values)

print("✅ Embeddings saved")
print("X shape:", embeddings.shape)
print("y shape:", df["label"].shape)
