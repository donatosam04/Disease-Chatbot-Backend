import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === CONFIG ===
MODEL_DIR = Path("models_saved/universal_disease_model")
MODEL_PATH = MODEL_DIR / "classifier.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# === LOAD ONCE (IMPORTANT) ===
classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def predict_top3(text: str):
    embedding = embed_model.encode([text])
    probs = classifier.predict_proba(embedding)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    return list(zip(top3_labels, top3_probs))
