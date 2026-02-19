import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =========================
# CONFIG
# =========================

MODEL_DIR = Path("models_saved/universal_disease_model")
MODEL_PATH = MODEL_DIR / "classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# =========================
# LOAD MODEL & ENCODER
# =========================

print("🔹 Loading model...")
clf = joblib.load(MODEL_PATH)

print("🔹 Loading label encoder...")
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("🔹 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# =========================
# TOP-3 PREDICTION FUNCTION
# =========================

def predict_top3(symptoms_text: str):
    """
    Returns Top-3 disease predictions with confidence.
    """

    # Embed input text
    embedding = embedder.encode([symptoms_text])

    # Get probabilities
    probs = clf.predict_proba(embedding)[0]

    # Get top-3 indices
    top3_idx = np.argsort(probs)[-3:][::-1]

    results = []
    for idx in top3_idx:
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        results.append({
            "disease": disease,
            "confidence": round(confidence, 4)
        })

    return results


# =========================
# CLI TEST
# =========================

if __name__ == "__main__":
    text = input("\n🩺 Enter symptoms: ")
    preds = predict_top3(text)

    print("\n🔍 Top-3 Predictions:")
    for i, p in enumerate(preds, 1):
        print(f"{i}. {p['disease']} ({p['confidence']*100:.2f}%)")
