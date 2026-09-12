import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

MODEL_DIR = Path("models_saved/universal_disease_model")
MODEL_PATH = MODEL_DIR / "classifier.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# =====================================================
# LOAD MODELS ONCE (IMPORTANT FOR PERFORMANCE)
# =====================================================

classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_top3(text: str):


    if not text or not text.strip():
        return []

    # Generate embedding (normalized for stability)
    embedding = embed_model.encode(
        [text],
        normalize_embeddings=True
    )
    # Get probability distribution
    probs = classifier.predict_proba(embedding)[0]

    # Sort probabilities descending
    top3_idx = np.argsort(probs)[-3:][::-1]

    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    # Convert numpy types to Python native floats
    results = [
        (str(label), float(prob))
        for label, prob in zip(top3_labels, top3_probs)
    ]
    return results