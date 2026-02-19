from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ============================
# FLASK APP
# ============================

app = Flask(__name__)

# ============================
# LOAD MODEL & ENCODER
# ============================

MODEL_DIR = Path("models_saved/universal_disease_model")

clf = joblib.load(MODEL_DIR / "classifier.pkl")
le = joblib.load(MODEL_DIR / "label_encoder.pkl")

# IMPORTANT: SAME MODEL USED IN TRAINING
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

CLASS_NAMES = le.classes_

print("✅ Model, encoder, and embedder loaded")

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    # 1️⃣ Encode text
    emb = embedder.encode([text])  # shape (1, 768)

    # 2️⃣ Predict probabilities
    probs = clf.predict_proba(emb)[0]

    # 3️⃣ Top predictions
    top_indices = np.argsort(probs)[::-1][:3]

    results = []
    for idx in top_indices:
        results.append({
            "disease": CLASS_NAMES[idx],
            "confidence": round(float(probs[idx]) * 100, 2)
        })

    return jsonify({
        "top_prediction": results[0],
        "top_3_predictions": results
    })

# ============================
# RUN
# ============================

if __name__ == "__main__":
    app.run(debug=True)
