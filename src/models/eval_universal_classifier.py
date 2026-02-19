import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

# ============================
# CONFIG
# ============================

MODEL_DIR = Path("models_saved/universal_disease_model")
MODEL_PATH = MODEL_DIR / "classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

VAL_CSV = "data/processed/combined_diseases_hard_val.csv"

OUT_DIR = Path("models_saved/evaluation/hard")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ============================
# LOAD MODEL & DATA
# ============================

print("🔹 Loading model & label encoder...")
clf = joblib.load(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

print("🔹 Loading sentence transformer...")
embedder = SentenceTransformer(EMBED_MODEL)

df = pd.read_csv(VAL_CSV)
assert {"text", "label"}.issubset(df.columns)

# keep only labels known to model
df = df[df["label"].isin(le.classes_)].reset_index(drop=True)

texts = df["text"].tolist()
true_labels = le.transform(df["label"].tolist())

class_names = le.classes_
num_classes = len(class_names)

print(f"🧪 Validation samples: {len(df)}")
print(f"🧬 Total diseases in model: {num_classes}")

# ============================
# EMBEDDINGS
# ============================

print("🔹 Encoding texts...")
X = embedder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ============================
# PREDICTION
# ============================

probs = clf.predict_proba(X)
preds = probs.argmax(axis=1)

# ============================
# ACCURACY
# ============================

acc = accuracy_score(true_labels, preds)
top3 = top_k_accuracy_score(true_labels, probs, k=3, labels=range(num_classes))

print(f"\n✅ HARD Validation Accuracy (Top-1): {acc*100:.2f}%")
print(f"✅ HARD Validation Accuracy (Top-3): {top3*100:.2f}%")

# ============================
# SAFE CLASSIFICATION REPORT
# ============================

present_class_ids = np.unique(true_labels)
present_class_names = le.inverse_transform(present_class_ids)

report = classification_report(
    true_labels,
    preds,
    labels=present_class_ids,
    target_names=present_class_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUT_DIR / "classification_report.csv")

print("📊 Classification report saved")

# ============================
# CONFUSION MATRIX (COUNTS)
# ============================

cm = confusion_matrix(
    true_labels,
    preds,
    labels=present_class_ids
)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=present_class_names,
    yticklabels=present_class_names
)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_counts.png", dpi=300)
plt.close()

# ============================
# PER-CLASS F1 SCORE
# ============================

f1_scores = report_df.loc[present_class_names]["f1-score"]

plt.figure(figsize=(12, 4))
sns.barplot(x=f1_scores.index, y=f1_scores.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_f1.png", dpi=300)
plt.close()

# ============================
# CONFIDENCE DISTRIBUTION
# ============================

max_conf = probs.max(axis=1)

plt.figure(figsize=(6, 4))
sns.histplot(max_conf, bins=20, kde=True)
plt.xlabel("Prediction Confidence")
plt.title("Prediction Confidence Distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_distribution.png", dpi=300)
plt.close()

print("\n📁 Evaluation graphs saved to:", OUT_DIR)
print("✅ Universal model evaluation complete")
