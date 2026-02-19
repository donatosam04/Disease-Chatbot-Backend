import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize

# ============================
# CONFIG
# ============================

MODEL_DIR = Path("models_saved/universal_disease_model")
EVAL_DIR  = Path("models_saved/universal_eval/hard")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = "data/processed/combined_diseases_hard_val.csv"

# ============================
# LOAD MODEL & DATA
# ============================

print("🔹 Loading model...")
clf = joblib.load(MODEL_DIR / "classifier.pkl")
le  = joblib.load(MODEL_DIR / "label_encoder.pkl")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_csv(VAL_CSV)
texts = df["text"].tolist()
y_true = le.transform(df["label"])
class_names = le.classes_
n_classes = len(class_names)

print("🔹 Encoding text...")
X = embedder.encode(texts, batch_size=32, show_progress_bar=True)

# ============================
# PREDICTIONS
# ============================

y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)

# ============================
# 1️⃣ TOP-K ACCURACY
# ============================

top1 = top_k_accuracy_score(y_true, y_proba, k=1)
top3 = top_k_accuracy_score(y_true, y_proba, k=3)
top5 = top_k_accuracy_score(y_true, y_proba, k=min(5, n_classes))

plt.figure(figsize=(6,4))
plt.bar(["Top-1", "Top-3", "Top-5"], [top1*100, top3*100, top5*100])
plt.ylabel("Accuracy (%)")
plt.title("Top-K Accuracy")
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(EVAL_DIR / "topk_accuracy.png", dpi=300)
plt.close()

# ============================
# 2️⃣ MACRO vs MICRO METRICS
# ============================

macro = precision_recall_fscore_support(y_true, y_pred, average="macro")
micro = precision_recall_fscore_support(y_true, y_pred, average="micro")

metrics = pd.DataFrame({
    "Macro": macro[:3],
    "Micro": micro[:3]
}, index=["Precision", "Recall", "F1-score"])

metrics.T.plot(kind="bar", figsize=(7,4))
plt.ylabel("Score")
plt.title("Macro vs Micro Metrics")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(EVAL_DIR / "macro_vs_micro.png", dpi=300)
plt.close()

# ============================
# 3️⃣ SUPPORT vs F1 SCORE
# ============================

support = df["label"].value_counts().sort_index()
f1_per_class = precision_recall_fscore_support(
    y_true, y_pred, average=None
)[2]

plt.figure(figsize=(8,5))
plt.scatter(support.values, f1_per_class)
for i, label in enumerate(class_names):
    plt.annotate(label, (support.values[i], f1_per_class[i]), fontsize=8)
plt.xlabel("Number of Samples")
plt.ylabel("F1 Score")
plt.title("Support vs F1 Score")
plt.tight_layout()
plt.savefig(EVAL_DIR / "support_vs_f1.png", dpi=300)
plt.close()

# ============================
# 4️⃣ OFF-DIAGONAL ERROR HEATMAP
# ============================

cm = confusion_matrix(y_true, y_pred)
np.fill_diagonal(cm, 0)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Reds", xticklabels=class_names, yticklabels=class_names)
plt.title("Error Heatmap (Misclassifications Only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(EVAL_DIR / "error_heatmap.png", dpi=300)
plt.close()

# ============================
# 5️⃣ CONFIDENCE DISTRIBUTION
# ============================

confidence = y_proba.max(axis=1)
correct = confidence[y_true == y_pred]
incorrect = confidence[y_true != y_pred]

plt.figure(figsize=(7,4))
sns.kdeplot(correct, label="Correct", fill=True)
sns.kdeplot(incorrect, label="Incorrect", fill=True)
plt.xlabel("Prediction Confidence")
plt.title("Prediction Confidence Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(EVAL_DIR / "confidence_distribution.png", dpi=300)
plt.close()

# ============================
# 6️⃣ ROC-AUC (ONE-vs-REST)
# ============================

y_bin = label_binarize(y_true, classes=range(n_classes))
roc_scores = []

for i in range(n_classes):
    roc_scores.append(
        roc_auc_score(y_bin[:, i], y_proba[:, i])
    )

plt.figure(figsize=(8,4))
sns.barplot(x=class_names, y=roc_scores)
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC")
plt.title("Per-Class ROC-AUC")
plt.ylim(0.5,1.0)
plt.tight_layout()
plt.savefig(EVAL_DIR / "roc_auc_per_class.png", dpi=300)
plt.close()

print("\n✅ EXTRA EVALUATION COMPLETE")
print("📁 Saved all graphs to:", EVAL_DIR)
