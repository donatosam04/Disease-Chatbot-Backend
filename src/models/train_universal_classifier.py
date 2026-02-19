import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ============================
# CONFIG
# ============================

EMBEDDINGS_PATH = "data/processed/universal_embeddings_v6.npy"
LABELS_PATH = "data/processed/universal_labels_v6.csv"

OUT_DIR = Path("models_saved/universal_disease_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# LOAD
# ============================

print("🔹 Loading embeddings & labels...")
X = np.load(EMBEDDINGS_PATH)
df = pd.read_csv(LABELS_PATH)

le = LabelEncoder()
y = le.fit_transform(df["label"])

print(f"🔹 Classes ({len(le.classes_)}): {list(le.classes_)}")

# ============================
# SPLIT
# ============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# MODEL
# ============================

print("🚀 Training XGBoost universal disease classifier...")

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# EVALUATION
# ============================

probs = model.predict_proba(X_val)
preds = probs.argmax(axis=1)

top1 = accuracy_score(y_val, preds)
top3 = top_k_accuracy_score(y_val, probs, k=3)

print("\n📊 Validation Metrics")
print(f"✅ Top-1 Accuracy : {top1*100:.2f}%")
print(f"✅ Top-3 Accuracy : {top3*100:.2f}%")

# ============================
# SAVE
# ============================

joblib.dump(model, OUT_DIR / "classifier.pkl")
joblib.dump(le, OUT_DIR / "label_encoder.pkl")

print("\n💾 Model saved to:", OUT_DIR)
print("✅ Universal disease classifier training complete")
