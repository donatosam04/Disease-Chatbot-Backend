import pandas as pd
from pathlib import Path

# ============================
# INPUT DATASETS
# ============================
BASE = "data/processed/combined_diseases_train.csv"
HF = "data/processed/hf_cleaned.csv"
MENDELEY = "data/processed/mendeley_cleaned.csv"

OUT = "data/processed/universal_disease_train_v2.csv"

# ============================
# LOAD
# ============================
dfs = []

for path in [BASE, HF, MENDELEY]:
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns)
    dfs.append(df)

combined = pd.concat(dfs).reset_index(drop=True)

# ============================
# CLEAN
# ============================
combined["text"] = combined["text"].astype(str)
combined["label"] = combined["label"].astype(str)

# Optional: remove very rare labels (<50)
combined = combined.groupby("label").filter(lambda x: len(x) >= 50)

# ============================
# STATS
# ============================
print("✅ Total samples:", len(combined))
print("✅ Unique diseases:", combined["label"].nunique())
print("\nLabel distribution:")
print(combined["label"].value_counts())

# ============================
# SAVE
# ============================
Path("data/processed").mkdir(parents=True, exist_ok=True)
combined.to_csv(OUT, index=False)

print(f"\n💾 Saved → {OUT}")
