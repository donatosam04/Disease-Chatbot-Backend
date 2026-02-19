import pandas as pd
from pathlib import Path

# ===============================
# INPUT DATASETS
# ===============================
INPUT_FILES = [
    "data/processed/respiratory10_bert.csv",
    "data/processed/gi5_train.csv",
    "data/processed/infectious_metabolic_train.csv",
]

OUT = "data/processed/combined_diseases_train.csv"
Path("data/processed").mkdir(parents=True, exist_ok=True)

dfs = []

for file in INPUT_FILES:
    df = pd.read_csv(file)
    assert {"text", "label"}.issubset(df.columns), f"{file} missing columns"
    dfs.append(df[["text", "label"]])

combined = pd.concat(dfs).reset_index(drop=True)

print("Combined dataset shape:", combined.shape)
print("\nLabel distribution:")
print(combined["label"].value_counts())

combined.to_csv(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
