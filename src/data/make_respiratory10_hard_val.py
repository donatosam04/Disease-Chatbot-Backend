import pandas as pd
from pathlib import Path

# ======================
# CONFIG
# ======================
SRC = "data/processed/respiratory10_augmented_train.csv"
OUT = "data/processed/respiratory10_hard_val.csv"

PER_CLASS = 200

TARGET_LABELS = [
    "allergy",
    "asthma",
    "common_cold",
    "pneumonia",
    "tuberculosis",
]

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(SRC)
assert {"text", "label"}.issubset(df.columns)

print("Available samples per label:")
print(df["label"].value_counts())

rows = []

for label in TARGET_LABELS:
    subset = df[df["label"] == label]

    if len(subset) == 0:
        print(f"❌ Skipping {label}: no samples available")
        continue

    if len(subset) < PER_CLASS:
        print(f"⚠️ Not enough samples for {label}, duplicating with shuffle")
        subset = subset.sample(
            PER_CLASS, replace=True, random_state=42
        )
    else:
        subset = subset.sample(
            PER_CLASS, replace=False, random_state=42
        )

    rows.append(subset)

# ======================
# SAVE
# ======================
hard_df = pd.concat(rows).sample(frac=1, random_state=42).reset_index(drop=True)
hard_df[["text", "label"]].to_csv(OUT, index=False)

print(f"✅ Hard validation saved → {OUT}")
print("Final hard label distribution:")
print(hard_df["label"].value_counts())
