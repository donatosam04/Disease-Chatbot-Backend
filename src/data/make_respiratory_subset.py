import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_CSV = "data/processed/bert_intent_val_reduced.csv"
OUT_DIR = Path("data/processed")
GROUP_NAME = "respiratory7"

DISEASES = [
    "pneumonia",
    "acute bronchitis",
    "asthma",
    "chronic obstructive pulmonary disease",
    "tuberculosis",
    "acute bronchiolitis",
    "acute bronchospasm"
]
# ---------------------------------------

df = pd.read_csv(INPUT_CSV)

print("\n ORIGINAL CLASS DISTRIBUTION")
print(df["label"].value_counts())

df_sub = df[df["label"].str.lower().isin(DISEASES)].copy()

print("\n FILTERED CLASS DISTRIBUTION")
print(df_sub["label"].value_counts())

# Shuffle
df_sub = df_sub.sample(frac=1, random_state=42)

# Split (80/20)
train = df_sub.groupby("label", group_keys=False).apply(
    lambda x: x.sample(frac=0.8, random_state=42)
)
val = df_sub.drop(train.index)

train_path = OUT_DIR / f"{GROUP_NAME}_train.csv"
val_path = OUT_DIR / f"{GROUP_NAME}_val.csv"

train.to_csv(train_path, index=False)
val.to_csv(val_path, index=False)

print(f"\n Saved:")
print(train_path)
print(val_path)
