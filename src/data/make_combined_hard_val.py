import pandas as pd
import random
from pathlib import Path

INPUT = "data/processed/combined_diseases_train.csv"
OUT = "data/processed/combined_diseases_hard_val.csv"

PER_CLASS = 150
CONFUSE_RATIO = 0.2

df = pd.read_csv(INPUT)
out = []

labels = df["label"].unique().tolist()

for label, subset in df.groupby("label"):
    subset = subset.sample(PER_CLASS, replace=True, random_state=42).copy()

    n_confuse = int(PER_CLASS * CONFUSE_RATIO)
    confuse_idx = subset.sample(n_confuse, random_state=42).index

    other_labels = [l for l in labels if l != label]

    for i in confuse_idx:
        other = random.choice(other_labels)
        subset.at[i, "text"] = (
            f"Patient reports symptoms similar to {label} "
            f"but also overlaps with {other}."
        )

    out.append(subset)

hard_df = pd.concat(out).sample(frac=1, random_state=42)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
hard_df.to_csv(OUT, index=False)

print("✅ HARD combined validation set created")
print(hard_df["label"].value_counts())
