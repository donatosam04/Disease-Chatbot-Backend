import pandas as pd
import random
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================

random.seed(42)

INPUT = "data/processed/gi5_val.csv"
OUT = "data/processed/gi5_hard_realistic.csv"

PER_CLASS = 200
CONFUSE_RATIO = 0.25   # stronger but realistic confusion

# Symptom pools per disease
SYMPTOMS = {
    "gastroenteritis": [
        "diarrhoea", "vomiting", "dehydration", "abdominal cramps",
        "loose stools", "stomach upset"
    ],
    "gerd": [
        "acid reflux", "heartburn", "chest discomfort",
        "regurgitation", "burning sensation"
    ],
    "hepatitis": [
        "fatigue", "jaundice", "loss of appetite",
        "upper abdominal pain", "dark urine"
    ],
    "peptic_ulcer": [
        "burning stomach pain", "bloating",
        "nausea", "indigestion", "pain on empty stomach"
    ],
}

# ======================================================
# LOAD BASE VALIDATION DATA
# ======================================================

df = pd.read_csv(INPUT)
assert {"text", "label"}.issubset(df.columns)

out_rows = []

for label, subset in df.groupby("label"):
    # Ensure enough samples
    subset = subset.sample(PER_CLASS, replace=True, random_state=42).copy()

    n_confuse = int(PER_CLASS * CONFUSE_RATIO)
    confuse_idx = subset.sample(n_confuse, random_state=42).index

    other_labels = [l for l in SYMPTOMS if l != label]

    for i in confuse_idx:
        other = random.choice(other_labels)

        mixed_symptoms = (
            random.sample(SYMPTOMS[label], 2)
            + random.sample(SYMPTOMS[other], 2)
        )
        random.shuffle(mixed_symptoms)

        subset.at[i, "text"] = (
            "Patient reports " + ", ".join(mixed_symptoms) + "."
        )

    out_rows.append(subset)

hard_df = pd.concat(out_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# ======================================================
# SAVE
# ======================================================

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
hard_df.to_csv(OUT, index=False)

# ======================================================
# SAFETY CHECKS
# ======================================================

print("✅ Strong-confusion GI-5 hard validation set created")
print(hard_df["label"].value_counts())

assert len(hard_df) == PER_CLASS * len(SYMPTOMS), "❌ Hard dataset size incorrect"
