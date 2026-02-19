# src/data/augment_respiratory10_training.py

import pandas as pd
from pathlib import Path

BASE_TRAIN = "data/processed/respiratory10_bert.csv"
EXTERNAL_DATA = "data/raw/symptom_disease_external_clean.csv"
OUT_TRAIN = "data/processed/respiratory10_augmented_train.csv"

TARGET_LABELS = {
    "allergy",
    "asthma",
    "common_cold",
    "pneumonia",
    "tuberculosis",
    "bronchitis",
    "covid19",
    "flu",
    "sinusitis",
    "lung_cancer",
}

MAX_EXTERNAL_PER_LABEL = 300  # safe cap to avoid domination

def main():
    base_df = pd.read_csv(BASE_TRAIN)
    print(f"Base training samples: {len(base_df)}")

    ext_df = pd.read_csv(EXTERNAL_DATA)

    # Safety checks
    assert {"text", "label"}.issubset(ext_df.columns), "External dataset must have text & label"

    # Keep only target diseases
    ext_df = ext_df[ext_df["label"].isin(TARGET_LABELS)]

    print("External samples before balancing:")
    print(ext_df["label"].value_counts())

    # Balance external data per label
    ext_balanced = (
        ext_df
        .groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_EXTERNAL_PER_LABEL), random_state=42))
    )

    print("External samples after balancing:")
    print(ext_balanced["label"].value_counts())

    # Merge
    combined = pd.concat(
        [base_df[["text", "label"]], ext_balanced[["text", "label"]]],
        ignore_index=True
    )

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    combined.to_csv(OUT_TRAIN, index=False)

    print(f"✅ Augmented training samples: {len(combined)}")
    print(f"📂 Saved → {OUT_TRAIN}")

if __name__ == "__main__":
    main()
