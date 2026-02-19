import pandas as pd
from pathlib import Path

INPUT = "data/processed/universal_disease_train_v2.csv"
OUTPUT = "data/processed/universal_disease_train_v3.csv"

# Load
df = pd.read_csv(INPUT)

# Label normalization map
LABEL_MAP = {
    "bronchial_asthma": "asthma",
    "hypertension_": "hypertension",
    "diabetes_": "diabetes",
    "urinary_tract_infection": "uti",
}

df["label"] = df["label"].replace(LABEL_MAP)

# Remove rare classes (<50 samples)
df = df.groupby("label").filter(lambda x: len(x) >= 50)

# Stats
print("✅ Samples:", len(df))
print("✅ Diseases:", df["label"].nunique())
print("\nLabel distribution:")
print(df["label"].value_counts())

# Save
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"\n💾 Saved → {OUTPUT}")
