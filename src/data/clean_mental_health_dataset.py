import pandas as pd
from pathlib import Path

IN_FILE = "data/raw/huggingface/mental_health_raw.csv"
OUT_FILE = "data/processed/mental_health_cleaned.csv"

df = pd.read_csv(IN_FILE)

# Map topics to disease-style labels
LABEL_MAP = {
    "depression": "depression",
    "anxiety": "anxiety_disorder",
    "stress": "stress_disorder",
    "bipolar disorder": "bipolar_disorder",
    "self-harm": "suicidal_ideation"
}

df["label"] = df["label"].str.lower().map(LABEL_MAP)
df = df.dropna()

print("✅ Samples:", len(df))
print("✅ Diseases:", df["label"].nunique())
print(df["label"].value_counts())

df.to_csv(OUT_FILE, index=False)
print(f"💾 Saved → {OUT_FILE}")
