import pandas as pd
from pathlib import Path

IN_FILE = "data/raw/huggingface/gretel_symptom_to_diagnosis.csv"
OUT_FILE = "data/processed/gretel_cleaned.csv"

df = pd.read_csv(IN_FILE)

# Rename correctly
df = df.rename(columns={
    "input_text": "text",
    "output_text": "label"
})

# Basic cleanup
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.lower().str.strip()

print("✅ Samples:", len(df))
print("✅ Unique diseases:", df["label"].nunique())
print(df["label"].value_counts().head(20))

Path("data/processed").mkdir(parents=True, exist_ok=True)
df[["text", "label"]].to_csv(OUT_FILE, index=False)

print(f"💾 Saved → {OUT_FILE}")
