import pandas as pd
import json

IN_FILE = "data/processed/gretel_cleaned.csv"
OUT_FILE = "data/processed/gretel_normalized.csv"
MAP_FILE = "src/data/label_normalization_map.json"

df = pd.read_csv(IN_FILE)

with open(MAP_FILE, "r") as f:
    label_map = json.load(f)

df["label"] = df["label"].apply(
    lambda x: label_map.get(x, x.replace(" ", "_"))
)

print("✅ Diseases after normalization:", df["label"].nunique())
print(df["label"].value_counts().head(20))

df.to_csv(OUT_FILE, index=False)
print(f"💾 Saved → {OUT_FILE}")
