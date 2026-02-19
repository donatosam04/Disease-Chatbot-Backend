import json
import pandas as pd
from pathlib import Path
import random

IN_FILE = "src/data/additional_diseases_templates.json"
OUT_FILE = "data/processed/additional_diseases.csv"

TARGET_SAMPLES = 300  # per disease

with open(IN_FILE, "r") as f:
    data = json.load(f)

rows = []

for disease, templates in data.items():
    for _ in range(TARGET_SAMPLES):
        text = random.choice(templates)
        rows.append({
            "text": text,
            "label": disease
        })

df = pd.DataFrame(rows)
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("✅ Additional diseases generated")
print(df["label"].value_counts())
