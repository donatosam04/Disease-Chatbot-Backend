import pandas as pd
import random
from pathlib import Path

random.seed(42)

INPUT = "data/processed/gi5_train.csv"
OUT   = "data/processed/gi5_train_contrastive.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

CONFUSIONS = {
    "gerd": "peptic_ulcer",
    "peptic_ulcer": "gerd",
    "gastroenteritis": "gerd",
    "hepatitis": "gastroenteritis"
}

df = pd.read_csv(INPUT)
rows = []

for _, r in df.iterrows():
    rows.append(r)

    label = r["label"]
    if label in CONFUSIONS:
        fake = r.copy()
        fake["label"] = CONFUSIONS[label]
        fake["text"] = (
            r["text"]
            + " However, symptoms do not clearly match the primary diagnosis."
        )
        rows.append(fake)

aug = pd.DataFrame(rows).sample(frac=1, random_state=42)
aug.to_csv(OUT, index=False)

print("✅ Contrastive GI-5 training set created")
print(aug["label"].value_counts())
