import pandas as pd
import random
from pathlib import Path

random.seed(42)

INPUT = "data/raw/kaggle_disease_symptoms/DiseaseAndSymptoms.csv"

OUT_TRAIN = "data/processed/gi5_train.csv"
OUT_VAL   = "data/processed/gi5_val.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Disease mapping
# -----------------------------
GI_MAP = {
    "gastroenteritis": ["gastroenteritis"],
    "gerd": ["gerd"],
    "hepatitis": [
        "hepatitis a", "hepatitis b", "hepatitis c",
        "hepatitis d", "hepatitis e"
    ],
    "peptic_ulcer": ["peptic ulcer disease"],
}

# -----------------------------
# Discriminative phrases
# -----------------------------
DISCRIMINATORS = {
    "gastroenteritis": [
        "acute onset of symptoms",
        "diarrhea is the dominant complaint",
        "history of contaminated food intake"
    ],
    "gerd": [
        "symptoms worsen when lying down",
        "burning sensation after meals",
        "acid regurgitation with sour taste"
    ],
    "hepatitis": [
        "yellowing of eyes and skin",
        "dark urine and pale stools",
        "persistent fatigue and jaundice"
    ],
    "peptic_ulcer": [
        "pain relieved after eating",
        "pain occurs on empty stomach",
        "night-time gnawing epigastric pain"
    ],
}

# -----------------------------
# Text generator
# -----------------------------
def make_text(symptoms, label):
    symptoms = [s.replace("_", " ").lower() for s in symptoms if pd.notna(s)]
    random.shuffle(symptoms)

    s1 = f"Patient reports {', '.join(symptoms[:3])}."
    s2 = random.choice(DISCRIMINATORS[label]) + "."
    s3 = "Symptoms have persisted for several days."

    return " ".join([s1, s2, s3])

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT)
sym_cols = [c for c in df.columns if c.lower().startswith("symptom")]

rows = []

for _, row in df.iterrows():
    disease = str(row["Disease"]).lower().strip()
    symptoms = [row[c] for c in sym_cols if pd.notna(row[c])]

    for label, aliases in GI_MAP.items():
        if disease in aliases and len(symptoms) >= 3:
            rows.append({
                "text": make_text(symptoms, label),
                "label": label
            })

df_all = pd.DataFrame(rows)

print("Raw distribution:")
print(df_all["label"].value_counts())

# -----------------------------
# Balanced split
# -----------------------------
train = df_all.groupby("label").sample(600, replace=True, random_state=42)
val   = df_all.groupby("label").sample(150, replace=True, random_state=43)

train.to_csv(OUT_TRAIN, index=False)
val.to_csv(OUT_VAL, index=False)

print("✅ GI-5 train & validation datasets created")
