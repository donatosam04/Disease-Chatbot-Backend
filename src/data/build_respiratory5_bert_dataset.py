import pandas as pd
import random
from pathlib import Path

# =========================
# Config
# =========================
RAW_DS = "data/raw/kaggle_disease_symptoms/DiseaseAndSymptoms.csv"
OUT_CSV = "data/processed/respiratory5_bert.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

DISEASE_MAP = {
    "asthma": ["bronchial asthma"],
    "common_cold": ["common cold"],
    "pneumonia": ["pneumonia"],
    "tuberculosis": ["tuberculosis"],
    "allergy": ["allergy"]
}

TEMPLATES = [
    "Patient reports {symptoms}.",
    "Symptoms include {symptoms}.",
    "The patient complains of {symptoms}.",
    "Clinical presentation shows {symptoms}.",
    "Individual experiences {symptoms}.",
    "Observed symptoms are {symptoms}.",
]

# =========================
def normalize(s):
    return str(s).strip().lower().replace("_", " ")

def build_sentence(symptoms):
    symptoms = [normalize(s) for s in symptoms if s and str(s) != "nan"]
    random.shuffle(symptoms)

    # dropout → improves generalization
    keep = max(3, int(len(symptoms) * random.uniform(0.6, 1.0)))
    symptoms = symptoms[:keep]

    return random.choice(TEMPLATES).format(
        symptoms=", ".join(symptoms)
    )

# =========================
df = pd.read_csv(RAW_DS)
symptom_cols = [c for c in df.columns if c.startswith("Symptom")]

rows = []

for _, row in df.iterrows():
    disease = normalize(row["Disease"])
    symptoms = [row[c] for c in symptom_cols if pd.notna(row[c])]

    for label, aliases in DISEASE_MAP.items():
        if disease in aliases and len(symptoms) >= 3:
            # generate multiple linguistic variants
            for _ in range(5):
                rows.append({
                    "text": build_sentence(symptoms),
                    "label": label
                })

out = pd.DataFrame(rows)

print("\nOriginal distribution:")
print(out["label"].value_counts())

# =========================
# Balance (CRITICAL for accuracy)
# =========================
TARGET = 1000
balanced = []

for label, grp in out.groupby("label"):
    if len(grp) >= TARGET:
        balanced.append(grp.sample(TARGET, random_state=42))
    else:
        balanced.append(grp.sample(TARGET, replace=True, random_state=42))

final = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced distribution:")
print(final["label"].value_counts())

final.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")
