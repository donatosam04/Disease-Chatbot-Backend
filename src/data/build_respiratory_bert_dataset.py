import pandas as pd
import random
from pathlib import Path

# =========================
# Config
# =========================
RAW_DS1 = "data/raw/kaggle_disease_symptoms/DiseaseAndSymptoms.csv"
RAW_DS2 = "data/raw/kaggle_disease_symptoms/symbipredict_2022.csv"
OUT_CSV = "data/processed/respiratory_bert_full.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# Robust alias map (substring-safe)
RESPIRATORY_DISEASES = {
    "asthma": ["asthma"],
    "acute bronchitis": ["bronchitis"],
    "acute bronchiolitis": ["bronchiolitis"],
    "pneumonia": ["pneumonia"],
    "acute bronchospasm": ["bronchospasm"]
}

TEMPLATES = [
    "Patient reports {symptoms}.",
    "Symptoms include {symptoms}.",
    "The individual experiences {symptoms}.",
    "Patient presents with {symptoms}.",
    "Clinical symptoms observed are {symptoms}."
]

# =========================
# Helper functions
# =========================
def normalize_symptom(s):
    return str(s).strip().lower().replace("_", " ")

def make_sentence(symptoms):
    symptoms = [normalize_symptom(s) for s in symptoms if pd.notna(s)]
    if len(symptoms) < 2:
        return None
    random.shuffle(symptoms)
    sent = random.choice(TEMPLATES)
    return sent.format(symptoms=", ".join(symptoms))

# =========================
# Load datasets
# =========================
df1 = pd.read_csv(RAW_DS1)
df2 = pd.read_csv(RAW_DS2)

rows = []

# =========================
# Process DiseaseAndSymptoms.csv
# =========================
symptom_cols_1 = [c for c in df1.columns if c.lower().startswith("symptom")]

for _, row in df1.iterrows():
    disease = str(row["Disease"]).strip().lower()
    symptoms = [row[c] for c in symptom_cols_1 if pd.notna(row[c])]

    sentence = make_sentence(symptoms)
    if not sentence:
        continue

    for target, aliases in RESPIRATORY_DISEASES.items():
        if any(a in disease for a in aliases):
            rows.append({
                "text": sentence,
                "label": target
            })
            break

# =========================
# Process symbipredict_2022.csv
# =========================
symptom_cols_2 = [c for c in df2.columns if c != "prognosis"]

for _, row in df2.iterrows():
    disease = str(row["prognosis"]).strip().lower()
    symptoms = [col for col in symptom_cols_2 if row[col] == 1]

    sentence = make_sentence(symptoms)
    if not sentence:
        continue

    for target, aliases in RESPIRATORY_DISEASES.items():
        if any(a in disease for a in aliases):
            rows.append({
                "text": sentence,
                "label": target
            })
            break

# =========================
# Create DataFrame
# =========================
out_df = pd.DataFrame(rows)

print("\nOriginal class distribution:")
print(out_df["label"].value_counts())

# =========================
# Balance dataset
# =========================
MIN_SAMPLES = 250

balanced = []
for label, group in out_df.groupby("label"):
    if len(group) >= MIN_SAMPLES:
        balanced.append(group.sample(MIN_SAMPLES, random_state=42))
    else:
        balanced.append(group.sample(MIN_SAMPLES, replace=True, random_state=42))

final_df = (
    pd.concat(balanced)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print("\nBalanced class distribution:")
print(final_df["label"].value_counts())

# =========================
# Save
# =========================
final_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved dataset to: {OUT_CSV}")
