import pandas as pd
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
RAW_DS1 = "data/raw/kaggle_disease_symptoms/DiseaseAndSymptoms.csv"
RAW_DS2 = "data/raw/kaggle_disease_symptoms/symbipredict_2022.csv"
OUT_CSV = "data/processed/respiratory10_bert.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# 10 diseases (respiratory-focused)
RESPIRATORY_10 = {
    "asthma": ["asthma", "bronchial asthma"],
    "pneumonia": ["pneumonia"],
    "tuberculosis": ["tuberculosis"],
    "common_cold": ["common cold"],
    "allergy": ["allergy"],
    "bronchitis": ["bronchitis"],
    "influenza": ["influenza", "flu"],
    "covid": ["covid", "covid-19"],
    "sinusitis": ["sinusitis"],
    "copd": ["copd", "chronic obstructive pulmonary disease"],
}

TEMPLATES = [
    "Patient reports {symptoms}.",
    "Symptoms include {symptoms}.",
    "The patient experiences {symptoms}.",
    "Clinical signs observed are {symptoms}.",
    "The individual presents with {symptoms}.",
]

# =========================
# HELPERS
# =========================
def normalize(s):
    return s.strip().lower().replace("_", " ")

def make_sentence(symptoms):
    symptoms = [normalize(s) for s in symptoms if s]
    random.shuffle(symptoms)
    return random.choice(TEMPLATES).format(symptoms=", ".join(symptoms))

# =========================
# LOAD DATA
# =========================
df1 = pd.read_csv(RAW_DS1)
df2 = pd.read_csv(RAW_DS2)

rows = []

# -------- Dataset 1 --------
symptom_cols = [c for c in df1.columns if c.lower().startswith("symptom")]

for _, row in df1.iterrows():
    disease = str(row["Disease"]).lower().strip()
    symptoms = [row[c] for c in symptom_cols if pd.notna(row[c])]

    for label, aliases in RESPIRATORY_10.items():
        if disease in aliases and len(symptoms) >= 2:
            rows.append({
                "text": make_sentence(symptoms),
                "label": label
            })

# -------- Dataset 2 --------
symptom_cols = [c for c in df2.columns if c != "prognosis"]

for _, row in df2.iterrows():
    disease = str(row["prognosis"]).lower().strip()
    symptoms = [col for col in symptom_cols if row[col] == 1]

    for label, aliases in RESPIRATORY_10.items():
        if disease in aliases and len(symptoms) >= 2:
            rows.append({
                "text": make_sentence(symptoms),
                "label": label
            })

df = pd.DataFrame(rows)

print("\nOriginal distribution:")
print(df["label"].value_counts())

# =========================
# BALANCE DATASET
# =========================
MIN_SAMPLES = 1000
balanced = []

for label, group in df.groupby("label"):
    if len(group) >= MIN_SAMPLES:
        balanced.append(group.sample(MIN_SAMPLES, random_state=42))
    else:
        balanced.append(group.sample(MIN_SAMPLES, replace=True, random_state=42))

final_df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced distribution:")
print(final_df["label"].value_counts())

final_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")
