import pandas as pd
import random
from pathlib import Path

# ===============================
# CONFIG
# ===============================
INPUT = "data/raw/kaggle_disease_symptoms/DiseaseAndSymptoms.csv"

OUT_TRAIN = "data/processed/respiratory10_train_realistic.csv"
OUT_HARD  = "data/processed/respiratory10_hard_val_realistic.csv"

TRAIN_PER_CLASS = 600
HARD_PER_CLASS  = 200

TARGET_DISEASES = {
    "Allergy": "allergy",
    "Bronchial Asthma": "asthma",
    "Common Cold": "common_cold",
    "Pneumonia": "pneumonia",
    "Tuberculosis": "tuberculosis",
    "Bronchitis": "bronchitis",
    "COVID-19": "covid19",
    "Flu": "flu",
    "Sinusitis": "sinusitis",
    "Lung Cancer": "lung_cancer",
}

TEMPLATES = [
    "I am experiencing {symptoms}.",
    "For the past few days, I have had {symptoms}.",
    "My main symptoms include {symptoms}.",
    "Lately I have been suffering from {symptoms}.",
    "I feel unwell with {symptoms}.",
    "I am dealing with {symptoms} and it is getting worse.",
]

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(INPUT)

assert "Disease" in df.columns, "Disease column missing"

# collect all symptom columns
SYMPTOM_COLS = [c for c in df.columns if c.lower().startswith("symptom")]

print(f"Detected {len(SYMPTOM_COLS)} symptom columns")

# ===============================
# HELPERS
# ===============================
def clean_symptoms(row):
    return [
        str(row[c]).strip().replace("_", " ").lower()
        for c in SYMPTOM_COLS
        if pd.notna(row[c]) and str(row[c]).strip()
    ]

def generate_sentences(symptoms, n):
    rows = []
    for _ in range(n):
        k = random.randint(2, min(5, len(symptoms)))
        chosen = random.sample(symptoms, k)
        sentence = random.choice(TEMPLATES).format(
            symptoms=", ".join(chosen)
        )
        rows.append(sentence)
    return rows

# ===============================
# BUILD DATASETS
# ===============================
train_rows = []
hard_rows = []

for disease_name, label in TARGET_DISEASES.items():
    subset = df[df["Disease"].str.strip() == disease_name]

    if subset.empty:
        print(f"⚠️ Disease not found: {disease_name}")
        continue

    symptoms = clean_symptoms(subset.iloc[0])

    if len(symptoms) < 3:
        print(f"⚠️ Not enough symptoms for {disease_name}")
        continue

    train_texts = generate_sentences(symptoms, TRAIN_PER_CLASS)
    hard_texts  = generate_sentences(symptoms, HARD_PER_CLASS)

    train_rows.extend(
        [{"text": t, "label": label} for t in train_texts]
    )
    hard_rows.extend(
        [{"text": t, "label": label} for t in hard_texts]
    )

# ===============================
# SAVE
# ===============================
train_df = pd.DataFrame(train_rows)
hard_df  = pd.DataFrame(hard_rows)

Path("data/processed").mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUT_TRAIN, index=False)
hard_df.to_csv(OUT_HARD, index=False)

print("\n✅ Dataset creation complete")
print("\nTRAIN distribution:")
print(train_df["label"].value_counts())

print("\nHARD validation distribution:")
print(hard_df["label"].value_counts())
