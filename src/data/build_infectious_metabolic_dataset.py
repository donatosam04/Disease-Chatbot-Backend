import pandas as pd
import random
from pathlib import Path

random.seed(42)

# ===============================
# OUTPUT
# ===============================
OUT = "data/processed/infectious_metabolic_train.csv"
Path("data/processed").mkdir(parents=True, exist_ok=True)

# ===============================
# DISEASE DEFINITIONS
# ===============================
DISEASE_SYMPTOMS = {
    "dengue": [
        "high fever", "severe headache", "joint pain", "muscle pain",
        "pain behind eyes", "skin rash", "fatigue"
    ],
    "malaria": [
        "fever with chills", "sweating", "headache",
        "nausea", "vomiting", "muscle pain", "fatigue"
    ],
    "typhoid": [
        "persistent fever", "abdominal pain", "loss of appetite",
        "weakness", "headache", "diarrhea"
    ],
    "diabetes": [
        "frequent urination", "increased thirst", "increased hunger",
        "fatigue", "blurred vision", "slow wound healing"
    ],
    "hypertension": [
        "headache", "dizziness", "chest pain",
        "shortness of breath", "fatigue", "blurred vision"
    ],
    "migraine": [
        "severe headache", "nausea", "vomiting",
        "sensitivity to light", "sensitivity to sound", "blurred vision"
    ],
    "uti": [
        "burning sensation during urination", "frequent urination",
        "lower abdominal pain", "cloudy urine", "strong urine odor"
    ],
}

TEMPLATES = [
    "I have been experiencing {symptoms} for the past few days.",
    "The patient reports {symptoms}.",
    "Symptoms include {symptoms}.",
    "I am suffering from {symptoms}.",
    "I have symptoms such as {symptoms}.",
]

SAMPLES_PER_DISEASE = 600

# ===============================
# GENERATE DATA
# ===============================
rows = []

for disease, symptoms in DISEASE_SYMPTOMS.items():
    for _ in range(SAMPLES_PER_DISEASE):
        k = random.randint(3, min(5, len(symptoms)))
        sampled = random.sample(symptoms, k)
        text = random.choice(TEMPLATES).format(
            symptoms=", ".join(sampled)
        )

        rows.append({
            "text": text,
            "label": disease
        })

df = pd.DataFrame(rows)

print("Generated dataset:")
print(df["label"].value_counts())

df.to_csv(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
