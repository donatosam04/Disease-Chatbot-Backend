import random
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
OUT_PATH = Path("data/processed/respiratory10_val.csv")
SAMPLES_PER_CLASS = 500     # 🔥 increase if needed
random.seed(42)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# EASY (CLEAR) DISEASE TEMPLATES
# =========================================================
DISEASE_TEMPLATES = {
    "allergy": [
        "I have frequent sneezing and itchy eyes",
        "My nose is runny and my eyes are watery",
        "I sneeze a lot when exposed to dust or pollen",
        "I have nasal congestion with itching",
    ],
    "asthma": [
        "I experience wheezing and shortness of breath",
        "I feel tightness in my chest and difficulty breathing",
        "Breathing becomes hard after physical activity",
        "I wheeze at night and feel breathless",
    ],
    "common_cold": [
        "I have a runny nose and mild fever",
        "I am sneezing with a sore throat",
        "I feel tired with nasal congestion",
        "I have cold symptoms and slight headache",
    ],
    "pneumonia": [
        "I have high fever and chest pain while breathing",
        "I feel severe cough with difficulty breathing",
        "My breathing feels heavy and painful",
        "I have persistent fever and chest discomfort",
    ],
    "tuberculosis": [
        "I have a persistent cough with weight loss",
        "I sweat at night and feel very weak",
        "There is blood in my cough and chest pain",
        "I have long lasting cough and night sweats",
    ],
    "bronchitis": [
        "I have frequent coughing with mucus",
        "My chest feels congested and I cough a lot",
        "I have chest tightness with productive cough",
        "I cough constantly with thick sputum",
    ],
    "covid19": [
        "I lost my sense of taste and smell",
        "I have fever with dry cough and fatigue",
        "I feel breathless with body aches",
        "I feel tired with fever and breathing difficulty",
    ],
    "flu": [
        "I have sudden fever with body pain",
        "I feel chills headache and fatigue",
        "I have high fever and muscle aches",
        "I feel weak with severe fever",
    ],
    "sinusitis": [
        "I have facial pain and nasal blockage",
        "My sinuses hurt and nose feels blocked",
        "I feel pressure around my eyes and nose",
        "I have thick nasal discharge with pain",
    ],
    "lung_cancer": [
        "I have chronic cough and chest pain",
        "I cough blood and feel constant fatigue",
        "I experience unexplained weight loss and coughing",
        "I have persistent chest pain and weakness",
    ],
}

# =========================================================
# TEXT VARIATIONS (SAFE)
# =========================================================
VARIATIONS = [
    "",
    " for the past few days",
    " especially in the morning",
    " and it is getting worse",
    " with mild discomfort",
    " for several weeks",
]

# =========================================================
# GENERATION
# =========================================================
rows = []

for label, templates in DISEASE_TEMPLATES.items():
    for _ in range(SAMPLES_PER_CLASS):
        base = random.choice(templates)
        variation = random.choice(VARIATIONS)
        rows.append({
            "text": base + variation,
            "label": label
        })

df = pd.DataFrame(rows)

# =========================================================
# SAVE
# =========================================================
df.to_csv(OUT_PATH, index=False)

print("✅ EASY validation dataset created")
print(df["label"].value_counts())
print(f"📂 Saved → {OUT_PATH}")
