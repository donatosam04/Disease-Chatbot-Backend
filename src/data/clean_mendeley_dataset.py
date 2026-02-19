import pandas as pd
from pathlib import Path

# ============================
# CONFIG
# ============================
INPUT = "data/raw/mendeley/symbipredict_2022.csv"
OUTPUT = "data/processed/mendeley_cleaned.csv"

# ============================
# LOAD
# ============================
df = pd.read_csv(INPUT)

print("🔹 Total samples:", len(df))
print("🔹 Columns:", list(df.columns))

# ============================
# IDENTIFY COLUMNS
# ============================
LABEL_COL = "prognosis"
SYMPTOM_COLS = [c for c in df.columns if c != LABEL_COL]

# ============================
# BUILD TEXT COLUMN
# ============================
def row_to_text(row):
    symptoms = [col.replace("_", " ") for col in SYMPTOM_COLS if row[col] == 1]
    return ", ".join(symptoms)

df["text"] = df.apply(row_to_text, axis=1)
df["label"] = df[LABEL_COL].str.lower().str.replace(" ", "_")

df = df[["text", "label"]]

# ============================
# FILTER RARE DISEASES
# ============================
df = df.groupby("label").filter(lambda x: len(x) >= 50)

print("\n✅ Filtered samples:", len(df))
print("✅ Unique diseases:", df["label"].nunique())
print("\nLabel distribution:")
print(df["label"].value_counts())

# ============================
# SAVE
# ============================
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"\n💾 Saved → {OUTPUT}")
