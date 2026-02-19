import json
import pandas as pd
from pathlib import Path

# ============================
# CONFIG
# ============================

BASE_DATASET = "data/processed/universal_disease_train_v5.csv"
ADDITIONAL_DATASET = "data/processed/additional_diseases.csv"  # adjust if name differs

OUT_PATH = "data/processed/universal_disease_train_v6.csv"
MAX_SAMPLES_PER_DISEASE = 800

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("🔹 Loading datasets...")

base_df = pd.read_csv(BASE_DATASET)
add_df = pd.read_csv(ADDITIONAL_DATASET)

# keep only required columns
base_df = base_df[["text", "label"]]
add_df = add_df[["text", "label"]]

# ============================
# COMBINE DATASETS
# ============================

combined = pd.concat(
    [base_df, add_df],
    ignore_index=True
)

print("📊 After merge total samples:", len(combined))
print("📊 After merge unique diseases:", combined["label"].nunique())

# ============================
# CLEAN
# ============================

combined["text"] = (
    combined["text"]
    .astype(str)
    .str.lower()
    .str.replace("_", " ", regex=False)
)

combined["label"] = combined["label"].astype(str).str.lower()
# ============================
# NORMALIZE LABELS
# ============================

MAP_FILE = "src/data/label_normalization_map.json"

if Path(MAP_FILE).exists():
    with open(MAP_FILE, "r") as f:
        label_map = json.load(f)

    combined["label"] = combined["label"].apply(
        lambda x: label_map.get(x, x)
    )

print("🧼 Unique labels after normalization:", combined["label"].nunique())

# ============================
# BALANCE
# ============================

print("🔹 Balancing dataset...")

balanced = (
    combined
    .groupby("label", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_DISEASE), random_state=42))
)

# ============================
# STATS
# ============================

print("\n✅ Total samples:", len(balanced))
print("✅ Unique diseases:", balanced["label"].nunique())
print("\nLabel distribution:")
print(balanced["label"].value_counts())

# ============================
# SAVE
# ============================

balanced.to_csv(OUT_PATH, index=False)
print(f"\n💾 Saved → {OUT_PATH}")
print("📊 Base dataset diseases:", base_df["label"].nunique())
print("📊 Additional dataset diseases:", add_df["label"].nunique())
print("📊 Additional labels:")
print(add_df["label"].value_counts())
